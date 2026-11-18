"""Allocation budgets and request bounds, independent of scheduler state."""

import math
import re


def seconds(value):
    """Parse an explicit [days-]HH:MM:SS duration, including hours > 23."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", str(value))
    if not match:
        raise ValueError(f"Expected [days-]HH:MM:SS duration, got {value!r}")
    days, hours, minutes, secs = (int(v or 0) for v in match.groups())
    result = days * 86400 + hours * 3600 + minutes * 60 + secs
    if result <= 0:
        raise ValueError("Continuation durations must be positive")
    return result


def duration(value):
    """Format seconds as HH:MM:SS, allowing hours greater than 23."""
    value = int(value)
    return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}"


def policy(values, maximum, scheduler, defaults=None):
    """Validate and normalize a stage's continuation policy.

    Parameters
    ----------
    values : dict
        Stage options; runtime_budget is required and durations are strings.
    maximum : str
        Single-allocation walltime ceiling from the resource profile.
    scheduler : {"slurm", "pbs"}
        Scheduler controlling request granularity and variable allocation support.
    defaults : dict, optional
        Site defaults overridden by explicit stage values.

    Returns
    -------
    dict
        Validated policy with durations expressed in seconds.

    Raises
    ------
    ValueError
        Options are unknown, inconsistent or cannot fit a usable allocation.
    """
    if not isinstance(values, dict):
        raise ValueError("continuation must be a mapping")
    merged = dict(defaults or {})
    merged.update(values)
    allowed = {
        "runtime_budget",
        "min_time",
        "checkpoint_margin",
        "site_min_time",
        "startup_allowance",
        "allow_pbs_shrink_to_fit",
        "max_attempts",
        "preferred_time",
        "max_infrastructure_retries",
    }
    if set(merged) - allowed:
        raise ValueError(
            f"Unknown continuation options: {sorted(set(merged) - allowed)}"
        )
    if scheduler not in ("slurm", "pbs"):
        raise ValueError("Continuation requires Slurm or PBS")
    if "runtime_budget" not in merged:
        raise ValueError("continuation requires runtime_budget")
    result = {
        "maximum": seconds(maximum),
        "preferred": seconds(merged.get("preferred_time", maximum)),
        "max_infrastructure_retries": merged.get("max_infrastructure_retries", 0),
        "runtime_budget": seconds(merged["runtime_budget"]),
        "checkpoint_margin": seconds(merged.get("checkpoint_margin", "00:15:00")),
        "site_min_time": seconds(merged.get("site_min_time", "00:01:00")),
        "startup_allowance": seconds(merged.get("startup_allowance", "00:01:00")),
        "minimum": seconds(merged["min_time"]) if merged.get("min_time") else None,
        "granularity": 60 if scheduler == "slurm" else 1,
        "max_attempts": merged.get("max_attempts", 100),
    }
    if type(result["max_attempts"]) is not int or result["max_attempts"] < 1:
        raise ValueError("max_attempts must be a positive integer")
    if scheduler == "slurm" and result["checkpoint_margin"] > 65535:
        raise ValueError("Slurm checkpoint_margin must be at most 65535 seconds")
    retries = result["max_infrastructure_retries"]
    if type(retries) is not int or retries < 0:
        raise ValueError("max_infrastructure_retries must be a non-negative integer")
    if result["preferred"] > result["maximum"]:
        raise ValueError("preferred_time must not exceed time")
    minimum = result["minimum"] or result["preferred"]
    if minimum > result["preferred"]:
        raise ValueError("min_time must not exceed preferred_time")
    if result["checkpoint_margin"] + result["startup_allowance"] >= minimum:
        raise ValueError("Allocation must exceed checkpoint_margin + startup_allowance")
    if result["site_min_time"] > minimum:
        raise ValueError("Allocation is shorter than site_min_time")
    if (
        scheduler == "pbs"
        and result["minimum"]
        and merged.get("allow_pbs_shrink_to_fit") is not True
    ):
        raise ValueError(
            "PBS min_time requires site validation and allow_pbs_shrink_to_fit: true"
        )
    if next_request(result, 0) is None:
        raise ValueError(
            "runtime_budget cannot fit a useful allocation at scheduler granularity"
        )
    return result


def next_request(config, used):
    """Choose the next allocation within the remaining runtime budget.

    Parameters
    ----------
    config : dict
        Normalized continuation policy.
    used : int
        Runtime already charged from confirmed scheduler accounting, in seconds.

    Returns
    -------
    dict or None
        Rounded request bounds and rounding loss, or None when the remaining
        budget cannot fit startup, checkpointing and the site's minimum duration.
    """
    remaining = config["runtime_budget"] - used
    granularity = config["granularity"]
    maximum = (
        int(min(config.get("preferred", config["maximum"]), remaining) // granularity)
        * granularity
    )
    threshold = max(
        config["site_min_time"],
        config["checkpoint_margin"] + config["startup_allowance"] + 1,
    )
    if maximum < threshold:
        return None
    minimum = config["minimum"]
    if minimum:
        minimum = min(maximum, math.ceil(minimum / granularity) * granularity)
    return {
        "time": duration(maximum),
        "min_time": duration(minimum) if minimum else None,
        "seconds": maximum,
        "rounding_seconds": min(config.get("preferred", config["maximum"]), remaining)
        - maximum,
    }


def profile_policy(catalog, profile, values):
    """Merge site minimums, validate allocation bounds, keep total budget separate."""
    from ..sites import allocation_limits, validate_allocation

    defaults = dict(profile.get("continuation_defaults") or {})
    limits = allocation_limits(catalog, profile)
    site_min = limits.get("min_time")
    if site_min:
        defaults["site_min_time"] = site_min
    scheduler = profile.get(
        "scheduler", "pbs" if profile.get("site") in ("anl", "polaris") else "slurm"
    )
    result = policy(values, profile["time"], scheduler, defaults)
    if site_min:
        result["site_min_time"] = max(result["site_min_time"], seconds(site_min))
    validate_allocation(catalog, profile)
    validate_allocation(
        catalog, dict(profile, time=duration(result["minimum"] or result["preferred"]))
    )
    if next_request(result, 0) is None:
        raise ValueError("Runtime budget cannot fit the site's minimum allocation")
    return result


def describe_policy(config):
    """Describe allocation bounds, budget and exhaustion behavior for dry-run output."""
    return (
        f"budget={duration(config['runtime_budget'])}; allocation="
        f"{duration(config['minimum']) if config['minimum'] else 'fixed'}..{duration(config['maximum'])}; "
        f"preferred={duration(config['preferred'])}; margin={duration(config['checkpoint_margin'])}; "
        f"startup={duration(config['startup_allowance'])}; site minimum={duration(config['site_min_time'])}; "
        f"infrastructure retries={config['max_infrastructure_retries']}; max attempts={config['max_attempts']}; "
        "exhaustion=stop and block consumers; final chunk=only if startup+margin+site minimum fit"
    )

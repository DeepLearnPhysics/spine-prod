"""Portable resource profiles and scheduler allocation limits.

Limits describe one allocation, not a training runtime budget. Site-mapped
training preserves source walltimes as cumulative continuation budgets.
"""

import re

from .continuation.policy import duration


def duration_seconds(value):
    """Parse explicit [days-]hours:minutes:seconds allocation durations."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", str(value))
    if not match:
        raise ValueError(f"Invalid allocation time {value!r}; use [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total == 0:
        raise ValueError("Allocation time must be positive, not unlimited")
    return total


def allocation_limits(catalog, profile):
    """Return configured limits for the effective site and queue/QOS."""
    site = profile.get("site", "s3df")
    policy = catalog.get("sites", {}).get(site, {})
    selector = policy.get("queue_field", "partition")
    return policy.get("limits", {}).get(profile.get(selector), {})


def validate_allocation(catalog, profile):
    """Reject known hard-limit violations without shortening the request."""
    limits = allocation_limits(catalog, profile)
    if not limits:
        return
    requested = duration_seconds(profile["time"])
    for field, compare in (
        ("min_time", lambda a, b: a < b),
        ("max_time", lambda a, b: a > b),
    ):
        bound = limits.get(field)
        if bound and compare(requested, duration_seconds(bound)):
            raise ValueError(
                f"{profile['site']} allocation time {profile['time']} violates "
                f"{field}={bound}. Set an explicit --time or use allocation "
                "continuation; training budgets are never silently truncated."
            )
    nodes = int(profile.get("nodes", 1))
    if not limits.get("min_nodes", 1) <= nodes <= limits.get("max_nodes", nodes):
        raise ValueError(f"Node count {nodes} violates queue limits: {limits}")
    gpus = int(profile.get("gpus_per_node", profile.get("gpus", 0)))
    if gpus > limits.get("max_gpus_per_node", gpus):
        raise ValueError(f"GPU count {gpus} violates queue limits: {limits}")


def map_stage(catalog, stage, site, overrides=None):
    """Translate a concrete profile through its allocation role.

    Scheduler-specific YAML fields belong to the original site. CPU and memory
    requirements, training budgets, and application settings survive translation; explicit
    launch overrides apply last in the destination site's vocabulary.
    """
    overrides = dict(overrides or {})
    training = stage.get("kind", "spine") == "spine" and stage.get("stage") == "train"
    if not training:
        overrides.pop("nodes", None)
    sites = catalog["sites"]
    site = {"alcf": "anl", "polaris": "anl"}.get(site, site)
    if site not in sites:
        raise ValueError(f"Unknown site: {site}. Available: {', '.join(sites)}")
    source_name = stage.get(
        "profile",
        "s3df_milano" if stage.get("kind") in ("filter", "report") else "s3df_ampere",
    )
    profiles = catalog["profiles"]
    if source_name not in profiles:
        raise ValueError(
            f"Site mapping requires a named resource profile: {source_name}"
        )
    source = profiles[source_name]
    result = dict(stage)
    result.setdefault("profile", source_name)
    if training:
        continuation = dict(stage.get("continuation", {}))
        continuation.setdefault("runtime_budget", stage.get("time", source["time"]))
        result["continuation"] = continuation
    if source["site"] != site:
        role = source.get("allocation_role")
        if role not in sites[site]["roles"]:
            raise ValueError(
                f"No {site} correspondence for profile {source_name} ({role})"
            )
        target_name = sites[site]["roles"][role]
        target = profiles[target_name]
        # Do not silently move a high-memory model onto a smaller GPU.
        if source.get("gpu_mem") and int(str(source["gpu_mem"]).rstrip("g")) > int(
            str(target["gpu_mem"]).rstrip("g")
        ):
            raise ValueError(
                f"{source_name} needs more GPU memory than {target_name}; "
                "select an appropriate --profile explicitly"
            )
        result["profile"] = target_name
        gpu_count = result.get("gpus_per_node", result.get("gpus"))
        if role == "cpu" and "mem_per_node" not in result:
            result.setdefault("mem_per_cpu", source.get("mem_per_cpu", "4g"))
        for field in (
            "partition",
            "qos",
            "queue",
            "constraint",
            "exclude",
            "account",
            "gpus",
            "gpus_per_node",
        ):
            result.pop(field, None)
        if gpu_count is not None:
            result["gpus" if site == "s3df" else "gpus_per_node"] = gpu_count
    result.update(overrides)
    if (
        result.get("cache_repository")
        and result.get("files_per_task") is None
        and "ntasks" not in overrides
        and "cache_ntasks" in sites[site]
    ):
        result["ntasks"] = sites[site]["cache_ntasks"]
    target = profiles[result["profile"]]
    if (
        int(result.get("nodes", target.get("nodes", 1))) > 1
        and "profile" not in overrides
        and target.get("multi_node_profile")
    ):
        result["profile"] = target["multi_node_profile"]
        target = profiles[result["profile"]]
        if "qos" not in overrides:
            result.pop("qos", None)
    if target["site"] != site:
        raise ValueError(
            f"--profile {result['profile']} does not belong to site {site}"
        )
    if site != "s3df" and not result.get("account") and not target.get("account"):
        raise ValueError(
            f"--site {site} requires --account (or an account in its profile)"
        )
    effective = dict(target, **{k: v for k, v in result.items() if k != "profile"})
    nodes = int(effective.get("nodes", 1))
    if nodes < 1:
        raise ValueError("Node count must be positive")
    if nodes > 1 and not training:
        raise ValueError("Multi-node allocations are supported only for training")
    selector = sites[site]["queue_field"]
    if effective.get(selector) not in sites[site]["limits"]:
        raise ValueError(
            f"No configured limits for {site} {selector}={effective.get(selector)}"
        )
    if site == "anl" and "mem_per_cpu" in result:
        memory = str(result.pop("mem_per_cpu"))
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgtKMGT])(?:[bB])?", memory)
        if not match:
            raise ValueError(f"PBS memory per CPU requires a unit, e.g. 4g: {memory}")
        result["mem_per_node"] = (
            f"{float(match[1]) * effective['cpus_per_task']:g}{match[2].lower()}b"
        )
    if site == "anl" and "mem_per_node" in result:
        result["mem_per_node"] = re.sub(
            r"([kmgt])$", r"\1b", str(result["mem_per_node"]), flags=re.IGNORECASE
        )
    return _continuation_allocation(catalog, result, target, overrides)


def _continuation_allocation(catalog, stage, profile, overrides):
    """Bound opted-in allocations without changing their total training budget.

    Explicit launch times and stage chunk preferences remain authoritative and
    are validated later. Only inherited ceilings and site defaults are adapted.
    """
    if "continuation" not in stage:
        return stage
    effective = dict(profile, **stage)
    ceiling = duration_seconds(effective["time"])
    hard_limit = allocation_limits(catalog, effective).get("max_time")
    if hard_limit and "time" not in overrides:
        ceiling = min(ceiling, duration_seconds(hard_limit))
    stage["time"] = duration(ceiling)
    defaults = dict(catalog["sites"][profile["site"]].get("continuation_defaults", {}))
    defaults.update(profile.get("continuation_defaults", {}))
    explicit = stage["continuation"]
    preferred = explicit.get(
        "preferred_time", defaults.get("preferred_time", stage["time"])
    )
    preferred_seconds = min(ceiling, duration_seconds(preferred))
    for key, bound in (("preferred_time", ceiling), ("min_time", preferred_seconds)):
        if key in defaults and key not in explicit:
            defaults[key] = duration(min(duration_seconds(defaults[key]), bound))
    stage["continuation"] = dict(defaults, **explicit)
    return stage

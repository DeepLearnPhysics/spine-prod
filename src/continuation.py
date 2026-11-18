"""Durable, fail-closed allocation continuation (contract version 1).

The submission host advances the workflow. Compute jobs never submit other jobs.
All walltimes are seconds internally; queue wait is never charged to the budget.
"""

import contextlib
import errno
import fcntl
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .run_manager import RunManager

CONTRACT = 1


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
    value = int(value)
    return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}"


def policy(values, maximum, scheduler, defaults=None):
    """Validate opt-in policy and merge operational site defaults."""
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
    if minimum > result["maximum"]:
        raise ValueError("min_time must not exceed time")
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
    """Round down to avoid overspending; allow a usable final short allocation."""
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


def validate_storage(path, site):
    """Require explicit Scratch continuation storage on NERSC before submission.

    Check canonical paths as well as the provided spelling (symlinks included).
    This is not a substitute for compute-host preflight on other filesystems.
    """
    if site != "nersc":
        return
    if not path:
        raise ValueError(
            "NERSC continuation requires an explicit --continuation-dir on $SCRATCH"
        )
    resolved = Path(path).expanduser().resolve()
    roots = ("/pscratch", "/global/cscratch1", "/global/cscratch2")
    if not any(str(resolved).startswith(root + "/") for root in roots):
        raise ValueError(
            f"NERSC --continuation-dir {resolved} must be on Scratch ($SCRATCH); "
            "CFS and other DVS paths cannot hold continuation locks. "
            "The workspace, training logs and checkpoints may remain on CFS."
        )


def preflight_storage(run_dir, site=None, continuation_dir=None):
    validate_storage(
        continuation_dir if continuation_dir is not None else run_dir, site
    )
    # Use the actual run's lock. Test publication on a unique probe without
    # modifying the manifest or deleting another process's probe.
    path = manifest_path(run_dir, continuation_dir)
    with locked(path):
        probe = path.parent / ("preflight-" + uuid.uuid4().hex + ".json")
        try:
            atomic_json(probe, {"probe": True})
            if read(probe) != {"probe": True}:
                raise RuntimeError(
                    "Continuation filesystem publication verification failed"
                )
        finally:
            probe.unlink(missing_ok=True)


def profile_policy(catalog, profile, values):
    """Merge site minimums, validate allocation bounds, keep total budget separate."""
    from .site_resources import allocation_limits, validate_allocation

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
    return (
        f"budget={duration(config['runtime_budget'])}; allocation="
        f"{duration(config['minimum']) if config['minimum'] else 'fixed'}..{duration(config['maximum'])}; "
        f"preferred={duration(config['preferred'])}; margin={duration(config['checkpoint_margin'])}; "
        f"startup={duration(config['startup_allowance'])}; site minimum={duration(config['site_min_time'])}; "
        f"infrastructure retries={config['max_infrastructure_retries']}; max attempts={config['max_attempts']}; "
        "exhaustion=stop and block consumers; final chunk=only if startup+margin+site minimum fit"
    )


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def locked(path):
    """Serialize readers that mutate state; replacement keeps readers atomic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno in (524, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise RuntimeError(
                    f"Continuation cannot lock {path}: this filesystem does not support "
                    "file locking from this host. On NERSC set --continuation-dir on "
                    "$SCRATCH, not CFS or Global Homes. Do not disable locking."
                ) from exc
            raise
        yield


def read(path):
    return json.loads(Path(path).read_text())


def manifest_path(run_dir, continuation_dir=None):
    return (
        Path(continuation_dir if continuation_dir is not None else run_dir)
        / ".spine-prod"
        / "continuation.json"
    )


def reserve(
    run_dir, identity, config, scheduler, attempt_dir, site=None, continuation_dir=None
):
    """Commit submission intent before contacting a scheduler. Never retry ambiguity."""
    path = manifest_path(run_dir, continuation_dir)
    with locked(path):
        state = (
            read(path)
            if path.exists()
            else {
                "contract": CONTRACT,
                "identity": identity,
                "generation": 0,
                "state": "pending",
                "used_seconds": 0,
                "attempts": [],
            }
        )
        if state["identity"] != identity:
            raise ValueError("Continuation scientific configuration/topology changed")
        if state["state"] not in ("pending", "paused") or (
            state["attempts"] and not state["attempts"][-1].get("accounted")
        ):
            raise ValueError(
                f"Cannot submit continuation in state {state['state']}; reconcile first"
            )
        config = dict(config)
        config["runtime_budget"] = state.get(
            "budget_override", config["runtime_budget"]
        )
        state["policy"] = config
        request = next_request(config, state["used_seconds"])
        if len(state["attempts"]) >= config["max_attempts"]:
            state.update(
                state="failed",
                reason="max_attempts reached",
                generation=state["generation"] + 1,
            )
            atomic_json(path, state)
            raise ValueError("Continuation max_attempts reached")
        if request is None:
            state.update(
                state="budget_exhausted",
                reason="runtime_budget_exhausted",
                generation=state["generation"] + 1,
            )
            atomic_json(path, state)
            raise ValueError("Continuation runtime budget exhausted")
        token = "spc-" + uuid.uuid4().hex[:20]
        attempt = {
            "token": token,
            "directory": str(attempt_dir),
            "scheduler": scheduler,
            "number": len(state["attempts"]) + 1,
            "site": site,
            "start_progress": state.get("progress", -1),
            "request": request,
            "intent_time": time.time(),
            "job_id": None,
        }
        state["attempts"].append(attempt)
        state.update(state="submitting", generation=state["generation"] + 1)
        atomic_json(path, state)
        return attempt


def submitted(run_dir, token, job_id, continuation_dir=None):
    path = manifest_path(run_dir, continuation_dir)
    with locked(path):
        state = read(path)
        attempt = state["attempts"][-1]
        if attempt["token"] != token:
            raise ValueError("Stale submission token")
        # An unknown response is deliberately left submitting for reconciliation.
        if job_id:
            if attempt.get("job_id") not in (None, job_id):
                raise ValueError("Submission token already belongs to another job")
            attempt["job_id"] = job_id
            if state["state"] == "submitting":
                state["state"] = "pending"
        state["generation"] += 1
        atomic_json(path, state)


def command(argv):
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=30
    ).stdout


def discover_submission(scheduler, token):
    """Reconcile an uncertain submission by its unique, exact scheduler name.

    No match is not proof of rejection (accounting can lag): leave the intent
    pending, never blindly submit again. Multiple matches require intervention.
    """
    if not re.fullmatch(r"spc-[a-f0-9]{20}", token):
        raise ValueError("Invalid continuation submission token")
    matches = set()
    if scheduler == "slurm":
        output = command(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "--starttime=1970-01-01",
                "--name=" + token,
                "--format=JobIDRaw,JobName%100",
            ]
        )
        for line in output.splitlines():
            fields = line.split("|")
            if (
                len(fields) >= 2
                and fields[1].strip() == token
                and fields[0].strip().isdigit()
            ):
                matches.add(fields[0].strip())
    else:
        jobs = json.loads(command(["qstat", "-xf", "-F", "json"]))["Jobs"]
        matches.update(
            job_id for job_id, data in jobs.items() if data.get("Job_Name") == token
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple jobs match submission intent {token}: {sorted(matches)}; reconcile manually"
        )
    return next(iter(matches), None)


def allocation(scheduler, job_id):
    """Read actual assigned start/deadline; never substitute requested walltime."""
    if scheduler == "slurm":
        data = json.loads(command(["scontrol", "show", "job", job_id, "--json"]))[
            "jobs"
        ][0]

        def scalar(value):
            return value.get("number") if isinstance(value, dict) else value

        start, end = scalar(data["start_time"]), scalar(data["end_time"])
    else:
        data = json.loads(command(["qstat", "-f", "-F", "json", job_id]))["Jobs"][
            job_id
        ]
        start = data["stime"]
        if isinstance(start, str) and not start.isdigit():
            start = time.mktime(time.strptime(start, "%a %b %d %H:%M:%S %Y"))
        start = float(start)
        end = start + seconds(data["Resource_List"]["walltime"])
    start, end = float(start), float(end)
    if not 0 < start < end or start > time.time() + 60:
        raise ValueError(
            "Scheduler did not supply a valid assigned allocation deadline"
        )
    return start, end


def accounting(scheduler, job_id):
    """Return None for live/unknown jobs; require confirmed terminal accounting."""
    if scheduler == "slurm":
        output = command(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--starttime=1970-01-01",
                "--format=JobIDRaw,State,ElapsedRaw,ExitCode",
            ]
        )
        for line in output.splitlines():
            fields = line.split("|")
            if len(fields) >= 4 and fields[0] == job_id:
                status = fields[1].split()[0].rstrip("+")
                terminal = {
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "NODE_FAIL",
                    "OUT_OF_MEMORY",
                    "PREEMPTED",
                    "BOOT_FAIL",
                    "DEADLINE",
                }
                if status in terminal:
                    return {
                        "status": status,
                        "seconds": int(fields[2]),
                        "success": status == "COMPLETED" and fields[3] == "0:0",
                        "exit_code": fields[3],
                    }
    else:
        data = json.loads(command(["qstat", "-xf", "-F", "json", job_id]))["Jobs"].get(
            job_id, {}
        )
        if data.get("job_state") == "F" and "Exit_status" in data:
            walltime = data.get("resources_used", {}).get("walltime")
            # A job that never started consumes zero allocation time.
            if walltime is None and "stime" in data:
                raise ValueError("PBS completed job is missing walltime accounting")
            elapsed = 0 if walltime in (None, "00:00:00") else seconds(walltime)
            return {
                "status": str(data["Exit_status"]),
                "seconds": elapsed,
                "success": int(data["Exit_status"]) == 0,
            }
    return None


def verify_outcome(run_dir, attempt):
    outcome = read(Path(attempt["directory"]) / "outcome.json")
    if outcome.get("contract") != CONTRACT or outcome.get("token") != attempt["token"]:
        raise ValueError("Missing or stale SPINE continuation outcome")
    if outcome.get("state") not in ("paused", "completed"):
        raise ValueError("Invalid SPINE continuation outcome")
    checkpoint = Path(outcome["checkpoint"]).resolve()
    if checkpoint.parent != (Path(run_dir) / "weights").resolve():
        raise ValueError("Continuation checkpoint is outside the run weights directory")
    digest = RunManager.checkpoint_digest(checkpoint)
    if not digest or digest != outcome["sha256"]:
        raise ValueError(
            "Continuation requires a committed checkpoint and matching SHA256"
        )
    if type(outcome.get("iteration")) is not int or outcome["iteration"] < -1:
        raise ValueError("Invalid checkpoint progress")
    return outcome


def recovery_checkpoint(run_dir, attempt):
    """Only use integrity-checked snapshots from the interrupted allocation.

    Native SPINE enforces the full-state contract on resume. Never recover from
    best.ckpt or infer terminal success from a periodic checkpoint.
    """
    failures = []
    for iteration, path in reversed(RunManager.checkpoints(Path(run_dir))):
        if iteration <= attempt.get("start_progress", -1):
            continue
        try:
            digest = RunManager.checkpoint_digest(path)
            if digest:
                return dict(
                    checkpoint=str(path.resolve()), sha256=digest, progress=iteration
                )
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    raise ValueError(
        "No new verified periodic checkpoint for recovery"
        + (": " + "; ".join(failures) if failures else "")
    )


def failure_reason(result):
    return {
        "PREEMPTED": "preemption",
        "NODE_FAIL": "node_failure",
        "BOOT_FAIL": "node_failure",
        "TIMEOUT": "allocation_expiry",
        "DEADLINE": "allocation_expiry",
        "CANCELLED": "user_cancelled",
        "OUT_OF_MEMORY": "out_of_memory",
    }.get(result["status"], "application_failure")


def reconcile(run_dir, continuation_dir=None):
    """Charge terminal accounting exactly once; retries require proven recovery."""
    path = manifest_path(run_dir, continuation_dir)
    with locked(path):
        state = read(path)
        if not state["attempts"]:
            return state
        attempt = state["attempts"][-1]
        if attempt.get("accounted"):
            return state
        if not attempt.get("job_id"):
            job_id = discover_submission(attempt["scheduler"], attempt["token"])
            if job_id is None:
                return state
            attempt["job_id"] = job_id
            attempt["submission_reconciled_at"] = time.time()
            if state["state"] == "submitting":
                state["state"] = "pending"
            state["generation"] += 1
            atomic_json(path, state)
        result = accounting(attempt["scheduler"], attempt["job_id"])
        if result is None:
            return state
        if type(result["seconds"]) is not int or result["seconds"] < 0:
            raise ValueError("Invalid scheduler runtime accounting")
        attempt.update(accounted=True, accounting=result)
        state["used_seconds"] += result["seconds"]
        if state["state"] != "cancelled":
            try:
                if result["success"]:
                    outcome = verify_outcome(run_dir, attempt)
                    previous = state.get("progress", -1)
                    validation_completed = (
                        len(state["attempts"]) > 1
                        and state["attempts"][-2]
                        .get("outcome", {})
                        .get("pending_validation")
                        is True
                        and outcome.get("pending_validation") is False
                        and outcome["iteration"] == previous
                    )
                    if (
                        outcome["state"] == "paused"
                        and outcome["iteration"] <= previous
                        and not validation_completed
                        and not (
                            len(state["attempts"]) == 1 and outcome["iteration"] == -1
                        )
                    ):
                        raise ValueError("Paused allocation made no training progress")
                    attempt["outcome"] = outcome
                    reason = outcome.get("reason") or (
                        "allocation_deadline"
                        if outcome["state"] == "paused"
                        else "training_limit"
                    )
                    state.update(
                        state=outcome["state"],
                        reason=reason,
                        checkpoint=outcome["checkpoint"],
                        sha256=outcome["sha256"],
                        progress=outcome["iteration"],
                    )
                    attempt["reason"] = reason
                else:
                    reason = failure_reason(result)
                    attempt["reason"] = reason
                    state.update(
                        state="cancelled" if reason == "user_cancelled" else "failed",
                        reason=reason,
                    )
                    failure_file = Path(attempt["directory"]) / "failure.json"
                    if failure_file.exists():
                        details = read(failure_file)
                        if details.get("token") == attempt["token"]:
                            attempt["failure"] = details
                    retries = state.get("infrastructure_retries", 0)
                    limit = state["policy"].get("max_infrastructure_retries", 0)
                    # Application failures/OOM/cancellation are never automatic retries.
                    if (
                        reason in ("preemption", "node_failure", "allocation_expiry")
                        and retries < limit
                    ):
                        recovered = recovery_checkpoint(run_dir, attempt)
                        state.update(
                            recovered,
                            state="paused",
                            infrastructure_retries=retries + 1,
                        )
                        attempt["recovery"] = recovered
                if state["state"] == "paused":
                    if next_request(state["policy"], state["used_seconds"]) is None:
                        state.update(
                            state="budget_exhausted", reason="runtime_budget_exhausted"
                        )
                    elif len(state["attempts"]) >= state["policy"]["max_attempts"]:
                        state.update(state="failed", reason="max_attempts_reached")
            except (OSError, ValueError, KeyError, TypeError) as error:
                state.update(
                    state="failed", reason="invalid_restart_state", detail=str(error)
                )
        state["generation"] += 1
        atomic_json(path, state)
        return state

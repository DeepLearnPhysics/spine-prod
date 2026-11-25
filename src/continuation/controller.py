"""Short-lived submission-host controller for continuation pipelines.

Stages are deliberately serialized in this first implementation. No consumer is
submitted against an intermediate allocation's successful exit code.
"""

import hashlib
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

from ..runs import RunManager
from .checkpoints import recovery_checkpoint, verify_outcome
from .policy import describe_policy, duration, next_request, profile_policy, seconds
from .scheduler import accounting, command, discover_submission, failure_reason
from .state import atomic_json, locked, manifest_path, read, validate_storage


def start(
    context,
    stages,
    workspace,
    dry_run=False,
    preload=False,
    warm_start=None,
    continuation_dir=None,
    cache_archives=None,
    audit_dir=None,
):
    """Create a durable controller and submit the first eligible stage.

    Parameters
    ----------
    context : Submitter
        Shared submission services and resource profiles.
    stages : sequence of dict
        Resolved stages in dependency order.
    workspace : str or pathlib.Path
        Pipeline output root.
    dry_run : bool, default False
        Describe the plan without registering a timer or submitting jobs.
    preload : bool, default False
        Resolve downloadable assets before submission.
    warm_start : str, optional
        Checkpoint used for declared module initialization.
    continuation_dir : str or pathlib.Path, optional
        Separate shared controller root; explicitly required on NERSC.

    Returns
    -------
    dict
        Stage names mapped to scheduler job identifiers from the initial pass.
    """
    from ..pipeline import PipelineRunner

    if not workspace:
        raise ValueError("Continuation pipelines require a workspace")
    path = (
        Path(continuation_dir or workspace).expanduser().resolve()
        / ".spine-prod"
        / "continuation_pipeline.json"
    )
    records = []
    for stage in stages:
        if stage.get("cleanup"):
            raise ValueError(
                "Continuation pipelines do not yet support automatic cleanup"
            )
        if stage.get("profile", "auto") == "auto":
            raise ValueError(
                "Continuation pipelines require explicit profiles for every stage"
            )
        options = PipelineRunner._submission_options(
            stage, None, warm_start_path=warm_start
        )
        if options.get("continuation") is not None and continuation_dir is not None:
            run_key = hashlib.sha256(
                str(Path(options["run_dir"]).resolve()).encode()
            ).hexdigest()
            options["continuation_dir"] = str(
                Path(continuation_dir).expanduser().resolve() / "runs" / run_key
            )
        records.append(
            {
                "name": stage["name"],
                "kind": stage.get("kind", "spine"),
                "options": options,
                "status": "waiting",
                "jobs": [],
            }
        )
        if stage["name"] in (cache_archives or {}):
            records.append(
                {
                    "name": stage["name"] + "__archive",
                    "kind": "archive",
                    "options": cache_archives[stage["name"]],
                    "status": "waiting",
                    "jobs": [],
                }
            )
    audit_dir = str(
        Path(audit_dir or Path(workspace) / "monitoring").expanduser().resolve()
    )
    automation_accounts = set()
    # Validate every operational policy before the first scheduler interaction.
    for record in records:
        options = record["options"]
        if options.get("continuation") is not None:
            profile = context.config_mgr.get_profile(
                options.get("profile", "auto"), None
            )
            from ..pipeline import PROFILE_FIELDS

            profile.update(
                {key: options[key] for key in PROFILE_FIELDS if key in options}
            )
            validate_storage(continuation_dir, profile.get("site"))
            if profile.get("site") == "nersc":
                automation_accounts.add(profile.get("account"))
            config = profile_policy(context.profiles, profile, options["continuation"])
            print(f"{record['name']}: {describe_policy(config)}")
    if len(automation_accounts) > 1:
        raise ValueError(
            "Automatic continuation requires one NERSC account per pipeline"
        )
    if automation_accounts:
        from .service import service_paths

        account = next(iter(automation_accounts))
        service_paths(account)
        print(
            "Automation plan: NERSC scrontab, every minute; will register and verify before training submission"
        )
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if dry_run:
        print(
            "Continuation strategy: submission-host advance; all consumers deferred until logical completion"
        )
        for index, record in enumerate(records):
            print(f"Stage {record['name']}: serialized completion gate")
            if index:
                print(
                    f"  Deferred until predecessors complete; options: {record['options']}"
                )
                continue
            method = getattr(
                context,
                "submit_" + ("job" if record["kind"] == "spine" else record["kind"]),
            )
            method(dry_run=True, **record["options"])
        return {}
    with locked(path):
        if path.exists():
            state = read(path)
            if state.get("audit_dir", audit_dir) != audit_dir:
                raise ValueError(
                    "Controller audit directory changed; use the original --audit-dir"
                )
            if state["definition_sha256"] != digest:
                raise ValueError("Controller definition changed; use a new workspace")
        else:
            atomic_json(
                path,
                {
                    "contract": 1,
                    "state": "active",
                    "definition_sha256": digest,
                    "basedir": str(context.basedir),
                    "audit_dir": audit_dir,
                    "records": records,
                    "preload": preload,
                    "generation": 0,
                },
            )
    write_tick_script(path, context.basedir)
    print(f"Durable controller: {path}")
    if automation_accounts:
        from .service import register

        registry = register(path, account, context.basedir)
        print(f"Automatic continuation registered: {registry}")
    else:
        print(
            "Automatic hosting is not yet configured for this site; periodic advance calls are required."
        )
    print(
        f"Unattended tick script: {path.parent / 'advance.sh'} (manual recovery/debugging)"
    )
    return advance(path, context)


def advance(path, context=None):
    """Reconcile a pipeline and submit at most one eligible stage or allocation.

    Parameters
    ----------
    path : str or pathlib.Path
        Durable pipeline manifest.
    context : Submitter, optional
        Submission services; reconstructed from the manifest when omitted.

    Returns
    -------
    dict
        Stage names mapped to their current scheduler job identifiers.

    Notes
    -----
    The pipeline lock serializes concurrent calls. A successful allocation exit
    alone never releases consumers of an incomplete logical training stage.
    """
    from ..submitter import Submitter
    from .audit import archive_audit

    path = Path(path)
    with locked(path):
        state = read(path)
        if state["state"] != "active":
            archive_audit(path)
            return {}
        context = context or Submitter(basedir=Path(state["basedir"]))
        for record in state["records"]:
            if record["status"] == "completed":
                continue
            options = dict(record["options"])
            is_continuation = options.get("continuation") is not None
            run_manifest = (
                manifest_path(options["run_dir"], options.get("continuation_dir"))
                if is_continuation
                else None
            )
            if run_manifest and run_manifest.exists():
                logical = reconcile(options["run_dir"], options.get("continuation_dir"))
                if not logical["attempts"]:
                    raise RuntimeError(
                        "Continuation manifest has no submission attempt; inspect its failure state"
                    )
                attempt = logical["attempts"][-1]
                if attempt.get("job_id"):
                    record["jobs"] = [attempt["job_id"]]
                if logical["state"] == "completed" and attempt.get("accounted"):
                    record["status"] = "completed"
                    atomic_json(path, state)
                    continue
                if logical["state"] != "paused" or not attempt.get("accounted"):
                    atomic_json(path, state)
                    if logical["state"] in ("failed", "cancelled", "budget_exhausted"):
                        archive_audit(path)
                    return {record["name"]: record["jobs"]}
                # Resume exactly the verified committed continuation artifact.
                options.update(resume=True, resume_from=logical["checkpoint"])
            elif record["status"] == "submitted":
                results = [
                    accounting(record["scheduler"], job) for job in record["jobs"]
                ]
                if any(result is None for result in results):
                    return {record["name"]: record["jobs"]}
                if not all(result["success"] for result in results):
                    record["status"] = "failed"
                    atomic_json(path, state)
                    archive_audit(path)
                    return {}
                record["status"] = "completed"
                atomic_json(path, state)
                continue
            elif record["status"] != "waiting":
                # A controller crash between scheduler submission and recording
                # its response is uncertain. Never repeat that submission.
                return {record["name"]: record["jobs"]}
            profile = context.config_mgr.get_profile(
                options.get("profile", "auto"), None
            )
            record["scheduler"] = profile.get(
                "scheduler",
                "pbs" if profile.get("site") in ("anl", "polaris") else "slurm",
            )
            record.update(status="submitting", token="spc-" + uuid.uuid4().hex[:20])
            options["job_name"] = record["token"]
            state["generation"] += 1
            atomic_json(path, state)
            method = getattr(
                context,
                "submit_" + ("job" if record["kind"] == "spine" else record["kind"]),
            )
            if record["kind"] == "spine":
                jobs = method(preload=state.get("preload", False), **options)
            else:
                jobs = method(**options)
            record["jobs"] = jobs
            if jobs:
                record["status"] = "submitted"
            atomic_json(path, state)
            return {record["name"]: jobs}
        state["state"] = "completed"
        state["generation"] += 1
        atomic_json(path, state)
        archive_audit(path)
        return {}


def operate(path, action, stage=None, value=None):
    """Rearm automatic hosting before changing a blocked pipeline's state."""
    if action in ("extend", "retry", "accept", "adopt"):
        from .service import reactivating

        with reactivating(read(path)):
            return _operate(path, action, stage, value)
    return _operate(path, action, stage, value)


def _operate(path, action, stage=None, value=None):
    """Serialize user operations with submission, including cancellation races."""
    path = Path(path)
    with locked(path):
        state = read(path)
        records = [r for r in state["records"] if stage is None or r["name"] == stage]
        if not records:
            raise ValueError(f"Unknown stage: {stage}")
        if action != "cancel" and (stage is None or len(records) != 1):
            raise ValueError("This operation requires --stage")
        if action == "cancel":
            state["state"] = "cancelled"
            atomic_json(path, state)  # Inhibit submissions even if qdel fails.
        for record in records:
            options = record["options"]
            run_path = (
                manifest_path(options["run_dir"], options.get("continuation_dir"))
                if options.get("continuation") is not None
                else None
            )
            if not run_path or not run_path.exists():
                if action != "cancel":
                    raise ValueError("Stage has no continuation manifest")
                for job in record["jobs"]:
                    command(
                        ["scancel" if record["scheduler"] == "slurm" else "qdel", job]
                    )
                continue
            with locked(run_path):
                logical = read(run_path)
                attempt = logical["attempts"][-1]
                if action == "extend":
                    budget = seconds(value)
                    if budget <= logical["policy"]["runtime_budget"]:
                        raise ValueError(
                            "New budget must be larger than the current budget"
                        )
                    logical["budget_override"] = budget
                    logical["policy"]["runtime_budget"] = budget
                    if logical["state"] == "budget_exhausted":
                        logical["state"] = "paused"
                elif action == "accept":
                    if logical["state"] not in (
                        "paused",
                        "budget_exhausted",
                    ) or not attempt.get("accounted"):
                        raise ValueError(
                            "Accept requires an accounted, checkpointed, inactive stage"
                        )
                    if (
                        RunManager.checkpoint_digest(Path(logical["checkpoint"]))
                        != logical["sha256"]
                    ):
                        raise ValueError("Checkpoint changed")
                    logical.update(state="completed", reason="user_accepted")
                elif action == "cancel":
                    logical.update(state="cancelled", reason="user_cancelled")
                elif action == "retry":
                    if logical["state"] != "failed" or not attempt.get("accounted"):
                        raise ValueError(
                            "Retry requires confirmed terminal failure accounting"
                        )
                    if len(logical["attempts"]) >= logical["policy"]["max_attempts"]:
                        raise ValueError("max_attempts reached")
                    valid = None
                    for _, candidate in reversed(
                        RunManager.checkpoints(Path(options["run_dir"]))
                    ):
                        try:
                            digest = RunManager.checkpoint_digest(candidate)
                            if digest:
                                valid = (candidate, digest)
                                break
                        except (OSError, ValueError):
                            continue
                    if valid is None:
                        raise ValueError(
                            "No valid periodic checkpoint to recover; use a new workspace after fixing the failure"
                        )
                    logical.update(
                        state="paused",
                        checkpoint=str(valid[0]),
                        sha256=valid[1],
                        reason="user_requested_retry",
                    )
                elif action == "adopt":
                    if attempt.get("job_id"):
                        raise ValueError("Attempt already has a scheduler job ID")
                    # Explicit adoption still verifies scheduler identity before trusting it.
                    if attempt["scheduler"] == "slurm":
                        data = json.loads(
                            command(["scontrol", "show", "job", value, "--json"])
                        )["jobs"][0]
                        name = data["name"]
                    else:
                        data = json.loads(
                            command(["qstat", "-xf", "-F", "json", value])
                        )["Jobs"][value]
                        name = data["Job_Name"]
                    if name != attempt["token"]:
                        raise ValueError(
                            "Scheduler job does not match submission token"
                        )
                    attempt["job_id"] = value
                    if logical["state"] == "submitting":
                        logical["state"] = "pending"
                else:
                    raise ValueError(f"Unknown operation: {action}")
                logical["generation"] += 1
                atomic_json(run_path, logical)
            if (
                action == "cancel"
                and attempt.get("job_id")
                and not attempt.get("accounted")
            ):
                command(
                    [
                        "scancel" if attempt["scheduler"] == "slurm" else "qdel",
                        attempt["job_id"],
                    ]
                )
        state["generation"] += 1
        atomic_json(path, state)


def status(path):
    """Print pipeline state, automation health and allocation accounting."""
    state = read(path)
    print(f"Pipeline: {state['state']}")
    from .service import health

    print("Automation: " + health(state))
    for record in state["records"]:
        options = record["options"]
        path = (
            manifest_path(options["run_dir"], options.get("continuation_dir"))
            if options.get("continuation") is not None
            else None
        )
        if path and path.exists():
            logical = reconcile(options["run_dir"], options.get("continuation_dir"))
            print(f"{record['name']}: {logical['state']} ({logical.get('reason', '')})")
            if logical.get("detail"):
                print(f"  detail: {logical['detail']}")
            print(
                f"  runtime: {duration(logical['used_seconds'])} / {duration(logical['policy']['runtime_budget'])}"
            )
            print(
                f"  checkpoint: {logical.get('checkpoint', 'none')}; iteration: {logical.get('progress', 'none')}"
            )
            for attempt in logical["attempts"]:
                print(
                    f"  allocation {attempt['number']}: job={attempt['job_id']} token={attempt['token']} "
                    f"request={attempt['request']} accounting={attempt.get('accounting', 'pending')}"
                )
        else:
            print(f"{record['name']}: {record['status']} {record['jobs']}")


def metrics(path, stream=None):
    """Export allocation evidence for fixed/variable/reference comparisons."""
    import csv
    import sys

    writer = csv.DictWriter(
        stream or sys.stdout,
        fieldnames=[
            "stage",
            "attempt",
            "job_id",
            "state",
            "requested_seconds",
            "assigned_seconds",
            "queue_seconds",
            "runtime_seconds",
            "accounting_confirmed",
            "pause_to_commit_seconds",
            "iteration",
            "checkpoint",
            "reason",
        ],
    )
    writer.writeheader()
    for record in read(path)["records"]:
        options = record["options"]
        if options.get("continuation") is None:
            continue
        run_path = manifest_path(options["run_dir"], options.get("continuation_dir"))
        if not run_path.exists():
            continue
        logical = reconcile(options["run_dir"], options.get("continuation_dir"))
        for attempt in logical["attempts"]:
            outcome = attempt.get("outcome", {})
            latency = ""
            if "committed_at" in outcome and "pause_requested_at" in attempt:
                latency = max(
                    0, outcome["committed_at"] - attempt["pause_requested_at"]
                )
            writer.writerow(
                dict(
                    stage=record["name"],
                    attempt=attempt["number"],
                    job_id=attempt["job_id"],
                    state=outcome.get("state", logical["state"]),
                    requested_seconds=attempt["request"]["seconds"],
                    assigned_seconds=attempt.get("assigned_seconds", ""),
                    queue_seconds=(
                        max(0, attempt["start"] - attempt["intent_time"])
                        if "start" in attempt
                        else ""
                    ),
                    runtime_seconds=attempt.get("accounting", {}).get(
                        "seconds", attempt.get("provisional_seconds", "")
                    ),
                    accounting_confirmed=bool(attempt.get("accounted")),
                    pause_to_commit_seconds=latency,
                    iteration=outcome.get("iteration", ""),
                    checkpoint=outcome.get("checkpoint", ""),
                    reason=attempt.get("reason", outcome.get("reason", "")),
                )
            )


def write_tick_script(manifest, basedir):
    """Pin the submission interpreter and container selection for periodic ticks.

    NERSC registration installs a managed timer; no login terminal or
    compute-node scheduler submission is required. Script and manifest share
    filesystem permissions. No credentials or general shell environment are saved.
    """
    manifest = Path(manifest).resolve()
    basedir = Path(basedir).resolve()
    script = manifest.parent / "advance.sh"
    lines = [
        "#!/bin/bash",
        "set -e",
        # scrontab's resource environment must not leak into sbatch/srun.
        'for key in ${!SLURM_@} ${!SBATCH_@} ${!SRUN_@}; do unset "$key"; done',
        "cd " + shlex.quote(str(basedir)),
    ]
    for key in (
        "PATH",
        "SPINE_CONTAINER_TAG",
        "SPINE_CONTAINER_PATH",
        "SPINE_CONTAINER_VERSION",
        "SPINE_LOCAL_PATH",
    ):
        if os.environ.get(key):
            lines.append(f"export {key}={shlex.quote(os.environ[key])}")
    lines.append("source " + shlex.quote(str(basedir / "configure.sh")))
    lines.append(
        "exec "
        + shlex.join(
            [
                sys.executable,
                str(basedir / "scripts/continue_training.py"),
                "advance",
                str(manifest),
            ]
        )
    )
    # Same controller lock prevents concurrent setup from truncating the launcher.
    with locked(manifest):
        temporary = script.with_name("advance-" + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text("\n".join(lines) + "\n")
            temporary.chmod(0o700)
            os.replace(temporary, script)
        finally:
            temporary.unlink(missing_ok=True)


def reconcile(run_dir, continuation_dir=None):
    """Charge completed accounting once and reconcile the training outcome.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Logical training directory.
    continuation_dir : str or pathlib.Path, optional
        Separate directory containing the run manifest.

    Returns
    -------
    dict
        Current run state after any accounting or recovery transition.

    Notes
    -----
    Unknown submissions are discovered by token before further action. Recovery
    requires a verified checkpoint and the configured retry budget; application
    failures never cause automatic resubmission.
    """
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

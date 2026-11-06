"""Short-lived submission-host controller for continuation pipelines.

Stages are deliberately serialized in this first implementation. No consumer is
submitted against an intermediate allocation's successful exit code.
"""

import hashlib
import json
import uuid
from pathlib import Path

from . import continuation as c


def start(context, stages, workspace, dry_run=False, preload=False, warm_start=None):
    from .pipeline import PipelineRunner

    if not workspace:
        raise ValueError("Continuation pipelines require a workspace")
    path = Path(workspace) / ".spine-prod" / "continuation_pipeline.json"
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
        records.append(
            {
                "name": stage["name"],
                "kind": stage.get("kind", "spine"),
                "options": options,
                "status": "waiting",
                "jobs": [],
            }
        )
    # Validate every operational policy before the first scheduler interaction.
    for record in records:
        options = record["options"]
        if options.get("continuation") is not None:
            profile = context.config_mgr.get_profile(
                options.get("profile", "auto"), None
            )
            scheduler = profile.get(
                "scheduler",
                "pbs" if profile.get("site") in ("anl", "polaris") else "slurm",
            )
            c.policy(
                options["continuation"],
                options.get("time", profile["time"]),
                scheduler,
                profile.get("continuation_defaults"),
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
    with c.locked(path):
        if path.exists():
            state = c.read(path)
            if state["definition_sha256"] != digest:
                raise ValueError("Controller definition changed; use a new workspace")
        else:
            c.atomic_json(
                path,
                {
                    "contract": 1,
                    "state": "active",
                    "definition_sha256": digest,
                    "basedir": str(context.basedir),
                    "records": records,
                    "preload": preload,
                    "generation": 0,
                },
            )
    print(f"Durable controller: {path}")
    print(
        f"Advance periodically on submission host: python3 scripts/continue_training.py advance {path}"
    )
    return advance(path, context)


def advance(path, context=None):
    """Perform one bounded pass. Safe to invoke concurrently from a site timer."""
    from .submitter import Submitter

    path = Path(path)
    with c.locked(path):
        state = c.read(path)
        if state["state"] != "active":
            return {}
        context = context or Submitter(basedir=Path(state["basedir"]))
        for record in state["records"]:
            if record["status"] == "completed":
                continue
            options = dict(record["options"])
            is_continuation = options.get("continuation") is not None
            run_manifest = (
                c.manifest_path(options["run_dir"]) if is_continuation else None
            )
            if run_manifest and run_manifest.exists():
                logical = c.reconcile(options["run_dir"])
                attempt = logical["attempts"][-1]
                if attempt.get("job_id"):
                    record["jobs"] = [attempt["job_id"]]
                if logical["state"] == "completed" and attempt.get("accounted"):
                    record["status"] = "completed"
                    c.atomic_json(path, state)
                    continue
                if logical["state"] != "paused" or not attempt.get("accounted"):
                    c.atomic_json(path, state)
                    return {record["name"]: record["jobs"]}
                # Resume exactly the verified committed continuation artifact.
                options.update(resume=True, resume_from=logical["checkpoint"])
            elif record["status"] == "submitted":
                results = [
                    c.accounting(record["scheduler"], job) for job in record["jobs"]
                ]
                if any(result is None for result in results):
                    return {record["name"]: record["jobs"]}
                if not all(result["success"] for result in results):
                    record["status"] = "failed"
                    c.atomic_json(path, state)
                    return {}
                record["status"] = "completed"
                c.atomic_json(path, state)
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
            c.atomic_json(path, state)
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
            c.atomic_json(path, state)
            return {record["name"]: jobs}
        state["state"] = "completed"
        state["generation"] += 1
        c.atomic_json(path, state)
        return {}


def operate(path, action, stage=None, value=None):
    """Serialize user operations with submission, including cancellation races."""
    path = Path(path)
    with c.locked(path):
        state = c.read(path)
        records = [r for r in state["records"] if stage is None or r["name"] == stage]
        if not records:
            raise ValueError(f"Unknown stage: {stage}")
        if action != "cancel" and (stage is None or len(records) != 1):
            raise ValueError("This operation requires --stage")
        if action == "cancel":
            state["state"] = "cancelled"
            c.atomic_json(path, state)  # Inhibit submissions even if qdel fails.
        for record in records:
            options = record["options"]
            run_path = (
                c.manifest_path(options["run_dir"])
                if options.get("continuation") is not None
                else None
            )
            if not run_path or not run_path.exists():
                if action != "cancel":
                    raise ValueError("Stage has no continuation manifest")
                for job in record["jobs"]:
                    c.command(
                        ["scancel" if record["scheduler"] == "slurm" else "qdel", job]
                    )
                continue
            with c.locked(run_path):
                logical = c.read(run_path)
                attempt = logical["attempts"][-1]
                if action == "extend":
                    budget = c.seconds(value)
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
                        c.RunManager.checkpoint_digest(Path(logical["checkpoint"]))
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
                        c.RunManager.checkpoints(Path(options["run_dir"]))
                    ):
                        try:
                            digest = c.RunManager.checkpoint_digest(candidate)
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
                            c.command(["scontrol", "show", "job", value, "--json"])
                        )["jobs"][0]
                        name = data["name"]
                    else:
                        data = json.loads(
                            c.command(["qstat", "-xf", "-F", "json", value])
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
                c.atomic_json(run_path, logical)
            if (
                action == "cancel"
                and attempt.get("job_id")
                and not attempt.get("accounted")
            ):
                c.command(
                    [
                        "scancel" if attempt["scheduler"] == "slurm" else "qdel",
                        attempt["job_id"],
                    ]
                )
        state["generation"] += 1
        c.atomic_json(path, state)


def status(path):
    state = c.read(path)
    print(f"Pipeline: {state['state']}")
    for record in state["records"]:
        options = record["options"]
        path = (
            c.manifest_path(options["run_dir"])
            if options.get("continuation") is not None
            else None
        )
        if path and path.exists():
            logical = c.reconcile(options["run_dir"])
            print(f"{record['name']}: {logical['state']} ({logical.get('reason', '')})")
            print(
                f"  runtime: {c.duration(logical['used_seconds'])} / {c.duration(logical['policy']['runtime_budget'])}"
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
    for record in c.read(path)["records"]:
        options = record["options"]
        if options.get("continuation") is None:
            continue
        run_path = c.manifest_path(options["run_dir"])
        if not run_path.exists():
            continue
        logical = c.reconcile(options["run_dir"])
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
                    reason=outcome.get("reason", ""),
                )
            )

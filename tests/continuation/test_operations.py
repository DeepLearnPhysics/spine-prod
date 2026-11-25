"""Controller recovery commands, durable transitions and stage gating."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.continuation import controller as ctl
from src.continuation import service
from src.continuation import state as storage

from .helpers import allocation, checkpoint, config


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    """Create a pipeline with a verified paused training allocation."""
    run = tmp_path / "train"
    attempt = allocation(run, 1)
    runpath = storage.manifest_path(run)
    logical = storage.read(runpath)
    cp, digest = checkpoint(run, 10)
    logical.update(state="paused", checkpoint=str(cp), sha256=digest, progress=10)
    logical["attempts"][0]["accounted"] = True
    storage.atomic_json(runpath, logical)
    record = dict(
        name="train",
        kind="spine",
        status="submitted",
        jobs=["1"],
        scheduler="slurm",
        options=dict(run_dir=str(run), continuation={}, profile="test"),
    )
    path = tmp_path / "pipeline.json"
    storage.atomic_json(
        path,
        dict(state="active", generation=0, basedir=str(tmp_path), records=[record]),
    )
    monkeypatch.setattr(ctl, "command", Mock(return_value=""))
    return path, runpath, attempt


def rewrite(path, **values):
    """Update only the fields relevant to a transition under test."""
    data = storage.read(path)
    data.update(values)
    storage.atomic_json(path, data)
    return data


@pytest.mark.parametrize(
    "action,stage,message",
    [
        ("extend", "missing", "Unknown stage"),
        ("extend", None, "requires --stage"),
        ("unknown", "train", "Unknown operation"),
    ],
)
def test_operation_rejects_invalid_target(controlled, action, stage, message):
    """Recovery commands must identify one known stage and action."""
    path, _, _ = controlled
    with pytest.raises(ValueError, match=message):
        ctl.operate(path, action, stage, "00:20:00")


@pytest.mark.parametrize("sched", ["slurm", "pbs"])
def test_cancel_noncontinuation_jobs(controlled, sched):
    """Cancellation inhibits the pipeline before cancelling ordinary jobs."""
    path, _, _ = controlled
    data = storage.read(path)
    record = data["records"][0]
    record["options"].pop("continuation")
    record["scheduler"] = sched
    storage.atomic_json(path, data)
    with pytest.raises(ValueError, match="no continuation"):
        ctl.operate(path, "extend", "train", "01:00:00")
    ctl.operate(path, "cancel")
    assert storage.read(path)["state"] == "cancelled"
    ctl.command.assert_called_once_with(
        ["scancel" if sched == "slurm" else "qdel", "1"]
    )


def test_extend_and_accept_verified_state(controlled):
    """An extension preserves a paused checkpoint; acceptance verifies its digest."""
    path, runpath, _ = controlled
    with pytest.raises(ValueError, match="larger"):
        ctl.operate(path, "extend", "train", "00:01:00")
    ctl.operate(path, "extend", "train", "00:20:00")
    assert storage.read(runpath)["budget_override"] == 1200
    ctl.operate(path, "accept", "train")
    assert storage.read(runpath)["reason"] == "user_accepted"
    with pytest.raises(ValueError, match="Accept requires"):
        ctl.operate(path, "accept", "train")
    rewrite(runpath, state="paused", sha256="wrong")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        ctl.operate(path, "accept", "train")


@pytest.mark.parametrize(
    "mode", ["active", "limit", "missing", "corrupt", "unsigned", "good"]
)
def test_retry_requires_terminal_accounting_and_verified_weights(
    controlled, monkeypatch, mode
):
    """Retry cannot resurrect a live attempt or use a damaged checkpoint."""
    path, runpath, _ = controlled
    data = storage.read(runpath)
    data["state"] = "failed"
    if mode == "active":
        data["attempts"][0]["accounted"] = False
    if mode == "limit":
        data["policy"]["max_attempts"] = 1
    storage.atomic_json(runpath, data)
    if mode == "missing":
        monkeypatch.setattr(ctl.RunManager, "checkpoints", lambda _: [])
    if mode == "corrupt":
        monkeypatch.setattr(
            ctl.RunManager, "checkpoint_digest", Mock(side_effect=ValueError("bad"))
        )
    if mode == "unsigned":
        monkeypatch.setattr(ctl.RunManager, "checkpoint_digest", lambda _: None)
    if mode != "good":
        with pytest.raises(ValueError):
            ctl.operate(path, "retry", "train")
    else:
        ctl.operate(path, "retry", "train")
        assert storage.read(runpath)["reason"] == "user_requested_retry"


@pytest.mark.parametrize("scheduler", ["slurm", "pbs"])
@pytest.mark.parametrize("pending", [False, True])
def test_adopt_verifies_identity(controlled, monkeypatch, scheduler, pending):
    """Adoption requires the exact submission token from the scheduler."""
    path, runpath, attempt = controlled
    with pytest.raises(ValueError, match="already has"):
        ctl.operate(path, "adopt", "train", "99")
    data = storage.read(runpath)
    data["state"] = "pending" if pending else "submitting"
    data["attempts"][0].update(job_id=None, scheduler=scheduler)
    storage.atomic_json(runpath, data)
    name = ["wrong"]
    monkeypatch.setattr(
        ctl,
        "command",
        lambda _: json.dumps(
            {"jobs": [{"name": name[0]}], "Jobs": {"99": {"Job_Name": name[0]}}}
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        ctl.operate(path, "adopt", "train", "99")
    name[0] = attempt["token"]
    ctl.operate(path, "adopt", "train", "99")
    assert storage.read(runpath)["state"] == "pending"
    assert storage.read(runpath)["attempts"][0]["job_id"] == "99"


def test_cancel_live_training_requests_scheduler_cancellation(controlled):
    """Cancellation of a live allocation is durable before scheduler contact."""
    path, runpath, _ = controlled
    data = storage.read(runpath)
    data["attempts"][0]["accounted"] = False
    storage.atomic_json(runpath, data)
    ctl.operate(path, "cancel", "train")
    assert storage.read(runpath)["state"] == "cancelled"
    ctl.command.assert_called_once_with(["scancel", "1"])


@pytest.mark.parametrize("state", ["pending", "completed", "failed"])
def test_ordinary_stage_accounting_gates_consumers(controlled, monkeypatch, state):
    """Ordinary stages also wait for confirmed successful accounting."""
    path, _, _ = controlled
    data = storage.read(path)
    data["records"][0]["options"].pop("continuation")
    storage.atomic_json(path, data)
    monkeypatch.setattr(
        ctl,
        "accounting",
        lambda *_: None if state == "pending" else {"success": state == "completed"},
    )
    ctl.advance(path, SimpleNamespace())
    observed = storage.read(path)
    assert (
        observed["records"][0]["status"]
        == {"pending": "submitted", "completed": "completed", "failed": "failed"}[state]
    )


def test_uncertain_nontraining_submission_is_not_repeated(controlled):
    """Unresolved submission intent must never issue a duplicate job."""
    path, _, _ = controlled
    data = storage.read(path)
    data["records"][0]["options"].pop("continuation")
    data["records"][0]["status"] = "submitting"
    storage.atomic_json(path, data)
    context = Mock()
    assert ctl.advance(path, context) == {"train": ["1"]}
    context.submit_job.assert_not_called()


@pytest.mark.parametrize("jobs", [[], ["9"]])
def test_controller_submits_nonspine_stage(controlled, jobs):
    """Non-training stage kinds use their own submission method."""
    path, _, _ = controlled
    data = storage.read(path)
    record = data["records"][0]
    record["options"].pop("continuation")
    record.update(kind="filter", status="waiting", jobs=[])
    storage.atomic_json(path, data)
    context = SimpleNamespace(
        config_mgr=SimpleNamespace(get_profile=lambda *_: {"site": "polaris"}),
        submit_filter=Mock(return_value=jobs),
    )
    ctl.advance(path, context)
    assert storage.read(path)["records"][0]["scheduler"] == "pbs"
    assert storage.read(path)["records"][0]["status"] == (
        "submitted" if jobs else "submitting"
    )
    context.submit_filter.assert_called_once()


def test_missing_attempt_fails_closed(controlled):
    """A corrupt logical manifest cannot trigger another submission."""
    path, runpath, _ = controlled
    rewrite(runpath, attempts=[])
    with pytest.raises(RuntimeError, match="no submission attempt"):
        ctl.advance(path, SimpleNamespace())


def test_status_and_metrics_cover_unstarted_and_failed_stages(controlled, capsys):
    """Inspection remains useful before any allocation and after failure."""
    path, runpath, _ = controlled
    rewrite(runpath, detail="diagnostic")
    data = storage.read(path)
    data["records"] += [
        dict(name="next", status="waiting", jobs=[], options={}),
        dict(
            name="unstarted",
            status="waiting",
            jobs=[],
            options={"continuation": {}, "run_dir": str(runpath.parent / "missing")},
        ),
    ]
    storage.atomic_json(path, data)
    ctl.status(path)
    output = capsys.readouterr().out
    assert (
        "diagnostic" in output
        and "next: waiting" in output
        and "allocation 1" in output
    )
    stream = io.StringIO()
    ctl.metrics(path, stream)
    assert "train,1,1" in stream.getvalue()


def test_reconcile_rejects_negative_accounting(controlled, monkeypatch):
    """Bad accounting cannot consume a negative budget."""
    _, runpath, _ = controlled
    data = storage.read(runpath)
    data["attempts"][0]["accounted"] = False
    storage.atomic_json(runpath, data)
    monkeypatch.setattr(ctl, "accounting", lambda *_: {"seconds": -1})
    with pytest.raises(ValueError, match="Invalid scheduler"):
        ctl.reconcile(runpath.parents[1])


def test_reconcile_attaches_failure_diagnostics_and_stops_attempt_limit(
    controlled, monkeypatch
):
    """Supervisor failures retain diagnostics; clean pauses still obey attempt caps."""
    _, runpath, attempt = controlled
    data = storage.read(runpath)
    data["attempts"][0]["accounted"] = False
    data["policy"]["max_attempts"] = 1
    storage.atomic_json(runpath, data)
    failure = Path(attempt["directory"]) / "failure.json"
    storage.atomic_json(failure, {"token": attempt["token"], "error": "disk"})
    monkeypatch.setattr(
        ctl,
        "accounting",
        lambda *_: {"seconds": 1, "success": False, "status": "FAILED"},
    )
    result = ctl.reconcile(runpath.parents[1])
    assert result["attempts"][0]["failure"]["error"] == "disk"
    result["attempts"][0]["accounted"] = False
    result["progress"] = 0
    storage.atomic_json(runpath, result)
    monkeypatch.setattr(
        ctl,
        "accounting",
        lambda *_: {"seconds": 1, "success": True, "status": "COMPLETED"},
    )
    assert ctl.reconcile(runpath.parents[1])["reason"] == "max_attempts_reached"


@pytest.mark.parametrize("scheduler_state", ["submitting", "pending"])
def test_reconcile_adopts_missing_response(controlled, monkeypatch, scheduler_state):
    """A scheduler job found by token is adopted without another submission."""
    _, runpath, _ = controlled
    data = storage.read(runpath)
    data["attempts"][0].update(job_id=None, accounted=False)
    data["state"] = scheduler_state
    storage.atomic_json(runpath, data)
    monkeypatch.setattr(ctl, "discover_submission", lambda *_: "restored")
    monkeypatch.setattr(ctl, "accounting", lambda *_: None)
    result = ctl.reconcile(runpath.parents[1])
    assert result["attempts"][0]["job_id"] == "restored"


def test_start_validates_before_submission(tmp_path, monkeypatch):
    """Invalid controller definitions fail before contacting a scheduler."""
    context = Mock()
    with pytest.raises(ValueError, match="workspace"):
        ctl.start(context, [], None)
    with pytest.raises(ValueError, match="cleanup"):
        ctl.start(context, [{"cleanup": True}], tmp_path)
    with pytest.raises(ValueError, match="explicit profiles"):
        ctl.start(context, [{}], tmp_path)
    context.submit_job.assert_not_called()


def test_start_is_idempotent_and_supports_manual_sites(tmp_path, monkeypatch):
    """Repeated starts reuse a matching manifest but reject changed definitions."""
    monkeypatch.setattr(ctl, "advance", Mock(return_value={}))
    monkeypatch.setattr(ctl, "write_tick_script", Mock())
    context = SimpleNamespace(basedir=tmp_path, submit_filter=Mock())
    stage = {
        "name": "filter",
        "kind": "filter",
        "profile": "test",
        "input_dir": "/input",
        "output_dir": "/output",
        "operation": "select",
    }
    # Submission mapping is independent of the persistent controller identity.
    from src.pipeline import PipelineRunner

    monkeypatch.setattr(PipelineRunner, "_submission_options", lambda *a, **k: {})
    ctl.start(context, [stage], tmp_path)
    ctl.start(context, [stage], tmp_path)
    with pytest.raises(ValueError, match="definition changed"):
        ctl.start(context, [dict(stage, name="changed")], tmp_path)
    ctl.start(context, [stage], tmp_path, dry_run=True)
    context.submit_filter.assert_called_once_with(dry_run=True)


def test_start_rejects_multiple_accounts(tmp_path, monkeypatch):
    """One pipeline cannot register competing account-specific timers."""
    from src.pipeline import PipelineRunner

    monkeypatch.setattr(
        PipelineRunner,
        "_submission_options",
        lambda stage, *a, **k: dict(
            profile=stage["profile"], continuation={}, run_dir=str(tmp_path)
        ),
    )
    context = SimpleNamespace(
        config_mgr=SimpleNamespace(
            get_profile=lambda name, _: {"site": "nersc", "account": name}
        ),
        profiles={},
    )
    monkeypatch.setattr(ctl, "validate_storage", lambda *_: None)
    monkeypatch.setattr(ctl, "profile_policy", lambda *_: config())
    with pytest.raises(ValueError, match="one site/account"):
        ctl.start(context, [dict(name=p, profile=p) for p in ("a", "b")], tmp_path)


@pytest.mark.parametrize("site", ["nersc", "s3df", "anl"])
def test_start_registers_before_advancing(tmp_path, monkeypatch, site):
    """Registration must succeed before the first NERSC allocation is submitted."""
    from src.pipeline import PipelineRunner

    monkeypatch.setattr(
        PipelineRunner,
        "_submission_options",
        lambda *a, **k: dict(continuation={}, run_dir=str(tmp_path)),
    )
    context = SimpleNamespace(
        basedir=tmp_path,
        config_mgr=SimpleNamespace(
            get_profile=lambda *a: {"site": site, "account": "project"}
        ),
        profiles={},
    )
    monkeypatch.setattr(ctl, "validate_storage", Mock())
    monkeypatch.setattr(ctl, "profile_policy", lambda *a: config())
    monkeypatch.setattr(
        service, "service_paths", lambda *a: (tmp_path, tmp_path / "registry")
    )
    register = Mock(return_value=tmp_path / "registry")
    monkeypatch.setattr(service, "register", register)
    monkeypatch.setattr(ctl, "advance", Mock(return_value={}))
    ctl.start(context, [dict(name="train", profile="test")], tmp_path)
    assert register.call_count == (site != "anl")
    ctl.advance.assert_called_once()


def test_unidentified_attempt_remains_pending(controlled, monkeypatch, capsys):
    """Uncertain submissions remain inspectable without duplicating allocations."""
    path, runpath, _ = controlled
    data = storage.read(runpath)
    data["attempts"][0]["job_id"] = None
    data["state"] = "pending"
    storage.atomic_json(runpath, data)
    assert ctl.advance(path, SimpleNamespace()) == {"train": ["1"]}
    ctl.status(path)
    assert "pending" in capsys.readouterr().out


def test_stale_failure_diagnostic_is_ignored(controlled, monkeypatch):
    """Diagnostics from a different attempt cannot contaminate this failure."""
    _, runpath, attempt = controlled
    data = storage.read(runpath)
    data["attempts"][0]["accounted"] = False
    storage.atomic_json(runpath, data)
    storage.atomic_json(Path(attempt["directory"]) / "failure.json", {"token": "stale"})
    monkeypatch.setattr(
        ctl,
        "accounting",
        lambda *a: {"seconds": 1, "success": False, "status": "FAILED"},
    )
    result = ctl.reconcile(runpath.parents[1])
    assert "failure" not in result["attempts"][0]


@pytest.mark.parametrize(
    "sibling, expected_status",
    [("RUNNING", "submitted"), ("FAILED", "failed"), ("COMPLETED", "completed")],
)
def test_array_stage_gates_consumers(controlled, monkeypatch, sibling, expected_status):
    """Only complete success of a cache array releases the next pipeline stage."""
    from src.continuation import scheduler

    path, _, _ = controlled
    data = storage.read(path)
    record = data["records"][0]
    record["options"].pop("continuation")
    record["jobs"] = ["42"]
    data["records"].append(
        dict(name="consumer", kind="filter", status="waiting", jobs=[], options={})
    )
    storage.atomic_json(path, data)
    monkeypatch.setattr(
        scheduler,
        "command",
        lambda _: f"42_1|COMPLETED|10|0:0\n42_2|{sibling}|20|0:0\n",
    )
    context = SimpleNamespace(
        config_mgr=SimpleNamespace(get_profile=lambda *a: {"scheduler": "slurm"}),
        submit_filter=Mock(return_value=["99"]),
    )
    ctl.advance(path, context)
    result = storage.read(path)
    assert result["records"][0]["status"] == expected_status
    assert context.submit_filter.call_count == (sibling == "COMPLETED")
    assert result["records"][1]["status"] == (
        "submitted" if sibling == "COMPLETED" else "waiting"
    )

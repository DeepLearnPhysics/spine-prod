"""Fail-closed storage, scheduler and checkpoint integrity checks."""

import errno
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.continuation import checkpoints, policy, scheduler, state

from .helpers import allocation, checkpoint, config


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("state", "invalid", "Invalid SPINE"),
        ("checkpoint", "/outside/run.ckpt", "outside"),
        ("sha256", "incorrect", "matching SHA256"),
        ("iteration", True, "progress"),
        ("iteration", -2, "progress"),
    ],
)
def test_outcome_rejects_invalid_commit(tmp_path, field, value, message):
    """A successful launcher must not make malformed checkpoint metadata valid."""
    attempt = allocation(tmp_path, 1)
    path = Path(attempt["directory"]) / "outcome.json"
    data = state.read(path)
    data[field] = value
    state.atomic_json(path, data)
    with pytest.raises(ValueError, match=message):
        checkpoints.verify_outcome(tmp_path, attempt)


@pytest.mark.parametrize("bad_digest", [None, ValueError("broken sidecar")])
def test_recovery_skips_uncommitted_newer_snapshot(tmp_path, monkeypatch, bad_digest):
    """Recovery must not treat unverified snapshots as resumable state."""
    checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 2)
    lookup = Mock(side_effect=[bad_digest])
    monkeypatch.setattr(checkpoints.RunManager, "checkpoint_digest", lookup)
    with pytest.raises(ValueError, match="No new verified"):
        checkpoints.recovery_checkpoint(tmp_path, {"start_progress": 1})


def test_storage_probe_detects_readback_corruption(tmp_path, monkeypatch):
    """Fail preflight and remove its probe when publication cannot be read back."""
    monkeypatch.setattr(state, "read", lambda _: {})
    with pytest.raises(RuntimeError, match="verification"):
        state.preflight_storage(tmp_path)
    assert not list(tmp_path.rglob("preflight-*.json"))


def test_unexpected_lock_error_is_preserved(tmp_path, monkeypatch):
    """Permission failures must retain their original diagnostic."""
    monkeypatch.setattr(
        state.fcntl, "flock", Mock(side_effect=OSError(errno.EACCES, "denied"))
    )
    with pytest.raises(OSError) as error:
        with state.locked(tmp_path / "manifest"):
            pytest.fail("lock unexpectedly acquired")
    assert error.value.errno == errno.EACCES


@pytest.mark.parametrize("limit", ["attempts", "budget"])
def test_reservation_persists_exhaustion(tmp_path, limit):
    """Persist terminal limits before refusing another scheduler submission."""
    allocation(tmp_path, 1)
    path = state.manifest_path(tmp_path)
    data = state.read(path)
    data["state"] = "paused"
    data["attempts"][-1]["accounted"] = True
    data["used_seconds"] = 720 if limit == "budget" else 0
    state.atomic_json(path, data)
    with pytest.raises(ValueError, match="exhausted|attempts"):
        state.reserve(
            tmp_path,
            "scientific-identity",
            config(max_attempts=1 if limit == "attempts" else 10),
            "slurm",
            tmp_path / "next",
        )
    assert state.read(path)["state"] == (
        "failed" if limit == "attempts" else "budget_exhausted"
    )


def test_submission_response_identity_is_immutable(tmp_path):
    """Stale and conflicting responses cannot overwrite a durable scheduler ID."""
    attempt = allocation(tmp_path, 1)
    with pytest.raises(ValueError, match="Stale"):
        state.submitted(tmp_path, "stale", "1")
    with pytest.raises(ValueError, match="another job"):
        state.submitted(tmp_path, attempt["token"], "2")
    state.submitted(tmp_path, attempt["token"], None)
    state.submitted(tmp_path, attempt["token"], "1")
    assert state.read(state.manifest_path(tmp_path))["attempts"][-1]["job_id"] == "1"


@pytest.mark.parametrize(
    "values,maximum,sched",
    [
        ([], "00:20:00", "slurm"),
        ({"runtime_budget": "01:00:00"}, "00:20:00", "unknown"),
        ({}, "00:20:00", "slurm"),
        (
            {"runtime_budget": "30:00:00", "checkpoint_margin": "20:00:00"},
            "24:00:00",
            "slurm",
        ),
        (
            {
                "runtime_budget": "01:00:00",
                "site_min_time": "00:30:00",
                "checkpoint_margin": "00:01:00",
            },
            "00:20:00",
            "slurm",
        ),
        (
            {"runtime_budget": "00:00:01", "checkpoint_margin": "00:01:00"},
            "00:20:00",
            "slurm",
        ),
    ],
)
def test_policy_rejects_unusable_requests(values, maximum, sched):
    """Invalid policies fail before any scheduler interaction."""
    with pytest.raises(ValueError):
        policy.policy(values, maximum, sched)


def test_scheduler_command_is_bounded(monkeypatch):
    """Scheduler subprocesses must have finite query timeouts."""
    run = Mock(return_value=SimpleNamespace(stdout="reply"))
    monkeypatch.setattr(scheduler.subprocess, "run", run)
    assert scheduler.command(["sacct"]) == "reply"
    assert run.call_args.kwargs["timeout"] == 30


def test_discovery_requires_exact_token_and_supports_pbs(monkeypatch):
    """Ignore unrelated jobs and reject malformed attempt identifiers."""
    with pytest.raises(ValueError, match="token"):
        scheduler.discover_submission("slurm", "invalid")
    token = "spc-" + "a" * 20
    monkeypatch.setattr(
        scheduler,
        "command",
        lambda _: json.dumps({"Jobs": {"12.server": {"Job_Name": token}, "other": {}}}),
    )
    assert scheduler.discover_submission("pbs", token) == "12.server"
    monkeypatch.setattr(
        scheduler,
        "command",
        lambda _: "malformed\n12|other\n12.batch|" + token + "\n13|" + token,
    )
    assert scheduler.discover_submission("slurm", token) == "13"


def test_pbs_timestamp_and_terminal_accounting(monkeypatch):
    """Read PBS native timestamp strings and reject missing elapsed time."""
    data = {
        "stime": "Wed Nov 18 01:00:00 2026",
        "Resource_List": {"walltime": "01:00:00"},
    }
    monkeypatch.setattr(scheduler.time, "time", lambda: 9999999999)
    monkeypatch.setattr(
        scheduler, "command", lambda _: json.dumps({"Jobs": {"job": data}})
    )
    start, end = scheduler.allocation("pbs", "job")
    assert end - start == 3600
    data.update(job_state="F", Exit_status=0)
    with pytest.raises(ValueError, match="missing walltime"):
        scheduler.accounting("pbs", "job")
    data["resources_used"] = {"walltime": "00:01:30"}
    assert scheduler.accounting("pbs", "job")["seconds"] == 90
    data["resources_used"]["walltime"] = "00:00:00"
    assert scheduler.accounting("pbs", "job")["seconds"] == 0
    del data["resources_used"]
    del data["stime"]
    assert scheduler.accounting("pbs", "job")["seconds"] == 0
    data["job_state"] = "R"
    assert scheduler.accounting("pbs", "job") is None
    data.update(job_state="F")
    del data["Exit_status"]
    assert scheduler.accounting("pbs", "job") is None


def test_slurm_accounting_ignores_steps_and_live_jobs(monkeypatch):
    """Only the exact terminal allocation row is authoritative."""
    monkeypatch.setattr(
        scheduler, "command", lambda _: "bad\n1.batch|COMPLETED|5|0:0\n1|RUNNING|5|0:0"
    )
    assert scheduler.accounting("slurm", "1") is None
    monkeypatch.setattr(
        scheduler,
        "command",
        lambda _: json.dumps({"jobs": [{"start_time": 100, "end_time": 50}]}),
    )
    with pytest.raises(ValueError, match="deadline"):
        scheduler.allocation("slurm", "1")


def test_site_minimum_cannot_be_bypassed_by_stage_budget(monkeypatch):
    """A stage's smaller minimum cannot make an unusable final allocation valid."""
    from src import sites

    monkeypatch.setattr(sites, "allocation_limits", lambda *a: {"min_time": "00:05:00"})
    monkeypatch.setattr(sites, "validate_allocation", lambda *a: None)
    with pytest.raises(ValueError, match="site's minimum"):
        policy.profile_policy(
            {},
            {"time": "00:10:00"},
            {
                "runtime_budget": "00:04:00",
                "site_min_time": "00:01:00",
                "checkpoint_margin": "00:00:10",
                "startup_allowance": "00:00:10",
            },
        )


@pytest.mark.parametrize(
    "sibling, expected", [("RUNNING", None), ("FAILED", False), ("COMPLETED", True)]
)
def test_array_accounting_requires_all_tasks(monkeypatch, sibling, expected):
    """An array's raw leader ID cannot establish success of its sibling tasks."""

    def query(argv):
        # Raw allocation IDs differ; JobID retains the shared array identity.
        identifiers = (
            ["42", "43"]
            if "--format=JobIDRaw,State,ElapsedRaw,ExitCode" in argv
            else ["42_1", "42_2"]
        )
        return f"{identifiers[0]}|COMPLETED|10|0:0\n{identifiers[1]}|{sibling}|20|0:0\n"

    monkeypatch.setattr(scheduler, "command", query)
    result = scheduler.accounting("slurm", "42")
    if expected is None:
        assert result is None
    else:
        assert result["success"] is expected
        assert result["seconds"] == 30


@pytest.mark.parametrize(
    "rows",
    [
        "",
        "42_0|COMPLETED|10|0:0\n42_7|PENDING|0|0:0",
        "42_7|RUNNING|20|0:0\n42_0|COMPLETED|10|0:0",
    ],
)
def test_array_accounting_waits_for_complete_terminal_records(monkeypatch, rows):
    """Sparse and zero-based arrays remain pending while any task is live."""
    monkeypatch.setattr(scheduler, "command", lambda _: rows)
    assert scheduler.accounting("slurm", "42") is None


def test_array_accounting_preserves_nonzero_exit_and_ignores_steps(monkeypatch):
    """Task exit codes remain authoritative even when the state says completed."""
    calls = []

    def query(argv):
        calls.append(argv)
        return (
            "42_0|COMPLETED|10|0:0\n42_7|COMPLETED|20|1:0\n"
            "42_7.batch|FAILED|20|1:0\n420_1|RUNNING|100|0:0"
        )

    monkeypatch.setattr(scheduler, "command", query)
    result = scheduler.accounting("slurm", "42")
    assert result == dict(
        status="COMPLETED", seconds=30, success=False, exit_code="1:0"
    )
    assert "--array" in calls[0]
    assert "--format=JobID,State,ElapsedRaw,ExitCode" in calls[0]


def test_single_array_task_query_does_not_include_siblings(monkeypatch):
    """An explicitly selected task is reconciled independently of its siblings."""
    monkeypatch.setattr(
        scheduler, "command", lambda _: "42_0|COMPLETED|10|0:0\n42_7|RUNNING|20|0:0"
    )
    assert scheduler.accounting("slurm", "42_0")["seconds"] == 10

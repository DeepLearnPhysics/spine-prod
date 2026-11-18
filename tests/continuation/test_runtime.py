"""Supervisor signal, deadline-discovery and failure-publication contracts."""

import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.continuation import runtime
from src.continuation import state as storage

from .helpers import config


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    """Create a pending allocation with a controlled clock and signal delivery."""
    directory = tmp_path / "attempt"
    directory.mkdir()
    attempt = storage.reserve(tmp_path, "identity", config(), "slurm", directory)
    storage.submitted(tmp_path, attempt["token"], "123")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    handlers = {}
    monkeypatch.setattr(
        runtime.signal, "signal", lambda sig, fn: handlers.update({sig: fn})
    )
    monkeypatch.setattr(runtime.time, "time", lambda: 100)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    monkeypatch.setattr(runtime, "allocation", lambda *_: (100, 340))
    child = Mock(pid=42, returncode=0)
    child.poll.return_value = 0
    monkeypatch.setattr(runtime.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(runtime.os, "killpg", Mock())
    return tmp_path, attempt, handlers, child


def test_runtime_rejects_stale_identity(supervisor):
    root, attempt, _, _ = supervisor
    with pytest.raises(ValueError, match="Stale"):
        runtime.run(root, "wrong", ["train"])
    path = storage.manifest_path(root)
    state = storage.read(path)
    state["attempts"][-1]["job_id"] = "other"
    storage.atomic_json(path, state)
    with pytest.raises(ValueError, match="Allocation ID"):
        runtime.run(root, attempt["token"], ["train"])


@pytest.mark.parametrize(
    "failure", ["discovery", "stale", "cancelled", "short", "terminated"]
)
def test_runtime_startup_failure_never_launches(supervisor, monkeypatch, failure):
    root, attempt, handlers, _ = supervisor

    def discover(*_):
        """Inject failure at the scheduler observation boundary."""
        if failure == "discovery":
            raise OSError("scheduler unavailable")
        if failure == "stale":
            (Path(attempt["directory"]) / "pause").touch()
        if failure == "cancelled":
            path = storage.manifest_path(root)
            state = storage.read(path)
            state["state"] = "cancelled"
            storage.atomic_json(path, state)
        if failure == "terminated":
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return (100, 105 if failure == "short" else 340)

    monkeypatch.setattr(runtime, "allocation", discover)
    with pytest.raises((ValueError, RuntimeError)):
        runtime.run(root, attempt["token"], ["train"])
    runtime.subprocess.Popen.assert_not_called()


@pytest.mark.parametrize("cancel_during_pause", [False, True])
def test_runtime_forwards_signals_once(supervisor, monkeypatch, cancel_during_pause):
    root, attempt, handlers, child = supervisor
    calls = []

    def poll():
        """Deliver simultaneous graceful, pause and termination requests."""
        calls.append(1)
        if len(calls) == 1:
            for sig in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM):
                handlers[sig](sig, None)
            if cancel_during_pause:
                path = storage.manifest_path(root)
                state = storage.read(path)
                state["state"] = "cancelled"
                storage.atomic_json(path, state)
        return None if len(calls) <= 2 else 0

    child.poll.side_effect = poll
    assert runtime.run(root, attempt["token"], ["train"]) == 0
    assert (Path(attempt["directory"]) / "pause").exists()
    assert (Path(attempt["directory"]) / "graceful_stop").exists()
    runtime.os.killpg.assert_called_once_with(42, signal.SIGTERM)
    state = storage.read(storage.manifest_path(root))
    assert state["state"] == ("cancelled" if cancel_during_pause else "pausing")
    assert state["attempts"][-1]["launcher_exit"] == 0


@pytest.mark.parametrize("kill", [False, True])
def test_runtime_cleans_up_child_when_polling_fails(supervisor, monkeypatch, kill):
    root, attempt, _, child = supervisor
    child.poll.return_value = None
    monkeypatch.setattr(
        runtime.time, "sleep", Mock(side_effect=OSError("poll failure"))
    )
    if kill:
        child.wait.side_effect = [subprocess.TimeoutExpired("train", 10), 0]
    with pytest.raises(OSError, match="poll failure"):
        runtime.run(root, attempt["token"], ["train"])
    assert runtime.os.killpg.call_args_list[0].args == (42, signal.SIGTERM)
    assert runtime.os.killpg.call_count == (2 if kill else 1)


def test_runtime_does_not_overwrite_replaced_attempt(supervisor):
    root, attempt, _, child = supervisor

    def poll():
        """Simulate replacement before the old supervisor publishes its exit."""
        path = storage.manifest_path(root)
        state = storage.read(path)
        state["attempts"][-1]["token"] = "replacement"
        storage.atomic_json(path, state)
        return 0

    child.poll.side_effect = poll
    runtime.run(root, attempt["token"], ["train"])
    assert (
        "launcher_exit" not in storage.read(storage.manifest_path(root))["attempts"][-1]
    )


@pytest.mark.parametrize("separator", [False, True])
def test_runtime_cli_exit_status(monkeypatch, separator):
    argv = ["runtime", "run", "token"] + (["--"] if separator else []) + ["train"]
    monkeypatch.setattr(runtime.sys, "argv", argv)
    run = Mock(return_value=7)
    monkeypatch.setattr(runtime, "run", run)
    with pytest.raises(SystemExit) as error:
        runtime.main()
    assert error.value.code == 7
    assert run.call_args.args == ("run", "token", ["train"])


@pytest.mark.parametrize("mode", ["matching", "stale", "missing"])
def test_runtime_cli_publishes_only_matching_failure(
    tmp_path, monkeypatch, capsys, mode
):
    attempt = storage.reserve(
        tmp_path, "identity", config(), "slurm", tmp_path / "attempt"
    )
    token = attempt["token"] if mode != "stale" else "stale"
    if mode == "missing":
        storage.manifest_path(tmp_path).unlink()
    monkeypatch.setattr(runtime.sys, "argv", ["runtime", str(tmp_path), token, "train"])
    monkeypatch.setattr(runtime, "run", Mock(side_effect=OSError("launcher failed")))
    with pytest.raises(SystemExit) as error:
        runtime.main()
    assert error.value.code == 1
    assert (tmp_path / "attempt/failure.json").exists() == (mode == "matching")
    assert "launcher failed" in capsys.readouterr().err

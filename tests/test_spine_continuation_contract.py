"""Run against patched SPINE with PYTHONPATH=<patched-checkout>/src.

The real Driver loop/checkpoint methods are exercised with CPU-only model and
loader doubles. GPU/DDP restart equivalence remains a site integration check.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("spine.continuation") is None,
    reason="Requires SPINE continuation-v1 patch; see docs/training-continuation.md",
)


def driver(tmp_path, monkeypatch, *, start=0, limit=10, scheduled=False, early=False):
    from spine.control import RunControl
    from spine.driver import Driver
    from spine.utils.stopwatch import StopwatchManager

    marker = tmp_path / "pause"
    outcome = tmp_path / "outcome.json"
    monkeypatch.setenv("SPINE_CONTINUATION_TOKEN", "attempt-1")
    monkeypatch.setenv("SPINE_PAUSE_FILE", str(marker))
    monkeypatch.setenv("SPINE_OUTCOME_FILE", str(outcome))
    drv = object.__new__(Driver)
    history = {
        "metrics": {"loss": 0.5},
        "early_stopping": {"bad_checks": 2},
        "best_checkpoint": {"best": 0.5},
    }
    saves = []

    def save(iteration, epoch, validation, **kwargs):
        cp = tmp_path / f"snapshot-{iteration}.ckpt"
        cp.write_bytes(f"step-{iteration}".encode())
        Path(str(cp) + ".sha256").write_text(
            f"{hashlib.sha256(cp.read_bytes()).hexdigest()}  {cp.name}\n"
        )
        saves.append((iteration, validation, kwargs))
        return str(cp)

    drv.model = SimpleNamespace(
        train=True,
        start_iteration=start,
        start_epoch=float(start),
        device=None,
        checkpoint_validation=history,
        should_save=lambda _: scheduled,
        step_scheduler=Mock(),
        save_state=save,
        save_best_state=Mock(),
        watch=StopwatchManager(),
    )
    drv.validation = SimpleNamespace(
        initial=False,
        num_iterations=1,
        io=SimpleNamespace(dataset_provenance=lambda: {}),
        run=Mock(return_value={"loss": 0.6}),
        update_best_checkpoint=Mock(return_value=False),
        update_early_stopping=Mock(return_value=early),
        checkpoint_state=Mock(return_value=history),
        close=Mock(),
    )
    drv.iterations = limit
    drv.epochs = None
    drv.run_control = RunControl()
    drv.main_process = True
    drv.distributed = False
    drv.rank = None
    drv.world_size = 1
    drv.cfg = {}
    drv.io = SimpleNamespace(
        has_loader=True,
        iter_per_epoch=1,
        prepare_iteration=Mock(),
        checkpoint_state=lambda next_iteration: {"next_iteration": next_iteration},
        dataset_provenance=lambda: {},
        close=Mock(),
    )
    drv.log_manager = SimpleNamespace(append_tensorboard=Mock(), close=Mock())
    drv.watch = StopwatchManager()
    drv.initialize_log = Mock()
    drv.reduce_training_metrics = Mock()
    drv.log_stdout = Mock()
    drv.log = Mock()
    drv.cleanup = Mock()

    def process(**kwargs):
        marker.touch()
        return {"loss": 1.0}

    drv.process = process
    return drv, saves, outcome, history


def test_pause_preserves_validation_patience_and_runtime(tmp_path, monkeypatch):
    drv, saves, outcome, history = driver(tmp_path, monkeypatch)
    drv.run()
    result = json.loads(outcome.read_text())
    assert result["state"] == "paused"
    assert result["iteration"] == 0
    drv.validation.run.assert_not_called()
    drv.validation.update_early_stopping.assert_not_called()
    drv.model.save_best_state.assert_not_called()
    assert all(
        call.args[0] != "validation" for call in drv.model.step_scheduler.call_args_list
    )
    assert saves[0][1] == history
    assert saves[0][2]["runtime_state"]["ranks"][0]["io"]["next_iteration"] == 1
    assert saves[0][2]["completion"] is None
    drv.cleanup.assert_called_once()


@pytest.mark.parametrize("early", [True, False])
def test_terminal_wins_over_pause(tmp_path, monkeypatch, early):
    drv, saves, outcome, _ = driver(
        tmp_path, monkeypatch, limit=10 if early else 1, scheduled=early, early=early
    )
    drv.run()
    result = json.loads(outcome.read_text())
    assert result["state"] == "completed"
    assert result["reason"] == ("early_stopping" if early else "training_limit")
    assert len(saves) == 1


def test_pause_resume_preserves_absolute_iteration_limit(tmp_path, monkeypatch):
    progress = []
    for start in range(3):
        directory = tmp_path / str(start)
        directory.mkdir()
        drv, saves, outcome, _ = driver(directory, monkeypatch, start=start, limit=3)
        drv.run()
        result = json.loads(outcome.read_text())
        progress.append(result["iteration"])
        assert result["state"] == ("completed" if start == 2 else "paused")
    assert progress == [0, 1, 2]


def test_checkpoint_failure_never_publishes_outcome(tmp_path, monkeypatch):
    drv, _, outcome, _ = driver(tmp_path, monkeypatch)
    drv.model.save_state = Mock(side_effect=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        drv.run()
    assert not outcome.exists()

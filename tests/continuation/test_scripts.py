"""Exercise the installed continuation command-line entry points."""

import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.continuation import controller, runtime, service, state

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.mark.parametrize(
    "action",
    [
        "advance",
        "status",
        "extend",
        "accept",
        "cancel",
        "adopt",
        "retry",
        "metrics",
        "preflight",
    ],
)
def test_controller_cli_dispatches_arguments(monkeypatch, tmp_path, capsys, action):
    """Each recovery command forwards its options and prints status when appropriate."""
    manifest = str(tmp_path / "pipeline.json")
    calls = {}
    for name in ("advance", "status", "operate", "metrics"):
        calls[name] = Mock()
        monkeypatch.setattr(controller, name, calls[name])
    preflight = Mock()
    monkeypatch.setattr(state, "preflight_storage", preflight)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "continue_training",
            action,
            manifest,
            "--stage",
            "train",
            "--value",
            "12:00:00",
            "--site",
            "nersc",
        ],
    )
    monkeypatch.syspath_prepend(str(SCRIPTS.parent))
    runpy.run_path(str(SCRIPTS / "continue_training.py"), run_name="__main__")
    if action == "preflight":
        preflight.assert_called_once_with(manifest, "nersc")
        assert "preflight passed" in capsys.readouterr().out
    elif action == "metrics":
        calls["metrics"].assert_called_once_with(manifest)
    else:
        calls["status"].assert_called_once_with(manifest)
        if action == "advance":
            calls["advance"].assert_called_once_with(manifest)
        elif action != "status":
            calls["operate"].assert_called_once_with(
                manifest, action, "train", "12:00:00"
            )
    assert calls["status"].call_count == (action not in ("preflight", "metrics"))


@pytest.mark.parametrize(
    "script, module, method, arguments",
    [
        ("continuation_runtime.py", runtime, "main", []),
        ("continuation_service.py", service, "tick_registry", ["registry.json"]),
    ],
)
def test_scheduler_entrypoint_dispatch(monkeypatch, script, module, method, arguments):
    """Scheduler wrappers invoke the same runtime and registry services as the API."""
    target = Mock()
    monkeypatch.setattr(module, method, target)
    monkeypatch.setattr(sys, "argv", [script, *arguments])
    monkeypatch.syspath_prepend(str(SCRIPTS.parent))
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    target.assert_called_once_with(*arguments)

"""Runtime guards must inspect the executable that will run the workload."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_spine_version

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_spine_version.py"


@pytest.mark.parametrize("version,code", [("1.3.0", 1), ("1.4.0", 0)])
def test_selected_executable_version(tmp_path, version, code):
    executable = tmp_path / "selected spine.py"
    executable.write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['--version']\n"
        f"print('SPINE {version}')\n"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), sys.executable, str(executable)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == code
    if code:
        assert "require SPINE >= 1.4.0" in result.stderr


def test_unverifiable_executable_fails_closed(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "missing-spine")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "runtime check failed" in result.stderr


@pytest.mark.parametrize(
    "output,expected_code",
    [
        ("SPINE 1.4.0\n", 0),
        ("SPINE 1.3.9\n", 1),
        ("SPINE unknown\n", 1),
        ("unexpected output\n", 1),
    ],
)
def test_version_check_main_covers_selected_command(
    monkeypatch, capsys, output, expected_code
):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "python3", "source/bin/run.py"])
    calls = []

    def fake_run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(check_spine_version.subprocess, "run", fake_run)
    assert check_spine_version.main() == expected_code
    assert calls == [
        (
            ["python3", "source/bin/run.py", "--version"],
            {"capture_output": True, "text": True, "check": True},
        )
    ]
    if expected_code:
        assert "SPINE runtime check failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error", [OSError("not found"), subprocess.CalledProcessError(2, "spine")]
)
def test_version_check_main_handles_execution_failure(monkeypatch, capsys, error):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    def fail_run(*_args, **_options):
        raise error

    monkeypatch.setattr(check_spine_version.subprocess, "run", fail_run)
    assert check_spine_version.main() == 1
    assert "SPINE runtime check failed" in capsys.readouterr().err


def test_version_check_script_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr(
        check_spine_version.subprocess,
        "run",
        lambda command, **_options: subprocess.CompletedProcess(
            command, 0, stdout="SPINE 1.4.0\n"
        ),
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(str(SCRIPT), run_name="__main__")

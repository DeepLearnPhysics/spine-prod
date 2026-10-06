"""Runtime guards must inspect the executable that will run the workload."""

import subprocess
import sys
from pathlib import Path

import pytest

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

"""Worker support is independent of the submission Python environment."""

import subprocess
import sys

import pytest

from src import require_python


def test_worker_imports_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "scripts/continuation_runtime.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_separate_worker_and_submission_minimums(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 9, 21))
    require_python((3, 9))
    with pytest.raises(ImportError, match="requires Python 3.10"):
        require_python()
    monkeypatch.setattr(sys, "version_info", (3, 8, 20))
    with pytest.raises(ImportError, match="requires Python 3.9"):
        require_python((3, 9))

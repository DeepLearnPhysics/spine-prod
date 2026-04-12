"""Tests for the preload_downloads utility."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "preload_downloads.py"


@pytest.fixture(name="preload_module")
def fixture_preload_module():
    """Load the preload utility as a module."""
    spec = importlib.util.spec_from_file_location("preload_downloads", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_config_path_accepts_existing_path(preload_module, tmp_path):
    """Test explicit existing config paths are accepted."""
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n")

    resolved = preload_module.resolve_config_path(str(config), tmp_path)

    assert resolved == config.resolve()


def test_resolve_config_path_falls_back_to_repo_config(preload_module, tmp_path):
    """Test config paths can be relative to the repository config directory."""
    config = tmp_path / "config" / "infer" / "example.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: {}\n")

    resolved = preload_module.resolve_config_path("infer/example.yaml", tmp_path)

    assert resolved == config.resolve()


def test_resolve_config_path_raises_for_missing_config(preload_module, tmp_path):
    """Test missing configs fail with a clear exception."""
    with pytest.raises(FileNotFoundError, match="Config not found"):
        preload_module.resolve_config_path("missing.yaml", tmp_path)


def test_import_does_not_import_spine(preload_module):
    """Test the utility does not load SPINE at import time."""
    assert "spine" not in sys.modules


def test_bootstrap_spine_adds_submodule_src(preload_module, tmp_path):
    """Test the bundled SPINE source path is added when present."""
    spine_src = tmp_path / "spine" / "src"
    spine_src.mkdir(parents=True)

    old_path = list(sys.path)
    try:
        preload_module.bootstrap_spine(tmp_path)
        assert str(spine_src) == sys.path[0]
    finally:
        sys.path[:] = old_path


def test_spine_config_import_does_not_import_h5py():
    """Test SPINE config imports do not load runtime IO dependencies."""
    spine_src = SCRIPT_PATH.parents[1] / "spine" / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(spine_src)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from spine.config.load import load_config_file; "
                "print('h5py' in sys.modules)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"

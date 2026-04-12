"""Tests for the preload_downloads utility."""

import importlib.util
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

"""Tests for the preload_downloads utility."""

import importlib.util
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


def test_collect_downloads_from_config_and_includes(preload_module, tmp_path):
    """Test collecting downloads from a config include tree."""
    base = tmp_path / "config" / "infer" / "detector"
    model_dir = base / "model"
    model_dir.mkdir(parents=True)

    full_chain = base / "full_chain.yaml"
    full_chain.write_text("include:\n  - model/model.yaml\n")

    model = model_dir / "model.yaml"
    model.write_text(
        "override:\n"
        "  model.weight_path: !download\n"
        "    url: https://example.com/model.ckpt\n"
        "    hash: abc123\n"
    )

    downloads = preload_module.collect_downloads(full_chain, tmp_path)

    assert len(downloads) == 1
    assert downloads[0].url == "https://example.com/model.ckpt"
    assert downloads[0].expected_hash == "abc123"
    assert downloads[0].source == model.resolve()


def test_url_to_filename_matches_spine_convention(preload_module):
    """Test URL cache filenames match SPINE's convention."""
    filename = preload_module.url_to_filename(
        "https://s3df.slac.stanford.edu/data/neutrino/spine/weights/2x2/"
        "2x2_snapshot_240819.ckpt"
    )

    assert filename == "5bee7a9d1a75a25e.ckpt"


def test_deduplicate_downloads(preload_module, tmp_path):
    """Test repeated downloads are collapsed while preserving order."""
    specs = [
        preload_module.DownloadSpec("https://example.com/a.ckpt", "hash-a", tmp_path),
        preload_module.DownloadSpec("https://example.com/a.ckpt", "hash-a", tmp_path),
        preload_module.DownloadSpec("https://example.com/b.ckpt", "hash-b", tmp_path),
    ]

    deduped = preload_module.deduplicate_downloads(specs)

    assert [spec.url for spec in deduped] == [
        "https://example.com/a.ckpt",
        "https://example.com/b.ckpt",
    ]

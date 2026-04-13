"""Tests for download preloading utilities."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import preload

REPO_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "preload_downloads.py"


def test_resolve_config_path_accepts_existing_path(tmp_path):
    """Test explicit existing config paths are accepted."""
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n")

    resolved = preload.resolve_config_path(str(config), tmp_path)

    assert resolved == config.resolve()


def test_resolve_config_path_falls_back_to_repo_config(tmp_path):
    """Test config paths can be relative to the repository config directory."""
    config = tmp_path / "config" / "infer" / "example.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: {}\n")

    resolved = preload.resolve_config_path("infer/example.yaml", tmp_path)

    assert resolved == config.resolve()


def test_resolve_config_path_raises_for_missing_config(tmp_path):
    """Test missing configs fail with a clear exception."""
    with pytest.raises(FileNotFoundError, match="Config not found"):
        preload.resolve_config_path("missing.yaml", tmp_path)


def test_import_does_not_import_spine():
    """Test importing src.preload does not load SPINE."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; " "import src.preload; " "print('spine' in sys.modules)"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_bootstrap_spine_adds_submodule_src(tmp_path):
    """Test the bundled SPINE source path is added when present."""
    spine_src = tmp_path / "spine" / "src"
    spine_src.mkdir(parents=True)

    old_path = list(sys.path)
    try:
        preload.bootstrap_spine(tmp_path)
        assert str(spine_src) == sys.path[0]
    finally:
        sys.path[:] = old_path


def test_spine_config_import_does_not_import_h5py():
    """Test SPINE config imports do not load runtime IO dependencies."""
    spine_src = REPO_ROOT / "spine" / "src"
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


def test_preload_downloads_loads_config(tmp_path):
    """Test preload_downloads resolves configs and calls SPINE's loader."""
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n")
    calls = []

    def load_config_file(path):
        calls.append(path)

    old_modules = {
        name: sys.modules.get(name)
        for name in [
            "spine",
            "spine.config",
            "spine.config.download",
            "spine.config.load",
        ]
    }

    download_module = type(
        "DownloadModule", (), {"get_cache_dir": staticmethod(lambda: tmp_path)}
    )()
    load_module = type(
        "LoadModule", (), {"load_config_file": staticmethod(load_config_file)}
    )()

    try:
        sys.modules["spine"] = type("SpineModule", (), {})()
        sys.modules["spine.config"] = type("ConfigModule", (), {})()
        sys.modules["spine.config.download"] = download_module
        sys.modules["spine.config.load"] = load_module

        loaded = preload.preload_downloads(str(config), tmp_path)
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert loaded == [config.resolve()]
    assert calls == [str(config.resolve())]


def test_script_wrapper_delegates_to_preload(tmp_path):
    """Test the optional script wrapper delegates to src.preload."""
    spec = importlib.util.spec_from_file_location("preload_downloads", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def fake_preload(configs, project_root, cache_dir=None):
        calls.append((configs, project_root, cache_dir))

    module.load_preload_downloads = lambda: fake_preload

    old_argv = sys.argv
    try:
        sys.argv = [
            str(SCRIPT_PATH),
            "--cache-dir",
            str(tmp_path / "cache"),
            "infer/2x2/full_chain_240819.yaml",
        ]
        assert module.main() == 0
    finally:
        sys.argv = old_argv

    assert calls == [
        (
            ["infer/2x2/full_chain_240819.yaml"],
            REPO_ROOT,
            str(tmp_path / "cache"),
        )
    ]

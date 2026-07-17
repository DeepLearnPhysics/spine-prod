"""Tests for download preloading utilities."""

import builtins
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


def test_bootstrap_spine_is_compatibility_noop(tmp_path):
    """Test the legacy bootstrap hook no longer adds bundled source paths."""
    spine_src = tmp_path / "spine" / "src"
    spine_src.mkdir(parents=True)

    old_path = list(sys.path)
    try:
        preload.bootstrap_spine(tmp_path)
        assert sys.path == old_path
    finally:
        sys.path[:] = old_path


def test_spine_config_loader_imports_when_available():
    """Test an optional local SPINE config loader can be imported."""
    try:
        loader_spec = importlib.util.find_spec("spine.config.load")
    except ModuleNotFoundError:
        loader_spec = None

    if loader_spec is None:
        pytest.skip("SPINE is not installed in this test environment")

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from spine.config.load import load_config_file; "
                "print(callable(load_config_file))"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


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


def test_preload_downloads_sets_environment_and_loads_multiple_configs(
    tmp_path, monkeypatch
):
    """Test cache settings and list inputs are forwarded to SPINE."""
    configs = [tmp_path / "first.yaml", tmp_path / "second.yaml"]
    for config in configs:
        config.touch()
    calls = []

    monkeypatch.setitem(
        sys.modules,
        "spine.config.download",
        type("DownloadModule", (), {"get_cache_dir": staticmethod(lambda: "cache")})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "spine.config.load",
        type(
            "LoadModule",
            (),
            {"load_config_file": staticmethod(lambda path: calls.append(path))},
        )(),
    )
    monkeypatch.delenv("SPINE_PROD_BASEDIR", raising=False)
    cache_dir = tmp_path / "~/cache"

    loaded = preload.preload_downloads(configs, tmp_path, cache_dir)

    assert loaded == [config.resolve() for config in configs]
    assert calls == [str(config.resolve()) for config in configs]
    assert os.environ["SPINE_PROD_BASEDIR"] == str(tmp_path)
    assert os.environ["SPINE_CACHE_DIR"] == str(cache_dir.expanduser())


def test_preload_downloads_reports_missing_spine_tools(tmp_path, monkeypatch):
    """Test an actionable error is raised when SPINE cannot be imported."""
    real_import = builtins.__import__

    def reject_spine(name, *args, **kwargs):
        if name.startswith("spine.config"):
            raise ImportError("spine unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_spine)

    with pytest.raises(RuntimeError, match="Could not import SPINE config tools"):
        preload.preload_downloads("config.yaml", tmp_path)


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


def test_script_wrapper_loads_preload_function():
    """Test the wrapper's isolated loader returns the implementation function."""
    spec = importlib.util.spec_from_file_location("preload_downloads", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loaded = module.load_preload_downloads()

    assert loaded.__name__ == "preload_downloads"
    assert loaded.__module__ == "spine_prod_preload"


def test_script_wrapper_reports_preload_errors(tmp_path, capsys):
    """Test wrapper failures become a nonzero exit code and concise message."""
    spec = importlib.util.spec_from_file_location("preload_downloads", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_preload_downloads = lambda: lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("download failed"))

    old_argv = sys.argv
    try:
        sys.argv = [str(SCRIPT_PATH), "config.yaml"]
        assert module.main() == 1
    finally:
        sys.argv = old_argv

    assert "ERROR: download failed" in capsys.readouterr().err

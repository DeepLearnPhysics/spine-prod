"""Utilities for preloading SPINE !download assets."""

import os
import sys
from pathlib import Path


def bootstrap_spine(project_root: Path):
    """Make the bundled SPINE submodule importable when present."""
    spine_src = project_root / "spine" / "src"
    if spine_src.exists() and str(spine_src) not in sys.path:
        sys.path.insert(0, str(spine_src))


def resolve_config_path(config: str, project_root: Path) -> Path:
    """Resolve a config path from cwd or the repository config directory."""
    path = Path(config).expanduser()
    if path.exists():
        return path.resolve()

    repo_config_path = project_root / "config" / config
    if repo_config_path.exists():
        return repo_config_path.resolve()

    raise FileNotFoundError(f"Config not found: {config}")


def preload_downloads(configs, project_root: Path, cache_dir=None):
    """Preload !download assets for one or more SPINE config files."""
    os.environ.setdefault("SPINE_PROD_BASEDIR", str(project_root))
    if cache_dir:
        os.environ["SPINE_CACHE_DIR"] = str(Path(cache_dir).expanduser())

    bootstrap_spine(project_root)

    try:
        from spine.config.download import get_cache_dir
        from spine.config.load import load_config_file
    except ImportError as exc:
        raise RuntimeError(
            "Could not import SPINE config tools. Source configure.sh or "
            "initialize the spine submodule before preloading downloads."
        ) from exc

    if isinstance(configs, (str, Path)):
        configs = [configs]

    print(f"Download cache: {get_cache_dir()}")

    loaded_configs = []
    for config in configs:
        config_path = resolve_config_path(str(config), project_root)
        print(f"\nLoading config: {config_path}")
        load_config_file(str(config_path))
        loaded_configs.append(config_path)

    print("\nPreload complete.")
    return loaded_configs

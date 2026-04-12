#!/usr/bin/env python3
"""Preload SPINE !download assets into the shared cache."""

import argparse
import os
import sys
from pathlib import Path


def get_project_root():
    """Return the spine-prod repository root."""
    return Path(__file__).resolve().parents[1]


def bootstrap_spine(project_root):
    """Make the bundled SPINE submodule importable when present."""
    spine_src = project_root / "spine" / "src"
    if spine_src.exists():
        sys.path.insert(0, str(spine_src))


def resolve_config_path(config, project_root):
    """Resolve a config path from cwd or the repository config directory."""
    path = Path(config).expanduser()
    if path.exists():
        return path.resolve()

    repo_config_path = project_root / "config" / config
    if repo_config_path.exists():
        return repo_config_path.resolve()

    raise FileNotFoundError("Config not found: {}".format(config))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Preload files referenced by SPINE !download tags. Run this on a "
            "login node before submitting jobs on systems where worker nodes "
            "do not have outbound network access."
        )
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="SPINE config files to load, e.g. infer/2x2/full_chain_240819.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        help=(
            "Override SPINE_CACHE_DIR. Defaults to SPINE's normal cache "
            "resolution, typically $SPINE_PROD_BASEDIR/.cache/weights."
        ),
    )
    return parser.parse_args()


def main():
    """Preload all downloads needed by the requested configs."""
    args = parse_args()
    project_root = get_project_root()

    os.environ.setdefault("SPINE_PROD_BASEDIR", str(project_root))
    if args.cache_dir:
        os.environ["SPINE_CACHE_DIR"] = str(Path(args.cache_dir).expanduser())

    bootstrap_spine(project_root)

    try:
        from spine.config.download import get_cache_dir
        from spine.config.load import load_config_file
    except ImportError as exc:
        print(
            "ERROR: Could not import SPINE config tools. Source configure.sh or "
            "initialize the spine submodule before running this script.",
            file=sys.stderr,
        )
        print("Import error: {}".format(exc), file=sys.stderr)
        return 1

    print("Download cache: {}".format(get_cache_dir()))

    for config in args.configs:
        try:
            config_path = resolve_config_path(config, project_root)
            print("\nLoading config: {}".format(config_path))
            load_config_file(str(config_path))
        except Exception as exc:
            print(
                "ERROR: Failed to preload {}: {}".format(config, exc), file=sys.stderr
            )
            return 1

    print("\nPreload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

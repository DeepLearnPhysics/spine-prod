#!/usr/bin/env python3
"""CLI wrapper for preloading SPINE !download assets."""

import argparse
import importlib.util
import sys
from pathlib import Path


def get_project_root():
    """Return the spine-prod repository root."""
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()


def load_preload_downloads():
    """Load src/preload.py without importing the full src package."""
    preload_path = PROJECT_ROOT / "src" / "preload.py"
    spec = importlib.util.spec_from_file_location("spine_prod_preload", preload_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preload_downloads


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Preload files referenced by SPINE !download tags. Useful when "
            "using spine-prod configs from an external production pipeline."
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
    try:
        preload_downloads = load_preload_downloads()
        preload_downloads(
            args.configs,
            PROJECT_ROOT,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

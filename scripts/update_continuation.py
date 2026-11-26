#!/usr/bin/env python3
"""Update future continuation requests without cancelling or resubmitting jobs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuation.update import update_campaign  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--min-time")
    parser.add_argument("--cache-dir")
    parser.add_argument("--checkpoint-margin")
    parser.add_argument("--cache-ntasks", type=int)
    parser.add_argument("--cache-time")
    parser.add_argument(
        "--apply", action="store_true", help="Save changes (default: dry run)"
    )
    args = parser.parse_args()
    update_campaign(
        args.manifest,
        args.workspace,
        args.min_time,
        args.cache_dir,
        args.apply,
        checkpoint_margin=args.checkpoint_margin,
        cache_ntasks=args.cache_ntasks,
        cache_time=args.cache_time,
    )


if __name__ == "__main__":
    main()

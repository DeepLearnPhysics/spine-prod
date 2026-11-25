#!/usr/bin/env python3
"""Run one bounded scan of a site-managed continuation registry."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import require_python  # noqa: E402

require_python()
from src.continuation.service import tick_registry  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry")
    parser.add_argument("--preflight-s3df", action="store_true")
    args = parser.parse_args()
    if args.preflight_s3df:
        from src.continuation.s3df import preflight_local

        preflight_local(args.registry)
    else:
        tick_registry(args.registry)

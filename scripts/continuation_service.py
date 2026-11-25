#!/usr/bin/env python3
"""Run one bounded scan of a NERSC continuation registry."""
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
    args = parser.parse_args()
    tick_registry(args.registry)

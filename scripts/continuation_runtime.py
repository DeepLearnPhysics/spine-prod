#!/usr/bin/env python3
"""Command-line entry point for compute-side allocation supervision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuation.runtime import main  # noqa: E402

if __name__ == "__main__":
    main()

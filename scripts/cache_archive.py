#!/usr/bin/env python3
"""Archive or explicitly restore cache data, never scheduler/controller state."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import require_python  # noqa: E402

require_python((3, 9))
from src.cache_archive import archive, restore  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["archive", "restore"])
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--stage")
    parser.add_argument("--publication-id")
    args = parser.parse_args()
    if args.action == "archive":
        if not args.stage:
            parser.error("archive requires --stage")
        print(archive(args.source, args.destination, args.stage, args.publication_id))
    else:
        restore(args.source, args.destination)
        print("Restored cache: " + args.destination)

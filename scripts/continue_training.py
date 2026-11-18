#!/usr/bin/env python3
"""Advance or inspect a durable continuation pipeline from a submission host."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuation.controller import advance, metrics, operate, status  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "advance",
            "status",
            "extend",
            "accept",
            "cancel",
            "adopt",
            "retry",
            "metrics",
            "preflight",
        ],
    )
    parser.add_argument("manifest")
    parser.add_argument("--stage")
    parser.add_argument(
        "--site", choices=["nersc", "s3df", "anl"], help="Site for preflight"
    )
    parser.add_argument(
        "--value", help="New total budget (extend), or scheduler job ID (adopt)"
    )
    args = parser.parse_args()
    if args.action == "preflight":
        from src.continuation.state import preflight_storage

        preflight_storage(args.manifest, args.site)
        print("Continuation storage preflight passed on this host: " + args.manifest)
        return
    if args.action == "metrics":
        metrics(args.manifest)
        return
    if args.action == "advance":
        advance(args.manifest)
    elif args.action != "status":
        operate(args.manifest, args.action, args.stage, args.value)
    status(args.manifest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and submit a short SBND 261029 pipeline on S3DF."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--train-source-list", required=True, type=Path)
    parser.add_argument("--validation-source-list", required=True, type=Path)
    parser.add_argument("--spine-path", required=True, type=Path)
    parser.add_argument(
        "--warm-start", type=Path, help="Optional full-chain checkpoint"
    )
    parser.add_argument("--train-files", type=positive, default=4)
    parser.add_argument("--validation-files", type=positive, default=4)
    parser.add_argument("--steps", type=positive, default=20)
    parser.add_argument("--account", help="Optional S3DF scheduler account override")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate jobs without submitting"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = args.workspace.expanduser().resolve()
    # Refuse to overwrite an existing experiment, including its smoke inputs.
    if workspace.exists():
        parser.error("workspace already exists; use a fresh smoke workspace")
    if not args.spine_path.expanduser().is_dir():
        parser.error("--spine-path must point to an existing SPINE checkout")
    if args.warm_start and not args.warm_start.expanduser().is_file():
        parser.error("--warm-start checkpoint does not exist")
    selected = {}
    for split, source, limit in [
        ("train", args.train_source_list, args.train_files),
        ("validation", args.validation_source_list, args.validation_files),
    ]:
        source = source.expanduser().resolve()
        lines = source.read_text().splitlines()
        files = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ][:limit]
        if not files:
            parser.error(f"empty {split} source list: {source}")
        # Require concrete S3DF paths rather than guessing a Polaris mirror.
        selected[split] = []
        for name in files:
            path = Path(name).expanduser()
            if not path.is_absolute() or not path.is_file():
                parser.error(
                    f"{split} input must be an existing absolute file path: {name}"
                )
            selected[split].append(str(path))
    workspace.mkdir(parents=True)
    inputs = workspace / "smoke_inputs"
    inputs.mkdir()
    cfg = yaml.safe_load((root / "pipelines/sbnd/full_chain_261029.yaml").read_text())
    for split, files in selected.items():
        path = inputs / f"{split}.list"
        path.write_text("\n".join(files) + "\n")
        cfg["variables"][f"{split}_source_list"] = str(path)
    for split in cfg["collections"]["splits"]:
        split["cache_tasks"] = 1
    for stage in cfg["stages"]:
        if stage.get("stage") == "train":
            stage["time"] = "02:00:00"
            stage.setdefault("set", []).extend(
                [
                    "base.epochs=null",
                    f"base.iterations={args.steps}",
                    "train.save_epoch=null",
                    f"train.save_step={args.steps}",
                    "validation.initial=true",
                ]
            )
        if stage["name"] == "evaluate_full_chain":
            stage["ntasks"] = 1
    pipeline = inputs / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Smoke pipeline: {pipeline}", flush=True)
    print(
        f"Training: {args.steps} updates per component at production GPU/batch settings.",
        flush=True,
    )
    print(
        "Caches, validation and evaluation process the selected files; metrics are for plumbing only.",
        flush=True,
    )
    command = [
        sys.executable,
        str(root / "submit.py"),
        "--pipeline",
        str(pipeline),
        "--workspace",
        str(workspace),
        "--spine-path",
        str(args.spine_path.expanduser().resolve()),
    ]
    if args.warm_start:
        command += ["--warm-start", str(args.warm_start.expanduser().resolve())]
    if args.account:
        command += ["--account", args.account]
    if args.dry_run:
        command += ["--dry-run"]
    if not args.prepare_only:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()

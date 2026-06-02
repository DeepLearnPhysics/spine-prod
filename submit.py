#!/usr/bin/env python3
"""SPINE Production Batch Submission System.

A modern, flexible job submission orchestrator for SPINE reconstruction
on batch-based HPC systems.

Usage:
    ./submit.py --config infer/icarus --source test.root
    ./submit.py --config infer/icarus/latest --source-list file_list.txt
    ./submit.py --config infer/icarus/latest --apply-mods data --source data/*.root --profile s3df_ampere
    ./submit.py --pipeline pipelines/icarus_production.yaml
    ./submit.py --config ... --source ... --central-dir
"""

import argparse
import sys
from pathlib import Path

from src import Submitter


def main():
    """Main entry point for the batch submission system."""
    parser = argparse.ArgumentParser(
        description="SPINE Production Batch Submission System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detector shorthand resolves to the latest composite config
  %(prog)s --config infer/icarus --source test.root

  # Basic submission with source list (recommended)
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source-list file_list.txt

  # Direct sources with glob pattern
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source data/*.root

  # Interactive mode (test locally without a batch scheduler)
  %(prog)s --interactive --config infer/icarus/full_chain_co_250625.yaml --source test.root
  %(prog)s -I --config infer/icarus/full_chain_co_250625.yaml --source-list files.txt --task-id 2

  # With custom profile
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source data/*.root --profile s3df_ampere

  # Apply modifiers at runtime
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source data/*.root --apply-mods data
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source data/*.root --apply-mods data lite

  # Override SPINE config values at runtime
  %(prog)s --config infer/generic/latest --source test.root --set base.world_size=0

  # List available modifiers for a config
  %(prog)s --list-mods infer/icarus/full_chain_co_250625.yaml

  # Split a file list across 20 tasks as evenly as possible
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source-list files.txt --ntasks 20

  # Multiple files per task, multiple concurrent tasks
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source-list files.txt --files-per-task 5 --ntasks 20

  # Use the input list already embedded in the config
  %(prog)s --config config/train/icarus/deghost/deghost.yaml

  # Pipeline mode
  %(prog)s --pipeline pipelines/icarus_production.yaml

  # Dry run (does not submit jobs, but shows what would be done)
  %(prog)s --config infer/icarus/full_chain_co_250625.yaml --source test.root --dry-run
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--config", "-c", help="SPINE configuration file")
    mode_group.add_argument("--pipeline", "-P", help="Pipeline YAML file")
    mode_group.add_argument(
        "--list-mods",
        metavar="CONFIG",
        help="List available modifiers for a configuration",
    )

    # Input files, global patterns or file list (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        "-s",
        nargs="+",
        help="Input files as direct paths or glob patterns (e.g., data/*.root)",
    )
    source_group.add_argument(
        "--source-list",
        "-S",
        nargs=1,
        help="Text file containing input file paths (one per line)",
    )

    # Configuration modifiers
    parser.add_argument(
        "--apply-mods",
        "-m",
        nargs="+",
        help='Apply modifier configs (e.g., "data flash" to compose with data and flash modifiers)',
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Override a SPINE config parameter using dot notation. May be "
            "specified multiple times."
        ),
    )

    # Resource configuration
    parser.add_argument(
        "--profile",
        "-p",
        default="auto",
        help="Resource profile (default: auto-detect)",
    )
    parser.add_argument(
        "--ntasks",
        "-n",
        type=int,
        help=(
            "Target number of tasks when --files-per-task is omitted, or array "
            "concurrency cap when --files-per-task is set"
        ),
    )
    parser.add_argument(
        "--files-per-task",
        type=int,
        help=(
            "Files per task. If omitted, all explicit inputs run in one task "
            "unless --ntasks requests an even split"
        ),
    )

    # Job configuration
    parser.add_argument("--job-name", "-j", help="Custom job name")
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Output file path or directory override. By default, outputs are "
            "written under the job output directory with input-derived prefixes."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        help=(
            "Override the suffix used for input-derived HDF5 output names "
            "(default: final config stem)."
        ),
    )
    parser.add_argument("--account", "-A", help="Batch scheduler account")
    parser.add_argument("--dependency", "-d", help="Batch scheduler dependency string")

    # Directory configuration
    dir_group = parser.add_mutually_exclusive_group()
    dir_group.add_argument(
        "--central-dir",
        action="store_true",
        help="Write job directories and logs to the central spine-prod/jobs directory "
        "instead of the current working directory (default).",
    )
    dir_group.add_argument(
        "--local-output",
        action="store_true",
        help="(DEPRECATED: This is now the default behavior and will be removed in a future version. "
        "Use --central-dir to write to spine-prod/jobs instead.)",
    )

    # Software paths
    parser.add_argument("--larcv-path", "-l", help="Custom LArCV installation path")
    parser.add_argument(
        "--flashmatch-path",
        help="Custom flash-matching installation path (sources configure.sh)",
    )
    parser.add_argument(
        "--spine-path",
        help=(
            "Override the SPINE executable with a checkout directory or explicit "
            "executable path. Directories resolve to bin/spine or bin/run.py."
        ),
    )
    parser.add_argument(
        "--flashmatch",
        "-F",
        action="store_true",
        help=(
            "Deprecated no-op. OpT0Finder is included in the tagged SPINE " "container."
        ),
    )
    parser.add_argument(
        "--cvmfs",
        action="store_true",
        help="Expose CVMFS inside the container. On S3DF this adds /cvmfs/ to "
        "Singularity binds; on NERSC this adds --module=cvmfs to Shifter.",
    )
    # Profile overrides
    partition_group = parser.add_mutually_exclusive_group()
    partition_group.add_argument("--partition", help="Override partition")
    partition_group.add_argument("--qos", help="Override QOS")
    partition_group.add_argument("--queue", help="Override PBS queue")

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpus", type=int, help="Override total GPU count")
    gpu_group.add_argument("--gpus-per-node", type=int, help="Override GPUs per node")

    cpu_group = parser.add_mutually_exclusive_group()
    cpu_group.add_argument("--cpus-per-task", type=int, help="Override CPUs per task")

    mem_group = parser.add_mutually_exclusive_group()
    mem_group.add_argument("--mem-per-cpu", help="Override memory per CPU")
    mem_group.add_argument("--mem-per-node", help="Override memory per node")

    parser.add_argument("--constraint", help="Override constraint")
    parser.add_argument("--nodes", type=int, help="Override number of nodes")
    parser.add_argument("--time", "-t", help="Override time limit")
    parser.add_argument(
        "--bind-paths",
        help=(
            "Override Singularity/Apptainer bind roots for batch templates that "
            "support them, as a comma-separated list."
        ),
    )

    # Execution
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without submitting",
    )
    exec_group.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        help="Run interactively without submitting to a batch scheduler (useful for testing)",
    )
    parser.add_argument(
        "--interactive-runtime",
        choices=["auto", "local", "container"],
        default="auto",
        help=(
            "Runtime for --interactive: local uses spine on PATH, container uses "
            "SPINE_CONTAINER_PATH/SPINE_CONTAINER_TAG, auto falls back to "
            "container if spine is missing (default: auto)."
        ),
    )

    parser.add_argument(
        "--task-id",
        type=int,
        default=1,
        help="Task ID to run in interactive mode (default: 1)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload SPINE !download assets on the submit host before execution.",
    )

    args = parser.parse_args()

    # Handle deprecated --local-output flag
    if getattr(args, "local_output", False):
        print(
            "WARNING: --local-output is deprecated and now the default behavior. "
            "This flag will be removed in a future version. "
            "Use --central-dir to write to spine-prod/jobs instead.",
            file=sys.stderr,
        )

    # Initialize submitter (central_dir True means write to spine-prod/jobs)
    submitter = Submitter(central_dir=getattr(args, "central_dir", False))

    # Handle --list-mods
    if args.list_mods:
        try:
            result = submitter.list_modifiers(args.list_mods)

            print(
                f"Available modifiers for {result['config_name']} "
                f"(version: {result['base_version'] or 'unversioned'}):"
            )
            if result["modifiers"]:
                for mod_name, mod_info in result["modifiers"].items():
                    versions_str = ", ".join(mod_info["available"])
                    print(
                        f"  {mod_name:12} -> {mod_info['selected']} (available: {versions_str})"
                    )

                mod_names = list(result["modifiers"].keys())
                print("\nUsage examples:")
                print(f"  --apply-mods {mod_names[0]}")
                print(f"  --apply-mods {mod_names[0]}:240719")
                print("  --apply-mods /path/to/custom_mod.yaml")
                if len(mod_names) > 1:
                    print(f"  --apply-mods {' '.join(mod_names[:2])}")
            else:
                config_path = Path(args.list_mods)
                print("  (none found)")
                print(
                    "\nNote: Modifiers should be in "
                    f"{config_path.parent.name}/modifier/ subdirectories or named "
                    f"{config_path.parent.name}_*_mod.yaml"
                )

        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        return 0

    # Interactive mode not supported for pipelines
    if args.interactive and args.pipeline:
        parser.error("--interactive mode is not supported with --pipeline")

    # Build profile overrides
    profile_overrides = {}
    for key in [
        "partition",
        "qos",
        "queue",
        "constraint",
        "gpus_per_node",
        "gpus",
        "cpus_per_task",
        "mem_per_cpu",
        "mem_per_node",
        "nodes",
        "time",
        "account",
        "bind_paths",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            profile_overrides[key] = value

    try:
        if args.pipeline:
            # Pipeline mode
            job_map = submitter.submit_pipeline(
                args.pipeline, dry_run=args.dry_run, preload=args.preload
            )
            print("\n=== Pipeline submitted ===")
            for stage, job_ids in job_map.items():
                print(f"{stage}: {', '.join(job_ids)}")

        elif args.interactive:
            # Interactive mode - run directly without SLURM
            files = args.source if args.source else args.source_list
            source_type = "source" if args.source else "source_list"

            exit_code = submitter.run_interactive(
                config=args.config,
                files=files,
                source_type=source_type,
                output=args.output,
                output_suffix=args.output_suffix,
                files_per_task=args.files_per_task,
                task_id=args.task_id,
                larcv_path=args.larcv_path,
                flashmatch_path=args.flashmatch_path,
                flashmatch=args.flashmatch,
                cvmfs=args.cvmfs,
                apply_mods=args.apply_mods,
                preload=args.preload,
                set_overrides=args.set_overrides,
                interactive_runtime=args.interactive_runtime,
                bind_paths=args.bind_paths,
                spine_path=args.spine_path,
            )
            return exit_code

        else:
            # Single job mode (batch submission)
            # Determine which source type was provided
            files = args.source if args.source else args.source_list
            source_type = "source" if args.source else "source_list"

            job_ids = submitter.submit_job(
                config=args.config,
                files=files,
                source_type=source_type,
                profile=args.profile,
                job_name=args.job_name,
                output=args.output,
                output_suffix=args.output_suffix,
                ntasks=args.ntasks,
                files_per_task=args.files_per_task,
                dependency=args.dependency,
                larcv_path=args.larcv_path,
                flashmatch_path=args.flashmatch_path,
                flashmatch=args.flashmatch,
                cvmfs=args.cvmfs,
                apply_mods=args.apply_mods,
                dry_run=args.dry_run,
                preload=args.preload,
                set_overrides=args.set_overrides,
                spine_path=args.spine_path,
                **profile_overrides,
            )

            if job_ids and not args.dry_run:
                print(f"\n=== Submitted job IDs: {', '.join(job_ids)} ===")

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

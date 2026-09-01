"""Interactive SPINE execution without scheduler submission."""

import subprocess
from pathlib import Path
from typing import List, Optional

from .component import SubmissionComponent


class InteractiveRunner(SubmissionComponent):
    """Prepare and execute one SPINE command in a local or container runtime."""

    def run_interactive(
        self,
        config: str,
        files: Optional[List[str]] = None,
        source_type: str = "source",
        output: Optional[str] = None,
        output_suffix: Optional[str] = None,
        in_place: bool = False,
        no_writer: bool = False,
        files_per_task: Optional[int] = None,
        task_id: int = 1,
        larcv_path: Optional[str] = None,
        flashmatch_path: Optional[str] = None,
        flashmatch: bool = False,
        cvmfs: bool = False,
        apply_mods: Optional[List[str]] = None,
        preload: bool = False,
        set_overrides: Optional[List[str]] = None,
        world_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        minibatch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        epochs: Optional[float] = None,
        iterations: Optional[int] = None,
        interactive_runtime: str = "auto",
        bind_paths: Optional[str] = None,
        spine_path: Optional[str] = None,
    ) -> int:
        """Run SPINE processing interactively (no SLURM submission).

        This mode performs all config composition and file preparation like
        submit_job(), but executes the SPINE command directly in the current
        shell instead of submitting to SLURM. Useful for testing configs.

        Parameters
        ----------
        config : str
            Path to SPINE configuration file
        files : List[str], optional
            List of input files (direct paths/globs or source list path). If not
            provided, SPINE uses the inputs already configured in ``config``.
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file)
        output : str, optional
            Output file path
        output_suffix : str, optional
            Output HDF5 suffix when output names are derived from input files
        in_place : bool, optional
            Leave the writer destination config-defined and suppress automatic
            ``--output*`` arguments.
        no_writer : bool, optional
            Deprecated and ignored. SPINE v0.15.3+ safely ignores output options
            when the configuration has no writer.
        files_per_task : int, optional
            Files to process per task. If omitted, all explicit input files run
            in a single task unless ``ntasks``-style splitting is requested by
            the caller before reaching interactive mode.
        task_id : int, optional
            Which task to run (1-indexed), by default 1
        larcv_path : str, optional
            Custom LArCV installation path
        flashmatch_path : str, optional
            Custom flash-matching installation path
        flashmatch : bool, optional
            Deprecated compatibility option. OpT0Finder is provided by the SPINE
            container and no external setup is needed.
        cvmfs : bool, optional
            Expose CVMFS inside the container, by default False
        apply_mods : List[str], optional
            List of modifiers to apply
        preload : bool, optional
            Preload !download assets before execution, by default False
        set_overrides : List[str], optional
            SPINE config overrides in KEY=VALUE form, passed as ``--set``.
        world_size : int, optional
            Number of local SPINE processes/devices.
        batch_size : int, optional
            Global SPINE data-loader batch size.
        minibatch_size : int, optional
            Per-process/GPU SPINE data-loader batch size.
        num_workers : int, optional
            Number of SPINE data-loader workers.
        epochs : float, optional
            Number of SPINE training epochs.
        iterations : int, optional
            Number of SPINE driver iterations.
        interactive_runtime : str, optional
            Runtime for interactive execution: 'auto', 'local', or 'container'.
        bind_paths : str, optional
            Extra container bind roots for interactive SIF execution, as a
            comma-separated list.
        spine_path : str, optional
            Override the SPINE executable with a checkout directory or an
            explicit executable path.

        Returns
        -------
        int
            Exit code from SPINE execution
        """
        if flashmatch and not flashmatch_path:
            self.context.batch.warn_flashmatch_noop()
        if no_writer:
            self.context.spine_cli.warn_no_writer_deprecated()
        if in_place and (output is not None or output_suffix is not None):
            raise ValueError(
                "--in-place cannot be combined with --output or --output-suffix"
            )

        if interactive_runtime not in ("auto", "local", "container"):
            raise ValueError(
                "interactive_runtime must be one of: 'auto', 'local', 'container'"
            )

        file_list = []
        if files:
            file_list = self.file_handler.parse_files(files, source_type)
            if not file_list:
                raise ValueError("No input files found")
            print(f"Found {len(file_list)} file(s) to process")
        else:
            if files_per_task is not None:
                raise ValueError(
                    "Cannot use --files-per-task without --source/--source-list"
                )
            if task_id != 1:
                raise ValueError("Cannot use --task-id without --source/--source-list")
            if output is not None or output_suffix is not None:
                raise ValueError(
                    "Cannot use --output/--output-suffix without "
                    "--source/--source-list"
                )
            print("No input files provided; using inputs defined in the config")

        # Detect detector
        detector = self.config_mgr.detect_detector(config)
        is_latest, config_name = self.context.batch.classify_config_request(config)

        job_name = f"interactive_{detector}_{config_name}"
        job_dir = self.batch_client.create_job_dir(job_name)

        # Handle "latest" config generation
        if is_latest:
            print(f"\nDetected 'latest' config request for {detector}")
            config = self.config_mgr.create_latest_config(detector, job_dir)

        # Apply modifiers if specified
        if apply_mods:
            config = self.config_mgr.create_composite_config(
                config, apply_mods, job_dir, detector=detector if is_latest else None
            )

        if preload:
            self.context.preload_downloads(config)

        task_file_list = None
        if file_list:
            max_array_size = self.profiles["defaults"]["max_array_size"]
            effective_files_per_task = self.context.batch.resolve_files_per_task(
                len(file_list), files_per_task=files_per_task
            )
            file_chunks = self.file_handler.chunk_files(
                file_list, max_array_size, effective_files_per_task
            )

            if task_id < 1 or task_id > len(file_chunks):
                raise ValueError(
                    f"Task ID {task_id} out of range (1-{len(file_chunks)})"
                )

            file_group_list = file_chunks[task_id - 1]
            task_file_list = job_dir / f"interactive_files_task_{task_id}.txt"
            with open(task_file_list, "w", encoding="utf-8") as f:
                for file_group in file_group_list:
                    for file_path in file_group:
                        f.write(f"{file_path}\n")

            total_files = sum(len(fg) for fg in file_group_list)
            print(f"\nRunning task {task_id}/{len(file_chunks)}")
            print(f"Processing {total_files} file(s):")
            for file_group in file_group_list:
                for file_path in file_group:
                    print(f"  {file_path}")
        else:
            print("\nRunning task 1/1")
            print("Processing inputs defined in the config")

        # Build command
        cmd_parts = []
        resolved_bind_paths = bind_paths

        # Cap Numba threads to the OpenBLAS build limit used in batch templates.
        cmd_parts.append("export NUMBA_NUM_THREADS=64")

        larcv_setup_cmd, larcv_bind_root = self.context.runtime.resolve_setup_path(
            larcv_path, "--larcv-path"
        )
        flashmatch_setup_cmd, flashmatch_bind_root = (
            self.context.runtime.resolve_setup_path(
                flashmatch_path, "--flashmatch-path"
            )
        )

        # Add environment setup if needed
        for setup_cmd in [larcv_setup_cmd, flashmatch_setup_cmd]:
            if setup_cmd:
                cmd_parts.append(setup_cmd)

        # Build SPINE command
        log_dir = job_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        output_dir = None
        if not in_place:
            output_dir, output_suffix = (
                self.context.spine_cli.default_writer_output_settings(
                    job_dir, config, output_suffix
                )
            )
        if file_list and output and not in_place:
            output_path = Path(output)
            if output_path.suffix:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path.mkdir(parents=True, exist_ok=True)
        elif file_list and not in_place:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_args = (
            self.context.spine_cli.format_output_args(output, output_dir, output_suffix)
            if file_list and not in_place
            else ""
        )
        spine_cli_overrides = self.context.spine_cli.format_set_overrides(set_overrides)
        spine_runtime_options = self.context.spine_cli.format_runtime_options(
            world_size,
            batch_size,
            minibatch_size,
            num_workers,
            epochs,
            iterations,
        )
        local_spine_cmd, extra_bind_root = self.context.runtime.resolve_spine_command(
            spine_path
        )
        extra_bind_roots = [
            root
            for root in [larcv_bind_root, flashmatch_bind_root, extra_bind_root]
            if root
        ]
        if extra_bind_roots:
            resolved_bind_paths = self.context.runtime.merge_bind_paths(
                bind_paths, extra_bind_roots
            )
        spine_cmd_parts = [local_spine_cmd or "spine"]
        if task_file_list is not None:
            spine_cmd_parts.extend(["-S", str(task_file_list)])
        spine_cmd_parts.extend(
            [output_args, "-c", str(config), "--log-dir", str(log_dir)]
        )
        spine_cmd = " ".join(part for part in spine_cmd_parts if part)
        spine_options = " ".join(
            part for part in [spine_runtime_options, spine_cli_overrides] if part
        )
        if spine_options:
            spine_cmd = f"{spine_cmd} {spine_options}"
        cmd_parts.append(spine_cmd)

        # Join with && for proper sequencing
        full_cmd = " && ".join(cmd_parts)
        if interactive_runtime == "local" and not local_spine_cmd:
            raise RuntimeError(
                "Interactive runtime 'local' requested, but no local SPINE command "
                "was found. Install 'spine' on PATH, pass --spine-path, or set "
                "SPINE_LOCAL_PATH."
            )
        if interactive_runtime == "container" or (
            interactive_runtime == "auto" and not local_spine_cmd
        ):
            full_cmd = self.context.runtime.build_interactive_container_command(
                full_cmd, cvmfs, bind_paths=resolved_bind_paths
            )

        print("\nExecuting:")
        print(f"  {full_cmd}\n")
        print("=" * 80)

        # Execute directly
        result = subprocess.run(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            check=False,
        )

        print("=" * 80)
        print(f"\nInteractive execution completed with exit code: {result.returncode}")
        print(f"Job directory: {job_dir}")
        if result.returncode == 0:
            if output:
                print(f"Output: {output}")
            else:
                print(f"Output directory: {output_dir}")
                print(f"Output suffix: {output_suffix}")

        return result.returncode

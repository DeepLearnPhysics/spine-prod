"""Single-job planning, template rendering, and scheduler submission."""

import math
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .client import PBSClient, SlurmClient
from .component import SubmissionComponent
from .run_manager import RunManager


class BatchRunner(SubmissionComponent):
    """Plan and submit one inference, training, or validation job."""

    def classify_config_request(self, config: str) -> Tuple[bool, str]:
        """Classify whether a config request should resolve to detector latest.

        Parameters
        ----------
        config : str
            User-provided config path or shorthand.

        Returns
        -------
        Tuple[bool, str]
            A tuple of ``(is_latest, config_name)`` where ``config_name`` is the
            normalized user-facing name for job naming.
        """
        config_path = Path(config)
        config_name = config_path.stem

        if config_name == "latest" or "latest" in config_path.parts:
            return True, "latest"

        config_str = config_path.as_posix().rstrip("/")
        for detector_config in self.profiles.get("detectors", {}).values():
            configs_dir = detector_config.get("configs_dir")
            if not configs_dir:
                continue

            rel_configs_dir = configs_dir.rstrip("/")
            if config_str == rel_configs_dir:
                return True, "latest"

            if (
                config_path.is_absolute()
                and config_path.resolve()
                == (self.basedir / "config" / rel_configs_dir).resolve()
            ):
                return True, "latest"

        return False, config_name

    @staticmethod
    def warn_flashmatch_noop() -> None:
        """Warn that the legacy flashmatch flag no longer changes behavior."""
        print(
            "WARNING: --flashmatch is deprecated and is now a no-op. "
            "OpT0Finder is already included in the SPINE container.",
            file=sys.stderr,
        )

    def get_batch_client(self, profile_config: Dict):
        """Get the batch client for a profile."""
        scheduler = profile_config.get("scheduler")
        site = profile_config.get("site", "s3df")

        if scheduler is None:
            scheduler = "pbs" if site in ("anl", "polaris") else "slurm"

        if scheduler == "slurm":
            return SlurmClient(self.basedir, self.jobs_dir)
        if scheduler == "pbs":
            return PBSClient(self.basedir, self.jobs_dir)

        raise ValueError(
            f"Unknown scheduler in profile: {scheduler}, must specify 'slurm' or 'pbs'"
        )

    def get_template_name(self, profile_config: Dict) -> str:
        """Get the job template for a profile."""
        if profile_config.get("template"):
            return profile_config["template"]

        site = profile_config.get("site", "s3df")
        if site == "nersc":
            return "job_template_nersc.sbatch"
        if site == "s3df":
            return "job_template_s3df.sbatch"
        if site in ("anl", "polaris"):
            return "job_template_anl.pbs"

        raise ValueError(
            f"Unknown site in profile: {site}, must specify 's3df', 'nersc', or 'anl'"
        )

    @staticmethod
    def resolve_files_per_task(
        num_files: int,
        ntasks: Optional[int] = None,
        files_per_task: Optional[int] = None,
    ) -> int:
        """Resolve the effective files-per-task policy for a submission."""
        if ntasks is not None and ntasks < 1:
            raise ValueError("--ntasks must be >= 1")

        if files_per_task is not None:
            if files_per_task < 1:
                raise ValueError("--files-per-task must be >= 1")
            return files_per_task

        if ntasks is not None:
            return max(1, math.ceil(num_files / ntasks))

        return max(1, num_files)

    def submit_job(
        self,
        config: str,
        files: Optional[List[str]] = None,
        source_type: str = "source",
        validation_files: Optional[List[str]] = None,
        validation_source_type: str = "source",
        named_sources: Optional[Mapping[str, Mapping[str, Any]]] = None,
        validation_named_sources: Optional[Mapping[str, Mapping[str, Any]]] = None,
        module_weights: Optional[Mapping[str, str]] = None,
        profile: str = "auto",
        job_name: Optional[str] = None,
        output: Optional[str] = None,
        output_suffix: Optional[str] = None,
        no_writer: bool = False,
        ntasks: Optional[int] = None,
        files_per_task: Optional[int] = None,
        dependency: Optional[str] = None,
        larcv_path: Optional[str] = None,
        flashmatch_path: Optional[str] = None,
        flashmatch: bool = False,
        cvmfs: bool = False,
        apply_mods: Optional[List[str]] = None,
        dry_run: bool = False,
        preload: bool = False,
        set_overrides: Optional[List[str]] = None,
        world_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        minibatch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        epochs: Optional[float] = None,
        iterations: Optional[int] = None,
        spine_path: Optional[str] = None,
        stage: str = "inference",
        run_dir: Optional[str] = None,
        resume: bool = False,
        resume_from: Optional[str] = None,
        validation_name: Optional[str] = None,
        rerun_validation: bool = False,
        tensorboard: bool = False,
        **profile_overrides,
    ) -> List[str]:
        """Submit batch job for SPINE processing.

        Parameters
        ----------
        config : str
            Path to SPINE configuration file
        files : List[str], optional
            List of input files (direct paths/globs or source list path). If not
            provided, SPINE uses the inputs already configured in ``config``.
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file),
            by default 'source'
        validation_files : List[str], optional
            Validation input files, globs, or one validation source-list path.
        validation_source_type : str, optional
            Either 'source' or 'source_list' for ``validation_files``.
        named_sources : mapping, optional
            Target-qualified source selectors for a composite dataset.
        validation_named_sources : mapping, optional
            Target-qualified validation selectors for a composite dataset.
        module_weights : mapping, optional
            Model module names mapped to checkpoint paths.
        profile : str, optional
            Resource profile name or 'auto', by default 'auto'
        job_name : str, optional
            Custom job name, by default None
        output : str, optional
            Output file path, by default None
        output_suffix : str, optional
            Output HDF5 suffix when output names are derived from input files,
            by default None
        no_writer : bool, optional
            Deprecated and ignored. SPINE v0.15.3+ safely ignores output options
            when the configuration has no writer.
        ntasks : int, optional
            Target number of tasks when ``files_per_task`` is omitted, or the
            scheduler array concurrency cap when ``files_per_task`` is set.
        files_per_task : int, optional
            Files to process per task. If omitted, all explicit input files run
            in a single task unless ``ntasks`` requests an even split.
        dependency : str, optional
            Batch scheduler dependency string, by default None
        larcv_path : str, optional
            Custom LArCV installation path, by default None
        flashmatch_path : str, optional
            Custom flash-matching installation path, by default None
        flashmatch : bool, optional
            Deprecated compatibility option. OpT0Finder is provided by the SPINE
            container and no external setup is needed.
        cvmfs : bool, optional
            Expose CVMFS inside the container, by default False
        apply_mods : List[str], optional
            List of modifiers to apply (e.g., ['data', 'flash']), by default None
        dry_run : bool, optional
            Show what would be submitted without submitting, by default False
        preload : bool, optional
            Preload !download assets before submitting, by default False
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
        spine_path : str, optional
            Override the SPINE executable with a checkout directory or an
            explicit executable path.
        stage : str, optional
            Lifecycle stage: inference, train, or validation.
        run_dir : str, optional
            Explicit persistent run directory. Required for train and validation.
        resume : bool, optional
            Resume training from the numerically latest checkpoint.
        resume_from : str, optional
            Resume training from a specific checkpoint within the run.
        validation_name : str, optional
            Name for a non-primary validation suite.
        rerun_validation : bool, optional
            Include checkpoints that already have complete validation logs.
        tensorboard : bool, optional
            Enable stage-specific TensorBoard event logging.
        **profile_overrides
            Override profile settings

        Returns
        -------
        List[str]
            List of submitted job IDs
        """
        if flashmatch and not flashmatch_path:
            self.warn_flashmatch_noop()
        if no_writer:
            self.context.spine_cli.warn_no_writer_deprecated()

        if stage not in ("inference", "train", "validation"):
            raise ValueError("stage must be one of: inference, train, validation")
        if stage != "inference" and not run_dir:
            raise ValueError(f"--run-dir is required for the {stage} stage")
        if stage != "train" and (resume or resume_from):
            raise ValueError("--resume and --resume-from are valid only for training")
        if stage != "validation" and (validation_name or rerun_validation):
            raise ValueError(
                "--validation-name and --rerun-validation are valid only for validation"
            )
        if validation_files and stage != "train":
            raise ValueError(
                "--val-source and --val-source-list are valid only for training"
            )
        if validation_named_sources and stage != "train":
            raise ValueError("Named validation sources are valid only for training")
        if files and named_sources:
            raise ValueError("Flat and named training sources cannot be combined")
        if validation_files and validation_named_sources:
            raise ValueError("Flat and named validation sources cannot be combined")
        if stage != "inference" and (ntasks is not None or files_per_task is not None):
            raise ValueError(
                "--ntasks and --files-per-task are valid only for inference"
            )

        file_list = []
        if files:
            file_list = self.file_handler.parse_files(files, source_type)
            if not file_list:
                raise ValueError("No input files found")
            print(f"Found {len(file_list)} file(s) to process")
        elif not named_sources:
            if ntasks is not None or files_per_task is not None:
                raise ValueError(
                    "Cannot use --ntasks/--files-per-task without "
                    "--source/--source-list"
                )
            if output is not None or output_suffix is not None:
                raise ValueError(
                    "Cannot use --output/--output-suffix without "
                    "--source/--source-list"
                )
            print("No input files provided; using inputs defined in the config")
        else:
            print(f"Using {len(named_sources)} named dataset source(s)")

        validation_file_list = []
        if validation_files:
            validation_file_list = self.file_handler.parse_files(
                validation_files, validation_source_type
            )
            if not validation_file_list:
                raise ValueError("No validation input files found")
            print(f"Found {len(validation_file_list)} validation file(s)")

        # Detect detector first
        detector = self.config_mgr.detect_detector(config)

        # Classify before resolving because latest may be a directory shorthand
        # or a virtual path that is materialized in the job workspace below.
        is_latest, config_name = self.classify_config_request(config)

        # Resolve ordinary files before persistent lifecycle bookkeeping reads
        # them. Latest shorthands are directories or virtual paths and are
        # materialized below instead.
        if not is_latest:
            config = str(self.config_mgr.resolve_config_path(config))

        if not job_name:
            job_name = f"spine_{detector}_{config_name}"

        if run_dir:
            job_dir = Path(run_dir).expanduser().resolve()
            if stage == "inference":
                if job_dir.exists() and any(job_dir.iterdir()):
                    raise ValueError(f"Inference run directory is not empty: {job_dir}")
                job_dir.mkdir(parents=True, exist_ok=True)
        else:
            job_dir = self.batch_client.create_job_dir(job_name)

        config_workspace = job_dir
        if stage != "inference":
            config_workspace = job_dir / ".spine-prod" / "configs"
            config_workspace.mkdir(parents=True, exist_ok=True)

        # Handle "latest" config generation
        if is_latest:
            print(f"\nDetected 'latest' config request for {detector}")
            config = self.config_mgr.create_latest_config(detector, config_workspace)
            config_name = Path(config).stem

        # Apply modifiers if specified
        original_config = config
        if apply_mods:
            # Pass detector if config was generated (to find modifiers in config dir)
            config = self.config_mgr.create_composite_config(
                config,
                apply_mods,
                config_workspace,
                detector=detector if is_latest else None,
            )

        if preload:
            self.context.preload_downloads(config)

        spine_cli_overrides = self.context.spine_cli.format_set_overrides(set_overrides)
        named_source_overrides = self.context.spine_cli.format_named_sources(
            named_sources
        )
        validation_named_source_overrides = self.context.spine_cli.format_named_sources(
            validation_named_sources, validation=True
        )
        module_weight_overrides = self.context.spine_cli.format_module_weights(
            module_weights
        )
        spine_cmd, extra_bind_root = self.context.runtime.resolve_spine_command(
            spine_path
        )
        _, larcv_bind_root = self.context.runtime.resolve_setup_path(
            larcv_path, "--larcv-path"
        )
        _, flashmatch_bind_root = self.context.runtime.resolve_setup_path(
            flashmatch_path, "--flashmatch-path"
        )

        # Detect detector and get profile
        profile_config = self.config_mgr.get_profile(profile, detector)
        site = profile_config.get("site", "s3df")
        if site == "s3df" and "gpus_per_node" in profile_overrides:
            raise ValueError("--gpus-per-node is not valid for S3DF profiles")
        if site != "s3df" and "gpus" in profile_overrides:
            raise ValueError(f"--gpus is not valid for {site.upper()} profiles")
        profile_config.update(profile_overrides)
        world_size = self.context.spine_cli.align_world_size(profile_config, world_size)
        spine_runtime_options = self.context.spine_cli.format_runtime_options(
            world_size,
            batch_size,
            minibatch_size,
            num_workers,
            epochs,
            iterations,
        )
        extra_bind_roots = [
            root
            for root in [larcv_bind_root, flashmatch_bind_root, extra_bind_root]
            if root
        ]
        if extra_bind_roots:
            bind_paths = profile_config.get("bind_paths")
            if not bind_paths:
                bind_paths = self.context.runtime.default_bind_paths_for_site(
                    profile_config.get("site", "s3df")
                )
            profile_config["bind_paths"] = self.context.runtime.merge_bind_paths(
                bind_paths, extra_bind_roots
            )

        # Get account
        account = profile_config.get("account")
        if not account and detector in self.profiles["detectors"]:
            profile_config["account"] = self.profiles["detectors"][detector].get(
                "account", self.profiles["defaults"]["account"]
            )

        submission_dir = None
        spine_log_dir = str(job_dir / "logs")
        lifecycle_args = []
        selected_checkpoints = []
        resume_checkpoint = None
        input_manifest = None
        validation_input_manifest = None
        if stage == "train":
            resume_checkpoint = RunManager.prepare_training_run(
                job_dir, config, resume=resume, resume_from=resume_from
            )
            submission_dir = RunManager.create_submission_dir(job_dir, "train")
            spine_log_dir = str(job_dir)
            lifecycle_args.extend(
                ["--weight-prefix", shlex.quote(str(job_dir / "weights" / "snapshot"))]
            )
            if resume_checkpoint is not None:
                lifecycle_args.extend(
                    ["--weight-path", shlex.quote(str(resume_checkpoint))]
                )
                lifecycle_args.append("--resume")
            if tensorboard:
                lifecycle_args.extend(
                    [
                        "--tensorboard",
                        "--tensorboard-dir",
                        shlex.quote(str(job_dir / "tensorboard" / "train")),
                    ]
                )
        elif stage == "validation":
            spine_log_path, selected_checkpoints = RunManager.prepare_validation(
                job_dir,
                config,
                validation_name=validation_name,
                rerun=rerun_validation,
            )
            spine_log_dir = str(spine_log_path)
            if not selected_checkpoints:
                print("Validation is up to date; no scheduler job submitted")
                return []
            submission_dir = RunManager.create_submission_dir(
                job_dir, "validation", validation_name
            )
            weight_list = submission_dir / "weights.txt"
            with open(weight_list, "w", encoding="utf-8") as stream:
                for checkpoint in selected_checkpoints:
                    stream.write(f"{checkpoint}\n")
            lifecycle_args.extend(["--weight-list", shlex.quote(str(weight_list))])
            lifecycle_args.extend(["--set", "model.weight_path=null"])
            if rerun_validation:
                lifecycle_args.extend(["--set", "base.overwrite_log=true"])
            if tensorboard:
                tensorboard_dir = job_dir / "tensorboard" / "validation"
                if validation_name:
                    tensorboard_dir /= validation_name
                lifecycle_args.extend(
                    [
                        "--tensorboard",
                        "--tensorboard-dir",
                        shlex.quote(str(tensorboard_dir)),
                    ]
                )

        if stage != "inference" and file_list:
            assert submission_dir is not None
            input_manifest = submission_dir / "inputs.txt"
            with open(input_manifest, "w", encoding="utf-8") as stream:
                for file_path in file_list:
                    stream.write(f"{file_path}\n")
            lifecycle_args.extend(["--source-list", shlex.quote(str(input_manifest))])

        if validation_file_list:
            assert submission_dir is not None
            validation_input_manifest = submission_dir / "validation_inputs.txt"
            with open(validation_input_manifest, "w", encoding="utf-8") as stream:
                for file_path in validation_file_list:
                    stream.write(f"{file_path}\n")
            lifecycle_args.extend(
                [
                    "--val-source-list",
                    shlex.quote(str(validation_input_manifest)),
                ]
            )

        extra_args = " ".join(lifecycle_args)
        spine_cli_overrides = " ".join(
            part
            for part in [
                spine_runtime_options,
                spine_cli_overrides,
                named_source_overrides,
                validation_named_source_overrides,
                module_weight_overrides,
                extra_args,
            ]
            if part
        )

        output_dir, output_suffix = (
            self.context.spine_cli.default_writer_output_settings(
                job_dir, config, output_suffix
            )
        )
        # Composite datasets have no flat file list, but their writer output
        # is still a first-class CLI override.
        if (file_list or named_sources) and output:
            output_path = Path(output)
            if output_path.suffix:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path.mkdir(parents=True, exist_ok=True)
        output_args = (
            self.context.spine_cli.format_output_args(output, output_dir, output_suffix)
            if (file_list or named_sources) and output
            else ""
        )

        file_list_pattern = None
        file_chunks = [[]]
        concurrent_task_limit = None
        if stage == "inference" and file_list:
            max_array_size = self.profiles["defaults"]["max_array_size"]
            effective_files_per_task = self.resolve_files_per_task(
                len(file_list), ntasks=ntasks, files_per_task=files_per_task
            )
            file_chunks = self.file_handler.chunk_files(
                file_list, max_array_size, effective_files_per_task
            )
            if files_per_task is not None and ntasks is not None:
                concurrent_task_limit = ntasks

        print(f"Splitting into {len(file_chunks)} array job(s)")

        job_ids = []
        chunk_dependency = dependency  # Track dependency for chunk chaining
        for chunk_idx, chunk in enumerate(file_chunks):
            chunk_name = f"chunk_{chunk_idx:03d}"
            task_dir_pattern = None
            chunk_output_args = output_args
            chunk_spine_log_dir = spine_log_dir
            if stage == "inference":
                scheduler_dir = job_dir / "scheduler" / chunk_name
            else:
                assert submission_dir is not None
                scheduler_dir = submission_dir
            scheduler_log_dir = scheduler_dir / "logs"
            scheduler_log_dir.mkdir(parents=True, exist_ok=True)

            # Render SBATCH script
            array_spec = None
            if len(chunk) > 1:
                array_spec = f"1-{len(chunk)}"
                if concurrent_task_limit is not None and concurrent_task_limit < len(
                    chunk
                ):
                    array_spec += f"%{concurrent_task_limit}"

            if stage == "inference" and file_list:
                task_chunk_dir = job_dir / "tasks" / chunk_name
                task_dir_pattern = str(task_chunk_dir / "task_*")
                file_list_pattern = f"{task_dir_pattern}/inputs.txt"
                for task_idx, file_group in enumerate(chunk, start=1):
                    task_dir = task_chunk_dir / f"task_{task_idx}"
                    (task_dir / "logs").mkdir(parents=True, exist_ok=True)
                    (task_dir / "output").mkdir(exist_ok=True)
                    task_file_list = task_dir / "inputs.txt"
                    with open(task_file_list, "w", encoding="utf-8") as f:
                        for file_path in file_group:
                            f.write(f"{file_path}\n")
                chunk_spine_log_dir = "$TASK_DIR/logs"
                if not output:
                    chunk_output_args = " ".join(
                        [
                            "--output-dir $TASK_DIR/output",
                            f"--output-suffix {shlex.quote(output_suffix)}",
                        ]
                    )
            else:
                file_list_pattern = None

            batch_client = self.get_batch_client(profile_config)
            template = batch_client.load_template(
                self.get_template_name(profile_config)
            )

            script_content = template.render(
                array_spec=array_spec,
                job_name=(
                    f"{job_name}_{chunk_idx}" if len(file_chunks) > 1 else job_name
                ),
                log_dir=str(scheduler_log_dir),
                dependency=chunk_dependency,
                basedir=str(self.basedir),
                file_list_pattern=file_list_pattern,
                input_manifest=(str(input_manifest) if input_manifest else None),
                task_dir_pattern=task_dir_pattern,
                spine_log_dir=chunk_spine_log_dir,
                config=config,
                output=output,
                output_dir=(
                    f"{task_dir_pattern}/output"
                    if task_dir_pattern and not output
                    else output_dir
                ),
                output_suffix=output_suffix,
                output_args=chunk_output_args,
                larcv_path=larcv_path,
                flashmatch_path=flashmatch_path,
                flashmatch=flashmatch,
                cvmfs=cvmfs,
                spine_cmd=spine_cmd or "spine",
                spine_cli_overrides=spine_cli_overrides,
                **profile_config,
            )

            # Write script
            script_path = scheduler_dir / f"submit{batch_client.script_suffix}"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            script_path.chmod(0o755)

            # Submit
            print(f"\nSubmitting chunk {chunk_idx + 1}/{len(file_chunks)}:")
            print(f"  Script: {script_path}")
            if file_list:
                print(f"  Files: {sum(len(group) for group in chunk)}")
            else:
                print("  Files: config-defined input list")
            print(f"  Profile: {profile} ({profile_config['description']})")
            if chunk_dependency:
                print(f"  Dependency: {chunk_dependency}")

            job_id = batch_client.submit(script_path, dry_run)
            if job_id:
                job_ids.append(job_id)
                print(f"  Job ID: {job_id}")

                # Chain dependencies: next chunk waits for this chunk to complete
                if len(file_chunks) > 1:
                    chunk_dependency = batch_client.dependency_afterok(job_id)

        # Save metadata
        from version import __version__

        metadata = {
            "spine_prod_version": __version__,
            "job_name": job_name,
            "stage": stage,
            "run_dir": str(job_dir),
            "detector": detector,
            "config": config,
            "original_config": original_config if apply_mods else config,
            "applied_modifiers": apply_mods or [],
            "set_overrides": set_overrides or [],
            "named_sources": named_sources or {},
            "validation_named_sources": validation_named_sources or {},
            "module_weights": module_weights or {},
            "world_size": world_size,
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "num_workers": num_workers,
            "epochs": epochs,
            "iterations": iterations,
            "source_type": source_type if files else None,
            "source_inputs": files or [],
            "source_manifest": str(input_manifest) if input_manifest else None,
            "validation_source_type": (
                validation_source_type if validation_files else None
            ),
            "validation_source_inputs": validation_files or [],
            "validation_source_manifest": (
                str(validation_input_manifest) if validation_input_manifest else None
            ),
            "larcv_path": larcv_path,
            "flashmatch_path": flashmatch_path,
            "spine_path": spine_path,
            "cvmfs": cvmfs,
            "no_writer": no_writer,
            "profile": profile,
            "profile_config": profile_config,
            "num_files": len(file_list) if file_list else None,
            "num_chunks": len(file_chunks),
            "files_per_task": files_per_task,
            "resolved_files_per_task": (
                self.resolve_files_per_task(
                    len(file_list), ntasks=ntasks, files_per_task=files_per_task
                )
                if stage == "inference" and file_list
                else None
            ),
            "ntasks": ntasks,
            "job_ids": job_ids,
            "output": output
            or (str(job_dir / "tasks") if stage == "inference" and file_list else None),
            "output_dir": (
                output_dir
                if output
                else (
                    str(job_dir / "tasks")
                    if stage == "inference" and file_list
                    else None
                )
            ),
            "output_suffix": output_suffix,
            "resume_checkpoint": (
                str(resume_checkpoint) if resume_checkpoint is not None else None
            ),
            "validation_name": validation_name,
            "selected_checkpoints": [str(path) for path in selected_checkpoints],
            "tensorboard": tensorboard,
            "submitted": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
        }
        metadata_dir = submission_dir or job_dir
        self.batch_client.save_job_metadata(metadata_dir, metadata)
        if stage == "train":
            RunManager.record_training_jobs(job_dir, job_ids)

        print(f"\nRun directory: {job_dir}")
        print(f"Submission metadata: {metadata_dir}/job_metadata.json")

        return job_ids

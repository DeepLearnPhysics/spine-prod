"""Main batch submission orchestrator for SPINE production."""

import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .client import PBSClient, SlurmClient
from .config_manager import ConfigManager
from .file_handler import FileHandler
from .preload import preload_downloads


class Submitter:
    """Orchestrates batch job submissions for SPINE production."""

    @staticmethod
    def _warn_flashmatch_noop() -> None:
        """Warn that the legacy flashmatch flag no longer changes behavior."""
        print(
            "WARNING: --flashmatch is deprecated and is now a no-op. "
            "OpT0Finder is already included in the SPINE container.",
            file=sys.stderr,
        )

    def __init__(self, basedir: Optional[Path] = None, central_dir: bool = False):
        """Initialize Submitter.

        Parameters
        ----------
        basedir : Optional[Path], optional
            Base directory of spine-prod, by default None (use script location)
        central_dir : bool, optional
            Write job dirs to spine-prod/jobs instead of cwd/jobs, by default False
        """
        # Always use the project base for resources
        self.basedir = basedir or Path(__file__).parent.parent

        # Initialize component modules
        self.config_mgr = ConfigManager(self.basedir)
        self.file_handler = FileHandler()

        if central_dir:
            jobs_dir = self.basedir / "jobs"
        else:
            jobs_dir = Path(os.getcwd()) / "jobs"
        jobs_dir.mkdir(exist_ok=True)

        self.jobs_dir = jobs_dir
        self.batch_client = SlurmClient(self.basedir, jobs_dir)

        # Check environment
        if not os.getenv("SPINE_PROD_BASEDIR") and central_dir:
            print("WARNING: SPINE_PROD_BASEDIR not set. Did you source configure.sh?")

    @property
    def profiles(self) -> Dict:
        """Get profiles from config manager."""
        return self.config_mgr.profiles

    def list_modifiers(self, config_path: str) -> Dict:
        """List available modifiers for a given configuration file.

        Parameters
        ----------
        config_path : str
            Path to the base configuration file

        Returns
        -------
        Dict
            Dictionary with 'base_version', 'config_name', and 'modifiers' keys
        """
        return self.config_mgr.list_modifiers(config_path)

    def _get_batch_client(self, profile_config: Dict):
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

    def _get_template_name(self, profile_config: Dict) -> str:
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
    def _format_spine_set_overrides(set_overrides: Optional[List[str]]) -> str:
        """Format SPINE ``--set`` overrides for shell execution."""
        if not set_overrides:
            return ""

        formatted = []
        for override in set_overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid --set override '{override}'. Expected KEY=VALUE."
                )
            if any(char.isspace() for char in override) or any(
                char in override for char in ("'", '"')
            ):
                raise ValueError(
                    f"Invalid --set override '{override}'. Whitespace and quotes "
                    "are not supported in submit.py --set values."
                )
            formatted.append(f"--set {override}")

        return " ".join(formatted)

    @staticmethod
    def _default_container_path() -> str:
        """Build the default local SIF path from the configured SPINE version."""
        version = os.environ.get("SPINE_CONTAINER_VERSION", "0.11.1")
        path_version = version.replace(".", "-")
        return f"/sdf/data/neutrino/images/spine_v{path_version}.sif"

    @staticmethod
    def _container_tag_for_cli() -> str:
        """Return the configured container tag in Docker/Podman CLI form."""
        version = os.environ.get("SPINE_CONTAINER_VERSION", "0.11.1")
        tag = os.environ.get(
            "SPINE_CONTAINER_TAG", f"docker:ghcr.io/deeplearnphysics/spine:{version}"
        )
        return tag.removeprefix("docker:")

    @staticmethod
    def _sif_runtime_executable() -> Optional[str]:
        """Return the Singularity/Apptainer executable for local SIF execution."""
        configured = os.environ.get("SPINE_CONTAINER_RUNTIME_BIN")
        if configured:
            runtime = shutil.which(configured)
            if not runtime:
                raise RuntimeError(
                    "SPINE_CONTAINER_RUNTIME_BIN is set, but no executable was found "
                    f"for: {configured}"
                )
            return runtime

        return shutil.which("singularity") or shutil.which("apptainer")

    @staticmethod
    def _sif_runtime_args() -> str:
        """Return additional args for interactive Singularity/Apptainer execution."""
        configured = os.environ.get("SPINE_CONTAINER_RUNTIME_ARGS", "")
        if not configured:
            return ""

        return " ".join(shlex.quote(arg) for arg in shlex.split(configured))

    def _build_interactive_container_command(self, inner_cmd: str, cvmfs: bool) -> str:
        """Build an interactive container command for local smoke tests."""
        container_path = os.environ.get(
            "SPINE_CONTAINER_PATH", self._default_container_path()
        )

        if Path(container_path).exists():
            singularity = self._sif_runtime_executable()
            if singularity:
                runtime_args = self._sif_runtime_args()
                runtime_args = f" {runtime_args}" if runtime_args else ""
                bind_paths = {str(Path.cwd()), str(self.basedir)}
                if cvmfs:
                    bind_paths.add("/cvmfs")
                bind_arg = ",".join(sorted(bind_paths))
                return (
                    f"{shlex.quote(singularity)} exec{runtime_args} --bind {shlex.quote(bind_arg)} "
                    f"--nv {shlex.quote(container_path)} bash -c "
                    f"{shlex.quote(inner_cmd)}"
                )

        docker = shutil.which("docker") or shutil.which("podman")
        if docker:
            workdir = str(Path.cwd())
            platform = os.environ.get("SPINE_CONTAINER_PLATFORM", "linux/amd64")
            platform_arg = f"--platform {shlex.quote(platform)} " if platform else ""
            volume_args = [f"-v {shlex.quote(f'{workdir}:{workdir}')}"]
            basedir = str(self.basedir)
            if basedir != workdir:
                volume_args.append(f"-v {shlex.quote(f'{basedir}:{basedir}')}")
            if cvmfs and Path("/cvmfs").exists():
                volume_args.append("-v /cvmfs:/cvmfs:ro")

            env_args = [
                f"-e SPINE_PROD_BASEDIR={shlex.quote(basedir)}",
                f"-e SPINE_CONFIG_PATH={shlex.quote(os.environ.get('SPINE_CONFIG_PATH', str(self.basedir / 'config')))}",
            ]
            if os.environ.get("ICARUS_DATA_DIR"):
                env_args.append(
                    f"-e ICARUS_DATA_DIR={shlex.quote(os.environ['ICARUS_DATA_DIR'])}"
                )

            return (
                f"{shlex.quote(docker)} run --rm {platform_arg}{' '.join(volume_args)} "
                f"-w {shlex.quote(workdir)} {' '.join(env_args)} "
                f"{shlex.quote(self._container_tag_for_cli())} bash -c "
                f"{shlex.quote(inner_cmd)}"
            )

        raise RuntimeError(
            "Interactive container runtime requested, but no usable runtime was "
            "found. Install spine on PATH, provide a readable SPINE_CONTAINER_PATH "
            "with singularity/apptainer, or make docker/podman available with "
            "SPINE_CONTAINER_TAG."
        )

    def run_interactive(
        self,
        config: str,
        files: List[str],
        source_type: str = "source",
        output: Optional[str] = None,
        files_per_task: int = 1,
        task_id: int = 1,
        larcv_basedir: Optional[str] = None,
        flashmatch: bool = False,
        cvmfs: bool = False,
        apply_mods: Optional[List[str]] = None,
        preload: bool = False,
        set_overrides: Optional[List[str]] = None,
        interactive_runtime: str = "auto",
    ) -> int:
        """Run SPINE processing interactively (no SLURM submission).

        This mode performs all config composition and file preparation like
        submit_job(), but executes the SPINE command directly in the current
        shell instead of submitting to SLURM. Useful for testing configs.

        Parameters
        ----------
        config : str
            Path to SPINE configuration file
        files : List[str]
            List of input files (direct paths/globs or source list path)
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file)
        output : str, optional
            Output file path
        files_per_task : int, optional
            Files to process per task, by default 1
        task_id : int, optional
            Which task to run (1-indexed), by default 1
        larcv_basedir : str, optional
            Custom LArCV installation path
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
        interactive_runtime : str, optional
            Runtime for interactive execution: 'auto', 'local', or 'container'.

        Returns
        -------
        int
            Exit code from SPINE execution
        """
        if flashmatch:
            self._warn_flashmatch_noop()

        if interactive_runtime not in ("auto", "local", "container"):
            raise ValueError(
                "interactive_runtime must be one of: 'auto', 'local', 'container'"
            )

        # Parse input files
        file_list = self.file_handler.parse_files(files, source_type)
        if not file_list:
            raise ValueError("No input files found")

        print(f"Found {len(file_list)} file(s) to process")

        # Detect detector
        detector = self.config_mgr.detect_detector(config)
        config_path = Path(config)
        config_name = config_path.stem

        # Check if this is a "latest" request
        is_latest = config_name == "latest" or "latest" in config_path.parts

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
            self._preload_downloads(config)

        # Determine output path
        if not output:
            output = str(job_dir / "output" / f"{job_name}.h5")

        # Ensure output directory exists
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        # Chunk files for processing
        max_array_size = self.profiles["defaults"]["max_array_size"]
        file_chunks = self.file_handler.chunk_files(
            file_list, max_array_size, files_per_task
        )

        # Validate task_id
        if task_id < 1 or task_id > len(file_chunks):
            raise ValueError(f"Task ID {task_id} out of range (1-{len(file_chunks)})")

        # Get the file group for this task (it's a list of file lists)
        file_group_list = file_chunks[task_id - 1]

        # Create temporary file list for this task
        task_file_list = job_dir / f"interactive_files_task_{task_id}.txt"
        with open(task_file_list, "w", encoding="utf-8") as f:
            for file_group in file_group_list:
                # Each file_group is a list of file paths
                for file_path in file_group:
                    f.write(f"{file_path}\n")

        # Count total files for display
        total_files = sum(len(fg) for fg in file_group_list)
        print(f"\nRunning task {task_id}/{len(file_chunks)}")
        print(f"Processing {total_files} file(s):")
        for file_group in file_group_list:
            for file_path in file_group:
                print(f"  {file_path}")

        # Build command
        cmd_parts = []

        # Add environment setup if needed
        if larcv_basedir:
            cmd_parts.append(f"source {larcv_basedir}/configure.sh")

        # Build SPINE command
        log_dir = job_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        spine_cli_overrides = self._format_spine_set_overrides(set_overrides)
        spine_cmd = (
            "spine "
            f"-S {task_file_list} "
            f"-o {output} "
            f"-c {config} "
            f"--log-dir {log_dir}"
        )
        if spine_cli_overrides:
            spine_cmd = f"{spine_cmd} {spine_cli_overrides}"
        cmd_parts.append(spine_cmd)

        # Join with && for proper sequencing
        full_cmd = " && ".join(cmd_parts)
        local_spine = shutil.which("spine")
        if interactive_runtime == "local" and not local_spine:
            raise RuntimeError(
                "Interactive runtime 'local' requested, but 'spine' is not on PATH."
            )
        if interactive_runtime == "container" or (
            interactive_runtime == "auto" and not local_spine
        ):
            full_cmd = self._build_interactive_container_command(full_cmd, cvmfs)

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
            print(f"Output: {output}")

        return result.returncode

    def submit_job(
        self,
        config: str,
        files: List[str],
        source_type: str = "source",
        profile: str = "auto",
        job_name: Optional[str] = None,
        output: Optional[str] = None,
        ntasks: Optional[int] = None,
        files_per_task: int = 1,
        dependency: Optional[str] = None,
        larcv_basedir: Optional[str] = None,
        flashmatch: bool = False,
        cvmfs: bool = False,
        apply_mods: Optional[List[str]] = None,
        dry_run: bool = False,
        preload: bool = False,
        set_overrides: Optional[List[str]] = None,
        **profile_overrides,
    ) -> List[str]:
        """Submit batch job for SPINE processing.

        Parameters
        ----------
        config : str
            Path to SPINE configuration file
        files : List[str]
            List of input files (direct paths/globs or source list path)
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file),
            by default 'source'
        profile : str, optional
            Resource profile name or 'auto', by default 'auto'
        job_name : str, optional
            Custom job name, by default None
        output : str, optional
            Output file path, by default None
        ntasks : int, optional
            Number of parallel tasks (default: auto), by default None
        files_per_task : int, optional
            Files to process per task, by default 1
        dependency : str, optional
            Batch scheduler dependency string, by default None
        larcv_basedir : str, optional
            Custom LArCV installation path, by default None
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
        **profile_overrides
            Override profile settings

        Returns
        -------
        List[str]
            List of submitted job IDs
        """
        if flashmatch:
            self._warn_flashmatch_noop()

        # Parse input files
        file_list = self.file_handler.parse_files(files, source_type)
        if not file_list:
            raise ValueError("No input files found")

        print(f"Found {len(file_list)} file(s) to process")

        # Detect detector first
        detector = self.config_mgr.detect_detector(config)

        # Create job directory first (needed for composite/latest config generation)
        config_path = Path(config)
        config_name = config_path.stem

        # Check if this is a "latest" request
        is_latest = config_name == "latest" or "latest" in config_path.parts

        if not job_name:
            job_name = f"spine_{detector}_{config_name}"

        job_dir = self.batch_client.create_job_dir(job_name)

        # Handle "latest" config generation
        if is_latest:
            print(f"\nDetected 'latest' config request for {detector}")
            config = self.config_mgr.create_latest_config(detector, job_dir)
            config_name = Path(config).stem

        # Apply modifiers if specified
        original_config = config
        if apply_mods:
            # Pass detector if config was generated (to find modifiers in config dir)
            config = self.config_mgr.create_composite_config(
                config, apply_mods, job_dir, detector=detector if is_latest else None
            )

        if preload:
            self._preload_downloads(config)

        spine_cli_overrides = self._format_spine_set_overrides(set_overrides)

        # Detect detector and get profile
        profile_config = self.config_mgr.get_profile(profile, detector)
        profile_config.update(profile_overrides)

        # Get account
        account = profile_config.get("account")
        if not account and detector in self.profiles["detectors"]:
            profile_config["account"] = self.profiles["detectors"][detector].get(
                "account", self.profiles["defaults"]["account"]
            )

        # Determine output path
        if not output:
            output = str(job_dir / "output" / f"{job_name}.h5")

        # Chunk files for array jobs
        max_array_size = self.profiles["defaults"]["max_array_size"]
        file_chunks = self.file_handler.chunk_files(
            file_list, max_array_size, files_per_task
        )

        print(f"Splitting into {len(file_chunks)} array job(s)")

        job_ids = []
        chunk_dependency = dependency  # Track dependency for chunk chaining
        for chunk_idx, chunk in enumerate(file_chunks):
            # Render SBATCH script
            array_spec = None
            if len(chunk) > 1:
                array_spec = f"1-{len(chunk)}"
                if ntasks and ntasks < len(chunk):
                    array_spec += f"%{ntasks}"

            # Create one file list per array task
            file_list_pattern = str(job_dir / f"files_chunk_{chunk_idx}_task_*.txt")
            for task_idx, file_group in enumerate(chunk, start=1):
                task_file_list = (
                    job_dir / f"files_chunk_{chunk_idx}_task_{task_idx}.txt"
                )
                with open(task_file_list, "w", encoding="utf-8") as f:
                    for file_path in file_group:
                        f.write(f"{file_path}\n")

            batch_client = self._get_batch_client(profile_config)
            template = batch_client.load_template(
                self._get_template_name(profile_config)
            )

            script_content = template.render(
                array_spec=array_spec,
                job_name=(
                    f"{job_name}_{chunk_idx}" if len(file_chunks) > 1 else job_name
                ),
                log_dir=str(job_dir / "logs"),
                dependency=chunk_dependency,
                basedir=str(self.basedir),
                file_list_pattern=file_list_pattern,
                config=config,
                output=output,
                larcv_basedir=larcv_basedir,
                flashmatch=flashmatch,
                cvmfs=cvmfs,
                spine_cli_overrides=spine_cli_overrides,
                **profile_config,
            )

            # Write script
            script_path = (
                job_dir / f"submit_chunk_{chunk_idx}{batch_client.script_suffix}"
            )
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            script_path.chmod(0o755)

            # Submit
            print(f"\nSubmitting chunk {chunk_idx + 1}/{len(file_chunks)}:")
            print(f"  Script: {script_path}")
            print(f"  Files: {len(chunk)}")
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
            "detector": detector,
            "config": config,
            "original_config": original_config if apply_mods else config,
            "applied_modifiers": apply_mods or [],
            "set_overrides": set_overrides or [],
            "cvmfs": cvmfs,
            "profile": profile,
            "profile_config": profile_config,
            "num_files": len(file_list),
            "num_chunks": len(file_chunks),
            "job_ids": job_ids,
            "output": output,
            "submitted": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
        }
        self.batch_client.save_job_metadata(job_dir, metadata)

        print(f"\nJob directory: {job_dir}")
        print(f"Job metadata: {job_dir}/job_metadata.json")

        return job_ids

    def submit_pipeline(
        self, pipeline_path: str, dry_run: bool = False, preload: bool = False
    ) -> Dict[str, List[str]]:
        """Submit multi-stage pipeline with dependencies.

        Parameters
        ----------
        pipeline_path : str
            Path to pipeline YAML file
        dry_run : bool, optional
            Show what would be submitted, by default False
        preload : bool, optional
            Preload !download assets before each stage submission, by default False

        Returns
        -------
        Dict[str, List[str]]
            Dict mapping stage names to job IDs
        """
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline = yaml.safe_load(f)

        print(f"Loading pipeline: {pipeline_path}")
        print(f"Stages: {len(pipeline['stages'])}\n")

        job_map = {}
        cleanup_map = {}  # Track stages that need cleanup

        for stage in pipeline["stages"]:
            stage_name = stage["name"]
            print(f"Stage: {stage_name}")

            # Build dependency string
            depends_on = stage.get("depends_on", [])
            dependency = None
            if depends_on:
                dep_jobs = []
                for dep_stage in depends_on:
                    if dep_stage in job_map:
                        dep_jobs.extend(job_map[dep_stage])
                if dep_jobs:
                    dependency = f"afterok:{':'.join(dep_jobs)}"

            # Submit stage
            job_ids = self.submit_job(
                config=stage["config"],
                files=stage["files"],
                profile=stage.get("profile", "auto"),
                job_name=stage.get("job_name", stage_name),
                output=stage.get("output"),
                ntasks=stage.get("ntasks"),
                files_per_task=stage.get("files_per_task", 1),
                dependency=dependency,
                larcv_basedir=stage.get("larcv_basedir"),
                flashmatch=stage.get("flashmatch", False),
                cvmfs=stage.get("cvmfs", False),
                dry_run=dry_run,
                preload=preload,
            )

            job_map[stage_name] = job_ids

            # Store cleanup info if requested
            cleanup = stage.get("cleanup", [])
            if cleanup:
                if not isinstance(cleanup, list):
                    cleanup = [cleanup]
                cleanup_map[stage_name] = {"paths": cleanup, "job_ids": job_ids}
                print(f"  Cleanup scheduled for: {', '.join(cleanup)}")

            print()

        # Schedule cleanup jobs for stages that have downstream dependencies
        if cleanup_map:
            print("\nScheduling cleanup jobs...")
            for stage_name, cleanup_info in cleanup_map.items():
                # Find all stages that depend on this one
                dependent_stages = []
                for stage in pipeline["stages"]:
                    if stage_name in stage.get("depends_on", []):
                        dependent_stages.append(stage["name"])

                if dependent_stages:
                    # Wait for all dependent stages to complete
                    dep_jobs = []
                    for dep_stage in dependent_stages:
                        if dep_stage in job_map:
                            dep_jobs.extend(job_map[dep_stage])

                    if dep_jobs:
                        dependency = f"afterok:{':'.join(dep_jobs)}"
                        print(
                            f"  {stage_name}: cleanup after {', '.join(dependent_stages)} complete"
                        )

                        self.batch_client.submit_cleanup_job(
                            paths_to_clean=cleanup_info["paths"],
                            job_name=f"cleanup_{stage_name}",
                            dependency=dependency,
                            dry_run=dry_run,
                        )
                else:
                    print(f"  {stage_name}: no cleanup (no dependent stages found)")

        return job_map

    # Delegation methods for backward compatibility with tests
    # These forward calls to the appropriate component modules

    def _detect_detector(self, config: str) -> str:
        """Delegate to ConfigManager.detect_detector."""
        return self.config_mgr.detect_detector(config)

    def _get_profile(self, profile_name: str, detector: Optional[str] = None) -> Dict:
        """Delegate to ConfigManager.get_profile."""
        return self.config_mgr.get_profile(profile_name, detector)

    def _extract_version(self, config_path: Path) -> Optional[str]:
        """Delegate to ConfigManager.extract_version."""
        return self.config_mgr.extract_version(config_path)

    def _resolve_modifier_version(
        self,
        mod_name: str,
        available_versions: List[Path],
        base_version: Optional[str],
        explicit_version: Optional[str],
    ) -> Path:
        """Delegate to ConfigManager.resolve_modifier_version."""
        return self.config_mgr.resolve_modifier_version(
            mod_name, available_versions, base_version, explicit_version
        )

    def _discover_modifiers(self, config: str) -> Dict[str, List[Path]]:
        """Delegate to ConfigManager.discover_modifiers."""
        return self.config_mgr.discover_modifiers(config)

    def _create_composite_config(
        self,
        base_config: str,
        modifiers: List[str],
        job_dir: Path,
        detector: str,
    ) -> str:
        """Delegate to ConfigManager.create_composite_config."""
        return self.config_mgr.create_composite_config(
            base_config, modifiers, job_dir, detector
        )

    def _parse_files(
        self, file_input: List[str], source_type: str = "source"
    ) -> List[str]:
        """Delegate to FileHandler.parse_files."""
        return self.file_handler.parse_files(file_input, source_type)

    def _chunk_files(
        self, files: List[str], max_array_size: int, files_per_task: int
    ) -> List[List[str]]:
        """Delegate to FileHandler.chunk_files."""
        return self.file_handler.chunk_files(files, max_array_size, files_per_task)

    def _create_job_dir(self, job_name: str) -> Path:
        """Delegate to BatchClient.create_job_dir."""
        return self.batch_client.create_job_dir(job_name)

    def _save_job_metadata(self, job_dir: Path, metadata: Dict):
        """Delegate to BatchClient.save_job_metadata."""
        return self.batch_client.save_job_metadata(job_dir, metadata)

    def _preload_downloads(self, config: str):
        """Preload !download assets for a config on the submit host."""
        print("\nPreloading !download assets:")
        print(f"  Config: {config}")
        preload_downloads(config, self.basedir)

    def _create_latest_config(self, detector: str, job_dir: Path) -> str:
        """Delegate to ConfigManager.create_latest_config."""
        return self.config_mgr.create_latest_config(detector, job_dir)

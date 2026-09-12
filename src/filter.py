"""Submission of standalone file-aware entry-filter jobs."""

import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .component import SubmissionComponent
from .run_manager import RunManager


class FilterRunner(SubmissionComponent):
    """Submit one ``spine-filter scan`` or ``build`` operation."""

    def submit_filter(
        self,
        config: str,
        operation: str,
        run_dir: str,
        sources: Optional[Sequence[str]] = None,
        source_list: Optional[str] = None,
        cache_dir: Optional[str] = None,
        output: Optional[str] = None,
        output_source_list: Optional[str] = None,
        workers: Optional[int] = None,
        force: bool = False,
        profile: str = "s3df_milano",
        job_name: str = "spine_filter",
        dependency: Optional[str] = None,
        spine_path: Optional[str] = None,
        cvmfs: bool = False,
        dry_run: bool = False,
        retry: bool = False,
        **profile_overrides: Any,
    ) -> List[str]:
        """Submit a persistent scan or manifest-build job.

        Scan jobs reuse compatible per-source records automatically. Build jobs
        validate those records and atomically publish the consolidated manifest
        and its companion source list.
        """
        self._validate_operation(
            operation,
            sources=sources,
            source_list=source_list,
            cache_dir=cache_dir,
            output=output,
            output_source_list=output_source_list,
            workers=workers,
            force=force,
        )

        source_config = self.config_mgr.resolve_config_path(config)
        detector = self.config_mgr.detect_detector(str(source_config))
        job_dir = Path(run_dir).expanduser().resolve()
        if job_dir.exists() and any(job_dir.iterdir()) and not retry:
            raise ValueError(f"Filter run directory is not empty: {job_dir}")
        job_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = RunManager.create_attempt_dir(job_dir)
        RunManager.expose_attempt_logs(job_dir, has_array=False)

        profile_config = self.config_mgr.get_profile(profile, detector)
        site = profile_config.get("site", "s3df")
        scheduler = profile_config.get("scheduler")
        if scheduler is None:
            scheduler = "pbs" if site in ("anl", "polaris") else "slurm"
        if scheduler == "pbs" and profile_overrides.get("exclude"):
            raise ValueError("--exclude is only supported by Slurm profiles")
        profile_config.update(profile_overrides)
        if not profile_config.get("account") and detector in self.profiles["detectors"]:
            profile_config["account"] = self.profiles["detectors"][detector].get(
                "account", self.profiles["defaults"]["account"]
            )

        filter_cmd, bind_root = self.context.runtime.resolve_spine_filter_command(
            spine_path
        )
        if bind_root:
            bind_paths = profile_config.get("bind_paths")
            if not bind_paths:
                bind_paths = self.context.runtime.default_bind_paths_for_site(site)
            profile_config["bind_paths"] = self.context.runtime.merge_bind_paths(
                bind_paths, [bind_root]
            )

        resolved_cache_dir = str(Path(cache_dir).expanduser().resolve())
        Path(resolved_cache_dir).mkdir(parents=True, exist_ok=True)
        source_args = self._format_sources(sources, source_list)
        command_parts = [
            filter_cmd or "spine-filter",
            operation,
            "--config",
            shlex.quote(str(source_config)),
            source_args,
            "--cache-dir",
            shlex.quote(resolved_cache_dir),
        ]
        if operation == "scan":
            if workers is not None:
                command_parts.extend(["--workers", str(workers)])
            if force:
                command_parts.append("--force")
        else:
            resolved_output = str(Path(output).expanduser().resolve())
            resolved_source_list = str(Path(output_source_list).expanduser().resolve())
            Path(resolved_output).parent.mkdir(parents=True, exist_ok=True)
            Path(resolved_source_list).parent.mkdir(parents=True, exist_ok=True)
            command_parts.extend(
                [
                    "--output",
                    shlex.quote(resolved_output),
                    "--output-source-list",
                    shlex.quote(resolved_source_list),
                ]
            )

        command = " ".join(command_parts)
        batch_client = self.context.batch.get_batch_client(profile_config)
        template = batch_client.load_template(
            self.context.batch.get_template_name(profile_config)
        )
        script = template.render(
            array_spec=None,
            job_name=job_name,
            log_dir=str(attempt_dir),
            stdout_path=str(attempt_dir / "stdout.log"),
            stderr_path=str(attempt_dir / "stderr.log"),
            dependency=dependency,
            basedir=str(self.basedir),
            config=str(source_config),
            output=output,
            output_args=None,
            output_dir=None,
            output_suffix=None,
            file_list_pattern=None,
            input_manifest=None,
            task_dir_pattern=None,
            spine_log_dir=None,
            larcv_path=None,
            flashmatch_path=None,
            flashmatch=False,
            cvmfs=cvmfs,
            spine_cmd=None,
            spine_cli_overrides=None,
            custom_command=command,
            command_label=f"SPINE Entry Filter {operation.title()}",
            **profile_config,
        )
        script_path = attempt_dir / f"submit{batch_client.script_suffix}"
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)

        print(f"  Script: {script_path}")
        print(f"  Operation: {operation}")
        print(f"  Counter cache: {resolved_cache_dir}")
        print(f"  Profile: {profile} ({profile_config['description']})")
        if dependency:
            print(f"  Dependency: {dependency}")

        job_id = batch_client.submit(script_path, dry_run)
        job_ids = [job_id] if job_id else []
        if job_id:
            print(f"  Job ID: {job_id}")

        from version import __version__

        metadata: Dict[str, Any] = {
            "spine_prod_version": __version__,
            "job_name": job_name,
            "kind": "filter",
            "operation": operation,
            "run_dir": str(job_dir),
            "config": str(source_config),
            "sources": list(sources or []),
            "source_list": source_list,
            "cache_dir": resolved_cache_dir,
            "output": output,
            "output_source_list": output_source_list,
            "workers": workers,
            "force": force,
            "spine_path": spine_path,
            "profile": profile,
            "profile_config": profile_config,
            "job_ids": job_ids,
            "submitted": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
        }
        batch_client.save_job_metadata(attempt_dir, metadata)
        return job_ids

    @staticmethod
    def _format_sources(
        sources: Optional[Sequence[str]], source_list: Optional[str]
    ) -> str:
        """Format the mutually exclusive filter input selectors."""
        if source_list is not None:
            return f"--source-list {shlex.quote(source_list)}"
        return "--source " + " ".join(shlex.quote(source) for source in sources or [])

    @staticmethod
    def _validate_operation(
        operation: str,
        *,
        sources: Optional[Sequence[str]],
        source_list: Optional[str],
        cache_dir: Optional[str],
        output: Optional[str],
        output_source_list: Optional[str],
        workers: Optional[int],
        force: bool,
    ) -> None:
        """Validate the operation-specific standalone CLI contract."""
        if operation not in ("scan", "build"):
            raise ValueError("Filter operation must be one of: scan, build")
        if bool(sources) == bool(source_list):
            raise ValueError(
                "Filter jobs require exactly one of sources or source_list"
            )
        if not cache_dir:
            raise ValueError("Filter jobs require cache_dir")
        if workers is not None and (isinstance(workers, bool) or workers < 1):
            raise ValueError("Filter workers must be at least one")
        if not isinstance(force, bool):
            raise TypeError("Filter force must be a boolean")
        if operation == "scan":
            if output is not None or output_source_list is not None:
                raise ValueError("Filter scan cannot define build outputs")
        else:
            if not output or not output_source_list:
                raise ValueError("Filter build requires output and output_source_list")
            if workers is not None or force:
                raise ValueError("Filter build cannot define workers or force")

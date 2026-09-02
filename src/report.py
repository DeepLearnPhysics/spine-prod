"""Submission of standalone metric-report reduction jobs."""

import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .component import SubmissionComponent
from .run_manager import RunManager


class ReportRunner(SubmissionComponent):
    """Submit a CPU-only ``spine-report`` job after metric inference."""

    def submit_report(
        self,
        config: str,
        input_dir: str,
        output_dir: str,
        run_dir: str,
        checkpoint: Optional[str] = None,
        dataset: Optional[str] = None,
        profile: str = "s3df_milano",
        job_name: str = "spine_report",
        dependency: Optional[str] = None,
        spine_path: Optional[str] = None,
        cvmfs: bool = False,
        dry_run: bool = False,
        retry: bool = False,
        **profile_overrides: Any,
    ) -> List[str]:
        """Submit one reduction over completed analyzer CSV shards.

        The source report configuration is copied into the immutable attempt
        directory after run-specific provenance is applied. The report itself
        is written to a stable output directory so a retry deterministically
        refreshes the production artifacts while retaining scheduler history.
        """
        source_config = self.config_mgr.resolve_config_path(config)
        detector = self.config_mgr.detect_detector(str(source_config))
        job_dir = Path(run_dir).expanduser().resolve()
        if job_dir.exists() and any(job_dir.iterdir()) and not retry:
            raise ValueError(f"Report run directory is not empty: {job_dir}")
        job_dir.mkdir(parents=True, exist_ok=True)

        attempt_dir = RunManager.create_attempt_dir(job_dir)
        RunManager.expose_attempt_logs(job_dir, has_array=False)
        resolved_config = self._materialize_config(
            source_config,
            attempt_dir,
            checkpoint=checkpoint,
            dataset=dataset,
        )

        profile_config = self.config_mgr.get_profile(profile, detector)
        profile_config.update(profile_overrides)
        if not profile_config.get("account") and detector in self.profiles["detectors"]:
            profile_config["account"] = self.profiles["detectors"][detector].get(
                "account", self.profiles["defaults"]["account"]
            )

        report_cmd, bind_root = self.context.runtime.resolve_spine_report_command(
            spine_path
        )
        if bind_root:
            bind_paths = profile_config.get("bind_paths")
            if not bind_paths:
                bind_paths = self.context.runtime.default_bind_paths_for_site(
                    profile_config.get("site", "s3df")
                )
            profile_config["bind_paths"] = self.context.runtime.merge_bind_paths(
                bind_paths, [bind_root]
            )

        input_path = str(Path(input_dir).expanduser().resolve())
        output_path = str(Path(output_dir).expanduser().resolve())
        Path(output_path).mkdir(parents=True, exist_ok=True)
        command = " ".join(
            [
                report_cmd or "spine-report",
                "--config",
                shlex.quote(str(resolved_config)),
                "--input-dir",
                shlex.quote(input_path),
                "--output-dir",
                shlex.quote(output_path),
            ]
        )

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
            config=str(resolved_config),
            output=output_path,
            output_args=None,
            output_dir=output_path,
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
            command_label="SPINE Metric Report",
            **profile_config,
        )
        script_path = attempt_dir / f"submit{batch_client.script_suffix}"
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)

        print(f"  Script: {script_path}")
        print(f"  Metric input: {input_path}")
        print(f"  Report output: {output_path}")
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
            "kind": "report",
            "run_dir": str(job_dir),
            "config": str(resolved_config),
            "source_config": str(source_config),
            "input_dir": input_path,
            "output_dir": output_path,
            "checkpoint": checkpoint,
            "dataset": dataset,
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
    def _materialize_config(
        source: Path,
        attempt_dir: Path,
        checkpoint: Optional[str],
        dataset: Optional[str],
    ) -> Path:
        """Copy a report recipe while injecting run-specific provenance."""
        document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            raise TypeError("Report configuration must contain a mapping")
        metadata = dict(document.get("metadata") or {})
        if checkpoint is not None:
            metadata["checkpoint"] = checkpoint
        if dataset is not None:
            metadata["dataset"] = dataset
        document["metadata"] = metadata

        destination = attempt_dir / "report.yaml"
        destination.write_text(
            yaml.safe_dump(document),
            encoding="utf-8",
        )
        return destination

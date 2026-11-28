"""CPU batch submission for verified cache archival after producer completion."""

import json
import shlex
from pathlib import Path

from ..runs import RunManager
from ..sites import validate_allocation
from .base import SubmissionComponent
from .render import render_script


class ArchiveRunner(SubmissionComponent):
    """Keep cache copying off the login/cron controller."""

    def submit_archive(
        self,
        source,
        destination,
        cache_stage,
        producer_run_dir,
        run_dir,
        profile,
        account=None,
        dependency=None,
        job_name="cache_archive",
        dry_run=False,
    ):
        resources = self.config_mgr.get_profile(profile)
        if account is not None:
            resources["account"] = account
        validate_allocation(self.profiles, resources)
        job_dir = Path(run_dir).expanduser().resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        attempt = RunManager.create_attempt_dir(job_dir)
        RunManager.expose_attempt_logs(job_dir, has_array=False)
        roots = [
            str(Path(path).expanduser().resolve())
            for path in (source, destination, self.basedir)
        ]
        resources["bind_paths"] = self.context.runtime.merge_bind_paths(
            resources.get("bind_paths")
            or self.context.runtime.default_bind_paths_for_site(resources.get("site")),
            roots,
        )
        Path(roots[1]).mkdir(parents=True, exist_ok=True)
        metadata = json.loads(
            (Path(producer_run_dir) / "latest/job_metadata.json").read_text()
        )
        publication = metadata.get("cache_publication_id")
        if not publication:
            raise ValueError("Cache producer has no publication identity")
        command = shlex.join(
            [
                "python3",
                str(self.basedir / "scripts/cache_archive.py"),
                "archive",
                roots[0],
                roots[1],
                "--stage",
                cache_stage,
                "--publication-id",
                publication,
            ]
        )
        client = self.context.batch.get_batch_client(resources)
        script = render_script(
            client,
            self.context.batch.get_template_name(resources),
            attempt,
            1,
            1,
            basedir=str(self.basedir),
            job_name=job_name,
            log_dir=str(attempt),
            stdout_path=str(attempt / "stdout.log"),
            stderr_path=str(attempt / "stderr.log"),
            dependency=dependency,
            custom_command=command,
            config="cache archive",
            output=roots[1],
            graceful_stop_file=str(attempt / "graceful_stop"),
            **resources,
        )
        print(f"Cache archive: {source} -> {destination}; script: {script}")
        job = client.submit(script, dry_run)
        return [job] if job else []

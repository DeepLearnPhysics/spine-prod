"""Scheduler-aware control of running SPINE production jobs."""

import shutil
from typing import Optional

from .client import PBSClient, SlurmClient
from .component import SubmissionComponent


class JobController(SubmissionComponent):
    """Deliver intentional control requests through the active scheduler."""

    SCHEDULER_COMMANDS = {"slurm": "scancel", "pbs": "qsig"}

    def graceful_stop(
        self,
        job_id: str,
        scheduler: Optional[str] = None,
        dry_run: bool = False,
    ) -> str:
        """Ask a training job to checkpoint and complete successfully.

        Parameters
        ----------
        job_id : str
            Scheduler job identifier. Array task identifiers are accepted.
        scheduler : {"slurm", "pbs"}, optional
            Scheduler to contact. If omitted, infer it from locally available
            scheduler commands.
        dry_run : bool, default False
            Print the scheduler command without sending a signal.

        Returns
        -------
        str
            Scheduler selected for the request.
        """
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("Graceful stop requires a non-empty job ID")
        if any(character.isspace() for character in job_id):
            raise ValueError("Scheduler job IDs cannot contain whitespace")

        scheduler = self.resolve_scheduler(scheduler)
        client_cls = SlurmClient if scheduler == "slurm" else PBSClient
        client = client_cls(self.basedir, self.jobs_dir)
        client.graceful_stop(job_id, dry_run)
        return scheduler

    @classmethod
    def resolve_scheduler(cls, scheduler: Optional[str] = None) -> str:
        """Resolve an explicit scheduler or detect one available on ``PATH``."""
        if scheduler is not None:
            if scheduler not in cls.SCHEDULER_COMMANDS:
                raise ValueError("Scheduler must be one of: slurm, pbs")
            return scheduler

        available = [
            name
            for name, command in cls.SCHEDULER_COMMANDS.items()
            if shutil.which(command) is not None
        ]
        if len(available) == 1:
            return available[0]
        if not available:
            raise RuntimeError(
                "Could not detect a scheduler: neither scancel nor qsig is on PATH"
            )
        raise RuntimeError(
            "Both Slurm and PBS commands are available; specify --scheduler"
        )

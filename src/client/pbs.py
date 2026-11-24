"""PBS batch client."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from .base import BatchClient


class PBSClient(BatchClient):
    """Handles PBS-specific operations and job submissions."""

    script_suffix = ".pbs"

    def submit(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """Submit a job to PBS and return the job ID."""
        return self.submit_qsub(script_path, dry_run)

    def submit_qsub(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """Submit job to PBS and return job ID."""
        if dry_run:
            print(f"[DRY RUN] Would submit: {script_path}")
            with open(script_path, "r", encoding="utf-8") as f:
                print(f.read())
            return None

        # PBS does not inherit arbitrary submission-host exports by default.
        # Forward runtime selection and distributed diagnostics without -V.
        names = [
            name
            for name in (
                "SPINE_CONTAINER_PATH",
                "SPINE_CONTAINER_VERSION",
                "SPINE_CONTAINER_TAG",
                "NCCL_DEBUG",
                "MASTER_PORT",
            )
            if name in os.environ
        ]
        command = ["qsub"]
        if names:
            command.extend(["-v", ",".join(names)])
        command.append(str(script_path))
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"ERROR: qsub failed: {result.stderr}")
            return None

        output = result.stdout.strip().split()
        if not output:
            print("ERROR: qsub did not return a job ID")
            return None

        # qsub returns the job ID directly, often with a server suffix.
        return output[0]

    def graceful_stop(self, job_id: str, dry_run: bool = False) -> None:
        """Send ``SIGUSR1`` to the job's signal-forwarding PBS shell."""
        command = ["qsig", "-s", "SIGUSR1", job_id]
        if dry_run:
            print(f"[DRY RUN] Would run: {' '.join(command)}")
            return

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Failed to signal PBS job {job_id}: {message}")

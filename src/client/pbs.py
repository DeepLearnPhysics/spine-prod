"""PBS batch client."""

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

        result = subprocess.run(
            ["qsub", str(script_path)],
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

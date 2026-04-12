"""SLURM batch client."""

import subprocess
from pathlib import Path
from typing import List, Optional

from .base import BatchClient


class SlurmClient(BatchClient):
    """Handles SLURM-specific operations and job submissions."""

    script_suffix = ".sbatch"

    def submit(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """Submit a job to SLURM and return the job ID."""
        return self.submit_sbatch(script_path, dry_run)

    def submit_sbatch(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """Submit job to SLURM and return job ID."""
        if dry_run:
            print(f"[DRY RUN] Would submit: {script_path}")
            with open(script_path, "r", encoding="utf-8") as f:
                print(f.read())
            return None

        result = subprocess.run(
            ["sbatch", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"ERROR: sbatch failed: {result.stderr}")
            return None

        # Extract job ID from "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        return job_id

    def submit_cleanup_job(
        self,
        paths_to_clean: List[str],
        job_name: str,
        dependency: str,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Submit a cleanup job to remove intermediate files."""
        if dry_run:
            print(f"[DRY RUN] Would schedule cleanup of: {', '.join(paths_to_clean)}")
            return None

        cleanup_script = f"""#!/bin/bash
#SBATCH --account=neutrino:ml
#SBATCH --partition=milano
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=0:10:00
#SBATCH --job-name={job_name}
#SBATCH --dependency={dependency}

echo "Cleanup job started: $(date)"
echo "Removing intermediate files..."

"""

        for path in paths_to_clean:
            cleanup_script += f"""if [ -e "{path}" ]; then
    echo "  Removing: {path}"
    rm -rf "{path}"
else
    echo "  Not found (already removed?): {path}"
fi

"""

        cleanup_script += """echo "Cleanup completed: $(date)"
"""

        cleanup_dir = self.create_job_dir(f"cleanup_{job_name}")
        script_path = cleanup_dir / "cleanup.sbatch"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(cleanup_script)
        script_path.chmod(0o755)

        job_id = self.submit_sbatch(script_path, dry_run)
        if job_id:
            print(f"  Cleanup job ID: {job_id}")
        return job_id

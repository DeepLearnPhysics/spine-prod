"""SLURM job submission and management."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Template


class SlurmClient:
    """Handles SLURM-specific operations and job submissions."""

    def __init__(self, basedir: Path, jobs_dir: Path):
        """Initialize SlurmClient.

        Parameters
        ----------
        basedir : Path
            Base directory of spine-prod
        jobs_dir : Path
            Directory for storing job artifacts
        """
        self.basedir = basedir
        self.jobs_dir = jobs_dir

    def load_template(self, template_name: str) -> Template:
        """Load SBATCH job template by name.

        Parameters
        ----------
        template_name : str
            Name of the template file (e.g., 'job_template_nersc.sbatch')

        Returns
        -------
        Template
            Jinja2 template object

        Raises
        ------
        FileNotFoundError
            If template file not found
        """
        template_path = self.basedir / "templates" / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            return Template(f.read())

    def create_job_dir(self, job_name: str) -> Path:
        """Create directory for job artifacts.

        Parameters
        ----------
        job_name : str
            Name of the job

        Returns
        -------
        Path
            Path to created job directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = self.jobs_dir / f"{timestamp}_{job_name}"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        return job_dir

    def save_job_metadata(self, job_dir: Path, metadata: Dict):
        """Save job metadata for tracking.

        Parameters
        ----------
        job_dir : Path
            Job directory path
        metadata : Dict
            Metadata dictionary to save
        """
        metadata_path = job_dir / "job_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def submit_sbatch(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """Submit job to SLURM and return job ID.

        Parameters
        ----------
        script_path : Path
            Path to SBATCH script
        dry_run : bool, optional
            If True, print script without submitting, by default False

        Returns
        -------
        Optional[str]
            Job ID if submitted, None otherwise
        """
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
        """Submit a cleanup job to remove intermediate files.

        Parameters
        ----------
        paths_to_clean : List[str]
            List of file/directory paths to remove
        job_name : str
            Name for the cleanup job
        dependency : str
            SLURM dependency specification
        dry_run : bool, optional
            If True, print without submitting, by default False

        Returns
        -------
        Optional[str]
            Job ID if submitted, None otherwise
        """
        if dry_run:
            print(f"[DRY RUN] Would schedule cleanup of: {', '.join(paths_to_clean)}")
            return None

        # Create cleanup script
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

        # Write cleanup script
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleanup_dir = self.jobs_dir / f"{timestamp}_cleanup_{job_name}"
        cleanup_dir.mkdir(parents=True, exist_ok=True)

        script_path = cleanup_dir / "cleanup.sbatch"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(cleanup_script)
        script_path.chmod(0o755)

        # Submit
        result = subprocess.run(
            ["sbatch", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"WARNING: Cleanup job submission failed: {result.stderr}")
            return None

        job_id = result.stdout.strip().split()[-1]
        print(f"  Cleanup job ID: {job_id}")
        return job_id

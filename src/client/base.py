"""Shared batch client utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from jinja2 import Template


class BatchClient:
    """Shared functionality for batch scheduler clients."""

    script_suffix = ".sh"

    def __init__(self, basedir: Path, jobs_dir: Path):
        """Initialize the batch client.

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
        """Load a job template by name."""
        template_path = self.basedir / "templates" / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            return Template(f.read())

    def create_job_dir(self, job_name: str) -> Path:
        """Create directory for job artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = self.jobs_dir / f"{timestamp}_{job_name}"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        return job_dir

    def save_job_metadata(self, job_dir: Path, metadata: Dict):
        """Save job metadata for tracking."""
        metadata_path = job_dir / "job_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def submit(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """Submit a batch script and return its job ID."""
        raise NotImplementedError

    def format_dependency(self, dependency: Optional[str]) -> Optional[str]:
        """Format a dependency string for scheduler directives."""
        return dependency

    def dependency_afterok(self, job_id: str) -> str:
        """Build an afterok dependency for a single upstream job."""
        return f"afterok:{job_id}"

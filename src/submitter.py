"""Public submission façade and component wiring."""

import os
from pathlib import Path
from typing import Any

from .client import SlurmClient
from .config import ConfigManager
from .control import JobController
from .files import FileHandler
from .pipeline import PipelineRunner
from .preload import preload_downloads
from .submission.archive import ArchiveRunner
from .submission.arguments import SpineCLI
from .submission.batch import BatchRunner
from .submission.filter import FilterRunner
from .submission.interactive import InteractiveRunner
from .submission.report import ReportRunner
from .submission.runtime import RuntimeResolver


class Submitter:
    """Compose focused services behind the stable submission API."""

    def __init__(self, basedir: Path | None = None, central_dir: bool = False):
        """Initialize shared state and wire submission components."""
        self.basedir = basedir or Path(__file__).parent.parent
        self.config_mgr = ConfigManager(self.basedir)
        self.file_handler = FileHandler()
        self.jobs_dir = self.basedir / "runs" if central_dir else Path.cwd() / "runs"
        self.batch_client = SlurmClient(self.basedir, self.jobs_dir)

        if central_dir and not os.environ.get("SPINE_PROD_BASEDIR"):
            print("WARNING: SPINE_PROD_BASEDIR not set. Did you source configure.sh?")

        # Components share managers and paths through this façade; operational
        # behavior remains in the component that owns it.
        self.runtime = RuntimeResolver(self)
        self.control = JobController(self)
        self.spine_cli = SpineCLI()
        self.batch = BatchRunner(self)
        self.interactive = InteractiveRunner(self)
        self.filter = FilterRunner(self)
        self.report = ReportRunner(self)
        self.archive = ArchiveRunner(self)
        self.pipeline = PipelineRunner(self)

    @property
    def profiles(self) -> dict:
        """Return scheduler and detector profiles."""
        return self.config_mgr.profiles

    def list_modifiers(self, config_path: str) -> dict:
        """List modifier versions available to a configuration."""
        return self.config_mgr.list_modifiers(config_path)

    def submit_job(self, *args: Any, **kwargs: Any) -> list[str]:
        """Submit one job through the batch component."""
        return self.batch.submit_job(*args, **kwargs)

    def run_interactive(self, *args: Any, **kwargs: Any) -> int:
        """Execute one job through the interactive component."""
        return self.interactive.run_interactive(*args, **kwargs)

    def submit_report(self, *args: Any, **kwargs: Any) -> list[str]:
        """Submit one completed-metric report reduction."""
        return self.report.submit_report(*args, **kwargs)

    def submit_filter(self, *args: Any, **kwargs: Any) -> list[str]:
        """Submit one file-aware entry-filter operation."""
        return self.filter.submit_filter(*args, **kwargs)

    def submit_archive(self, **kwargs: Any) -> list[str]:
        """Submit a verified cache snapshot after producer accounting."""
        return self.archive.submit_archive(**kwargs)

    def submit_pipeline(self, *args: Any, **kwargs: Any) -> dict[str, list[str]]:
        """Submit an ordered workflow through the pipeline component."""
        return self.pipeline.submit_pipeline(*args, **kwargs)

    def graceful_stop(self, *args: Any, **kwargs: Any) -> str:
        """Request successful checkpointed completion of a training job."""
        return self.control.graceful_stop(*args, **kwargs)

    def preload_downloads(self, config: str):
        """Materialize SPINE download directives before execution."""
        config = self.config_mgr.normalize_config_request(config)
        print("\nPreloading !download assets:")
        print(f"  Config: {config}")
        preload_downloads(config, self.basedir)

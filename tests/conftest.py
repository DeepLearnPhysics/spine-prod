"""Shared pytest fixtures for spine-prod tests."""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally use other fixtures as parameters

import sys
from pathlib import Path

import pytest

# Add parent directory to path to import submit
sys.path.insert(0, str(Path(__file__).parent.parent))

from submit import SlurmSubmitter  # noqa: E402


@pytest.fixture
def workspace_root():
    """Return path to workspace root."""
    return Path(__file__).parent.parent


@pytest.fixture
def infer_root(workspace_root):
    """Return path to infer directory."""
    return workspace_root / "config" / "infer"


@pytest.fixture
def mock_submitter(workspace_root, tmp_path):
    """Create a SlurmSubmitter instance with mocked paths."""
    # Use real basedir but mock job_dir - pass Path not str
    submitter = SlurmSubmitter(basedir=workspace_root)
    submitter.jobs_dir = tmp_path / "test_job"
    submitter.jobs_dir.mkdir(parents=True, exist_ok=True)
    # Also update the SlurmClient's jobs_dir since that's where job dirs are created
    submitter.slurm_client.jobs_dir = submitter.jobs_dir
    return submitter

"""Tests for scheduler-aware job control."""

from unittest.mock import patch

import pytest

from src.client import PBSClient, SlurmClient


def test_graceful_stop_uses_explicit_scheduler(mock_submitter):
    """An explicit scheduler should select its native signal client."""
    with patch.object(SlurmClient, "graceful_stop") as stop:
        selected = mock_submitter.graceful_stop(
            "12345", scheduler="slurm", dry_run=True
        )

    assert selected == "slurm"
    stop.assert_called_once_with("12345", True)

    with patch.object(PBSClient, "graceful_stop") as stop:
        selected = mock_submitter.graceful_stop("678.server", scheduler="pbs")

    assert selected == "pbs"
    stop.assert_called_once_with("678.server", False)


@pytest.mark.parametrize("job_id", ["", "   ", "123 456"])
def test_graceful_stop_rejects_invalid_job_ids(mock_submitter, job_id):
    """Job IDs must be present and safe to pass as one subprocess argument."""
    with pytest.raises(ValueError, match="job ID|whitespace"):
        mock_submitter.graceful_stop(job_id, scheduler="slurm")


def test_scheduler_resolution_validates_explicit_name(mock_submitter):
    """Only scheduler clients implemented by spine-prod may be selected."""
    with pytest.raises(ValueError, match="slurm, pbs"):
        mock_submitter.graceful_stop("123", scheduler="lsf")


def test_scheduler_resolution_detects_one_available_command(mock_submitter):
    """Automatic selection should work when exactly one scheduler is present."""
    with patch(
        "src.control.shutil.which",
        side_effect=lambda command: "/bin/scancel" if command == "scancel" else None,
    ), patch.object(SlurmClient, "graceful_stop") as stop:
        assert mock_submitter.graceful_stop("123") == "slurm"

    stop.assert_called_once_with("123", False)


@pytest.mark.parametrize(
    ("available", "message"),
    [
        (set(), "neither scancel nor qsig"),
        ({"scancel", "qsig"}, "Both Slurm and PBS"),
    ],
)
def test_scheduler_resolution_rejects_ambiguous_environment(
    mock_submitter, available, message
):
    """Missing or ambiguous scheduler commands require explicit selection."""
    with patch(
        "src.control.shutil.which",
        side_effect=lambda command: f"/bin/{command}" if command in available else None,
    ):
        with pytest.raises(RuntimeError, match=message):
            mock_submitter.graceful_stop("123")

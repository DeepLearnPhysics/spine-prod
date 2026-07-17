"""Tests for SLURM submission and cleanup behavior."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from src.client.slurm import SlurmClient


@pytest.fixture
def client(tmp_path):
    return SlurmClient(tmp_path, tmp_path / "jobs")


def test_submit_parses_sbatch_job_id(client, tmp_path):
    result = CompletedProcess([], 0, "Submitted batch job 12345\n", "")

    with patch("src.client.slurm.subprocess.run", return_value=result):
        assert client.submit(tmp_path / "job.sbatch") == "12345"


def test_submit_returns_none_when_sbatch_fails(client, tmp_path, capsys):
    result = CompletedProcess([], 1, "", "scheduler unavailable")

    with patch("src.client.slurm.subprocess.run", return_value=result):
        assert client.submit(tmp_path / "job.sbatch") is None

    assert "sbatch failed: scheduler unavailable" in capsys.readouterr().out


def test_dry_run_prints_script_without_invoking_sbatch(client, tmp_path, capsys):
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\necho test\n")

    with patch("src.client.slurm.subprocess.run") as run:
        assert client.submit(script, dry_run=True) is None

    run.assert_not_called()
    assert "echo test" in capsys.readouterr().out


def test_cleanup_dry_run_does_not_create_job_directory(client, capsys):
    result = client.submit_cleanup_job(
        ["one.root", "two.root"], "cleanup", "afterok:42", dry_run=True
    )

    assert result is None
    assert not client.jobs_dir.exists()
    assert "one.root, two.root" in capsys.readouterr().out


def test_cleanup_writes_executable_script_and_submits_it(client):
    with patch.object(client, "submit_sbatch", return_value="99") as submit:
        assert (
            client.submit_cleanup_job(["/tmp/intermediate.root"], "reco", "afterok:42")
            == "99"
        )

    script = submit.call_args.args[0]
    content = script.read_text()
    assert script.stat().st_mode & 0o111
    assert "#SBATCH --dependency=afterok:42" in content
    assert 'rm -rf "/tmp/intermediate.root"' in content
    submit.assert_called_once_with(script, False)


def test_cleanup_returns_none_when_submission_returns_no_job_id(client, capsys):
    with patch.object(client, "submit_sbatch", return_value=None):
        assert (
            client.submit_cleanup_job(["temporary.root"], "reco", "afterok:42") is None
        )

    assert "Cleanup job ID" not in capsys.readouterr().out

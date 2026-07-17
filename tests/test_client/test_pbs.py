"""Tests for PBS submission behavior."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from src.client.pbs import PBSClient


@pytest.fixture
def client(tmp_path):
    return PBSClient(tmp_path, tmp_path)


def test_submit_parses_qsub_job_id(client, tmp_path):
    script = tmp_path / "job.pbs"
    script.touch()
    result = CompletedProcess([], 0, "12345.server\n", "")

    with patch("src.client.pbs.subprocess.run", return_value=result) as run:
        assert client.submit(script) == "12345.server"

    run.assert_called_once()


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (CompletedProcess([], 1, "", "queue unavailable"), "qsub failed"),
        (CompletedProcess([], 0, "  \n", ""), "did not return a job ID"),
    ],
)
def test_submit_returns_none_for_qsub_failures(
    client, tmp_path, result, message, capsys
):
    with patch("src.client.pbs.subprocess.run", return_value=result):
        assert client.submit(tmp_path / "job.pbs") is None

    assert message in capsys.readouterr().out


def test_dry_run_prints_script_without_invoking_qsub(client, tmp_path, capsys):
    script = tmp_path / "job.pbs"
    script.write_text("#!/bin/bash\necho test\n")

    with patch("src.client.pbs.subprocess.run") as run:
        assert client.submit(script, dry_run=True) is None

    run.assert_not_called()
    output = capsys.readouterr().out
    assert "[DRY RUN]" in output
    assert "echo test" in output

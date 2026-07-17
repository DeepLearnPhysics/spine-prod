"""Tests for shared batch-client behavior."""

import json

import pytest

from src.client.base import BatchClient


def test_load_template_reads_repository_template(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "job.sh").write_text("Hello {{ name }}")
    client = BatchClient(tmp_path, tmp_path / "jobs")

    assert client.load_template("job.sh").render(name="SPINE") == "Hello SPINE"


def test_load_template_rejects_missing_template(tmp_path):
    client = BatchClient(tmp_path, tmp_path / "jobs")

    with pytest.raises(FileNotFoundError, match="Template not found"):
        client.load_template("missing.sh")


def test_create_job_dir_and_save_metadata(tmp_path):
    client = BatchClient(tmp_path, tmp_path / "jobs")

    job_dir = client.create_job_dir("reco")
    client.save_job_metadata(job_dir, {"job_id": "123"})

    assert job_dir.name.endswith("_reco")
    assert (job_dir / "logs").is_dir()
    assert (job_dir / "output").is_dir()
    assert json.loads((job_dir / "job_metadata.json").read_text()) == {"job_id": "123"}


def test_base_submission_and_dependency_helpers(tmp_path):
    client = BatchClient(tmp_path, tmp_path)

    with pytest.raises(NotImplementedError):
        client.submit(tmp_path / "job.sh")
    assert client.format_dependency(None) is None
    assert client.format_dependency("afterany:1") == "afterany:1"
    assert client.dependency_afterok("42") == "afterok:42"

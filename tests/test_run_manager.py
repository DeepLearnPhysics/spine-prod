"""Tests for persistent run and training lifecycle management."""

import hashlib
import json
from pathlib import Path

import pytest

from src.run_manager import RunManager


def write_config(path: Path, value: str = "base: {}\n") -> str:
    """Write and return a small configuration path."""
    path.write_text(value, encoding="utf-8")
    return str(path)


def initialize_run(tmp_path):
    """Create a training run and return its paths."""
    config = write_config(tmp_path / "train.yaml")
    run_dir = tmp_path / "default"
    assert RunManager.prepare_training_run(run_dir, config) is None
    return run_dir, config


def checkpoint(run_dir: Path, iteration: int, name: str = "snapshot") -> Path:
    """Create a checkpoint fixture."""
    path = run_dir / "weights" / f"{name}-{iteration}.ckpt"
    path.touch()
    return path


def checksum_sidecar(path: Path, digest=None, filename=None) -> Path:
    """Write a conventional checksum sidecar for a checkpoint fixture."""
    digest = digest or hashlib.sha256(path.read_bytes()).hexdigest()
    filename = filename or path.name
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(f"{digest}  {filename}\n", encoding="utf-8")
    return sidecar


def validation_log(log_dir: Path, iteration: int, content: str = "iter,loss\n0,1\n"):
    """Create a validation log associated with a checkpoint iteration."""
    path = log_dir / f"inference_log-{iteration + 1:07d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_config_digest_and_new_training_run(tmp_path):
    run_dir, config = initialize_run(tmp_path)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["kind"] == "training"
    assert metadata["training_config"] == str(Path(config).resolve())
    assert metadata["training_config_sha256"] == RunManager.config_digest(config)
    assert (run_dir / "weights").is_dir()
    assert (run_dir / "tensorboard" / "train").is_dir()
    assert (run_dir / "tensorboard" / "validation").is_dir()


def test_new_training_run_rejects_existing_run_and_unrelated_content(tmp_path):
    run_dir, config = initialize_run(tmp_path)
    assert RunManager.prepare_training_run(run_dir, config) is None
    RunManager.record_training_jobs(run_dir, [])
    RunManager.record_training_jobs(run_dir, ["123"])
    with pytest.raises(ValueError, match="already exists"):
        RunManager.prepare_training_run(run_dir, config)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["job_ids"] == ["123"]
    assert metadata["last_submitted"]

    other = tmp_path / "other"
    other.mkdir()
    (other / "unexpected").touch()
    with pytest.raises(ValueError, match="not empty"):
        RunManager.prepare_training_run(other, config)


def test_new_training_run_allows_internal_config_workspace(tmp_path):
    config = write_config(tmp_path / "train.yaml")
    run_dir = tmp_path / "default"
    (run_dir / ".spine-prod" / "configs").mkdir(parents=True)

    assert RunManager.prepare_training_run(run_dir, config) is None


def test_resume_requires_matching_run_config_and_checkpoint(tmp_path):
    config = write_config(tmp_path / "train.yaml")
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="no training run"):
        RunManager.prepare_training_run(missing, config, resume=True)

    run_dir, config = initialize_run(tmp_path)
    write_config(Path(config), "base:\n  epochs: 2\n")
    with pytest.raises(ValueError, match="different training configuration"):
        RunManager.prepare_training_run(run_dir, config, resume=True)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    metadata["training_config_sha256"] = RunManager.config_digest(config)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="no checkpoints"):
        RunManager.prepare_training_run(run_dir, config, resume=True)


def test_checkpoint_order_latest_and_explicit_resume(tmp_path):
    run_dir, config = initialize_run(tmp_path)
    first = checkpoint(run_dir, 9)
    latest = checkpoint(run_dir, 100)
    checkpoint(run_dir, 500, name="other")

    assert RunManager.checkpoints(run_dir) == [(9, first), (100, latest)]
    assert RunManager.latest_checkpoint(run_dir) == latest
    assert RunManager.prepare_training_run(run_dir, config, resume=True) == latest
    assert (
        RunManager.prepare_training_run(run_dir, config, resume_from=str(first))
        == first.resolve()
    )
    assert RunManager.latest_checkpoint(tmp_path / "empty") is None


def test_checkpoint_digest_verifies_sidecars_and_accepts_legacy_files(tmp_path):
    checkpoint_path = tmp_path / "snapshot-1.ckpt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        RunManager.checkpoint_digest(checkpoint_path)

    checkpoint_path.write_bytes(b"checkpoint")
    expected = hashlib.sha256(b"checkpoint").hexdigest()

    assert RunManager.checkpoint_digest(checkpoint_path) is None
    checksum_sidecar(checkpoint_path)
    assert RunManager.checkpoint_digest(checkpoint_path) == expected

    Path(f"{checkpoint_path}.sha256").write_text("malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        RunManager.checkpoint_digest(checkpoint_path)

    checksum_sidecar(checkpoint_path, filename="another.ckpt")
    with pytest.raises(ValueError, match="names 'another.ckpt'"):
        RunManager.checkpoint_digest(checkpoint_path)

    checksum_sidecar(checkpoint_path, digest="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        RunManager.checkpoint_digest(checkpoint_path)


def test_resume_and_validation_reject_corrupt_checkpoints(tmp_path):
    run_dir, config = initialize_run(tmp_path)
    saved = checkpoint(run_dir, 9)
    checksum_sidecar(saved, digest="0" * 64)

    with pytest.raises(ValueError, match="does not match"):
        RunManager.prepare_training_run(run_dir, config, resume=True)

    validation_config = write_config(tmp_path / "val.yaml")
    with pytest.raises(ValueError, match="does not match"):
        RunManager.prepare_validation(run_dir, validation_config)


def test_explicit_resume_checkpoint_must_belong_to_run(tmp_path):
    run_dir, _ = initialize_run(tmp_path)
    outside = tmp_path / "snapshot-1.ckpt"
    outside.touch()
    with pytest.raises(ValueError, match="weights directory"):
        RunManager.resolve_resume_checkpoint(run_dir, str(outside))

    invalid = checkpoint(run_dir, 1, name="checkpoint")
    with pytest.raises(ValueError, match="Invalid checkpoint name"):
        RunManager.resolve_resume_checkpoint(run_dir, str(invalid))


def test_validation_log_directory_and_completion_detection(tmp_path):
    run_dir = tmp_path / "run"
    assert RunManager.validation_log_dir(run_dir, None) == run_dir
    assert RunManager.validation_log_dir(run_dir, "data-v1") == (
        run_dir / "validation" / "data-v1"
    )
    with pytest.raises(ValueError, match="validation-name"):
        RunManager.validation_log_dir(run_dir, "bad name")

    complete = validation_log(run_dir, 9)
    assert RunManager.checkpoint_is_validated(run_dir, 9)
    complete.write_text("iter,loss\n", encoding="utf-8")
    assert not RunManager.checkpoint_is_validated(run_dir, 9)
    complete.unlink()
    complete.mkdir()
    assert not RunManager.checkpoint_is_validated(run_dir, 9)


def test_prepare_validation_selects_only_missing_checkpoints(tmp_path):
    run_dir, _ = initialize_run(tmp_path)
    validation_config = write_config(tmp_path / "val.yaml", "base:\n  train: null\n")
    first = checkpoint(run_dir, 9)
    second = checkpoint(run_dir, 19)
    validation_log(run_dir, 9)

    log_dir, selected = RunManager.prepare_validation(run_dir, validation_config)

    assert log_dir == run_dir
    assert selected == [second]
    state = json.loads(
        (run_dir / ".spine-prod" / "validation" / "primary.json").read_text()
    )
    assert state["name"] == "primary"
    assert state["config_sha256"] == RunManager.config_digest(validation_config)

    named_dir, named_selected = RunManager.prepare_validation(
        run_dir, validation_config, validation_name="data"
    )
    assert named_dir == run_dir / "validation" / "data"
    assert named_selected == [first, second]


def test_prepare_validation_checks_run_checkpoints_and_identity(tmp_path):
    config = write_config(tmp_path / "val.yaml")
    with pytest.raises(ValueError, match="No training run"):
        RunManager.prepare_validation(tmp_path / "missing", config)

    run_dir, _ = initialize_run(tmp_path)
    with pytest.raises(ValueError, match="No checkpoints"):
        RunManager.prepare_validation(run_dir, config)

    saved = checkpoint(run_dir, 3)
    RunManager.prepare_validation(run_dir, config)
    write_config(Path(config), "base:\n  iterations: 2\n")
    with pytest.raises(ValueError, match="different configuration"):
        RunManager.prepare_validation(run_dir, config)

    log_dir, selected = RunManager.prepare_validation(run_dir, config, rerun=True)
    assert log_dir == run_dir
    assert selected == [saved]


def test_create_stage_submission_directories(tmp_path):
    train = RunManager.create_submission_dir(tmp_path, "train")
    validation = RunManager.create_submission_dir(tmp_path, "validation", "data")

    assert train.parent == tmp_path / "submissions" / "train"
    assert (train / "logs").is_dir()
    assert validation.parent == tmp_path / "submissions" / "validation" / "data"
    assert (validation / "logs").is_dir()

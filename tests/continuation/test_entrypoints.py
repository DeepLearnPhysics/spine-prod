"""Continuation routing and public submission validation."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

import src
from src.continuation import controller
from src.pipeline.validation import StageValidator


def test_lazy_import_rejects_unknown_attribute():
    """Unknown public attributes retain normal Python import behavior."""
    with pytest.raises(AttributeError):
        getattr(src, "unknown")


def test_old_python_has_actionable_error(monkeypatch):
    """Login-node Python failures explain which environment to load."""
    monkeypatch.setattr(sys, "version_info", (3, 8, 0))
    with pytest.raises(ImportError, match="module load python"):
        exec(compile(Path(src.__file__).read_text(), src.__file__, "exec"), {})


@pytest.mark.parametrize(
    "stage",
    [dict(stage="validation", continuation={}), dict(stage="train", continuation=True)],
)
def test_continuation_requires_training_mapping(stage):
    """Invalid lifecycle policies fail during document validation."""
    with pytest.raises(ValueError, match="continuation"):
        StageValidator._validate_lifecycle("train", stage)


@pytest.mark.parametrize("slice_stage", [False, True])
def test_pipeline_routes_to_durable_controller(
    mock_submitter, tmp_path, monkeypatch, slice_stage
):
    """Continuation pipelines must use logical completion gates without slicing."""
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                workspace=str(tmp_path / "run"),
                stages=[
                    dict(
                        name="train",
                        stage="train",
                        config="config/train/generic/uresnet/train_240718.yaml",
                        profile="s3df_ampere",
                        run_dir="${workspace}/train",
                        continuation={"runtime_budget": "24:00:00"},
                    )
                ],
            )
        )
    )
    start = Mock(return_value={"train": ["1"]})
    monkeypatch.setattr(controller, "start", start)
    if slice_stage:
        with pytest.raises(ValueError, match="stage slicing"):
            mock_submitter.submit_pipeline(str(path), from_stage="train", dry_run=True)
        start.assert_not_called()
    else:
        assert mock_submitter.submit_pipeline(str(path), dry_run=True) == {
            "train": ["1"]
        }
        start.assert_called_once()


@pytest.mark.parametrize("dry_run", [False, True])
def test_batch_continuation_shared_workdir(
    mock_submitter, tmp_path, monkeypatch, dry_run
):
    """Sites permitting shared state render a supervised training allocation."""
    monkeypatch.setattr(
        mock_submitter.batch, "get_batch_client", lambda *a: mock_submitter.batch_client
    )
    monkeypatch.setattr(mock_submitter.batch_client, "submit", Mock(return_value="1"))
    mock_submitter.submit_job(
        config="config/train/generic/uresnet/train_240718.yaml",
        profile="s3df_ampere",
        stage="train",
        run_dir=str(tmp_path / "run"),
        continuation={"runtime_budget": "24:00:00"},
        dry_run=dry_run,
    )
    mock_submitter.batch_client.submit.assert_called_once()
    assert mock_submitter.batch_client.submit.call_args.args[1] is dry_run


def test_batch_rejects_inference_continuation(mock_submitter):
    """Continuation cannot supervise inference jobs."""
    with pytest.raises(ValueError, match="ordinary training"):
        mock_submitter.submit_job(
            config="config/train/generic/uresnet/train_240718.yaml",
            profile="s3df_ampere",
            continuation={},
        )

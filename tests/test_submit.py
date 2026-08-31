"""Tests for the submit.py command-line entry point."""

import sys
from unittest.mock import Mock, patch

import pytest

import submit


def run_main(*args, submitter=None):
    """Run the CLI with a mocked Submitter instance."""
    submitter = submitter or Mock()
    with patch.object(sys, "argv", ["submit.py", *args]), patch.object(
        submit, "Submitter", return_value=submitter
    ) as submitter_class:
        result = submit.main()
    return result, submitter, submitter_class


def test_list_modifiers_prints_discovered_versions(capsys):
    submitter = Mock()
    submitter.list_modifiers.return_value = {
        "config_name": "full_chain_250101",
        "base_version": "250101",
        "modifiers": {
            "data": {"selected": "250101", "available": ["241201", "250101"]}
        },
    }

    result, _, submitter_class = run_main(
        "--list-mods", "infer/example/full_chain_250101.yaml", submitter=submitter
    )

    assert result == 0
    submitter_class.assert_called_once_with(central_dir=False)
    submitter.list_modifiers.assert_called_once_with(
        "infer/example/full_chain_250101.yaml"
    )
    assert "data" in capsys.readouterr().out


def test_list_modifiers_reports_lookup_errors(capsys):
    submitter = Mock()
    submitter.list_modifiers.side_effect = FileNotFoundError("missing config")

    result, _, _ = run_main("--list-mods", "missing.yaml", submitter=submitter)

    assert result == 1
    assert "ERROR: missing config" in capsys.readouterr().err


def test_list_modifiers_prints_multiple_usage_example(capsys):
    submitter = Mock()
    submitter.list_modifiers.return_value = {
        "config_name": "full_chain.yaml",
        "base_version": None,
        "modifiers": {
            "data": {"selected": "250101", "available": ["250101"]},
            "lite": {"selected": "250102", "available": ["250102"]},
        },
    }

    result, _, _ = run_main("--list-mods", "full_chain.yaml", submitter=submitter)

    assert result == 0
    output = capsys.readouterr().out
    assert "version: unversioned" in output
    assert "--apply-mods data lite" in output


def test_list_modifiers_explains_empty_result(capsys):
    submitter = Mock()
    submitter.list_modifiers.return_value = {
        "config_name": "full_chain.yaml",
        "base_version": None,
        "modifiers": {},
    }

    result, _, _ = run_main(
        "--list-mods", "infer/example/full_chain.yaml", submitter=submitter
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "(none found)" in output
    assert "example/modifier/" in output


def test_interactive_mode_forwards_runtime_options():
    submitter = Mock()
    submitter.run_interactive.return_value = 7

    result, _, _ = run_main(
        "--config",
        "config.yaml",
        "--source-list",
        "files.txt",
        "--interactive",
        "--task-id",
        "3",
        "--interactive-runtime",
        "container",
        "--world-size",
        "4",
        "--minibatch-size",
        "2",
        "--num-workers",
        "8",
        "--epochs",
        "25",
        "--set",
        "model.detect_anomaly=true",
        submitter=submitter,
    )

    assert result == 7
    submitter.run_interactive.assert_called_once()
    kwargs = submitter.run_interactive.call_args.kwargs
    assert kwargs["files"] == ["files.txt"]
    assert kwargs["source_type"] == "source_list"
    assert kwargs["task_id"] == 3
    assert kwargs["interactive_runtime"] == "container"
    assert kwargs["world_size"] == 4
    assert kwargs["minibatch_size"] == 2
    assert kwargs["num_workers"] == 8
    assert kwargs["epochs"] == 25
    assert kwargs["set_overrides"] == ["model.detect_anomaly=true"]


def test_batch_mode_forwards_profile_overrides(capsys):
    submitter = Mock()
    submitter.submit_job.return_value = ["123", "124"]

    result, _, submitter_class = run_main(
        "--config",
        "config.yaml",
        "--source",
        "input.root",
        "--central-dir",
        "--partition",
        "gpu",
        "--gpus-per-node",
        "4",
        "--bind-paths",
        "/data,/scratch",
        "--batch-size",
        "16",
        "--iterations",
        "100",
        submitter=submitter,
    )

    assert result == 0
    submitter_class.assert_called_once_with(central_dir=True)
    kwargs = submitter.submit_job.call_args.kwargs
    assert kwargs["source_type"] == "source"
    assert kwargs["files"] == ["input.root"]
    assert kwargs["partition"] == "gpu"
    assert kwargs["gpus_per_node"] == 4
    assert kwargs["bind_paths"] == "/data,/scratch"
    assert kwargs["batch_size"] == 16
    assert kwargs["iterations"] == 100
    assert "Submitted job IDs: 123, 124" in capsys.readouterr().out


def test_batch_mode_forwards_run_lifecycle_options():
    submitter = Mock()
    submitter.submit_job.return_value = []

    result, _, _ = run_main(
        "--config",
        "val.yaml",
        "--stage",
        "validation",
        "--run-dir",
        "/runs/default",
        "--validation-name",
        "data",
        "--rerun-validation",
        "--tensorboard",
        submitter=submitter,
    )

    assert result == 0
    kwargs = submitter.submit_job.call_args.kwargs
    assert kwargs["stage"] == "validation"
    assert kwargs["run_dir"] == "/runs/default"
    assert kwargs["validation_name"] == "data"
    assert kwargs["rerun_validation"] is True
    assert kwargs["tensorboard"] is True


def test_batch_mode_forwards_training_and_validation_sources():
    """Training and validation sources are forwarded independently."""
    submitter = Mock()
    submitter.submit_job.return_value = []

    result, _, _ = run_main(
        "--config",
        "train.yaml",
        "--stage",
        "train",
        "--run-dir",
        "/runs/default",
        "--source-list",
        "train.txt",
        "--val-source-list",
        "validation.txt",
        submitter=submitter,
    )

    assert result == 0
    kwargs = submitter.submit_job.call_args.kwargs
    assert kwargs["files"] == ["train.txt"]
    assert kwargs["source_type"] == "source_list"
    assert kwargs["validation_files"] == ["validation.txt"]
    assert kwargs["validation_source_type"] == "source_list"


def test_pipeline_mode_prints_stage_jobs(capsys):
    submitter = Mock()
    submitter.submit_pipeline.return_value = {"reco": ["42"], "post": ["43"]}

    result, _, _ = run_main(
        "--pipeline",
        "pipeline.yaml",
        "--workspace",
        "/runs/benchmark",
        "--preload",
        submitter=submitter,
    )

    assert result == 0
    submitter.submit_pipeline.assert_called_once_with(
        "pipeline.yaml",
        dry_run=False,
        preload=True,
        overrides={},
        workspace="/runs/benchmark",
    )
    output = capsys.readouterr().out
    assert "reco: 42" in output
    assert "post: 43" in output


def test_submission_errors_return_failure(capsys):
    submitter = Mock()
    submitter.submit_job.side_effect = RuntimeError("scheduler unavailable")

    result, _, _ = run_main("--config", "config.yaml", submitter=submitter)

    assert result == 1
    assert "ERROR: scheduler unavailable" in capsys.readouterr().err


def test_local_output_warns_that_option_is_deprecated(capsys):
    submitter = Mock()
    submitter.submit_job.return_value = []

    result, _, _ = run_main(
        "--config", "config.yaml", "--local-output", submitter=submitter
    )

    assert result == 0
    assert "--local-output is deprecated" in capsys.readouterr().err


def test_pipeline_rejects_interactive_mode():
    with pytest.raises(SystemExit, match="2"):
        run_main("--pipeline", "pipeline.yaml", "--interactive")


def test_pipeline_mode_forwards_global_overrides():
    submitter = Mock()
    submitter.submit_pipeline.return_value = {}

    result, _, _ = run_main(
        "--pipeline",
        "pipeline.yaml",
        "--spine",
        "/software/spine-dev",
        "--profile",
        "s3df_hopper",
        "--account",
        "neutrino",
        "--gpus",
        "4",
        "--time",
        "12:00:00",
        "--iterations",
        "10",
        "--cvmfs",
        submitter=submitter,
    )

    assert result == 0
    assert submitter.submit_pipeline.call_args.kwargs["overrides"] == {
        "spine_path": "/software/spine-dev",
        "profile": "s3df_hopper",
        "account": "neutrino",
        "gpus": 4,
        "time": "12:00:00",
        "iterations": 10,
        "cvmfs": True,
    }
    assert submitter.submit_pipeline.call_args.kwargs["workspace"] is None


def test_workspace_is_rejected_outside_pipeline_mode():
    """A pipeline workspace must not silently behave like a job run directory."""
    with pytest.raises(SystemExit, match="2"):
        run_main("--config", "config.yaml", "--workspace", "/runs/benchmark")


def test_cli_rejects_undeclared_long_option_abbreviations():
    with pytest.raises(SystemExit, match="2"):
        run_main("--pipeline", "pipeline.yaml", "--spin", "/software/spine-dev")


@pytest.mark.parametrize(
    "stage_args",
    [
        ("--source", "input.root"),
        ("--apply-mods", "data"),
        ("--set", "base.seed=7"),
        ("--ntasks", "4"),
        ("--output", "/tmp/output.h5"),
        ("--task-id", "2"),
    ],
)
def test_pipeline_mode_rejects_stage_specific_cli_options(stage_args):
    with pytest.raises(SystemExit, match="2"):
        run_main("--pipeline", "pipeline.yaml", *stage_args)


@pytest.mark.parametrize(
    "args",
    [
        ("--config", "train.yaml", "--interactive", "--stage", "train"),
        ("--pipeline", "pipeline.yaml", "--run-dir", "/runs/default"),
    ],
)
def test_non_batch_modes_reject_run_lifecycle_options(args):
    with pytest.raises(SystemExit, match="2"):
        run_main(*args)


@pytest.mark.parametrize(
    "args",
    [
        ("--config", "train.yaml", "--interactive", "--val-source", "val.root"),
        ("--pipeline", "pipeline.yaml", "--val-source", "val.root"),
    ],
)
def test_non_batch_modes_reject_validation_sources(args):
    with pytest.raises(SystemExit, match="2"):
        run_main(*args)

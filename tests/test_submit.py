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
        "--set",
        "base.world_size=0",
        submitter=submitter,
    )

    assert result == 7
    submitter.run_interactive.assert_called_once()
    kwargs = submitter.run_interactive.call_args.kwargs
    assert kwargs["files"] == ["files.txt"]
    assert kwargs["source_type"] == "source_list"
    assert kwargs["task_id"] == 3
    assert kwargs["interactive_runtime"] == "container"
    assert kwargs["set_overrides"] == ["base.world_size=0"]


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
    assert "Submitted job IDs: 123, 124" in capsys.readouterr().out


def test_pipeline_mode_prints_stage_jobs(capsys):
    submitter = Mock()
    submitter.submit_pipeline.return_value = {"reco": ["42"], "post": ["43"]}

    result, _, _ = run_main(
        "--pipeline", "pipeline.yaml", "--preload", submitter=submitter
    )

    assert result == 0
    submitter.submit_pipeline.assert_called_once_with(
        "pipeline.yaml", dry_run=False, preload=True
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

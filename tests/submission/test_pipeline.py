"""Submission integration tests for pipeline behavior."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.runs import RunManager


class TestPipelineSubmission:
    """Tests for multi-stage dependencies and cleanup scheduling."""

    def test_submit_pipeline_indexes_each_latest_attempt(
        self, mock_submitter, tmp_path
    ):
        """A workspace should expose one shallow link per submitted stage."""
        workspace = tmp_path / "workspace"
        run_dir = workspace / "cache" / "train" / "segmentation"
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "workspace": str(workspace),
                    "stages": [
                        {
                            "name": "cache_train_segmentation",
                            "config": "cache.yaml",
                            "run_dir": str(run_dir),
                        }
                    ],
                }
            )
        )

        def submit_attempt(**options):
            RunManager.create_attempt_dir(Path(options["run_dir"]))
            return ["10"]

        with patch.object(mock_submitter, "submit_job", side_effect=submit_attempt):
            result = mock_submitter.submit_pipeline(str(pipeline_path))

        assert result == {"cache_train_segmentation": ["10"]}
        link = workspace / "logs" / "cache_train_segmentation"
        assert link.is_symlink()
        assert link.resolve() == (run_dir / "latest").resolve()

    @pytest.mark.parametrize("warm_start", [None, "/weights/full-chain.ckpt"])
    def test_submit_pipeline_forwards_warm_start_only_to_declared_training_stage(
        self, mock_submitter, tmp_path, warm_start
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "train",
                            "config": "train.yaml",
                            "stage": "train",
                            "run_dir": "/tmp/train",
                            "warm_start": {"grappa": "grappa_shower"},
                        },
                        {
                            "name": "cache",
                            "config": "cache.yaml",
                            "depends_on": ["train"],
                        },
                    ]
                }
            )
        )

        with patch.object(
            mock_submitter, "submit_job", side_effect=[["10"], ["20"]]
        ) as submit_job:
            result = mock_submitter.submit_pipeline(
                str(pipeline_path), warm_start=warm_start
            )

        assert result == {"train": ["10"], "cache": ["20"]}
        train = submit_job.call_args_list[0].kwargs
        cache = submit_job.call_args_list[1].kwargs
        assert train["warm_start_path"] == warm_start
        assert train["warm_start_modules"] == (
            {"grappa": "grappa_shower"} if warm_start else None
        )
        assert cache["warm_start_path"] is None
        assert cache["warm_start_modules"] is None

    @pytest.mark.parametrize("warm_start", ["", 4])
    def test_submit_pipeline_rejects_invalid_warm_start_path(
        self, mock_submitter, tmp_path, warm_start
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "train",
                            "config": "train.yaml",
                            "stage": "train",
                            "run_dir": "/tmp/train",
                            "warm_start": {"model": "full.model"},
                        }
                    ]
                }
            )
        )

        with pytest.raises(ValueError, match="non-empty string"):
            mock_submitter.submit_pipeline(str(pipeline_path), warm_start=warm_start)

    def test_submit_pipeline_rejects_warm_start_without_stage_mappings(
        self, mock_submitter, tmp_path
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump({"stages": [{"name": "inference", "config": "infer.yaml"}]})
        )

        with pytest.raises(ValueError, match="declares no warm_start mappings"):
            mock_submitter.submit_pipeline(
                str(pipeline_path), warm_start="/weights/full.ckpt"
            )

    def test_submit_pipeline_forwards_cli_source_and_override_fields(
        self, mock_submitter, tmp_path
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "train",
                            "config": "train.yaml",
                            "stage": "train",
                            "source_list": "train_files.txt",
                            "val_source_list": "validation_files.txt",
                            "val_entry_fraction_range": [0.0, 0.5],
                            "run_dir": "/tmp/train",
                            "time": "08:00:00",
                            "set": "model.weight_path=/tmp/seed.ckpt",
                        },
                        {
                            "name": "configured_inputs",
                            "config": "mixed.yaml",
                            "stage": "train",
                            "run_dir": "/tmp/configured-inputs",
                            "num_files": 2,
                            "val_num_files": 1,
                            "num_entries": 10000,
                            "val_num_entries": 1000,
                            "sources": {
                                "larcv": {"source": "raw.root"},
                                "hdf5": {"source": "cache.h5"},
                            },
                            "validation_sources": {
                                "larcv": {"source": "validation.root"},
                                "hdf5": {"source": "validation.h5"},
                            },
                            "weight_path": "/tmp/full-seed.ckpt",
                            "module_weight": {"uresnet_ppn": "/tmp/snapshot-best.ckpt"},
                        },
                        {
                            "name": "export",
                            "config": "model.yaml",
                            "depends_on": ["configured_inputs"],
                            "run_dir": "/tmp/export",
                            "export_weights": "/tmp/full-chain.ckpt",
                            "module_weight": {"uresnet_ppn": "/tmp/snapshot-best.ckpt"},
                        },
                    ]
                }
            )
        )

        with patch.object(
            mock_submitter, "submit_job", side_effect=[["10"], ["20"], ["30"]]
        ) as submit_job:
            result = mock_submitter.submit_pipeline(
                str(pipeline_path),
                stage_module_weights=[
                    ["configured_inputs", "uresnet_ppn=/tmp/cli-seed.ckpt"],
                    ["configured_inputs", "graph_spice=/tmp/graph-seed.ckpt"],
                ],
            )

        assert result == {
            "train": ["10"],
            "configured_inputs": ["20"],
            "export": ["30"],
        }
        train = submit_job.call_args_list[0].kwargs
        assert train["files"] == ["train_files.txt"]
        assert train["source_type"] == "source_list"
        assert train["validation_files"] == ["validation_files.txt"]
        assert train["validation_source_type"] == "source_list"
        assert train["val_entry_fraction_range"] == [0.0, 0.5]
        assert train["set_overrides"] == ["model.weight_path=/tmp/seed.ckpt"]
        assert train["time"] == "08:00:00"

        configured = submit_job.call_args_list[1].kwargs
        assert configured["files"] is None
        assert configured["named_sources"] == {
            "larcv": {"source": "raw.root"},
            "hdf5": {"source": "cache.h5"},
        }
        assert configured["validation_named_sources"] == {
            "larcv": {"source": "validation.root"},
            "hdf5": {"source": "validation.h5"},
        }
        assert configured["module_weights"] == {
            "uresnet_ppn": "/tmp/cli-seed.ckpt",
            "graph_spice": "/tmp/graph-seed.ckpt",
        }
        assert configured["weight_path"] == "/tmp/full-seed.ckpt"
        assert configured["num_entries"] == 10000
        assert configured["val_num_entries"] == 1000
        assert configured["num_files"] == 2
        assert configured["val_num_files"] == 1

        export = submit_job.call_args_list[2].kwargs
        assert export["dependency"] == "afterok:20"
        assert export["export_weights"] == "/tmp/full-chain.ckpt"
        assert export["module_weights"] == {"uresnet_ppn": "/tmp/snapshot-best.ckpt"}

    def test_submit_pipeline_dispatches_report_stage(self, mock_submitter, tmp_path):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "evaluate",
                            "config": "evaluate.yaml",
                            "source": "validation.root",
                            "run_dir": "/tmp/metrics/raw",
                            "weight_path": "/tmp/full-chain.ckpt",
                        },
                        {
                            "name": "report",
                            "kind": "report",
                            "depends_on": ["evaluate"],
                            "config": "report.yaml",
                            "input_dir": "/tmp/metrics/raw/latest",
                            "output_dir": "/tmp/metrics/report/artifacts",
                            "run_dir": "/tmp/metrics/report",
                            "checkpoint": "/tmp/full-chain.ckpt",
                            "dataset": "validation.root",
                            "dataset_selection": {"entry_fraction_range": [0.5, 1.0]},
                        },
                    ]
                }
            )
        )

        with (
            patch.object(mock_submitter, "submit_job", return_value=["10"]) as submit,
            patch.object(
                mock_submitter, "submit_report", return_value=["20"]
            ) as submit_report,
        ):
            result = mock_submitter.submit_pipeline(str(pipeline_path))

        assert result == {"evaluate": ["10"], "report": ["20"]}
        assert submit.call_args.kwargs["weight_path"] == "/tmp/full-chain.ckpt"
        report = submit_report.call_args.kwargs
        assert report["dependency"] == "afterok:10"
        assert report["input_dir"] == "/tmp/metrics/raw/latest"
        assert report["output_dir"] == "/tmp/metrics/report/artifacts"
        assert report["checkpoint"] == "/tmp/full-chain.ckpt"
        assert report["dataset_selection"] == {"entry_fraction_range": [0.5, 1.0]}

    def test_submit_pipeline_dispatches_filter_stage(self, mock_submitter, tmp_path):
        """Standalone filter stages preserve dependencies and operation fields."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "scan",
                            "kind": "filter",
                            "operation": "scan",
                            "config": "filter.yaml",
                            "source": ["one.root", "two.root"],
                            "cache_dir": "/tmp/counts",
                            "run_dir": "/tmp/filter/scan",
                            "workers": 4,
                        },
                        {
                            "name": "build",
                            "kind": "filter",
                            "operation": "build",
                            "depends_on": ["scan"],
                            "config": "filter.yaml",
                            "source": ["one.root", "two.root"],
                            "cache_dir": "/tmp/counts",
                            "output": "/tmp/accepted.yaml",
                            "output_source_list": "/tmp/accepted.txt",
                            "run_dir": "/tmp/filter/build",
                        },
                    ]
                }
            )
        )

        with patch.object(
            mock_submitter, "submit_filter", side_effect=[["10"], ["20"]]
        ) as submit_filter:
            result = mock_submitter.submit_pipeline(str(pipeline_path))

        assert result == {"scan": ["10"], "build": ["20"]}
        assert submit_filter.call_args_list[0].kwargs["sources"] == [
            "one.root",
            "two.root",
        ]
        assert submit_filter.call_args_list[0].kwargs["workers"] == 4
        assert submit_filter.call_args_list[1].kwargs["dependency"] == "afterok:10"
        assert submit_filter.call_args_list[1].kwargs["output"] == (
            "/tmp/accepted.yaml"
        )

    def test_submit_pipeline_rejects_invalid_report_dataset_selection(
        self, mock_submitter, tmp_path
    ):
        """Report provenance selections must be structured mappings."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "report",
                            "kind": "report",
                            "config": "report.yaml",
                            "input_dir": "/tmp/metrics/raw",
                            "output_dir": "/tmp/metrics/report/artifacts",
                            "run_dir": "/tmp/metrics/report",
                            "dataset_selection": [0.5, 1.0],
                        }
                    ]
                }
            )
        )

        with pytest.raises(TypeError, match="dataset_selection must be a mapping"):
            mock_submitter.submit_pipeline(str(pipeline_path))

    def test_submit_pipeline_forwards_in_place_cache_extension(
        self, mock_submitter, tmp_path
    ):
        """Pipeline cache stages should expose config-owned writer routing."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "append",
                            "config": "cache.yaml",
                            "source": "cache.spine-cache",
                            "in_place": True,
                            "cache_repository": "cache.spine-cache",
                            "cache_stage": "fragmentation",
                        }
                    ]
                }
            )
        )

        with patch.object(mock_submitter, "submit_job", return_value=["10"]) as submit:
            result = mock_submitter.submit_pipeline(str(pipeline_path))

        assert result == {"append": ["10"]}
        assert submit.call_args.kwargs["in_place"] is True
        assert submit.call_args.kwargs["cache_repository"] == "cache.spine-cache"
        assert submit.call_args.kwargs["cache_stage"] == "fragmentation"

    @pytest.mark.parametrize(
        ("stage_fields", "message"),
        [
            ({"in_place": True, "output": "cache.h5"}, "cannot be combined"),
            ({"in_place": "yes"}, "in_place must be a boolean"),
            (
                {"in_place": True, "stage": "train", "run_dir": "/tmp/train"},
                "in_place requires stage=inference",
            ),
            ({"cache_repository": "cache.spine-cache"}, "must define"),
            (
                {
                    "cache_repository": "cache.spine-cache",
                    "cache_stage": "fragmentation",
                    "stage": "train",
                    "run_dir": "/tmp/train",
                    "source": "input.root",
                },
                "cache publication requires stage=inference",
            ),
            (
                {
                    "cache_repository": "cache.spine-cache",
                    "cache_stage": "fragmentation",
                },
                "cache publication requires inputs",
            ),
        ],
    )
    def test_submit_pipeline_rejects_invalid_in_place_usage(
        self, mock_submitter, tmp_path, stage_fields, message
    ):
        """In-place routing must be explicit, boolean, and inference-only."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "invalid",
                            "config": "cache.yaml",
                            **stage_fields,
                        }
                    ]
                }
            )
        )

        with pytest.raises((TypeError, ValueError), match=message):
            mock_submitter.submit_pipeline(str(pipeline_path))

    def test_submit_pipeline_layers_defaults_stage_values_and_cli_overrides(
        self, mock_submitter, tmp_path
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "defaults": {
                        "profile": "s3df_ampere",
                        "time": "04:00:00",
                        "spine_path": "/software/default-spine",
                        "gpus_per_node": 2,
                    },
                    "stages": [
                        {
                            "name": "first",
                            "config": "first.yaml",
                            "time": "08:00:00",
                            "spine_path": "/software/stage-spine",
                        },
                        {
                            "name": "second",
                            "config": "second.yaml",
                            "profile": "s3df_milano",
                        },
                    ],
                }
            )
        )

        with patch.object(
            mock_submitter, "submit_job", side_effect=[["10"], ["20"]]
        ) as submit_job:
            result = mock_submitter.submit_pipeline(
                str(pipeline_path),
                overrides={
                    "profile": "s3df_hopper",
                    "spine_path": "/software/cli-spine",
                    "gpus": 4,
                    "exclude": "sdfampere014",
                },
            )

        assert result == {"first": ["10"], "second": ["20"]}
        for call in submit_job.call_args_list:
            assert call.kwargs["profile"] == "s3df_hopper"
            assert call.kwargs["spine_path"] == "/software/cli-spine"
            assert call.kwargs["gpus"] == 4
            assert call.kwargs["exclude"] == "sdfampere014"
            assert "gpus_per_node" not in call.kwargs
        assert submit_job.call_args_list[0].kwargs["time"] == "08:00:00"
        assert submit_job.call_args_list[1].kwargs["time"] == "04:00:00"

    def test_submit_pipeline_restarts_ordered_suffix(
        self, mock_submitter, tmp_path, capsys
    ):
        """Restart should skip completed stages and rebuild new dependencies."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {"name": "prepare", "config": "prepare.yaml"},
                        {
                            "name": "cache_train",
                            "config": "cache.yaml",
                            "depends_on": ["prepare"],
                        },
                        {
                            "name": "cache_validation",
                            "config": "cache.yaml",
                            "depends_on": ["prepare"],
                        },
                        {
                            "name": "train",
                            "config": "train.yaml",
                            "stage": "train",
                            "run_dir": "/tmp/train",
                            "depends_on": ["cache_train", "cache_validation"],
                        },
                    ]
                }
            )
        )

        with patch.object(
            mock_submitter,
            "submit_job",
            side_effect=[["20"], ["21"], ["30"]],
        ) as submit_job:
            result = mock_submitter.submit_pipeline(
                str(pipeline_path),
                from_stage="cache_train",
                to_stage="cache_validation",
            )

        assert result == {
            "cache_train": ["20"],
            "cache_validation": ["21"],
        }
        assert submit_job.call_args_list[0].kwargs["dependency"] is None
        assert submit_job.call_args_list[1].kwargs["dependency"] is None
        assert all(call.kwargs["retry"] for call in submit_job.call_args_list)
        output = capsys.readouterr().out
        assert "Skipped as completed: prepare" in output
        assert "Reusing completed dependencies: prepare" in output
        assert "Not selected after stop: train" in output

    def test_submit_pipeline_rejects_unknown_restart_before_submission(
        self, mock_submitter, tmp_path
    ):
        """A typo in the restart boundary must not submit any jobs."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump({"stages": [{"name": "prepare", "config": "prepare.yaml"}]})
        )

        with patch.object(mock_submitter, "submit_job") as submit_job:
            with pytest.raises(ValueError, match="Unknown pipeline restart stage"):
                mock_submitter.submit_pipeline(str(pipeline_path), from_stage="missing")

        submit_job.assert_not_called()

    @pytest.mark.parametrize("skip", [False, True])
    def test_submit_pipeline_selects_sparse_stages_and_contracts_dependencies(
        self, mock_submitter, tmp_path, capsys, skip
    ):
        """Sparse retries retain order and nearest selected dependencies."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {"name": "source", "config": "source.yaml"},
                        {
                            "name": "cache_train",
                            "config": "cache.yaml",
                            "depends_on": ["source"],
                        },
                        {
                            "name": "cache_validation",
                            "config": "cache.yaml",
                            "depends_on": ["source"],
                        },
                        {
                            "name": "train",
                            "config": "train.yaml",
                            "depends_on": ["cache_train", "cache_validation"],
                        },
                        {
                            "name": "particle_train",
                            "config": "particle.yaml",
                            "depends_on": ["train"],
                        },
                        {
                            "name": "particle_validation",
                            "config": "particle.yaml",
                            "depends_on": ["train"],
                        },
                    ]
                }
            )
        )

        with patch.object(
            mock_submitter,
            "submit_job",
            side_effect=[["20"], ["21"], ["30"], ["31"]],
        ) as submit_job:
            result = mock_submitter.submit_pipeline(
                str(pipeline_path),
                skip_stages=["source", "train"] if skip else None,
                select_stages=(
                    None
                    if skip
                    else [
                        "particle_validation",
                        "cache_validation",
                        "particle_train",
                        "cache_train",
                    ]
                ),
            )

        assert list(result) == [
            "cache_train",
            "cache_validation",
            "particle_train",
            "particle_validation",
        ]
        assert submit_job.call_args_list[0].kwargs["dependency"] is None
        assert submit_job.call_args_list[1].kwargs["dependency"] is None
        assert submit_job.call_args_list[2].kwargs["dependency"] == "afterok:20:21"
        assert submit_job.call_args_list[3].kwargs["dependency"] == "afterok:20:21"
        assert all(call.kwargs["retry"] for call in submit_job.call_args_list)
        output = capsys.readouterr().out
        label = "Skipped as completed" if skip else "Not selected"
        assert f"{label}: source, train" in output

    def test_report_rejects_non_mapping_configuration(self, tmp_path):
        """Report recipes must be mappings before provenance is injected."""
        from src.submission.report import ReportRunner

        source = tmp_path / "source.yaml"
        source.write_text("- invalid\n", encoding="utf-8")
        attempt = tmp_path / "attempt"
        attempt.mkdir()

        with pytest.raises(TypeError, match="must contain a mapping"):
            ReportRunner._materialize_config(source, attempt, None, None)

    def test_submit_report_rejects_existing_run_without_retry(
        self, mock_submitter, tmp_path
    ):
        """An existing report attempt requires an explicit retry."""
        source = tmp_path / "report.yaml"
        source.write_text("metrics: {}\n", encoding="utf-8")
        run_dir = tmp_path / "report-run"
        run_dir.mkdir()
        (run_dir / "existing").touch()

        with pytest.raises(ValueError, match="run directory is not empty"):
            mock_submitter.submit_report(
                config=str(source),
                input_dir=str(tmp_path / "input"),
                output_dir=str(tmp_path / "output"),
                run_dir=str(run_dir),
            )

    def test_submit_report_rejects_slurm_exclusion_for_pbs(
        self, mock_submitter, tmp_path
    ):
        """PBS report jobs must reject the Slurm-only exclusion option."""
        source = tmp_path / "report.yaml"
        source.write_text("metrics: {}\n", encoding="utf-8")

        with (
            patch.object(
                mock_submitter.config_mgr,
                "get_profile",
                return_value={
                    "site": "anl",
                    "scheduler": "pbs",
                    "description": "PBS test",
                },
            ),
            pytest.raises(ValueError, match="only supported by Slurm"),
        ):
            mock_submitter.submit_report(
                config=str(source),
                input_dir=str(tmp_path / "input"),
                output_dir=str(tmp_path / "output"),
                run_dir=str(tmp_path / "run"),
                exclude="node01",
            )

    @pytest.mark.parametrize("bind_paths", [None, "/existing"])
    @pytest.mark.parametrize("job_id", ["42", None])
    def test_submit_report_adds_checkout_bind_account_and_dependency(
        self, mock_submitter, tmp_path, capsys, bind_paths, job_id
    ):
        """Report jobs derive omitted site resources and bind a source checkout."""
        source = tmp_path / "report.yaml"
        source.write_text("metrics: {}\n", encoding="utf-8")
        profile = {
            "site": "s3df",
            "scheduler": "slurm",
            "description": "Slurm test",
            "bind_paths": bind_paths,
        }

        with (
            patch.object(
                mock_submitter.config_mgr,
                "detect_detector",
                return_value="icarus",
            ),
            patch.object(
                mock_submitter.config_mgr, "get_profile", return_value=profile
            ),
            patch.object(
                mock_submitter.runtime,
                "resolve_spine_report_command",
                return_value=("python3 -m spine.bin.report", "/checkout"),
            ),
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value=job_id),
        ):
            assert mock_submitter.submit_report(
                config=str(source),
                input_dir=str(tmp_path / "input"),
                output_dir=str(tmp_path / "output"),
                run_dir=str(tmp_path / "run"),
                dependency="afterok:41",
                dry_run=job_id is None,
            ) == ([job_id] if job_id else [])

        assert profile["bind_paths"] == f"{bind_paths or '/sdf/'},/checkout"
        assert profile["account"]
        output = capsys.readouterr().out
        assert "Dependency: afterok:41" in output
        assert ("Job ID: 42" in output) is bool(job_id)

    def test_resolve_spine_report_command_path_forms_and_fallbacks(
        self, mock_submitter, tmp_path
    ):
        """Report resolution supports overrides and ignores the host PATH."""
        checkout = tmp_path / "checkout"
        report_module = checkout / "src" / "spine" / "bin" / "report.py"
        report_module.parent.mkdir(parents=True)
        report_module.touch()

        for configured in (checkout / "bin" / "spine", checkout / "custom"):
            command, bind_root = mock_submitter.runtime.resolve_spine_report_command(
                str(configured)
            )
            assert command.endswith("python3 -m spine.bin.report")
            assert bind_root == str(checkout)

        with pytest.raises(RuntimeError, match="does not provide spine.bin.report"):
            mock_submitter.runtime.resolve_spine_report_command(str(tmp_path / "bad"))

        with patch(
            "src.submission.runtime.shutil.which", return_value="/usr/bin/spine-report"
        ):
            assert mock_submitter.runtime.resolve_spine_report_command() == (
                None,
                None,
            )

        with patch("src.submission.runtime.shutil.which", return_value=None):
            assert mock_submitter.runtime.resolve_spine_report_command() == (None, None)

    @pytest.mark.parametrize(
        ("value", "error", "message"),
        [
            ((0.0,), ValueError, "exactly two values"),
            (("zero", 1.0), TypeError, "bounds must be numbers"),
            ((float("nan"), 1.0), ValueError, "bounds must be finite"),
        ],
    )
    def test_validate_entry_fraction_range_rejects_malformed_bounds(
        self, mock_submitter, value, error, message
    ):
        """Entry ranges require two finite numeric bounds."""
        with pytest.raises(error, match=message):
            mock_submitter.spine_cli.validate_fraction_range("--entry", value)

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (
                {
                    "stage": "train",
                    "run_dir": "/tmp/train",
                    "resume": True,
                    "weight_path": "/weights/start.ckpt",
                },
                "weight-path cannot be combined",
            ),
            (
                {"val_entry_fraction_range": (0.0, 0.5)},
                "valid only for training",
            ),
            (
                {"val_num_entries": 100},
                "valid only for training",
            ),
            (
                {"num_entries": 100, "entry_fraction_range": (0.0, 0.5)},
                "cannot be combined",
            ),
            (
                {"stage": "train", "run_dir": "/tmp/train", "in_place": True},
                "in-place is valid only for inference",
            ),
        ],
    )
    def test_submit_job_rejects_additional_lifecycle_conflicts(
        self, mock_submitter, options, message
    ):
        """Single-job validation rejects incompatible lifecycle controls."""
        with pytest.raises(ValueError, match=message):
            mock_submitter.submit_job(config="config.yaml", **options)

    def test_submit_pipeline_rejects_reversed_stage_range(
        self, mock_submitter, tmp_path
    ):
        """A bounded restart must retain forward pipeline order."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {"name": "first", "config": "first.yaml"},
                        {"name": "second", "config": "second.yaml"},
                    ]
                }
            )
        )

        with pytest.raises(ValueError, match="must not precede"):
            mock_submitter.submit_pipeline(
                str(pipeline_path),
                from_stage="second",
                to_stage="first",
            )

    def test_submit_pipeline_allows_weight_override_for_skipped_stage(
        self, mock_submitter, tmp_path
    ):
        """A reusable restart command may retain seeds for earlier stages."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {"name": "train", "config": "train.yaml"},
                        {"name": "cache", "config": "cache.yaml"},
                    ]
                }
            )
        )

        with patch.object(mock_submitter, "submit_job", return_value=[]) as submit_job:
            result = mock_submitter.submit_pipeline(
                str(pipeline_path),
                from_stage="cache",
                stage_module_weights=[["train", "model=/weights/model.ckpt"]],
            )

        assert result == {"cache": []}
        assert submit_job.call_count == 1

    def test_submit_pipeline_allows_weight_override_for_deferred_stage(
        self, mock_submitter, tmp_path
    ):
        """A reusable bounded command may retain seeds for later stages."""
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {"name": "first", "config": "first.yaml"},
                        {"name": "later", "config": "later.yaml"},
                    ]
                }
            )
        )

        with patch.object(mock_submitter, "submit_job", return_value=[]) as submit_job:
            result = mock_submitter.submit_pipeline(
                str(pipeline_path),
                to_stage="first",
                stage_module_weights=[
                    ["later", "model=/weights/later.ckpt"],
                ],
            )

        assert result == {"first": []}
        assert submit_job.call_count == 1

    @pytest.mark.parametrize(
        ("pipeline", "message"),
        [
            (
                {"unexpected": True, "stages": []},
                "Unknown pipeline field",
            ),
            (
                {
                    "defaults": {"source": "input.root"},
                    "stages": [{"name": "stage", "config": "config.yaml"}],
                },
                "defaults contain stage-specific",
            ),
            (
                {
                    "stages": [
                        {
                            "name": "stage",
                            "config": "config.yaml",
                            "typo_profile": "s3df_hopper",
                        }
                    ]
                },
                "unknown field",
            ),
        ],
    )
    def test_submit_pipeline_rejects_unknown_fields_before_submission(
        self, mock_submitter, tmp_path, pipeline, message
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(yaml.safe_dump(pipeline))

        with patch.object(mock_submitter, "submit_job") as submit_job:
            with pytest.raises(ValueError, match=message):
                mock_submitter.submit_pipeline(str(pipeline_path))

        submit_job.assert_not_called()

    def test_submit_pipeline_rejects_unknown_cli_override(
        self, mock_submitter, tmp_path
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump({"stages": [{"name": "stage", "config": "config.yaml"}]})
        )

        with patch.object(mock_submitter, "submit_job") as submit_job:
            with pytest.raises(ValueError, match="Unknown pipeline override"):
                mock_submitter.submit_pipeline(
                    str(pipeline_path), overrides={"source": "input.root"}
                )

        submit_job.assert_not_called()

    @pytest.mark.parametrize(
        "conflicting_keys",
        [
            {"source": "input.root", "source_list": "inputs.txt"},
            {
                "val_source": "validation.root",
                "val_source_list": "validation.txt",
            },
        ],
    )
    def test_submit_pipeline_rejects_conflicting_source_fields(
        self, mock_submitter, tmp_path, conflicting_keys
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "ambiguous",
                            "config": "config.yaml",
                            **conflicting_keys,
                        }
                    ]
                }
            )
        )

        with pytest.raises(ValueError, match="must specify only one"):
            mock_submitter.submit_pipeline(str(pipeline_path))

    @pytest.mark.parametrize(
        ("stage_fields", "message"),
        [
            (
                {
                    "source": "raw.root",
                    "sources": {"larcv": {"source": "raw.root"}},
                },
                "cannot combine sources",
            ),
            (
                {
                    "val_source": "validation.root",
                    "validation_sources": {"larcv": {"source": "validation.root"}},
                },
                "cannot combine validation_sources",
            ),
            ({"module_weight": ["bad"]}, "module_weight must be a mapping"),
            ({"export_weights": 4}, "export_weights must be a string"),
            (
                {"entry_fraction_range": [0.8, 0.2]},
                "0 <= START < STOP <= 1",
            ),
            ({"num_entries": 0}, "positive integer"),
            ({"num_files": 0}, "positive integer"),
            (
                {
                    "num_entries": 10,
                    "entry_fraction_range": [0.0, 0.5],
                },
                "cannot combine num_entries",
            ),
        ],
    )
    def test_submit_pipeline_rejects_invalid_structured_fields(
        self, mock_submitter, tmp_path, stage_fields, message
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "invalid",
                            "config": "config.yaml",
                            **stage_fields,
                        }
                    ]
                }
            )
        )

        with pytest.raises((TypeError, ValueError), match=message):
            mock_submitter.submit_pipeline(str(pipeline_path))

    def test_submit_pipeline_chains_stages_and_schedules_cleanup(
        self, mock_submitter, tmp_path, capsys
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "prepare",
                            "config": "prepare.yaml",
                            "files": ["raw.root"],
                            "cleanup": "intermediate.root",
                        },
                        {
                            "name": "reconstruct",
                            "config": "reconstruct.yaml",
                            "files": ["intermediate.root"],
                            "depends_on": ["prepare"],
                            "profile": "s3df_ampere",
                            "output_suffix": "reco",
                            "larcv_basedir": "/software/larcv",
                            "flashmatch": True,
                            "cvmfs": True,
                        },
                        {
                            "name": "orphan",
                            "config": "orphan.yaml",
                            "files": ["other.root"],
                            "cleanup": ["unused.root"],
                        },
                    ]
                }
            )
        )

        with (
            patch.object(
                mock_submitter,
                "submit_job",
                side_effect=[["10"], ["20", "21"], ["30"]],
            ) as submit_job,
            patch.object(
                mock_submitter.batch_client, "submit_cleanup_job", return_value="40"
            ) as cleanup,
        ):
            result = mock_submitter.submit_pipeline(
                str(pipeline_path), dry_run=True, preload=True
            )

        assert result == {
            "prepare": ["10"],
            "reconstruct": ["20", "21"],
            "orphan": ["30"],
        }
        assert submit_job.call_count == 3
        reconstruct = submit_job.call_args_list[1].kwargs
        assert reconstruct["dependency"] == "afterok:10"
        assert reconstruct["allow_missing_inputs"] is True
        assert reconstruct["larcv_path"] == "/software/larcv"
        assert reconstruct["flashmatch"] is True
        assert reconstruct["cvmfs"] is True
        assert reconstruct["preload"] is True
        assert submit_job.call_args_list[0].kwargs["allow_missing_inputs"] is False
        cleanup.assert_called_once_with(
            paths_to_clean=["intermediate.root"],
            job_name="cleanup_prepare",
            dependency="afterok:20:21",
            dry_run=True,
        )
        assert "orphan: no cleanup" in capsys.readouterr().out

    def test_submit_pipeline_skips_cleanup_without_downstream_job_ids(
        self, mock_submitter, tmp_path
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "prepare",
                            "config": "prepare.yaml",
                            "files": ["raw.root"],
                            "cleanup": ["temporary.root"],
                        },
                        {
                            "name": "consume",
                            "config": "consume.yaml",
                            "files": ["temporary.root"],
                            "depends_on": ["prepare"],
                        },
                    ]
                }
            )
        )

        with (
            patch.object(mock_submitter, "submit_job", side_effect=[["10"], []]),
            patch.object(mock_submitter.batch_client, "submit_cleanup_job") as cleanup,
        ):
            result = mock_submitter.submit_pipeline(str(pipeline_path))

        assert result == {"prepare": ["10"], "consume": []}
        cleanup.assert_not_called()

    def test_submit_pipeline_rejects_unknown_dependency_before_submission(
        self, mock_submitter, tmp_path
    ):
        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "stages": [
                        {
                            "name": "standalone",
                            "config": "standalone.yaml",
                            "files": ["input.root"],
                            "depends_on": ["external_stage"],
                        }
                    ]
                }
            )
        )

        with patch.object(mock_submitter, "submit_job") as submit_job:
            with pytest.raises(ValueError, match="unknown or later stage"):
                mock_submitter.submit_pipeline(str(pipeline_path))

        submit_job.assert_not_called()


def test_pipeline_runner_forwards_source_lists_to_expanded_training(
    mock_submitter, tmp_path
):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "train_source_list": "old-train.list",
                    "validation_source_list": "old-val.list",
                },
                "stages": [
                    {
                        "name": "train",
                        "config": "train.yaml",
                        "stage": "train",
                        "run_dir": str(tmp_path / "train"),
                        "source_list": "${train_source_list}",
                        "val_source_list": "${validation_source_list}",
                    }
                ],
            }
        )
    )
    with patch.object(mock_submitter, "submit_job", return_value=["123"]) as job:
        result = mock_submitter.submit_pipeline(
            str(pipeline),
            source_list="/sdf/train.list",
            val_source_list="/sdf/val.list",
        )
    assert result == {"train": ["123"]}
    assert job.call_args.kwargs["files"] == ["/sdf/train.list"]
    assert job.call_args.kwargs["source_type"] == "source_list"
    assert job.call_args.kwargs["validation_files"] == ["/sdf/val.list"]
    assert job.call_args.kwargs["validation_source_type"] == "source_list"

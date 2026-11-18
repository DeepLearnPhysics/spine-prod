"""Submission integration tests for batch behavior."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.client import PBSClient, SlurmClient
from src.runs import RunManager
from src.submitter import Submitter


class TestSubmitterHelpers:
    """Tests for scheduler, path, and template selection helpers."""

    def test_format_spine_runtime_options(self, mock_submitter):
        assert mock_submitter.spine_cli.format_runtime_options(
            world_size=4,
            batch_size=16,
            minibatch_size=2,
            num_workers=8,
            epochs=25.0,
            iterations=100,
        ) == (
            "--world-size 4 --batch-size 16 --minibatch-size 2 "
            "--num-workers 8 --epochs 25.0 --iterations 100"
        )
        assert mock_submitter.spine_cli.format_runtime_options() == ""

        with pytest.raises(ValueError, match="managed from the GPU allocation"):
            mock_submitter.spine_cli.format_set_overrides(["base.world_size=4"])

    def test_format_spine_entry_fraction_ranges(self, mock_submitter):
        """Dataset partitions use SPINE's explicit half-open CLI options."""
        assert mock_submitter.spine_cli.format_entry_fraction_ranges(
            entry_fraction_range=(0.5, 1.0),
            val_entry_fraction_range=(0.0, 0.5),
        ) == ("--entry-fraction-range 0.5 1.0 " "--val-entry-fraction-range 0.0 0.5")

        with pytest.raises(ValueError, match="0 <= START < STOP <= 1"):
            mock_submitter.spine_cli.format_entry_fraction_ranges(
                entry_fraction_range=(0.5, 0.5)
            )

    def test_format_spine_num_entries(self, mock_submitter):
        """Exact dataset limits use SPINE's first-class CLI options."""
        assert mock_submitter.spine_cli.format_num_entries(10000, 1000) == (
            "--num-entries 10000 --val-num-entries 1000"
        )
        assert mock_submitter.spine_cli.format_num_entries() == ""
        for value in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="positive integer"):
                mock_submitter.spine_cli.format_num_entries(value)

    def test_format_spine_entry_filters(self, mock_submitter):
        """Eligibility manifests are quoted independently for train and val."""
        assert mock_submitter.spine_cli.format_entry_filters(
            "/filters/train accepted.yaml", "/filters/validation.yaml"
        ) == (
            "--entry-filter '/filters/train accepted.yaml' "
            "--val-entry-filter /filters/validation.yaml"
        )
        assert mock_submitter.spine_cli.format_entry_filters() == ""
        with pytest.raises(ValueError, match="non-empty path"):
            mock_submitter.spine_cli.format_entry_filters("")

    def test_format_spine_named_sources_and_module_weights(self, mock_submitter):
        sources = {
            "larcv": {"source_list": ["raw files.txt"]},
            "hdf5": {"source": "/cache/*.h5"},
        }
        assert mock_submitter.spine_cli.format_named_sources(sources) == (
            "--source 'hdf5=/cache/*.h5' " "--source-list 'larcv=raw files.txt'"
        )
        assert mock_submitter.spine_cli.format_named_sources(
            sources, validation=True
        ) == (
            "--val-source 'hdf5=/cache/*.h5' " "--val-source-list 'larcv=raw files.txt'"
        )
        assert (
            mock_submitter.spine_cli.format_module_weights(
                {"uresnet_ppn": "/weights/best.ckpt"}
            )
            == "--module-weight uresnet_ppn=/weights/best.ckpt"
        )
        assert (
            mock_submitter.spine_cli.format_weight_path("/weights/full chain.ckpt")
            == "--weight-path '/weights/full chain.ckpt'"
        )
        assert (
            mock_submitter.spine_cli.format_export_weights("/weights/full chain.ckpt")
            == "--export-weights '/weights/full chain.ckpt'"
        )

        with pytest.raises(ValueError, match="exactly one"):
            mock_submitter.spine_cli.format_named_sources(
                {"larcv": {"source": "raw.root", "source_list": "raw.txt"}}
            )

        assert mock_submitter.spine_cli.format_named_sources(None) == ""
        assert mock_submitter.spine_cli.format_module_weights(None) == ""
        assert mock_submitter.spine_cli.format_weight_path(None) == ""
        assert mock_submitter.spine_cli.format_export_weights(None) == ""
        with pytest.raises(TypeError, match="must be a mapping"):
            mock_submitter.spine_cli.format_named_sources({"larcv": "raw.root"})
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_submitter.spine_cli.format_named_sources({"larcv": {"source": []}})
        with pytest.raises(ValueError, match="accepts exactly one"):
            mock_submitter.spine_cli.format_named_sources(
                {"larcv": {"source_list": ["one.txt", "two.txt"]}}
            )
        with pytest.raises(ValueError, match="require a module and path"):
            mock_submitter.spine_cli.format_module_weights({"uresnet_ppn": ""})

    def test_align_world_size_with_scheduler_gpus(self, mock_submitter):
        assert (
            mock_submitter.spine_cli.align_world_size({"site": "s3df", "gpus": 4}, None)
            == 4
        )
        assert (
            mock_submitter.spine_cli.align_world_size(
                {"site": "nersc", "gpus_per_node": 4}, None
            )
            == 4
        )
        assert (
            mock_submitter.spine_cli.align_world_size({"site": "s3df", "gpus": 0}, None)
            == 0
        )
        assert mock_submitter.spine_cli.align_world_size({"site": "custom"}, 2) == 2

        with pytest.raises(ValueError, match="conflicts with the scheduler"):
            mock_submitter.spine_cli.align_world_size(
                {"site": "s3df", "gpus": 2}, requested_world_size=4
            )
        with pytest.raises(ValueError, match="Multi-node"):
            mock_submitter.spine_cli.align_world_size(
                {"site": "anl", "gpus_per_node": 4, "nodes": 2}, None
            )

    def test_classify_config_request_covers_shorthand_and_absolute_paths(
        self, mock_submitter, workspace_root
    ):
        assert mock_submitter.batch.classify_config_request("infer/icarus") == (
            True,
            "latest",
        )
        assert mock_submitter.batch.classify_config_request(
            str(workspace_root / "config" / "infer" / "icarus")
        ) == (True, "latest")
        assert mock_submitter.batch.classify_config_request("infer/icarus/latest") == (
            True,
            "latest",
        )
        assert mock_submitter.batch.classify_config_request("convert/icarus") == (
            True,
            "latest",
        )
        assert mock_submitter.batch.classify_config_request(
            "convert/dune-vd-10kt-1x8x6"
        ) == (True, "latest")
        assert mock_submitter.batch.classify_config_request("convert/protodune-hd") == (
            True,
            "latest",
        )
        assert mock_submitter.batch.classify_config_request(
            "convert/icarus/latest"
        ) == (True, "latest")
        assert mock_submitter.batch.classify_config_request(
            str(workspace_root / "config" / "convert" / "icarus")
        ) == (True, "latest")
        assert mock_submitter.batch.classify_config_request(
            "config/convert/icarus"
        ) == (True, "latest")
        with pytest.warns(FutureWarning, match="dune10kt-1x2x6.*deprecated"):
            assert mock_submitter.batch.classify_config_request(
                "infer/dune10kt-1x2x6"
            ) == (True, "latest")
        assert mock_submitter.batch.classify_config_request("custom.yaml") == (
            False,
            "custom",
        )
        mock_submitter.config_mgr.profiles["detectors"]["without_configs"] = {}
        assert mock_submitter.batch.classify_config_request("still_custom.yaml") == (
            False,
            "still_custom",
        )

    def test_resolve_setup_path_requires_configure_script(
        self, mock_submitter, tmp_path
    ):
        assert mock_submitter.runtime.resolve_setup_path(None, "--tool") == (None, None)
        with pytest.raises(RuntimeError, match="configure.sh"):
            mock_submitter.runtime.resolve_setup_path(str(tmp_path), "--tool")

    def test_batch_client_and_template_selection(self, mock_submitter):
        assert isinstance(
            mock_submitter.batch.get_batch_client({"site": "s3df"}), SlurmClient
        )
        assert isinstance(
            mock_submitter.batch.get_batch_client({"site": "polaris"}), PBSClient
        )
        assert (
            mock_submitter.batch.get_template_name({"template": "custom.j2"})
            == "custom.j2"
        )
        assert (
            mock_submitter.batch.get_template_name({"site": "nersc"})
            == "job_template_nersc.sbatch"
        )
        assert (
            mock_submitter.batch.get_template_name({"site": "s3df"})
            == "job_template_s3df.sbatch"
        )

        with pytest.raises(ValueError, match="Unknown scheduler"):
            mock_submitter.batch.get_batch_client({"scheduler": "other"})
        with pytest.raises(ValueError, match="Unknown site"):
            mock_submitter.batch.get_template_name({"site": "other"})

    def test_output_and_bind_path_helpers(self, mock_submitter, tmp_path):
        output = tmp_path / "result.h5"
        assert (
            mock_submitter.spine_cli.format_output_args(str(output), "unused", "unused")
            == f"--output {output}"
        )
        assert (
            mock_submitter.spine_cli.format_output_args(
                None, "/tmp/job output", "reco output"
            )
            == "--output-dir '/tmp/job output' --output-suffix 'reco output'"
        )
        assert (
            mock_submitter.runtime.merge_bind_paths(
                " /data, /scratch, /data ", ["/extra", ""]
            )
            == "/data,/scratch,/extra"
        )
        assert mock_submitter.runtime.merge_bind_paths(None) is None
        assert mock_submitter.runtime.default_bind_paths_for_site("nersc") is None
        assert mock_submitter.batch.resolve_files_per_task(10, files_per_task=3) == 3

    def test_resolve_spine_command_accepts_explicit_binary(
        self, mock_submitter, tmp_path
    ):
        checkout = tmp_path / "checkout"
        binary = checkout / "bin" / "spine"
        binary.parent.mkdir(parents=True)
        binary.touch()

        command, bind_root = mock_submitter.runtime.resolve_spine_command(str(binary))

        assert command == str(binary)
        assert bind_root == str(checkout)

    def test_resolve_spine_report_command_uses_checkout_module(
        self, mock_submitter, tmp_path
    ):
        checkout = tmp_path / "checkout"
        report_module = checkout / "src" / "spine" / "bin" / "report.py"
        report_module.parent.mkdir(parents=True)
        report_module.touch()

        command, bind_root = mock_submitter.runtime.resolve_spine_report_command(
            str(checkout)
        )

        assert command.startswith(f"env PYTHONPATH={checkout / 'src'}:")
        assert command.endswith("python3 -m spine.bin.report")
        assert bind_root == str(checkout)

    def test_resolve_spine_filter_command_uses_checkout_module(
        self, mock_submitter, tmp_path
    ):
        checkout = tmp_path / "checkout"
        filter_module = checkout / "src" / "spine" / "bin" / "filter.py"
        filter_module.parent.mkdir(parents=True)
        filter_module.touch()

        command, bind_root = mock_submitter.runtime.resolve_spine_filter_command(
            str(checkout)
        )

        assert command.startswith(f"env PYTHONPATH={checkout / 'src'}:")
        assert command.endswith("python3 -m spine.bin.filter")
        assert bind_root == str(checkout)

        for configured in (checkout / "bin" / "spine", checkout / "custom"):
            command, bind_root = mock_submitter.runtime.resolve_spine_filter_command(
                str(configured)
            )
            assert command.endswith("python3 -m spine.bin.filter")
            assert bind_root == str(checkout)

        with pytest.raises(RuntimeError, match="does not provide spine.bin.filter"):
            mock_submitter.runtime.resolve_spine_filter_command(str(tmp_path / "bad"))

        with patch(
            "src.submission.runtime.shutil.which", return_value="/usr/bin/spine-filter"
        ):
            assert mock_submitter.runtime.resolve_spine_filter_command() == (
                None,
                None,
            )
        with patch("src.submission.runtime.shutil.which", return_value=None):
            assert mock_submitter.runtime.resolve_spine_filter_command() == (None, None)

    def test_resolve_spine_cache_command_uses_checkout_module(
        self, mock_submitter, tmp_path
    ):
        """Cache maintenance must follow the selected SPINE checkout."""
        checkout = tmp_path / "checkout"
        cache_module = checkout / "src" / "spine" / "bin" / "cache.py"
        cache_module.parent.mkdir(parents=True)
        cache_module.touch()

        command, bind_root = mock_submitter.runtime.resolve_spine_cache_command(
            str(checkout)
        )
        assert command.startswith(f"env PYTHONPATH={checkout / 'src'}:")
        assert command.endswith("python3 -m spine.bin.cache")
        assert bind_root == str(checkout)

        for configured in (checkout / "bin" / "spine", checkout / "custom"):
            command, bind_root = mock_submitter.runtime.resolve_spine_cache_command(
                str(configured)
            )
            assert command.endswith("python3 -m spine.bin.cache")
            assert bind_root == str(checkout)

        with pytest.raises(RuntimeError, match="does not provide spine.bin.cache"):
            mock_submitter.runtime.resolve_spine_cache_command(str(tmp_path / "bad"))

        with patch(
            "src.submission.runtime.shutil.which", return_value="/usr/bin/spine-cache"
        ):
            assert mock_submitter.runtime.resolve_spine_cache_command() == (
                None,
                None,
            )
        with patch("src.submission.runtime.shutil.which", return_value=None):
            assert mock_submitter.runtime.resolve_spine_cache_command() == (None, None)

    @pytest.mark.parametrize("bind_paths", [None, "/existing"])
    def test_submit_filter_scan_builds_persistent_scheduler_job(
        self, mock_submitter, tmp_path, bind_paths
    ):
        """A scan job records its exact reusable counter-cache invocation."""
        run_dir = tmp_path / "filter-scan"
        cache_dir = tmp_path / "counts"
        config = "filter/protodune-sp/space_points_260210.yaml"
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(
                mock_submitter.runtime,
                "resolve_spine_filter_command",
                return_value=("spine-filter", "/checkout"),
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="41"),
        ):
            job_ids = mock_submitter.submit_filter(
                config=config,
                operation="scan",
                sources=["/data/input one.root", "/data/input-two.root"],
                cache_dir=str(cache_dir),
                run_dir=str(run_dir),
                workers=16,
                bind_paths=bind_paths,
                force=True,
                dependency="afterok:40",
            )

        assert job_ids == ["41"]
        attempt = (run_dir / "latest").resolve()
        script = (attempt / "submit.sbatch").read_text(encoding="utf-8")
        assert "spine-filter scan" in script
        expected_bind = bind_paths or "/sdf/"
        assert f'BIND_PATHS="{expected_bind},/checkout"' in script
        assert "--source '/data/input one.root' /data/input-two.root" in script
        assert "--workers 16 --force" in script
        assert "#SBATCH --dependency=afterok:40" in script
        metadata = json.loads((attempt / "job_metadata.json").read_text())
        assert metadata["kind"] == "filter"
        assert metadata["operation"] == "scan"
        assert metadata["workers"] == 16
        assert metadata["force"] is True

    def test_submit_filter_build_publishes_manifest_paths(
        self, mock_submitter, tmp_path
    ):
        """A build job owns both final filter artifacts and supports dry runs."""
        run_dir = tmp_path / "filter-build"
        output = tmp_path / "artifacts" / "accepted.yaml"
        source_output = tmp_path / "artifacts" / "accepted.txt"
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(
                mock_submitter.runtime,
                "resolve_spine_filter_command",
                return_value=(None, None),
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value=None),
        ):
            job_ids = mock_submitter.submit_filter(
                config="filter/protodune-sp/space_points_260210.yaml",
                operation="build",
                source_list="/data/files.txt",
                cache_dir=str(tmp_path / "counts"),
                output=str(output),
                output_source_list=str(source_output),
                run_dir=str(run_dir),
                dry_run=True,
            )

        assert job_ids == []
        script = ((run_dir / "latest").resolve() / "submit.sbatch").read_text()
        assert "spine-filter build" in script
        assert "--source-list /data/files.txt" in script
        assert f"--output {output}" in script
        assert f"--output-source-list {source_output}" in script
        assert output.parent.is_dir()

    def test_submit_filter_uses_detector_account_fallback(
        self, mock_submitter, tmp_path
    ):
        """Filter jobs inherit the detector account when a profile omits it."""
        detector_account = mock_submitter.profiles["detectors"]["protodune-sp"][
            "account"
        ]
        profile = {
            "site": "s3df",
            "scheduler": "slurm",
            "description": "Account fallback test",
        }
        with (
            patch.object(
                mock_submitter.config_mgr,
                "get_profile",
                return_value=profile,
            ),
            patch.object(
                mock_submitter.runtime,
                "resolve_spine_filter_command",
                return_value=("spine-filter", None),
            ),
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value=None),
        ):
            mock_submitter.submit_filter(
                config="filter/protodune-sp/space_points_260210.yaml",
                operation="scan",
                sources=["/data/input.root"],
                cache_dir=str(tmp_path / "counts"),
                run_dir=str(tmp_path / "run"),
                dry_run=True,
            )

        metadata = json.loads(
            ((tmp_path / "run" / "latest").resolve() / "job_metadata.json").read_text()
        )
        assert metadata["profile_config"]["account"] == detector_account

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"operation": "other"}, ValueError, "one of: scan, build"),
            ({"sources": None}, ValueError, "exactly one"),
            (
                {"sources": ["input.root"], "source_list": "files.txt"},
                ValueError,
                "exactly one",
            ),
            ({"cache_dir": None}, ValueError, "require cache_dir"),
            ({"workers": 0}, ValueError, "at least one"),
            ({"force": "yes"}, TypeError, "must be a boolean"),
            ({"output": "filter.yaml"}, ValueError, "cannot define build outputs"),
            (
                {"operation": "build"},
                ValueError,
                "requires output and output_source_list",
            ),
            (
                {
                    "operation": "build",
                    "output": "filter.yaml",
                    "output_source_list": "files.txt",
                    "workers": 2,
                },
                ValueError,
                "cannot define workers or force",
            ),
        ],
    )
    def test_filter_runner_rejects_invalid_operations(
        self, mock_submitter, kwargs, error, message
    ):
        """Standalone validation rejects ambiguous or cross-operation fields."""
        values = {
            "operation": "scan",
            "sources": ["input.root"],
            "source_list": None,
            "cache_dir": "/tmp/counts",
            "output": None,
            "output_source_list": None,
            "workers": None,
            "force": False,
        }
        values.update(kwargs)
        with pytest.raises(error, match=message):
            mock_submitter.filter._validate_operation(**values)

    def test_filter_runner_rejects_existing_run_and_pbs_exclusion(
        self, mock_submitter, tmp_path
    ):
        """Persistent filter attempts and scheduler controls fail explicitly."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "existing").touch()
        base = {
            "config": "filter/protodune-sp/space_points_260210.yaml",
            "operation": "scan",
            "sources": ["input.root"],
            "cache_dir": str(tmp_path / "counts"),
            "run_dir": str(run_dir),
        }
        with pytest.raises(ValueError, match="run directory is not empty"):
            mock_submitter.submit_filter(**base)

        base["run_dir"] = str(tmp_path / "pbs-run")
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
            mock_submitter.submit_filter(**base, exclude="node01")

    def test_submit_report_materializes_provenance_and_scheduler_job(
        self, mock_submitter, tmp_path
    ):
        source_config = tmp_path / "report.yaml"
        source_config.write_text(
            yaml.safe_dump(
                {
                    "include": "test/common/full_chain/report_v1.yaml",
                    "metadata": {"dataset": None, "checkpoint": None},
                    "metrics": {
                        "segmentation": {
                            "name": "segment_confusion",
                            "source": "**/*segment.csv",
                        }
                    },
                }
            )
        )
        run_dir = tmp_path / "report-run"
        input_dir = tmp_path / "raw" / "latest"
        output_dir = run_dir / "artifacts"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(
                mock_submitter.runtime,
                "resolve_spine_report_command",
                return_value=(None, None),
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="42"),
        ):
            job_ids = mock_submitter.submit_report(
                config=str(source_config),
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                run_dir=str(run_dir),
                checkpoint="/weights/full.ckpt",
                dataset="validation.root",
                dataset_selection={"entry_fraction_range": [0.5, 1.0]},
            )

        assert job_ids == ["42"]
        attempt = (run_dir / "latest").resolve()
        resolved = yaml.safe_load((attempt / "report.yaml").read_text())
        assert resolved["metadata"] == {
            "dataset": "validation.root",
            "checkpoint": "/weights/full.ckpt",
            "dataset_selection": {"entry_fraction_range": [0.5, 1.0]},
        }
        assert resolved["include"] == "test/common/full_chain/report_v1.yaml"
        script = (attempt / "submit.sbatch").read_text()
        assert "spine-report --config" in script
        assert 'scripts/check_spine_version.py\\" spine' in script
        assert 'scripts/check_spine_version.py\\" None' not in script
        assert f"--input-dir {input_dir}" in script
        assert f"--output-dir {output_dir}" in script

    def test_report_config_materialization_supports_legacy_pyyaml(self, tmp_path):
        """Generated report recipes must not require PyYAML's sort_keys option."""
        from src.submission.report import ReportRunner

        source = tmp_path / "source.yaml"
        source.write_text("metadata: {}\nmetrics: {}\n", encoding="utf-8")
        attempt = tmp_path / "attempt"
        attempt.mkdir()
        safe_dump = yaml.safe_dump

        def legacy_safe_dump(document, **kwargs):
            if "sort_keys" in kwargs:
                raise TypeError("dump_all() got an unexpected keyword argument")
            return safe_dump(document, **kwargs)

        with patch(
            "src.submission.report.yaml.safe_dump", side_effect=legacy_safe_dump
        ):
            destination = ReportRunner._materialize_config(
                source,
                attempt,
                checkpoint="weights.ckpt",
                dataset="test.root",
            )

        resolved = yaml.safe_load(destination.read_text(encoding="utf-8"))
        assert resolved["metadata"] == {
            "checkpoint": "weights.ckpt",
            "dataset": "test.root",
        }

    def test_container_helpers_cover_fallbacks(self, mock_submitter):
        with patch.dict(
            os.environ,
            {"SPINE_PROD_CONFIGURED": "1"},
            clear=True,
        ):
            assert (
                mock_submitter.runtime.container_version()
                == mock_submitter.runtime.default_container_version()
            )

        def find_apptainer(command):
            return "/usr/bin/apptainer" if command == "apptainer" else None

        with patch.dict(os.environ, {}, clear=True), patch(
            "src.submission.runtime.shutil.which", side_effect=find_apptainer
        ):
            assert (
                mock_submitter.runtime.sif_runtime_executable() == "/usr/bin/apptainer"
            )

    def test_init_uses_central_jobs_directory_and_warns_without_environment(
        self, workspace_root, tmp_path, capsys
    ):
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "profiles.yaml").write_text(
            (workspace_root / "templates" / "profiles.yaml").read_text()
        )

        with patch.dict(os.environ, {}, clear=True):
            submitter = Submitter(basedir=tmp_path, central_dir=True)

        assert submitter.jobs_dir == tmp_path / "runs"
        assert not submitter.jobs_dir.exists()
        assert "SPINE_PROD_BASEDIR not set" in capsys.readouterr().out

    def test_init_does_not_create_default_jobs_directory(
        self, workspace_root, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        submitter = Submitter(basedir=workspace_root)

        assert submitter.jobs_dir == tmp_path / "runs"
        assert not submitter.jobs_dir.exists()

    def test_runtime_helpers_report_invalid_configuration(
        self, mock_submitter, tmp_path
    ):
        with (
            patch.dict(
                os.environ,
                {"SPINE_CONTAINER_RUNTIME_BIN": "missing-runtime"},
                clear=False,
            ),
            patch("src.submission.runtime.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="no executable was found"),
        ):
            mock_submitter.runtime.sif_runtime_executable()

        with (
            patch.dict(
                os.environ,
                {"SPINE_CONTAINER_PATH": str(tmp_path / "missing.sif")},
                clear=False,
            ),
            patch("src.submission.runtime.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="no usable runtime"),
        ):
            mock_submitter.runtime.build_interactive_container_command("spine", False)

    def test_sif_container_command_adds_cvmfs_bind(self, mock_submitter, tmp_path):
        container = tmp_path / "spine.sif"
        container.touch()
        with (
            patch.dict(
                os.environ,
                {"SPINE_CONTAINER_PATH": str(container)},
                clear=False,
            ),
            patch.object(
                mock_submitter.runtime,
                "sif_runtime_executable",
                return_value="/usr/bin/apptainer",
            ),
        ):
            command = mock_submitter.runtime.build_interactive_container_command(
                "spine -c config.yaml", True
            )

        assert "/cvmfs" in command
        assert "/usr/bin/apptainer exec" in command

    def test_container_command_falls_back_from_sif_to_docker_with_environment(
        self, mock_submitter, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        original_exists = Path.exists

        def report_cvmfs_exists(path):
            if path == Path("/cvmfs"):
                return True
            return original_exists(path)

        def find_docker(command):
            return "/usr/bin/docker" if command == "docker" else None

        with (
            patch.dict(
                os.environ,
                {
                    "SPINE_CONTAINER_PATH": str(tmp_path / "missing.sif"),
                    "ICARUS_DATA_DIR": "/data/icarus",
                    "SBND_DATA_DIR": "/data/sbnd",
                },
                clear=False,
            ),
            patch("src.submission.runtime.Path.exists", report_cvmfs_exists),
            patch("src.submission.runtime.shutil.which", side_effect=find_docker),
        ):
            command = mock_submitter.runtime.build_interactive_container_command(
                "spine -c config.yaml", True
            )

        assert f"{mock_submitter.basedir}:{mock_submitter.basedir}" in command
        assert "-v /cvmfs:/cvmfs:ro" in command
        assert "-e ICARUS_DATA_DIR=/data/icarus" in command
        assert "-e SBND_DATA_DIR=/data/sbnd" in command

    def test_existing_sif_falls_back_to_docker_without_sif_runtime(
        self, mock_submitter, tmp_path
    ):
        container = tmp_path / "spine.sif"
        container.touch()

        def find_docker(command):
            return "/usr/bin/docker" if command == "docker" else None

        with (
            patch.dict(
                os.environ,
                {"SPINE_CONTAINER_PATH": str(container)},
                clear=False,
            ),
            patch.object(
                mock_submitter.runtime, "sif_runtime_executable", return_value=None
            ),
            patch("src.submission.runtime.shutil.which", side_effect=find_docker),
        ):
            command = mock_submitter.runtime.build_interactive_container_command(
                "spine -c config.yaml", False
            )

        assert "/usr/bin/docker run" in command
        assert "apptainer" not in command


class TestJobDirectory:
    """Tests for job directory creation."""

    def test_create_job_dir(self, mock_submitter):
        """Test creating timestamped job directory."""
        job_name = "test_job"
        job_dir = mock_submitter.batch_client.create_job_dir(job_name)

        assert job_dir.exists()
        assert job_dir.is_dir()
        assert job_name in str(job_dir)
        # Should have timestamp in name
        assert any(char.isdigit() for char in job_dir.name)

    def test_job_dir_under_jobs(self, mock_submitter):
        """Test that job dir is created under jobs/."""
        job_dir = mock_submitter.batch_client.create_job_dir("test")
        assert mock_submitter.jobs_dir in job_dir.parents


class TestJobMetadata:
    """Tests for job metadata handling."""


class TestBatchSpineOverride:
    """Tests for explicit SPINE runtime overrides in batch mode."""

    def test_submit_job_exercises_chunked_latest_submission_flow(
        self, mock_submitter, tmp_path, monkeypatch, capsys
    ):
        inputs = []
        for index in range(5):
            path = tmp_path / f"input_{index}.root"
            path.touch()
            inputs.append(str(path))
        larcv_root = tmp_path / "larcv"
        larcv_root.mkdir()
        (larcv_root / "configure.sh").touch()
        latest = tmp_path / "latest.yaml"
        composite = tmp_path / "composite.yaml"
        latest.touch()
        composite.touch()
        output_dir = tmp_path / "explicit_output"

        profile_config = mock_submitter.config_mgr.get_profile("auto", "icarus")
        profile_config.pop("bind_paths", None)
        profile_config.pop("account", None)
        profile_config["site"] = "s3df"
        profile_config["scheduler"] = "slurm"
        monkeypatch.setitem(mock_submitter.profiles["defaults"], "max_array_size", 2)

        with (
            patch.object(
                mock_submitter.config_mgr,
                "create_latest_config",
                return_value=str(latest),
            ) as create_latest,
            patch.object(
                mock_submitter.config_mgr,
                "create_composite_config",
                return_value=str(composite),
            ) as create_composite,
            patch.object(
                mock_submitter.config_mgr,
                "get_profile",
                return_value=profile_config,
            ),
            patch.object(mock_submitter, "preload_downloads") as preload,
            patch.object(SlurmClient, "submit", side_effect=["10", "20", "30"]),
            patch("src.submission.runtime.shutil.which", return_value=None),
        ):
            job_ids = mock_submitter.submit_job(
                config="infer/icarus",
                files=inputs,
                output=str(output_dir),
                ntasks=1,
                files_per_task=1,
                dependency="afterok:5",
                larcv_path=str(larcv_root),
                flashmatch=True,
                apply_mods=["data"],
                preload=True,
            )

        assert job_ids == ["10", "20", "30"]
        assert output_dir.is_dir()
        job_dir = create_latest.call_args.args[1]
        create_composite.assert_called_once_with(
            str(latest),
            ["data"],
            job_dir,
            detector="icarus",
            family="infer",
        )
        preload.assert_called_once_with(str(composite))
        scripts = sorted(job_dir.glob("attempts/*/submit_*.sbatch"))
        assert len(scripts) == 3
        assert "#SBATCH --array=1-2%1" in scripts[0].read_text()
        assert "#SBATCH --dependency=afterok:5" in scripts[0].read_text()
        assert "#SBATCH --kill-on-invalid-dep=yes" in scripts[0].read_text()
        assert "#SBATCH --dependency=afterok:10" in scripts[1].read_text()
        assert "tasks/002_1/inputs.txt" in scripts[2].read_text()
        assert "tasks/002_*/inputs.txt" not in scripts[2].read_text()
        attempt = scripts[0].parent
        assert not list(attempt.glob("tasks/*/logs"))
        assert not list(attempt.glob("tasks/*/output"))
        assert not (attempt / "scheduler").exists()
        assert profile_config["bind_paths"].startswith("/sdf/")
        assert profile_config["account"]
        assert "--flashmatch is deprecated" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"joint_file_mode": "other"}, "joint_file_mode must be either"),
            ({"stage": "unknown"}, "stage must be one of"),
            ({"stage": "train"}, "--run-dir is required"),
            ({"resume": True}, "valid only for training"),
            (
                {"warm_start_path": "/weights/full.ckpt"},
                "must be provided together",
            ),
            (
                {"warm_start_modules": {"model": "full.model"}},
                "must be provided together",
            ),
            (
                {
                    "warm_start_path": "/weights/full.ckpt",
                    "warm_start_modules": {"model": "full.model"},
                },
                "valid only for training",
            ),
            (
                {
                    "stage": "train",
                    "run_dir": "/tmp/train",
                    "warm_start_path": "/weights/full.ckpt",
                    "warm_start_modules": {"model": "full.model"},
                    "weight_path": "/weights/other.ckpt",
                },
                "cannot be combined with weight_path",
            ),
            ({"validation_name": "data"}, "valid only for validation"),
            ({"val_entry_filter": "/filters/val.yaml"}, "valid only for training"),
            ({"val_num_files": 1}, "valid only for training"),
            ({"num_files": 1}, "requires --source/--source-list"),
            (
                {"stage": "train", "run_dir": "/tmp/train", "val_num_files": 1},
                "requires --val-source/--val-source-list",
            ),
            (
                {
                    "stage": "train",
                    "run_dir": "/tmp/train",
                    "val_num_entries": 10,
                    "val_entry_fraction_range": (0.0, 0.5),
                },
                "cannot be combined with --val-entry-fraction-range",
            ),
            (
                {"joint_file_mode": "paired"},
                "require named primary and secondary",
            ),
            (
                {
                    "stage": "train",
                    "run_dir": "/tmp/train",
                    "joint_file_mode": "paired",
                    "named_sources": {
                        "primary": {"source": "primary.root"},
                        "secondary": {"source": "secondary.root"},
                    },
                },
                "valid only for inference jobs",
            ),
            (
                {"validation_named_sources": {"larcv": {"source": "val.root"}}},
                "Named validation sources are valid only for training",
            ),
            (
                {
                    "files": ["train.root"],
                    "named_sources": {"larcv": {"source": "train.root"}},
                },
                "Flat and named training sources cannot be combined",
            ),
            (
                {
                    "stage": "train",
                    "run_dir": "/tmp/train",
                    "validation_files": ["val.root"],
                    "validation_named_sources": {"larcv": {"source": "val.root"}},
                },
                "Flat and named validation sources cannot be combined",
            ),
        ],
    )
    def test_submit_job_validates_lifecycle_options(
        self, mock_submitter, kwargs, message
    ):
        """Test lifecycle-only options are rejected outside their stage."""
        with pytest.raises(ValueError, match=message):
            mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml", **kwargs
            )

    @pytest.mark.parametrize(
        ("profile", "kwargs", "message"),
        [
            (
                "s3df_ampere",
                {"gpus_per_node": 2},
                "--gpus-per-node is not valid",
            ),
            ("nersc_gpu", {"gpus": 2}, "--gpus is not valid"),
        ],
    )
    def test_submit_job_rejects_site_incompatible_gpu_options(
        self, mock_submitter, profile, kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            mock_submitter.submit_job(
                config="infer/generic/full_chain_240718.yaml",
                profile=profile,
                **kwargs,
            )

    def test_submit_job_rejects_node_exclusions_for_pbs(self, mock_submitter):
        """PBS has no portable equivalent of Slurm's node exclusion."""
        with pytest.raises(ValueError, match="only supported by Slurm"):
            mock_submitter.submit_job(
                config="infer/generic/full_chain_240718.yaml",
                profile="anl_polaris_debug",
                exclude="x3001c0s1b0n0",
            )

    def test_submit_job_forwards_explicit_training_sources(
        self, mock_submitter, tmp_path
    ):
        """Training sources are normalized into persistent submission manifests."""
        train_source = tmp_path / "train.root"
        validation_source = tmp_path / "validation.root"
        train_source.touch()
        validation_source.touch()
        run_dir = tmp_path / "run"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train"),
        ):
            assert mock_submitter.submit_job(
                config="train/generic/uresnet/train_240718.yaml",
                files=[str(train_source)],
                validation_files=[str(validation_source)],
                num_entries=10000,
                val_num_entries=1000,
                entry_filter="/filters/train.yaml",
                val_entry_filter="/filters/validation.yaml",
                stage="train",
                run_dir=str(run_dir),
            ) == ["train"]

        submission = run_dir / "latest"
        train_manifest = submission / "inputs.txt"
        validation_manifest = submission / "validation_inputs.txt"
        assert train_manifest.read_text().splitlines() == [str(train_source)]
        assert validation_manifest.read_text().splitlines() == [str(validation_source)]

        script = (submission / "submit.sbatch").read_text(encoding="utf-8")
        assert (
            str(
                mock_submitter.basedir
                / "config"
                / "train"
                / "generic"
                / "uresnet"
                / "train_240718.yaml"
            )
            in script
        )
        assert f"--source-list {train_manifest.resolve()}" in script
        assert f"--val-source-list {validation_manifest.resolve()}" in script
        assert "--num-entries 10000" in script
        assert "--val-num-entries 1000" in script
        assert "--entry-filter /filters/train.yaml" in script
        assert "--val-entry-filter /filters/validation.yaml" in script
        marker = submission.resolve() / "graceful_stop"
        assert f"--graceful-stop-file {marker}" in script
        assert f'SPINE_PROD_GRACEFUL_STOP_FILE="{marker}"' in script
        assert not marker.exists()
        assert "#SBATCH --array=" not in script
        assert not (run_dir / "tasks").exists()

        metadata = json.loads((submission / "job_metadata.json").read_text())
        assert metadata["source_manifest"] == str(train_manifest.resolve())
        assert metadata["validation_source_manifest"] == str(
            validation_manifest.resolve()
        )
        assert metadata["num_entries"] == 10000
        assert metadata["val_num_entries"] == 1000
        assert metadata["entry_filter"] == "/filters/train.yaml"
        assert metadata["val_entry_filter"] == "/filters/validation.yaml"
        assert metadata["graceful_stop_file"] == str(marker)

    def test_submit_job_limits_training_files_before_writing_manifests(
        self, mock_submitter, tmp_path
    ):
        train_files = [tmp_path / f"train-{index}.root" for index in range(3)]
        validation_files = [tmp_path / f"validation-{index}.root" for index in range(2)]
        for path in train_files + validation_files:
            path.touch()
        train_list = tmp_path / "train.txt"
        validation_list = tmp_path / "validation.txt"
        train_list.write_text("\n".join(map(str, train_files)))
        validation_list.write_text("\n".join(map(str, validation_files)))
        run_dir = tmp_path / "run"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train"),
        ):
            mock_submitter.submit_job(
                config="train/generic/uresnet/train_240718.yaml",
                files=[str(train_list)],
                source_type="source_list",
                validation_files=[str(validation_list)],
                validation_source_type="source_list",
                num_files=2,
                val_num_files=1,
                stage="train",
                run_dir=str(run_dir),
            )

        submission = run_dir / "latest"
        assert (submission / "inputs.txt").read_text().splitlines() == list(
            map(str, train_files[:2])
        )
        assert (submission / "validation_inputs.txt").read_text().splitlines() == [
            str(validation_files[0])
        ]
        metadata = json.loads((submission / "job_metadata.json").read_text())
        assert metadata["num_file_limit"] == 2
        assert metadata["val_num_file_limit"] == 1
        assert metadata["num_files"] == 2
        assert metadata["validation_num_files"] == 1

    def test_submit_job_preserves_future_pipeline_training_sources(
        self, mock_submitter, tmp_path, capsys
    ):
        """Dependent jobs may reference exact files created by predecessors."""
        train_source = tmp_path / "cache" / "train.h5"
        validation_source = tmp_path / "cache" / "validation.h5"
        run_dir = tmp_path / "run"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train"),
        ):
            assert mock_submitter.submit_job(
                config="train/generic/uresnet/train_240718.yaml",
                files=[str(train_source)],
                validation_files=[str(validation_source)],
                stage="train",
                run_dir=str(run_dir),
                dependency="afterok:123",
                allow_missing_inputs=True,
            ) == ["train"]

        submission = run_dir / "latest"
        assert (submission / "inputs.txt").read_text().splitlines() == [
            str(train_source)
        ]
        assert (submission / "validation_inputs.txt").read_text().splitlines() == [
            str(validation_source)
        ]
        assert "Files: 1" in capsys.readouterr().out

    def test_submit_job_forwards_named_sources_and_module_weights(
        self, mock_submitter, tmp_path
    ):
        """Composite sources and module checkpoints use native SPINE flags."""
        run_dir = tmp_path / "run"
        named_sources = {
            "larcv": {"source": "/raw/train.root"},
            "hdf5": {"source": "/cache/train.h5"},
        }
        validation_sources = {
            "larcv": {"source": "/raw/test.root"},
            "hdf5": {"source": "/cache/test.h5"},
        }

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train"),
        ):
            assert mock_submitter.submit_job(
                config=(
                    "train/generic/graph_spice/" "train_from_uresnet_ppn_240805.yaml"
                ),
                named_sources=named_sources,
                validation_named_sources=validation_sources,
                module_weights={"graph_spice": "/weights/seed.ckpt"},
                stage="train",
                run_dir=str(run_dir),
            ) == ["train"]

        submission = run_dir / "latest"
        script = (submission / "submit.sbatch").read_text(encoding="utf-8")
        assert "--source larcv=/raw/train.root hdf5=/cache/train.h5" in script
        assert "--val-source larcv=/raw/test.root hdf5=/cache/test.h5" in script
        assert "--module-weight graph_spice=/weights/seed.ckpt" in script
        assert "--set io.loader.dataset" not in script

        metadata = json.loads((submission / "job_metadata.json").read_text())
        assert metadata["named_sources"] == named_sources
        assert metadata["validation_named_sources"] == validation_sources
        assert metadata["module_weights"] == {"graph_spice": "/weights/seed.ckpt"}

    def test_submit_job_applies_warm_start_and_explicit_override(
        self, mock_submitter, tmp_path
    ):
        """A fresh stage expands a full-chain seed into weights-only imports."""
        run_dir = tmp_path / "run"
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train"),
        ):
            mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml",
                module_weights={"ppn": "/weights/ppn-only.ckpt"},
                warm_start_path="/weights/full-chain.ckpt",
                warm_start_modules={
                    "uresnet": "uresnet_ppn.uresnet",
                    "ppn": "uresnet_ppn.ppn",
                },
                stage="train",
                run_dir=str(run_dir),
            )

        submission = run_dir / "latest"
        script = (submission / "submit.sbatch").read_text(encoding="utf-8")
        assert "--module-weight uresnet=/weights/full-chain.ckpt" in script
        assert "ppn=/weights/ppn-only.ckpt" in script
        assert "--set model.modules.uresnet.model_name=uresnet_ppn.uresnet" in script
        assert "model.modules.ppn.model_name" not in script
        assert "--resume" not in script
        metadata = json.loads((submission / "job_metadata.json").read_text())
        assert metadata["warm_start_applied"] is True
        assert metadata["warm_start_path"] == "/weights/full-chain.ckpt"

    def test_submit_job_resume_suppresses_warm_start(self, mock_submitter, tmp_path):
        """A stage checkpoint is authoritative when retrying a warm-started run."""
        run_dir = tmp_path / "run"
        config = "config/train/generic/uresnet/train_240718.yaml"
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="first"),
        ):
            mock_submitter.submit_job(
                config=config,
                warm_start_path="/weights/full-chain.ckpt",
                warm_start_modules={"uresnet": "uresnet_ppn.uresnet"},
                stage="train",
                run_dir=str(run_dir),
            )

        checkpoint = run_dir / "weights" / "snapshot-5.ckpt"
        checkpoint.touch()
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="retry"),
        ):
            mock_submitter.submit_job(
                config=config,
                warm_start_path="/weights/full-chain.ckpt",
                warm_start_modules={"uresnet": "uresnet_ppn.uresnet"},
                stage="train",
                run_dir=str(run_dir),
                retry=True,
            )

        submission = run_dir / "latest"
        script = (submission / "submit.sbatch").read_text(encoding="utf-8")
        assert f"--weight-path {checkpoint}" in script
        assert "--resume" in script
        assert "--module-weight uresnet=/weights/full-chain.ckpt" not in script
        assert "model.modules.uresnet.model_name" not in script
        metadata = json.loads((submission / "job_metadata.json").read_text())
        assert metadata["warm_start_applied"] is False

    def test_submit_job_forwards_output_for_named_sources(
        self, mock_submitter, tmp_path
    ):
        """Composite cache jobs can append into an explicit output directory."""
        output = tmp_path / "cache" / "train"
        named_sources = {
            "larcv": {"source": "/raw/train.root"},
            "hdf5": {"source": "/cache/train_cache.h5"},
        }

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="cache"),
        ):
            assert mock_submitter.submit_job(
                config="cache/generic/graph_spice/fragment_graphs_240805.yaml",
                named_sources=named_sources,
                output=str(output),
                output_suffix="cache",
                allow_missing_inputs=True,
            ) == ["cache"]

        script = next(
            mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch")
        ).read_text(encoding="utf-8")
        assert f"--output-dir {output}" in script
        assert "--output-suffix cache" in script
        assert output.is_dir()

    def test_submit_job_arrays_aligned_named_source_lists(
        self, mock_submitter, tmp_path
    ):
        """Mixed inference arrays create one aligned manifest per target."""
        run_dir = tmp_path / "run"
        paths = [tmp_path / f"input_{index}.root" for index in range(2)]
        for path in paths:
            path.touch()
        manifest = tmp_path / "primary.txt"
        manifest.write_text("\n".join(map(str, paths)), encoding="utf-8")
        cache = tmp_path / "cache.spine-cache"
        cache.mkdir()
        source_lists = {
            "primary": {"source_list": str(manifest)},
            "cache": {"source": str(cache)},
        }

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="cache"),
        ):
            assert mock_submitter.submit_job(
                config="cache/generic/graph_spice/fragment_graphs_240805.yaml",
                named_sources=source_lists,
                run_dir=str(run_dir),
                in_place=True,
                files_per_task=1,
                ntasks=2,
            ) == ["cache"]

        attempt = run_dir / "latest"
        script = (attempt / "submit.sbatch").read_text(encoding="utf-8")
        assert "#SBATCH --array=1-2" in script
        assert " -S $TASK_FILE_LIST" not in script
        assert f"--source cache={cache}" in script
        assert "--source-list primary=$TASK_DIR/primary.txt" in script
        for index in (1, 2):
            task_dir = attempt / "tasks" / f"000_{index}"
            assert (task_dir / "primary.txt").read_text().strip() == str(
                paths[index - 1]
            )
            assert not (task_dir / "cache.txt").exists()

        metadata = json.loads((attempt / "job_metadata.json").read_text())
        assert metadata["num_files"] == 2
        assert metadata["resolved_files_per_task"] == 1

    def test_submit_job_limits_named_sources_before_array_splitting(
        self, mock_submitter, tmp_path
    ):
        run_dir = tmp_path / "run"
        paths = [tmp_path / f"input_{index}.root" for index in range(3)]
        for path in paths:
            path.touch()
        manifest = tmp_path / "primary.txt"
        manifest.write_text("\n".join(map(str, paths)), encoding="utf-8")
        cache = tmp_path / "cache.spine-cache"
        cache.mkdir()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="cache"),
        ):
            mock_submitter.submit_job(
                config="cache/generic/graph_spice/fragment_graphs_240805.yaml",
                named_sources={
                    "primary": {"source_list": str(manifest)},
                    "cache": {"source": str(cache)},
                },
                run_dir=str(run_dir),
                in_place=True,
                num_files=2,
                files_per_task=1,
            )

        attempt = run_dir / "latest"
        script = (attempt / "submit.sbatch").read_text(encoding="utf-8")
        assert "#SBATCH --array=1-2" in script
        assert not (attempt / "tasks" / "000_3").exists()
        metadata = json.loads((attempt / "job_metadata.json").read_text())
        assert metadata["num_file_limit"] == 2
        assert metadata["num_files"] == 2

    def test_submit_job_splits_joint_primary_and_shares_secondary(
        self, mock_submitter, tmp_path
    ):
        run_dir = tmp_path / "run"
        primary = [tmp_path / f"primary-{index}.root" for index in range(3)]
        secondary = [tmp_path / f"secondary-{index}.root" for index in range(2)]
        for path in primary + secondary:
            path.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="joint"),
        ):
            mock_submitter.submit_job(
                config="infer/nd-lar/full_chain_260409.yaml",
                named_sources={
                    "primary": {"source": list(map(str, primary))},
                    "secondary": {"source": list(map(str, secondary))},
                },
                apply_mods=["joint:240819"],
                run_dir=str(run_dir),
                num_files=2,
                files_per_task=1,
            )

        attempt = run_dir / "latest"
        script = (attempt / "submit.sbatch").read_text(encoding="utf-8")
        assert "#SBATCH --array=1-2" in script
        assert "--output-dir $TASK_DIR/output" in script
        assert "--source-list primary=$TASK_DIR/primary.txt" in script
        assert "secondary=$TASK_DIR/secondary.txt" in script
        for index in (1, 2):
            task_dir = attempt / "tasks" / f"000_{index}"
            assert (task_dir / "primary.txt").read_text().splitlines() == [
                str(primary[index - 1])
            ]
            assert (task_dir / "secondary.txt").read_text().splitlines() == list(
                map(str, secondary)
            )
            assert (task_dir / "output").is_dir()

        metadata = json.loads((attempt / "job_metadata.json").read_text())
        assert metadata["num_files"] == 2

    @pytest.mark.parametrize(
        "options,error,message",
        [
            ({"joint_validate_trees": "yes"}, TypeError, "must be a boolean"),
            (
                {"joint_validate_trees": True, "joint_file_mode": "broadcast"},
                ValueError,
                "requires balanced",
            ),
            (
                {"named_sources": None},
                ValueError,
                "require named primary and secondary",
            ),
            ({"stage": "train"}, ValueError, "valid only for inference"),
            ({"extra_source": True}, ValueError, "exactly primary and secondary"),
            ({"allow_missing_inputs": True}, ValueError, "must survey existing inputs"),
            ({"num_entries": 2}, ValueError, "entry selection overrides"),
            (
                {"entry_fraction_range": (0.0, 0.5)},
                ValueError,
                "entry selection overrides",
            ),
            (
                {"set_overrides": ["io.loader.dataset.primary.n_skip=1"]},
                ValueError,
                "manages dataset and sampler",
            ),
            (
                {"set_overrides": ["io.loader.sampler=joint_sequential"]},
                ValueError,
                "manages dataset and sampler",
            ),
        ],
    )
    def test_balanced_joint_rejects_conflicting_options(
        self, mock_submitter, tmp_path, options, error, message
    ):
        """Reject unsupported combinations before surveying or submitting work."""
        options = dict(options)
        sources = {
            "primary": {"source": ["primary.root"]},
            "secondary": {"source": ["secondary.root"]},
        }
        if options.pop("extra_source", False):
            sources["extra"] = {"source": ["extra.root"]}
        kwargs = {
            "joint_file_mode": "balanced",
            "named_sources": sources,
            "run_dir": str(tmp_path / "run"),
        }
        kwargs.update(options)
        with patch("src.submission.batch.survey_sources") as survey, patch.object(
            mock_submitter.batch_client, "submit"
        ) as submit:
            with pytest.raises(error, match=message):
                mock_submitter.submit_job(
                    config="infer/nd-lar/full_chain_260409.yaml", **kwargs
                )
            survey.assert_not_called()
            submit.assert_not_called()

    @pytest.mark.parametrize(
        "spine_command",
        ["spine", "/opt/spine/bin/spine", "python3 /opt/spine/bin/run.py"],
    )
    def test_submit_job_balances_joint_entries(
        self, mock_submitter, tmp_path, spine_command
    ):
        run_dir = tmp_path / "run"
        primary = [tmp_path / f"p-{index}.root" for index in range(5)]
        secondary = [tmp_path / "secondary.root"]
        for path in primary + secondary:
            path.touch()
        with (
            patch.object(
                mock_submitter.runtime,
                "resolve_spine_command",
                return_value=(spine_command, None),
            ),
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="joint"),
            patch(
                "src.submission.batch.survey_sources",
                return_value={"primary": [5] * 5, "secondary": [200]},
            ) as survey,
        ):
            mock_submitter.submit_job(
                config="infer/nd-lar/full_chain_260409.yaml",
                named_sources={
                    "primary": {"source": list(map(str, primary))},
                    "secondary": {"source": list(map(str, secondary))},
                },
                apply_mods=["joint:240819"],
                run_dir=str(run_dir),
                joint_file_mode="balanced",
                joint_validate_trees=True,
                set_overrides=["base.seed=123"],
                files_per_task=2,
                ntasks=2,
            )
        assert survey.call_args.kwargs["validate_trees"] is True
        metadata = json.loads((run_dir / "latest" / "job_metadata.json").read_text())
        assert metadata["joint_validate_trees"] is True
        attempt = run_dir / "latest"
        script = (attempt / "submit.sbatch").read_text()
        assert "#SBATCH --array=1-3%2" in script
        assert (
            'io.loader.dataset.secondary.n_skip=$(cat "$TASK_DIR/secondary.skip")'
            in script
        )
        assert "io.loader.sampler.drop_last=false" in script
        # Exercise both quoting layers with a fake container and executable.
        import subprocess

        assert f" exec {spine_command} " in script
        command_line = next(
            line
            for line in script.splitlines()
            if line.startswith('SINGULARITY_INNER_CMD="$SINGULARITY_INNER_CMD exec ')
        ).replace(f" exec {spine_command} ", " fake_spine ", 1)
        container_line = next(
            line for line in script.splitlines() if line.startswith("SINGULARITY_CMD=")
        )
        task = attempt / "tasks" / "000_2"
        shell = r"""fake_spine() { printf '%s\0' "$@"; }
export -f fake_spine
singularity() {
    while [ "$1" != bash ]; do shift; done
    "$@"
}
TASK_DIR="$1"
BIND_PATHS=/unused
SPINE_CONTAINER_PATH=/unused.sif
"""
        result = subprocess.run(
            [
                "bash",
                "-c",
                shell
                + command_line
                + "\n"
                + container_line
                + '\neval "$SINGULARITY_CMD"',
                "bash",
                str(task),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        arguments = result.stdout.split("\0")[:-1]
        assert "io.loader.dataset.secondary.n_skip=10" in arguments
        assert "io.loader.sampler={}" in arguments
        assert "io.loader.sampler.name=joint_sequential" in arguments
        assert "io.loader.sampler.drop_last=false" in arguments
        plan = json.loads((attempt / "joint_shards.json").read_text())
        assert [s["entries"] for s in plan["shards"]] == [10, 10, 5]
        for index, skip in enumerate((0, 10, 20), 1):
            task = attempt / "tasks" / f"000_{index}"
            assert (task / "secondary.txt").read_text().splitlines() == list(
                map(str, secondary)
            )
            assert int((task / "secondary.skip").read_text()) == skip
            assert len((task / "primary.txt").read_text().splitlines()) <= 2

    @pytest.mark.parametrize(
        "files_per_task,ntasks,num_files,max_array_size,expected_entries",
        [
            (None, 2, None, 99, [8, 2]),
            (2, 1, None, 1, [3, 7]),
            (None, None, 2, 99, [3]),
        ],
    )
    def test_balanced_submission_surveys_real_files_and_writes_disjoint_tasks(
        self,
        mock_submitter,
        tmp_path,
        monkeypatch,
        files_per_task,
        ntasks,
        num_files,
        max_array_size,
        expected_entries,
    ):
        """Exercise real ROOT surveys through task manifests and scheduler chunks."""
        uproot = pytest.importorskip("uproot")
        import numpy as np

        sizes = {"primary": [0, 3, 5, 4, 2], "secondary": [4, 4, 2]}
        paths = {}
        for role, counts in sizes.items():
            paths[role] = []
            for index, count in enumerate(counts):
                path = tmp_path / f"{role}-{index:02d}.root"
                with uproot.recreate(path) as source:
                    for name in ("data_tree", "truth_tree"):
                        source.mktree(name, {"entry": "int64"})
                        if count:
                            source[name].extend(
                                {"entry": np.arange(count, dtype=np.int64)}
                            )
                paths[role].append(str(path))
        run_dir = tmp_path / "run"
        monkeypatch.setitem(
            mock_submitter.profiles["defaults"], "max_array_size", max_array_size
        )
        with patch.object(
            mock_submitter.batch,
            "get_batch_client",
            return_value=mock_submitter.batch_client,
        ), patch.object(
            mock_submitter.batch_client, "submit", return_value="joint"
        ) as submit:
            mock_submitter.submit_job(
                config="infer/nd-lar/full_chain_260409.yaml",
                apply_mods=["joint:240819"],
                joint_file_mode="balanced",
                joint_validate_trees=True,
                named_sources={
                    role: {"source": values if num_files else list(reversed(values))}
                    for role, values in paths.items()
                },
                run_dir=str(run_dir),
                num_files=num_files,
                files_per_task=files_per_task,
                ntasks=ntasks,
                dry_run=True,
            )
        attempt = run_dir / "latest"
        plan = json.loads((attempt / "joint_shards.json").read_text())
        assert [shard["entries"] for shard in plan["shards"]] == expected_entries
        assert plan["counts"]["primary"] == sizes["primary"][:num_files]
        tasks = sorted((attempt / "tasks").iterdir())
        assert len(tasks) == len(expected_entries)
        assert submit.call_count == (len(tasks) + max_array_size - 1) // max_array_size
        consumed = {role: [] for role in paths}
        for task, shard in zip(tasks, plan["shards"]):
            count = int((task / "entries.count").read_text())
            assert count == shard["entries"]
            for role in paths:
                files = (task / f"{role}.txt").read_text().splitlines()
                assert files == shard[role]["files"]
                skip = int((task / f"{role}.skip").read_text())
                entries = [
                    (path, entry)
                    for path in files
                    for entry in range(sizes[role][paths[role].index(path)])
                ]
                consumed[role].extend(entries[skip : skip + count])
        for role in paths:
            eligible = [
                (path, entry)
                for path, count in zip(paths[role], sizes[role])
                for entry in range(count)
            ]
            assert consumed[role] == eligible[: sum(expected_entries)]
        if num_files:
            cache = json.loads((run_dir / "entry_counts.json").read_text())
            assert all(path not in cache for path in paths["primary"][num_files:])
        for script in attempt.glob("submit*.sbatch"):
            if max_array_size == 1 or len(tasks) == 1:
                assert "#SBATCH --array=" not in script.read_text()

    def test_submit_job_pairs_joint_files_up_to_shorter_list(
        self, mock_submitter, tmp_path, capsys
    ):
        run_dir = tmp_path / "run"
        primary = [tmp_path / f"primary-{index}.root" for index in range(3)]
        secondary = [tmp_path / f"secondary-{index}.root" for index in range(2)]
        for path in primary + secondary:
            path.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="joint"),
        ):
            mock_submitter.submit_job(
                config="infer/nd-lar/full_chain_260409.yaml",
                named_sources={
                    "primary": {"source": list(map(str, primary))},
                    "secondary": {"source": list(map(str, secondary))},
                },
                apply_mods=["joint:240819"],
                run_dir=str(run_dir),
                joint_file_mode="paired",
                files_per_task=1,
            )

        attempt = run_dir / "latest"
        script = (attempt / "submit.sbatch").read_text(encoding="utf-8")
        assert "#SBATCH --array=1-2" in script
        assert "--output-dir $TASK_DIR/output" in script
        assert 'SPINE_PROD_TASK_OUTPUT_DIR="$TASK_DIR/output"' in script
        assert "flock -x 9" in script
        for index in (1, 2):
            task_dir = attempt / "tasks" / f"000_{index}"
            assert (task_dir / "primary.txt").read_text().splitlines() == [
                str(primary[index - 1])
            ]
            assert (task_dir / "secondary.txt").read_text().splitlines() == [
                str(secondary[index - 1])
            ]
            assert (task_dir / "output").is_dir()
        assert not (attempt / "tasks" / "000_3").exists()

        output = capsys.readouterr().out
        assert "Primary files:   3" in output
        assert "Secondary files: 2" in output
        assert "File pairs:      2" in output
        assert "Ignored primary: 1" in output

        metadata = json.loads((attempt / "job_metadata.json").read_text())
        assert metadata["joint_file_mode"] == "paired"
        assert metadata["num_files"] == 2
        output_manifest = attempt.resolve() / "outputs.txt"
        assert output_manifest.is_file()
        assert metadata["output_manifest"] == str(output_manifest)

    def test_submit_job_reports_ignored_secondary_joint_files(
        self, mock_submitter, tmp_path, capsys
    ):
        """Paired submission reports truncation from either dataset."""
        primary = [tmp_path / "primary.root"]
        secondary = [tmp_path / f"secondary-{index}.root" for index in range(2)]
        for path in primary + secondary:
            path.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="joint"),
        ):
            mock_submitter.submit_job(
                config="infer/nd-lar/full_chain_260409.yaml",
                named_sources={
                    "primary": {"source": list(map(str, primary))},
                    "secondary": {"source": list(map(str, secondary))},
                },
                joint_file_mode="paired",
                run_dir=str(tmp_path / "run"),
            )

        assert "Ignored secondary: 1" in capsys.readouterr().out

    def test_submit_job_limits_named_validation_sources(self, mock_submitter, tmp_path):
        """Named validation datasets honor their independent file limit."""
        validation = [tmp_path / f"validation-{index}.root" for index in range(2)]
        for path in validation:
            path.touch()
        run_dir = tmp_path / "run"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train"),
        ):
            mock_submitter.submit_job(
                config="train/generic/uresnet/train_240718.yaml",
                validation_named_sources={
                    "primary": {"source": list(map(str, validation))}
                },
                val_num_files=1,
                stage="train",
                run_dir=str(run_dir),
            )

        metadata = json.loads((run_dir / "latest/job_metadata.json").read_text())
        assert metadata["validation_num_files"] == 1

    def test_submit_job_places_single_task_joint_output_under_attempt(
        self, mock_submitter, tmp_path
    ):
        run_dir = tmp_path / "run"
        primary = [tmp_path / f"primary-{index}.root" for index in range(2)]
        secondary = [tmp_path / f"secondary-{index}.root" for index in range(2)]
        for path in primary + secondary:
            path.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="joint"),
        ):
            mock_submitter.submit_job(
                config="infer/nd-lar/full_chain_260409.yaml",
                named_sources={
                    "primary": {"source": list(map(str, primary))},
                    "secondary": {"source": list(map(str, secondary))},
                },
                apply_mods=["joint:240819"],
                run_dir=str(run_dir),
                joint_file_mode="paired",
            )

        attempt = (run_dir / "latest").resolve()
        task_dir = attempt / "tasks" / "000_1"
        script = (attempt / "submit.sbatch").read_text(encoding="utf-8")
        assert "#SBATCH --array=" not in script
        assert f'TASK_DIR="{task_dir}"' in script
        assert "--output-dir $TASK_DIR/output" in script
        assert (task_dir / "output").is_dir()

        metadata = json.loads((attempt / "job_metadata.json").read_text())
        assert metadata["output"] == str(attempt / "tasks")
        assert metadata["output_dir"] == str(attempt / "tasks")

    def test_submit_job_writes_expected_stage_cache_source_list(
        self, mock_submitter, tmp_path
    ):
        """A first cache stage can publish paths for dependent array jobs."""
        sources = [tmp_path / "first.root", tmp_path / "second.root"]
        for source in sources:
            source.touch()
        output = tmp_path / "cache"
        source_list = output / "cache_file_list.txt"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="cache"),
        ):
            assert mock_submitter.submit_job(
                config="cache/generic/uresnet_ppn/segmentation_240805.yaml",
                files=[str(source) for source in sources],
                output=str(output),
                output_suffix="cache",
                output_source_list=str(source_list),
            ) == ["cache"]

        assert source_list.read_text(encoding="utf-8").splitlines() == [
            str(output / "first_cache.h5"),
            str(output / "second_cache.h5"),
        ]
        metadata = json.loads(
            next(mock_submitter.jobs_dir.glob("**/job_metadata.json")).read_text()
        )
        assert metadata["output_source_list"] == str(source_list)

    def test_submit_job_fences_parallel_cache_publication(
        self, mock_submitter, tmp_path
    ):
        """One publication ID and source-count barrier span an entire array."""
        sources = [tmp_path / "first.root", tmp_path / "second.root"]
        for source in sources:
            source.touch()
        repository = tmp_path / "cache" / "train.spine-cache"
        run_dir = tmp_path / "cache" / "segmentation"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="cache"),
            patch(
                "src.submission.batch.uuid.uuid4", return_value=Mock(hex="attempt-id")
            ),
        ):
            assert mock_submitter.submit_job(
                config="cache/generic/uresnet_ppn/segmentation_240805.yaml",
                files=[str(path) for path in sources],
                output=str(repository),
                run_dir=str(run_dir),
                files_per_task=1,
                cache_repository=str(repository),
                cache_stage="segmentation",
            ) == ["cache"]

        script = (run_dir / "latest" / "submit.sbatch").read_text()
        assert "export SPINE_CACHE_PUBLICATION_ID=attempt-id" in script
        assert (
            f"spine-cache begin {repository} segmentation "
            '--publication-id \\"\\$SPINE_CACHE_PUBLICATION_ID\\"'
        ) in script
        assert "--set io.writer.parallel=true" in script
        assert "--set io.writer.expected_sources=2" in script
        metadata = json.loads((run_dir / "latest" / "job_metadata.json").read_text())
        assert metadata["cache_repository"] == str(repository)
        assert metadata["cache_stage"] == "segmentation"
        assert metadata["cache_publication_id"] == "attempt-id"

    @pytest.mark.parametrize(
        "options",
        [
            {"cache_repository": "/tmp/cache.spine-cache"},
            {"cache_stage": "segmentation"},
            {
                "cache_repository": "/tmp/cache.spine-cache",
                "cache_stage": "segmentation",
                "stage": "train",
                "run_dir": "/tmp/train",
            },
        ],
    )
    def test_submit_job_rejects_invalid_cache_publication(
        self, mock_submitter, options
    ):
        """Fenced cache publication requires a complete inference contract."""
        with pytest.raises(ValueError, match="cache|Cache"):
            mock_submitter.submit_job(config="config.yaml", **options)

    def test_submit_job_rejects_cache_publication_without_sources(
        self, mock_submitter, tmp_path
    ):
        """A parallel cache barrier cannot be inferred without source files."""
        with pytest.raises(ValueError, match="requires input sources"):
            mock_submitter.submit_job(
                config="cache/generic/uresnet_ppn/segmentation_240805.yaml",
                run_dir=str(tmp_path / "cache"),
                cache_repository=str(tmp_path / "train.spine-cache"),
                cache_stage="segmentation",
            )

    def test_cache_training_uses_scalar_source_overrides(
        self, mock_submitter, tmp_path
    ):
        """Logical cache repositories must not be converted to source lists."""
        run_dir = tmp_path / "train"
        train_cache = tmp_path / "train.spine-cache"
        validation_cache = tmp_path / "validation.spine-cache"
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train"),
        ):
            mock_submitter.submit_job(
                config="train/generic/grappa_inter/train_from_particle_cache_260828.yaml",
                files=[str(train_cache)],
                validation_files=[str(validation_cache)],
                stage="train",
                run_dir=str(run_dir),
                allow_missing_inputs=True,
            )

        script = (run_dir / "latest" / "submit.sbatch").read_text()
        assert f"--source {train_cache}" in script
        assert f"--val-source {validation_cache}" in script
        assert "--source-list" not in script
        assert "--val-source-list" not in script

    def test_submit_job_rejects_incomplete_output_source_list_contract(
        self, mock_submitter, tmp_path
    ):
        """Publishing future cache paths requires deterministic output naming."""
        source = tmp_path / "input.root"
        source.touch()

        with pytest.raises(ValueError, match="explicit output directory and suffix"):
            mock_submitter.submit_job(
                config="cache/generic/uresnet_ppn/segmentation_240805.yaml",
                files=[str(source)],
                output_source_list=str(tmp_path / "cache_files.txt"),
            )

    def test_submit_job_exports_composed_module_weights(self, mock_submitter, tmp_path):
        """A model-only batch job should forward composition options to SPINE."""
        run_dir = tmp_path / "weights" / "export"
        destination = tmp_path / "weights" / "full_chain.ckpt"
        module_weights = {
            "uresnet_ppn": "/weights/uresnet.ckpt",
            "graph_spice": "/weights/graph_spice.ckpt",
        }

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="export"),
        ):
            assert mock_submitter.submit_job(
                config="model/generic/full_chain/model_240805.yaml",
                module_weights=module_weights,
                export_weights=str(destination),
                profile="s3df_milano",
                run_dir=str(run_dir),
            ) == ["export"]

        script = (run_dir / "latest" / "submit.sbatch").read_text(encoding="utf-8")
        assert "--module-weight uresnet_ppn=/weights/uresnet.ckpt" in script
        assert "graph_spice=/weights/graph_spice.ckpt" in script
        assert f"--export-weights {destination}" in script
        assert " -S " not in script

        metadata = json.loads((run_dir / "latest" / "job_metadata.json").read_text())
        assert metadata["export_weights"] == str(destination)

    @pytest.mark.parametrize(
        "options",
        [
            {"files": ["input.root"]},
            {"named_sources": {"larcv": {"source": "input.root"}}},
            {"output": "output.h5"},
            {"stage": "train", "run_dir": "/tmp/train"},
        ],
    )
    def test_submit_job_rejects_export_runtime_conflicts(self, mock_submitter, options):
        """Model export must remain a terminal model-only operation."""
        with pytest.raises(ValueError, match="export-weights"):
            mock_submitter.submit_job(
                config="model/generic/full_chain/model_240805.yaml",
                export_weights="/weights/full_chain.ckpt",
                **options,
            )

    def test_submit_job_rejects_validation_sources_outside_training(
        self, mock_submitter, tmp_path
    ):
        """Checkpoint-bound validation sources apply only to training runs."""
        source = tmp_path / "validation.root"
        source.touch()
        with pytest.raises(ValueError, match="valid only for training"):
            mock_submitter.submit_job(
                config="config/infer/generic/full_chain_240718.yaml",
                validation_files=[str(source)],
            )

    def test_submit_job_rejects_empty_validation_sources(
        self, mock_submitter, tmp_path
    ):
        """Training validation sources must resolve to at least one file."""
        missing_source = tmp_path / "missing.root"
        with pytest.raises(ValueError, match="No validation input files found"):
            mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml",
                validation_files=[str(missing_source)],
                stage="train",
                run_dir=str(tmp_path / "run"),
            )

    def test_submit_job_rejects_training_source_splitting(
        self, mock_submitter, tmp_path
    ):
        """Training datasets must not be split into independent array jobs."""
        source = tmp_path / "train.root"
        source.touch()
        with pytest.raises(ValueError, match="valid only for inference"):
            mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml",
                files=[str(source)],
                stage="train",
                run_dir=str(tmp_path / "run"),
                files_per_task=1,
            )

    def test_submit_job_rejects_nonempty_explicit_inference_run(
        self, mock_submitter, tmp_path
    ):
        """Test explicit inference runs cannot mix with existing artifacts."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "artifact").touch()
        with pytest.raises(ValueError, match="Inference run directory is not empty"):
            mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml",
                run_dir=str(run_dir),
            )

    def test_submit_job_creates_explicit_inference_run(self, mock_submitter, tmp_path):
        """Test inference may use a caller-selected new run directory."""
        run_dir = tmp_path / "chosen-run"
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value=None),
        ):
            assert (
                mock_submitter.submit_job(
                    config="config/train/generic/uresnet/train_240718.yaml",
                    run_dir=str(run_dir),
                )
                == []
            )

        assert run_dir.is_dir()
        assert (run_dir / "latest" / "job_metadata.json").is_file()

    def test_submit_job_retry_preserves_prior_inference_attempt(
        self, mock_submitter, tmp_path
    ):
        """A pipeline retry should write new scheduler artifacts beside the old."""
        run_dir = tmp_path / "cache-stage"
        config = "config/train/generic/uresnet/train_240718.yaml"
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(
                mock_submitter.batch_client,
                "submit",
                side_effect=["first", "retry"],
            ),
        ):
            assert mock_submitter.submit_job(config=config, run_dir=str(run_dir)) == [
                "first"
            ]
            original_script = run_dir / "latest" / "submit.sbatch"
            original_attempt = original_script.resolve().parent
            original_script = original_attempt / "submit.sbatch"
            original_text = original_script.read_text(encoding="utf-8")

            assert mock_submitter.submit_job(
                config=config,
                run_dir=str(run_dir),
                retry=True,
            ) == ["retry"]

        attempts = list(run_dir.glob("attempts/*"))
        assert len(attempts) == 2
        assert all((attempt / "submit.sbatch").is_file() for attempt in attempts)
        assert all((attempt / "job_metadata.json").is_file() for attempt in attempts)
        assert (run_dir / "latest").resolve() != original_attempt
        assert original_script.read_text(encoding="utf-8") == original_text

    def test_submit_training_and_resume_share_run_artifacts(
        self, mock_submitter, tmp_path
    ):
        """Test training creation and resume reuse logs, weights, and metadata."""
        run_dir = tmp_path / "experiments" / "deghost" / "default"
        config = "config/train/generic/uresnet/train_240718.yaml"

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train-1"),
        ):
            assert mock_submitter.submit_job(
                config=config,
                profile="s3df_ampere",
                stage="train",
                run_dir=str(run_dir),
                tensorboard=True,
            ) == ["train-1"]

        first_script = run_dir / "latest" / "submit.sbatch"
        first_text = first_script.read_text(encoding="utf-8")
        assert f"--log-dir {run_dir}" in first_text
        assert f"--weight-prefix {run_dir}/weights/snapshot" in first_text
        assert (
            f"--tensorboard --tensorboard-dir {run_dir}/tensorboard/train" in first_text
        )
        assert "base.tensorboard=" not in first_text
        assert (run_dir / "run_metadata.json").is_file()

        saved = run_dir / "weights" / "snapshot-99999.ckpt"
        saved.touch()
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="train-2"),
        ):
            assert mock_submitter.submit_job(
                config=config,
                profile="s3df_ampere",
                stage="train",
                run_dir=str(run_dir),
                resume=True,
            ) == ["train-2"]

        scripts = sorted(run_dir.glob("attempts/*/submit.sbatch"))
        resumed_text = scripts[-1].read_text(encoding="utf-8")
        assert f"--weight-path {saved}" in resumed_text
        assert "--resume" in resumed_text
        assert "restore_optimizer" not in resumed_text
        metadata = json.loads((scripts[-1].parent / "job_metadata.json").read_text())
        assert metadata["stage"] == "train"
        assert metadata["resume_checkpoint"] == str(saved)

    @pytest.mark.parametrize("tensorboard", [False, True])
    def test_submit_validation_selects_missing_checkpoints_and_noops(
        self, mock_submitter, tmp_path, capsys, tensorboard
    ):
        """Test validation submits only checkpoints without complete CSV logs."""
        run_dir = tmp_path / "experiments" / "deghost" / "default"
        train_config = "config/train/generic/uresnet/train_240718.yaml"
        val_config = "config/test/generic/full_chain/evaluate_240718.yaml"
        RunManager.prepare_training_run(run_dir, train_config)
        first = run_dir / "weights" / "snapshot-9.ckpt"
        second = run_dir / "weights" / "snapshot-19.ckpt"
        first.touch()
        second.touch()
        (run_dir / "inference_log-0000010.csv").write_text(
            "iter,loss\n0,1\n", encoding="utf-8"
        )

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(
                mock_submitter.batch_client, "submit", return_value="val-1"
            ) as submit,
        ):
            assert mock_submitter.submit_job(
                config=val_config,
                profile="s3df_ampere",
                stage="validation",
                run_dir=str(run_dir),
                tensorboard=tensorboard,
            ) == ["val-1"]
            submit.assert_called_once()

        submission = run_dir / "latest"
        assert submission.is_dir()
        assert (submission / "weights.txt").read_text().splitlines() == [str(second)]
        script = (submission / "submit.sbatch").read_text(encoding="utf-8")
        assert f"--log-dir {run_dir}" in script
        assert f"--weight-list {(submission / 'weights.txt').resolve()}" in script
        assert "model.weight_path=null" in script
        if tensorboard:
            assert (
                f"--tensorboard --tensorboard-dir {run_dir}/tensorboard/validation"
                in script
            )
        else:
            assert "--tensorboard" not in script
            assert not (run_dir / "tensorboard").exists()

        (run_dir / "inference_log-0000020.csv").write_text(
            "iter,loss\n0,1\n", encoding="utf-8"
        )
        with patch.object(mock_submitter.batch_client, "submit") as submit:
            assert (
                mock_submitter.submit_job(
                    config=val_config,
                    profile="s3df_ampere",
                    stage="validation",
                    run_dir=str(run_dir),
                )
                == []
            )
            submit.assert_not_called()
        assert "Validation is up to date" in capsys.readouterr().out

    def test_submit_named_validation_can_rerun_all_checkpoints(
        self, mock_submitter, tmp_path
    ):
        """Test named validation gets isolated logs and an explicit overwrite plan."""
        run_dir = tmp_path / "run"
        train_config = "config/train/generic/uresnet/train_240718.yaml"
        val_config = "config/test/generic/full_chain/evaluate_240718.yaml"
        RunManager.prepare_training_run(run_dir, train_config)
        saved = run_dir / "weights" / "snapshot-4.ckpt"
        saved.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="val"),
        ):
            assert mock_submitter.submit_job(
                config=val_config,
                profile="s3df_ampere",
                stage="validation",
                run_dir=str(run_dir),
                validation_name="data",
                rerun_validation=True,
                tensorboard=True,
            ) == ["val"]

        submission = run_dir / "latest"
        script = (submission / "submit.sbatch").read_text(encoding="utf-8")
        assert f"--log-dir {run_dir}/validation/data" in script
        assert "base.overwrite_log=true" in script
        assert (
            f"--tensorboard --tensorboard-dir "
            f"{run_dir}/tensorboard/validation/data" in script
        )

    def test_submit_job_rejects_missing_explicit_inputs(self, mock_submitter, tmp_path):
        with pytest.raises(ValueError, match="No input files found"):
            mock_submitter.submit_job(
                config="config.yaml", files=[str(tmp_path / "missing.root")]
            )

    def test_submit_job_accepts_custom_name_existing_binds_and_output_file(
        self, mock_submitter, tmp_path
    ):
        input_file = tmp_path / "input.root"
        input_file.touch()
        larcv_root = tmp_path / "larcv"
        larcv_root.mkdir()
        (larcv_root / "configure.sh").touch()
        output = tmp_path / "nested" / "result.h5"
        profile_config = mock_submitter.config_mgr.get_profile("auto", "sbnd")
        profile_config["bind_paths"] = "/existing"

        with (
            patch.object(
                mock_submitter.config_mgr,
                "get_profile",
                return_value=profile_config,
            ),
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value=None),
        ):
            assert (
                mock_submitter.submit_job(
                    "infer/generic/full_chain_240718.yaml",
                    files=[str(input_file)],
                    job_name="custom_job",
                    output=str(output),
                    larcv_path=str(larcv_root),
                    gpus=4,
                    batch_size=8,
                    num_workers=4,
                    epochs=2.5,
                    dry_run=True,
                )
                == []
            )

        assert output.parent.is_dir()
        assert profile_config["bind_paths"] == f"/existing,{larcv_root}"
        script = next(
            mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch")
        ).read_text(encoding="utf-8")
        assert "#SBATCH --gpus=4" in script
        assert "--world-size 4 --batch-size 8 --num-workers 4 --epochs 2.5" in script
        metadata_path = next(mock_submitter.jobs_dir.glob("**/job_metadata.json"))
        assert json.loads(metadata_path.read_text())["world_size"] == 4

    def test_submit_job_uses_spine_path_and_merges_bind_root(
        self, mock_submitter, tmp_path
    ):
        """Test batch submission writes the overridden SPINE command into scripts."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        spine_checkout = tmp_path / "spine"
        larcv_root = tmp_path / "larcv"
        flashmatch_root = tmp_path / "flashmatch"
        (spine_checkout / "bin").mkdir(parents=True)
        larcv_root.mkdir()
        flashmatch_root.mkdir()
        (larcv_root / "configure.sh").write_text("export LARCV=1\n", encoding="utf-8")
        (flashmatch_root / "configure.sh").write_text(
            "export FMATCH=1\n", encoding="utf-8"
        )
        run_py = spine_checkout / "bin" / "run.py"
        run_py.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                profile="s3df_ampere",
                larcv_path=str(larcv_root),
                flashmatch_path=str(flashmatch_root),
                spine_path=str(spine_checkout),
            )

        assert job_ids == ["12345"]

        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1

        script = scripts[0].read_text(encoding="utf-8")
        assert f"python3 {run_py} -S $TASK_FILE_LIST" in script
        assert str(spine_checkout) in script
        assert f'source "{larcv_root / "configure.sh"}"' in script
        assert f'source "{flashmatch_root / "configure.sh"}"' in script
        assert str(larcv_root) in script
        assert str(flashmatch_root) in script

    def test_submit_job_keeps_default_s3df_bind_root_with_spine_path(
        self, mock_submitter, tmp_path
    ):
        """Test S3DF keeps the default /sdf bind when adding a custom spine path."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        spine_checkout = Path("/sdf/data/neutrino/software/spine-dev")

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
            patch.object(
                mock_submitter.runtime,
                "resolve_spine_command",
                return_value=(
                    f"python3 {spine_checkout / 'bin' / 'run.py'}",
                    str(spine_checkout),
                ),
            ),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/infer/dune-hd-10kt-1x2x6/full_chain_260510.yaml",
                files=[str(input_file)],
                profile="s3df_ampere",
                spine_path=str(spine_checkout),
            )

        assert job_ids == ["12345"]

        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1

        script = scripts[0].read_text(encoding="utf-8")
        assert 'BIND_PATHS="/sdf/,/sdf/data/neutrino/software/spine-dev"' in script

    def test_submit_job_accepts_output_suffix(self, mock_submitter, tmp_path):
        """Test batch submission can override the derived output suffix."""
        input_file = tmp_path / "input.root"
        input_file.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                profile="s3df_ampere",
                output_suffix="custom_reco",
            )

        assert job_ids == ["12345"]

        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1

        script = scripts[0].read_text(encoding="utf-8")
        assert "--output-suffix custom_reco" in script

    def test_submit_job_defaults_to_single_task_over_all_files(
        self, mock_submitter, tmp_path
    ):
        """Test explicit file lists default to one task containing all files."""
        input_files = []
        for idx in range(3):
            input_file = tmp_path / f"input_{idx}.root"
            input_file.touch()
            input_files.append(str(input_file))

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=input_files,
                profile="s3df_ampere",
            )

        assert job_ids == ["12345"]

        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1
        script = scripts[0].read_text(encoding="utf-8")
        assert "#SBATCH --array=" not in script
        assert 'TASK_DIR="' not in script
        assert f"--output-dir {scripts[0].parent}/output" in script
        assert f"--log-dir {scripts[0].parent}" in script
        assert f'SPINE_PROD_TASK_OUTPUT_DIR="{scripts[0].parent}/output"' in script
        assert f'SPINE_PROD_OUTPUT_MANIFEST="{scripts[0].parent}/outputs.txt"' in script

        manifest = scripts[0].parent / "inputs.txt"
        assert manifest.read_text(encoding="utf-8").strip().splitlines() == input_files
        assert (scripts[0].parent / "output").is_dir()
        assert (scripts[0].parent / "outputs.txt").is_file()
        assert not (scripts[0].parent / "logs").exists()
        assert not (scripts[0].parent / "tasks").exists()

    def test_submit_job_in_place_leaves_writer_destination_config_defined(
        self, mock_submitter, tmp_path
    ):
        """In-place jobs must not create or pass a default writer destination."""
        input_file = tmp_path / "cache.h5"
        input_file.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/cache/generic/grappa_shower_track/particle_graphs_240805.yaml",
                files=[str(input_file)],
                profile="s3df_ampere",
                in_place=True,
            )

        assert job_ids == ["12345"]
        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1
        script = scripts[0].read_text(encoding="utf-8")
        assert "--output " not in script
        assert "--output-dir" not in script
        assert "--output-suffix" not in script
        assert not (scripts[0].parent / "output").exists()

        metadata = json.loads(
            (scripts[0].parent / "job_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["in_place"] is True
        assert metadata["output"] is None
        assert metadata["output_dir"] is None

    def test_submit_job_rejects_in_place_with_output(self, mock_submitter):
        """Config-owned and explicit writer destinations are mutually exclusive."""
        with pytest.raises(ValueError, match="--in-place cannot be combined"):
            mock_submitter.submit_job(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=["input.root"],
                in_place=True,
                output="output.h5",
            )

    def test_submit_job_no_writer_is_deprecated_and_ignored(
        self, mock_submitter, tmp_path, capsys
    ):
        """Test deprecated no-writer leaves automatic output options enabled."""
        input_file = tmp_path / "input.root"
        input_file.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                profile="s3df_ampere",
                no_writer=True,
            )

        assert job_ids == ["12345"]

        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1
        script = scripts[0].read_text(encoding="utf-8")
        assert " -S $TASK_FILE_LIST" in script
        assert "--output-dir" in script
        assert "--output-suffix full_chain_co_260316" in script
        assert "Output directory:" in script
        warning = capsys.readouterr().out
        assert "--no-writer is deprecated and ignored" in warning

    def test_submit_job_ignores_no_writer_with_output_suffix(
        self, mock_submitter, tmp_path
    ):
        """Test an explicit output suffix wins over deprecated no-writer."""
        input_file = tmp_path / "input.root"
        input_file.touch()

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                profile="s3df_ampere",
                no_writer=True,
                output_suffix="custom_reco",
            )

        assert job_ids == ["12345"]
        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1
        assert "--output-suffix custom_reco" in scripts[0].read_text(encoding="utf-8")

    def test_submit_job_uses_config_inputs_when_files_omitted(self, mock_submitter):
        """Test batch submission can defer input discovery to the config."""
        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml",
                profile="s3df_ampere",
            )

        assert job_ids == ["12345"]

        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1
        script = scripts[0].read_text(encoding="utf-8")
        assert "Using input files defined in the config" in script
        assert " -S $TASK_FILE_LIST" not in script
        assert "--output-dir" not in script
        assert "--output-suffix" not in script
        assert "Output: defined in config" in script

    def test_submit_job_rejects_split_flags_without_files(self, mock_submitter):
        """Test config-driven inputs cannot be combined with submit-time splitting."""
        with pytest.raises(
            ValueError,
            match="Cannot use --ntasks/--files-per-task without --source/--source-list",
        ):
            mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml",
                profile="s3df_ampere",
                ntasks=2,
            )

    def test_submit_job_rejects_output_without_files(self, mock_submitter):
        """Test config-owned batch runs reject inference output overrides."""
        with pytest.raises(
            ValueError,
            match="Cannot use --output/--output-suffix without --source/--source-list",
        ):
            mock_submitter.submit_job(
                config="config/train/generic/uresnet/train_240718.yaml",
                profile="s3df_ampere",
                output_suffix="custom_reco",
            )

    def test_submit_job_uses_ntasks_as_target_task_count(
        self, mock_submitter, tmp_path
    ):
        """Test ntasks alone spreads explicit files across roughly even task sizes."""
        input_files = []
        for idx in range(10):
            input_file = tmp_path / f"input_{idx}.root"
            input_file.touch()
            input_files.append(str(input_file))

        with (
            patch.object(
                mock_submitter.batch,
                "get_batch_client",
                return_value=mock_submitter.batch_client,
            ),
            patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
        ):
            job_ids = mock_submitter.submit_job(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=input_files,
                profile="s3df_ampere",
                ntasks=3,
            )

        assert job_ids == ["12345"]

        scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
        assert len(scripts) == 1
        script = scripts[0].read_text(encoding="utf-8")
        assert "#SBATCH --array=1-3" in script
        assert "%3" not in script

        task_lists = sorted(
            mock_submitter.jobs_dir.glob("**/attempts/*/tasks/000_*/inputs.txt")
        )
        assert len(task_lists) == 3
        task_sizes = [
            len(path.read_text(encoding="utf-8").strip().splitlines())
            for path in task_lists
        ]
        assert task_sizes == [4, 4, 2]

    def test_save_job_metadata(self, mock_submitter, tmp_path):
        """Test saving job metadata to JSON."""
        metadata = {
            "config": "infer/icarus/latest.yaml",
            "files": ["file1.root", "file2.root"],
            "profile": "s3df_ampere",
            "timestamp": "2026-01-05T10:00:00",
        }

        mock_submitter.batch_client.save_job_metadata(tmp_path, metadata)

        metadata_file = tmp_path / "job_metadata.json"
        assert metadata_file.exists()

        # Verify contents
        with open(metadata_file) as f:
            loaded = json.load(f)
            assert loaded["config"] == metadata["config"]
            assert loaded["profile"] == metadata["profile"]
            assert len(loaded["files"]) == 2


class TestCVMFSOption:
    """Tests for opt-in CVMFS container exposure."""

    def _render_template(self, mock_submitter, template_name, **kwargs):
        """Render a job template with minimal defaults."""
        template = mock_submitter.batch_client.load_template(template_name)
        defaults = {
            "account": "test-account",
            "partition": "test-partition",
            "qos": "test-qos",
            "queue": "debug",
            "constraint": None,
            "gpus": 0,
            "gpus_per_node": 0,
            "nodes": 1,
            "system": "polaris",
            "cpus_per_node": 32,
            "cpus_per_task": 1,
            "mem_per_cpu": "1g",
            "mem": "1g",
            "filesystems": "home:grand:eagle",
            "place": None,
            "exclusive": False,
            "exclude": None,
            "time": "00:10:00",
            "array_spec": None,
            "job_name": "test-job",
            "log_dir": "/tmp/logs",
            "dependency": None,
            "config": "/tmp/config.yaml",
            "output": "/tmp/output.h5",
            "output_dir": "/tmp/output",
            "output_suffix": "config",
            "output_args": "--output /tmp/output.h5",
            "basedir": "/tmp/spine-prod",
            "file_list_pattern": "/tmp/files_*.txt",
            "task_dir_pattern": "/tmp/tasks/task_*",
            "spine_log_dir": "/tmp/spine-logs",
            "larcv_path": None,
            "flashmatch_path": None,
            "flashmatch": False,
            "cvmfs": False,
            "expandable_segments": False,
            "bind_paths": None,
            "spine_cmd": "spine",
            "spine_cli_overrides": "",
            "graceful_stop_file": None,
        }
        defaults.update(kwargs)
        return template.render(**defaults)

    @pytest.mark.parametrize(
        "template_name",
        ["job_template_s3df.sbatch", "job_template_nersc.sbatch"],
    )
    def test_slurm_full_node_profiles_request_exclusive_access(
        self, mock_submitter, template_name
    ):
        """Slurm templates should render explicit full-node exclusivity."""
        shared = self._render_template(mock_submitter, template_name)
        exclusive = self._render_template(mock_submitter, template_name, exclusive=True)

        assert "#SBATCH --exclusive" not in shared
        assert "#SBATCH --exclusive" in exclusive

    @pytest.mark.parametrize(
        "template_name",
        ["job_template_s3df.sbatch", "job_template_nersc.sbatch"],
    )
    def test_slurm_profiles_render_node_exclusions(self, mock_submitter, template_name):
        """Slurm templates should pass node exclusions to the scheduler."""
        script = self._render_template(
            mock_submitter,
            template_name,
            exclude="gpu042,gpu043",
        )

        assert "#SBATCH --exclude=gpu042,gpu043" in script

    def test_polaris_full_node_profiles_request_exclusive_placement(
        self, mock_submitter
    ):
        """Polaris full-node profiles should render PBS exclusive placement."""
        script = self._render_template(
            mock_submitter,
            "job_template_anl.pbs",
            gpus_per_node=4,
            cpus_per_node=64,
            cpus_per_task=64,
            place="scatter:excl",
        )

        assert "#PBS -l select=1:system=polaris:ncpus=64:ngpus=4" in script
        assert "#PBS -l place=scatter:excl" in script

    def test_s3df_does_not_bind_cvmfs_by_default(self, mock_submitter):
        """Test S3DF leaves CVMFS out of bind paths by default."""
        script = self._render_template(mock_submitter, "job_template_s3df.sbatch")

        assert 'BIND_PATHS="/sdf/"' in script
        assert "/cvmfs/" not in script

    def test_s3df_honors_custom_bind_paths(self, mock_submitter):
        """Test S3DF bind roots can be overridden through template context."""
        script = self._render_template(
            mock_submitter,
            "job_template_s3df.sbatch",
            bind_paths="/sdf/,/eaf/,/tmp/work",
        )

        assert 'BIND_PATHS="/sdf/,/eaf/,/tmp/work"' in script

    def test_s3df_binds_cvmfs_when_requested(self, mock_submitter):
        """Test S3DF adds CVMFS to bind paths when requested."""
        script = self._render_template(
            mock_submitter, "job_template_s3df.sbatch", cvmfs=True
        )

        assert 'BIND_PATHS="/sdf/"' in script
        assert 'BIND_PATHS="$BIND_PATHS,/cvmfs/"' in script

    def test_nersc_does_not_load_cvmfs_module_by_default(self, mock_submitter):
        """Test NERSC leaves Shifter CVMFS module out by default."""
        script = self._render_template(mock_submitter, "job_template_nersc.sbatch")

        assert "--module=cvmfs" not in script

    def test_nersc_loads_cvmfs_module_when_requested(self, mock_submitter):
        """Test NERSC adds Shifter CVMFS module when requested."""
        script = self._render_template(
            mock_submitter, "job_template_nersc.sbatch", cvmfs=True
        )

        assert 'SHIFTER_MODULES+=("--module=cvmfs")' in script

    @pytest.mark.parametrize("allocated_cpus", [None, "16", "31", "32"])
    def test_nersc_srun_uses_runtime_cpu_allocation(
        self, mock_submitter, allocated_cpus
    ):
        """Explicit step CPUs take precedence over stale TRES environment values."""
        script = self._render_template(
            mock_submitter, "job_template_nersc.sbatch", cpus_per_task=16
        )
        # Exercise the rendered shell expansion without Slurm or a container.
        assignment = next(
            line for line in script.splitlines() if line.startswith('RUN_CMD="')
        )
        env = dict(os.environ, SLURM_TRES_PER_TASK="cpu=16")
        env.pop("SLURM_CPUS_PER_TASK", None)
        if allocated_cpus is not None:
            env["SLURM_CPUS_PER_TASK"] = allocated_cpus
        result = subprocess.run(
            [
                "bash",
                "-c",
                assignment + '\nset -- $RUN_CMD\nprintf "%s\\n" "$1" "$4" "$5"',
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.splitlines() == ["srun", "-c", allocated_cpus or "16"]

    @pytest.mark.parametrize(
        "template_name",
        [
            "job_template_s3df.sbatch",
            "job_template_nersc.sbatch",
            "job_template_anl.pbs",
        ],
    )
    def test_expandable_segments_is_opt_in(self, mock_submitter, template_name):
        """Allocator expansion is exported only when explicitly requested."""
        default_script = self._render_template(mock_submitter, template_name)
        enabled_script = self._render_template(
            mock_submitter, template_name, expandable_segments=True
        )

        assert "PYTORCH_CUDA_ALLOC_CONF" not in default_script
        assert (
            'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+'
            '${PYTORCH_CUDA_ALLOC_CONF},}expandable_segments:True"' in enabled_script
        )

    @pytest.mark.parametrize(
        "template_name",
        ["job_template_s3df.sbatch", "job_template_nersc.sbatch"],
    )
    def test_slurm_dependencies_cancel_when_they_become_invalid(
        self, mock_submitter, template_name
    ):
        """Slurm descendants should not remain pending after upstream failure."""
        script = self._render_template(
            mock_submitter,
            template_name,
            dependency="afterok:123",
        )

        assert "#SBATCH --dependency=afterok:123" in script
        assert "#SBATCH --kill-on-invalid-dep=yes" in script

    @pytest.mark.parametrize(
        "template_name",
        ["job_template_s3df.sbatch", "job_template_nersc.sbatch"],
    )
    def test_slurm_jobs_without_dependencies_need_no_invalid_dependency_policy(
        self, mock_submitter, template_name
    ):
        """Independent jobs should not receive an irrelevant Slurm directive."""
        script = self._render_template(mock_submitter, template_name)

        assert "--kill-on-invalid-dep" not in script

    def test_anl_template_uses_pbs_and_array_index(self, mock_submitter):
        """Test ANL template uses PBS directives and PBS array variables."""
        script = self._render_template(
            mock_submitter,
            "job_template_anl.pbs",
            array_spec="1-4",
            gpus_per_node=1,
        )

        assert "#PBS -A test-account" in script
        assert "#PBS -q debug" in script
        assert "#PBS -J 1-4" in script
        assert "ngpus=1" in script
        assert "${PBS_ARRAY_INDEX}" in script
        assert "apptainer exec" in script
        assert "spine -S" in script

    def test_anl_uses_native_pbs_dependency_deletion_semantics(self, mock_submitter):
        """PBS needs only afterok; it has no Slurm-style opt-in directive."""
        script = self._render_template(
            mock_submitter,
            "job_template_anl.pbs",
            dependency="afterok:123.server",
        )

        assert "#PBS -W depend=afterok:123.server" in script
        assert "kill-on-invalid-dep" not in script

    def test_templates_allow_config_defined_inputs(self, mock_submitter):
        """Test batch templates can omit submit-time source lists entirely."""
        script = self._render_template(
            mock_submitter,
            "job_template_s3df.sbatch",
            file_list_pattern=None,
            output=None,
            output_args="",
        )

        assert "Using input files defined in the config" in script
        assert " -S $TASK_FILE_LIST" not in script
        assert "--output-dir" not in script
        assert "--output-suffix" not in script
        assert "Output: defined in config" in script

    def test_templates_include_spine_set_overrides(self, mock_submitter):
        """Test SPINE --set overrides are rendered into batch commands."""
        script = self._render_template(
            mock_submitter,
            "job_template_s3df.sbatch",
            spine_cli_overrides="--set base.world_size=0 --set io.loader.batch_size=1",
        )

        assert "--set base.world_size=0" in script
        assert "--set io.loader.batch_size=1" in script

    def test_templates_allow_custom_spine_command(self, mock_submitter):
        """Test batch templates can override the SPINE executable."""
        script = self._render_template(
            mock_submitter,
            "job_template_s3df.sbatch",
            spine_cmd="python3 /tmp/spine/bin/run.py",
            bind_paths="/sdf/,/tmp/spine",
        )

        assert "python3 /tmp/spine/bin/run.py -S $TASK_FILE_LIST" in script
        assert 'BIND_PATHS="/sdf/,/tmp/spine"' in script

    def test_templates_use_writer_directory_and_suffix_by_default(self, mock_submitter):
        """Test batch templates can avoid forcing a writer file name."""
        script = self._render_template(
            mock_submitter,
            "job_template_s3df.sbatch",
            output=None,
            output_dir="/tmp/job/output",
            output_suffix="full_chain_260501",
            output_args=(
                "--output-dir /tmp/job/output " "--output-suffix full_chain_260501"
            ),
        )

        assert "-o /tmp/output.h5" not in script
        assert "--output-dir /tmp/job/output" in script
        assert "--output-suffix full_chain_260501" in script

    def test_templates_use_writer_file_name_for_explicit_output(self, mock_submitter):
        """Test explicit output files use SPINE's dedicated output option."""
        script = self._render_template(
            mock_submitter,
            "job_template_s3df.sbatch",
            output="/tmp/output.h5",
            output_args="--output /tmp/output.h5",
        )

        assert " -o " not in script
        assert "--output /tmp/output.h5" in script

    def test_templates_source_custom_software_paths(self, mock_submitter):
        """Test batch templates source custom software configure scripts."""
        script = self._render_template(
            mock_submitter,
            "job_template_s3df.sbatch",
            larcv_path="/tmp/larcv",
            flashmatch_path="/tmp/flashmatch",
        )

        assert 'source "/tmp/larcv/configure.sh"' in script
        assert 'source "/tmp/flashmatch/configure.sh"' in script

    @pytest.mark.parametrize(
        "template_name",
        [
            "job_template_s3df.sbatch",
            "job_template_nersc.sbatch",
            "job_template_anl.pbs",
        ],
    )
    def test_training_templates_translate_graceful_stop_to_marker(
        self, mock_submitter, template_name
    ):
        """Training wrappers should translate USR1 without signaling children."""
        marker = "/tmp/attempt/graceful_stop"
        script = self._render_template(
            mock_submitter,
            template_name,
            graceful_stop_file=marker,
        )

        assert "trap request_graceful_stop USR1" in script
        assert f'SPINE_PROD_GRACEFUL_STOP_FILE="{marker}"' in script
        assert 'touch -- "$SPINE_PROD_GRACEFUL_STOP_FILE"' in script
        assert 'kill -USR1 "$SPINE_PROD_WORKLOAD_PID"' not in script
        assert 'eval "exec $RUN_CMD" &' in script
        assert "exec spine -S $TASK_FILE_LIST" in script
        assert "scripts/check_spine_version.py" in script
        assert script.index("scripts/check_spine_version.py") < script.index(
            "exec spine -S $TASK_FILE_LIST"
        )
        syntax = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr

    @pytest.mark.parametrize(
        "template_name",
        [
            "job_template_s3df.sbatch",
            "job_template_nersc.sbatch",
            "job_template_anl.pbs",
        ],
    )
    def test_nontraining_templates_do_not_install_graceful_stop(
        self, mock_submitter, template_name
    ):
        """Nontraining wrappers should not expose graceful-stop machinery."""
        script = self._render_template(mock_submitter, template_name)

        assert "trap request_graceful_stop USR1" not in script
        assert "SPINE_PROD_GRACEFUL_STOP_FILE" not in script


class TestBatchClientSelection:
    """Tests for scheduler selection by Submitter."""

    def test_submitter_selects_pbs_client_for_anl(self, mock_submitter):
        """Test ANL profiles select the PBS client."""
        client = mock_submitter.batch.get_batch_client(
            {"site": "anl", "scheduler": "pbs"}
        )

        assert isinstance(client, PBSClient)
        assert mock_submitter.batch.get_template_name({"site": "anl"}) == (
            "job_template_anl.pbs"
        )


class TestPreloadDownloads:
    """Tests for submit-time download preloading."""

    def test_preload_downloads_invokes_helper(self, mock_submitter, workspace_root):
        """Test submitter invokes the preload helper with the requested config."""
        with patch("src.submitter.preload_downloads") as preload:
            mock_submitter.preload_downloads("infer/2x2/full_chain_240819.yaml")

        preload.assert_called_once_with(
            "infer/2x2/full_chain_240819.yaml", workspace_root
        )

    def test_preload_downloads_raises_on_failure(self, mock_submitter):
        """Test preload failures stop submission before jobs are queued."""
        with patch("src.submitter.preload_downloads", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                mock_submitter.preload_downloads("infer/2x2/full_chain_240819.yaml")

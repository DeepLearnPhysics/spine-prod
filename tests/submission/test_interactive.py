"""Submission integration tests for interactive behavior."""

import io
import os
from unittest.mock import patch

import pytest


class TestInteractiveExecution:
    """Tests for direct interactive execution."""

    def test_run_interactive_rejects_invalid_runtime(self, mock_submitter):
        with pytest.raises(ValueError, match="interactive_runtime"):
            mock_submitter.run_interactive("config.yaml", interactive_runtime="other")

    def test_run_interactive_rejects_in_place_output_override(self, mock_submitter):
        """In-place execution cannot also redirect the writer."""
        with pytest.raises(ValueError, match="--in-place cannot be combined"):
            mock_submitter.run_interactive(
                "config.yaml", in_place=True, output="result.h5"
            )

    def test_run_interactive_rejects_entry_count_with_fraction(self, mock_submitter):
        with pytest.raises(ValueError, match="cannot be combined"):
            mock_submitter.run_interactive(
                "config.yaml",
                num_entries=10,
                entry_fraction_range=(0.0, 0.5),
            )

    def test_run_interactive_requires_files_for_file_limit(self, mock_submitter):
        with pytest.raises(ValueError, match="requires --source/--source-list"):
            mock_submitter.run_interactive("config.yaml", num_files=1)

    def test_run_interactive_rejects_missing_and_invalid_task_inputs(
        self, mock_submitter, tmp_path
    ):
        with pytest.raises(ValueError, match="--files-per-task"):
            mock_submitter.run_interactive("config.yaml", files_per_task=2)
        with pytest.raises(ValueError, match="--task-id"):
            mock_submitter.run_interactive("config.yaml", task_id=2)

        missing = tmp_path / "missing.root"
        with pytest.raises(ValueError, match="No input files found"):
            mock_submitter.run_interactive("config.yaml", files=[str(missing)])

        input_file = tmp_path / "input.root"
        input_file.touch()
        with pytest.raises(ValueError, match="Task ID 2 out of range"):
            mock_submitter.run_interactive(
                "config.yaml",
                files=[str(input_file)],
                files_per_task=1,
                task_id=2,
            )

    @pytest.mark.parametrize("output_name", ["result.h5", "result_dir"])
    def test_run_interactive_creates_explicit_output_location(
        self, mock_submitter, tmp_path, output_name, capsys
    ):
        input_file = tmp_path / "input.root"
        input_file.touch()
        output = tmp_path / "nested" / output_name
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.submission.interactive.subprocess.run", return_value=completed),
        ):
            assert (
                mock_submitter.run_interactive(
                    "config.yaml",
                    files=[str(input_file)],
                    output=str(output),
                    interactive_runtime="local",
                )
                == 0
            )

        expected_directory = output.parent if output.suffix else output
        assert expected_directory.is_dir()
        assert f"Output: {output}" in capsys.readouterr().out

    def test_run_interactive_composes_latest_modifiers_and_preloads(
        self, mock_submitter, tmp_path
    ):
        latest = tmp_path / "latest.yaml"
        composite = tmp_path / "composite.yaml"
        latest.touch()
        composite.touch()
        completed = type("Completed", (), {"returncode": 1})()

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
            patch.object(mock_submitter, "preload_downloads") as preload,
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.submission.interactive.subprocess.run", return_value=completed),
        ):
            assert (
                mock_submitter.run_interactive(
                    "infer/icarus",
                    apply_mods=["data"],
                    preload=True,
                    interactive_runtime="local",
                )
                == 1
            )

        create_latest.assert_called_once()
        create_composite.assert_called_once_with(
            str(latest),
            ["data"],
            create_latest.call_args.args[1],
            detector="icarus",
            family="infer",
        )
        preload.assert_called_once_with(str(composite))

    def test_run_interactive_resolves_latest_conversion_bundle(
        self, mock_submitter, tmp_path
    ):
        """Conversion shorthand must select from the conversion family."""
        latest = tmp_path / "icarus_truth_latest_240812.yaml"
        latest.touch()
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch.object(
                mock_submitter.config_mgr,
                "create_latest_config",
                return_value=str(latest),
            ) as create_latest,
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.submission.interactive.subprocess.run", return_value=completed),
        ):
            assert (
                mock_submitter.run_interactive(
                    "convert/icarus", interactive_runtime="local"
                )
                == 0
            )

        args = create_latest.call_args
        assert args.args[0] == "icarus"
        assert args.kwargs == {"family": "convert"}

    def test_run_interactive_uses_bash_for_source(self, mock_submitter, tmp_path):
        """Test interactive mode uses bash so source commands work."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                flashmatch=True,
                world_size=1,
                minibatch_size=2,
                num_workers=4,
                iterations=10,
                set_overrides=["model.detect_anomaly=true"],
                expandable_segments=True,
                interactive_runtime="local",
            )

        assert exit_code == 0
        run.assert_called_once()
        _, kwargs = run.call_args
        assert kwargs["shell"] is True
        assert kwargs["executable"] == "/bin/bash"
        assert "export NUMBA_NUM_THREADS=64" in run.call_args.args[0]
        assert "expandable_segments:True" in run.call_args.args[0]
        assert "spine -S" in run.call_args.args[0]
        assert " -o " not in run.call_args.args[0]
        assert "--output-dir " in run.call_args.args[0]
        assert "--output-suffix full_chain_co_260316" in run.call_args.args[0]
        assert "--world-size 1" in run.call_args.args[0]
        assert "--minibatch-size 2" in run.call_args.args[0]
        assert "--num-workers 4" in run.call_args.args[0]
        assert "--iterations 10" in run.call_args.args[0]
        assert "--set model.detect_anomaly=true" in run.call_args.args[0]
        assert "FMATCH_BASEDIR" not in run.call_args.args[0]

    def test_run_interactive_uses_config_inputs_without_writer_overrides(
        self, mock_submitter
    ):
        """Test config-owned interactive runs do not override IO or output."""
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/train/generic/uresnet/train_240718.yaml",
                interactive_runtime="local",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "spine -S" not in command
        assert "--output-dir" not in command
        assert "--output-suffix" not in command

    def test_run_interactive_no_writer_is_deprecated_and_ignored(
        self, mock_submitter, tmp_path, capsys
    ):
        """Test deprecated no-writer leaves automatic output options enabled."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                no_writer=True,
                interactive_runtime="local",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "spine -S" in command
        assert "--output-dir" in command
        assert "--output-suffix full_chain_co_260316" in command
        warning = capsys.readouterr().out
        assert "--no-writer is deprecated and ignored" in warning

    def test_run_interactive_rejects_output_without_files(self, mock_submitter):
        """Test config-owned interactive runs reject inference output overrides."""
        with pytest.raises(
            ValueError,
            match="Cannot use --output/--output-suffix without --source/--source-list",
        ):
            mock_submitter.run_interactive(
                config="config/train/generic/uresnet/train_240718.yaml",
                output_suffix="custom_reco",
            )

    def test_run_interactive_ignores_no_writer_with_output_suffix(
        self, mock_submitter, tmp_path
    ):
        """Test an explicit output suffix wins over deprecated no-writer."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                no_writer=True,
                output_suffix="custom_reco",
                entry_filter="/filters/accepted.yaml",
                interactive_runtime="local",
            )

        assert exit_code == 0
        assert "--output-suffix custom_reco" in run.call_args.args[0]

    def test_run_interactive_sources_custom_software_paths(
        self, mock_submitter, tmp_path
    ):
        """Test custom software paths source their configure scripts before SPINE."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        larcv_root = tmp_path / "larcv"
        flashmatch_root = tmp_path / "flashmatch"
        larcv_root.mkdir()
        flashmatch_root.mkdir()
        (larcv_root / "configure.sh").write_text("export LARCV=1\n", encoding="utf-8")
        (flashmatch_root / "configure.sh").write_text(
            "export FMATCH=1\n", encoding="utf-8"
        )
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                larcv_path=str(larcv_root),
                flashmatch_path=str(flashmatch_root),
                interactive_runtime="local",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert f"source {larcv_root / 'configure.sh'}" in command
        assert f"source {flashmatch_root / 'configure.sh'}" in command
        assert "spine -S" in command

    def test_run_interactive_flashmatch_is_warned_noop(self, mock_submitter, tmp_path):
        """Test --flashmatch is accepted but does not change execution."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.submission.interactive.subprocess.run", return_value=completed),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                flashmatch=True,
                interactive_runtime="local",
            )

        assert exit_code == 0
        assert "no-op" in stderr.getvalue()

    def test_run_interactive_rejects_invalid_set_override(
        self, mock_submitter, tmp_path
    ):
        """Test invalid --set overrides fail before execution."""
        input_file = tmp_path / "input.root"
        input_file.touch()

        with pytest.raises(ValueError, match="Expected KEY=VALUE"):
            mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                set_overrides=["base.world_size"],
                interactive_runtime="local",
            )

    def test_run_interactive_rejects_quoted_set_override(
        self, mock_submitter, tmp_path
    ):
        """Test unsafe --set values fail before script rendering."""
        input_file = tmp_path / "input.root"
        input_file.touch()

        with pytest.raises(ValueError, match="Whitespace and quotes"):
            mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                set_overrides=["model.name='full chain'"],
                interactive_runtime="local",
            )

    def test_run_interactive_auto_falls_back_to_docker_container(
        self, mock_submitter, tmp_path
    ):
        """Test auto interactive mode falls back to SPINE_CONTAINER_TAG when spine is absent."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        completed = type("Completed", (), {"returncode": 0})()

        def fake_which(command):
            return "/usr/bin/docker" if command == "docker" else None

        with (
            patch.dict(
                os.environ,
                {
                    "SPINE_CONTAINER_PATH": str(tmp_path / "missing.sif"),
                    "SPINE_CONTAINER_TAG": "docker:ghcr.io/deeplearnphysics/spine:9.8.7",
                },
                clear=False,
            ),
            patch("src.submission.runtime.shutil.which", side_effect=fake_which),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                world_size=0,
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "export NUMBA_NUM_THREADS=64" in command
        assert "docker run --rm" in command
        assert "--platform linux/amd64" in command
        assert "ghcr.io/deeplearnphysics/spine:9.8.7" in command
        assert "docker:ghcr" not in command
        assert "spine -S" in command
        assert "--world-size 0" in command

    def test_default_container_version_comes_from_repo_file(
        self, mock_submitter, workspace_root
    ):
        """Test Python fallback reads the same default used by configure.sh."""
        expected = (
            (workspace_root / "DEFAULT_SPINE_VERSION")
            .read_text(encoding="utf-8")
            .strip()
        )

        with patch.dict(os.environ, {}, clear=True):
            assert mock_submitter.runtime.default_container_version() == expected
            assert mock_submitter.runtime.container_version() == expected
            assert mock_submitter.runtime.container_tag_for_cli().endswith(
                f":{expected}"
            )
            expected_path_version = (
                expected[1:] if expected.startswith("v") else expected
            )
            assert mock_submitter.runtime.default_container_path().endswith(
                f"spine_v{expected_path_version.replace('.', '-')}.sif"
            )

    def test_container_version_env_override_requires_configured_env(
        self, mock_submitter
    ):
        """Test direct version env overrides are ignored without configure.sh."""
        expected = mock_submitter.runtime.default_container_version()

        with patch.dict(os.environ, {"SPINE_CONTAINER_VERSION": "9.8.7"}, clear=True):
            assert mock_submitter.runtime.container_version() == expected

        with patch.dict(
            os.environ,
            {"SPINE_PROD_CONFIGURED": "1", "SPINE_CONTAINER_VERSION": "9.8.7"},
            clear=True,
        ):
            assert mock_submitter.runtime.container_version() == "9.8.7"

    def test_container_tag_strips_release_prefix(self, mock_submitter):
        """Test Git-style release versions map to unprefixed GHCR tags."""
        with patch.dict(
            os.environ,
            {"SPINE_PROD_CONFIGURED": "1", "SPINE_CONTAINER_VERSION": "v9.8.7"},
            clear=True,
        ):
            assert (
                mock_submitter.runtime.container_tag_for_cli()
                == "ghcr.io/deeplearnphysics/spine:9.8.7"
            )

    def test_run_interactive_container_uses_configured_sif_runtime(
        self, mock_submitter, tmp_path
    ):
        """Test interactive container mode honors a configured SIF runtime binary."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        container = tmp_path / "spine.sif"
        container.touch()
        completed = type("Completed", (), {"returncode": 0})()

        def fake_which(command):
            if command == "spine":
                return None
            if command == "/cvmfs/eaf.opensciencegrid.org/apptainer/bin/apptainer":
                return command
            return None

        with (
            patch.dict(
                os.environ,
                {
                    "SPINE_CONTAINER_PATH": str(container),
                    "SPINE_CONTAINER_RUNTIME_BIN": "/cvmfs/eaf.opensciencegrid.org/apptainer/bin/apptainer",
                },
                clear=False,
            ),
            patch("src.submission.runtime.shutil.which", side_effect=fake_which),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                interactive_runtime="container",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "/cvmfs/eaf.opensciencegrid.org/apptainer/bin/apptainer exec" in command
        assert str(container) in command

    def test_run_interactive_container_uses_configured_sif_runtime_args(
        self, mock_submitter, tmp_path
    ):
        """Test interactive container mode appends configured SIF runtime args."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        container = tmp_path / "spine.sif"
        container.touch()
        completed = type("Completed", (), {"returncode": 0})()

        def fake_which(command):
            if command == "spine":
                return None
            if command == "/cvmfs/eaf.opensciencegrid.org/apptainer/bin/apptainer":
                return command
            return None

        with (
            patch.dict(
                os.environ,
                {
                    "SPINE_CONTAINER_PATH": str(container),
                    "SPINE_CONTAINER_RUNTIME_BIN": "/cvmfs/eaf.opensciencegrid.org/apptainer/bin/apptainer",
                    "SPINE_CONTAINER_RUNTIME_ARGS": "--env LD_PRELOAD= --env LC_ALL=C.UTF-8",
                },
                clear=False,
            ),
            patch("src.submission.runtime.shutil.which", side_effect=fake_which),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                interactive_runtime="container",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "--env LD_PRELOAD=" in command
        assert "--env LC_ALL=C.UTF-8" in command
        assert str(container) in command

    def test_run_interactive_container_uses_bind_path_overrides(
        self, mock_submitter, tmp_path
    ):
        """Test interactive container mode appends configured bind path overrides."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        container = tmp_path / "spine.sif"
        container.touch()
        completed = type("Completed", (), {"returncode": 0})()

        def fake_which(command):
            if command == "spine":
                return None
            if (
                command
                == "/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer"
            ):
                return command
            return None

        with (
            patch.dict(
                os.environ,
                {
                    "SPINE_CONTAINER_PATH": str(container),
                    "SPINE_CONTAINER_RUNTIME_BIN": "/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer",
                },
                clear=False,
            ),
            patch("src.submission.runtime.shutil.which", side_effect=fake_which),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                interactive_runtime="container",
                bind_paths="/exp/dune",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "--bind" in command
        assert "/exp/dune" in command
        assert str(container) in command

    def test_run_interactive_local_uses_spine_path_run_py(
        self, mock_submitter, tmp_path
    ):
        """Test local interactive mode can target a SPINE checkout via --spine-path."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        spine_checkout = tmp_path / "spine"
        (spine_checkout / "bin").mkdir(parents=True)
        run_py = spine_checkout / "bin" / "run.py"
        run_py.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value=None),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                interactive_runtime="local",
                spine_path=str(spine_checkout),
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert str(run_py) in command
        assert "spine -S" not in command
        assert "python3 " in command

    def test_run_interactive_accepts_output_suffix(self, mock_submitter, tmp_path):
        """Test interactive mode can override the derived output suffix."""
        input_file = tmp_path / "input.root"
        input_file.touch()
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                output_suffix="custom_reco",
                entry_filter="/filters/accepted.yaml",
                interactive_runtime="local",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "--output-suffix custom_reco" in command
        assert "--entry-filter /filters/accepted.yaml" in command

    def test_run_interactive_in_place_omits_output_overrides(
        self, mock_submitter, tmp_path
    ):
        """Interactive cache extension should preserve config-owned routing."""
        input_file = tmp_path / "cache.h5"
        input_file.touch()
        completed = type("Completed", (), {"returncode": 0})()

        with (
            patch("src.submission.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch(
                "src.submission.interactive.subprocess.run", return_value=completed
            ) as run,
        ):
            exit_code = mock_submitter.run_interactive(
                config="config/cache/generic/grappa_shower_track/particle_graphs_240805.yaml",
                files=[str(input_file)],
                in_place=True,
                interactive_runtime="local",
            )

        assert exit_code == 0
        command = run.call_args.args[0]
        assert "--output " not in command
        assert "--output-dir" not in command
        assert "--output-suffix" not in command

    def test_run_interactive_local_rejects_invalid_spine_path(
        self, mock_submitter, tmp_path
    ):
        """Test invalid --spine-path values fail clearly."""
        input_file = tmp_path / "input.root"
        input_file.touch()

        with (pytest.raises(RuntimeError, match="--spine-path"),):
            mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                interactive_runtime="local",
                spine_path=str(tmp_path / "missing-checkout"),
            )

    def test_run_interactive_local_requires_spine_on_path(
        self, mock_submitter, tmp_path
    ):
        """Test local interactive mode fails clearly if spine is unavailable."""
        input_file = tmp_path / "input.root"
        input_file.touch()

        with (
            patch("src.submission.runtime.shutil.which", return_value=None),
            pytest.raises(
                RuntimeError, match="SPINE.*PATH|--spine-path|SPINE_LOCAL_PATH"
            ),
        ):
            mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                interactive_runtime="local",
            )

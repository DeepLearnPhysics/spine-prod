"""Tests for Submitter orchestration and integration behavior.

This module provides integration tests for the batch submission system,
testing actual orchestration paths in src.submitter.

Tests include:
1. **Modifier Discovery**: Tests _discover_modifiers() and list_modifiers()
2. **Version Resolution**: Tests _resolve_modifier_version()
3. **Latest Config Generation**: Tests _create_latest_config()
4. **Configuration Loading**: Tests load_profiles()

These tests exercise the actual Python code to provide meaningful coverage metrics.
"""

import io
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.client import PBSClient, SlurmClient
from src.run_manager import RunManager
from src.submitter import Submitter


class TestModifierDiscovery:
    """Tests for modifier discovery and resolution."""

    def test_discover_modifiers_icarus(self, mock_submitter, infer_root):
        """Test discovering modifiers for ICARUS detector."""
        # Test with a real ICARUS config that has modifiers
        icarus_configs = list((infer_root / "icarus").glob("full_chain_*.yaml"))
        if not icarus_configs:
            pytest.skip("No ICARUS configs found for testing")

        config_path = str(icarus_configs[0])
        modifiers = mock_submitter.config_mgr.discover_modifiers(config_path)

        # Should be a dict with modifier names as keys
        assert isinstance(modifiers, dict)

        # If there are modifiers, verify structure
        for mod_name, versions in modifiers.items():
            assert isinstance(mod_name, str)
            assert isinstance(versions, list)
            # Each version should be a Path object
            for version_path in versions:
                assert isinstance(version_path, Path)
                assert version_path.exists()

    def test_list_modifiers_public_api(self, mock_submitter, infer_root):
        """Test the public list_modifiers() API."""
        icarus_configs = list((infer_root / "icarus").glob("full_chain_*.yaml"))
        if not icarus_configs:
            pytest.skip("No ICARUS configs found for testing")

        config_path = str(icarus_configs[0])
        result = mock_submitter.list_modifiers(config_path)

        # Should return a dict with detector info and modifiers
        assert isinstance(result, dict)
        assert "config_name" in result
        assert "base_version" in result
        assert "modifiers" in result

        # Verify modifiers structure
        modifiers = result["modifiers"]
        for mod_name, mod_info in modifiers.items():
            assert "available" in mod_info
            assert "selected" in mod_info
            assert isinstance(mod_info["available"], list)


class TestVersionResolution:
    """Tests for modifier version resolution logic."""

    def test_resolve_modifier_version_explicit(self, mock_submitter):
        """Test resolving modifier with explicit version."""
        # Create mock version paths
        versions = [
            Path("/fake/mod_data_240719.yaml"),
            Path("/fake/mod_data_250115.yaml"),
            Path("/fake/mod_data_250625.yaml"),
        ]

        # Resolve with explicit version
        result = mock_submitter.config_mgr.resolve_modifier_version(
            mod_name="data",
            available_versions=versions,
            base_version="250625",
            explicit_version="250115",
        )

        assert result == Path("/fake/mod_data_250115.yaml")

    def test_resolve_modifier_version_latest(self, mock_submitter):
        """Test resolving modifier to latest version."""
        versions = [
            Path("/fake/mod_data_240719.yaml"),
            Path("/fake/mod_data_250115.yaml"),
            Path("/fake/mod_data_250625.yaml"),
        ]

        result = mock_submitter.config_mgr.resolve_modifier_version(
            mod_name="data",
            available_versions=versions,
            base_version="250625",
            explicit_version=None,
        )

        # Should pick version matching base or latest
        assert result == Path("/fake/mod_data_250625.yaml")

    def test_resolve_modifier_version_fallback(self, mock_submitter):
        """Test fallback to latest when base version not found."""
        versions = [
            Path("/fake/mod_data_240719.yaml"),
            Path("/fake/mod_data_250115.yaml"),
        ]

        result = mock_submitter.config_mgr.resolve_modifier_version(
            mod_name="data",
            available_versions=versions,
            base_version="250625",  # Not available
            explicit_version=None,
        )

        # Should fall back to latest available
        assert result == Path("/fake/mod_data_250115.yaml")


class TestLatestConfigGeneration:
    """Tests for dynamic 'latest' config generation."""

    def test_create_latest_config_icarus(self, mock_submitter, infer_root):
        """Test creating a 'latest' config for ICARUS."""
        icarus_dir = infer_root / "icarus"
        if not icarus_dir.exists():
            pytest.skip("ICARUS configs not found")

        # Test with icarus detector
        config_path = mock_submitter.config_mgr.create_latest_config(
            detector="icarus", job_dir=mock_submitter.jobs_dir
        )

        # Should create a config file in job_dir
        config_path_obj = Path(config_path)
        assert config_path_obj.exists()
        assert "latest" in config_path_obj.name
        assert config_path_obj.suffix == ".yaml"

        # Config should be valid YAML
        with open(config_path) as f:
            config = yaml.safe_load(f)
            assert "include" in config
            assert isinstance(config["include"], list)

    def test_create_latest_config_with_modifiers(self, mock_submitter, infer_root):
        """Test creating a 'latest' config."""
        icarus_dir = infer_root / "icarus"
        if not icarus_dir.exists():
            pytest.skip("ICARUS configs not found")

        config_path = mock_submitter.config_mgr.create_latest_config(
            detector="icarus", job_dir=mock_submitter.jobs_dir
        )

        config_path_obj = Path(config_path)
        assert config_path_obj.exists()

        # Config should be valid YAML with includes
        with open(config_path) as f:
            config = yaml.safe_load(f)
            includes = config.get("include", [])
            # Should have base, io, model, post components
            assert len(includes) >= 4


class TestProfileLoading:
    """Tests for profile configuration loading."""

    def test_load_profiles_default(self, workspace_root):
        """Test loading default profiles.yaml."""
        submitter = Submitter(basedir=workspace_root)

        # Should load profiles successfully
        assert submitter.profiles is not None
        assert "profiles" in submitter.profiles
        assert "detectors" in submitter.profiles

        # Verify some expected profiles exist
        profiles = submitter.profiles["profiles"]
        assert "s3df_ampere" in profiles or "gpu_large" in profiles

    def test_detector_config_paths(self, workspace_root):
        """Test that detector configs_dir paths are correct."""
        submitter = Submitter(basedir=workspace_root)

        detectors = submitter.profiles.get("detectors", {})
        for detector_name, detector_config in detectors.items():
            configs_dir = detector_config.get("configs_dir", "")
            # Should use infer/ not config/
            assert "infer/" in configs_dir
            assert "config/" not in configs_dir

    def test_full_node_gpu_profiles_request_complete_nodes(self, workspace_root):
        """Full-node profiles must expose every GPU and its CPU allocation."""
        profiles = Submitter(basedir=workspace_root).profiles["profiles"]
        expected = {
            "s3df_hopper_full": ("gpus", 4, "cpus_per_task", 224),
            "s3df_ampere_full": ("gpus", 4, "cpus_per_task", 112),
            "nersc_gpu_full": (
                "gpus_per_node",
                4,
                "cpus_per_task",
                128,
            ),
            "nersc_gpu_full_80gb": (
                "gpus_per_node",
                4,
                "cpus_per_task",
                128,
            ),
            "anl_polaris_capacity_full": (
                "gpus_per_node",
                4,
                "cpus_per_node",
                64,
            ),
            "anl_polaris_debug_full": (
                "gpus_per_node",
                4,
                "cpus_per_node",
                64,
            ),
        }

        for name, (gpu_key, gpus, cpu_key, cpus) in expected.items():
            assert profiles[name][gpu_key] == gpus
            assert profiles[name][cpu_key] == cpus
            if profiles[name]["site"] in ("s3df", "nersc"):
                assert profiles[name]["exclusive"] is True
            else:
                assert profiles[name]["place"] == "scatter:excl"

        # Preserve commands using the historical NERSC terminology.
        for old_name, new_name in (
            ("nersc_gpu_exclusive", "nersc_gpu_full"),
            ("nersc_gpu_exclusive_80gb", "nersc_gpu_full_80gb"),
        ):
            for key in ("gpus_per_node", "cpus_per_task", "constraint", "gpu_mem"):
                assert profiles[old_name][key] == profiles[new_name][key]

    def test_polaris_profiles_scale_cpus_with_gpus(self, workspace_root):
        """Polaris profiles should request 16 logical CPUs per A100."""
        profiles = Submitter(basedir=workspace_root).profiles["profiles"]

        for queue in ("capacity", "debug"):
            single = profiles[f"anl_polaris_{queue}"]
            full = profiles[f"anl_polaris_{queue}_full"]
            assert single["gpus_per_node"] == 1
            assert single["cpus_per_node"] == 16
            assert single["cpus_per_task"] == 16
            assert full["gpus_per_node"] == 4
            assert full["cpus_per_node"] == 64
            assert full["cpus_per_task"] == 64


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_basedir_from_env(self, workspace_root):
        """Test SPINE_PROD_BASEDIR environment variable."""
        with patch.dict(
            os.environ, {"SPINE_PROD_BASEDIR": str(workspace_root)}, clear=False
        ):
            submitter = Submitter()
            assert submitter.basedir == workspace_root

    def test_basedir_explicit_override(self, workspace_root, tmp_path):
        """Test explicit basedir overrides environment."""
        with patch.dict(os.environ, {"SPINE_PROD_BASEDIR": str(tmp_path)}, clear=False):
            submitter = Submitter(basedir=workspace_root)
            assert submitter.basedir == workspace_root


class TestConfigPathHandling:
    """Tests for configuration path resolution."""

    def test_detect_latest_config(self, mock_submitter):
        """Test detecting 'latest' config shorthand."""
        test_cases = [
            "icarus/latest",
            "icarus/latest.yaml",
            "infer/icarus/latest",
            "infer/icarus",
            "infer/dune10kt-1x2x6",
        ]

        for config_path in test_cases:
            is_latest, config_name = mock_submitter.batch.classify_config_request(
                config_path
            )
            assert is_latest is True
            assert config_name == "latest"

    def test_absolute_vs_relative_paths(self, workspace_root):
        """Test handling of absolute vs relative config paths."""
        # Both should work

        rel_path = "config/infer/icarus/full_chain_co_250625.yaml"
        abs_path = workspace_root / rel_path

        # Check paths exist
        assert abs_path.exists(), f"Config not found: {abs_path}"


class TestDetectorDetection:
    """Tests for detector auto-detection."""

    def test_detect_detector_icarus(self, mock_submitter):
        """Test auto-detecting ICARUS from config path."""
        result = mock_submitter.config_mgr.detect_detector("infer/icarus/latest.yaml")
        assert result == "icarus"

    def test_detect_detector_sbnd(self, mock_submitter):
        """Test auto-detecting SBND from config path."""
        result = mock_submitter.config_mgr.detect_detector(
            "infer/sbnd/full_chain_240720.yaml"
        )
        assert result == "sbnd"

    def test_detect_detector_2x2(self, mock_submitter):
        """Test auto-detecting 2x2 from config path."""
        result = mock_submitter.config_mgr.detect_detector("infer/2x2/latest")
        assert result == "2x2"

    def test_detect_detector_generic(self, mock_submitter):
        """Test fallback for unknown detectors."""
        result = mock_submitter.config_mgr.detect_detector("some/random/config.yaml")
        assert result == "unknown_detector"

    def test_detect_detector_dune10kt_1x2x6(self, mock_submitter):
        """Test auto-detecting DUNE10kt-1x2x6 from config path."""
        result = mock_submitter.config_mgr.detect_detector("infer/dune10kt-1x2x6")
        assert result == "dune10kt-1x2x6"


class TestVersionExtraction:
    """Tests for version extraction from config names."""

    def test_extract_version_yymmdd(self, mock_submitter):
        """Test extracting YYMMDD version format."""
        config_path = Path("full_chain_co_250625.yaml")
        version = mock_submitter.config_mgr.extract_version(config_path)
        assert version == "250625"

    def test_extract_version_data_modifier(self, mock_submitter):
        """Test extracting version from data modifier."""
        config_path = Path("mod_data_250115.yaml")
        version = mock_submitter.config_mgr.extract_version(config_path)
        assert version == "250115"

    def test_extract_version_no_version(self, mock_submitter):
        """Test handling files without version."""
        config_path = Path("base_common.yaml")
        version = mock_submitter.config_mgr.extract_version(config_path)
        assert version is None

    def test_extract_version_legacy_format(self, mock_submitter):
        """Test extracting version from legacy format."""
        config_path = Path("full_chain_240719.yaml")
        version = mock_submitter.config_mgr.extract_version(config_path)
        assert version == "240719"


class TestFileHandling:
    """Tests for file parsing and handling."""

    def test_parse_files_single_file(self, mock_submitter, tmp_path):
        """Test parsing single file."""
        test_file = tmp_path / "test.root"
        test_file.touch()

        files = mock_submitter.file_handler.parse_files([str(test_file)])
        assert len(files) == 1
        assert files[0] == str(test_file)

    def test_parse_files_glob_pattern(self, mock_submitter, tmp_path):
        """Test parsing glob patterns."""
        # Create test files
        for i in range(3):
            (tmp_path / f"data_{i}.root").touch()

        pattern = str(tmp_path / "data_*.root")
        files = mock_submitter.file_handler.parse_files([pattern])
        assert len(files) == 3
        assert all(f.endswith(".root") for f in files)

    def test_parse_files_from_list(self, mock_submitter, tmp_path):
        """Test parsing files from a source list file."""
        # Create test files
        test_files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.root"
            f.touch()
            test_files.append(str(f))

        # Create file list
        file_list = tmp_path / "files.txt"
        file_list.write_text("\n".join(test_files))

        files = mock_submitter.file_handler.parse_files(
            [str(file_list)], source_type="source_list"
        )
        assert len(files) == 3
        assert all(f.endswith(".root") for f in files)

    def test_parse_files_direct_paths(self, mock_submitter, tmp_path):
        """Test parsing direct file paths."""
        # Create test files
        files_to_create = []
        for i in range(3):
            f = tmp_path / f"file_{i}.root"
            f.touch()
            files_to_create.append(str(f))

        files = mock_submitter.file_handler.parse_files(
            files_to_create, source_type="source"
        )
        assert len(files) == 3
        assert all(f.endswith(".root") for f in files)


class TestFileChunking:
    """Tests for file chunking logic."""

    def test_chunk_files_basic(self, mock_submitter):
        """Test basic file chunking."""
        files = [f"file_{i}.root" for i in range(10)]

        chunks = mock_submitter.file_handler.chunk_files(
            files, max_array_size=99, files_per_task=2
        )
        # 10 files / 2 per task = 5 groups, all fit in one chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 5  # 5 groups

    def test_chunk_files_multiple_chunks(self, mock_submitter):
        """Test chunking with array size limit."""
        # Create enough files to exceed max_array_size
        files = [f"file_{i}.root" for i in range(50)]

        chunks = mock_submitter.file_handler.chunk_files(
            files, max_array_size=10, files_per_task=1
        )
        # 50 files / 1 per task = 50 groups, split into chunks of 10
        assert len(chunks) == 5
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_chunk_files_per_task(self, mock_submitter):
        """Test multiple files per task."""
        files = [f"file_{i}.root" for i in range(9)]

        chunks = mock_submitter.file_handler.chunk_files(
            files, max_array_size=99, files_per_task=3
        )
        # 9 files / 3 per task = 3 groups
        assert len(chunks) == 1
        assert len(chunks[0]) == 3
        # Each group should have 3 files as a list
        assert len(chunks[0][0]) == 3
        assert isinstance(chunks[0][0], list)

    def test_resolve_files_per_task_defaults_to_all_files(self, mock_submitter):
        """Test omitted splitting flags collapse explicit inputs into one task."""
        assert mock_submitter.batch.resolve_files_per_task(9) == 9

    def test_resolve_files_per_task_uses_ntasks_for_even_split(self, mock_submitter):
        """Test ntasks alone distributes files roughly evenly across tasks."""
        assert mock_submitter.batch.resolve_files_per_task(10, ntasks=3) == 4

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"ntasks": 0}, "--ntasks must be >= 1"),
            ({"files_per_task": 0}, "--files-per-task must be >= 1"),
        ],
    )
    def test_resolve_files_per_task_rejects_nonpositive_values(
        self, mock_submitter, kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            mock_submitter.batch.resolve_files_per_task(10, **kwargs)

    def test_parse_named_sources_preserves_aligned_file_order(
        self, mock_submitter, tmp_path
    ):
        """Composite source lists resolve into aligned target sequences."""
        larcv = [tmp_path / f"input_{index}.root" for index in range(2)]
        hdf5 = [tmp_path / f"input_{index}_cache.h5" for index in range(2)]
        for path in [*larcv, *hdf5]:
            path.touch()
        larcv_list = tmp_path / "larcv.txt"
        hdf5_list = tmp_path / "hdf5.txt"
        larcv_list.write_text("\n".join(map(str, larcv)), encoding="utf-8")
        hdf5_list.write_text("\n".join(map(str, hdf5)), encoding="utf-8")

        resolved = mock_submitter.file_handler.parse_named_sources(
            {
                "larcv": {"source_list": str(larcv_list)},
                "hdf5": {"source_list": str(hdf5_list)},
            }
        )
        assert resolved == {
            "larcv": [str(path) for path in larcv],
            "hdf5": [str(path) for path in hdf5],
        }

    @pytest.mark.parametrize(
        ("sources", "message"),
        [
            ({"larcv": {}}, "must specify exactly one"),
            ({"larcv": {"source": []}}, "contains no input files"),
        ],
    )
    def test_parse_named_sources_rejects_invalid_targets(
        self, mock_submitter, sources, message
    ):
        with pytest.raises(ValueError, match=message):
            mock_submitter.file_handler.parse_named_sources(sources)

    def test_parse_named_sources_rejects_unaligned_counts(
        self, mock_submitter, tmp_path
    ):
        """Every mixed source target must contribute one file per pair."""
        paths = [tmp_path / f"input_{index}.root" for index in range(2)]
        for path in paths:
            path.touch()
        with pytest.raises(ValueError, match="aligned file counts"):
            mock_submitter.file_handler.parse_named_sources(
                {
                    "larcv": {"source": [str(path) for path in paths]},
                    "hdf5": {"source": str(paths[0])},
                }
            )


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
            "convert/icarus/latest"
        ) == (True, "latest")
        assert mock_submitter.batch.classify_config_request(
            str(workspace_root / "config" / "convert" / "icarus")
        ) == (True, "latest")
        assert mock_submitter.batch.classify_config_request(
            "config/convert/icarus"
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

        with patch("src.runtime.shutil.which", return_value="/usr/bin/spine-filter"):
            assert mock_submitter.runtime.resolve_spine_filter_command() == (
                "/usr/bin/spine-filter",
                None,
            )
        with patch("src.runtime.shutil.which", return_value=None):
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

        with patch("src.runtime.shutil.which", return_value="/usr/bin/spine-cache"):
            assert mock_submitter.runtime.resolve_spine_cache_command() == (
                "/usr/bin/spine-cache",
                None,
            )
        with patch("src.runtime.shutil.which", return_value=None):
            assert mock_submitter.runtime.resolve_spine_cache_command() == (None, None)

    def test_submit_filter_scan_builds_persistent_scheduler_job(
        self, mock_submitter, tmp_path
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
                force=True,
                dependency="afterok:40",
            )

        assert job_ids == ["41"]
        attempt = (run_dir / "latest").resolve()
        script = (attempt / "submit.sbatch").read_text(encoding="utf-8")
        assert "spine-filter scan" in script
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
        assert f"--input-dir {input_dir}" in script
        assert f"--output-dir {output_dir}" in script

    def test_report_config_materialization_supports_legacy_pyyaml(self, tmp_path):
        """Generated report recipes must not require PyYAML's sort_keys option."""
        from src.report import ReportRunner

        source = tmp_path / "source.yaml"
        source.write_text("metadata: {}\nmetrics: {}\n", encoding="utf-8")
        attempt = tmp_path / "attempt"
        attempt.mkdir()
        safe_dump = yaml.safe_dump

        def legacy_safe_dump(document, **kwargs):
            if "sort_keys" in kwargs:
                raise TypeError("dump_all() got an unexpected keyword argument")
            return safe_dump(document, **kwargs)

        with patch("src.report.yaml.safe_dump", side_effect=legacy_safe_dump):
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
            "src.runtime.shutil.which", side_effect=find_apptainer
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
            patch("src.runtime.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="no executable was found"),
        ):
            mock_submitter.runtime.sif_runtime_executable()

        with (
            patch.dict(
                os.environ,
                {"SPINE_CONTAINER_PATH": str(tmp_path / "missing.sif")},
                clear=False,
            ),
            patch("src.runtime.shutil.which", return_value=None),
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
            patch("src.runtime.Path.exists", report_cvmfs_exists),
            patch("src.runtime.shutil.which", side_effect=find_docker),
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
            patch("src.runtime.shutil.which", side_effect=find_docker),
        ):
            command = mock_submitter.runtime.build_interactive_container_command(
                "spine -c config.yaml", False
            )

        assert "/usr/bin/docker run" in command
        assert "apptainer" not in command


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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed),
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed),
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
            str(latest), ["data"], create_latest.call_args.args[1], detector="icarus"
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed),
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
                interactive_runtime="local",
            )

        assert exit_code == 0
        run.assert_called_once()
        _, kwargs = run.call_args
        assert kwargs["shell"] is True
        assert kwargs["executable"] == "/bin/bash"
        assert "export NUMBA_NUM_THREADS=64" in run.call_args.args[0]
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed),
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
            patch("src.runtime.shutil.which", side_effect=fake_which),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", side_effect=fake_which),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", side_effect=fake_which),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", side_effect=fake_which),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value=None),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value="/usr/bin/spine"),
            patch("src.interactive.subprocess.run", return_value=completed) as run,
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
            patch("src.runtime.shutil.which", return_value=None),
            pytest.raises(
                RuntimeError, match="SPINE.*PATH|--spine-path|SPINE_LOCAL_PATH"
            ),
        ):
            mock_submitter.run_interactive(
                config="config/infer/sbnd/full_chain_co_260316.yaml",
                files=[str(input_file)],
                interactive_runtime="local",
            )


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
            patch("src.runtime.shutil.which", return_value=None),
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
            str(latest), ["data"], job_dir, detector="icarus"
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
            ({"stage": "unknown"}, "stage must be one of"),
            ({"stage": "train"}, "--run-dir is required"),
            ({"resume": True}, "valid only for training"),
            ({"validation_name": "data"}, "valid only for validation"),
            ({"val_entry_filter": "/filters/val.yaml"}, "valid only for training"),
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
                entry_fraction_range=(0.0, 1.0),
                val_entry_fraction_range=(0.0, 0.5),
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
        assert "--entry-fraction-range 0.0 1.0" in script
        assert "--val-entry-fraction-range 0.0 0.5" in script
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
        assert metadata["entry_fraction_range"] == [0.0, 1.0]
        assert metadata["val_entry_fraction_range"] == [0.0, 0.5]
        assert metadata["entry_filter"] == "/filters/train.yaml"
        assert metadata["val_entry_filter"] == "/filters/validation.yaml"
        assert metadata["graceful_stop_file"] == str(marker)

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
            patch("src.batch.uuid.uuid4", return_value=Mock(hex="attempt-id")),
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

    def test_submit_validation_selects_missing_checkpoints_and_noops(
        self, mock_submitter, tmp_path, capsys
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
                tensorboard=True,
            ) == ["val-1"]
            submit.assert_called_once()

        submission = run_dir / "latest"
        assert submission.is_dir()
        assert (submission / "weights.txt").read_text().splitlines() == [str(second)]
        script = (submission / "submit.sbatch").read_text(encoding="utf-8")
        assert f"--log-dir {run_dir}" in script
        assert f"--weight-list {(submission / 'weights.txt').resolve()}" in script
        assert "model.weight_path=null" in script
        assert (
            f"--tensorboard --tensorboard-dir {run_dir}/tensorboard/validation"
            in script
        )

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
                config="config/infer/dune10kt-1x2x6/full_chain_260510.yaml",
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

        manifest = scripts[0].parent / "inputs.txt"
        assert manifest.read_text(encoding="utf-8").strip().splitlines() == input_files
        assert (scripts[0].parent / "output").is_dir()
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


class TestCompositeConfig:
    """Tests for composite config creation."""

    def test_create_composite_config_basic(self, mock_submitter, tmp_path, infer_root):
        """Test creating a composite config."""
        # Use a real ICARUS config
        icarus_configs = list((infer_root / "icarus").glob("full_chain_*.yaml"))
        if not icarus_configs:
            pytest.skip("No ICARUS configs found")

        base_config = str(icarus_configs[0])
        composite_path = mock_submitter.config_mgr.create_composite_config(
            base_config=base_config,
            modifiers=[],
            job_dir=tmp_path,
            detector="icarus",
        )

        assert Path(composite_path).exists()
        assert "composite" in composite_path

        # Verify it's valid YAML
        with open(composite_path) as f:
            config = yaml.safe_load(f)
            assert "include" in config

    def test_create_composite_with_modifiers(
        self, mock_submitter, tmp_path, infer_root
    ):
        """Test creating composite config with modifiers."""
        # We need a config that's in the same directory as modifiers
        # The top-level full_chain configs don't have modifiers in their directory
        pytest.skip(
            "Composite config with modifiers requires special directory structure"
        )

    def test_create_composite_from_latest_avoids_duplicate_composite_suffix(
        self, mock_submitter, tmp_path
    ):
        """Test composing a generated latest config keeps one composite suffix."""
        latest_config = mock_submitter.config_mgr.create_latest_config(
            "nd-lar", tmp_path
        )

        composite_path = mock_submitter.config_mgr.create_composite_config(
            base_config=latest_config,
            modifiers=["single"],
            job_dir=tmp_path,
            detector="nd-lar",
        )

        latest_stem = Path(latest_config).stem
        if latest_stem.endswith("_composite"):
            latest_stem = latest_stem[: -len("_composite")]
        expected_name = f"{latest_stem}_single_composite.yaml"

        assert Path(composite_path).exists()
        assert Path(composite_path).name == expected_name
        assert "_composite_single_composite" not in Path(composite_path).name


class TestProfileSelection:
    """Tests for profile selection and validation."""

    def test_get_profile_explicit(self, mock_submitter):
        """Test getting an explicit profile."""
        profile = mock_submitter.config_mgr.get_profile("s3df_ampere")
        assert profile is not None
        assert "partition" in profile or "nodes" in profile

    def test_get_profile_with_detector(self, mock_submitter):
        """Test getting profile with detector defaults."""
        profile = mock_submitter.config_mgr.get_profile("auto", detector="icarus")
        assert profile is not None

    def test_get_profile_auto_fallback(self, mock_submitter):
        """Test auto profile selection with fallback."""
        # Should fall back to default if available
        profile = mock_submitter.config_mgr.get_profile("auto", detector="generic")
        assert profile is not None


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

    def test_submit_pipeline_selects_sparse_stages_and_contracts_dependencies(
        self, mock_submitter, tmp_path, capsys
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
                select_stages=[
                    "particle_validation",
                    "cache_validation",
                    "particle_train",
                    "cache_train",
                ],
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
        assert "Not selected: source, train" in output

    def test_report_rejects_non_mapping_configuration(self, tmp_path):
        """Report recipes must be mappings before provenance is injected."""
        from src.report import ReportRunner

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

    def test_submit_report_adds_checkout_bind_account_and_dependency(
        self, mock_submitter, tmp_path, capsys
    ):
        """Report jobs derive omitted site resources and bind a source checkout."""
        source = tmp_path / "report.yaml"
        source.write_text("metrics: {}\n", encoding="utf-8")
        profile = {
            "site": "s3df",
            "scheduler": "slurm",
            "description": "Slurm test",
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
            patch.object(mock_submitter.batch_client, "submit", return_value="42"),
        ):
            assert mock_submitter.submit_report(
                config=str(source),
                input_dir=str(tmp_path / "input"),
                output_dir=str(tmp_path / "output"),
                run_dir=str(tmp_path / "run"),
                dependency="afterok:41",
            ) == ["42"]

        assert profile["bind_paths"] == "/sdf/,/checkout"
        assert profile["account"]
        assert "Dependency: afterok:41" in capsys.readouterr().out

    def test_resolve_spine_report_command_path_forms_and_fallbacks(
        self, mock_submitter, tmp_path
    ):
        """Report resolution supports executable-like paths and PATH fallback."""
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

        with patch("src.runtime.shutil.which", return_value="/usr/bin/spine-report"):
            assert mock_submitter.runtime.resolve_spine_report_command() == (
                "/usr/bin/spine-report",
                None,
            )

        with patch("src.runtime.shutil.which", return_value=None):
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

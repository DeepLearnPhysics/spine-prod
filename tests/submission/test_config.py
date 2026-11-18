"""Submission integration tests for config behavior."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

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
            configs_dir = detector_config.get("configs_dir")
            if configs_dir is None:
                # Conversion-only detectors do not need an inference tree.
                continue

            # Inference directories use SPINE_CONFIG_PATH-relative paths.
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
            "infer/dune-hd-10kt-1x2x6",
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

    def test_detect_detector_dune_hd_10kt_1x2x6(self, mock_submitter):
        """Test auto-detecting DUNE-HD 10 kt module 1x2x6."""
        result = mock_submitter.config_mgr.detect_detector("infer/dune-hd-10kt-1x2x6")
        assert result == "dune-hd-10kt-1x2x6"

    def test_detect_detector_accepts_deprecated_dune_hd_alias(self, mock_submitter):
        """Test that the former DUNE-HD name resolves with a visible warning."""
        with pytest.warns(FutureWarning, match="dune10kt-1x2x6.*deprecated"):
            result = mock_submitter.config_mgr.detect_detector("infer/dune10kt-1x2x6")
        assert result == "dune-hd-10kt-1x2x6"


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

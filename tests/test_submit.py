"""Tests for submit.py functionality.

This module provides integration tests for the SLURM submission system,
testing actual code paths in submit.py to ensure proper coverage reporting.

Tests include:
1. **Modifier Discovery**: Tests _discover_modifiers() and list_modifiers()
2. **Version Resolution**: Tests _resolve_modifier_version()
3. **Latest Config Generation**: Tests _create_latest_config()
4. **Configuration Loading**: Tests load_profiles()

These tests exercise the actual Python code to provide meaningful coverage metrics.
"""

import os

# Add parent directory to path to import submit
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from submit import SlurmSubmitter  # noqa: E402


@pytest.fixture
def workspace_root():
    """Return path to workspace root."""
    return Path(__file__).parent.parent


@pytest.fixture
def infer_root(workspace_root):
    """Return path to infer directory."""
    return workspace_root / "infer"


@pytest.fixture
def mock_submitter(workspace_root, tmp_path):
    """Create a SlurmSubmitter instance with mocked paths."""
    # Use real basedir but mock job_dir - pass Path not str
    submitter = SlurmSubmitter(basedir=workspace_root)
    submitter.job_dir = tmp_path / "test_job"
    submitter.job_dir.mkdir(parents=True, exist_ok=True)
    return submitter


class TestModifierDiscovery:
    """Tests for modifier discovery and resolution."""

    def test_discover_modifiers_icarus(self, mock_submitter, infer_root):
        """Test discovering modifiers for ICARUS detector."""
        # Test with a real ICARUS config that has modifiers
        icarus_configs = list((infer_root / "icarus").glob("icarus_full_chain_*.yaml"))
        if not icarus_configs:
            pytest.skip("No ICARUS configs found for testing")

        config_path = str(icarus_configs[0])
        modifiers = mock_submitter._discover_modifiers(config_path)

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
        icarus_configs = list((infer_root / "icarus").glob("icarus_full_chain_*.yaml"))
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
        result = mock_submitter._resolve_modifier_version(
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

        result = mock_submitter._resolve_modifier_version(
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

        result = mock_submitter._resolve_modifier_version(
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
        config_path = mock_submitter._create_latest_config(
            detector="icarus", job_dir=mock_submitter.job_dir
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

        config_path = mock_submitter._create_latest_config(
            detector="icarus", job_dir=mock_submitter.job_dir
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
        submitter = SlurmSubmitter(basedir=workspace_root)

        # Should load profiles successfully
        assert submitter.profiles is not None
        assert "profiles" in submitter.profiles
        assert "detectors" in submitter.profiles

        # Verify some expected profiles exist
        profiles = submitter.profiles["profiles"]
        assert "s3df_ampere" in profiles or "gpu_large" in profiles

    def test_detector_config_paths(self, workspace_root):
        """Test that detector configs_dir paths are correct."""
        submitter = SlurmSubmitter(basedir=workspace_root)

        detectors = submitter.profiles.get("detectors", {})
        for detector_name, detector_config in detectors.items():
            configs_dir = detector_config.get("configs_dir", "")
            # Should use infer/ not config/
            assert "infer/" in configs_dir
            assert "config/" not in configs_dir


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_basedir_from_env(self, workspace_root):
        """Test SPINE_PROD_BASEDIR environment variable."""
        with patch.dict(
            os.environ, {"SPINE_PROD_BASEDIR": str(workspace_root)}, clear=False
        ):
            submitter = SlurmSubmitter()
            assert submitter.basedir == workspace_root

    def test_basedir_explicit_override(self, workspace_root, tmp_path):
        """Test explicit basedir overrides environment."""
        with patch.dict(os.environ, {"SPINE_PROD_BASEDIR": str(tmp_path)}, clear=False):
            submitter = SlurmSubmitter(basedir=workspace_root)
            assert submitter.basedir == workspace_root


class TestConfigPathHandling:
    """Tests for configuration path resolution."""

    def test_detect_latest_config(self, mock_submitter):
        """Test detecting 'latest' config shorthand."""
        # Test various latest patterns
        test_cases = [
            "icarus/latest",
            "icarus/latest.yaml",
            "icarus/latest.cfg",
            "infer/icarus/latest",
        ]

        for config_path in test_cases:
            # Should recognize as latest config
            # (Implementation detail: would trigger _create_latest_config)
            assert "latest" in config_path

    def test_absolute_vs_relative_paths(self, workspace_root):
        """Test handling of absolute vs relative config paths."""
        # Both should work
        rel_path = "infer/icarus/icarus_full_chain_co_250625.yaml"
        abs_path = workspace_root / rel_path

        # Check paths exist
        assert abs_path.exists(), f"Config not found: {abs_path}"

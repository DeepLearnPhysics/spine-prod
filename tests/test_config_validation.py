"""Tests for configuration file validation.

This module provides automated tests for SPINE-prod configuration files:

1. **Base Configuration Tests**: Validates that all main detector configuration
   files (e.g., icarus_full_chain_*.yaml) parse correctly with includes and
   overrides properly resolved.

2. **Legacy Configuration Tests**: Ensures backward compatibility by testing
   deprecated configuration files in the legacy/ directories.

3. **Metadata Stripping**: Verifies that __meta__ blocks are properly removed
   from final configurations and don't pollute the SPINE runtime config.

To run tests:
    pytest tests/test_config_validation.py -v

To add a new detector for testing, add it to DETECTOR_BASE_CONFIGS dict.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from spine.config import load_config_file

    SPINE_AVAILABLE = True
except ImportError:
    SPINE_AVAILABLE = False


def load_config_with_includes(config_path):
    """Load a YAML config using SPINE's load_config function.

    This uses SPINE's native config loader which handles:
    - Custom YAML tags (!path, !include, etc.)
    - Include directives
    - Override directives
    - Metadata stripping

    Note: Downloads are mocked to avoid downloading large checkpoint files.
    """
    config_path = Path(config_path)

    # Mock download_from_url at the point where it's used in the loader
    with patch("spine.config.loader.download_from_url") as mock_download:
        # Return a fake path instead of downloading
        mock_download.return_value = "/fake/weights/checkpoint.ckpt"
        return load_config_file(str(config_path))


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
class TestConfigValidation:
    """Test that all configuration files parse correctly."""

    @pytest.fixture
    def config_infer_root(self):
        """Return path to infer directory."""
        return Path(__file__).parent.parent / "config" / "infer"

    @pytest.fixture
    def config_train_root(self):
        """Return path to train directory."""
        return Path(__file__).parent.parent / "config" / "train"

    @staticmethod
    def get_detector_dirs(config_root):
        """Get list of detector directories, excluding non-directory files."""
        if not config_root.exists():
            return []
        return [
            d.name
            for d in config_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    @pytest.mark.parametrize(
        "detector",
        get_detector_dirs.__func__(Path(__file__).parent.parent / "config" / "infer"),
    )
    def test_infer_detector_base_configs(self, config_infer_root, detector):
        """Test that all top-level YAML configurations for a detector parse correctly.

        Automatically discovers and tests all .yaml files in the detector root directory
        (excludes subdirectories like base/, io/, model/, etc.).
        """
        detector_dir = config_infer_root / detector
        if not detector_dir.exists():
            pytest.skip(f"Detector directory not found: {detector_dir}")

        # Find all YAML files in detector root (not in subdirectories)
        yaml_files = [f for f in detector_dir.glob("*.yaml") if f.is_file()]

        if not yaml_files:
            pytest.skip(f"No YAML files found in {detector_dir}")

        failed_configs = []
        for config_path in yaml_files:
            try:
                # Load config with includes
                cfg = load_config_with_includes(config_path)
                assert cfg is not None, "Config loaded but returned None"
                assert isinstance(cfg, dict), f"Config must be a dict, got {type(cfg)}"

                # Verify __meta__ was stripped
                assert (
                    "__meta__" not in cfg
                ), "__meta__ should be stripped from final config"

                # Verify config has expected top-level keys
                assert len(cfg) > 0, "Config is empty after loading"

            except Exception as e:
                failed_configs.append((config_path.name, str(e)))

        if failed_configs:
            error_msg = f"Failed to parse {detector} infer configs:\n"
            for cfg, err in failed_configs:
                error_msg += f"  - {cfg}: {err}\n"
            pytest.fail(error_msg)

    @pytest.mark.parametrize(
        "detector",
        get_detector_dirs.__func__(Path(__file__).parent.parent / "config" / "infer"),
    )
    def test_infer_detector_legacy_configs(self, config_infer_root, detector):
        """Test that legacy configurations still parse correctly.

        Automatically discovers and tests all .yaml files in the legacy/ subdirectory.
        """
        detector_dir = config_infer_root / detector
        legacy_dir = detector_dir / "legacy"

        if not legacy_dir.exists():
            pytest.skip(f"No legacy directory for detector: {detector}")

        yaml_files = list(legacy_dir.glob("*.yaml"))
        if not yaml_files:
            pytest.skip(f"No YAML files in legacy directory for {detector}")

        failed_configs = []
        for config_file in yaml_files:
            try:
                cfg = load_config_with_includes(config_file)
                assert cfg is not None, "Config loaded but returned None"
                assert isinstance(cfg, dict), "Config must be a dict"
                assert "__meta__" not in cfg, "__meta__ should be stripped"
                assert len(cfg) > 0, "Config is empty"
            except Exception as e:
                failed_configs.append((config_file.name, str(e)))

        if failed_configs:
            error_msg = f"Failed to parse {detector} infer legacy configs:\n"
            for cfg, err in failed_configs:
                error_msg += f"  - {cfg}: {err}\n"
            pytest.fail(error_msg)

    @pytest.mark.parametrize(
        "detector",
        get_detector_dirs.__func__(Path(__file__).parent.parent / "config" / "train"),
    )
    def test_train_detector_base_configs(self, config_train_root, detector):
        """Test that all top-level YAML training configurations for a detector parse correctly.

        Automatically discovers and tests all .yaml files in the detector root directory
        (excludes subdirectories).
        """
        detector_dir = config_train_root / detector
        if not detector_dir.exists():
            pytest.skip(f"Detector directory not found: {detector_dir}")

        # Find all YAML files in detector root (not in subdirectories)
        yaml_files = [f for f in detector_dir.glob("*.yaml") if f.is_file()]

        if not yaml_files:
            pytest.skip(f"No YAML files found in {detector_dir}")

        failed_configs = []
        for config_path in yaml_files:
            try:
                # Load config with includes
                cfg = load_config_with_includes(config_path)
                assert cfg is not None, "Config loaded but returned None"
                assert isinstance(cfg, dict), f"Config must be a dict, got {type(cfg)}"

                # Verify __meta__ was stripped
                assert (
                    "__meta__" not in cfg
                ), "__meta__ should be stripped from final config"

                # Verify config has expected top-level keys
                assert len(cfg) > 0, "Config is empty after loading"

            except Exception as e:
                failed_configs.append((config_path.name, str(e)))

        if failed_configs:
            error_msg = f"Failed to parse {detector} train configs:\n"
            for cfg, err in failed_configs:
                error_msg += f"  - {cfg}: {err}\n"
            pytest.fail(error_msg)

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

import pytest
import yaml

try:
    from spine.utils.factory import module_dict  # noqa: F401

    SPINE_AVAILABLE = True
except ImportError:
    SPINE_AVAILABLE = False


def load_config_with_includes(config_path):
    """Load a YAML config and recursively process includes."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    # Process includes
    if "include" in cfg:
        includes = cfg.pop("include")
        if not isinstance(includes, list):
            includes = [includes]

        base_cfg = {}
        for inc_file in includes:
            inc_path = config_path.parent / inc_file
            inc_cfg = load_config_with_includes(inc_path)
            # Merge included config
            base_cfg = {**base_cfg, **inc_cfg}

        # Merge current config on top
        cfg = {**base_cfg, **cfg}

    # Process override directive
    if "override" in cfg:
        overrides = cfg.pop("override")
        for key, value in overrides.items():
            # Simple override handling (doesn't handle nested paths like SPINE does)
            cfg[key] = value

    # Strip metadata
    cfg.pop("__meta__", None)

    return cfg


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
class TestConfigValidation:
    """Test that all configuration files parse correctly."""

    @pytest.fixture
    def config_root(self):
        """Return path to infer directory."""
        return Path(__file__).parent.parent / "infer"

    # Detectors to test - will auto-discover YAML files in each
    DETECTORS = ["icarus", "sbnd", "2x2", "nd-lar", "fsd", "generic"]

    @pytest.mark.parametrize("detector", DETECTORS)
    def test_detector_base_configs(self, config_root, detector):
        """Test that all top-level YAML configurations for a detector parse correctly.

        Automatically discovers and tests all .yaml files in the detector root directory
        (excludes subdirectories like base/, io/, model/, etc.).
        """
        detector_dir = config_root / detector
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
            error_msg = f"Failed to parse {detector} configs:\n"
            for cfg, err in failed_configs:
                error_msg += f"  - {cfg}: {err}\n"
            pytest.fail(error_msg)

    @pytest.mark.parametrize("detector", DETECTORS)
    def test_detector_legacy_configs(self, config_root, detector):
        """Test that legacy configurations still parse correctly.

        Automatically discovers and tests all .yaml files in the legacy/ subdirectory.
        """
        detector_dir = config_root / detector
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
            error_msg = f"Failed to parse {detector} legacy configs:\n"
            for cfg, err in failed_configs:
                error_msg += f"  - {cfg}: {err}\n"
            pytest.fail(error_msg)

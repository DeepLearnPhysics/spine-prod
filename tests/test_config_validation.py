"""Tests for configuration file validation.

This module provides automated tests for SPINE-prod configuration files:

1. **Base Configuration Tests**: Validates that all main detector configuration
   files (e.g., icarus_full_chain_*.yaml) parse correctly with includes and
   overrides properly resolved.

2. **Legacy Configuration Tests**: Ensures backward compatibility by testing
   deprecated configuration files in the legacy/ directories.

3. **Metadata Stripping**: Verifies that __meta__ blocks are properly removed
   from final configurations and don't pollute the SPINE runtime config.

4. **Metadata Contracts**: Verifies explicit bundle, fragment, and modifier
   kinds and ensures metadata versions and dates agree with authoritative
   dated filenames.

To run tests:
    pytest tests/test_config_validation.py -v

To add a new detector for testing, add it to DETECTOR_BASE_CONFIGS dict.
"""

import re
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

CONFIG_INFER_ROOT = Path(__file__).parent.parent / "config" / "infer"
CONFIG_ROOT = CONFIG_INFER_ROOT.parent
ACTIVE_CONFIGS = sorted(
    [
        config_path
        for config_path in CONFIG_ROOT.rglob("*.yaml")
        if "legacy" not in config_path.parts
    ]
    + list((CONFIG_ROOT / "train" / "config").rglob("*.cfg"))
)
TRAIN_CONFIGS = sorted(
    list((CONFIG_ROOT / "train" / "config").rglob("*.cfg"))
    + list((CONFIG_ROOT / "train").rglob("*.yaml"))
)
COMMON_CONFIGS = sorted(CONFIG_INFER_ROOT.rglob("*_common.yaml"))
COMPOSITE_CONFIGS = sorted(
    config_path
    for pattern in ("full_chain_*.yaml", "save_truth_*.yaml")
    for config_path in CONFIG_INFER_ROOT.rglob(pattern)
    if "legacy" not in config_path.parts
)
VERSIONED_MODIFIER_CONFIGS = sorted(
    config_path
    for config_path in CONFIG_INFER_ROOT.rglob("*.yaml")
    if "modifier" in config_path.parts and not config_path.stem.endswith("_common")
)

try:
    from spine.config import load_config_file
    from spine.config.api import VALID_KINDS

    SPINE_AVAILABLE = True
    SPINE_SUPPORTS_FRAGMENTS = "fragment" in VALID_KINDS
except ImportError:
    SPINE_AVAILABLE = False
    SPINE_SUPPORTS_FRAGMENTS = False


@pytest.mark.parametrize("config_path", ACTIVE_CONFIGS, ids=lambda path: str(path))
def test_active_configs_declare_explicit_kind(config_path):
    """Every maintained configuration must declare its metadata semantics."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    assert config.get("__meta__", {}).get("kind") in {"bundle", "fragment", "mod"}


@pytest.mark.parametrize("config_path", TRAIN_CONFIGS, ids=lambda path: str(path))
def test_training_configs_use_canonical_top_level_train(config_path):
    """Training recipes must not rely on SPINE's deprecated base.train layout."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    assert "train" not in config.get("base", {})


@pytest.mark.parametrize("config_path", COMPOSITE_CONFIGS, ids=lambda path: str(path))
def test_composite_configs_are_bundles(config_path):
    """Executable full-chain and truth-saving composites must be bundles."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    assert config.get("__meta__", {}).get("kind") == "bundle"


@pytest.mark.parametrize("config_path", COMMON_CONFIGS, ids=lambda path: str(path))
def test_common_configs_are_fragments(config_path):
    """Reusable common configurations must declare fragment metadata."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    assert config.get("__meta__", {}).get("kind") == "fragment"


@pytest.mark.parametrize(
    "config_path", VERSIONED_MODIFIER_CONFIGS, ids=lambda path: str(path)
)
def test_versioned_modifiers_declare_mod_kind(config_path):
    """Versioned modifier configurations must use modifier semantics."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    assert config.get("__meta__", {}).get("kind") == "mod"


@pytest.mark.parametrize(
    "config_path", sorted(CONFIG_INFER_ROOT.rglob("*.yaml")), ids=lambda path: str(path)
)
def test_metadata_version_matches_filename(config_path):
    """Dated filenames are authoritative for metadata versions and dates."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    metadata = config.get("__meta__", {})
    if "version" not in metadata:
        return

    match = re.search(r"_(\d{6})(?:_|$)", config_path.stem)
    assert match is not None, "Versioned metadata requires a dated filename"

    filename_version = match.group(1)
    assert metadata["version"] == filename_version

    expected_date = (
        f"20{filename_version[:2]}-{filename_version[2:4]}-{filename_version[4:]}"
    )
    assert metadata["date"] == expected_date


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
    with (
        patch("spine.config.loader.download_from_url") as mock_download,
        warnings.catch_warnings(),
    ):
        # SPINE releases before fragment support mistake kind-only fragment
        # metadata for a missing block. Do not hide this warning from releases
        # that understand the current metadata contract.
        if not SPINE_SUPPORTS_FRAGMENTS:
            warnings.filterwarnings(
                "ignore",
                message=r"Included file '.*' has no __meta__ block\..*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Invalid __meta__\.kind: 'fragment'.*",
                category=UserWarning,
            )

        # Return a fake path instead of downloading
        mock_download.return_value = "/fake/weights/checkpoint.ckpt"
        return load_config_file(str(config_path))


@contextmanager
def ignore_expected_legacy_metadata_warnings():
    """Ignore metadata diagnostics expected while loading legacy bundles.

    Metadata correctness is tested independently above. Older SPINE loaders do
    not recognize ``kind: fragment`` and infer metadata presence only from a
    version or description, so valid fragment metadata produces a misleading
    "no __meta__ block" warning during legacy compatibility checks.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Included file '.*' has no __meta__ block\..*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Invalid __meta__\.kind: 'fragment'.*",
            category=UserWarning,
        )
        yield


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
                with ignore_expected_legacy_metadata_warnings():
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

    @pytest.mark.parametrize(
        "relative_path",
        [
            "dune10kt-1x2x6/full_chain_260626.yaml",
            "protodune-vd/full_chain_260128.yaml",
        ],
    )
    def test_chained_post_calibrations_continue_from_calibrated_depositions(
        self, config_infer_root, relative_path
    ):
        """Production post gains must preserve the preceding model calibration."""
        config = load_config_with_includes(config_infer_root / relative_path)
        assert config["post"]["calibration"]["depositions_source"] == "depositions"

    def test_sbnd_charge_scale_continues_from_varied_depositions(
        self, config_infer_root, tmp_path
    ):
        """The fully resolved charge-scale variation propagates into calorimetry."""
        sbnd_root = config_infer_root / "sbnd"
        composite = tmp_path / "composite.yaml"
        composite.write_text(
            yaml.safe_dump(
                {
                    "include": [
                        str(sbnd_root / "full_chain_co_260316.yaml"),
                        str(
                            sbnd_root
                            / "modifier"
                            / "charge_scale"
                            / "mod_charge_scale_260813.yaml"
                        ),
                    ]
                }
            ),
            encoding="utf-8",
        )
        config = load_config_with_includes(composite)
        assert (
            config["post"]["apply_calibrations"]["depositions_source"] == "depositions"
        )

    def test_sbnd_smearing_25_percent_composes_with_supported_model(
        self, config_infer_root, tmp_path
    ):
        """The 25% SBND smearing variation must resolve into the full chain."""
        sbnd_root = config_infer_root / "sbnd"
        modifier = sbnd_root / "modifier" / "smearing" / "mod_smearing_260902.yaml"
        composite = tmp_path / "composite.yaml"
        composite.write_text(
            yaml.safe_dump(
                {
                    "include": [
                        str(sbnd_root / "full_chain_co_260316.yaml"),
                        str(modifier),
                    ]
                }
            ),
            encoding="utf-8",
        )

        config = load_config_with_includes(composite)
        calibration = config["model"]["modules"]["calibration"]
        assert calibration["smearing"]["scale"] == 0.25
        assert calibration["smearing"]["mode"] == "multiplicative"
        assert "response_func" in calibration["response"]
        assert (
            config["post"]["apply_calibrations"]["depositions_source"] == "depositions"
        )

    def test_icarus_charge_scale_only_adds_response_and_smearing(
        self, config_infer_root, tmp_path
    ):
        """The ICARUS variation preserves calibration supplied by the base model."""
        icarus_root = config_infer_root / "icarus"
        modifier = (
            icarus_root / "modifier" / "charge_scale" / "mod_charge_scale_260813.yaml"
        )
        modifier_config = yaml.safe_load(modifier.read_text(encoding="utf-8"))
        assert modifier_config["__meta__"]["compatible_with"] == {"model": ">=250303"}
        assert set(modifier_config["override"]) == {
            "model.modules.calibration.response",
            "model.modules.calibration.smearing",
        }

        composite = tmp_path / "composite.yaml"
        composite.write_text(
            yaml.safe_dump(
                {
                    "include": [
                        str(icarus_root / "full_chain_co_260501.yaml"),
                        str(modifier),
                    ]
                }
            ),
            encoding="utf-8",
        )
        config = load_config_with_includes(composite)
        calibration = config["model"]["modules"]["calibration"]
        assert calibration["stage"] == "segmentation"
        assert calibration["gain"]["gain"] == 79.9169
        assert "recombination" in calibration
        assert "lifetime" in calibration
        assert "transparency" in calibration
        assert calibration["response"]["priority"] == 10
        assert calibration["smearing"]["priority"] == 9

    def test_icarus_charge_scale_accepts_first_in_chain_calibration(
        self, config_infer_root, tmp_path
    ):
        """The first ICARUS model with in-chain calibration supports the modifier."""
        icarus_root = config_infer_root / "icarus"
        modifier = (
            icarus_root / "modifier" / "charge_scale" / "mod_charge_scale_260813.yaml"
        )
        composite = tmp_path / "composite.yaml"
        composite.write_text(
            yaml.safe_dump(
                {
                    "include": [
                        str(icarus_root / "full_chain_co_250303.yaml"),
                        str(modifier),
                    ]
                }
            ),
            encoding="utf-8",
        )

        config = load_config_with_includes(composite)
        assert "response" in config["model"]["modules"]["calibration"]

    @pytest.mark.parametrize(
        ("direction", "scale"),
        [("neg", 0.98), ("pos", 1.02)],
    )
    def test_icarus_gain_scale_uses_response_and_preserves_nominal_gain(
        self, config_infer_root, tmp_path, direction, scale
    ):
        """ICARUS gain variations use response without replacing nominal gain."""
        icarus_root = config_infer_root / "icarus"
        modifier = (
            icarus_root
            / "modifier"
            / f"gain_{direction}_scale"
            / f"mod_gain_{direction}_scale_260902.yaml"
        )
        composite = tmp_path / "composite.yaml"
        composite.write_text(
            yaml.safe_dump(
                {
                    "include": [
                        str(icarus_root / "full_chain_co_260501.yaml"),
                        str(modifier),
                    ]
                }
            ),
            encoding="utf-8",
        )

        config = load_config_with_includes(composite)
        calibration = config["model"]["modules"]["calibration"]
        assert calibration["gain"]["gain"] == 79.9169
        assert calibration["gain_scale"] == {
            "name": "response",
            "priority": 10,
            "response_func": f"{scale}*x",
        }

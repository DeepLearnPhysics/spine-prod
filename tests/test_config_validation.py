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

import os
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.pipeline import PipelineDefinition

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
GENERIC_TRAIN_BUNDLES = sorted(
    (CONFIG_ROOT / "train" / "generic").glob("*/train_*.yaml")
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


@pytest.mark.parametrize(
    "config_path", GENERIC_TRAIN_BUNDLES, ids=lambda path: str(path)
)
def test_generic_training_bundles_pin_their_model_revision(config_path):
    """Every dated generic training bundle must pin one matching model revision."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model_version = config["__meta__"]["version"]
    model_includes = [
        include for include in config["include"] if include.startswith("model/generic/")
    ]
    assert len(model_includes) == 1
    assert model_includes[0].endswith(f"model_{model_version}.yaml")


def test_generic_training_variants_follow_component_model_changes():
    """Sister recipes exist only for components that changed between releases."""
    variants = {
        model_dir.name: {
            yaml.safe_load(config_path.read_text(encoding="utf-8"))["__meta__"][
                "version"
            ]
            for config_path in model_dir.glob("train_*.yaml")
        }
        for model_dir in (CONFIG_ROOT / "train" / "generic").iterdir()
        if model_dir.is_dir() and list(model_dir.glob("train_*.yaml"))
    }
    assert variants == {
        "uresnet": {"240718"},
        "uresnet_ppn": {"240718", "240805"},
        "graph_spice": {"240718", "240805"},
        "grappa_shower": {"240718", "260828"},
        "grappa_track": {"240718", "260828"},
        "grappa_inter": {"240718", "240805", "260828"},
    }


def test_generic_models_and_cache_producers_name_dated_component_fragments():
    """Composition sites must expose their exact network and loss revisions."""
    components = {
        "uresnet": ("240718",),
        "graph_spice": ("240718", "240805"),
        "grappa_shower": ("240718", "260828"),
        "grappa_track": ("240718", "260828"),
        "grappa_inter": ("240718", "240805", "260828"),
    }
    for component, versions in components.items():
        for version in versions:
            model_path = (
                CONFIG_ROOT / "model" / "generic" / component / f"model_{version}.yaml"
            )
            model = yaml.load(model_path.read_text(), Loader=yaml.BaseLoader)
            module_name = {
                "uresnet": "uresnet",
                "graph_spice": "graph_spice",
            }.get(component, "grappa")
            assert model["override"][f"model.modules.{module_name}"] == (
                f"model/generic/{component}/network_{version}.yaml"
            )
            loss_name = {
                "uresnet": "uresnet_loss",
                "graph_spice": "graph_spice_loss",
            }.get(component, "grappa_loss")
            assert model["override"][f"model.modules.{loss_name}"] == (
                f"model/generic/{component}/loss_{version}.yaml"
            )

    # UResNet-PPN shares leaf modules with standalone UResNet while also
    # providing one nested fragment for full-chain composition.
    for version in ("240718", "240805"):
        model_path = (
            CONFIG_ROOT / "model" / "generic" / "uresnet_ppn" / f"model_{version}.yaml"
        )
        model = yaml.load(model_path.read_text(), Loader=yaml.BaseLoader)
        overrides = model["override"]
        assert overrides["model.modules.uresnet"] == (
            "model/generic/uresnet/network_240718.yaml"
        )
        assert overrides["model.modules.uresnet_loss"] == (
            "model/generic/uresnet/loss_240718.yaml"
        )
        assert overrides["model.modules.ppn"] == (
            f"model/generic/uresnet_ppn/ppn_{version}.yaml"
        )
        assert overrides["model.modules.ppn_loss"] == (
            f"model/generic/uresnet_ppn/ppn_loss_{version}.yaml"
        )

        network_path = (
            CONFIG_ROOT
            / "model"
            / "generic"
            / "uresnet_ppn"
            / f"network_{version}.yaml"
        )
        network = yaml.load(network_path.read_text(), Loader=yaml.BaseLoader)
        assert network["uresnet"] == "model/generic/uresnet/network_240718.yaml"
        assert network["ppn"] == (f"model/generic/uresnet_ppn/ppn_{version}.yaml")

    segmentation_cache_path = (
        CONFIG_ROOT / "cache" / "generic" / "uresnet_ppn" / "segmentation_240805.yaml"
    )
    segmentation_cache = yaml.load(
        segmentation_cache_path.read_text(), Loader=yaml.BaseLoader
    )
    assert segmentation_cache["model"]["modules"]["uresnet_ppn"] == (
        "model/generic/uresnet_ppn/network_240805.yaml"
    )

    fragment_cache_path = (
        CONFIG_ROOT
        / "cache"
        / "generic"
        / "graph_spice"
        / "fragment_graphs_240805.yaml"
    )
    fragment_cache = yaml.load(fragment_cache_path.read_text(), Loader=yaml.BaseLoader)
    fragment_modules = fragment_cache["model"]["modules"]
    assert fragment_modules["graph_spice"] == (
        "model/generic/graph_spice/network_240805.yaml"
    )
    assert fragment_modules["grappa_shower"] == (
        "model/generic/grappa_shower/network_240718.yaml"
    )
    assert fragment_modules["grappa_track"] == (
        "model/generic/grappa_track/network_240718.yaml"
    )

    particle_cache_path = (
        CONFIG_ROOT
        / "cache"
        / "generic"
        / "grappa_shower_track"
        / "particle_graphs_240805.yaml"
    )
    particle_cache = yaml.load(particle_cache_path.read_text(), Loader=yaml.BaseLoader)
    particle_modules = particle_cache["model"]["modules"]
    assert particle_modules["grappa_shower"] == (
        "model/generic/grappa_shower/network_240718.yaml"
    )
    assert particle_modules["grappa_track"] == (
        "model/generic/grappa_track/network_240718.yaml"
    )
    assert particle_modules["grappa_inter"] == (
        "model/generic/grappa_inter/network_240805.yaml"
    )

    # The authoritative full-chain model selects the same component revisions.
    full_chain_path = (
        CONFIG_ROOT / "model" / "generic" / "full_chain" / "model_240805.yaml"
    )
    full_chain = yaml.load(full_chain_path.read_text(), Loader=yaml.BaseLoader)
    full_chain_overrides = full_chain["override"]
    assert full_chain_overrides["model.modules.uresnet_ppn"] == (
        "model/generic/uresnet_ppn/network_240805.yaml"
    )
    assert full_chain_overrides["model.modules.uresnet_ppn_loss"] == (
        "model/generic/uresnet_ppn/loss_240805.yaml"
    )
    selected_modules = dict(fragment_modules)
    selected_modules.update(particle_modules)
    for component in ("graph_spice", "grappa_shower", "grappa_track", "grappa_inter"):
        assert (
            full_chain_overrides[f"model.modules.{component}"]
            == selected_modules[component]
        )


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_generic_segmentation_cache_reuses_uresnet_ppn_revision():
    """The cache producer must run the same network as standalone training."""
    cache = load_config_with_includes(
        CONFIG_ROOT / "cache" / "generic" / "uresnet_ppn" / "segmentation_240805.yaml"
    )
    standalone = load_config_with_includes(
        CONFIG_ROOT / "model" / "generic" / "uresnet_ppn" / "model_240805.yaml"
    )

    cache_module = cache["model"]["modules"]["uresnet_ppn"]
    standalone_modules = standalone["model"]["modules"]
    assert cache_module["uresnet"] == standalone_modules["uresnet"]
    assert cache_module["ppn"] == standalone_modules["ppn"]
    assert cache_module["model_name"] == ""
    assert cache_module["weight_path"] == ""
    assert cache["base"]["split_output"] is True
    assert "split" not in cache["io"]["writer"]
    assert cache["io"]["writer"]["stage"] == "segmentation"
    assert cache["io"]["writer"]["overwrite_stage"] is True
    assert cache["io"]["writer"]["keys"] == [
        "seg_pred",
        "ppn_points",
        "clust_label_adapt",
    ]
    assert cache["model"]["network_input"] == {
        "data": "data",
        "seg_label": "seg_label",
        "clust_label": "clust_label",
    }
    stage_config = cache["model"]["modules"]["chain"]["stages"][0]["config"]
    assert stage_config["adapt_labels"] == {
        "break_eps": 1.1,
        "break_metric": "chebyshev",
        "break_p": 2.0,
        "break_classes": [0, 1, 2, 3],
        "weighted": True,
    }
    cache_schema = cache["io"]["loader"]["dataset"]["schema"]
    assert set(cache_schema) == {"data", "seg_label", "clust_label"}
    assert cache_schema["clust_label"]["particle_event"] == "particle_corrected"
    assert cache_schema["clust_label"]["shape_precedence"] == [2, 1, 0, 3, 4, 6]


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
@pytest.mark.parametrize("version", ["240718", "240805"])
def test_generic_uresnet_ppn_matches_generic_full_chain(version):
    """Standalone and full-chain UResNet-PPN must share dated leaf modules."""
    standalone = load_config_with_includes(
        CONFIG_ROOT / "model" / "generic" / "uresnet_ppn" / f"model_{version}.yaml"
    )["model"]["modules"]
    full_chain = load_config_with_includes(
        CONFIG_INFER_ROOT / "generic" / "model" / f"model_{version}.yaml"
    )["model"]["modules"]

    assert full_chain["uresnet_ppn"] == {
        "uresnet": standalone["uresnet"],
        "ppn": standalone["ppn"],
    }
    assert full_chain["uresnet_ppn_loss"] == {
        "uresnet_loss": standalone["uresnet_loss"],
        "ppn_loss": standalone["ppn_loss"],
    }


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_generic_graph_spice_can_train_from_segmentation_cache():
    """The cached recipe must merge truth with the canonical prediction product."""
    config = load_config_with_includes(
        CONFIG_ROOT
        / "train"
        / "generic"
        / "graph_spice"
        / "train_from_uresnet_ppn_240805.yaml"
    )

    dataset = config["io"]["loader"]["dataset"]
    assert dataset["name"] == "mixed"
    assert set(dataset["larcv"]["schema"]) == {"data"}
    assert dataset["hdf5"]["staged"] is True
    assert dataset["hdf5"]["stage"] == "segmentation"
    assert dataset["hdf5"]["keys"] == ["seg_pred", "clust_label_adapt"]
    assert "schema" not in dataset["hdf5"]
    assert set(config["validation"]["sources"]) == {"larcv", "hdf5"}
    assert config["model"]["network_input"]["seg_label"] == "seg_pred"
    assert config["model"]["loss_input"]["seg_label"] == "seg_pred"
    assert config["model"]["loss_input"]["clust_label"] == "clust_label_adapt"


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_generic_fragment_cache_materializes_both_grappa_training_contracts():
    """One Graph-SPICE pass must feed independent shower and track jobs."""
    config = load_config_with_includes(
        CONFIG_ROOT
        / "cache"
        / "generic"
        / "graph_spice"
        / "fragment_graphs_240805.yaml"
    )

    loader = config["io"]["loader"]
    assert loader["num_workers"] == 0
    assert loader["dataset"]["hdf5"]["keep_open"] is False
    assert config["io"]["writer"]["keep_open"] is False
    assert config["io"]["writer"]["overwrite_stage"] is True
    assert "ppn_points" not in config["io"]["writer"]["keys"]
    assert "clust_label_adapt" not in config["io"]["writer"]["keys"]

    chain = config["model"]["modules"]["chain"]
    assert chain["inputs"] == ["seg_pred", "ppn_points"]
    assert config["model"]["network_input"]["ppn_points"] == "ppn_points"
    assert [stage["name"] for stage in chain["stages"]] == [
        "fragmentation",
        "particle_aggregation",
    ]
    assert config["model"]["network_input"]["seg_pred"] == "seg_pred"

    modules = config["model"]["modules"]
    assert modules["graph_spice"]["model_name"] == ""
    assert modules["graph_spice"]["weight_path"] == ""
    assert modules["grappa_shower"]["return_features"] is True
    assert modules["grappa_track"]["return_features"] is True
    assert modules["grappa_shower_loss"]["return_targets"] is True
    assert modules["grappa_track_loss"]["return_targets"] is True

    # Cache-only controls aside, these are the authoritative standalone modules.
    revisions = {
        "graph_spice": ("240805", "graph_spice", None),
        "grappa_shower": ("240718", "grappa", "grappa_loss"),
        "grappa_track": ("240718", "grappa", "grappa_loss"),
    }
    for component, (version, network_key, loss_key) in revisions.items():
        standalone = load_config_with_includes(
            CONFIG_ROOT / "model" / "generic" / component / f"model_{version}.yaml"
        )["model"]["modules"]
        cache_network = deepcopy(modules[component])
        for key in ("model_name", "weight_path", "return_features"):
            cache_network.pop(key, None)
        assert cache_network == standalone[network_key]
        if loss_key is not None:
            cache_loss = deepcopy(modules[f"{component}_loss"])
            cache_loss.pop("return_targets")
            assert cache_loss == standalone[loss_key]

    keys = set(config["io"]["writer"]["keys"])
    for path in ("shower", "track"):
        prefix = f"{path}_fragment"
        assert {
            f"{prefix}_clusts",
            f"{prefix}_edge_index",
            f"{prefix}_node_features",
            f"{prefix}_edge_features",
        }.issubset(keys)
    assert {
        "particle_aggregation_shower_node_target",
        "particle_aggregation_shower_node_valid",
        "particle_aggregation_shower_edge_target",
        "particle_aggregation_shower_edge_valid",
        "particle_aggregation_track_edge_target",
        "particle_aggregation_track_edge_valid",
    }.issubset(keys)


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
@pytest.mark.parametrize(
    ("component", "bundle", "prefix", "objectives"),
    [
        (
            "grappa_shower",
            "train_from_fragment_cache_240718.yaml",
            "shower_fragment",
            ("node", "edge"),
        ),
        (
            "grappa_track",
            "train_from_fragment_cache_240718.yaml",
            "track_fragment",
            ("edge",),
        ),
    ],
)
def test_generic_particle_grappas_train_only_from_cached_graphs(
    component, bundle, prefix, objectives
):
    """Cached GrapPA recipes must bypass graph construction and encoding."""
    config = load_config_with_includes(
        CONFIG_ROOT / "train" / "generic" / component / bundle
    )

    dataset = config["io"]["loader"]["dataset"]
    assert dataset["name"] == "hdf5"
    assert dataset["staged"] is True
    assert dataset["stage"] == "fragmentation"
    assert config["model"]["network_input"] == {
        "data": "data",
        "clusts": f"{prefix}_clusts",
        "edge_index": f"{prefix}_edge_index",
        "node_features": f"{prefix}_node_features",
        "edge_features": f"{prefix}_edge_features",
    }
    assert set(config["model"]["modules"]["grappa"]) == {"nodes", "gnn_model"}
    assert set(config["model"]["loss_input"]) == {
        key
        for objective in objectives
        for key in (f"{objective}_target", f"{objective}_valid")
    }


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_generic_particle_cache_and_inter_training_share_one_graph_contract():
    """The second cache must provide every input and target used by inter."""
    cache = load_config_with_includes(
        CONFIG_ROOT
        / "cache"
        / "generic"
        / "grappa_shower_track"
        / "particle_graphs_240805.yaml"
    )
    training = load_config_with_includes(
        CONFIG_ROOT
        / "train"
        / "generic"
        / "grappa_inter"
        / "train_from_particle_cache_240805.yaml"
    )

    loader = cache["io"]["loader"]
    assert loader["num_workers"] == 0
    assert loader["dataset"]["keep_open"] is False
    assert loader["dataset"]["stage_map"] == {
        "ppn_points": "segmentation",
        "clust_label_adapt": "segmentation",
    }
    assert cache["io"]["writer"]["keep_open"] is False
    assert cache["io"]["writer"]["overwrite_stage"] is True

    chain = cache["model"]["modules"]["chain"]
    assert chain["inputs"] == [
        "ppn_points",
        "fragment_clusts",
        "fragment_shapes",
    ]
    assert cache["model"]["network_input"]["ppn_points"] == "ppn_points"
    assert [stage["name"] for stage in chain["stages"]] == [
        "particle_aggregation",
        "interaction_aggregation",
    ]
    assert cache["model"]["modules"]["grappa_shower"]["model_name"] == ""
    assert cache["model"]["modules"]["grappa_track"]["model_name"] == ""
    assert cache["model"]["modules"]["grappa_inter"]["return_features"] is True
    assert cache["model"]["modules"]["grappa_inter_loss"]["return_targets"] is True

    cache_modules = cache["model"]["modules"]
    revisions = {
        "grappa_shower": "240718",
        "grappa_track": "240718",
        "grappa_inter": "240805",
    }
    for component, version in revisions.items():
        standalone = load_config_with_includes(
            CONFIG_ROOT / "model" / "generic" / component / f"model_{version}.yaml"
        )["model"]["modules"]
        cache_network = deepcopy(cache_modules[component])
        for key in ("model_name", "weight_path", "return_features"):
            cache_network.pop(key, None)
        assert cache_network == standalone["grappa"]
        if component == "grappa_inter":
            cache_loss = deepcopy(cache_modules["grappa_inter_loss"])
            cache_loss.pop("return_targets")
            assert cache_loss == standalone["grappa_loss"]

    writer_keys = set(cache["io"]["writer"]["keys"])
    reader_keys = set(training["io"]["loader"]["dataset"]["keys"])
    assert reader_keys.issubset(writer_keys)
    assert training["model"]["network_input"] == {
        "data": "data",
        "clusts": "particle_clusts",
        "edge_index": "particle_edge_index",
        "node_features": "particle_node_features",
        "edge_features": "particle_edge_features",
    }
    assert set(training["model"]["modules"]["grappa"]) == {"nodes", "gnn_model"}


def test_generic_full_chain_training_pipeline_has_expected_fan_out_and_join():
    """Shower and track training fan out, then jointly gate the inter cache."""
    pipeline_path = (
        CONFIG_ROOT.parent / "pipelines" / "generic" / "full_chain_240805.yaml"
    )
    pipeline = PipelineDefinition.load(
        str(pipeline_path), workspace_override="/path/to/workflow"
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}

    fragment_dependencies = [
        "cache_train_fragment_graphs",
        "cache_validation_fragment_graphs",
    ]
    assert stages["train_grappa_shower"]["depends_on"] == fragment_dependencies
    assert stages["train_grappa_track"]["depends_on"] == fragment_dependencies

    particle_dependencies = ["train_grappa_shower", "train_grappa_track"]
    assert stages["cache_train_particle_graphs"]["depends_on"] == particle_dependencies
    assert (
        stages["cache_validation_particle_graphs"]["depends_on"]
        == particle_dependencies
    )
    assert stages["train_grappa_inter"]["depends_on"] == [
        "cache_train_particle_graphs",
        "cache_validation_particle_graphs",
    ]
    assert all(
        stage["time"] == "08:00:00"
        for stage in pipeline.stages
        if stage["name"] != "export_full_chain_weights"
    )

    # Every materialization stage extends one source-derived cache per split.
    train_cache = "/path/to/workflow/cache/train/train_cache.h5"
    validation_cache = "/path/to/workflow/cache/validation/test_cache.h5"
    for name in ("cache_train_segmentation", "cache_train_fragment_graphs"):
        assert stages[name]["output"] == "/path/to/workflow/cache/train"
        assert stages[name]["output_suffix"] == "cache"
    for name in (
        "cache_validation_segmentation",
        "cache_validation_fragment_graphs",
    ):
        assert stages[name]["output"] == "/path/to/workflow/cache/validation"
        assert stages[name]["output_suffix"] == "cache"
    assert stages["cache_train_particle_graphs"]["source"] == train_cache
    assert stages["cache_train_particle_graphs"]["output"] == train_cache
    assert "output_suffix" not in stages["cache_train_particle_graphs"]
    assert stages["cache_validation_particle_graphs"]["source"] == validation_cache
    assert stages["cache_validation_particle_graphs"]["output"] == validation_cache
    assert "output_suffix" not in stages["cache_validation_particle_graphs"]
    assert stages["train_grappa_inter"]["source"] == train_cache
    assert stages["train_grappa_inter"]["val_source"] == validation_cache

    expected_run_dirs = {
        "cache_train_segmentation": "/path/to/workflow/cache/train/segmentation",
        "cache_validation_segmentation": (
            "/path/to/workflow/cache/validation/segmentation"
        ),
        "cache_train_fragment_graphs": "/path/to/workflow/cache/train/fragmentation",
        "cache_validation_fragment_graphs": (
            "/path/to/workflow/cache/validation/fragmentation"
        ),
        "cache_train_particle_graphs": (
            "/path/to/workflow/cache/train/particle_aggregation"
        ),
        "cache_validation_particle_graphs": (
            "/path/to/workflow/cache/validation/particle_aggregation"
        ),
    }
    for name, run_dir in expected_run_dirs.items():
        assert stages[name]["run_dir"] == run_dir

    export = stages["export_full_chain_weights"]
    assert export["depends_on"] == ["train_grappa_inter"]
    assert export["config"] == "model/generic/full_chain/model_240805.yaml"
    assert export["profile"] == "s3df_milano"
    assert export["time"] == "00:30:00"
    assert export["export_weights"] == (
        "/path/to/workflow/weights/full_chain_240805.ckpt"
    )
    assert set(export["module_weight"]) == {
        "uresnet_ppn",
        "graph_spice",
        "grappa_shower",
        "grappa_track",
        "grappa_inter",
    }


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
@pytest.mark.parametrize("version", ["240718", "240805"])
def test_generic_inference_wraps_authoritative_full_chain_model(version):
    """Inference should add published weights without redefining the model."""
    composed = load_config_with_includes(
        CONFIG_ROOT / "model" / "generic" / "full_chain" / f"model_{version}.yaml"
    )
    inference = load_config_with_includes(
        CONFIG_INFER_ROOT / "generic" / "model" / f"model_{version}.yaml"
    )

    weight_path = inference["model"].pop("weight_path")
    assert weight_path == "/fake/weights/checkpoint.ckpt"
    assert inference == composed


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_default_uresnet_matches_common_full_chain():
    """The common full chain must compose the shared UResNet leaves."""
    network = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "uresnet" / "network_v1.yaml"
    )
    loss = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "uresnet" / "loss_v1.yaml"
    )
    full_chain = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "full_chain" / "base_v1.yaml"
    )

    assert network == full_chain["model"]["modules"]["uresnet_ppn"]["uresnet"]
    assert loss == full_chain["model"]["modules"]["uresnet_ppn_loss"]["uresnet_loss"]


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_default_uresnet_ppn_matches_common_full_chain():
    """The common full chain must compose the shared point-proposal leaves."""
    ppn = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "uresnet_ppn" / "ppn_v1.yaml"
    )
    ppn_loss = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "uresnet_ppn" / "ppn_loss_v1.yaml"
    )
    full_chain = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "full_chain" / "base_v1.yaml"
    )

    assert ppn == full_chain["model"]["modules"]["uresnet_ppn"]["ppn"]
    assert ppn_loss == full_chain["model"]["modules"]["uresnet_ppn_loss"]["ppn_loss"]


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_default_uresnet_ppn_reuses_default_uresnet():
    """UResNet-PPN must resolve the shared UResNet network and loss fragments."""
    uresnet = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "uresnet" / "base_v1.yaml"
    )
    uresnet_ppn = load_config_with_includes(
        CONFIG_ROOT / "model" / "common" / "uresnet_ppn" / "base_v1.yaml"
    )

    assert (
        uresnet_ppn["model"]["modules"]["uresnet"]
        == uresnet["model"]["modules"]["uresnet"]
    )
    assert (
        uresnet_ppn["model"]["modules"]["uresnet_loss"]
        == uresnet["model"]["modules"]["uresnet_loss"]
    )


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
@pytest.mark.parametrize("version", ["240718", "240805"])
def test_generic_graph_spice_matches_generic_full_chain(version):
    """The generic Graph-SPICE definition must match the generic full chain."""
    graph_spice = load_config_with_includes(
        CONFIG_ROOT / "model" / "generic" / "graph_spice" / f"model_{version}.yaml"
    )
    full_chain = load_config_with_includes(
        CONFIG_INFER_ROOT / "generic" / "model" / f"model_{version}.yaml"
    )

    modules = graph_spice["model"]["modules"]
    full_chain_modules = full_chain["model"]["modules"]
    assert modules["graph_spice"] == full_chain_modules["graph_spice"]
    assert modules["graph_spice_loss"] == full_chain_modules["graph_spice_loss"]


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
@pytest.mark.parametrize(
    ("component", "version"),
    [
        ("grappa_shower", "240718"),
        ("grappa_track", "240718"),
        ("grappa_inter", "240718"),
        ("grappa_inter", "240805"),
    ],
)
def test_generic_grappa_matches_generic_full_chain(component, version):
    """Each generic GrapPA definition must match its full-chain component."""
    grappa = load_config_with_includes(
        CONFIG_ROOT / "model" / "generic" / component / f"model_{version}.yaml"
    )
    full_chain = load_config_with_includes(
        CONFIG_INFER_ROOT / "generic" / "model" / f"model_{version}.yaml"
    )

    modules = grappa["model"]["modules"]
    full_chain_modules = full_chain["model"]["modules"]

    assert modules["grappa"] == full_chain_modules[component]
    assert modules["grappa_loss"] == full_chain_modules[f"{component}_loss"]


def test_standalone_interaction_grappa_has_unambiguous_primary_targets():
    """Particle-group nodes must not use fragment-level target disambiguation."""
    loss_path = CONFIG_ROOT / "model" / "common" / "grappa_inter" / "loss_v1.yaml"
    with open(loss_path, "r", encoding="utf-8") as config_file:
        primary_loss = yaml.safe_load(config_file)["node_loss"]["primary"]

    assert "use_closest" not in primary_loss
    assert "secondary_label" not in primary_loss


@pytest.mark.skipif(not SPINE_AVAILABLE, reason="SPINE not available")
def test_graph_spice_uses_a_dedicated_uresnet_embedder():
    """Graph-SPICE must own its specialized UResNet configuration."""
    embedder_path = CONFIG_ROOT / "model" / "common" / "graph_spice" / "uresnet_v1.yaml"
    with open(embedder_path, "r", encoding="utf-8") as config_file:
        embedder = yaml.safe_load(config_file)

    assert "include" not in embedder
    assert embedder["num_input"] == 4
    assert embedder["input_kernel"] == 5
    assert embedder["spatial_size"] is None
    assert "num_classes" not in embedder


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
    "config_path", sorted(CONFIG_ROOT.rglob("*.yaml")), ids=lambda path: str(path)
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
        patch.dict(os.environ, {"SPINE_CONFIG_PATH": str(CONFIG_ROOT)}),
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
        """Test that all runnable YAML training bundles parse correctly."""
        detector_dir = config_train_root / detector
        if not detector_dir.exists():
            pytest.skip(f"Detector directory not found: {detector_dir}")

        yaml_files = []
        for config_path in detector_dir.rglob("*.yaml"):
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = yaml.load(config_file, Loader=yaml.BaseLoader)
            if config.get("__meta__", {}).get("kind") == "bundle":
                yaml_files.append(config_path)

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

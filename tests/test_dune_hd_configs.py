"""DUNE-HD candidate calibration, training, and release-isolation contracts."""

from pathlib import Path

import pytest

from src.config_manager import ConfigManager
from src.pipeline import PipelineDefinition
from tests.config_migration import fingerprint

spine_config = pytest.importorskip("spine.config")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
VERSIONS = ["260202", "260510", "260626"]
COMPONENTS = [
    "uresnet_deghost",
    "uresnet_ppn",
    "graph_spice",
    "grappa_shower",
    "grappa_track",
    "grappa_inter",
]


@pytest.fixture(autouse=True)
def config_environment(monkeypatch):
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(CONFIG))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(ROOT))


def load(path):
    return spine_config.load_config_file(str(CONFIG / path), download=False)


@pytest.mark.parametrize("version", VERSIONS)
def test_released_inference_is_unchanged(version):
    expected = {
        "260202": "2a1e97937f6346a55684c5205f55c186cf31adf0dd5018a56d2e24a905749b7d",
        "260510": "a76df0ba1a020801adb849e03e80720212aade305e84629cfc749f69b5478e9a",
        "260626": "ada9a48ac07ce559a84bb1e51c78c5df506b87b8a06c37b845fdc15ee4f8b3d0",
    }
    assert (
        fingerprint(load(f"infer/dune-hd-10kt-1x2x6/full_chain_{version}.yaml"), CONFIG)
        == expected[version]
    )


def test_latest_still_selects_released_inference(tmp_path):
    generated = ConfigManager(ROOT).create_latest_config("dune-hd-10kt-1x2x6", tmp_path)
    latest = spine_config.load_config_file(generated, download=False)
    released = load("infer/dune-hd-10kt-1x2x6/full_chain_260626.yaml")
    assert fingerprint(latest, CONFIG) == fingerprint(released, CONFIG)


def test_fd_architectures_preserve_detector_specific_parameters():
    released = load("infer/dune-hd-10kt-1x2x6/full_chain_260626.yaml")["model"][
        "modules"
    ]
    candidate = load("model/dune-hd-10kt-1x2x6/full_chain/model_261022.yaml")["model"][
        "modules"
    ]
    for component in COMPONENTS:
        assert candidate[component] == released[component]
    assert candidate["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 6


def test_filters_and_initial_resource_budget():
    pipeline = PipelineDefinition.load(
        ROOT / "pipelines/dune-hd-10kt-1x2x6/full_chain_261022.yaml",
        workspace_override="/tmp/fd-training",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    for split in ["train", "validation"]:
        build = stages[f"build_{split}_filter"]
        assert load(build["config"])["filters"]["sparse3d_reco"]["max_count"] == 500000
        assert stages[f"cache_{split}_deghosting"]["entry_filter"] == build["output"]
    for component, batch, hours in [
        ("uresnet_deghost", 16, "5-00:00:00"),
        ("uresnet_ppn", 32, "3-00:00:00"),
        ("graph_spice", 32, "2-00:00:00"),
        ("grappa_shower", 16, "1-00:00:00"),
        ("grappa_track", 32, "16:00:00"),
        ("grappa_inter", 128, "08:00:00"),
    ]:
        stage = stages[f"train_{component}"]
        cfg = load(stage["config"])
        loader = cfg["io"]["loader"]
        assert (
            loader["batch_size" if component == "uresnet_deghost" else "minibatch_size"]
            == batch
        )
        assert stage["time"] == hours
        assert cfg["model"].get("weight_path") is None
        assert stage["val_entry_fraction_range"] == [0.0, 0.5]
    assert stages["evaluate_full_chain"]["entry_fraction_range"] == [0.5, 1.0]


def test_candidate_calibrates_charge_before_recombination_without_double_scaling():
    model = load("model/dune-hd-10kt-1x2x6/full_chain/model_261022.yaml")
    cache = load("cache/dune-hd-10kt-1x2x6/uresnet_deghost/deghosting_261022.yaml")
    calibration = load("model/dune-hd-10kt-1x2x6/calibration/calibration_261022.yaml")
    assert calibration["stages"][0] == {"name": "gain", "gain": 1.0 / 0.005045}
    assert [stage["name"] for stage in calibration["stages"]] == [
        "gain",
        "recombination",
        "lifetime",
    ]
    assert model["model"]["modules"]["calibration"] == calibration
    assert (
        cache["model"]["modules"]["chain"]["stages"][1]["config"]["calibration"]
        == calibration
    )
    candidate_post = load("infer/dune-hd-10kt-1x2x6/post/candidate/post_261022.yaml")
    assert "calibration" not in {
        stage["name"] for stage in candidate_post["post"]["stages"]
    }
    evaluation = load("test/dune-hd-10kt-1x2x6/full_chain/evaluate_261022.yaml")
    assert evaluation["model"]["modules"] == model["model"]["modules"]
    assert "calibration" not in {
        stage["name"] for stage in evaluation["post"]["stages"]
    }
    assert evaluation["geo"] == cache["geo"] == model["geo"]
    assert evaluation["io"]["writer"] is None
    assert "flashes" not in evaluation["io"]["loader"]["dataset"]["schema"]


def test_candidate_materializes_exact_training_targets_including_orientation():
    model = load("model/dune-hd-10kt-1x2x6/full_chain/model_261022.yaml")["model"][
        "modules"
    ]
    cache = load(
        "cache/dune-hd-10kt-1x2x6/grappa_shower_track/particle_graphs_261022.yaml"
    )
    training = load(
        "train/dune-hd-10kt-1x2x6/grappa_inter/train_from_particle_cache_261022.yaml"
    )
    objective = model["grappa_inter_loss"]
    assert objective == cache["model"]["modules"]["grappa_inter_loss"]
    assert objective == training["model"]["modules"]["grappa_loss"]
    assert set(objective["node_loss"]) == {"type", "primary", "orient"}
    for name in ["type", "primary"]:
        assert objective["node_loss"][name]["min_iou"] == 0.5
        assert objective["node_loss"][name]["match_target"] == "group"
    assert objective["node_loss"]["primary"]["use_closest"] is True
    assert objective["node_loss"]["primary"]["secondary_label"] == 0
    standalone = load("model/dune-hd-10kt-1x2x6/grappa_inter/model_261022.yaml")
    assert (
        "min_iou"
        not in standalone["model"]["modules"]["grappa_loss"]["node_loss"]["type"]
    )
    assert model["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 6
    produced = set(cache["io"]["writer"]["keys"])
    assert produced == set(training["io"]["loader"]["dataset"]["keys"])
    for key in ["node_orient_target", "node_orient_valid"]:
        assert training["model"]["loss_input"][key] == f"particle_graph_{key}"
        assert f"particle_graph_{key}" in produced
    assert cache["model"]["network_input"]["coord_label"] == "coord_label"


def test_candidate_pipeline_uses_new_calibration_targets_and_checkpoint_end_to_end():
    pipeline = PipelineDefinition.load(
        ROOT / "pipelines/dune-hd-10kt-1x2x6/full_chain_261022.yaml",
        workspace_override="/tmp/vd-candidate",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    assert len(stages) == 21
    for stage in pipeline.stages:
        assert not stage.get("expandable_segments", False)
        cfg = load(stage["config"])
        if stage.get("stage") == "train":
            assert stage["config"].endswith("261022.yaml")
            assert set(stage["warm_start"]) <= set(cfg["model"]["modules"])
            assert stage["val_entry_fraction_range"] == [0.0, 0.5]
    for split in ["train", "validation"]:
        assert stages[f"cache_{split}_deghosting"]["config"].endswith(
            "deghosting_261022.yaml"
        )
        assert stages[f"cache_{split}_particle_graphs"]["config"].endswith(
            "particle_graphs_261022.yaml"
        )
    assert stages["export_full_chain_weights"]["config"].endswith("model_261022.yaml")
    assert stages["evaluate_full_chain"]["config"].endswith("evaluate_261022.yaml")
    checkpoint = stages["export_full_chain_weights"]["export_weights"]
    assert stages["evaluate_full_chain"]["weight_path"] == checkpoint
    assert stages["report_full_chain"]["checkpoint"] == checkpoint
    assert stages["report_full_chain"]["config"].endswith("report_261022.yaml")
    ppn = load(stages["train_uresnet_ppn"]["config"])
    assert ppn["base"]["epochs"] == 100.0
    report = load(stages["report_full_chain"]["config"])
    assert "particle_orientation" in report["metrics"]["nodes"]["tasks"]
    assert "deghosting" in report["metrics"]


@pytest.mark.parametrize("version", VERSIONS)
def test_legacy_pipeline_preserves_released_model_and_postprocessing(version):
    pipeline = PipelineDefinition.load(
        ROOT / f"pipelines/dune-hd-10kt-1x2x6/full_chain_{version}.yaml",
        workspace_override=f"/tmp/fd-{version}",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    assert len(stages) == 21
    for stage in pipeline.stages:
        cfg = load(stage["config"])
        if stage.get("stage") == "train":
            assert cfg["model"].get("weight_path") is None
            assert set(stage["warm_start"]) <= set(cfg["model"]["modules"])
    released = load(f"infer/dune-hd-10kt-1x2x6/full_chain_{version}.yaml")
    exported = load(stages["export_full_chain_weights"]["config"])
    evaluated = load(stages["evaluate_full_chain"]["config"])
    assert exported["model"]["modules"] == released["model"]["modules"]
    assert evaluated["model"]["modules"] == released["model"]["modules"]
    # Metrics enable fragment matching in adapted space and add PPN records.
    expected_post = released["post"]["stages"]
    for stage in expected_post:
        if stage["name"] == "match":
            stage.update(truth_point_mode="points_adapt", fragment=True)
    assert evaluated["post"]["stages"] == expected_post + [{"name": "ppn"}]
    dataset_root = (
        "/sdf/data/neutrino/dune/sim/mpvmpr_v0"
        if version == "260202"
        else "/sdf/data/neutrino/dune/sim/hd/mpvmpr_v1"
    )
    assert (
        stages["train_uresnet_deghost"]["source_list"]
        == f"{dataset_root}/train_file_list.txt"
    )
    cache = load(stages["cache_train_deghosting"]["config"])
    assert (
        cache["model"]["modules"]["chain"]["stages"][1]["config"]["calibration"]
        == released["model"]["modules"]["calibration"]
    )
    particle_cache = load(stages["cache_train_particle_graphs"]["config"])
    trainer = load(stages["train_grappa_inter"]["config"])
    assert (
        particle_cache["model"]["modules"]["grappa_inter_loss"]
        == released["model"]["modules"]["grappa_inter_loss"]
    )
    assert (
        trainer["model"]["modules"]["grappa_loss"]
        == released["model"]["modules"]["grappa_inter_loss"]
    )
    assert set(particle_cache["io"]["writer"]["keys"]) == set(
        trainer["io"]["loader"]["dataset"]["keys"]
    )
    assert not any("orient" in key for key in particle_cache["io"]["writer"]["keys"])

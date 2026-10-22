"""ProtoDUNE-VD deployed-model and staged-training contracts."""

import hashlib
import json
from pathlib import Path

import pytest

from src.config_manager import ConfigManager
from src.pipeline import PipelineDefinition
from tests.config_migration import fingerprint

spine_config = pytest.importorskip("spine.config")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
VERSIONS = ["260118", "260128"]
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
def test_released_inference_preserves_original_behavior(version):
    expected = {
        "260118": "a1325af161252a03684635b487513fd751cbd3a9082a9e507c3141435bdfcf52",
        "260128": "3228a984929de8f5241de39319bae1975f72be019576e39b492f786e626957f5",
    }
    inference = load(f"infer/protodune-vd/full_chain_{version}.yaml")
    assert fingerprint(inference, CONFIG) == expected[version]
    shared = load(f"model/protodune-vd/full_chain/model_{version}.yaml")
    assert (
        shared["geo"]
        == inference["geo"]
        == {
            "detector": "protodune-vd",
            "tag": "protodunevd_v5",
        }
    )
    released = inference["model"]
    assert released.pop("weight_path") is not None
    shared_model = shared["model"]
    assert shared_model.pop("weight_path") is None
    assert shared_model == released
    digest = hashlib.sha256(
        json.dumps(shared_model, sort_keys=True).encode()
    ).hexdigest()
    assert digest == "7684daf82682b4cc1fb0b14fdcd8faed9125de3d0e8f8838e4e93fc6cd0c1f8b"
    assert set(released["modules"]["grappa_inter_loss"]["node_loss"]) == {
        "type",
        "primary",
    }
    assert released["modules"]["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 5


def test_component_models_and_cache_producers_pin_deployed_networks():
    modules = load("model/protodune-vd/full_chain/model_260118.yaml")["model"][
        "modules"
    ]
    for component in COMPONENTS:
        standalone = load(f"model/protodune-vd/{component}/model_260118.yaml")["model"][
            "modules"
        ]
        if component == "uresnet_ppn":
            for leaf in ["uresnet", "ppn"]:
                assert standalone[leaf] == modules[component][leaf]
                assert (
                    standalone[leaf + "_loss"]
                    == modules[component + "_loss"][leaf + "_loss"]
                )
        else:
            name = (
                "uresnet"
                if component == "uresnet_deghost"
                else ("grappa" if component.startswith("grappa") else component)
            )
            network = standalone[name]
            deployed = dict(modules[component])
            if component in {"grappa_track", "grappa_inter"}:
                deployed["nodes"] = dict(deployed["nodes"])
                if component == "grappa_track":
                    deployed["nodes"]["make_groups"] = False
                    deployed["nodes"].pop("grouping_method")
                else:
                    deployed["nodes"].pop("grouping_through_track")
            assert network == deployed
            assert standalone[name + "_loss"] == modules[component + "_loss"]
    cache_components = {
        "uresnet_deghost/deghosting": ["uresnet_deghost"],
        "uresnet_ppn/segmentation": ["uresnet_ppn"],
        "graph_spice/fragment_graphs": [
            "graph_spice",
            "grappa_shower",
            "grappa_shower_loss",
            "grappa_track",
            "grappa_track_loss",
        ],
        "grappa_shower_track/particle_graphs": [
            "grappa_shower",
            "grappa_track",
            "grappa_inter",
            "grappa_inter_loss",
        ],
    }
    for recipe, names in cache_components.items():
        cache = load(f"cache/protodune-vd/{recipe}_260118.yaml")
        for name in names:
            actual = dict(cache["model"]["modules"][name])
            actual.pop("weight_path", None)
            actual.pop("model_name", None)
            assert actual == modules[name]


def test_deghosting_calibration_cache_preserves_point_domain():
    deghosting = load("cache/protodune-vd/uresnet_deghost/deghosting_260118.yaml")
    calibration = load("model/protodune-vd/calibration/calibration_260118.yaml")
    assert deghosting["geo"] == {"detector": "protodune-vd", "tag": "protodunevd_v5"}
    assert deghosting["io"]["writer"]["keys"] == ["data_calib", "orig_index"]
    stages = deghosting["model"]["modules"]["chain"]["stages"]
    assert stages[0]["config"]["charge_rescaling"] == "collection"
    assert stages[1]["config"]["calibration"] == calibration
    assert calibration["stages"][0] == {"name": "gain", "gain": 14.0}
    for version in VERSIONS:
        training = load(
            f"train/protodune-vd/uresnet_ppn/train_from_deghost_cache_{version}.yaml"
        )
        assert training["model"]["network_input"]["data"] == "data_calib"
        assert training["model"]["network_input"]["orig_index"] == "orig_index"
        assert (
            training["model"]["modules"]["chain"]["stages"][0]["config"]["adapt_labels"]
            is False
        )
        assert training["base"]["epochs"] == 100.0
        info = training["io"]["loader"]["dataset"]["primary"]["schema"]["clust_label"][
            "particle_info"
        ]
        assert info["particle_event"] == "particle_corrected"
        assert info["neutrino_event"] == "neutrino_mpv"
        assert info["type_include_mpr"] is False


def test_interaction_cache_has_only_supported_targets():
    cache = load("cache/protodune-vd/grappa_shower_track/particle_graphs_260118.yaml")
    produced = set(cache["io"]["writer"]["keys"])
    assert not any("orient" in key for key in produced)
    for version in VERSIONS:
        training = load(
            f"train/protodune-vd/grappa_inter/train_from_particle_cache_{version}.yaml"
        )
        consumed = set(training["io"]["loader"]["dataset"]["keys"])
        assert consumed == produced
        assert not any("orient" in key for key in training["model"]["loss_input"])
        assert set(training["model"]["modules"]["grappa_loss"]["node_loss"]) == {
            "type",
            "primary",
        }


@pytest.mark.parametrize("version", VERSIONS)
def test_pipeline_resolves_filter_six_trainers_and_complete_evaluation(version):
    pipeline = PipelineDefinition.load(
        ROOT / f"pipelines/protodune-vd/full_chain_{version}.yaml",
        workspace_override="/tmp/vd-training",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    assert len(stages) == 21
    for stage in pipeline.stages:
        assert not stage.get("expandable_segments", False)
        cfg = load(stage["config"])
        if stage.get("stage") == "train":
            assert set(stage["warm_start"]) <= set(cfg["model"]["modules"])
            assert stage["val_entry_fraction_range"] == [0.0, 0.5]
    assert stages["train_uresnet_ppn"]["warm_start"] == {"uresnet_ppn": "uresnet_ppn"}
    assert (
        stages["train_uresnet_deghost"]["source_list"]
        == "/sdf/data/neutrino/pdune/sim/vd/mpvmpr_v0/train_file_list.txt"
    )
    assert [
        stage["name"] for stage in pipeline.stages if stage.get("stage") == "train"
    ] == [f"train_{component}" for component in COMPONENTS]
    for split in ["train", "validation"]:
        path = f"/tmp/vd-training/cache/{split}.spine-cache"
        assert stages[f"cache_{split}_deghosting"]["cache_repository"] == path
        assert stages[f"cache_{split}_deghosting"]["output"] == path
        for cache_stage in ["segmentation", "fragment_graphs", "particle_graphs"]:
            assert (
                stages[f"cache_{split}_{cache_stage}"]["sources"]["cache"]["source"]
                == path
            )
        assert stages[f"cache_{split}_particle_graphs"]["depends_on"] == [
            "train_grappa_shower",
            "train_grappa_track",
        ]
    evaluation = stages["evaluate_full_chain"]
    assert evaluation["entry_fraction_range"] == [0.5, 1.0]
    assert (
        evaluation["weight_path"]
        == stages["export_full_chain_weights"]["export_weights"]
    )
    assert stages["report_full_chain"]["depends_on"] == ["evaluate_full_chain"]


@pytest.mark.parametrize("version", VERSIONS)
def test_evaluation_uses_release_postprocessing_and_ghost_metrics(version):
    evaluation = load(f"test/protodune-vd/full_chain/evaluate_{version}.yaml")
    model = load(f"model/protodune-vd/full_chain/model_{version}.yaml")
    assert evaluation["model"]["modules"] == model["model"]["modules"]
    schema = evaluation["io"]["loader"]["dataset"]["schema"]
    assert "flashes" not in schema and "crthits" not in schema
    assert schema["particles"]["neutrino_event"] == "neutrino_mpv"
    processors = {stage["name"]: stage for stage in evaluation["post"]["stages"]}
    assert processors["match"]["truth_point_mode"] == "points_adapt"
    assert processors["match"]["ghost"] is True
    if version == "260128":
        assert processors["calibration"]["gain"]["gain"] == 14.2857
    else:
        assert "calibration" not in processors
    analyzers = {stage["name"]: stage for stage in evaluation["ana"]["stages"]}
    assert analyzers["segment_eval"]["ghost"] is True
    report = load("test/protodune-vd/full_chain/report_260118.yaml")
    assert report["metrics"]["deghosting"]["class_mapping"]["Ghost"] == ["ghost"]


def test_latest_inference_keeps_released_checkpoint(tmp_path):
    generated = ConfigManager(ROOT).create_latest_config("protodune-vd", tmp_path)
    latest = spine_config.load_config_file(generated, download=False)
    released = load("infer/protodune-vd/full_chain_260128.yaml")
    assert vars(latest["model"]["weight_path"]) == vars(
        released["model"]["weight_path"]
    )
    assert latest["model"]["modules"] == released["model"]["modules"]
    assert latest["post"] == released["post"]
    assert {
        p.name for p in (CONFIG / "infer/protodune-vd").glob("full_chain_*.yaml")
    } == {"full_chain_260118.yaml", "full_chain_260128.yaml"}


def test_candidate_calibrates_charge_before_recombination_without_double_scaling():
    model = load("model/protodune-vd/full_chain/model_261014.yaml")
    cache = load("cache/protodune-vd/uresnet_deghost/deghosting_261014.yaml")
    calibration = load("model/protodune-vd/calibration/calibration_261014.yaml")
    assert calibration["stages"][0] == {"name": "gain", "gain": 1.0 / 0.005}
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
    candidate_post = load("infer/protodune-vd/post/candidate/post_261014.yaml")
    assert "calibration" not in {
        stage["name"] for stage in candidate_post["post"]["stages"]
    }
    evaluation = load("test/protodune-vd/full_chain/evaluate_261014.yaml")
    assert evaluation["model"]["modules"] == model["model"]["modules"]
    assert "calibration" not in {
        stage["name"] for stage in evaluation["post"]["stages"]
    }
    assert evaluation["geo"] == cache["geo"] == model["geo"]
    assert evaluation["io"]["writer"] is None
    assert "flashes" not in evaluation["io"]["loader"]["dataset"]["schema"]


def test_candidate_materializes_exact_training_targets_including_orientation():
    model = load("model/protodune-vd/full_chain/model_261014.yaml")["model"]["modules"]
    cache = load("cache/protodune-vd/grappa_shower_track/particle_graphs_261014.yaml")
    training = load(
        "train/protodune-vd/grappa_inter/train_from_particle_cache_261014.yaml"
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
    standalone = load("model/protodune-vd/grappa_inter/model_261014.yaml")
    assert (
        "min_iou"
        not in standalone["model"]["modules"]["grappa_loss"]["node_loss"]["type"]
    )
    assert model["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 5
    produced = set(cache["io"]["writer"]["keys"])
    assert produced == set(training["io"]["loader"]["dataset"]["keys"])
    for key in ["node_orient_target", "node_orient_valid"]:
        assert training["model"]["loss_input"][key] == f"particle_graph_{key}"
        assert f"particle_graph_{key}" in produced
    assert cache["model"]["network_input"]["coord_label"] == "coord_label"


def test_candidate_pipeline_uses_new_calibration_targets_and_checkpoint_end_to_end():
    pipeline = PipelineDefinition.load(
        ROOT / "pipelines/protodune-vd/full_chain_261014.yaml",
        workspace_override="/tmp/vd-candidate",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    assert len(stages) == 21
    for stage in pipeline.stages:
        assert not stage.get("expandable_segments", False)
        cfg = load(stage["config"])
        if stage.get("stage") == "train":
            assert stage["config"].endswith("261014.yaml")
            assert set(stage["warm_start"]) <= set(cfg["model"]["modules"])
            assert stage["val_entry_fraction_range"] == [0.0, 0.5]
    for split in ["train", "validation"]:
        assert stages[f"cache_{split}_deghosting"]["config"].endswith(
            "deghosting_261014.yaml"
        )
        assert stages[f"cache_{split}_particle_graphs"]["config"].endswith(
            "particle_graphs_261014.yaml"
        )
    assert stages["export_full_chain_weights"]["config"].endswith("model_261014.yaml")
    assert stages["evaluate_full_chain"]["config"].endswith("evaluate_261014.yaml")
    checkpoint = stages["export_full_chain_weights"]["export_weights"]
    assert stages["evaluate_full_chain"]["weight_path"] == checkpoint
    assert stages["report_full_chain"]["checkpoint"] == checkpoint
    assert stages["report_full_chain"]["config"].endswith("report_261014.yaml")
    ppn = load(stages["train_uresnet_ppn"]["config"])
    assert ppn["base"]["epochs"] == 100.0
    report = load(stages["report_full_chain"]["config"])
    assert "particle_orientation" in report["metrics"]["nodes"]["tasks"]
    assert "deghosting" in report["metrics"]

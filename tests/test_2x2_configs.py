"""2x2 migration contracts for deployed models and cache transitions."""

import hashlib
import json
from pathlib import Path

import pytest

from src.pipeline import PipelineDefinition

spine_config = pytest.importorskip("spine.config")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def config_environment(monkeypatch):
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(ROOT / "config"))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(ROOT))


def load(path):
    return spine_config.load_config_file(str(ROOT / "config" / path), download=False)


@pytest.mark.parametrize("version", ["240719", "240819"])
def test_deployed_model_is_preserved_and_shared(version):
    # Fingerprints of the original inference model, excluding checkpoint paths.
    expected = {
        "240719": "d0baadf433d0d58268ab1ebdf3d7ab7328f29d7a34ec8dbbeffcaf0a763da93b",
        "240819": "7a5b7d528dc54f19acf59ff3fbcc7aa3f6b36b0046e34e195da2e9865ddf34bc",
    }
    shared = load(f"model/2x2/full_chain/model_{version}.yaml")["model"]
    deployed = load(f"infer/2x2/model/model_{version}.yaml")["model"]
    assert deployed.pop("weight_path") is not None
    assert deployed == shared
    assert (
        hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest()
        == expected[version]
    )
    evaluation = load(f"test/2x2/full_chain/evaluate_{version}.yaml")["model"]
    assert evaluation["modules"] == shared["modules"]


@pytest.mark.parametrize("version", ["240719", "240819"])
def test_training_and_cache_use_deployed_components(version):
    modules = load(f"model/2x2/full_chain/model_{version}.yaml")["model"]["modules"]
    for component in [
        "uresnet_ppn",
        "graph_spice",
        "grappa_shower",
        "grappa_track",
        "grappa_inter",
    ]:
        standalone = load(f"model/2x2/{component}/model_{version}.yaml")["model"][
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
            target = "grappa" if component.startswith("grappa") else component
            assert standalone[target] == modules[component]
            assert standalone[target + "_loss"] == modules[component + "_loss"]
    cache_stages = {
        "uresnet_ppn/segmentation": ["uresnet_ppn"],
        "graph_spice/fragment_graphs": [
            "graph_spice",
            "grappa_shower",
            "grappa_track",
            "grappa_shower_loss",
            "grappa_track_loss",
        ],
        "grappa_shower_track/particle_graphs": [
            "grappa_shower",
            "grappa_track",
            "grappa_inter",
            "grappa_inter_loss",
        ],
    }
    for recipe, names in cache_stages.items():
        cache = load(f"cache/2x2/{recipe}_{version}.yaml")
        for name in names:
            actual = dict(cache["model"]["modules"][name])
            actual.pop("model_name", None)
            actual.pop("weight_path", None)
            assert actual == modules[name]


@pytest.mark.parametrize("version,dataset", [("240719", "v1"), ("240819", "v2")])
def test_pipeline_resolves_complete_workflow(version, dataset):
    pipeline = PipelineDefinition.load(
        ROOT / f"pipelines/2x2/full_chain_{version}.yaml",
        workspace_override="/tmp/2x2-test",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    assert len(stages) == 14
    assert (
        stages["train_uresnet_ppn"]["source_list"]
        == f"/sdf/data/neutrino/2x2/sim/mpvmpr_{dataset}/train_file_list.txt"
    )
    batches = {
        "uresnet_ppn": 1024,
        "graph_spice": 256,
        "grappa_shower": 256,
        "grappa_track": 256,
        "grappa_inter": 512,
    }
    for component, batch in batches.items():
        stage = stages[f"train_{component}"]
        cfg = load(stage["config"])
        assert (
            cfg["io"]["loader"].get(
                "batch_size", cfg["io"]["loader"].get("minibatch_size")
            )
            == batch
        )
        assert cfg["base"]["epochs"] == 50.0
        assert stage["warm_start"]
        assert stage["val_entry_fraction_range"] == [0.0, 0.5]
        if component.startswith("grappa"):
            assert cfg["io"]["loader"]["dataset"]["provider"] == "cache"
            assert "node_encoder" not in cfg["model"]["modules"]["grappa"]
    assert stages["evaluate_full_chain"]["entry_fraction_range"] == [0.5, 1.0]
    assert (
        stages["evaluate_full_chain"]["weight_path"]
        == stages["export_full_chain_weights"]["export_weights"]
    )
    assert stages["report_full_chain"]["depends_on"] == ["evaluate_full_chain"]
    ppn = load(stages["train_uresnet_ppn"]["config"])
    assert ppn["io"]["loader"]["dataset"]["schema"]["ppn_label"][
        "include_point_tagging"
    ] == (version == "240719")


def test_current_tasks_match_latest_shared_policies():
    current = load("model/2x2/full_chain/model_261007.yaml")["model"]["modules"]
    old = load("model/2x2/full_chain/model_240819.yaml")["model"]["modules"]
    assert current["uresnet_ppn"] == old["uresnet_ppn"]
    assert current["graph_spice"] == old["graph_spice"]
    assert current["uresnet_ppn_loss"]["ppn_loss"] == load(
        "model/common/uresnet_ppn/ppn_loss_v1.yaml"
    )
    assert current["grappa_shower_loss"] == load(
        "model/common/grappa_shower/loss_v1.yaml"
    )
    latest = load("model/nd-lar/grappa_inter/loss_full_chain_260924.yaml")
    assert current["grappa_inter_loss"] == latest
    for name in ["grappa_shower", "grappa_track", "grappa_inter"]:
        assert current[name]["graph"] == old[name]["graph"]
        assert current[name]["node_encoder"]["dir_max_dist"] == 5
        assert current[name]["node_encoder"]["dedx_max_dist"] == 5
    assert current["grappa_inter"]["nodes"]["grouping_through_track"] is True
    assert current["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 6
    fragment = load("cache/2x2/graph_spice/fragment_graphs_261007.yaml")["model"][
        "modules"
    ]
    particle = load("cache/2x2/grappa_shower_track/particle_graphs_261007.yaml")[
        "model"
    ]["modules"]
    for names, cache in [
        (["grappa_shower", "grappa_shower_loss", "grappa_track"], fragment),
        (["grappa_inter", "grappa_inter_loss"], particle),
    ]:
        for name in names:
            assert cache[name] == current[name]
    training = load("train/2x2/grappa_inter/train_from_particle_cache_261007.yaml")
    assert training["model"]["modules"]["grappa_loss"] == particle["grappa_inter_loss"]
    assert load("model/2x2/grappa_inter/model_261007.yaml")["model"]["modules"][
        "grappa_loss"
    ] == load("model/common/grappa_inter/loss_v1.yaml")


def test_current_pipeline_augmentation_and_metric_truth():
    import yaml

    pipeline = PipelineDefinition.load(
        ROOT / "pipelines/2x2/full_chain_261007.yaml",
        workspace_override="/tmp/2x2-current",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    assert len(stages) == 14
    for name, mod in [("uresnet_ppn", "uresnet"), ("graph_spice", "graph_spice")]:
        stage = stages[f"train_{name}"]
        assert stage["apply_mods"] == [f"augment_{mod}:261007"]
        cfg = spine_config.load_config(
            yaml.safe_dump(
                {
                    "include": [
                        stage["config"],
                        f"train/2x2/modifier/augment_{mod}/mod_augment_{mod}_261007.yaml",
                    ]
                }
            ),
            root_dir=str(ROOT / "config"),
            download=False,
        )
        assert cfg["geo"] == {"detector": "2x2", "tag": "mr5-0"}
        augment = cfg["io"]["loader"]["dataset"]["augment"]["stages"]
        assert [a["axis"] for a in augment] == [0, 1, 2]
        assert all(a["p"] == 0.5 and a["keep_meta"] for a in augment)
        backbone = (
            cfg["model"]["modules"]["uresnet"]
            if name == "uresnet_ppn"
            else cfg["model"]["modules"]["graph_spice"]["embedder"]["uresnet"]
        )
        assert backbone["lattice"] == {"period": "auto"}
    evaluation = load(stages["evaluate_full_chain"]["config"])
    schema = evaluation["io"]["loader"]["dataset"]["schema"]
    assert "neutrino_event" not in schema["clust_label"]
    assert "neutrino_event" not in schema["particles"]
    assert "neutrinos" not in schema
    report = load(stages["report_full_chain"]["config"])
    assert report == load("test/common/full_chain/report_v1.yaml")
    assert report["metrics"]["ppn"]["distance_unit"] == "cm"
    assert (
        stages["evaluate_full_chain"]["weight_path"]
        == stages["export_full_chain_weights"]["export_weights"]
    )
    assert stages["evaluate_full_chain"]["entry_fraction_range"] == [0.5, 1.0]
    infer = load("infer/2x2/full_chain_261007.yaml")
    assert infer["model"]["weight_path"] is None
    assert infer["model"]["modules"] == evaluation["model"]["modules"]

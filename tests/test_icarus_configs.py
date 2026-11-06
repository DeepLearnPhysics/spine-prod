"""ICARUS deployed parity and two-cryostat staged training contracts."""

import copy
import json
from pathlib import Path

import pytest

from src.config_manager import ConfigManager
from src.pipeline import PipelineDefinition
from tests.config_migration import fingerprint

spine_config = pytest.importorskip("spine.config")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
BASELINE = json.loads(
    (ROOT / "tests/fixtures/icarus_inference_baseline.json").read_text()
)
REVISIONS = [name.removeprefix("full_chain_") for name in BASELINE]
COMPONENTS = (
    "uresnet_deghost",
    "uresnet_ppn",
    "graph_spice",
    "grappa_shower",
    "grappa_track",
    "grappa_inter",
)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(CONFIG))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(ROOT))
    monkeypatch.setenv("ICARUS_DATA_DIR", "/test/icarus-data")


def load(path):
    return spine_config.load_config_file(str(CONFIG / path), download=False)


def pipeline(revision, **kwargs):
    return PipelineDefinition.load(
        ROOT / f"pipelines/icarus/full_chain_{revision}.yaml",
        workspace_override=f"/tmp/icarus-{revision}",
        **kwargs,
    )


@pytest.mark.parametrize("chain", BASELINE)
def test_deployed_inference_is_unchanged(chain):
    assert fingerprint(load(f"infer/icarus/{chain}.yaml"), CONFIG) == BASELINE[chain]


def test_latest_stays_on_released_may_model(tmp_path):
    path = ConfigManager(ROOT).create_latest_config("icarus", tmp_path)
    assert (
        fingerprint(spine_config.load_config_file(path, download=False), CONFIG)
        == BASELINE["full_chain_co_260501"]
    )


@pytest.mark.parametrize("revision", REVISIONS)
def test_historical_model_and_cache_parity(revision):
    stages = {s["name"]: s for s in pipeline(revision).stages}
    configs = {name: load(s["config"]) for name, s in stages.items()}
    released = load(f"infer/icarus/full_chain_{revision}.yaml")
    modules = released["model"]["modules"]
    exported, evaluation = (
        configs["export_full_chain_weights"],
        configs["evaluate_full_chain"],
    )
    assert exported["model"]["weight_path"] is None
    assert evaluation["model"]["weight_path"] is None
    assert exported["model"]["modules"] == evaluation["model"]["modules"] == modules
    assert exported["geo"] == evaluation["geo"] == released["geo"]
    for name, keys in {
        "cache_train_deghosting": ["uresnet_deghost"],
        "cache_train_segmentation": ["uresnet_ppn"],
        "cache_train_fragment_graphs": [
            "graph_spice",
            "grappa_shower",
            "grappa_shower_loss",
            "grappa_track",
            "grappa_track_loss",
        ],
        "cache_train_particle_graphs": [
            "grappa_shower",
            "grappa_track",
            "grappa_inter",
            "grappa_inter_loss",
        ],
    }.items():
        for key in keys:
            actual = copy.deepcopy(configs[name]["model"]["modules"][key])
            actual.pop("model_name", None)
            actual.pop("weight_path", None)
            assert actual == modules[key]
    deghost = configs["cache_train_deghosting"]
    chain = deghost["model"]["modules"]["chain"]["stages"]
    assert chain[0]["config"] == modules["chain"]["stages"][0]["config"]
    assert chain[1]["config"]["calibration"] == (
        modules.get("calibration") or {"stages": []}
    )
    assert (
        deghost["io"]["loader"]["dataset"]["schema"]["data"]
        == released["io"]["loader"]["dataset"]["schema"]["data"]
    )
    assert (
        configs["train_grappa_inter"]["model"]["modules"]["grappa_loss"]
        == modules["grappa_inter_loss"]
    )
    # Historical ICARUS has an orientation head but no orientation objective.
    assert "orient" in modules["grappa_inter"]["gnn_model"]["node_pred"]
    assert "orient" not in modules["grappa_inter_loss"]["node_loss"]
    expected = copy.deepcopy(released["post"]["stages"])
    expected = [s for s in expected if s["name"] != "flash_match"]
    for s in expected:
        if s["name"] == "match":
            s.update(ghost=True, truth_point_mode="points_adapt", fragment=True)
    assert evaluation["post"]["stages"] == expected + [{"name": "ppn"}]
    assert evaluation["io"]["writer"] is None
    assert evaluation["io"]["loader"]["collate_fn"]["split"] is True
    assert "run_info" in evaluation["io"]["loader"]["dataset"]["schema"]
    assert not {"flashes", "crthits"} & set(
        evaluation["io"]["loader"]["dataset"]["schema"]
    )


@pytest.mark.parametrize("revision", REVISIONS)
def test_standalone_networks_preserve_released_parameters(revision):
    modules = load(f"model/icarus/full_chain/model_{revision}.yaml")["model"]["modules"]
    for component in COMPONENTS:
        standalone = load(f"model/icarus/{component}/model_{revision}.yaml")["model"][
            "modules"
        ]
        if component == "uresnet_ppn":
            for leaf in ("uresnet", "ppn"):
                assert standalone[leaf] == modules[component][leaf]
                assert (
                    standalone[leaf + "_loss"]
                    == modules[component + "_loss"][leaf + "_loss"]
                )
            continue
        key = (
            "uresnet"
            if component == "uresnet_deghost"
            else "grappa" if component.startswith("grappa") else component
        )
        expected = copy.deepcopy(modules[component])
        if component == "grappa_track":
            expected["nodes"]["make_groups"] = False
            expected["nodes"].pop("grouping_method", None)
        if component == "grappa_inter":
            expected["nodes"].pop("grouping_through_track", None)
        assert standalone[key] == expected
        assert standalone[key + "_loss"] == modules[component + "_loss"]


@pytest.mark.parametrize("revision", [*REVISIONS, "261106"])
def test_pipeline_alignment_sources_and_resources(revision):
    stages = {
        s["name"]: s
        for s in pipeline(
            revision,
            source_list="/inputs/train.list",
            val_source_list="/inputs/test.list",
        ).stages
    }
    assert len(stages) == 21
    configs = {name: load(s["config"]) for name, s in stages.items()}
    for split, source in [
        ("train", "/inputs/train.list"),
        ("validation", "/inputs/test.list"),
    ]:
        for op in ["scan", "build"]:
            assert stages[f"{op}_{split}_filter"]["source_list"] == source
        assert stages[f"cache_{split}_deghosting"]["source_list"] == source
        for step in ["segmentation", "fragment_graphs", "particle_graphs"]:
            assert (
                stages[f"cache_{split}_{step}"]["sources"]["primary"]["source_list"]
                == source
            )
        assert stages[f"cache_{split}_deghosting"]["ntasks"] == (
            4 if split == "train" else 1
        )
    for component, batch in zip(COMPONENTS, [8, 12, 25, 12, 16, 64]):
        stage = stages[f"train_{component}"]
        cfg = configs[f"train_{component}"]
        key = "batch_size" if component == "uresnet_deghost" else "minibatch_size"
        assert cfg["io"]["loader"][key] == batch
        assert cfg["io"]["loader"]["collate_fn"] == "all"
        assert cfg["validation"]["early_stopping"]["patience"] == 5
        assert stage["val_entry_fraction_range"] == [0.0, 0.5]
        assert set(stage["warm_start"]) <= set(cfg["model"]["modules"])
    for consumer, producer in [
        ("train_grappa_inter", "cache_train_particle_graphs"),
        ("train_grappa_shower", "cache_train_fragment_graphs"),
        ("train_grappa_track", "cache_train_fragment_graphs"),
    ]:
        keys = configs[consumer]["io"]["loader"]["dataset"]["keys"]
        assert set(keys) <= set(configs[producer]["io"]["writer"]["keys"])
        assert any("orient" in key for key in keys) == (
            revision == "261106" and consumer == "train_grappa_inter"
        )
    raw = configs["cache_train_deghosting"]
    schema = raw["io"]["loader"]["dataset"]["schema"]
    assert schema["data"]["num_features"] == 8
    assert len(schema["data"]["sparse_event_list"]) == 16
    assert schema["sources"]["num_features"] == 2
    assert raw["model"]["network_input"]["run_info"] == "run_info"
    assert "run_info" in schema
    assert raw["io"]["loader"]["collate_fn"] == "all"
    assert raw["geo"] == {"detector": "icarus", "tag": "icarus_v4"}
    filt = configs["build_train_filter"]["filters"]
    assert filt == {
        "sparse3d_reco_cryoE": {"max_count": 500000},
        "sparse3d_reco_cryoW": {"max_count": 500000},
    }
    evaluation = stages["evaluate_full_chain"]
    assert evaluation["source_list"] == "/inputs/test.list"
    assert evaluation["entry_fraction_range"] == [0.5, 1.0]
    assert evaluation["ntasks"] == 4 and "files_per_task" not in evaluation
    assert (
        evaluation["weight_path"]
        == stages["export_full_chain_weights"]["export_weights"]
        == stages["report_full_chain"]["checkpoint"]
    )


def test_candidate_targets_calibration_and_augmentation():
    stages = {s["name"]: s for s in pipeline("261106").stages}
    configs = {name: load(s["config"]) for name, s in stages.items()}
    latest = load("model/icarus/full_chain/model_co_260501.yaml")["model"]["modules"]
    candidate = configs["export_full_chain_weights"]["model"]["modules"]
    for key in latest:
        if key != "grappa_inter_loss":
            assert candidate[key] == latest[key]
    for name in [
        "export_full_chain_weights",
        "evaluate_full_chain",
        "cache_train_particle_graphs",
    ]:
        loss = configs[name]["model"]["modules"]["grappa_inter_loss"]["node_loss"]
        assert loss["type"]["min_iou"] == loss["primary"]["min_iou"] == 0.5
        assert (
            loss["type"]["match_target"] == loss["primary"]["match_target"] == "group"
        )
        assert (
            loss["primary"]["use_closest"] and loss["primary"]["secondary_label"] == 0
        )
        assert loss["orient"] == {"provider": "orient", "loss": "ce"}
    standalone = configs["train_grappa_inter"]["model"]["modules"]["grappa_loss"][
        "node_loss"
    ]
    assert "min_iou" not in standalone["type"]
    assert "use_closest" not in standalone["primary"]
    assert "orient" in standalone
    for name in [
        "train_uresnet_ppn",
        "cache_train_segmentation",
        "evaluate_full_chain",
    ]:
        data = configs[name]["io"]["loader"]["dataset"]
        info = data.get("primary", data)["schema"]["clust_label"]["particle_info"]
        assert not any(
            info[k]
            for k in [
                "type_include_secondary",
                "type_include_mpr",
                "primary_include_mpr",
            ]
        )
    for component, mod, backbone in [
        ("uresnet_deghost", "augment_uresnet", "uresnet"),
        ("uresnet_ppn", "augment_ppn", "uresnet_ppn"),
        ("graph_spice", "augment_graph_spice", "graph_spice"),
    ]:
        stage = stages[f"train_{component}"]
        assert not stage.get("apply_mods")
        assert "augment" not in configs[f"train_{component}"]["io"]["loader"]["dataset"]
        cfg = spine_config.load_config(
            f"include:\n  - {stage['config']}\n  - train/icarus/modifier/{mod}/mod_{mod}_261106.yaml\n",
            root_dir=str(CONFIG),
            download=False,
        )
        net = cfg["model"]["modules"][backbone]
        if component == "uresnet_ppn":
            net = net["uresnet"]
        elif component == "graph_spice":
            net = net["embedder"]["uresnet"]
        assert net["lattice"] == {"period": "auto"}
        flips = cfg["io"]["loader"]["dataset"]["augment"]["stages"]
        assert [s["axis"] for s in flips] == [0, 1, 2]
        assert all(s["keep_meta"] == (component != "uresnet_ppn") for s in flips)
        assert cfg["geo"]["detector"] == "icarus"


def test_unsplit_cache_collation_keeps_both_cryostats_in_raw_index_domain(monkeypatch):
    import numpy as np
    from spine.data import IndexData, Meta, TensorData
    from spine.geo import GeoManager
    from spine.io.collate import CollateAll
    from spine.io.unwrap import Unwrapper

    # Cache products and raw labels must share a logical event axis even when
    # its points lie in different cryostats. No geometry-dependent re-split.
    monkeypatch.setattr(GeoManager, "get_instance_if_initialized", lambda: None)
    meta = Meta(
        lower=np.zeros(3),
        upper=np.full(3, 200.0),
        size=np.ones(3),
        count=np.full(3, 200),
    )
    events = []
    for event_id, selected in enumerate(([0, 2], [1, 3])):
        coords = np.array([[1, 1, 1], [2, 2, 2], [101, 1, 1], [102, 2, 2]])
        labels = np.arange(4)[:, None] + 10 * event_id
        selected = np.asarray(selected)
        events.append(
            {
                "index": event_id,
                "seg_label": TensorData(coords=coords, features=labels, meta=meta),
                "data_calib": TensorData(
                    coords=coords[selected], features=np.ones((2, 1)), meta=meta
                ),
                "orig_index": IndexData(selected, span=4),
            }
        )
    batch = CollateAll(data_keys=events[0].keys())(events)
    aligned = batch["seg_label"].features.data[batch["orig_index"].index]
    np.testing.assert_array_equal(aligned.ravel(), [0, 2, 11, 13])
    np.testing.assert_array_equal(batch["orig_index"].index, [0, 2, 5, 7])
    result = Unwrapper()(batch)
    for i, selected in enumerate(([0, 2], [1, 3])):
        np.testing.assert_array_equal(result["orig_index"][i].features, selected)
        np.testing.assert_array_equal(
            result["data_calib"][i].coords, result["seg_label"][i].coords[selected]
        )


@pytest.mark.parametrize(
    "revision, dataset",
    [
        ("240719", 2),
        ("240812", 2),
        ("co_240812", 2),
        ("250115", 3),
        ("co_250115", 3),
        ("co_250303", 3),
        ("co_250625", 4),
        ("co_260212", 4),
        ("co_260325", 4),
        ("co_260306", 5),
        ("co_260410", 5),
        ("co_260501", 5),
        ("261106", 5),
    ],
)
def test_pipeline_pins_documented_dataset(revision, dataset):
    stage = next(
        s for s in pipeline(revision).stages if s["name"] == "train_uresnet_deghost"
    )
    root = f"/sdf/data/neutrino/icarus/sim/mpvmpr_v{dataset}"
    assert stage["source_list"] == f"{root}/train_file_list.txt"
    assert stage["val_source_list"] == f"{root}/test_file_list.txt"


def test_candidate_ppn_reflection_preserves_both_cryostat_index_domains(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from spine.data import IndexData, Meta, TensorData
    from spine.geo import GeoManager
    from spine.io.augment import AugmentManager

    cfg = spine_config.load_config(
        "include:\n  - train/icarus/uresnet_ppn/train_from_deghost_cache_261106.yaml\n  - train/icarus/modifier/augment_ppn/mod_augment_ppn_261106.yaml\n",
        root_dir=str(CONFIG),
        download=False,
    )
    # The reflection manager needs the center only; no geometry download is
    # needed to test that a moved image frame preserves cross-cryostat indices.
    geo = SimpleNamespace(tpc=SimpleNamespace(center=np.zeros(3)))
    monkeypatch.setattr(GeoManager, "get_instance", lambda: geo)
    options = cfg["io"]["loader"]["dataset"]["augment"]
    for flip in options["stages"]:
        flip["p"] = 1.0
    manager = AugmentManager(**options)
    meta = Meta(
        lower=np.array([-10.0, -2.0, -2.0]),
        upper=np.array([10.0, 8.0, 8.0]),
        size=np.ones(3),
        count=np.array([20, 10, 10]),
    )
    coords = np.array([[1, 1, 1], [2, 2, 2], [11, 3, 3], [12, 4, 4]])
    indices = np.array([0, 2])
    data = {
        "seg_label": TensorData(
            coords=coords.copy(), features=np.arange(4)[:, None], meta=meta
        ),
        "data_calib": TensorData(
            coords=coords[indices].copy(),
            features=np.array([[10.0], [20.0]]),
            meta=meta,
        ),
        "orig_index": IndexData(indices.copy(), span=4),
    }
    state = np.random.get_state()
    try:
        result = manager(data)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(result["orig_index"].features, indices)
    np.testing.assert_array_equal(
        result["data_calib"].coords, result["seg_label"].coords[indices]
    )
    assert len(result["seg_label"]) == 4

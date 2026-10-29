"""SBND deployed-model parity and staged retraining contracts."""

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
    (ROOT / "tests/fixtures/sbnd_inference_baseline.json").read_text()
)
MODEL_REVISIONS = (
    "240720",
    "240814",
    "250328",
    "250901",
    "260304",
    "260316",
    "260501",
    "260521",
)
MODEL_CHAINS = [name for name in BASELINE if name.split("_")[-1] in MODEL_REVISIONS]
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
    monkeypatch.setenv("SBND_DATA_DIR", "/test/sbnd-data")


def load(path):
    return spine_config.load_config_file(str(CONFIG / path), download=False)


@pytest.mark.parametrize("chain", BASELINE)
def test_deployed_inference_is_unchanged(chain):
    assert fingerprint(load(f"infer/sbnd/{chain}.yaml"), CONFIG) == BASELINE[chain]


def test_latest_still_selects_october_deployment(tmp_path):
    path = ConfigManager(ROOT).create_latest_config("sbnd", tmp_path)
    assert fingerprint(spine_config.load_config_file(path, download=False), CONFIG) == (
        BASELINE["full_chain_co_261019"]
    )


@pytest.mark.parametrize("chain", MODEL_CHAINS)
def test_pipeline_model_cache_and_evaluation_contracts(chain):
    revision = chain.split("_")[-1]
    pipeline = PipelineDefinition.load(
        ROOT / f"pipelines/sbnd/full_chain_{revision}.yaml",
        workspace_override=f"/tmp/sbnd-{revision}",
    )
    stages = {s["name"]: s for s in pipeline.stages}
    assert len(stages) == 21
    configs = {name: load(stage["config"]) for name, stage in stages.items()}
    released = load(f"infer/sbnd/{chain}.yaml")
    modules = released["model"]["modules"]
    exported = configs["export_full_chain_weights"]
    evaluation = configs["evaluate_full_chain"]
    assert exported["model"]["weight_path"] is None
    assert exported["model"]["modules"] == evaluation["model"]["modules"] == modules
    assert exported["geo"] == evaluation["geo"] == released["geo"]
    assert modules["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 5
    for component in COMPONENTS:
        stage = stages[f"train_{component}"]
        cfg = configs[f"train_{component}"]
        assert cfg["model"].get("weight_path") is None
        assert set(stage["warm_start"]) <= set(cfg["model"]["modules"])
        assert stage["val_entry_fraction_range"] == [0.0, 0.5]
    assert stages["evaluate_full_chain"]["entry_fraction_range"] == [0.5, 1.0]
    checkpoint = stages["export_full_chain_weights"]["export_weights"]
    assert stages["evaluate_full_chain"]["weight_path"] == checkpoint
    assert stages["report_full_chain"]["checkpoint"] == checkpoint
    # Historical charge-only models get a cache-only identity calibration.
    deghost = configs["cache_train_deghosting"]
    deghost_stages = deghost["model"]["modules"]["chain"]["stages"]
    assert deghost["geo"] == released["geo"]
    assert deghost_stages[0]["config"] == modules["chain"]["stages"][0]["config"]
    assert deghost_stages[1]["config"]["calibration"] == modules.get(
        "calibration", {"stages": []}
    )
    for stage_name, names in {
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
        for name in names:
            actual = dict(configs[stage_name]["model"]["modules"][name])
            actual.pop("model_name", None)
            actual.pop("weight_path", None)
            assert actual == modules[name]
    producer = configs["cache_train_particle_graphs"]
    consumer = configs["train_grappa_inter"]
    assert set(producer["io"]["writer"]["keys"]) == set(
        consumer["io"]["loader"]["dataset"]["keys"]
    )
    assert consumer["model"]["modules"]["grappa_loss"] == modules["grappa_inter_loss"]
    assert set(modules["grappa_inter_loss"]["node_loss"]) == {
        "type",
        "primary",
        "orient",
    }
    for kind in ("target", "valid"):
        key = f"node_orient_{kind}"
        assert consumer["model"]["loss_input"][key] == f"particle_graph_{key}"
    # Keep reconstruction/calorimetry post-processing; optical matching belongs
    # to production inference and is deliberately absent from training metrics.
    expected = copy.deepcopy(released["post"]["stages"])
    expected = [s for s in expected if s["name"] not in {"flash_match", "crt_match"}]
    for post in expected:
        if post["name"] == "match":
            post.update(ghost=True, truth_point_mode="points_adapt", fragment=True)
    assert evaluation["post"]["stages"] == expected + [{"name": "ppn"}]
    assert evaluation["io"]["writer"] is None
    schema = evaluation["io"]["loader"]["dataset"]["schema"]
    assert not {"flashes", "flashes_xa", "crthits", "run_info"} & set(schema)
    for name in ["train_uresnet_ppn", "cache_train_segmentation"]:
        info = configs[name]["io"]["loader"]["dataset"]["primary"]["schema"][
            "clust_label"
        ]["particle_info"]
        # Preserve the deployed SBND parser defaults, including MPR labels.
        assert all(
            info[k]
            for k in (
                "type_include_secondary",
                "type_include_mpr",
                "primary_include_mpr",
            )
        )
    assert (
        configs["build_train_filter"]["filters"]["sparse3d_reco"]["max_count"] == 500000
    )
    if revision < "250000":
        train = "/sdf/data/neutrino/sbnd/simulation/mpvmpr_v01/train.list"
        test = "/sdf/data/neutrino/sbnd/simulation/mpvmpr_v01/test.list"
        profile = "s3df_ampere_full"
    elif revision < "260500":
        train = "/lus/eagle/projects/neutrinoGPU/bearc/simulation/mpvmpr_v02/train/files.txt"
        test = "/lus/eagle/projects/neutrinoGPU/bearc/simulation/mpvmpr_v02/test/larcv/files.txt"
        profile = "s3df_ampere_full"
    else:
        root = "/lus/eagle/projects/neutrinoGPU/jmueller/spine_train/datasets/v10_14_02/generic"
        train, test = f"{root}/train/files.txt", f"{root}/test/files.txt"
        profile = "s3df_ampere_full"
    assert stages["train_uresnet_deghost"]["source_list"] == train
    assert stages["train_uresnet_deghost"]["val_source_list"] == test
    assert stages["train_uresnet_deghost"]["profile"] == profile
    readme = (CONFIG / "infer/sbnd/README.md").read_text()
    assert train in readme and test in readme


def test_charge_only_cache_identity_executes_without_changing_values():
    import numpy as np
    from spine.calib import CalibrationManager
    from spine.geo import GeoManager

    cfg = load("cache/sbnd/uresnet_deghost/deghosting_240720.yaml")
    GeoManager.initialize_or_get(**cfg["geo"])
    options = cfg["model"]["modules"]["chain"]["stages"][1]["config"]["calibration"]
    manager = CalibrationManager(**options)
    points, charge = np.zeros((2, 3)), np.array([10.0, 25.0])
    actual_points, actual_charge = manager(
        points, charge, sources=np.array([[0, 0], [0, 1]])
    )
    np.testing.assert_array_equal(actual_points, points)
    np.testing.assert_array_equal(actual_charge, charge)


@pytest.mark.parametrize(
    "version",
    ["240720", "240814", "250328", "250901", "260304", "260316", "260501", "260521"],
)
def test_standalone_components_preserve_released_networks_and_losses(version):
    modules = load(f"model/sbnd/full_chain/model_{version}.yaml")["model"]["modules"]
    for component in COMPONENTS:
        standalone = load(f"model/sbnd/{component}/model_{version}.yaml")["model"][
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
            else ("grappa" if component.startswith("grappa") else component)
        )
        expected = copy.deepcopy(modules[component])
        if component == "grappa_track":
            expected["nodes"]["make_groups"] = False
            expected["nodes"].pop("grouping_method")
        elif component == "grappa_inter":
            expected["nodes"].pop("grouping_through_track")
        assert standalone[key] == expected
        assert standalone[key + "_loss"] == modules[component + "_loss"]


def test_one_pipeline_per_model_including_candidate():
    pipelines = {
        p.stem.removeprefix("full_chain_")
        for p in (ROOT / "pipelines/sbnd").glob("*.yaml")
    }
    models = {
        p.stem.removeprefix("model_")
        for p in (CONFIG / "model/sbnd/full_chain").glob("model_*.yaml")
    }
    assert pipelines == models == set(MODEL_REVISIONS) | {"261029"}


@pytest.mark.parametrize("chain", BASELINE)
def test_each_inference_revision_keeps_its_evaluator(chain):
    revision = chain.split("_")[-1]
    released = load(f"infer/sbnd/{chain}.yaml")
    evaluation = load(f"test/sbnd/full_chain/evaluate_{revision}.yaml")
    assert evaluation["model"]["modules"] == released["model"]["modules"]


def test_candidate_calibration_targets_and_pipeline_are_consistent():
    pipeline = PipelineDefinition.load(
        ROOT / "pipelines/sbnd/full_chain_261029.yaml",
        workspace_override="/tmp/sbnd-candidate",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    configs = {name: load(stage["config"]) for name, stage in stages.items()}
    model = configs["export_full_chain_weights"]
    modules = model["model"]["modules"]
    released = load("model/sbnd/full_chain/model_260521.yaml")["model"]["modules"]
    for component in COMPONENTS:
        assert modules[component] == released[component]
    assert modules["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 5
    calibration = modules["calibration"]
    assert calibration["stages"][0] == {"name": "gain", "gain": 1 / 0.02019}
    assert calibration["stages"][1:] == released["calibration"]["stages"][1:]
    cache = configs["cache_train_deghosting"]
    assert (
        cache["model"]["modules"]["chain"]["stages"][1]["config"]["calibration"]
        == calibration
    )
    producer = configs["cache_train_particle_graphs"]
    trainer = configs["train_grappa_inter"]
    objective = modules["grappa_inter_loss"]
    assert (
        producer["model"]["modules"]["grappa_inter_loss"]
        == trainer["model"]["modules"]["grappa_loss"]
        == objective
    )
    assert set(objective["node_loss"]) == {"type", "primary", "orient"}
    for task in ["type", "primary"]:
        assert objective["node_loss"][task]["min_iou"] == 0.5
        assert objective["node_loss"][task]["match_target"] == "group"
    assert objective["node_loss"]["primary"]["use_closest"] is True
    assert objective["node_loss"]["primary"]["secondary_label"] == 0
    standalone = load("model/sbnd/grappa_inter/model_261029.yaml")
    assert (
        "min_iou"
        not in standalone["model"]["modules"]["grappa_loss"]["node_loss"]["type"]
    )
    assert set(producer["io"]["writer"]["keys"]) == set(
        trainer["io"]["loader"]["dataset"]["keys"]
    )
    evaluation = configs["evaluate_full_chain"]
    assert evaluation["model"]["modules"] == modules
    assert (
        evaluation["geo"]
        == model["geo"]
        == cache["geo"]
        == {"detector": "sbnd", "tag": "sbndv2-6"}
    )
    assert not {"apply_calibrations", "flash_match"} & {
        p["name"] for p in evaluation["post"]["stages"]
    }
    for name in [
        "train_uresnet_ppn",
        "cache_train_segmentation",
        "evaluate_full_chain",
    ]:
        dataset = configs[name]["io"]["loader"]["dataset"]
        schema = dataset.get("primary", dataset)["schema"]
        options = schema["clust_label"]["particle_info"]
        assert not any(
            options[key]
            for key in [
                "type_include_secondary",
                "type_include_mpr",
                "primary_include_mpr",
            ]
        )
    assert configs["train_uresnet_ppn"]["base"]["epochs"] == 100.0
    for component, mod, backbone in [
        ("uresnet_deghost", "augment_uresnet", "uresnet"),
        ("uresnet_ppn", "augment_ppn", "uresnet_ppn"),
        ("graph_spice", "augment_graph_spice", "graph_spice"),
    ]:
        stage = stages[f"train_{component}"]
        assert not stage.get("apply_mods")
        default = configs[f"train_{component}"]
        assert "augment" not in default["io"]["loader"]["dataset"]
        default_net = default["model"]["modules"][backbone]
        if component == "uresnet_ppn":
            default_net = default_net["uresnet"]
        elif component == "graph_spice":
            default_net = default_net["embedder"]["uresnet"]
        assert "lattice" not in default_net
        # Optional modifiers remain available for later augmentation validation.
        cfg = spine_config.load_config(
            "include:\n  - "
            + stage["config"]
            + "\n  - train/sbnd/modifier/"
            + mod
            + "/mod_"
            + mod
            + "_261029.yaml\n",
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
        assert [f["axis"] for f in flips] == [0, 1, 2]
        assert all(
            f["keep_meta"] == (component != "uresnet_ppn")
            and f["use_geo_center"]
            and f["p"] == 0.5
            for f in flips
        )
    for stage in pipeline.stages:
        if stage.get("stage") == "train":
            assert stage["config"].endswith("261029.yaml")
            assert stage["val_entry_fraction_range"] == [0.0, 0.5]
    assert stages["evaluate_full_chain"]["entry_fraction_range"] == [0.5, 1.0]
    checkpoint = stages["export_full_chain_weights"]["export_weights"]
    assert (
        stages["evaluate_full_chain"]["weight_path"]
        == stages["report_full_chain"]["checkpoint"]
        == checkpoint
    )
    assert "deghosting" in configs["report_full_chain"]["metrics"]
    assert (
        "shower_fragment_primary"
        in configs["report_full_chain"]["metrics"]["nodes"]["tasks"]
    )


def test_candidate_ppn_reflection_keeps_raw_to_cached_index_mapping():
    import numpy as np
    from spine.data import Meta, TensorData
    from spine.geo import GeoManager
    from spine.io.augment import AugmentManager

    cfg = spine_config.load_config(
        "include:\n  - train/sbnd/uresnet_ppn/train_from_deghost_cache_261029.yaml\n  - train/sbnd/modifier/augment_ppn/mod_augment_ppn_261029.yaml\n",
        root_dir=str(CONFIG),
        download=False,
    )
    geo = GeoManager.initialize_or_get(**cfg["geo"])
    options = cfg["io"]["loader"]["dataset"]["augment"]
    for flip in options["stages"]:
        flip["p"] = 1.0
    manager = AugmentManager(**options)
    # An asymmetric image frame would crop rows under a fixed-frame reflection.
    lower = geo.tpc.center - 2.0
    meta = Meta(
        lower=lower, upper=lower + 10, size=np.ones(3), count=np.array([10] * 3)
    )
    coords = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    indices = np.array([0, 2])
    data = {
        "seg_label": TensorData(
            coords=coords.copy(), features=np.arange(3)[:, None], meta=meta
        ),
        "data_calib": TensorData(
            coords=coords[indices].copy(),
            features=np.array([[10.0], [20.0]]),
            meta=meta,
        ),
        "orig_index": indices.copy(),
    }
    state = np.random.get_state()
    try:
        result = manager(data)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(result["orig_index"], indices)
    np.testing.assert_array_equal(
        result["data_calib"].coords, result["seg_label"].coords[indices]
    )
    np.testing.assert_array_equal(result["data_calib"].features, [[10.0], [20.0]])
    assert len(result["seg_label"].features) == 3


@pytest.mark.parametrize("revision", [*MODEL_REVISIONS, "261029"])
def test_sbnd_starting_batches_and_s3df_profiles(revision):
    pipeline = PipelineDefinition.load(
        ROOT / f"pipelines/sbnd/full_chain_{revision}.yaml",
        workspace_override=f"/tmp/sbnd-budget-{revision}",
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    for stage in pipeline.stages:
        assert stage["profile"].startswith("s3df_")
        assert "gpus_per_node" not in stage
        if stage["name"].startswith(("scan_", "build_", "export_", "report_")):
            assert stage["profile"] == "s3df_milano"
    for component, batch, profile in [
        ("uresnet_deghost", 16, "s3df_ampere_full"),
        ("uresnet_ppn", 25, "s3df_ampere_full"),
        ("graph_spice", 50, "s3df_ampere_full"),
        ("grappa_shower", 32, "s3df_ampere"),
        ("grappa_track", 32, "s3df_ampere"),
        ("grappa_inter", 128, "s3df_ampere"),
    ]:
        stage = stages[f"train_{component}"]
        loader = load(stage["config"])["io"]["loader"]
        key = "batch_size" if component == "uresnet_deghost" else "minibatch_size"
        assert loader[key] == batch
        assert stage["profile"] == profile
        assert stage["time"] == (
            "5-00:00:00" if component == "uresnet_deghost" else "3-00:00:00"
        )


def test_cli_source_lists_propagate_to_every_sbnd_input_and_report():
    train, validation = "/sdf/new/train.list", "/sdf/new/test.list"
    pipeline = PipelineDefinition.load(
        ROOT / "pipelines/sbnd/full_chain_261029.yaml",
        workspace_override="/tmp/sbnd-cli-sources",
        source_list=train,
        val_source_list=validation,
    )
    stages = {stage["name"]: stage for stage in pipeline.stages}
    for split, source in [("train", train), ("validation", validation)]:
        for operation in ["scan", "build"]:
            assert stages[f"{operation}_{split}_filter"]["source_list"] == source
        assert stages[f"cache_{split}_deghosting"]["source_list"] == source
        for step in ["segmentation", "fragment_graphs", "particle_graphs"]:
            assert (
                stages[f"cache_{split}_{step}"]["sources"]["primary"]["source_list"]
                == source
            )
    assert stages["train_uresnet_deghost"]["source_list"] == train
    assert stages["train_uresnet_deghost"]["val_source_list"] == validation
    assert stages["train_uresnet_ppn"]["sources"]["primary"]["source_list"] == train
    assert (
        stages["train_uresnet_ppn"]["validation_sources"]["primary"]["source_list"]
        == validation
    )
    assert stages["evaluate_full_chain"]["source_list"] == validation
    assert stages["report_full_chain"]["dataset"] == validation

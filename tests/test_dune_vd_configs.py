"""DUNE-VD candidate geometry, cache alignment, and HD inheritance contracts."""

from pathlib import Path

import pytest

from src.pipeline import PipelineDefinition

spine_config = pytest.importorskip("spine.config")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
VD = "dune-vd-10kt-1x8x6"
HD = "dune-hd-10kt-1x2x6"
GEO = {"detector": VD, "tag": "dunevd10kt_3view_30deg_v3_refactored_1x8x6ref"}


@pytest.fixture(autouse=True)
def config_environment(monkeypatch):
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(CONFIG))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(ROOT))


def load(path):
    return spine_config.load_config_file(str(CONFIG / path), download=False)


def pipeline(**kwargs):
    return PipelineDefinition.load(
        ROOT / f"pipelines/{VD}/full_chain_261117.yaml",
        workspace_override="/tmp/vd-training",
        **kwargs,
    )


def test_pipeline_geometry_and_all_recipes_resolve():
    from spine.geo.factories import geo_factory

    geo_factory(**GEO)
    assert load(f"convert/{VD}/truth_260911.yaml")["geo"] == GEO
    stages = pipeline().stages
    assert len(stages) == 21
    for stage in stages:
        cfg = load(stage["config"])
        if "model" in cfg:
            assert cfg["geo"] == GEO
        if stage.get("stage") == "train":
            assert cfg["model"].get("weight_path") is None
            assert set(stage["warm_start"]) <= set(cfg["model"]["modules"])
            assert stage["val_entry_fraction_range"] == [0.0, 0.5]
        if "io" in cfg:
            # Preserve raw event identity across materialized orig_index/graphs.
            assert cfg["io"]["loader"]["collate_fn"] == "all"
    evaluation = load(f"test/{VD}/full_chain/evaluate_261117.yaml")
    assert evaluation["model"]["network_input"]["clust_label"] == "clust_label"
    assert evaluation["io"]["writer"] is None


def test_calibration_is_shared_by_cache_export_and_evaluation(tmp_path, monkeypatch):
    # A VD-only calibration change must reach every calibrated execution path.
    import shutil

    configs = tmp_path / "config"
    shutil.copytree(CONFIG, configs)
    path = configs / f"model/{VD}/calibration/calibration_261117.yaml"
    path.write_text(path.read_text().replace("198.21605550049554", "123.456"))
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(configs))

    def local(path):
        return spine_config.load_config_file(str(configs / path), download=False)

    expected = local(f"model/{VD}/calibration/calibration_261117.yaml")
    assert expected["stages"][0]["gain"] == 123.456
    model = local(f"model/{VD}/full_chain/model_261117.yaml")
    cache = local(f"cache/{VD}/uresnet_deghost/deghosting_261117.yaml")
    evaluation = local(f"test/{VD}/full_chain/evaluate_261117.yaml")
    assert model["model"]["modules"]["calibration"] == expected
    assert evaluation["model"]["modules"] == model["model"]["modules"]
    calibration_stage = next(
        s
        for s in cache["model"]["modules"]["chain"]["stages"]
        if s["name"] == "calibration"
    )
    assert calibration_stage["config"]["calibration"] == expected
    assert "calibration" not in {s["name"] for s in evaluation["post"]["stages"]}


def test_hd_networks_and_cached_objectives_are_preserved():
    model = load(f"model/{VD}/full_chain/model_261117.yaml")["model"]["modules"]
    hd = load(f"model/{HD}/full_chain/model_261022.yaml")["model"]["modules"]
    assert model == hd  # Calibration is explicitly provisional HD baseline.
    assert model["grappa_inter"]["gnn_model"]["node_pred"]["type"] == 6
    producer = load(f"cache/{VD}/grappa_shower_track/particle_graphs_261117.yaml")
    trainer = load(f"train/{VD}/grappa_inter/train_from_particle_cache_261117.yaml")
    loss = model["grappa_inter_loss"]
    assert producer["model"]["modules"]["grappa_inter_loss"] == loss
    assert trainer["model"]["modules"]["grappa_loss"] == loss
    assert set(loss["node_loss"]) == {"type", "primary", "orient"}
    assert set(producer["io"]["writer"]["keys"]) == set(
        trainer["io"]["loader"]["dataset"]["keys"]
    )
    for task in ("type", "primary"):
        assert loss["node_loss"][task]["min_iou"] == 0.5
    for key in ("node_orient_target", "node_orient_valid"):
        assert trainer["model"]["loss_input"][key] == f"particle_graph_{key}"


def test_source_overrides_and_checkpoint_flow():
    stages = {
        s["name"]: s
        for s in pipeline(
            source_list="/vd/train.txt", val_source_list="/vd/test.txt"
        ).stages
    }
    for split, path in (("train", "/vd/train.txt"), ("validation", "/vd/test.txt")):
        scan = stages[f"scan_{split}_filter"]
        assert scan["source_list"] == path
        assert scan["time"] == "02:00:00"
        assert stages[f"cache_{split}_deghosting"]["source_list"] == path
        for suffix in ("segmentation", "fragment_graphs", "particle_graphs"):
            assert (
                stages[f"cache_{split}_{suffix}"]["sources"]["primary"]["source_list"]
                == path
            )
    assert stages["train_uresnet_deghost"]["val_source_list"] == "/vd/test.txt"
    evaluation = stages["evaluate_full_chain"]
    assert evaluation["source_list"] == "/vd/test.txt"
    assert evaluation["entry_fraction_range"] == [0.5, 1.0]
    assert evaluation["ntasks"] == 4
    assert "files_per_task" not in evaluation
    checkpoint = stages["export_full_chain_weights"]["export_weights"]
    assert evaluation["weight_path"] == checkpoint
    assert stages["report_full_chain"]["checkpoint"] == checkpoint
    assert stages["report_full_chain"]["dataset"] == "/vd/test.txt"


def test_default_inputs_do_not_silently_select_hd_data():
    stages = {s["name"]: s for s in pipeline().stages}
    assert (
        stages["scan_train_filter"]["source_list"]
        == "/sdf/data/neutrino/dune/sim/vd/mpvmpr_v0/train_file_list.txt"
    )
    assert (
        stages["scan_validation_filter"]["source_list"]
        == "/sdf/data/neutrino/dune/sim/vd/mpvmpr_v0/test_file_list.txt"
    )

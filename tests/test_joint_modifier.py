"""Validation for joint-dataset inference modifiers."""

from spine.config import load_config


def test_nd_lar_joint_modifier_wraps_inference_dataset(workspace_root, monkeypatch):
    config_root = workspace_root / "config"
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(config_root))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(workspace_root))
    config = load_config(
        """
include:
  - infer/nd-lar/full_chain_260409.yaml
  - infer/nd-lar/modifier/joint/mod_joint_240819.yaml
""",
        root_dir=str(config_root),
    )

    dataset = config["io"]["loader"]["dataset"]
    assert dataset["name"] == "joint"
    assert dataset["primary"] == {"file_keys": None}
    assert dataset["secondary"] == {"file_keys": None}
    assert dataset["base"]["name"] == "larcv"
    assert dataset["base"]["file_keys"] is None
    assert {"run_info", "flashes", "trigger"}.isdisjoint(dataset["base"]["schema"])
    assert {"run_info", "flashes", "trigger"}.isdisjoint(config["io"]["writer"]["keys"])


def test_nd_lar_dataset_refactor_preserves_standard_inference(
    workspace_root, monkeypatch
):
    config_root = workspace_root / "config"
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(config_root))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(workspace_root))
    config = load_config(
        "include: infer/nd-lar/full_chain_260409.yaml\n",
        root_dir=str(config_root),
    )

    dataset = config["io"]["loader"]["dataset"]
    assert dataset["name"] == "larcv"
    assert dataset["file_keys"] is None
    assert set(dataset["schema"]) == {
        "data",
        "seg_label",
        "ppn_label",
        "clust_label",
        "coord_label",
        "particles",
        "neutrinos",
        "meta",
        "run_info",
        "trigger",
        "flashes",
    }

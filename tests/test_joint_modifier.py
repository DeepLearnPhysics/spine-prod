"""Validation for joint-dataset inference modifiers."""

import pytest
import yaml
from spine.config import load_config

JOINT_CASES = (
    ("generic", "full_chain_240718.yaml", "240718", {"run_info"}, True),
    ("generic", "full_chain_240805.yaml", "240805", {"run_info"}, False),
    (
        "2x2",
        "full_chain_240719.yaml",
        "240719",
        {"run_info", "trigger", "flashes"},
        True,
    ),
    (
        "2x2",
        "full_chain_240819.yaml",
        "240819",
        {"run_info", "trigger", "flashes"},
        False,
    ),
    (
        "icarus",
        "full_chain_240719.yaml",
        "240719",
        {"run_info", "flashes", "crthits"},
        True,
    ),
    (
        "icarus",
        "full_chain_co_260501.yaml",
        "240812",
        {"run_info", "flashes", "crthits"},
        False,
    ),
    (
        "sbnd",
        "full_chain_240720.yaml",
        "240720",
        {"run_info", "flashes", "flashes_xa", "crthits"},
        False,
    ),
    (
        "sbnd",
        "full_chain_co_260521.yaml",
        "240918",
        {"run_info", "flashes", "flashes_xa", "crthits"},
        False,
    ),
    (
        "dune-hd-10kt-1x2x6",
        "full_chain_260626.yaml",
        "260202",
        {"run_info", "flashes", "crthits"},
        False,
    ),
    (
        "protodune-sp",
        "full_chain_260906.yaml",
        "260210",
        {"run_info", "flashes", "crthits"},
        False,
    ),
    (
        "protodune-vd",
        "full_chain_260128.yaml",
        "260118",
        {"run_info", "flashes", "crthits"},
        False,
    ),
)


@pytest.mark.parametrize(
    "detector,config_name,modifier_version,excluded,point_tagging",
    JOINT_CASES,
    ids=lambda value: str(value),
)
def test_joint_modifiers_wrap_detector_inference_datasets(
    workspace_root,
    monkeypatch,
    detector,
    config_name,
    modifier_version,
    excluded,
    point_tagging,
):
    config_root = workspace_root / "config"
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(config_root))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(workspace_root))
    config = load_config(
        f"""
include:
  - infer/{detector}/{config_name}
  - infer/{detector}/modifier/joint/mod_joint_{modifier_version}.yaml
""",
        root_dir=str(config_root),
        download=False,
    )

    dataset = config["io"]["loader"]["dataset"]
    assert dataset["name"] == "joint"
    assert dataset["primary"] == {"file_keys": None}
    assert dataset["secondary"] == {"file_keys": None}
    assert dataset["base"]["name"] == "larcv"
    assert excluded.isdisjoint(dataset["base"]["schema"])
    assert excluded.isdisjoint(config["io"]["writer"]["keys"])
    assert (
        dataset["base"]["schema"]["ppn_label"]["include_point_tagging"] is point_tagging
    )
    assert "flash_match" not in config.get("post", {})


def test_all_detector_joint_modifiers_are_mc_only(workspace_root):
    joint_root = workspace_root / "config" / "infer"
    modifier_paths = sorted(joint_root.glob("*/modifier/joint/mod_joint_[0-9]*.yaml"))
    detectors = {path.parts[-4] for path in modifier_paths}

    assert detectors == {
        "2x2",
        "dune-hd-10kt-1x2x6",
        "generic",
        "icarus",
        "nd-lar",
        "protodune-sp",
        "protodune-vd",
        "sbnd",
    }
    for path in modifier_paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["__meta__"]["applies_to"] == ["mc"]


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

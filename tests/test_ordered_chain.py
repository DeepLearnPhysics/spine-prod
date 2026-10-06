"""Contracts for maintained ordered chains and SBND pre-deghosting scaling."""

from pathlib import Path

import pytest
import yaml

spine_config = pytest.importorskip("spine.config")

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
INFERENCE_CONFIGS = sorted((CONFIG_ROOT / "infer").glob("*/full_chain_*.yaml"))
SBND_CONFIGS = sorted((CONFIG_ROOT / "infer" / "sbnd").glob("full_chain_*.yaml"))
SCALE_MODIFIER = "infer/sbnd/modifier/predeghost_scale/mod_predeghost_scale_261005.yaml"


@pytest.fixture(autouse=True)
def config_environment(monkeypatch):
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(CONFIG_ROOT))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(CONFIG_ROOT.parent))
    monkeypatch.setenv("SBND_DATA_DIR", "/test/sbnd-data")


def compose(*paths):
    return spine_config.load_config(
        yaml.safe_dump({"include": [str(path) for path in paths]}),
        root_dir=str(CONFIG_ROOT),
        download=False,
    )


@pytest.mark.parametrize("path", INFERENCE_CONFIGS, ids=lambda path: str(path))
def test_inference_chains_have_ordered_stages_and_valid_references(path):
    """Every maintained inference variant keeps its complete reconstruction path."""
    modules = compose(path)["model"]["modules"]
    assert set(modules["chain"]) == {"stages"}
    stages = modules["chain"]["stages"]
    names = [stage["name"] for stage in stages]
    assert len(names) == len(set(names))
    assert names[-4:] == [
        "segmentation",
        "fragmentation",
        "particle_aggregation",
        "interaction_aggregation",
    ]
    if "deghosting" in names:
        assert names[0] == "deghosting"
    if "calibration_before_segmentation" in names:
        assert names.index("calibration_before_segmentation") + 1 == names.index(
            "segmentation"
        )
        assert "stage" not in modules["calibration"]
    for stage in stages:
        uses = stage.get("uses", [])
        if isinstance(uses, str):
            uses = [uses]
        assert all(key in modules for key in uses)
        loss = stage.get("loss", {})
        losses = [loss] if isinstance(loss, str) else loss.values()
        assert all(key in modules for key in losses)


@pytest.mark.parametrize("path", SBND_CONFIGS, ids=lambda path: path.name)
def test_predeghost_scale_preserves_existing_chain_and_accepts_factor(path):
    """A changed factor affects only the added stage on every SBND model."""
    baseline = compose(path)
    scaled = compose(path, SCALE_MODIFIER)
    spine_config.apply_overrides(
        scaled,
        [
            "model.modules.chain.stages~={update: {name: predeghost_scale, "
            "changes: {config: {calibration: {stages: [{name: gain, gain: 1.1}]}}}}}"
        ],
    )
    modules = scaled["model"]["modules"]
    stages = modules["chain"]["stages"]
    predeghost = stages.pop(0)
    assert predeghost == {
        "name": "predeghost_scale",
        "provider": "calibration",
        "config": {
            "mode": "apply",
            "calibration": {"stages": [{"name": "gain", "gain": 1.1}]},
        },
    }
    assert stages[0]["name"] == "deghosting"
    assert modules == baseline["model"]["modules"]
    assert scaled["post"] == baseline["post"]
    assert scaled["model"]["network_input"]["meta"] == "meta"


@pytest.mark.parametrize("scale_first", [True, False])
@pytest.mark.parametrize(
    "base,modifiers",
    [
        ("full_chain_co_260521.yaml", ["data/mod_data_260501.yaml"]),
        ("full_chain_co_250901.yaml", ["data_sim_gain/mod_data_sim_gain_260910.yaml"]),
        (
            "full_chain_co_260316.yaml",
            [
                "gain_pos_scale/mod_gain_pos_scale_260818.yaml",
                "smearing/mod_smearing_260902.yaml",
            ],
        ),
    ],
)
def test_scale_composes_with_existing_calibration_modifiers(
    base, modifiers, scale_first
):
    """Both insertion orders preserve the pre-segmentation calibration slot."""
    modifiers = [f"infer/sbnd/modifier/{modifier}" for modifier in modifiers]
    base = f"infer/sbnd/{base}"
    baseline = compose(base, *modifiers)
    sequence = (
        [SCALE_MODIFIER, *modifiers] if scale_first else [*modifiers, SCALE_MODIFIER]
    )
    scaled = compose(base, *sequence)
    modules = scaled["model"]["modules"]
    stages = modules["chain"]["stages"]
    assert stages.pop(0)["name"] == "predeghost_scale"
    assert modules == baseline["model"]["modules"]
    assert [stage["name"] for stage in stages[:3]] == [
        "deghosting",
        "calibration_before_segmentation",
        "segmentation",
    ]

"""Release migration contracts, including actual manager construction/execution."""

import json
import warnings
from pathlib import Path

import pytest
import yaml

from scripts.check_spine_version import validate_version
from tests.config_migration import fingerprint

spine_config = pytest.importorskip("spine.config")
ROOT = Path(__file__).resolve().parents[1] / "config"
BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/spine140_baseline.json").read_text()
)["cases"]


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(ROOT))
    monkeypatch.setenv("SPINE_PROD_BASEDIR", str(ROOT.parent))
    monkeypatch.setenv("SBND_DATA_DIR", "/test/sbnd-data")
    monkeypatch.setenv("ICARUS_DATA_DIR", "/test/icarus-data")


def compose(*paths):
    # Incomplete fragments belong in a composition. No unexpected warning is
    # allowed in the resolved production cases retained in the fixture.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return spine_config.load_config(
            yaml.safe_dump({"include": list(paths)}),
            root_dir=str(ROOT),
            download=False,
        )


@pytest.mark.parametrize("case", BASELINE, ids=lambda c: " + ".join(c["includes"]))
def test_pre_migration_behavior(case):
    """Compare providers, values, order, inputs and outputs, not YAML spelling."""
    if "rejected" in case:
        with pytest.raises(spine_config.errors.ConfigError, match=case["rejected"]):
            compose(*case["includes"])
    else:
        if case.get("direct"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                cfg = spine_config.load_config_file(
                    str(ROOT / case["includes"][0]), download=False
                )
        else:
            cfg = compose(*case["includes"])
        assert fingerprint(cfg, ROOT) == case["sha256"]


def test_split_sbnd_data_modifier_has_time_containment():
    cfg = compose(
        "infer/sbnd/full_chain_co_250818.yaml",
        "infer/sbnd/modifier/data/mod_data_250818.yaml",
    )
    stages = cfg["post"]["stages"]
    matching = [s for s in stages if s["name"] == "time_containment"]
    assert len(matching) == 1
    assert matching[0]["run_mode"] == "reco"


@pytest.mark.parametrize("version", ["1.3.9", "1.4.0rc1", "1.4.0.dev1", "unknown"])
def test_old_runtime_rejected(version):
    with pytest.raises(RuntimeError):
        validate_version(version)


@pytest.mark.parametrize(
    "version", ["1.4.0", "v1.4.0", "1.4.0+local", "1.4.1", "2.0.0"]
)
def test_released_runtime_accepted(version):
    validate_version(version)


def test_calibration_managers_execute_real_sbnd_gain():
    pytest.importorskip("torch")
    import numpy as np
    from spine.calib import CalibrationManager
    from spine.geo import GeoManager

    cfg = compose(
        "infer/sbnd/full_chain_co_260501.yaml",
        "infer/sbnd/modifier/predeghost_scale/mod_predeghost_scale_261005.yaml",
    )
    GeoManager.initialize_or_get(**cfg["geo"])
    modules = cfg["model"]["modules"]
    pre = CalibrationManager(**modules["chain"]["stages"][0]["config"]["calibration"])
    later = CalibrationManager(**modules["calibration"])
    assert list(later.modules) == [s["name"] for s in modules["calibration"]["stages"]]
    points = np.zeros((2, 3))
    charges = np.array([10.0, 20.0])
    sources = np.array([[0, 0], [0, 1]])
    _, scaled = pre(points, charges, sources=sources)
    np.testing.assert_allclose(scaled, 1.03 * charges)
    _, restored = pre(points, scaled, sources=sources, inverse=True)
    np.testing.assert_allclose(restored, charges)
    # A second independent manager can calibrate the preserved charge again.
    _, calibrated = later(points, charges, sources=sources)
    assert np.all(np.isfinite(calibrated))


def test_post_and_analysis_managers_construct(tmp_path):
    pytest.importorskip("torch")
    from spine.ana import AnaManager
    from spine.post import PostManager

    cfg = compose(
        "infer/generic/full_chain_240718.yaml",
        "test/common/full_chain/analyzers_v1.yaml",
    )
    post = PostManager(cfg["post"])
    ana = AnaManager(cfg["ana"], log_dir=str(tmp_path), prefix="smoke")
    assert list(post.modules) == [s["name"] for s in cfg["post"]["stages"]]
    assert list(ana.modules) == [s["name"] for s in cfg["ana"]["stages"]]


def test_training_augmentation_constructs_and_preserves_features():
    pytest.importorskip("torch")
    import numpy as np
    from spine.data import Meta, TensorData
    from spine.geo import GeoManager
    from spine.io.augment import AugmentManager

    cfg = compose("train/nd-lar/uresnet/train_augmented_260912.yaml")
    geo = GeoManager.initialize_or_get(**cfg["geo"])
    augment = cfg["io"]["loader"]["dataset"]["augment"]
    manager = AugmentManager(**augment)
    assert len(manager.modules) == len(augment["stages"])
    # Fixed RNG state makes this runtime smoke repeatable without changing p.
    state = np.random.get_state()
    try:
        np.random.seed(7)
        center = geo.tpc.center
        meta = Meta(
            lower=center - 500.0,
            upper=center + 500.0,
            size=np.ones(3),
            count=np.array([1000] * 3),
        )
        coords = np.array([[500, 500, 500], [501, 501, 501]])
        features = np.array([[10.0], [20.0]])
        data = {
            "meta": meta,
            "data": TensorData(coords=coords, features=features.copy(), meta=meta),
        }
        result = manager(data)
        np.testing.assert_array_equal(result["data"].features, features)
        assert result["data"].coords.shape == coords.shape
    finally:
        np.random.set_state(state)

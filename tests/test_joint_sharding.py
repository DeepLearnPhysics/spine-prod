"""Balanced shards preserve entry coverage across unequal file boundaries."""

from unittest.mock import patch

import pytest

from src.joint_sharding import plan_shards, survey_sources


def test_primary_file_limit_and_disjoint_secondary_ranges():
    sources = {"primary": ["p0", "p1", "p2", "p3", "p4"], "secondary": ["s0", "s1"]}
    shards = plan_shards(sources, {"primary": [3, 4, 2, 6, 5], "secondary": [9, 20]}, 2)
    assert [s["entries"] for s in shards] == [7, 8, 5]
    assert [s["primary"]["files"] for s in shards] == [
        ["p0", "p1"],
        ["p2", "p3"],
        ["p4"],
    ]
    assert [s["secondary"] for s in shards] == [
        {"files": ["s0"], "n_skip": 0},
        {"files": ["s0", "s1"], "n_skip": 7},
        {"files": ["s1"], "n_skip": 6},
    ]
    assert [s["start"] for s in shards] == [0, 7, 15]


def test_secondary_exhaustion_truncates_only_last_group():
    sources = {"primary": ["p0", "p1", "p2"], "secondary": ["s0", "s1"]}
    shards = plan_shards(sources, {"primary": [5, 5, 5], "secondary": [0, 7]}, 1)
    assert [s["entries"] for s in shards] == [5, 2]
    assert [s["secondary"]["n_skip"] for s in shards] == [0, 5]
    assert all(s["secondary"]["files"] == ["s1"] for s in shards)


def test_empty_primary_groups_and_exact_secondary_boundary():
    sources = {"primary": ["p0", "p1", "p2"], "secondary": ["s0", "s1", "s2"]}
    shards = plan_shards(sources, {"primary": [0, 5, 5], "secondary": [5, 0, 5]}, 1)
    assert [s["secondary"]["files"] for s in shards] == [["s0"], ["s2"]]
    assert [s["secondary"]["n_skip"] for s in shards] == [0, 0]
    with pytest.raises(ValueError, match="no usable"):
        plan_shards(sources, {"primary": [0, 0, 0], "secondary": [1, 1, 1]}, 1)


def test_survey_cache_reused_and_invalidated(tmp_path):
    source = tmp_path / "input.root"
    source.write_bytes(b"first")
    sources = {"primary": [str(source)], "secondary": [str(source)]}
    cache = tmp_path / "counts.json"
    with patch("src.joint_sharding.count_root_entries", return_value=5) as count:
        assert survey_sources(sources, cache) == {"primary": [5], "secondary": [5]}
        survey_sources(sources, cache)
        assert count.call_count == 1
        source.write_bytes(b"changed size")
        survey_sources(sources, cache)
        assert count.call_count == 2


def test_root_survey_checks_tree_counts(tmp_path):
    uproot = pytest.importorskip("uproot")
    import numpy as np

    from src.joint_sharding import count_root_entries

    path = tmp_path / "events.root"
    with uproot.recreate(path) as source:
        for name in ("sparse3d_data_tree", "particle_truth_tree"):
            source.mktree(name, {"entry": "int64"})
            source[name].extend({"entry": np.arange(7)})
    assert count_root_entries(path) == 7
    with uproot.update(path) as source:
        source.mktree("bad_tree", {"entry": "int64"})
    # A sorted reference tree is enough for sharding, even with other counts.
    assert count_root_entries(path) == 0
    with pytest.raises(ValueError, match="equally sized"):
        count_root_entries(path, validate_trees=True)


def test_full_validation_upgrades_reference_cache(tmp_path):
    """A cached reference count cannot satisfy a subsequent full validation."""
    source = tmp_path / "input.root"
    source.touch()
    cache = tmp_path / "counts.json"
    sources = {"primary": [str(source)]}
    with patch("src.joint_sharding.count_root_entries", return_value=5) as count:
        survey_sources(sources, cache)
        survey_sources(sources, cache, validate_trees=True)
        assert count.call_count == 2
        count.assert_called_with(str(source), validate_trees=True)
        survey_sources(sources, cache)
        survey_sources(sources, cache, validate_trees=True)
        assert count.call_count == 2


@pytest.mark.parametrize("backend", ["uproot", "ROOT"])
@pytest.mark.parametrize("validate", [False, True])
def test_reference_survey_opens_only_one_tree(monkeypatch, backend, validate):
    """Both backends stop at the sorted reference unless validation is requested."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.joint_sharding import count_root_entries

    opened = []
    counts = {"a_tree": 7, "b_tree": 9}

    def get_tree(name):
        opened.append(name)
        return SimpleNamespace(
            num_entries=counts[name],
            GetEntries=lambda: counts[name],
            InheritsFrom=lambda _: True,
        )

    source = MagicMock()
    source.__enter__.return_value = source
    if backend == "uproot":
        source.keys.return_value = ["b_tree", "a_tree"]
        source.__getitem__.side_effect = get_tree
        monkeypatch.setitem(
            sys.modules, "uproot", SimpleNamespace(open=lambda _: source)
        )
    else:
        source.IsZombie.return_value = False
        source.GetListOfKeys.return_value = [
            SimpleNamespace(GetName=lambda name=name: name) for name in reversed(counts)
        ]
        source.Get.side_effect = get_tree
        monkeypatch.setitem(sys.modules, "uproot", None)
        monkeypatch.setitem(
            sys.modules,
            "ROOT",
            SimpleNamespace(TFile=SimpleNamespace(Open=lambda *_: source)),
        )
    if validate:
        with pytest.raises(ValueError, match="equally sized"):
            count_root_entries("input.root", validate_trees=True)
        assert opened == ["a_tree", "b_tree"]
    else:
        assert count_root_entries("input.root") == 7
        assert opened == ["a_tree"]
    if backend == "ROOT":
        source.Close.assert_called_once()

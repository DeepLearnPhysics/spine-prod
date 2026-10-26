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
    with pytest.raises(ValueError, match="equally sized"):
        count_root_entries(path)

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
        if name == "1_fake_tree":
            return SimpleNamespace(InheritsFrom=lambda _: False)
        return SimpleNamespace(
            num_entries=counts[name],
            GetEntries=lambda: counts[name],
            InheritsFrom=lambda _: True,
        )

    source = MagicMock()
    source.__enter__.return_value = source
    if backend == "uproot":
        source.keys.return_value = ["b_tree", "a_tree", "0_metadata", "1_fake_tree"]
        source.__getitem__.side_effect = get_tree
        monkeypatch.setitem(
            sys.modules, "uproot", SimpleNamespace(open=lambda _: source)
        )
    else:
        source.IsZombie.return_value = False
        source.GetListOfKeys.return_value = [
            SimpleNamespace(GetName=lambda name=name: name)
            for name in ("b_tree", "a_tree", "0_metadata", "1_fake_tree")
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
        assert opened == ["1_fake_tree", "a_tree", "b_tree"]
    else:
        assert count_root_entries("input.root") == 7
        assert opened == ["1_fake_tree", "a_tree"]
    if backend == "ROOT":
        source.Close.assert_called_once()


def test_survey_requires_a_root_backend(monkeypatch):
    """Missing optional backends produce an actionable error."""
    import sys

    from src.joint_sharding import count_root_entries

    monkeypatch.setitem(sys.modules, "uproot", None)
    monkeypatch.setitem(sys.modules, "ROOT", None)
    with pytest.raises(RuntimeError, match="uproot or PyROOT"):
        count_root_entries("input.root")


@pytest.mark.parametrize("zombie", [False, True])
def test_pyroot_rejects_unreadable_files(monkeypatch, zombie):
    """Both missing and zombie ROOT handles fail before metadata access."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.joint_sharding import count_root_entries

    source = MagicMock() if zombie else None
    if zombie:
        source.IsZombie.return_value = True
    monkeypatch.setitem(sys.modules, "uproot", None)
    monkeypatch.setitem(
        sys.modules,
        "ROOT",
        SimpleNamespace(TFile=SimpleNamespace(Open=lambda *_: source)),
    )
    with pytest.raises(ValueError, match="Cannot open ROOT file"):
        count_root_entries("missing.root")


def test_root_file_without_larcv_trees(tmp_path):
    """A readable ROOT file without an event tree cannot drive sharding."""
    uproot = pytest.importorskip("uproot")
    from src.joint_sharding import count_root_entries

    path = tmp_path / "empty.root"
    with uproot.recreate(path):
        pass
    with pytest.raises(ValueError, match="LArCV"):
        count_root_entries(path)


@pytest.mark.parametrize("cached_count", [None, True, -1, "5"])
def test_survey_discards_invalid_cached_counts(tmp_path, cached_count):
    """Matching file signatures do not make invalid count values trustworthy."""
    import json

    source = tmp_path / "input.root"
    source.touch()
    stat = source.stat()
    cache = tmp_path / "counts.json"
    cache.write_text(
        json.dumps(
            {
                str(source.resolve()): {
                    "signature": [stat.st_size, stat.st_mtime_ns],
                    "entries": cached_count,
                }
            }
        )
    )
    with patch("src.joint_sharding.count_root_entries", return_value=5) as count:
        assert survey_sources({"primary": [str(source)]}, cache) == {"primary": [5]}
        count.assert_called_once()


def test_survey_invalidates_on_mtime_without_size_change(tmp_path):
    """Same-sized file replacements still invalidate the survey cache."""
    import os

    source = tmp_path / "input.root"
    source.write_bytes(b"first")
    sources = {"primary": [str(source)]}
    with patch("src.joint_sharding.count_root_entries", side_effect=[5, 6]) as count:
        survey_sources(sources, tmp_path / "cache.json")
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert survey_sources(sources, tmp_path / "cache.json") == {"primary": [6]}
        assert count.call_count == 2


def test_failed_validation_does_not_upgrade_or_overwrite_cache(tmp_path):
    """A failed tree survey leaves the last successful cache intact."""
    source = tmp_path / "input.root"
    source.touch()
    sources = {"primary": [str(source)]}
    cache = tmp_path / "counts.json"
    with patch("src.joint_sharding.count_root_entries", return_value=5):
        survey_sources(sources, cache)
    before = cache.read_bytes()
    with patch(
        "src.joint_sharding.count_root_entries", side_effect=ValueError("mismatch")
    ):
        with pytest.raises(ValueError, match="mismatch"):
            survey_sources(sources, cache, validate_trees=True)
    assert cache.read_bytes() == before


def test_shards_cover_each_usable_event_once_exhaustively():
    """Check coverage against flat event inventories across small file layouts."""
    from itertools import product

    sources = {"primary": ["p0", "p1", "p2"], "secondary": ["s0", "s1"]}
    for sizes in product(range(3), repeat=5):
        counts = {"primary": sizes[:3], "secondary": sizes[3:]}
        inventory = {
            role: [
                (path, entry)
                for path, size in zip(paths, counts[role])
                for entry in range(size)
            ]
            for role, paths in sources.items()
        }
        usable = min(map(len, inventory.values()))
        for width in (1, 2, 3):
            if not usable:
                with pytest.raises(ValueError, match="no usable"):
                    plan_shards(sources, counts, width)
                continue
            shards = plan_shards(sources, counts, width)
            consumed = {role: [] for role in sources}
            for shard in shards:
                assert 0 < len(shard["primary"]["files"]) <= width
                assert shard["entries"] > 0
                assert shard["start"] == len(consumed["primary"])
                for role in sources:
                    selection = shard[role]
                    entries = [
                        event
                        for event in inventory[role]
                        if event[0] in selection["files"]
                    ]
                    first = selection["n_skip"]
                    consumed[role].extend(entries[first : first + shard["entries"]])
            for role in sources:
                assert consumed[role] == inventory[role][:usable], (sizes, width, role)

"""Tests for source-file parsing and array chunking."""

import pytest

from src.file_handler import FileHandler


@pytest.fixture
def handler():
    return FileHandler()


def test_parse_source_list_ignores_blank_lines(handler, tmp_path):
    source_list = tmp_path / "files.txt"
    source_list.write_text("first.root\n\n second.root \n")

    assert handler.parse_files([str(source_list)], "source_list") == [
        "first.root",
        "second.root",
    ]


def test_parse_source_list_rejects_repeated_paths(handler, tmp_path):
    source_list = tmp_path / "files.txt"
    source_list.write_text("first.root\nsecond.root\nfirst.root\n")

    with pytest.raises(ValueError, match="repeated file paths.*first.root"):
        handler.parse_files([str(source_list)], "source_list")


def test_parse_source_list_requires_one_path(handler):
    with pytest.raises(ValueError, match="exactly one"):
        handler.parse_files(["one.txt", "two.txt"], "source_list")


def test_parse_direct_sources_expands_globs_and_warns_for_missing(
    handler, tmp_path, capsys
):
    first = tmp_path / "first.root"
    second = tmp_path / "second.root"
    first.touch()
    second.touch()
    missing = tmp_path / "missing.root"

    files = handler.parse_files([str(tmp_path / "*.root"), str(missing)])

    assert files == [str(first), str(second)]
    assert f"WARNING: File not found: {missing}" in capsys.readouterr().out


def test_parse_direct_sources_sorts_each_glob(handler, monkeypatch):
    matches = {
        "first/*.root": ["first/b.root", "first/a.root"],
        "second/*.root": ["second/d.root", "second/c.root"],
    }
    monkeypatch.setattr("src.file_handler.glob.glob", matches.__getitem__)

    assert handler.parse_files(list(matches)) == [
        "first/a.root",
        "first/b.root",
        "second/c.root",
        "second/d.root",
    ]


def test_parse_direct_sources_preserves_expected_pipeline_output(handler, tmp_path):
    """An exact future path can be written into a dependent job manifest."""
    future = tmp_path / "upstream_cache.h5"

    assert handler.parse_files([str(future)], allow_missing=True) == [str(future)]


def test_allow_missing_preserves_unresolved_glob(handler, tmp_path):
    """A dependent job may expand its upstream cache glob at runtime."""
    pattern = str(tmp_path / "*.h5")

    assert handler.parse_files([pattern], allow_missing=True) == [pattern]


def test_named_sources_allow_one_shared_cache_repository(handler, tmp_path):
    """A scalar cache accompanies a partitionable primary source collection."""
    first = tmp_path / "first.root"
    second = tmp_path / "second.root"
    cache = tmp_path / "train.spine-cache"
    first.touch()
    second.touch()
    cache.mkdir()

    assert handler.parse_named_sources(
        {
            "primary": {"source": [str(first), str(second)]},
            "cache": {"source": str(cache)},
        }
    ) == {
        "primary": [str(first), str(second)],
        "cache": [str(cache)],
    }


def test_named_joint_sources_allow_independent_file_counts(handler, tmp_path):
    primary = [tmp_path / f"primary-{index}.root" for index in range(2)]
    secondary = [tmp_path / f"secondary-{index}.root" for index in range(3)]
    for path in primary + secondary:
        path.touch()

    assert handler.parse_named_sources(
        {
            "primary": {"source": list(map(str, primary))},
            "secondary": {"source": list(map(str, secondary))},
        },
        aligned=False,
    ) == {
        "primary": list(map(str, primary)),
        "secondary": list(map(str, secondary)),
    }


def test_named_sources_reject_invalid_cache_cardinality(handler, tmp_path):
    """A cache role always denotes exactly one logical repository."""
    caches = [tmp_path / "first.spine-cache", tmp_path / "second.spine-cache"]
    for cache in caches:
        cache.mkdir()
    with pytest.raises(ValueError, match="requires one repository"):
        handler.parse_named_sources({"cache": {"source": list(map(str, caches))}})


def test_named_sources_reject_cache_without_partition_driver(handler, tmp_path):
    """A cache-only inference array has no raw target to divide into tasks."""
    cache = tmp_path / "train.spine-cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="require another target"):
        handler.parse_named_sources({"cache": {"source": str(cache)}})


def test_limit_named_sources_preserves_alignment_and_shared_cache(handler):
    sources = {
        "primary": ["raw-1.root", "raw-2.root", "raw-3.root"],
        "features": ["one.h5", "two.h5", "three.h5"],
        "cache": ["train.spine-cache"],
    }

    assert handler.limit_named_sources(sources, 2) == {
        "primary": ["raw-1.root", "raw-2.root"],
        "features": ["one.h5", "two.h5"],
        "cache": ["train.spine-cache"],
    }


def test_limit_joint_sources_only_truncates_primary(handler):
    sources = {
        "primary": ["primary-1.root", "primary-2.root"],
        "secondary": ["secondary-1.root", "secondary-2.root"],
    }
    assert handler.limit_named_sources(sources, 1, primary_only=True) == {
        "primary": ["primary-1.root"],
        "secondary": ["secondary-1.root", "secondary-2.root"],
    }


def test_pair_joint_sources_uses_shorter_ordered_source(handler):
    sources = {
        "primary": ["primary-1.root", "primary-2.root", "primary-3.root"],
        "secondary": ["secondary-1.root", "secondary-2.root"],
    }

    assert handler.pair_joint_sources(sources) == {
        "primary": ["primary-1.root", "primary-2.root"],
        "secondary": ["secondary-1.root", "secondary-2.root"],
    }
    assert handler.pair_joint_sources(sources, num_files=1) == {
        "primary": ["primary-1.root"],
        "secondary": ["secondary-1.root"],
    }


def test_pair_joint_sources_requires_both_roles(handler):
    with pytest.raises(ValueError, match="primary and secondary"):
        handler.pair_joint_sources({"primary": ["primary.root"]})


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_limit_files_requires_positive_integer(handler, value):
    with pytest.raises(ValueError, match="positive integer"):
        handler.limit_files(["one.root"], value)


def test_stage_cache_output_paths_follow_writer_naming(handler, tmp_path):
    """Predicted cache paths use each source basename and configured suffix."""
    assert handler.stage_cache_output_paths(
        ["/input/a.root", "/other/b.larcv.root"], str(tmp_path), "cache"
    ) == [
        str(tmp_path / "a_cache.h5"),
        str(tmp_path / "b.larcv_cache.h5"),
    ]


def test_stage_cache_output_paths_reject_duplicate_basenames(handler, tmp_path):
    """Two source directories cannot silently claim one output cache path."""
    with pytest.raises(ValueError, match="globally unique source basenames"):
        handler.stage_cache_output_paths(
            ["/first/data.root", "/second/data.root"], str(tmp_path), "cache"
        )


def test_duplicates_preserves_first_duplicate_order(handler):
    assert handler._duplicates(["a", "b", "a", "c", "b", "a"]) == ["a", "b"]


@pytest.mark.parametrize(
    ("files", "max_array_size", "files_per_task", "expected"),
    [
        ([], 2, 2, []),
        (["a", "b", "c"], 10, 2, [[["a", "b"], ["c"]]]),
        (
            ["a", "b", "c", "d", "e"],
            2,
            1,
            [[["a"], ["b"]], [["c"], ["d"]], [["e"]]],
        ),
    ],
)
def test_chunk_files(files, max_array_size, files_per_task, expected, handler):
    assert handler.chunk_files(files, max_array_size, files_per_task) == expected

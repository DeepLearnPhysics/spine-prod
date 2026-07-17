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

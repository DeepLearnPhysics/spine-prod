"""Configuration-independent tests for SPINE command-line formatting."""

import shlex

import pytest

from src.spine_cli import SpineCLI


@pytest.mark.parametrize("validation", [False, True])
def test_named_sources_can_contain_only_file_lists(validation):
    """List-only composite inputs keep target names and paths with spaces intact."""
    sources = {
        "primary": {"source_list": "primary files.txt"},
        "secondary": {"source_list": ["secondary.txt"]},
    }
    formatted = SpineCLI.format_named_sources(sources, validation=validation)
    expected_option = "--val-source-list" if validation else "--source-list"
    assert shlex.split(formatted) == [
        expected_option,
        "primary=primary files.txt",
        "secondary=secondary.txt",
    ]

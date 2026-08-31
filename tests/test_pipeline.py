"""Focused tests for pipeline loading and variable expansion."""

import pytest
import yaml

from src.pipeline import PipelineDefinition


def write_pipeline(tmp_path, document):
    """Write one pipeline document and return its path."""
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_pipeline_expands_workspace_and_variables_recursively(tmp_path):
    """Local variables should expand throughout nested stage structures."""
    path = write_pipeline(
        tmp_path,
        {
            "workspace": "/workflow",
            "variables": {
                "train_source": "/data/train.root",
                "cache_root": "${workspace}/cache",
                "checkpoint": "${workspace}/train/model/weights/best.ckpt",
            },
            "defaults": {"spine_path": "${workspace}/software/spine"},
            "stages": [
                {
                    "name": "cache",
                    "config": "cache.yaml",
                    "source": "${train_source}",
                    "run_dir": "${cache_root}/train",
                    "output": "${cache_root}/train",
                    "module_weight": {"model": "${checkpoint}"},
                    "bind_paths": ["${workspace}", "/data"],
                }
            ],
        },
    )

    stage = PipelineDefinition.load(str(path)).stages[0]
    assert stage["source"] == "/data/train.root"
    assert stage["run_dir"] == "/workflow/cache/train"
    assert stage["output"] == "/workflow/cache/train"
    assert stage["module_weight"] == {
        "model": "/workflow/train/model/weights/best.ckpt"
    }
    assert stage["bind_paths"] == ["/workflow", "/data"]
    assert stage["spine_path"] == "/workflow/software/spine"


@pytest.mark.parametrize(
    ("document", "error", "message"),
    [
        (
            {"workspace": 1, "stages": [{"name": "job", "config": "x.yaml"}]},
            TypeError,
            "workspace must be a string",
        ),
        (
            {"workspace": "", "stages": [{"name": "job", "config": "x.yaml"}]},
            ValueError,
            "workspace must not be empty",
        ),
        (
            {"variables": [], "stages": [{"name": "job", "config": "x.yaml"}]},
            TypeError,
            "variables must be a mapping",
        ),
        (
            {
                "variables": {"workspace": "/bad"},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            ValueError,
            "workspace.*reserved",
        ),
        (
            {
                "variables": {"bad-name": "value"},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            ValueError,
            "valid identifiers",
        ),
        (
            {
                "variables": {"count": 2},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            TypeError,
            "count.*must be a string",
        ),
        (
            {
                "variables": {"first": "${second}", "second": "${first}"},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            ValueError,
            "Cyclic pipeline variable",
        ),
        (
            {
                "stages": [
                    {
                        "name": "job",
                        "config": "x.yaml",
                        "source": "${missing}",
                    }
                ]
            },
            ValueError,
            "Undefined pipeline variable 'missing'",
        ),
    ],
)
def test_pipeline_rejects_invalid_variable_declarations(
    tmp_path, document, error, message
):
    """Malformed or unresolved substitutions must fail before submission."""
    path = write_pipeline(tmp_path, document)

    with pytest.raises(error, match=message):
        PipelineDefinition.load(str(path))

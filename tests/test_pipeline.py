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


def test_pipeline_expands_collection_stage_templates(tmp_path):
    """Each collection item should become an ordinary validated stage."""
    path = write_pipeline(
        tmp_path,
        {
            "workspace": "/workflow",
            "variables": {
                "train_source": "/data/train.root",
                "validation_source": "/data/test.root",
            },
            "collections": {
                "splits": [
                    {
                        "name": "train",
                        "source": "${train_source}",
                        "cache_dir": "${workspace}/cache/train",
                    },
                    {
                        "name": "validation",
                        "source": "${validation_source}",
                        "cache_dir": "${workspace}/cache/validation",
                    },
                ]
            },
            "stages": [
                {
                    "name": "prepare",
                    "config": "prepare.yaml",
                },
                {
                    "name": "cache_${split.name}",
                    "for_each": {"collection": "splits", "as": "split"},
                    "depends_on": ["prepare"],
                    "config": "cache.yaml",
                    "source": "${split.source}",
                    "run_dir": "${split.cache_dir}/stage",
                    "output": "${split.cache_dir}",
                },
                {
                    "name": "consume",
                    "config": "consume.yaml",
                    "depends_on": ["cache_train", "cache_validation"],
                },
            ],
        },
    )

    stages = PipelineDefinition.load(str(path)).stages
    assert [stage["name"] for stage in stages] == [
        "prepare",
        "cache_train",
        "cache_validation",
        "consume",
    ]
    assert stages[1]["source"] == "/data/train.root"
    assert stages[1]["run_dir"] == "/workflow/cache/train/stage"
    assert stages[2]["source"] == "/data/test.root"
    assert stages[2]["output"] == "/workflow/cache/validation"


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
            {"collections": [], "stages": [{"name": "job", "config": "x.yaml"}]},
            TypeError,
            "collections must be a mapping",
        ),
        (
            {
                "collections": {"splits": []},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            ValueError,
            "splits.*non-empty list",
        ),
        (
            {
                "collections": {"splits": [{"name": 1}]},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            TypeError,
            "value 'name' must be a string",
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
        (
            {
                "stages": [
                    {
                        "name": "job_${split.name}",
                        "config": "x.yaml",
                        "for_each": {"collection": "missing", "as": "split"},
                    }
                ]
            },
            ValueError,
            "unknown collection: missing",
        ),
        (
            {
                "collections": {"splits": [{"name": "train"}]},
                "stages": [
                    {
                        "name": "job_${split.missing}",
                        "config": "x.yaml",
                        "for_each": {"collection": "splits", "as": "split"},
                    }
                ],
            },
            ValueError,
            "Undefined pipeline variable 'split.missing'",
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

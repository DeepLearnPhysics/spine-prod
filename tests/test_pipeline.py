"""Focused tests for pipeline loading and variable expansion."""

import pytest
import yaml

from src.pipeline import PipelineDefinition, PipelineRunner


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


def test_null_workspace_requires_launch_override(tmp_path):
    """Portable pipelines should fail before expansion without a workspace."""
    path = write_pipeline(
        tmp_path,
        {
            "workspace": None,
            "stages": [
                {
                    "name": "train",
                    "config": "train.yaml",
                    "run_dir": "${workspace}/train",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="workspace is null.*--workspace"):
        PipelineDefinition.load(str(path))


def test_workspace_override_resolves_portable_pipeline(tmp_path):
    """A launch workspace should resolve variables in stages and collections."""
    path = write_pipeline(
        tmp_path,
        {
            "workspace": None,
            "collections": {
                "splits": [{"name": "train", "output": "${workspace}/cache/train"}]
            },
            "stages": [
                {
                    "name": "cache_${split.name}",
                    "for_each": {"collection": "splits", "as": "split"},
                    "config": "cache.yaml",
                    "output": "${split.output}",
                }
            ],
        },
    )

    definition = PipelineDefinition.load(
        str(path), workspace_override="/runs/benchmark"
    )

    assert definition.workspace == "/runs/benchmark"
    assert definition.stages[0]["output"] == "/runs/benchmark/cache/train"


def test_workspace_override_replaces_yaml_default(tmp_path):
    """The launch value should take precedence over a concrete YAML workspace."""
    path = write_pipeline(
        tmp_path,
        {
            "workspace": "/yaml/default",
            "stages": [
                {
                    "name": "job",
                    "config": "job.yaml",
                    "run_dir": "${workspace}/job",
                }
            ],
        },
    )

    definition = PipelineDefinition.load(str(path), workspace_override="/cli/override")

    assert definition.workspace == "/cli/override"
    assert definition.stages[0]["run_dir"] == "/cli/override/job"


def test_pipeline_may_omit_unused_workspace(tmp_path):
    """Legacy pipelines need no workspace when they do not reference one."""
    path = write_pipeline(
        tmp_path,
        {"stages": [{"name": "job", "config": "job.yaml"}]},
    )

    definition = PipelineDefinition.load(str(path))

    assert definition.workspace is None
    assert definition.stages[0]["name"] == "job"


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


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"source": "input.root"}, "cannot be combined with data sources"),
        ({"output": "output.h5"}, "cannot be combined with writer output"),
        ({"stage": "train", "run_dir": "/tmp/train"}, "requires stage=inference"),
    ],
)
def test_pipeline_rejects_export_runtime_conflicts(tmp_path, extra, message):
    """Weight composition must remain a terminal model-only stage."""
    path = write_pipeline(
        tmp_path,
        {
            "stages": [
                {
                    "name": "export",
                    "config": "model.yaml",
                    "export_weights": "/weights/full-chain.ckpt",
                    **extra,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match=message):
        PipelineDefinition.load(str(path))


def test_pipeline_rejects_stage_names_that_are_not_safe_paths(tmp_path):
    """Stage names also serve as workspace log-link names."""
    path = write_pipeline(
        tmp_path,
        {"stages": [{"name": "../escape", "config": "model.yaml"}]},
    )

    with pytest.raises(ValueError, match="name may contain only"):
        PipelineDefinition.load(str(path))


@pytest.mark.parametrize(
    ("document", "workspace_override", "error", "message"),
    [
        ([], None, TypeError, "must contain a mapping"),
        ({"stages": []}, None, ValueError, "non-empty stages list"),
        (
            {"stages": [{"name": "job", "config": "x.yaml"}]},
            1,
            TypeError,
            "override must be a string",
        ),
        (
            {"stages": [{"name": "job", "config": "x.yaml"}]},
            "",
            ValueError,
            "override must not be empty",
        ),
        (
            {
                "variables": {"first": "${missing}"},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            None,
            ValueError,
            "Undefined pipeline variable: missing",
        ),
        (
            {
                "collections": {"bad-name": [{"name": "train"}]},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            None,
            ValueError,
            "collection names must be valid identifiers",
        ),
        (
            {
                "collections": {"splits": [{}]},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            None,
            ValueError,
            "item 1 must not be empty",
        ),
        (
            {
                "collections": {"splits": [{"bad-name": "train"}]},
                "stages": [{"name": "job", "config": "x.yaml"}],
            },
            None,
            ValueError,
            "keys must be valid identifiers",
        ),
        ({"stages": ["bad"]}, None, TypeError, "stage 1 must be a mapping"),
        (
            {
                "collections": {"splits": [{"name": "train"}]},
                "stages": [
                    {
                        "name": "job",
                        "config": "x.yaml",
                        "for_each": {
                            "collection": "splits",
                            "as": "split",
                            "unknown": True,
                        },
                    }
                ],
            },
            None,
            ValueError,
            "contains unknown field",
        ),
        (
            {
                "collections": {"splits": [{"name": "train"}]},
                "stages": [
                    {
                        "name": "job",
                        "config": "x.yaml",
                        "for_each": {"collection": 1, "as": "split"},
                    }
                ],
            },
            None,
            TypeError,
            "collection must be a non-empty string",
        ),
        (
            {
                "collections": {"splits": [{"name": "train"}]},
                "stages": [
                    {
                        "name": "job",
                        "config": "x.yaml",
                        "for_each": {"collection": "splits", "as": "bad-name"},
                    }
                ],
            },
            None,
            ValueError,
            "as must be a valid identifier",
        ),
        (
            {
                "variables": {"split": "reserved"},
                "collections": {"splits": [{"name": "train"}]},
                "stages": [
                    {
                        "name": "job",
                        "config": "x.yaml",
                        "for_each": {"collection": "splits", "as": "split"},
                    }
                ],
            },
            None,
            ValueError,
            "conflicts with a pipeline variable",
        ),
    ],
)
def test_pipeline_rejects_additional_malformed_documents(
    tmp_path, document, workspace_override, error, message
):
    """All malformed document shapes should fail before scheduler contact."""
    path = write_pipeline(tmp_path, document)

    with pytest.raises(error, match=message):
        PipelineDefinition.load(str(path), workspace_override=workspace_override)


def test_pipeline_expands_tuples_and_rejects_conflicting_aliases():
    """Tuple recursion and legacy alias ambiguity have deterministic behavior."""
    assert PipelineDefinition._expand_variables(
        ("${value}",), {"value": "expanded"}, "test"
    ) == ("expanded",)

    with pytest.raises(ValueError, match="both larcv_path and larcv_basedir"):
        PipelineDefinition._normalize_aliases(
            {"larcv_path": "/new", "larcv_basedir": "/legacy"}, "test"
        )


@pytest.mark.parametrize(
    ("stage", "error", "message"),
    [
        (None, TypeError, "must be a mapping"),
        ({"name": "job"}, ValueError, "must define name and config"),
        ({"name": 1, "config": "x.yaml"}, TypeError, "name must be a string"),
        (
            {"name": "job", "config": "x.yaml", "depends_on": "first"},
            TypeError,
            "depends_on must be a list",
        ),
        (
            {"name": "job", "config": "x.yaml", "export_weights": ""},
            ValueError,
            "export_weights must not be empty",
        ),
        (
            {"name": "job", "config": "x.yaml", "weight_path": ""},
            ValueError,
            "weight_path must be a non-empty string",
        ),
        (
            {"name": "job", "config": "x.yaml", "kind": "other"},
            ValueError,
            "invalid kind",
        ),
        (
            {"name": "job", "config": "x.yaml", "stage": "other"},
            ValueError,
            "invalid lifecycle stage",
        ),
        (
            {"name": "job", "config": "x.yaml", "stage": "train"},
            ValueError,
            "requires run_dir",
        ),
        (
            {"name": "job", "config": "x.yaml", "resume": True},
            ValueError,
            "can resume only",
        ),
        (
            {"name": "job", "config": "x.yaml", "validation_name": "test"},
            ValueError,
            "validation lifecycle options require",
        ),
        (
            {"name": "job", "config": "x.yaml", "val_source": "val.root"},
            ValueError,
            "validation inputs require stage=train",
        ),
        (
            {
                "name": "job",
                "config": "x.yaml",
                "val_entry_fraction_range": [0.0, 0.5],
            },
            ValueError,
            "validation entry range requires stage=train",
        ),
        (
            {
                "name": "job",
                "config": "x.yaml",
                "stage": "validation",
                "run_dir": "/run",
                "ntasks": 2,
            },
            ValueError,
            "task splitting requires stage=inference",
        ),
        (
            {
                "name": "report",
                "config": "report.yaml",
                "kind": "report",
            },
            ValueError,
            "requires: run_dir, input_dir, output_dir",
        ),
        (
            {
                "name": "report",
                "config": "report.yaml",
                "kind": "report",
                "run_dir": "/run",
                "input_dir": "/input",
                "output_dir": "/output",
                "source": "input.root",
            },
            ValueError,
            "cannot use SPINE field",
        ),
    ],
)
def test_pipeline_rejects_additional_invalid_stage_contracts(stage, error, message):
    """Stage validation covers every lifecycle and structured-field contract."""
    with pytest.raises(error, match=message):
        PipelineDefinition._resolve_stage(stage, 1, {}, {}, set())


def test_pipeline_rejects_duplicate_stage_names():
    """A stage name may appear only once in a pipeline."""
    with pytest.raises(ValueError, match="Duplicate pipeline stage name"):
        PipelineDefinition._resolve_stage(
            {"name": "duplicate", "config": "x.yaml"},
            2,
            {},
            {},
            {"duplicate"},
        )


def test_pipeline_log_index_rejects_existing_regular_path(tmp_path):
    """The shallow log index must not overwrite user-owned entries."""
    workspace = tmp_path / "workspace"
    log_path = workspace / "logs" / "stage"
    log_path.parent.mkdir(parents=True)
    log_path.touch()

    with pytest.raises(ValueError, match="log index path is not a symlink"):
        PipelineRunner._prepare_log_index(str(workspace), [{"name": "stage"}])


@pytest.mark.parametrize(
    ("from_stage", "to_stage", "message"),
    [
        ("", None, "from_stage must be a non-empty string"),
        (None, "", "to_stage must be a non-empty string"),
        (None, "missing", "Unknown pipeline stop stage"),
    ],
)
def test_pipeline_stage_selection_rejects_invalid_boundaries(
    from_stage, to_stage, message
):
    """Restart boundaries must be non-empty names present in the pipeline."""
    with pytest.raises(ValueError, match=message):
        PipelineRunner._select_stages(
            [{"name": "stage"}], from_stage=from_stage, to_stage=to_stage
        )

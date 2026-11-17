"""Cross-site allocation contracts, including the complete VD candidate."""

from unittest.mock import patch

import pytest
import yaml

from src.pipeline import PipelineDefinition
from src.site_resources import (
    allocation_limits,
    duration_seconds,
    map_stage,
    validate_allocation,
)


@pytest.fixture
def catalog(workspace_root):
    return yaml.safe_load((workspace_root / "templates/profiles.yaml").read_text())


@pytest.mark.parametrize(
    "value,seconds", [("5-00:00:00", 432000), ("168:00:00", 604800), ("00:05:01", 301)]
)
def test_duration(value, seconds):
    assert duration_seconds(value) == seconds


def test_invalid_duration():
    with pytest.raises(ValueError, match="Invalid allocation time"):
        duration_seconds("1:99:00")


@pytest.mark.parametrize(
    "site,prefix", [("nersc", "nersc"), ("alcf", "anl_polaris"), ("s3df", "s3df")]
)
def test_vd_mapping(catalog, workspace_root, site, prefix):
    definition = PipelineDefinition.load(
        str(workspace_root / "tests/fixtures/dune_vd_pipeline_261117.yaml"),
        workspace_override="/tmp/vd",
    )
    stages = [
        map_stage(catalog, stage, site, {"account": "project", "time": "24:00:00"})
        for stage in definition.stages
    ]
    assert len(stages) == 21
    for stage in stages:
        assert stage["profile"].startswith(prefix)
        effective = dict(
            catalog["profiles"][stage["profile"]],
            **{k: v for k, v in stage.items() if k != "profile"},
        )
        validate_allocation(catalog, effective)
    assert stages[0]["cpus_per_task"] == 16
    assert stages[0]["workers"] == 16
    train = next(s for s in stages if s["name"] == "train_uresnet_deghost")
    assert catalog["profiles"][train["profile"]]["allocation_role"] == "full_gpu_node"
    if site == "nersc":
        assert stages[0]["mem_per_cpu"] == "4g"


@pytest.mark.parametrize(
    "site,stage,message",
    [
        ("missing", {}, "Unknown site"),
        ("nersc", {"profile": "auto"}, "named resource"),
        ("alcf", {"profile": "nersc_gpu_80gb"}, "No anl correspondence"),
        ("nersc", {"profile": "s3df_hopper"}, "more GPU memory"),
        ("nersc", {"profile": "s3df_ampere"}, "requires --account"),
    ],
)
def test_mapping_errors(catalog, site, stage, message):
    with pytest.raises(ValueError, match=message):
        map_stage(catalog, stage, site)


def test_explicit_wrong_site_profile(catalog):
    with pytest.raises(ValueError, match="does not belong"):
        map_stage(
            catalog, {"profile": "s3df_ampere"}, "nersc", {"profile": "s3df_ampere"}
        )


def test_custom_gpu_count_and_cpu_memory(catalog):
    result = map_stage(
        catalog,
        {"profile": "s3df_ampere", "gpus": 2, "partition": "ampere"},
        "nersc",
        {"account": "project"},
    )
    assert result["gpus_per_node"] == 2
    assert "partition" not in result
    result = map_stage(
        catalog,
        {"profile": "s3df_milano", "mem_per_node": "64g"},
        "nersc",
        {"account": "project"},
    )
    assert result["mem_per_node"] == "64g"
    assert "mem_per_cpu" not in result


@pytest.mark.parametrize(
    "profile,changes,message",
    [
        ("nersc_gpu_full", {"time": "5-00:00:00"}, "max_time"),
        ("nersc_gpu", {"gpus_per_node": 4}, "GPU count"),
        ("anl_polaris_debug", {"time": "00:01:00"}, "min_time"),
        ("anl_polaris_capacity", {"nodes": 5}, "Node count"),
        ("anl_polaris_capacity", {"queue": "prod"}, "Node count"),
    ],
)
def test_limits(catalog, profile, changes, message):
    with pytest.raises(ValueError, match=message):
        validate_allocation(catalog, dict(catalog["profiles"][profile], **changes))


def test_unknown_policy_is_not_an_invented_limit():
    assert allocation_limits({}, {}) == {}
    validate_allocation({}, {})


def test_pipeline_rejects_late_limit_before_any_submission(mock_submitter, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {
                        "name": "scan",
                        "config": "anything.yaml",
                        "profile": "s3df_milano",
                    },
                    {
                        "name": "train",
                        "config": "anything.yaml",
                        "profile": "s3df_ampere_full",
                        "time": "5-00:00:00",
                    },
                ]
            }
        )
    )
    with patch.object(mock_submitter, "submit_job") as submit:
        with pytest.raises(ValueError, match="Stage train.*max_time"):
            mock_submitter.submit_pipeline(
                str(path), site="nersc", overrides={"account": "project"}
            )
        submit.assert_not_called()


def test_pipeline_maps_selected_stages_before_dispatch(mock_submitter, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {
                        "name": "scan",
                        "config": "anything.yaml",
                        "profile": "s3df_milano",
                        "cpus_per_task": 16,
                    },
                    {
                        "name": "train",
                        "config": "anything.yaml",
                        "profile": "s3df_ampere_full",
                        "time": "5-00:00:00",
                    },
                ]
            }
        )
    )
    with patch.object(mock_submitter, "submit_job", return_value=["123"]) as submit:
        mock_submitter.submit_pipeline(
            str(path), site="nersc", to_stage="scan", overrides={"account": "project"}
        )
        assert submit.call_args.kwargs["profile"] == "nersc_cpu"
        assert submit.call_args.kwargs["mem_per_cpu"] == "4g"


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"nodes": 2}, "one node"),
        ({"queue": "custom"}, "No configured limits"),
        ({"mem_per_cpu": "garbage"}, "requires a unit"),
    ],
)
def test_portable_invalid_overrides(catalog, changes, message):
    with pytest.raises(ValueError, match=message):
        map_stage(catalog, {"profile": "s3df_milano"}, "alcf", changes)


def test_defaults_and_reverse_gpu_mapping(catalog):
    assert map_stage(catalog, {}, "s3df")["profile"] == "s3df_ampere"
    assert (
        map_stage(catalog, {"kind": "report"}, "alcf")["profile"] == "anl_polaris_cpu"
    )
    result = map_stage(catalog, {"profile": "nersc_gpu", "gpus_per_node": 2}, "s3df")
    assert result["gpus"] == 2
    assert "gpus_per_node" not in result


def test_polaris_scan_memory(catalog):
    result = map_stage(catalog, {"profile": "s3df_milano", "cpus_per_task": 16}, "alcf")
    assert result["mem_per_node"] == "64gb"
    assert "mem_per_cpu" not in result


@pytest.mark.parametrize(
    "site,profile,expected",
    [
        (
            "nersc",
            "s3df_milano",
            ["--constraint=cpu", "--cpus-per-task=16", "--mem-per-cpu=4g"],
        ),
        ("alcf", "s3df_milano", ["ncpus=16", "#PBS -l mem=64gb"]),
        (
            "nersc",
            "s3df_ampere_full",
            ["--constraint=gpu", "--gpus-per-node=4", "--mem=0"],
        ),
    ],
)
def test_mapped_scheduler_directives(catalog, workspace_root, site, profile, expected):
    from jinja2 import Environment, FileSystemLoader

    stage = map_stage(
        catalog, {"profile": profile, "cpus_per_task": 16}, site, {"account": "project"}
    )
    effective = dict(catalog["profiles"][stage["profile"]], **stage)
    template = "job_template_anl.pbs" if site == "alcf" else "job_template_nersc.sbatch"
    script = (
        Environment(loader=FileSystemLoader(workspace_root / "templates"))
        .get_template(template)
        .render(**effective)
    )
    for directive in expected:
        assert directive in script


def test_unlimited_time_is_not_a_finite_allocation():
    with pytest.raises(ValueError, match="positive"):
        duration_seconds("00:00:00")


def test_pbs_explicit_node_memory_units(catalog):
    result = map_stage(
        catalog, {"profile": "s3df_milano", "mem_per_node": "64g"}, "alcf"
    )
    assert result["mem_per_node"] == "64gb"

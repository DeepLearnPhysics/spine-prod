"""Exercise the DUNE HD and VD pipelines across the supported sites."""

import pytest
import yaml

from src.pipeline import PipelineDefinition
from src.sites import map_stage, validate_allocation


@pytest.fixture(
    params=[
        "pipelines/dune-vd-10kt-1x8x6/full_chain_261117.yaml",
        "pipelines/dune-hd-10kt-1x2x6/full_chain_261022.yaml",
    ],
    ids=["vd", "hd"],
)
def pipeline_path(request):
    return request.param


@pytest.fixture
def catalog(workspace_root):
    return yaml.safe_load((workspace_root / "templates/profiles.yaml").read_text())


@pytest.mark.parametrize(
    "site,prefix", [("nersc", "nersc"), ("alcf", "anl_polaris"), ("s3df", "s3df")]
)
def test_dune_mapping(catalog, workspace_root, site, prefix, pipeline_path):
    definition = PipelineDefinition.load(
        str(workspace_root / pipeline_path),
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


@pytest.mark.parametrize("site", ["nersc", "s3df", "alcf"])
def test_dune_budgets_and_site_requests(catalog, workspace_root, site, pipeline_path):
    from src.continuation.policy import next_request, profile_policy

    definition = PipelineDefinition.load(
        str(workspace_root / pipeline_path),
        workspace_override="/tmp/vd",
    )
    training = [s for s in definition.stages if s.get("stage") == "train"]
    assert len(training) == 6
    budgets = [120, 72, 48, 24, 16, 8]
    for stage, hours in zip(training, budgets):
        mapped = map_stage(catalog, stage, site, {"account": "project"})
        effective = dict(catalog["profiles"][mapped["profile"]], time=mapped["time"])
        config = profile_policy(catalog, effective, mapped["continuation"])
        assert config["runtime_budget"] == hours * 3600
        assert config["maximum"] == min(hours, 48 if site == "nersc" else hours) * 3600
        request = next_request(config, 0)
        assert request["seconds"] == min(hours, 24 if site == "nersc" else hours) * 3600
        assert request["min_time"] == ("04:00:00" if site == "nersc" else None)
        assert stage["continuation"] == {"runtime_budget": stage["time"]}
        assert next_request(config, hours * 3600) is None
    scan = map_stage(catalog, definition.stages[0], site, {"account": "project"})
    assert scan["time"] == definition.stages[0]["time"]
    assert "continuation" not in scan


def test_explicit_overlong_request_is_rejected(catalog):
    from src.continuation.policy import profile_policy

    stage = {
        "profile": "s3df_ampere_full",
        "time": "120:00:00",
        "continuation": {"runtime_budget": "120:00:00"},
    }
    result = map_stage(
        catalog, stage, "nersc", {"account": "project", "time": "72:00:00"}
    )
    assert result["time"] == "72:00:00"
    with pytest.raises(ValueError, match="max_time"):
        profile_policy(
            catalog,
            dict(catalog["profiles"][result["profile"]], time=result["time"]),
            result["continuation"],
        )


def test_explicit_preferences_and_short_defaults(catalog):
    from src.continuation.policy import profile_policy

    stage = {
        "profile": "s3df_ampere",
        "time": "48:00:00",
        "continuation": {
            "runtime_budget": "120:00:00",
            "preferred_time": "06:00:00",
            "min_time": "01:00:00",
        },
    }
    result = map_stage(catalog, stage, "nersc", {"account": "project"})
    assert result["continuation"]["preferred_time"] == "06:00:00"
    assert result["continuation"]["min_time"] == "01:00:00"
    stage["time"] = "01:00:00"
    stage["continuation"] = {"runtime_budget": "02:00:00"}
    result = map_stage(catalog, stage, "nersc", {"account": "project"})
    assert result["continuation"]["preferred_time"] == "01:00:00"
    assert result["continuation"]["min_time"] == "01:00:00"
    profile_policy(
        catalog,
        dict(catalog["profiles"][result["profile"]], time=result["time"]),
        result["continuation"],
    )


def test_noncontinuation_time_is_not_clamped(catalog):
    stage = {"profile": "s3df_ampere", "time": "120:00:00"}
    mapped = map_stage(catalog, stage, "nersc", {"account": "project"})
    assert mapped["time"] == "120:00:00"
    with pytest.raises(ValueError, match="max_time"):
        validate_allocation(
            catalog, dict(catalog["profiles"][mapped["profile"]], time=mapped["time"])
        )


def test_dune_nersc_dry_run_preflights_all_training_stages(
    mock_submitter, tmp_path, monkeypatch, capsys, pipeline_path
):
    from unittest.mock import Mock

    from src.continuation import service

    monkeypatch.setattr(
        service, "service_paths", lambda account: (tmp_path, tmp_path / "registry")
    )
    first_stage = Mock(return_value=[])
    monkeypatch.setattr(mock_submitter, "submit_filter", first_stage)
    register = Mock(side_effect=AssertionError("Dry-run must not install automation"))
    monkeypatch.setattr(service, "register", register)
    result = mock_submitter.submit_pipeline(
        str(mock_submitter.basedir / pipeline_path),
        workspace=str(tmp_path / "vd"),
        site="nersc",
        overrides={"account": "project"},
        continuation_dir="/pscratch/sd/u/user/vd-state",
        dry_run=True,
    )
    assert result == {}
    first_stage.assert_called_once()
    assert first_stage.call_args.kwargs["profile"] == "nersc_cpu"
    output = capsys.readouterr().out
    assert (
        "budget=120:00:00; allocation=04:00:00..48:00:00; preferred=24:00:00" in output
    )
    assert (
        "budget=08:00:00; allocation=04:00:00..08:00:00; preferred=08:00:00" in output
    )
    register.assert_not_called()

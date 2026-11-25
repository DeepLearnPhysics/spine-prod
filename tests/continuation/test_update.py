"""Operational updates preserve live attempts and serialize against the controller."""

import copy
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.continuation import update
from src.continuation.state import atomic_json, read


@pytest.fixture
def campaign(tmp_path, mock_submitter):
    workspace = tmp_path / "workspace"
    training = dict(
        profile="nersc_gpu_full",
        run_dir=str(workspace / "train/deghost"),
        time="48:00:00",
        continuation=dict(
            runtime_budget="120:00:00",
            min_time="04:00:00",
            preferred_time="24:00:00",
            checkpoint_margin="00:15:00",
            startup_allowance="00:05:00",
        ),
    )
    repository = str(workspace / "cache/train.spine-cache")
    state = dict(
        state="active",
        basedir=str(mock_submitter.basedir),
        definition_sha256="original-launch",
        generation=4,
        records=[
            dict(
                name="scan",
                status="completed",
                jobs=["1"],
                options=dict(
                    profile="nersc_cpu", cache_dir=str(workspace / "filter/counts")
                ),
            ),
            dict(name="train", status="submitted", jobs=["2"], options=training),
            dict(
                name="cache",
                status="waiting",
                jobs=[],
                options=dict(
                    profile="nersc_gpu",
                    cache_repository=repository,
                    output=repository,
                    cache_stage="deghosting",
                    run_dir=str(workspace / "cache/train/deghosting"),
                ),
            ),
            dict(
                name="next",
                status="waiting",
                jobs=[],
                options=dict(
                    copy.deepcopy(training),
                    run_dir=str(workspace / "train/next"),
                    source=repository,
                ),
            ),
        ],
    )
    for record in state["records"]:
        record["kind"] = "spine"
    path = tmp_path / "state/.spine-prod/continuation_pipeline.json"
    atomic_json(path, state)
    return path, workspace, state


def test_preview_apply_idempotence_and_history(campaign, mock_submitter, monkeypatch):
    path, workspace, original = campaign
    scheduler = Mock(side_effect=AssertionError("must not submit or cancel"))
    monkeypatch.setattr(mock_submitter, "submit_job", scheduler)
    args = dict(
        minimum="01:00:00", cache_dir="/pscratch/user/vd-cache", context=mock_submitter
    )
    assert update.update_campaign(path, workspace, **args) is None
    assert read(path) == original
    assert not list(path.parent.glob("*before-update*"))
    backup = update.update_campaign(path, workspace, apply=True, **args)
    assert read(backup) == original
    changed = read(path)
    assert changed["records"][0] == original["records"][0]
    assert changed["records"][1]["options"]["continuation"]["min_time"] == "01:00:00"
    assert changed["records"][1]["jobs"] == ["2"]
    assert changed["records"][1]["status"] == "submitted"
    assert (
        changed["records"][2]["options"]["output"]
        == "/pscratch/user/vd-cache/train.spine-cache"
    )
    assert (
        changed["records"][3]["options"]["source"]
        == "/pscratch/user/vd-cache/train.spine-cache"
    )
    assert (
        changed["records"][3]["options"]["continuation"]["runtime_budget"]
        == "120:00:00"
    )
    assert changed["definition_sha256"] == original["definition_sha256"]
    assert changed["generation"] == 5
    assert len(changed["configuration_updates"]) == 1
    assert (
        update.update_campaign(
            path, workspace, minimum="01:00:00", apply=True, context=mock_submitter
        )
        is None
    )
    scheduler.assert_not_called()


@pytest.mark.parametrize(
    "mode",
    [
        "inactive",
        "workspace",
        "started",
        "populated",
        "file",
        "no-producer",
        "invalid-min",
        "cfs",
    ],
)
def test_rejects_unsafe_changes_before_writing(campaign, mock_submitter, mode):
    path, workspace, state = campaign
    minimum, target = "01:00:00", "/pscratch/user/cache"
    if mode == "inactive":
        state["state"] = "cancelled"
    elif mode == "workspace":
        workspace = workspace / "wrong"
    elif mode == "started":
        state["records"][2]["status"] = "submitted"
    elif mode in ("populated", "file"):
        source = Path(state["records"][2]["options"]["cache_repository"])
        source.parent.mkdir(parents=True)
        if mode == "populated":
            source.mkdir()
            (source / "manifest.json").write_text("{}")
        else:
            source.write_text("data")
    elif mode == "no-producer":
        del state["records"][2]["options"]["cache_repository"]
    elif mode == "invalid-min":
        minimum = "00:10:00"
    else:
        target = "/global/cfs/cache"
    atomic_json(path, state)
    with pytest.raises(ValueError):
        update.update_campaign(path, workspace, minimum, target, True, mock_submitter)
    assert read(path) == state
    assert not list(path.parent.glob("*before-update*"))


def test_backup_failure_preserves_controller(campaign, mock_submitter, monkeypatch):
    path, workspace, state = campaign
    monkeypatch.setattr(update, "atomic_json", lambda p, d: atomic_json(p, {}))
    with pytest.raises(ValueError, match="backup verification"):
        update.update_campaign(
            path, workspace, minimum="01:00:00", apply=True, context=mock_submitter
        )
    assert read(path) == state


def test_options_required_and_empty_cache_allowed(campaign, mock_submitter):
    path, workspace, state = campaign
    with pytest.raises(ValueError, match="Provide"):
        update.update_campaign(path, workspace, context=mock_submitter)
    source = Path(state["records"][2]["options"]["cache_repository"])
    source.mkdir(parents=True)
    update.update_campaign(
        path,
        workspace,
        cache_dir="/pscratch/user/cache",
        apply=True,
        context=mock_submitter,
    )
    assert read(path)["records"][1] == state["records"][1]


def test_cli_preview(campaign, monkeypatch):
    path, workspace, state = campaign
    script = Path(__file__).resolve().parents[2] / "scripts/update_continuation.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            str(path),
            "--workspace",
            str(workspace),
            "--min-time",
            "01:00:00",
            "--cache-dir",
            "/pscratch/user/cache",
        ],
    )
    runpy.run_path(str(script), run_name="__main__")
    assert read(path) == state


@pytest.mark.parametrize("content", ["empty", "file", "data"])
def test_destination_validation(campaign, mock_submitter, tmp_path, content):
    path, workspace, state = campaign
    target_root = tmp_path / "scratch"
    target = target_root / "train.spine-cache"
    target_root.mkdir()
    if content == "file":
        target.write_text("existing")
    else:
        target.mkdir()
        if content == "data":
            (target / "manifest.json").write_text("{}")
    # Exercise destination validation with a local shared filesystem profile.
    for record in state["records"]:
        record["options"]["profile"] = "s3df_ampere"
    atomic_json(path, state)
    if content == "empty":
        update.update_campaign(
            path, workspace, cache_dir=str(target_root), context=mock_submitter
        )
    else:
        with pytest.raises(ValueError, match="Destination cache repository"):
            update.update_campaign(
                path,
                workspace,
                cache_dir=str(target_root),
                apply=True,
                context=mock_submitter,
            )
    assert read(path) == state


def test_next_resume_keeps_checkpoint_and_uses_updated_minimum(
    campaign, mock_submitter, monkeypatch
):
    from src.continuation import controller
    from src.continuation.state import manifest_path

    path, workspace, original = campaign
    run_dir = original["records"][1]["options"]["run_dir"]
    logical_path = manifest_path(run_dir)
    logical = dict(
        state="paused",
        checkpoint="/durable/restart.ckpt",
        used_seconds=275239,
        attempts=[dict(job_id="2", accounted=True)],
    )
    atomic_json(logical_path, logical)
    update.update_campaign(
        path,
        workspace,
        minimum="01:00:00",
        cache_dir="/pscratch/user/cache",
        apply=True,
        context=mock_submitter,
    )
    assert read(logical_path) == logical
    monkeypatch.setattr(controller, "reconcile", lambda *a: logical)
    submit = Mock(return_value=["3"])
    monkeypatch.setattr(mock_submitter, "submit_job", submit)
    controller.advance(path, mock_submitter)
    assert submit.call_args.kwargs["continuation"]["min_time"] == "01:00:00"
    assert submit.call_args.kwargs["resume_from"] == "/durable/restart.ckpt"
    assert submit.call_args.kwargs["resume"] is True
    assert read(logical_path)["used_seconds"] == 275239

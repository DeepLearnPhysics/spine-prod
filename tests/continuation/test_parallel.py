"""Independent branches retain logical completion and retry boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.continuation import controller, service
from src.continuation.state import atomic_json, manifest_path, read
from src.continuation.update import enable_parallel


def record(name, deps, status="waiting", continuation=False, root=None):
    options = {"run_dir": str(root / name), "profile": "test"}
    if continuation:
        options["continuation"] = {}
    return dict(
        name=name,
        depends_on=deps,
        status=status,
        kind="spine",
        jobs=[name] if status == "submitted" else [],
        options=options,
        scheduler="slurm",
    )


def campaign(tmp_path, records):
    path = tmp_path / "pipeline.json"
    atomic_json(path, dict(state="active", generation=0, records=records))
    submit = Mock(side_effect=lambda **kw: [kw["run_dir"].split("/")[-1]])
    context = SimpleNamespace(
        config_mgr=SimpleNamespace(get_profile=lambda *_: {"scheduler": "slurm"}),
        submit_job=submit,
    )
    return path, context


def test_parallel_cache_failure_retry_and_join(tmp_path, monkeypatch):
    records = [
        record("train", [], root=tmp_path),
        record("val", [], root=tmp_path),
        record("consumer", ["train", "val"], root=tmp_path),
    ]
    records[0]["options"]["cache_repository"] = str(tmp_path / "repo")
    path, ctx = campaign(tmp_path, records)
    monkeypatch.setattr(controller, "accounting", lambda *_: None)
    assert set(controller.advance(path, ctx)) == {"train", "val"}
    controller.advance(path, ctx)
    assert ctx.submit_job.call_count == 2
    monkeypatch.setattr(
        controller,
        "accounting",
        lambda _, job: {"success": False} if job == "train" else None,
    )
    controller.advance(path, ctx)
    assert service.needs_tick(path)  # failed sibling must not retire the monitor
    controller._operate(path, "retry-cache", "train")
    assert read(path)["records"][0]["options"]["retry"]
    controller.advance(path, ctx)
    monkeypatch.setattr(controller, "accounting", lambda *_: {"success": True})
    controller.advance(path, ctx)
    assert ctx.submit_job.call_count == 4
    controller.advance(path, ctx)
    assert read(path)["state"] == "completed"


def test_training_siblings_resume_and_gate_consumers(tmp_path, monkeypatch):
    records = [record(n, [], "submitted", True, tmp_path) for n in ("shower", "track")]
    records.append(record("cache", ["shower", "track"], root=tmp_path))
    path, ctx = campaign(tmp_path, records)
    logical = {}
    for name in ("shower", "track"):
        logical[name] = dict(
            state="paused",
            checkpoint=name + ".ckpt",
            attempts=[dict(accounted=True, job_id=name)],
        )
        atomic_json(manifest_path(tmp_path / name), logical[name])
    monkeypatch.setattr(
        controller, "reconcile", lambda run, *_: logical[run.split("/")[-1]]
    )
    controller.advance(path, ctx)
    assert [c.kwargs["resume_from"] for c in ctx.submit_job.call_args_list] == [
        "shower.ckpt",
        "track.ckpt",
    ]
    logical["shower"]["state"] = "completed"
    logical["track"]["state"] = "running"
    controller.advance(path, ctx)
    assert ctx.submit_job.call_count == 2
    logical["track"]["state"] = "completed"
    controller.advance(path, ctx)
    assert ctx.submit_job.call_count == 3


@pytest.mark.parametrize(
    "detector,version",
    [("dune-hd-10kt-1x2x6", "261022"), ("dune-vd-10kt-1x8x6", "261117")],
)
def test_upgrade_real_campaign(tmp_path, detector, version):
    from src.pipeline import PipelineDefinition

    source = f"pipelines/{detector}/full_chain_{version}.yaml"
    stages = PipelineDefinition.load(source, workspace_override=str(tmp_path)).stages
    records = [
        dict(
            name=s["name"],
            kind=s.get("kind", "spine"),
            status="waiting",
            jobs=[],
            options={"run_dir": s["run_dir"]},
        )
        for s in stages
    ]
    records[0]["status"] = "completed"
    active = next(r for r in records if r["name"] == "train_grappa_shower")
    active.update(status="submitted", jobs=["existing-job"])
    path, _ = campaign(tmp_path, records)
    original = read(path)
    enable_parallel(path, source, str(tmp_path))
    assert read(path) == original
    backup = enable_parallel(path, source, str(tmp_path), True)
    assert read(backup) == original
    upgraded = read(path)
    graph = controller.dependencies(upgraded["records"])
    assert graph["train_grappa_shower"] == graph["train_grappa_track"]
    assert graph["cache_train_deghosting"] == graph["cache_validation_deghosting"]
    for before, after in zip(records, upgraded["records"]):
        assert {k: v for k, v in after.items() if k != "depends_on"} == before
    assert enable_parallel(path, source, str(tmp_path), True) is None


def test_invalid_graph_rejected_before_submission(tmp_path):
    path, ctx = campaign(tmp_path, [record("a", ["missing"], root=tmp_path)])
    with pytest.raises(ValueError, match="dependencies"):
        controller.advance(path, ctx)
    ctx.submit_job.assert_not_called()


def test_monitor_retires_only_when_all_remaining_branches_blocked(tmp_path):
    records = [
        record("a", [], "failed", root=tmp_path),
        record("b", [], "submitted", True, tmp_path),
        record("join", ["a", "b"], root=tmp_path),
    ]
    path, _ = campaign(tmp_path, records)
    logical = dict(state="failed", attempts=[dict(accounted=False)])
    atomic_json(manifest_path(tmp_path / "b"), logical)
    assert service.needs_tick(path)
    logical["attempts"][0]["accounted"] = True
    atomic_json(manifest_path(tmp_path / "b"), logical)
    assert not service.needs_tick(path)


def test_legacy_manifests_stay_sequential(tmp_path, monkeypatch):
    records = [record(n, [], root=tmp_path) for n in ("a", "b")]
    for item in records:
        del item["depends_on"]
    path, ctx = campaign(tmp_path, records)
    monkeypatch.setattr(controller, "accounting", lambda *_: None)
    controller.advance(path, ctx)
    controller.advance(path, ctx)
    assert ctx.submit_job.call_count == 1


def test_upgrade_rejects_wrong_workspace_without_changes(tmp_path):
    path, _ = campaign(tmp_path, [record("wrong", [], root=tmp_path)])
    original = read(path)
    with pytest.raises(ValueError, match="stages/order"):
        enable_parallel(
            path,
            "pipelines/dune-hd-10kt-1x2x6/full_chain_261022.yaml",
            str(tmp_path),
            True,
        )
    assert read(path) == original

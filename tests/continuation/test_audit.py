"""Historical audit publication and fail-safe timer retirement."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.continuation import audit, controller, service
from src.continuation.state import atomic_json, manifest_path, read

from . import test_service
from .helpers import allocation

fake_scrontab = test_service.fake_scrontab
service_environment = test_service.service_environment


def test_audit_records_outcomes_and_log_without_restore(tmp_path):
    run = tmp_path / "run"
    attempt = allocation(run, 1, "completed")
    atomic_json(Path(attempt["directory"]) / "job_metadata.json", {"job": "123"})
    manifest = tmp_path / "continuation_pipeline.json"
    state = dict(
        state="completed",
        audit_dir=str(tmp_path / "durable"),
        records=[
            dict(
                name="train",
                status="completed",
                jobs=["123"],
                options=dict(continuation={}, run_dir=str(run)),
            ),
            dict(
                name="future",
                status="waiting",
                options=dict(continuation={}, run_dir=str(tmp_path / "absent")),
            ),
            dict(name="cache", status="completed", options={}),
        ],
    )
    atomic_json(manifest, state)
    (tmp_path / "controller.log").write_text("monitor output\n")
    path = audit.archive_audit(manifest)
    payload = read(path)
    assert payload["historical_only"] is True
    assert payload["controller_log"] == "monitor output\n"
    assert payload["summary"][0]["jobs"] == ["123"]
    assert "1/outcome.json" in payload["stages"]["train"]["attempt_files"]
    assert "1/job_metadata.json" in payload["stages"]["train"]["attempt_files"]
    assert read(path.parent / "latest.json")["snapshot"] == path.name
    assert audit.archive_audit(manifest) == path
    assert read(manifest) == state
    assert manifest_path(run).exists()


def test_audit_verification_failure(tmp_path, monkeypatch):
    manifest = tmp_path / "pipeline.json"
    atomic_json(manifest, dict(audit_dir=str(tmp_path / "audit"), records=[]))
    monkeypatch.setattr(audit, "atomic_json", lambda p, d: atomic_json(p, {}))
    with pytest.raises(ValueError, match="timer retained"):
        audit.archive_audit(manifest)
    assert not list((tmp_path / "audit").rglob("latest.json"))


def test_failed_audit_keeps_timer_then_retries_without_submission(
    service_environment, monkeypatch
):
    _, registry, manifest = service_environment
    schedule, _ = fake_scrontab(monkeypatch)
    service.register(manifest, "project", Path.cwd())
    state = read(manifest)
    state.update(state="completed", audit_dir=str(manifest.parent / "audit"))
    atomic_json(manifest, state)
    advance = Mock(side_effect=AssertionError("completed work must not run again"))
    monkeypatch.setattr(controller, "advance", advance)
    with monkeypatch.context() as patch:
        patch.setattr(
            audit, "atomic_json", Mock(side_effect=OSError("archive offline"))
        )
        service.tick_registry(registry)
    assert "# BEGIN spine-prod" in schedule[0]
    assert "archive offline" in read(registry)["cleanup_error"]
    service.tick_registry(registry)
    assert "# BEGIN spine-prod" not in schedule[0]
    assert list((manifest.parent / "audit").rglob("latest.json"))
    advance.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
def test_archive_stage_serializes_controller(
    mock_submitter, tmp_path, monkeypatch, success
):
    stages = [
        dict(
            name="cache",
            config="cache.yaml",
            profile="s3df_ampere",
            run_dir=str(tmp_path / "cache-job"),
        ),
        dict(
            name="consumer",
            config="train.yaml",
            profile="s3df_ampere",
            run_dir=str(tmp_path / "train"),
        ),
    ]
    archive_plan = dict(profile="s3df_milano", run_dir=str(tmp_path / "archive-job"))
    submit = Mock(side_effect=[["producer"], ["consumer"]])
    archive = Mock(return_value=["archive"])
    monkeypatch.setattr(mock_submitter, "submit_job", submit)
    monkeypatch.setattr(mock_submitter, "submit_archive", archive)
    controller.start(
        mock_submitter, stages, tmp_path, cache_archives={"cache": archive_plan}
    )
    manifest = tmp_path / ".spine-prod/continuation_pipeline.json"
    assert [r["name"] for r in read(manifest)["records"]] == [
        "cache",
        "cache__archive",
        "consumer",
    ]
    with pytest.raises(ValueError, match="audit directory changed"):
        controller.start(
            mock_submitter,
            stages,
            tmp_path,
            cache_archives={"cache": archive_plan},
            audit_dir=str(tmp_path / "other"),
        )
    monkeypatch.setattr(controller, "accounting", lambda *a: {"success": True})
    assert controller.advance(manifest, mock_submitter) == {
        "cache__archive": ["archive"]
    }
    assert submit.call_count == 1
    monkeypatch.setattr(controller, "accounting", lambda *a: {"success": success})
    controller.advance(manifest, mock_submitter)
    assert submit.call_count == (2 if success else 1)
    if success:
        controller.advance(manifest, mock_submitter)
    assert list((tmp_path / "monitoring").rglob("latest.json"))


def test_failed_logical_stage_is_archived(mock_submitter, tmp_path, monkeypatch):
    run = tmp_path / "train"
    allocation(run, 1)
    logical_path = manifest_path(run)
    logical = read(logical_path)
    logical["state"] = "failed"
    logical["reason"] = "training_error"
    atomic_json(logical_path, logical)
    manifest = tmp_path / "pipeline.json"
    atomic_json(
        manifest,
        dict(
            state="active",
            audit_dir=str(tmp_path / "audit"),
            records=[
                dict(
                    name="train",
                    status="submitted",
                    jobs=["1"],
                    options=dict(continuation={}, run_dir=str(run)),
                )
            ],
        ),
    )
    monkeypatch.setattr(controller, "reconcile", lambda *a: logical)
    controller.advance(manifest, mock_submitter)
    index = next((tmp_path / "audit").rglob("latest.json"))
    payload = read(index.parent / read(index)["snapshot"])
    assert payload["summary"][0]["reason"] == "training_error"


def test_disabled_registry_entries_do_not_require_audit(
    service_environment, monkeypatch
):
    _, registry, manifest = service_environment
    fake_scrontab(monkeypatch)
    service.register(manifest, "project", Path.cwd())
    data = read(registry)
    data["pipelines"][str(manifest)]["enabled"] = False
    atomic_json(registry, data)
    monkeypatch.setattr(
        audit, "archive_audit", Mock(side_effect=AssertionError("disabled"))
    )
    assert service.retire_if_idle(registry)

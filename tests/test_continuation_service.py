"""Test unattended registration and execution without a live scheduler."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import continuation as c
from src import continuation_pipeline as pipeline
from src import continuation_service as service


@pytest.fixture
def service_environment(tmp_path, monkeypatch):
    root = tmp_path / "automation"
    registry = root / "project" / "registry.json"
    monkeypatch.setattr(service, "service_paths", lambda _: (root, registry))
    manifest = tmp_path / "run" / "continuation_pipeline.json"
    c.atomic_json(manifest, {"state": "active", "records": []})
    return root, registry, manifest


def fake_scrontab(monkeypatch, initial=""):
    schedule = [initial]
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv == ["scrontab", "-l"]:
            return SimpleNamespace(returncode=0, stdout=schedule[0], stderr="")
        assert argv == ["scrontab", "-"]
        schedule[0] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(service.subprocess, "run", run)
    return schedule, calls


def test_registration_preserves_other_jobs_and_is_idempotent(
    service_environment, monkeypatch
):
    _, registry, manifest = service_environment
    original = "# user job\n0 * * * * /some/script\n"
    schedule, calls = fake_scrontab(monkeypatch, original)
    service.register(manifest, "project", Path.cwd())
    service.register(manifest, "project", Path.cwd())
    assert schedule[0].startswith(original)
    assert schedule[0].count("# BEGIN spine-prod") == 1
    assert calls.count(["scrontab", "-"]) == 1
    assert len(c.read(registry)["pipelines"]) == 1
    assert c.read(registry)["pipelines"][str(manifest)]["enabled"]
    assert c.read(manifest)["automation"]["backend"] == "nersc-scrontab"
    assert list(registry.parent.glob("scrontab-before-*"))[0].read_text() == original


def test_registration_failure_disables_pipeline(service_environment, monkeypatch):
    _, registry, manifest = service_environment

    def fail(argv, **kwargs):
        if "-l" in argv:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.CalledProcessError(1, argv, stderr="denied")

    monkeypatch.setattr(service.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        service.register(manifest, "project", Path.cwd())
    assert not c.read(registry)["pipelines"][str(manifest)]["enabled"]
    assert "automation" not in c.read(manifest)


def test_read_schedule_errors_are_not_treated_as_empty(monkeypatch):
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="Permission denied"
        ),
    )
    with pytest.raises(RuntimeError, match="Permission denied"):
        service.read_schedule()
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="scrontab: no scrontab for user"
        ),
    )
    assert service.read_schedule() == ""


def test_schedule_edit_race_never_overwrites(service_environment, monkeypatch):
    _, _, manifest = service_environment
    reads = iter(["original\n", "changed\n"])
    monkeypatch.setattr(service, "read_schedule", lambda: next(reads))
    monkeypatch.setattr(
        service.subprocess, "run", lambda *a, **k: pytest.fail("must not install")
    )
    with pytest.raises(RuntimeError, match="changed during"):
        service.register(manifest, "project", Path.cwd())


def test_tick_runs_real_script_cleans_environment_and_skips_completed(
    service_environment, monkeypatch
):
    _, registry, manifest = service_environment
    result = manifest.parent / "result"
    (manifest.parent / "advance.sh").write_text(
        "#!/bin/bash\nset -e\n"
        'test -z "${SLURM_MEM_PER_CPU+x}"\n'
        'test -z "${SBATCH_QOS+x}"\n'
        'test -z "${SRUN_CPUS_PER_TASK+x}"\n'
        f'echo ran >> "{result}"\n'
    )
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "2048")
    monkeypatch.setenv("SBATCH_QOS", "cron")
    monkeypatch.setenv("SRUN_CPUS_PER_TASK", "1")
    c.atomic_json(registry, {"pipelines": {str(manifest): {"enabled": True}}})
    service.tick_registry(registry)
    assert result.read_text() == "ran\n"
    entry = c.read(registry)["pipelines"][str(manifest)]
    assert entry["last_checked"]
    assert entry["last_error"] is None
    c.atomic_json(manifest, {"state": "completed", "records": []})
    service.tick_registry(registry)
    assert result.read_text() == "ran\n"


def test_tick_error_does_not_disable_other_pipelines(service_environment):
    _, registry, manifest = service_environment
    other = manifest.parent / "other" / "controller.json"
    c.atomic_json(other, {"state": "completed", "records": []})
    c.atomic_json(
        registry,
        {
            "pipelines": {
                str(manifest): {"enabled": True},
                str(other): {"enabled": True},
            }
        },
    )
    service.tick_registry(registry)
    entries = c.read(registry)["pipelines"]
    assert entries[str(manifest)]["last_error"]
    assert entries[str(other)]["last_checked"]
    assert entries[str(other)]["last_error"] is None


@pytest.mark.parametrize("logical_state", ["failed", "cancelled", "budget_exhausted"])
def test_blocked_training_is_not_advanced(service_environment, logical_state):
    _, _, manifest = service_environment
    run = manifest.parent / "train"
    c.atomic_json(c.manifest_path(run), {"state": logical_state})
    c.atomic_json(
        manifest,
        {
            "state": "active",
            "records": [
                {
                    "status": "submitted",
                    "options": {"run_dir": str(run), "continuation": {}},
                }
            ],
        },
    )
    assert not service.needs_tick(manifest)


def test_start_never_submits_training_if_registration_fails(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from src.submitter import Submitter

    context = Submitter()
    context.submit_job = Mock()
    monkeypatch.setattr(c, "validate_storage", lambda *_: None)
    monkeypatch.setattr(
        service, "service_paths", lambda _: (tmp_path, tmp_path / "registry")
    )

    def fail(*_):
        raise RuntimeError("registration denied")

    monkeypatch.setattr(service, "register", fail)
    stages = [
        {
            "name": "train",
            "stage": "train",
            "profile": "nersc_gpu_full",
            "account": "project",
            "config": "train.yaml",
            "run_dir": str(tmp_path / "train"),
            "continuation": {"runtime_budget": "120:00:00"},
        }
    ]
    with pytest.raises(RuntimeError, match="registration denied"):
        pipeline.start(
            context, stages, str(tmp_path), continuation_dir=str(tmp_path / "state")
        )
    context.submit_job.assert_not_called()


def test_health_reports_missing_and_stale_registry(service_environment, monkeypatch):
    _, registry, manifest = service_environment
    fake_scrontab(monkeypatch)
    service.register(manifest, "project", Path.cwd())
    state = c.read(manifest)
    assert "awaiting" in service.health(state)
    data = c.read(registry)
    data["pipelines"][str(manifest)]["last_checked"] = 1
    c.atomic_json(registry, data)
    assert "stale" in service.health(state)
    registry.unlink()
    assert service.health(state) == "registry unavailable"


def test_service_drives_two_pauses_and_releases_consumer_once(
    service_environment, monkeypatch
):
    from tests.test_continuation import allocation

    _, registry, manifest = service_environment
    run = manifest.parent / "train"
    control = manifest.parent / "scratch"
    c.atomic_json(
        manifest,
        {
            "state": "active",
            "basedir": str(Path.cwd()),
            "generation": 0,
            "records": [
                {
                    "name": "train",
                    "kind": "spine",
                    "status": "waiting",
                    "jobs": [],
                    "options": {
                        "run_dir": str(run),
                        "continuation_dir": str(control),
                        "continuation": {},
                        "profile": "test",
                    },
                },
                {
                    "name": "consumer",
                    "kind": "spine",
                    "status": "waiting",
                    "jobs": [],
                    "options": {"config": "consumer"},
                },
            ],
        },
    )
    c.atomic_json(registry, {"pipelines": {str(manifest): {"enabled": True}}})
    submissions = []

    def submit(**options):
        submissions.append(options)
        number = len(submissions)
        if "continuation" in options:
            allocation(
                run,
                number,
                "completed" if number == 3 else "paused",
                continuation_dir=control,
            )
        return [str(number)]

    context = SimpleNamespace(
        config_mgr=SimpleNamespace(get_profile=lambda *_: {"scheduler": "slurm"}),
        submit_job=submit,
    )
    # Simulate completed allocations; use the real reconcile/advance algorithms.
    monkeypatch.setattr(
        c,
        "accounting",
        lambda *_: {"success": True, "seconds": 200, "status": "COMPLETED"},
    )

    def launch(argv, **kwargs):
        assert argv == ["bash", str(manifest.parent / "advance.sh")]
        pipeline.advance(manifest, context)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(service.subprocess, "run", launch)
    for _ in range(7):
        service.tick_registry(registry)
    assert len(submissions) == 4
    assert submissions[1]["resume_from"].endswith("snapshot-10.ckpt")
    assert submissions[2]["resume_from"].endswith("snapshot-20.ckpt")
    assert submissions[3]["config"] == "consumer"
    assert c.read(manifest)["state"] == "completed"
    assert c.read(c.manifest_path(run, control))["used_seconds"] == 600


def test_two_pipelines_share_one_schedule(service_environment, monkeypatch):
    _, registry, manifest = service_environment
    schedule, calls = fake_scrontab(monkeypatch)
    other = manifest.parent / "another.json"
    c.atomic_json(other, {"state": "active", "records": []})
    service.register(manifest, "project", Path.cwd())
    service.register(other, "project", Path.cwd())
    assert len(c.read(registry)["pipelines"]) == 2
    assert schedule[0].count("# BEGIN spine-prod") == 1
    assert calls.count(["scrontab", "-"]) == 1

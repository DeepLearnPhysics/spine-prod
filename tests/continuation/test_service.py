"""Test unattended registration and execution without a live scheduler."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.continuation import controller as pipeline
from src.continuation import service
from src.continuation import state as storage


@pytest.fixture
def service_environment(tmp_path, monkeypatch):
    root = tmp_path / "automation"
    registry = root / "project" / "registry.json"
    monkeypatch.setattr(service, "service_paths", lambda *args: (root, registry))
    manifest = tmp_path / "run" / "continuation_pipeline.json"
    storage.atomic_json(manifest, {"state": "active", "records": []})
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
    assert len(storage.read(registry)["pipelines"]) == 1
    assert storage.read(registry)["pipelines"][str(manifest)]["enabled"]
    assert storage.read(manifest)["automation"]["backend"] == "nersc-scrontab"
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
    assert not storage.read(registry)["pipelines"][str(manifest)]["enabled"]
    assert "automation" not in storage.read(manifest)


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
    monkeypatch.setattr(service, "read_schedule", lambda *args: next(reads))
    monkeypatch.setattr(
        service.subprocess, "run", lambda *a, **k: pytest.fail("must not install")
    )
    with pytest.raises(RuntimeError, match="changed during"):
        service.register(manifest, "project", Path.cwd())


def test_tick_runs_real_script_cleans_environment_and_skips_completed(
    service_environment, monkeypatch
):
    monkeypatch.setattr(service, "read_schedule", lambda *args: "")
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
    storage.atomic_json(registry, {"pipelines": {str(manifest): {"enabled": True}}})
    service.tick_registry(registry)
    assert result.read_text() == "ran\n"
    entry = storage.read(registry)["pipelines"][str(manifest)]
    assert entry["last_checked"]
    assert entry["last_error"] is None
    storage.atomic_json(manifest, {"state": "completed", "records": []})
    service.tick_registry(registry)
    assert result.read_text() == "ran\n"


def test_tick_error_does_not_disable_other_pipelines(service_environment):
    _, registry, manifest = service_environment
    other = manifest.parent / "other" / "controller.json"
    storage.atomic_json(other, {"state": "completed", "records": []})
    storage.atomic_json(
        registry,
        {
            "pipelines": {
                str(manifest): {"enabled": True},
                str(other): {"enabled": True},
            }
        },
    )
    service.tick_registry(registry)
    entries = storage.read(registry)["pipelines"]
    assert entries[str(manifest)]["last_error"]
    assert entries[str(other)]["last_checked"]
    assert entries[str(other)]["last_error"] is None


@pytest.mark.parametrize("logical_state", ["failed", "cancelled", "budget_exhausted"])
def test_blocked_training_is_not_advanced(service_environment, logical_state):
    _, _, manifest = service_environment
    run = manifest.parent / "train"
    storage.atomic_json(storage.manifest_path(run), {"state": logical_state})
    storage.atomic_json(
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


@pytest.mark.parametrize("profile", ["nersc_gpu_full", "s3df_ampere_full"])
def test_start_never_submits_training_if_registration_fails(
    tmp_path, monkeypatch, profile
):
    from unittest.mock import Mock

    from src.submitter import Submitter

    context = Submitter()
    context.submit_job = Mock()
    monkeypatch.setattr(pipeline, "validate_storage", lambda *_: None)
    monkeypatch.setattr(
        service, "service_paths", lambda *args: (tmp_path, tmp_path / "registry")
    )

    def fail(*_):
        raise RuntimeError("registration denied")

    monkeypatch.setattr(service, "register", fail)
    stages = [
        {
            "name": "train",
            "stage": "train",
            "profile": profile,
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
    state = storage.read(manifest)
    assert "awaiting" in service.health(state)
    data = storage.read(registry)
    data["pipelines"][str(manifest)]["last_checked"] = 1
    storage.atomic_json(registry, data)
    assert "stale" in service.health(state)
    registry.unlink()
    assert service.health(state) == "registry unavailable"


def test_service_drives_two_pauses_and_releases_consumer_once(
    service_environment, monkeypatch
):
    from .helpers import allocation

    _, registry, manifest = service_environment
    run = manifest.parent / "train"
    control = manifest.parent / "scratch"
    storage.atomic_json(
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
    storage.atomic_json(registry, {"pipelines": {str(manifest): {"enabled": True}}})
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
        pipeline,
        "accounting",
        lambda *_: {"success": True, "seconds": 200, "status": "COMPLETED"},
    )

    schedule = [service.schedule_block("project", registry, Path.cwd())]

    def launch(argv, **kwargs):
        if argv == ["scrontab", "-l"]:
            return SimpleNamespace(returncode=0, stdout=schedule[0], stderr="")
        if argv == ["scrontab", "-"]:
            schedule[0] = kwargs["input"]
            return SimpleNamespace(returncode=0, stdout="", stderr="")
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
    assert storage.read(manifest)["state"] == "completed"
    assert storage.read(storage.manifest_path(run, control))["used_seconds"] == 600

    assert schedule[0] == ""
    assert storage.read(registry)["schedule_state"] == "stopped"


def test_two_pipelines_share_one_schedule(service_environment, monkeypatch):
    _, registry, manifest = service_environment
    schedule, calls = fake_scrontab(monkeypatch)
    other = manifest.parent / "another.json"
    storage.atomic_json(other, {"state": "active", "records": []})
    service.register(manifest, "project", Path.cwd())
    service.register(other, "project", Path.cwd())
    assert len(storage.read(registry)["pipelines"]) == 2
    assert schedule[0].count("# BEGIN spine-prod") == 1
    assert calls.count(["scrontab", "-"]) == 1


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize(
    "message", ["no crontab for drielsma\n", "scrontab: no scrontab for drielsma\n"]
)
def test_first_use_scrontab_diagnostic_on_either_stream(monkeypatch, stream, message):
    result = SimpleNamespace(returncode=1, stdout="", stderr="")
    setattr(result, stream, message)
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: result)
    assert service.read_schedule() == ""


@pytest.mark.parametrize(
    "stdout,stderr",
    [
        ("permission denied", ""),
        ("no crontab for drielsma", "connection failed"),
        ("* * * * * /existing/task", "no crontab for drielsma"),
        ("", ""),
    ],
)
def test_scrontab_unexpected_failure_is_not_empty(monkeypatch, stdout, stderr):
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr),
    )
    with pytest.raises(RuntimeError) as error:
        service.read_schedule()
    assert stdout in str(error.value)
    assert stderr in str(error.value)
    if not stdout and not stderr:
        assert "exit status 1" in str(error.value)


def test_retirement_preserves_unrelated_entries_and_relaunch_rearms(
    service_environment, monkeypatch
):
    _, registry, manifest = service_environment
    unrelated = "# user task\n0 * * * * /user/task\n"
    schedule, calls = fake_scrontab(monkeypatch, unrelated)
    service.register(manifest, "project", Path.cwd())
    # The all-completed transition has yet to run, so still needs a tick.
    assert not service.retire_if_idle(registry)
    state = storage.read(manifest)
    state["state"] = "completed"
    storage.atomic_json(manifest, state)
    assert service.retire_if_idle(registry)
    assert schedule[0] == unrelated
    assert "stopped" in service.health(storage.read(manifest))
    # Repeated cleanup is a no-op; a new launch installs one block again.
    installed = calls.count(["scrontab", "-"])
    assert service.retire_if_idle(registry)
    assert calls.count(["scrontab", "-"]) == installed
    service.register(manifest, "project", Path.cwd())
    assert schedule[0].count("# BEGIN spine-prod") == 1
    assert storage.read(registry)["schedule_state"] == "active"


def test_retirement_keeps_timer_for_other_active_pipeline(
    service_environment, monkeypatch
):
    _, registry, manifest = service_environment
    schedule, _ = fake_scrontab(monkeypatch)
    service.register(manifest, "project", Path.cwd())
    other = manifest.parent / "other.json"
    storage.atomic_json(other, {"state": "active", "records": []})
    service.register(other, "project", Path.cwd())
    storage.atomic_json(manifest, {"state": "completed", "records": []})
    assert not service.retire_if_idle(registry)
    assert "# BEGIN spine-prod" in schedule[0]
    other.unlink()
    with pytest.raises(OSError):
        service.retire_if_idle(registry)
    assert "# BEGIN spine-prod" in schedule[0]


@pytest.mark.parametrize(
    "action,logical_state,value",
    [
        ("extend", "budget_exhausted", "00:24:00"),
        ("retry", "failed", None),
        ("accept", "paused", None),
    ],
)
def test_operator_actions_rearm_stopped_service(
    service_environment, monkeypatch, action, logical_state, value
):
    from .helpers import allocation

    _, registry, manifest = service_environment
    schedule, _ = fake_scrontab(monkeypatch)
    run = manifest.parent / "train"
    attempt = allocation(run, 1)
    logical = storage.read(storage.manifest_path(run))
    outcome = storage.read(Path(attempt["directory"]) / "outcome.json")
    logical.update(
        state=logical_state, checkpoint=outcome["checkpoint"], sha256=outcome["sha256"]
    )
    logical["attempts"][0]["accounted"] = True
    storage.atomic_json(storage.manifest_path(run), logical)
    storage.atomic_json(
        manifest,
        {
            "state": "active",
            "generation": 0,
            "basedir": str(Path.cwd()),
            "records": [
                {
                    "name": "train",
                    "status": "submitted",
                    "jobs": ["1"],
                    "options": {"run_dir": str(run), "continuation": {}},
                }
            ],
        },
    )
    service.register(manifest, "project", Path.cwd())
    # accept can be invoked on a manually disabled timer too.
    if action == "accept":
        schedule[0] = ""
    else:
        assert service.retire_if_idle(registry)
    assert schedule[0] == ""
    pipeline.operate(manifest, action, stage="train", value=value)
    assert schedule[0].count("# BEGIN spine-prod") == 1
    assert storage.read(registry)["schedule_state"] == "active"
    assert storage.read(storage.manifest_path(run))["state"] == (
        "completed" if action == "accept" else "paused"
    )


def test_retirement_waits_for_reactivation_state_change(
    service_environment, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    _, registry, manifest = service_environment
    schedule, _ = fake_scrontab(monkeypatch)
    storage.atomic_json(
        manifest, {"state": "completed", "records": [], "basedir": str(Path.cwd())}
    )
    service.register(manifest, "project", Path.cwd())
    service.retire_if_idle(registry)
    attempted = Event()

    def cleanup():
        attempted.set()
        return service.retire_if_idle(registry)

    with ThreadPoolExecutor() as pool:
        with service.reactivating(storage.read(manifest)):
            future = pool.submit(cleanup)
            assert attempted.wait(2)
            state = storage.read(manifest)
            state["state"] = "active"
            storage.atomic_json(manifest, state)
        assert future.result(timeout=2) is False
    assert "# BEGIN spine-prod" in schedule[0]


@pytest.mark.parametrize(
    "account,scratch",
    [("bad/account", "/pscratch/user"), ("project", "/pscratch/with space")],
)
def test_service_paths_reject_invalid_names(monkeypatch, account, scratch):
    """Unsafe account and path characters cannot enter a cron entry."""
    monkeypatch.setenv("SCRATCH", scratch)
    with pytest.raises(ValueError):
        service.service_paths(account)


def test_schedule_rejects_shell_expansion_and_ambiguous_markers(tmp_path):
    """Only complete unique managed blocks may be replaced or removed."""
    with pytest.raises(ValueError):
        service.schedule_block("project", tmp_path / "registry", tmp_path / "$bad")
    block = service.schedule_block("project", tmp_path / "registry", tmp_path)
    with pytest.raises(RuntimeError):
        service.merge_schedule(block + block, block)
    with pytest.raises(RuntimeError):
        service.remove_schedule(block + block, "project")
    assert service.merge_schedule(block.rstrip("\n"), block) == block
    assert service.remove_schedule(block.rstrip("\n"), "project") == ""
    assert service.remove_schedule("unrelated", "project") == "unrelated"


def test_registration_detects_external_edit(service_environment, monkeypatch):
    """Do not overwrite a scrontab edited while registration was preparing."""
    _, registry, manifest = service_environment
    from unittest.mock import Mock

    monkeypatch.setattr(
        service, "read_schedule", Mock(side_effect=["", "external edit"])
    )
    with pytest.raises(RuntimeError, match="changed during registration"):
        service.register(manifest, "project", Path.cwd())
    assert not storage.read(registry)["pipelines"][str(manifest)]["enabled"]


@pytest.mark.parametrize("verify_failure", [False, True])
def test_retirement_races_leave_diagnostic(
    service_environment, monkeypatch, verify_failure
):
    """Failed timer retirement must be visible rather than silently forgotten."""
    from unittest.mock import Mock

    _, registry, manifest = service_environment
    storage.atomic_json(manifest, {"state": "completed", "records": []})
    storage.atomic_json(registry, {"pipelines": {str(manifest): {"enabled": True}}})
    block = service.schedule_block("project", registry, Path.cwd())
    replies = (
        [block, block, "different"] if verify_failure else [block, "external edit"]
    )
    monkeypatch.setattr(service, "read_schedule", Mock(side_effect=replies))
    monkeypatch.setattr(service.subprocess, "run", Mock())
    service.tick_registry(registry)
    assert storage.read(registry)["cleanup_error"]


@pytest.mark.parametrize("mode", ["deadline", "disabled", "removed"])
def test_tick_registry_handles_bounded_and_changed_registrations(
    service_environment, monkeypatch, mode
):
    """Respect scan time limits, disabled entries and concurrent deregistration."""
    from unittest.mock import Mock

    _, registry, manifest = service_environment
    storage.atomic_json(
        registry, {"pipelines": {str(manifest): {"enabled": mode != "disabled"}}}
    )
    monkeypatch.setattr(service, "retire_if_idle", Mock())
    monkeypatch.setattr(
        service.time,
        "monotonic",
        Mock(side_effect=[0, 2000 if mode == "deadline" else 1]),
    )

    def needs_tick(_):
        """Simulate removal while a tick is checking a pipeline."""
        storage.atomic_json(registry, {"pipelines": {}})
        return False

    monkeypatch.setattr(service, "needs_tick", needs_tick)
    service.tick_registry(registry)
    assert "last_checked" not in storage.read(registry)["pipelines"].get(
        str(manifest), {}
    )


def test_needs_tick_handles_completed_and_failed_records(tmp_path):
    """Completed predecessors are skipped; failed ordinary stages stop monitoring."""
    manifest = tmp_path / "pipeline.json"
    storage.atomic_json(
        manifest,
        {"state": "active", "records": [{"status": "completed"}, {"status": "failed"}]},
    )
    assert not service.needs_tick(manifest)
    storage.atomic_json(
        manifest, {"state": "active", "records": [{"status": "completed"}]}
    )
    assert service.needs_tick(manifest)


def test_health_reports_all_registration_failures(tmp_path, monkeypatch):
    """Expose absent, disabled, errored and stale automation states."""
    registry = tmp_path / "registry.json"
    state = {"automation": {"registry": str(registry), "manifest": "pipeline"}}
    assert service.health({}) == "manual"
    assert service.health(state) == "registry unavailable"
    for data, expected in [
        ({"cleanup_error": "failure"}, "controller retirement error: failure"),
        ({"pipelines": {"pipeline": {"enabled": False}}}, "disabled"),
        (
            {"pipelines": {"pipeline": {"enabled": True, "last_error": "failed"}}},
            "controller error: failed",
        ),
    ]:
        storage.atomic_json(registry, data)
        assert service.health(state) == expected


def test_registration_requires_readback(service_environment, monkeypatch):
    """A successful scrontab exit alone does not prove timer installation."""
    _, _, manifest = service_environment
    monkeypatch.setattr(service, "read_schedule", lambda *args: "")
    monkeypatch.setattr(
        service.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    with pytest.raises(RuntimeError, match="could not be verified"):
        service.register(manifest, "project", Path.cwd())

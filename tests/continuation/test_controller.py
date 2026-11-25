"""Allocation lifecycle, accounting and consumer gating regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Template

from src.continuation import controller as pipeline
from src.continuation import policy as policies
from src.continuation import scheduler as scheduling
from src.continuation import state as storage

from .helpers import allocation, checkpoint, config


@pytest.mark.parametrize("bad", ["0:00:00", "1:60:00", "10", -2, True])
def test_duration_rejects_invalid(bad):
    with pytest.raises(ValueError):
        policies.seconds(bad)


def test_policy_final_chunk_and_rounding():
    cfg = config(min_time="00:02:00")
    assert policies.next_request(cfg, 599)["seconds"] == 120
    assert policies.next_request(cfg, 599)["rounding_seconds"] == 1
    assert policies.next_request(cfg, 659)["seconds"] == 60
    assert policies.next_request(cfg, 661) is None
    assert policies.seconds("2-03:00:00") == 51 * 3600


@pytest.mark.parametrize(
    "values",
    [
        {"min_time": "00:05:00"},
        {"checkpoint_margin": "00:04:00"},
        {"unknown": True},
        {"max_attempts": 0},
        {"max_attempts": True},
    ],
)
def test_policy_rejects_invalid(values):
    with pytest.raises(ValueError):
        config(**values)


def test_pbs_variable_is_explicitly_gated():
    values = {
        "runtime_budget": "01:00:00",
        "min_time": "00:10:00",
        "checkpoint_margin": "00:01:00",
    }
    with pytest.raises(ValueError, match="site validation"):
        policies.policy(values, "00:20:00", "pbs")
    assert (
        policies.policy({**values, "allow_pbs_shrink_to_fit": True}, "00:20:00", "pbs")[
            "minimum"
        ]
        == 600
    )


def test_pause_resume_accounting_and_duplicate_intent(tmp_path, monkeypatch):
    allocation(tmp_path, 1)
    with pytest.raises(ValueError, match="reconcile"):
        storage.reserve(
            tmp_path, "scientific-identity", config(), "slurm", tmp_path / "duplicate"
        )
    monkeypatch.setattr(
        pipeline,
        "accounting",
        lambda *_: {"success": True, "seconds": 200, "status": "COMPLETED"},
    )
    state = pipeline.reconcile(tmp_path)
    assert state["state"] == "paused"
    assert state["used_seconds"] == 200
    assert pipeline.reconcile(tmp_path)["used_seconds"] == 200
    allocation(tmp_path, 2, "completed")
    assert pipeline.reconcile(tmp_path)["state"] == "completed"
    with pytest.raises(ValueError):
        storage.reserve(
            tmp_path, "changed-config", config(), "slurm", tmp_path / "changed"
        )


@pytest.mark.parametrize(
    "failure", ["checksum", "missing", "stale", "no_progress", "hard_kill"]
)
def test_fail_closed_and_charge_failed_allocation(tmp_path, monkeypatch, failure):
    attempt = allocation(tmp_path, 1)
    outcome = Path(attempt["directory"]) / "outcome.json"
    data = storage.read(outcome)
    if failure == "checksum":
        Path(data["checkpoint"]).write_bytes(b"partial")
    elif failure == "missing":
        outcome.unlink()
    elif failure == "stale":
        data["token"] = "previous-attempt"
        storage.atomic_json(outcome, data)
    elif failure == "no_progress":
        path = storage.manifest_path(tmp_path)
        state = storage.read(path)
        state["progress"] = data["iteration"]
        storage.atomic_json(path, state)
    monkeypatch.setattr(
        pipeline,
        "accounting",
        lambda *_: {
            "success": failure != "hard_kill",
            "seconds": 240,
            "status": "TIMEOUT",
        },
    )
    state = pipeline.reconcile(tmp_path)
    assert state["state"] == "failed"
    assert state["used_seconds"] == 240


def test_cancel_during_checkpointing_inhibits_completion(tmp_path, monkeypatch):
    allocation(tmp_path, 1, "completed")
    path = storage.manifest_path(tmp_path)
    state = storage.read(path)
    state["state"] = "cancelled"
    storage.atomic_json(path, state)
    monkeypatch.setattr(
        pipeline, "accounting", lambda *_: {"success": True, "seconds": 200}
    )
    assert pipeline.reconcile(tmp_path)["state"] == "cancelled"


def test_actual_pbs_deadline_not_requested_maximum(monkeypatch):
    start = 1700000000
    monkeypatch.setattr(
        scheduling,
        "command",
        lambda _: json.dumps(
            {
                "Jobs": {
                    "1.server": {
                        "stime": start,
                        "Resource_List": {
                            "walltime": "00:06:00",
                            "max_walltime": "24:00:00",
                        },
                    }
                }
            }
        ),
    )
    assert scheduling.allocation("pbs", "1.server") == (start, start + 360)


def test_actual_slurm_deadline_typed_timestamps(monkeypatch):
    monkeypatch.setattr(
        scheduling,
        "command",
        lambda _: json.dumps(
            {
                "jobs": [
                    {
                        "start_time": {"number": 1700000000},
                        "end_time": {"number": 1700000360},
                    }
                ]
            }
        ),
    )
    assert scheduling.allocation("slurm", "1") == (1700000000, 1700000360)


@pytest.mark.parametrize("split", [False, True])
def test_two_pauses_then_completion_submits_consumer_exactly_once(
    tmp_path, monkeypatch, split
):
    run = tmp_path / "train"
    continuation_dir = tmp_path / "scratch" if split else None
    path = tmp_path / "controller.json"
    storage.atomic_json(
        path,
        {
            "state": "active",
            "basedir": str(tmp_path),
            "generation": 0,
            "records": [
                {
                    "name": "train",
                    "kind": "spine",
                    "status": "waiting",
                    "jobs": [],
                    "options": {
                        "run_dir": str(run),
                        "continuation_dir": str(continuation_dir) if split else None,
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
    calls = []

    def submit(**options):
        calls.append(options)
        if "continuation" in options:
            number = len([v for v in calls if "continuation" in v])
            allocation(
                run,
                number,
                "completed" if number == 3 else "paused",
                continuation_dir=continuation_dir,
            )
        return [str(len(calls))]

    context = SimpleNamespace(
        config_mgr=SimpleNamespace(get_profile=lambda *_: {"scheduler": "slurm"}),
        submit_job=submit,
    )
    # Queued/running allocations have no accounting, even if an outcome exists.
    monkeypatch.setattr(pipeline, "accounting", lambda *_: None)
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    assert len(calls) == 1
    monkeypatch.setattr(
        pipeline, "accounting", lambda *_: {"success": True, "seconds": 200}
    )
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    assert len(calls) == 3
    assert all("continuation" in call for call in calls)
    assert calls[1]["resume_from"].endswith("snapshot-10.ckpt")
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    assert len(calls) == 4
    assert storage.read(path)["state"] == "completed"


@pytest.mark.parametrize("site", ["nersc.sbatch", "s3df.sbatch", "anl.pbs"])
def test_directives_and_shell_syntax(site, tmp_path):
    import subprocess

    source = Path("templates") / ("job_template_" + site)
    rendered = Template(source.read_text()).render(
        time="01:00:00",
        continuation_attempt={"token": "spc-test", "request": {"min_time": "00:10:00"}},
        continuation_policy={"checkpoint_margin": 60},
        continuation_run_dir=str(tmp_path),
        continuation_dir="/scratch/control",
        basedir=str(Path.cwd()),
        graceful_stop_file="/tmp/graceful-stop",
        gpus_per_node=1,
        cpus_per_task=1,
        spine_cmd="spine",
        config="test.yaml",
        account="test",
    )
    if site.endswith("pbs"):
        assert "min_walltime=00:10:00,max_walltime=01:00:00" in rendered
        assert "#PBS -l walltime=" not in rendered
    else:
        assert "--time-min=00:10:00" in rendered
        assert "--signal=B:USR2@60" in rendered
    assert "--require-continuation" in rendered
    assert "continuation_runtime.py" in rendered
    assert '--continuation-dir "/scratch/control"' in rendered
    script = tmp_path / "job.sh"
    script.write_text(rendered)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_concurrent_reservations_submit_only_one_intent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def reserve(number):
        try:
            return storage.reserve(
                tmp_path, "identity", config(), "slurm", tmp_path / str(number)
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, [1, 2]))
    assert len([result for result in results if result]) == 1
    assert len(storage.read(storage.manifest_path(tmp_path))["attempts"]) == 1


def test_atomic_publish_interruption_leaves_previous_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    storage.atomic_json(path, {"generation": 1})

    def fail(*_):
        raise OSError("interrupted before rename")

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError):
        storage.atomic_json(path, {"generation": 2})
    assert storage.read(path) == {"generation": 1}


def test_budget_extension_and_explicit_accept(tmp_path, monkeypatch):
    run = tmp_path / "train"
    allocation(run, 1)
    monkeypatch.setattr(
        pipeline, "accounting", lambda *_: {"success": True, "seconds": 720}
    )
    assert pipeline.reconcile(run)["state"] == "budget_exhausted"
    path = tmp_path / "pipeline.json"
    storage.atomic_json(
        path,
        {
            "state": "active",
            "generation": 0,
            "records": [
                {
                    "name": "train",
                    "options": {"run_dir": str(run), "continuation": {}},
                    "jobs": ["1"],
                }
            ],
        },
    )
    pipeline.operate(path, "extend", "train", "00:20:00")
    assert storage.read(storage.manifest_path(run))["state"] == "paused"
    assert storage.read(storage.manifest_path(run))["budget_override"] == 1200
    pipeline.operate(path, "accept", "train")
    assert storage.read(storage.manifest_path(run))["state"] == "completed"


def test_shell_trap_supervisor_and_launcher_handle_repeated_pause(tmp_path):
    """Exercise real Bash wait/traps and supervisor processes without a scheduler."""
    import os
    import shlex
    import signal
    import subprocess
    import sys
    import time

    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    attempt = storage.reserve(tmp_path, "identity", config(), "slurm", attempt_dir)
    storage.submitted(tmp_path, attempt["token"], "123")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    scheduler = bindir / "scontrol"
    now = int(time.time())
    scheduler.write_text(
        "#!/bin/sh\n"
        + "printf '%s\\n' '"
        + json.dumps({"jobs": [{"start_time": now, "end_time": now + 240}]})
        + "'\n"
    )
    scheduler.chmod(0o755)
    # The worker uses the compute host interpreter, independent of submission.
    (bindir / "python3").symlink_to(sys.executable)
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, time\np=pathlib.Path("
        + repr(str(attempt_dir / "pause"))
        + ")\n"
        "while not p.exists(): time.sleep(.02)\ntime.sleep(.2)\n"
    )
    rendered = Template(Path("templates/job_template_nersc.sbatch").read_text()).render(
        continuation_attempt=attempt,
        continuation_policy=config(),
        continuation_run_dir=str(tmp_path),
        basedir=str(Path.cwd()),
        gpus_per_node=1,
    )
    block = rendered.split("# Keep the batch shell responsive", 1)[1].split(
        "trap - USR1 USR2 TERM INT", 1
    )[0]
    script = tmp_path / "shell.sh"
    script.write_text(
        "#!/bin/bash\nRUN_CMD="
        + shlex.quote(f"{sys.executable} {child}")
        + "\n# Keep the batch shell responsive"
        + block
        + "\nexit $EXIT_STATUS\n"
    )
    env = {
        **os.environ,
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "SLURM_JOB_ID": "123",
    }
    process = subprocess.Popen(
        ["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 10
        while storage.read(storage.manifest_path(tmp_path))["state"] != "running":
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        # Supervisor installs its handlers immediately after committing running.
        time.sleep(0.1)
        os.kill(process.pid, signal.SIGUSR2)
        time.sleep(0.05)
        os.kill(process.pid, signal.SIGUSR2)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr.decode()
        assert (attempt_dir / "pause").is_file()
        assert not (attempt_dir / "graceful_stop").exists()
        state = storage.read(storage.manifest_path(tmp_path))
        assert state["state"] == "pausing"
        assert state["attempts"][0]["launcher_exit"] == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_metrics_separates_queue_runtime_and_pause_latency(tmp_path, monkeypatch):
    import csv
    import io

    run = tmp_path / "train"
    attempt = allocation(run, 1)
    path = storage.manifest_path(run)
    logical = storage.read(path)
    record = logical["attempts"][0]
    record.update(
        start=record["intent_time"] + 40, pause_requested_at=100, assigned_seconds=240
    )
    storage.atomic_json(path, logical)
    outcome_path = Path(attempt["directory"]) / "outcome.json"
    outcome = storage.read(outcome_path)
    outcome["committed_at"] = 109
    storage.atomic_json(outcome_path, outcome)
    controller = tmp_path / "controller.json"
    storage.atomic_json(
        controller,
        {
            "records": [
                {"name": "train", "options": {"run_dir": str(run), "continuation": {}}}
            ]
        },
    )
    monkeypatch.setattr(
        pipeline, "accounting", lambda *_: {"success": True, "seconds": 205}
    )
    stream = io.StringIO()
    pipeline.metrics(controller, stream)
    row = list(csv.DictReader(io.StringIO(stream.getvalue())))[0]
    assert float(row["queue_seconds"]) == 40
    assert row["runtime_seconds"] == "205"
    assert row["pause_to_commit_seconds"] == "9"
    assert row["accounting_confirmed"] == "True"


@pytest.mark.parametrize("capability,expected", [("2", 0), ("1", 1), ("0", 1)])
def test_runtime_probes_actual_executable_contract(tmp_path, capability, expected):
    import subprocess
    import sys

    executable = tmp_path / "spine.py"
    executable.write_text(
        "import sys\nprint("
        + repr(capability)
        + " if sys.argv[-1] == '--continuation-contract' else 'SPINE 1.4.4')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_spine_version.py",
            "--require-continuation",
            sys.executable,
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected


@pytest.mark.parametrize(
    "path",
    [
        "/global/cfs/cdirs/m5252/run",
        "/global/homes/d/user/run",
        "/global/u2/u/user/run",
        "/dvs_ro/cfs/cdirs/m5252/run",
    ],
)
def test_nersc_dvs_workspace_rejected_before_submission(path):
    with pytest.raises(ValueError, match=r"\$SCRATCH"):
        storage.validate_storage(path, "nersc")


def test_nersc_symlink_into_cfs_rejected(tmp_path):
    link = tmp_path / "workspace"
    link.symlink_to("/global/cfs/cdirs/m5252/run")
    with pytest.raises(ValueError, match="DVS"):
        storage.validate_storage(link, "nersc")


def test_dvs_errno_has_actionable_message_and_never_enters_lock(tmp_path, monkeypatch):
    def unsupported(*_):
        raise OSError(524, "Unknown error 524")

    monkeypatch.setattr(storage.fcntl, "flock", unsupported)
    with pytest.raises(RuntimeError, match="Do not disable locking"):
        with storage.locked(tmp_path / "manifest.json"):
            pytest.fail("An unsupported lock must not be bypassed")


def test_storage_preflight_leaves_scientific_state_unchanged(tmp_path):
    path = storage.manifest_path(tmp_path)
    storage.atomic_json(path, {"sentinel": 1})
    storage.preflight_storage(tmp_path, "anl")
    assert storage.read(path) == {"sentinel": 1}
    assert not list(path.parent.glob("preflight-*.json"))


def test_preferred_chunk_and_site_minimum():
    import yaml

    catalog = yaml.safe_load(Path("templates/profiles.yaml").read_text())
    profile = dict(
        catalog["profiles"]["nersc_gpu_full"], time="48:00:00", qos="preempt"
    )
    cfg = policies.profile_policy(
        catalog,
        profile,
        {
            "runtime_budget": "120:00:00",
            "preferred_time": "24:00:00",
            "site_min_time": "00:01:00",
            "checkpoint_margin": "00:15:00",
        },
    )
    assert cfg["site_min_time"] == 7200
    assert policies.next_request(cfg, 0)["seconds"] == 86400
    assert policies.next_request(cfg, 119 * 3600) is None
    assert policies.next_request(cfg, 118 * 3600)["seconds"] == 7200
    assert cfg["runtime_budget"] == 120 * 3600
    with pytest.raises(ValueError, match="max_time"):
        policies.profile_policy(
            catalog, dict(profile, time="120:00:00"), {"runtime_budget": "240:00:00"}
        )


@pytest.mark.parametrize(
    "values",
    [
        {"preferred_time": "00:05:00"},
        {"preferred_time": "00:01:00", "min_time": "00:02:00"},
        {"max_infrastructure_retries": -1},
        {"max_infrastructure_retries": True},
    ],
)
def test_invalid_chunk_and_retry_policy(values):
    with pytest.raises(ValueError):
        config(**values)


@pytest.mark.parametrize(
    "scheduler_state,expected,recoverable",
    [
        ("PREEMPTED", "preemption", True),
        ("NODE_FAIL", "node_failure", True),
        ("TIMEOUT", "allocation_expiry", True),
        ("FAILED", "application_failure", False),
        ("OUT_OF_MEMORY", "out_of_memory", False),
        ("CANCELLED", "user_cancelled", False),
    ],
)
def test_bounded_infrastructure_recovery_never_retries_application_errors(
    tmp_path, monkeypatch, scheduler_state, expected, recoverable
):
    attempt = allocation(tmp_path, 1)
    Path(
        attempt["directory"], "outcome.json"
    ).unlink()  # hard kill, periodic artifact survives
    path = storage.manifest_path(tmp_path)
    state = storage.read(path)
    state["policy"]["max_infrastructure_retries"] = 1
    storage.atomic_json(path, state)
    monkeypatch.setattr(
        pipeline,
        "accounting",
        lambda *_: {"status": scheduler_state, "success": False, "seconds": 240},
    )
    result = pipeline.reconcile(tmp_path)
    assert result["used_seconds"] == 240
    assert result["reason"] == expected
    assert result["state"] == (
        "paused"
        if recoverable
        else "cancelled" if scheduler_state == "CANCELLED" else "failed"
    )
    assert pipeline.reconcile(tmp_path)["used_seconds"] == 240
    if recoverable:
        directory = tmp_path / "attempts/2"
        directory.mkdir()
        second = storage.reserve(
            tmp_path,
            "scientific-identity",
            config(max_infrastructure_retries=1),
            "slurm",
            directory,
        )
        storage.submitted(tmp_path, second["token"], "2")
        checkpoint(tmp_path, 20)
        result = pipeline.reconcile(tmp_path)
        assert result["state"] == "failed"
        assert result["infrastructure_retries"] == 1
        assert result["used_seconds"] == 480


def test_preemption_without_new_checkpoint_stops(tmp_path, monkeypatch):
    allocation(tmp_path, 1)
    path = storage.manifest_path(tmp_path)
    state = storage.read(path)
    state["policy"]["max_infrastructure_retries"] = 2
    state["attempts"][0]["start_progress"] = 10
    storage.atomic_json(path, state)
    monkeypatch.setattr(
        pipeline,
        "accounting",
        lambda *_: {"status": "PREEMPTED", "success": False, "seconds": 120},
    )
    state = pipeline.reconcile(tmp_path)
    assert state["state"] == "failed"
    assert "No new verified" in state["detail"]
    assert state["attempts"][0]["reason"] == "preemption"


def test_periodic_controller_script_runs_with_minimal_environment(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    import sys

    basedir = tmp_path / "repo with spaces"
    scripts = basedir / "scripts"
    scripts.mkdir(parents=True)
    # Use the real configure.sh, which legitimately reads unset shell variables.
    (basedir / "configure.sh").write_text(Path("configure.sh").read_text())
    (basedir / "DEFAULT_SPINE_VERSION").write_text("1.4.4\n")
    (scripts / "continue_training.py").write_text(
        "import os, sys\nassert sys.argv[1] == 'advance'\nassert os.environ['SPINE_CONTAINER_TAG'] == 'custom-tag'\nassert 'SLURM_MEM_PER_CPU' not in os.environ\n"
    )
    manifest = tmp_path / "run/.spine-prod/continuation_pipeline.json"
    manifest.parent.mkdir(parents=True)
    monkeypatch.setenv("SPINE_CONTAINER_TAG", "custom-tag")
    pipeline.write_tick_script(manifest, basedir)
    script = manifest.parent / "advance.sh"
    result = subprocess.run(
        ["bash", str(script)],
        env={"PATH": os.environ["PATH"], "SLURM_MEM_PER_CPU": "2048"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert sys.executable in script.read_text()
    assert (script.stat().st_mode & 0o777) == 0o700


def test_all_training_stages_report_policy_in_dry_run(tmp_path, capsys, monkeypatch):
    from unittest.mock import Mock

    from src.submitter import Submitter

    monkeypatch.setenv("SCRATCH", "/pscratch/sd/t/test")
    context = Submitter()
    context.submit_job = Mock(return_value=[])
    stages = [
        {
            "name": name,
            "stage": "train",
            "config": "train.yaml",
            "run_dir": "/global/cfs/cdirs/project/" + name,
            "profile": "nersc_gpu_full",
            "account": "project",
            "time": "48:00:00",
            "continuation": {
                "runtime_budget": "120:00:00",
                "preferred_time": "24:00:00",
            },
        }
        for name in ["first", "second"]
    ]
    pipeline.start(
        context,
        stages,
        str(tmp_path),
        dry_run=True,
        continuation_dir="/pscratch/sd/t/test/continuation",
    )
    output = capsys.readouterr().out
    assert output.count("budget=120:00:00") == 2
    assert "preferred=24:00:00" in output
    assert "exhaustion=stop and block consumers" in output
    context.submit_job.assert_called_once()
    options = context.submit_job.call_args.kwargs
    assert options["run_dir"] == "/global/cfs/cdirs/project/first"
    assert options["continuation_dir"].startswith(
        "/pscratch/sd/t/test/continuation/runs/"
    )
    assert not (tmp_path / ".spine-prod/continuation_pipeline.json").exists()


def test_unknown_submission_reconciles_token_and_charges_once(tmp_path, monkeypatch):
    attempt = allocation(tmp_path, 1)
    path = storage.manifest_path(tmp_path)
    state = storage.read(path)
    state["attempts"][0]["job_id"] = None
    state["state"] = "submitting"
    storage.atomic_json(path, state)
    commands = []

    def query(argv):
        commands.append(argv)
        if any(arg.startswith("--name=") for arg in argv):
            return "42|" + attempt["token"] + "|\n"
        return "42|COMPLETED|200|0:0|\n"

    monkeypatch.setattr(scheduling, "command", query)
    result = pipeline.reconcile(tmp_path)
    assert result["attempts"][0]["job_id"] == "42"
    assert result["state"] == "paused"
    assert result["used_seconds"] == 200
    pipeline.reconcile(tmp_path)
    assert len(commands) == 2


def test_unknown_submission_with_no_match_never_resubmits(tmp_path, monkeypatch):
    attempt = storage.reserve(
        tmp_path, "identity", config(), "slurm", tmp_path / "attempt"
    )
    monkeypatch.setattr(scheduling, "command", lambda _: "")
    state = pipeline.reconcile(tmp_path)
    assert state["state"] == "submitting"
    assert state["attempts"][0]["token"] == attempt["token"]
    with pytest.raises(ValueError, match="reconcile"):
        storage.reserve(tmp_path, "identity", config(), "slurm", tmp_path / "other")


def test_recovery_rejects_ambiguous_scheduler_matches(monkeypatch):
    token = "spc-" + "a" * 20
    monkeypatch.setattr(scheduling, "command", lambda _: f"1|{token}|\n2|{token}|\n")
    with pytest.raises(RuntimeError, match="Multiple jobs"):
        scheduling.discover_submission("slurm", token)


def test_flock_failure_emits_traceback_and_diagnostic_without_launch(tmp_path):
    import os
    import subprocess
    import sys

    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    attempt = storage.reserve(tmp_path, "identity", config(), "slurm", attempt_dir)
    storage.submitted(tmp_path, attempt["token"], "123")
    script = str(Path("scripts/continuation_runtime.py").resolve())
    code = (
        "import runpy, sys, src.continuation.state as storage\n"
        "def fail(*args): raise OSError(524, 'Unknown error 524')\n"
        "storage.fcntl.flock = fail\n"
        f"sys.argv = [{script!r}, {str(tmp_path)!r}, {attempt['token']!r}, '--', 'does-not-exist']\n"
        f"runpy.run_path({script!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SLURM_JOB_ID": "123"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" in result.stderr
    assert "Do not disable locking" in result.stderr
    assert storage.read(attempt_dir / "failure.json")["token"] == attempt["token"]
    assert storage.read(storage.manifest_path(tmp_path))["state"] == "pending"


def test_pipeline_stops_cfs_request_before_any_scheduler_call(tmp_path, mock_submitter):
    from unittest.mock import Mock

    import yaml

    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        yaml.safe_dump(
            {
                "workspace": "/global/cfs/cdirs/m5252/run",
                "stages": [
                    {
                        "name": "train",
                        "stage": "train",
                        "config": "train.yaml",
                        "profile": "nersc_gpu_full",
                        "run_dir": "${workspace}/train",
                        "continuation": {"runtime_budget": "120:00:00"},
                    }
                ],
            }
        )
    )
    mock_submitter.submit_job = Mock()
    with pytest.raises(ValueError, match="explicit --continuation-dir"):
        mock_submitter.submit_pipeline(str(pipeline_file), dry_run=True)
    mock_submitter.submit_job.assert_not_called()


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("assigned,used", [(40, 0), (240, 680)])
def test_watchdog_uses_actual_deadline_and_remaining_budget(
    tmp_path, monkeypatch, assigned, used, split
):
    continuation_dir = tmp_path / "scratch" if split else None
    attempt_dir = (continuation_dir or tmp_path) / "attempt"
    attempt_dir.mkdir(parents=True)
    attempt = storage.reserve(
        tmp_path,
        "identity",
        config(),
        "slurm",
        attempt_dir,
        continuation_dir=continuation_dir,
    )
    storage.submitted(tmp_path, attempt["token"], "123", continuation_dir)
    manifest = storage.manifest_path(tmp_path, continuation_dir)
    state = storage.read(manifest)
    state["used_seconds"] = used
    storage.atomic_json(manifest, state)
    from src.continuation import runtime as module

    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda _: now.__setitem__(0, now[0] + 1))
    monkeypatch.setattr(module, "allocation", lambda *_: (1000, 1000 + assigned))
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    class Child:
        returncode = None

        def poll(self):
            assert now[0] <= 1031, "Watchdog incorrectly used requested walltime"
            if (attempt_dir / "pause").exists():
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Child())
    assert (
        module.run(str(tmp_path), attempt["token"], ["fake-launcher"], continuation_dir)
        == 0
    )
    state = storage.read(manifest)
    assert state["attempts"][0]["pause_requested_at"] == 1030
    assert state["attempts"][0]["assigned_seconds"] == assigned
    assert state["state"] == "pausing"


@pytest.mark.parametrize("profile", ["s3df_ampere", "s3df_ampere_full"])
def test_training_budget_survives_nersc_site_mapping(profile):
    """A site allocation ceiling must not truncate the logical training budget."""
    import yaml

    from src.sites import map_stage

    catalog = yaml.safe_load(Path("templates/profiles.yaml").read_text())
    mapped = map_stage(
        catalog,
        {"stage": "train", "profile": profile},
        "nersc",
        {"account": "project", "time": "48:00:00"},
    )
    effective = dict(catalog["profiles"][mapped["profile"]], time=mapped["time"])
    config = policies.profile_policy(
        catalog,
        effective,
        {"runtime_budget": "120:00:00", "preferred_time": "24:00:00"},
    )
    assert config["runtime_budget"] == 432000
    assert config["maximum"] == 172800
    assert policies.next_request(config, 0)["seconds"] == 86400
    assert policies.next_request(config, 4 * 86400)["seconds"] == 86400
    assert policies.next_request(config, 5 * 86400) is None


@pytest.mark.parametrize(
    "path",
    [None, "/tmp/state", "/global/cfs/cdirs/project/state", "/pscratch-looking/state"],
)
def test_nersc_requires_explicit_scratch_state(path):
    with pytest.raises(ValueError, match="continuation-dir"):
        storage.validate_storage(path, "nersc")


def test_nersc_scratch_state_and_symlink_accepted(tmp_path):
    storage.validate_storage("/pscratch/sd/u/user/continuation", "nersc")
    link = tmp_path / "state"
    link.symlink_to("/pscratch/sd/u/user/continuation")
    storage.validate_storage(link, "nersc")


def test_split_batch_keeps_training_artifacts_and_moves_control_files(
    tmp_path, mock_submitter, monkeypatch
):
    run = tmp_path / "cfs" / "train"
    control = tmp_path / "scratch" / "state"
    monkeypatch.setattr(
        mock_submitter.batch, "get_batch_client", lambda *_: mock_submitter.batch_client
    )
    monkeypatch.setattr(mock_submitter.batch_client, "submit", lambda *_: "42")
    mock_submitter.submit_job(
        config="config/train/generic/uresnet/train_240718.yaml",
        profile="s3df_ampere",
        stage="train",
        run_dir=str(run),
        continuation={"runtime_budget": "06:00:00"},
        continuation_dir=str(control),
    )
    assert not storage.manifest_path(run).exists()
    state = storage.read(storage.manifest_path(run, control))
    attempt = state["attempts"][0]
    assert attempt["job_id"] == "42"
    assert Path(attempt["directory"]).is_relative_to(control)
    script = (run / "latest" / "submit.sbatch").read_text()
    assert f"--log-dir {run}" in script
    assert f"--weight-prefix {run}/weights/snapshot" in script
    assert f'--continuation-dir "{control}"' in script
    assert f"SPINE_PAUSE_FILE={attempt['directory']}/pause" in script
    assert f"SPINE_OUTCOME_FILE={attempt['directory']}/outcome.json" in script
    assert f"--graceful-stop-file {attempt['directory']}/graceful_stop" in script
    assert (run / "latest" / "job_metadata.json").exists()
    assert not list(control.rglob("*.log"))


def test_split_preflight_locks_only_control_storage(tmp_path, monkeypatch):
    run, control = tmp_path / "cfs", tmp_path / "scratch"
    paths = []
    original = storage.locked

    def locked(path):
        paths.append(path)
        return original(path)

    monkeypatch.setattr(storage, "locked", locked)
    storage.preflight_storage(run, continuation_dir=control)
    assert paths == [storage.manifest_path(run, control)]
    assert not run.exists()


def test_initial_pending_validation_checkpoint_is_usable(tmp_path, monkeypatch):
    allocation(tmp_path, 1, iteration=-1)
    monkeypatch.setattr(
        pipeline, "accounting", lambda *_: {"success": True, "seconds": 200}
    )
    assert pipeline.reconcile(tmp_path)["state"] == "paused"


def test_finishing_pending_validation_counts_as_progress_once(tmp_path, monkeypatch):
    first = allocation(tmp_path, 1, iteration=10)
    path = Path(first["directory"]) / "outcome.json"
    result = storage.read(path)
    result["pending_validation"] = True
    storage.atomic_json(path, result)
    monkeypatch.setattr(
        pipeline, "accounting", lambda *_: {"success": True, "seconds": 200}
    )
    assert pipeline.reconcile(tmp_path)["state"] == "paused"
    second = allocation(tmp_path, 2, iteration=10)
    path = Path(second["directory"]) / "outcome.json"
    result = storage.read(path)
    result["pending_validation"] = False
    storage.atomic_json(path, result)
    assert pipeline.reconcile(tmp_path)["state"] == "paused"
    allocation(tmp_path, 3, iteration=10)
    assert pipeline.reconcile(tmp_path)["state"] == "failed"

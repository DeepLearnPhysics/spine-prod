"""Allocation lifecycle, accounting and consumer gating regression tests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Template

from src import continuation as c
from src import continuation_pipeline as pipeline


def config(**extra):
    return c.policy(
        {
            "runtime_budget": "00:12:00",
            "checkpoint_margin": "00:00:10",
            "startup_allowance": "00:00:05",
            **extra,
        },
        "00:04:00",
        "slurm",
    )


def checkpoint(run, iteration):
    path = run / "weights" / f"snapshot-{iteration}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"restart-state-{iteration}".encode())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{sha}  {path.name}\n")
    return path, sha


def allocation(run, number, outcome="paused", iteration=None):
    attempt_dir = run / "attempts" / str(number)
    attempt_dir.mkdir(parents=True)
    attempt = c.reserve(run, "scientific-identity", config(), "slurm", attempt_dir)
    c.submitted(run, attempt["token"], str(number))
    cp, sha = checkpoint(run, iteration if iteration is not None else number * 10)
    c.atomic_json(
        attempt_dir / "outcome.json",
        dict(
            contract=1,
            token=attempt["token"],
            state=outcome,
            checkpoint=str(cp),
            sha256=sha,
            iteration=iteration if iteration is not None else number * 10,
        ),
    )
    return attempt


@pytest.mark.parametrize("bad", ["0:00:00", "1:60:00", "10", -2, True])
def test_duration_rejects_invalid(bad):
    with pytest.raises(ValueError):
        c.seconds(bad)


def test_policy_final_chunk_and_rounding():
    cfg = config(min_time="00:02:00")
    assert c.next_request(cfg, 599)["seconds"] == 120
    assert c.next_request(cfg, 599)["rounding_seconds"] == 1
    assert c.next_request(cfg, 659)["seconds"] == 60
    assert c.next_request(cfg, 661) is None
    assert c.seconds("2-03:00:00") == 51 * 3600


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
        c.policy(values, "00:20:00", "pbs")
    assert (
        c.policy({**values, "allow_pbs_shrink_to_fit": True}, "00:20:00", "pbs")[
            "minimum"
        ]
        == 600
    )


def test_pause_resume_accounting_and_duplicate_intent(tmp_path, monkeypatch):
    allocation(tmp_path, 1)
    with pytest.raises(ValueError, match="reconcile"):
        c.reserve(
            tmp_path, "scientific-identity", config(), "slurm", tmp_path / "duplicate"
        )
    monkeypatch.setattr(
        c,
        "accounting",
        lambda *_: {"success": True, "seconds": 200, "status": "COMPLETED"},
    )
    state = c.reconcile(tmp_path)
    assert state["state"] == "paused"
    assert state["used_seconds"] == 200
    assert c.reconcile(tmp_path)["used_seconds"] == 200
    allocation(tmp_path, 2, "completed")
    assert c.reconcile(tmp_path)["state"] == "completed"
    with pytest.raises(ValueError):
        c.reserve(tmp_path, "changed-config", config(), "slurm", tmp_path / "changed")


@pytest.mark.parametrize(
    "failure", ["checksum", "missing", "stale", "no_progress", "hard_kill"]
)
def test_fail_closed_and_charge_failed_allocation(tmp_path, monkeypatch, failure):
    attempt = allocation(tmp_path, 1)
    outcome = Path(attempt["directory"]) / "outcome.json"
    data = c.read(outcome)
    if failure == "checksum":
        Path(data["checkpoint"]).write_bytes(b"partial")
    elif failure == "missing":
        outcome.unlink()
    elif failure == "stale":
        data["token"] = "previous-attempt"
        c.atomic_json(outcome, data)
    elif failure == "no_progress":
        path = c.manifest_path(tmp_path)
        state = c.read(path)
        state["progress"] = data["iteration"]
        c.atomic_json(path, state)
    monkeypatch.setattr(
        c,
        "accounting",
        lambda *_: {
            "success": failure != "hard_kill",
            "seconds": 240,
            "status": "TIMEOUT",
        },
    )
    state = c.reconcile(tmp_path)
    assert state["state"] == "failed"
    assert state["used_seconds"] == 240


def test_cancel_during_checkpointing_inhibits_completion(tmp_path, monkeypatch):
    allocation(tmp_path, 1, "completed")
    path = c.manifest_path(tmp_path)
    state = c.read(path)
    state["state"] = "cancelled"
    c.atomic_json(path, state)
    monkeypatch.setattr(c, "accounting", lambda *_: {"success": True, "seconds": 200})
    assert c.reconcile(tmp_path)["state"] == "cancelled"


def test_actual_pbs_deadline_not_requested_maximum(monkeypatch):
    start = 1700000000
    monkeypatch.setattr(
        c,
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
    assert c.allocation("pbs", "1.server") == (start, start + 360)


def test_actual_slurm_deadline_typed_timestamps(monkeypatch):
    monkeypatch.setattr(
        c,
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
    assert c.allocation("slurm", "1") == (1700000000, 1700000360)


def test_two_pauses_then_completion_submits_consumer_exactly_once(
    tmp_path, monkeypatch
):
    run = tmp_path / "train"
    path = tmp_path / "controller.json"
    c.atomic_json(
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
            allocation(run, number, "completed" if number == 3 else "paused")
        return [str(len(calls))]

    context = SimpleNamespace(
        config_mgr=SimpleNamespace(get_profile=lambda *_: {"scheduler": "slurm"}),
        submit_job=submit,
    )
    # Queued/running allocations have no accounting, even if an outcome exists.
    monkeypatch.setattr(c, "accounting", lambda *_: None)
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    assert len(calls) == 1
    monkeypatch.setattr(c, "accounting", lambda *_: {"success": True, "seconds": 200})
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    assert len(calls) == 3
    assert all("continuation" in call for call in calls)
    assert calls[1]["resume_from"].endswith("snapshot-10.ckpt")
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    pipeline.advance(path, context)
    assert len(calls) == 4
    assert c.read(path)["state"] == "completed"


@pytest.mark.parametrize("site", ["nersc.sbatch", "s3df.sbatch", "anl.pbs"])
def test_directives_and_shell_syntax(site, tmp_path):
    import subprocess

    source = Path("templates") / ("job_template_" + site)
    rendered = Template(source.read_text()).render(
        time="01:00:00",
        continuation_attempt={"token": "spc-test", "request": {"min_time": "00:10:00"}},
        continuation_policy={"checkpoint_margin": 60},
        continuation_run_dir=str(tmp_path),
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
    script = tmp_path / "job.sh"
    script.write_text(rendered)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_concurrent_reservations_submit_only_one_intent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def reserve(number):
        try:
            return c.reserve(
                tmp_path, "identity", config(), "slurm", tmp_path / str(number)
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, [1, 2]))
    assert len([result for result in results if result]) == 1
    assert len(c.read(c.manifest_path(tmp_path))["attempts"]) == 1


def test_atomic_publish_interruption_leaves_previous_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    c.atomic_json(path, {"generation": 1})

    def fail(*_):
        raise OSError("interrupted before rename")

    monkeypatch.setattr(c.os, "replace", fail)
    with pytest.raises(OSError):
        c.atomic_json(path, {"generation": 2})
    assert c.read(path) == {"generation": 1}


def test_budget_extension_and_explicit_accept(tmp_path, monkeypatch):
    run = tmp_path / "train"
    allocation(run, 1)
    monkeypatch.setattr(c, "accounting", lambda *_: {"success": True, "seconds": 720})
    assert c.reconcile(run)["state"] == "budget_exhausted"
    path = tmp_path / "pipeline.json"
    c.atomic_json(
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
    assert c.read(c.manifest_path(run))["state"] == "paused"
    assert c.read(c.manifest_path(run))["budget_override"] == 1200
    pipeline.operate(path, "accept", "train")
    assert c.read(c.manifest_path(run))["state"] == "completed"


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
    attempt = c.reserve(tmp_path, "identity", config(), "slurm", attempt_dir)
    c.submitted(tmp_path, attempt["token"], "123")
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
    # Use the same interpreter/dependencies as pytest on the compute-host side.
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
        while c.read(c.manifest_path(tmp_path))["state"] != "running":
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
        state = c.read(c.manifest_path(tmp_path))
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
    path = c.manifest_path(run)
    logical = c.read(path)
    record = logical["attempts"][0]
    record.update(
        start=record["intent_time"] + 40, pause_requested_at=100, assigned_seconds=240
    )
    c.atomic_json(path, logical)
    outcome_path = Path(attempt["directory"]) / "outcome.json"
    outcome = c.read(outcome_path)
    outcome["committed_at"] = 109
    c.atomic_json(outcome_path, outcome)
    controller = tmp_path / "controller.json"
    c.atomic_json(
        controller,
        {
            "records": [
                {"name": "train", "options": {"run_dir": str(run), "continuation": {}}}
            ]
        },
    )
    monkeypatch.setattr(c, "accounting", lambda *_: {"success": True, "seconds": 205})
    stream = io.StringIO()
    pipeline.metrics(controller, stream)
    row = list(csv.DictReader(io.StringIO(stream.getvalue())))[0]
    assert float(row["queue_seconds"]) == 40
    assert row["runtime_seconds"] == "205"
    assert row["pause_to_commit_seconds"] == "9"
    assert row["accounting_confirmed"] == "True"


@pytest.mark.parametrize("capability,expected", [("1", 0), ("0", 1)])
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

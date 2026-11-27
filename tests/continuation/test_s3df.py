"""S3DF cron registration without contacting production SSH or schedulers."""

import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.continuation import controller, s3df, service
from src.continuation import state as storage


def test_transport_is_noninteractive_and_local_on_cron_host(monkeypatch):
    monkeypatch.setattr(s3df.socket, "gethostname", lambda: "sdfiana008")
    argv = s3df.command(["/usr/bin/crontab", "-l"])
    assert argv[-2:] == ["sdfcron001", "/usr/bin/crontab -l"]
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=yes" in argv
    monkeypatch.setattr(
        s3df.socket, "gethostname", lambda: "sdfcron001.slac.stanford.edu"
    )
    assert s3df.command(["/usr/bin/crontab", "-l"]) == ["/usr/bin/crontab", "-l"]


def test_s3df_paths_and_block(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root, registry = service.service_paths("neutrino:ml-dev", "s3df")
    assert root == tmp_path / ".spine-prod/automation/s3df"
    assert registry.parent.name == "neutrino:ml-dev"
    with pytest.raises(ValueError, match="valid account"):
        service.service_paths("bad\naccount", "s3df")
    block = s3df.schedule_block("neutrino:ml-dev", registry, tmp_path / "repo space")
    assert "* * * * * /usr/bin/env" in block
    assert "/usr/bin/flock -n" in block
    assert s3df.SLURM_BIN in block
    assert "ssh" not in block
    assert "#SCRON" not in block
    assert "controller.log" in block
    for suffix in ("%", "\n", "\r"):
        with pytest.raises(ValueError, match="Unsupported"):
            s3df.schedule_block("account", registry, tmp_path / ("bad" + suffix))


def test_preflight_uses_cron_environment_and_reports_errors(tmp_path, monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(s3df.subprocess, "run", run)
    monkeypatch.setattr(s3df.socket, "gethostname", lambda: "sdfiana008")
    s3df.preflight(tmp_path / "manifest", tmp_path)
    argv = shlex.split(run.call_args.args[0][-1])
    assert argv[:3] == ["/usr/bin/env", "PATH=" + s3df.CRON_PATH, sys.executable]
    assert argv[-1] == "--preflight-s3df"
    run.return_value = SimpleNamespace(returncode=1, stderr="denied", stdout="")
    with pytest.raises(RuntimeError, match="denied"):
        s3df.preflight(tmp_path / "manifest", tmp_path)


def test_local_preflight_checks_storage_and_dependencies(tmp_path, monkeypatch):
    import shutil

    manifest = tmp_path / "control/manifest.json"
    storage.atomic_json(
        manifest,
        {
            "basedir": str(Path.cwd()),
            "records": [
                {
                    "options": {
                        "run_dir": str(tmp_path / "run"),
                        "continuation_dir": str(tmp_path / "state"),
                    }
                }
            ],
        },
    )
    state = storage.read(manifest)
    state["records"].append({"options": {"run_dir": str(tmp_path / "consumer")}})
    storage.atomic_json(manifest, state)
    (manifest.parent / "advance.sh").write_text("#!/bin/bash\n")
    monkeypatch.setattr(s3df.socket, "gethostname", lambda: "sdfiana008")
    with pytest.raises(RuntimeError, match="must run on"):
        s3df.preflight_local(manifest)
    monkeypatch.setattr(s3df.socket, "gethostname", lambda: "sdfcron001")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Missing cron-host command"):
        s3df.preflight_local(manifest)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    s3df.preflight_local(manifest)
    assert (tmp_path / "run").is_dir()
    assert (tmp_path / "state").is_dir()
    (manifest.parent / "advance.sh").unlink()
    with pytest.raises(RuntimeError, match="cannot read"):
        s3df.preflight_local(manifest)


def test_register_tick_retire_and_reactivate(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(s3df.socket, "gethostname", lambda: "sdfiana008")
    monkeypatch.setattr(s3df, "preflight", Mock())
    manifest = tmp_path / "run/manifest.json"
    storage.atomic_json(
        manifest, {"state": "active", "records": [], "basedir": str(Path.cwd())}
    )
    original = "# Existing user task\n0 * * * * /my/task\n"
    schedule = [original]
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        command = shlex.split(argv[-1]) if argv[0] == "ssh" else argv
        if command == ["/usr/bin/crontab", "-l"]:
            return SimpleNamespace(returncode=0, stdout=schedule[0], stderr="")
        if command == ["/usr/bin/crontab", "-"]:
            schedule[0] = kwargs["input"]
            return SimpleNamespace(returncode=0)
        assert argv == ["bash", str(manifest.parent / "advance.sh")]
        assert not any(k.startswith("SLURM_") for k in kwargs["env"])
        state = storage.read(manifest)
        state["state"] = "completed"
        storage.atomic_json(manifest, state)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(service.subprocess, "run", run)
    registry = service.register(manifest, "neutrino:ml-dev", Path.cwd(), "s3df")
    service.register(manifest, "neutrino:ml-dev", Path.cwd(), "s3df")
    assert schedule[0].startswith(original)
    assert schedule[0].count("# BEGIN spine-prod") == 1
    assert storage.read(manifest)["automation"]["backend"] == "s3df-crontab"
    assert service.retire_if_idle(registry) is False
    # Cron executes locally, without SSH or an inherited scheduler allocation.
    monkeypatch.setattr(s3df.socket, "gethostname", lambda: "sdfcron001")
    calls.clear()
    monkeypatch.setenv("SLURM_JOB_ID", "unexpected")
    service.tick_registry(registry)
    assert all(argv[0] != "ssh" for argv in calls)
    assert schedule[0] == original
    assert storage.read(registry)["schedule_state"] == "stopped"
    with service.reactivating(storage.read(manifest)):
        assert "# BEGIN spine-prod" in schedule[0]
    assert storage.read(registry)["schedule_state"] == "active"


def test_s3df_tick_script_preserves_slurm_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    manifest = tmp_path / "manifest.json"
    controller.write_tick_script(manifest, Path.cwd(), "s3df")
    script = (tmp_path / "advance.sh").read_text()
    assert f'export PATH={s3df.SLURM_BIN}:"$PATH"' in script
    subprocess.run(["bash", "-n", str(tmp_path / "advance.sh")], check=True)


def test_preflight_entrypoint(monkeypatch):
    import runpy

    target = Mock()
    monkeypatch.setattr(s3df, "preflight_local", target)
    monkeypatch.setattr(
        sys, "argv", ["continuation_service.py", "manifest.json", "--preflight-s3df"]
    )
    runpy.run_path("scripts/continuation_service.py", run_name="__main__")
    target.assert_called_once_with("manifest.json")


def test_preflight_failure_never_installs_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    preflight = Mock(side_effect=RuntimeError("SSH denied"))
    monkeypatch.setattr(s3df, "preflight", preflight)
    schedule = Mock()
    monkeypatch.setattr(service, "read_schedule", schedule)
    with pytest.raises(RuntimeError, match="SSH denied"):
        service.register(tmp_path / "manifest", "neutrino:ml-dev", tmp_path, "s3df")
    schedule.assert_not_called()


def test_empty_regular_crontab(monkeypatch):
    monkeypatch.setattr(
        service.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="no crontab for drielsma"
            )
        ),
    )
    assert service.read_schedule("s3df") == ""

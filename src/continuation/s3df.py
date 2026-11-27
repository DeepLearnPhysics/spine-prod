"""S3DF cron transport; training and reconciliation remain in the shared service."""

import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

HOST = "sdfcron001"
SLURM_BIN = "/opt/slurm/slurm-curr/bin"
CRON_PATH = SLURM_BIN + ":/usr/local/bin:/usr/bin:/bin"


def command(argv):
    """Use local commands on the cron host, otherwise noninteractive SSH."""
    if socket.gethostname().split(".")[0] == HOST:
        return argv
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=yes",
        HOST,
        shlex.join(argv),
    ]


def preflight(manifest, basedir):
    """Check cron-host dependencies and shared state before registering jobs."""
    result = subprocess.run(
        command(
            [
                "/usr/bin/env",
                "PATH=" + CRON_PATH,
                sys.executable,
                str(Path(basedir).resolve() / "scripts/continuation_service.py"),
                str(manifest),
                "--preflight-s3df",
            ]
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(
            "S3DF cron-host preflight failed; training not submitted: "
            + (result.stderr or result.stdout)
        )


def preflight_local(manifest):
    """Run on sdfcron001 under the same minimal PATH as the cron entry."""
    import shutil

    from src import Submitter  # Verify submission dependencies on the cron host.

    from .state import preflight_storage, read

    if socket.gethostname().split(".")[0] != HOST:
        raise RuntimeError("S3DF cron preflight must run on sdfcron001")
    for executable in ("sbatch", "sacct", "squeue", "scancel", "crontab", "flock"):
        if not shutil.which(executable):
            raise RuntimeError("Missing cron-host command: " + executable)
    state = read(manifest)
    Submitter(basedir=Path(state["basedir"]))
    paths = {Path(manifest).parent}
    for record in state["records"]:
        options = record["options"]
        paths.add(Path(options["run_dir"]))
        if options.get("continuation_dir"):
            paths.add(Path(options["continuation_dir"]))
    for path in paths:
        preflight_storage(path, "s3df")
    if not os.access(Path(manifest).parent / "advance.sh", os.R_OK):
        raise RuntimeError("Cron host cannot read the pipeline advance script")


def schedule_block(account, registry, basedir):
    """Generate a non-overlapping, minimal-environment cron invocation."""
    import hashlib

    key = hashlib.sha256(account.encode()).hexdigest()[:16]
    argv = [
        "/usr/bin/env",
        "PATH=" + CRON_PATH,
        "/usr/bin/flock",
        "-n",
        str(registry.parent / "cron-invocation.lock"),
        sys.executable,
        str(Path(basedir).resolve() / "scripts/continuation_service.py"),
        str(registry),
    ]
    invocation = (
        shlex.join(argv)
        + " >> "
        + shlex.quote(str(registry.parent / "controller.log"))
        + " 2>&1"
    )
    if any(c in invocation for c in ("\n", "\r", "%")):
        raise ValueError("Unsupported character in automatic controller path")
    return (
        f"# BEGIN spine-prod continuation {key}\n"
        f"* * * * * {invocation}\n"
        f"# END spine-prod continuation {key}\n"
    )

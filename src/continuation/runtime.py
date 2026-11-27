"""Supervise one real container launcher, using the assigned scheduler deadline."""

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .. import require_python
from .scheduler import allocation
from .state import atomic_json, locked, manifest_path, preflight_storage, read

require_python((3, 9))


def run(run_dir, token, argv, continuation_dir=None):
    """Supervise a launcher using the actual allocation deadline.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Logical training output directory.
    token : str
        Token identifying the allocation's durable submission intent.
    argv : sequence of str
        Launcher command, passed directly to subprocess.Popen.
    continuation_dir : str or pathlib.Path, optional
        Separate shared directory containing the run manifest.

    Returns
    -------
    int
        Launcher exit code after all child processes have been cleaned up.

    Notes
    -----
    Signal handlers only set flags. Manifest locking and control-file publication
    run in the polling loop so a signal cannot deadlock an interrupted transaction.
    """
    path = manifest_path(run_dir, continuation_dir)
    state = read(path)
    attempt = state["attempts"][-1]
    if attempt["token"] != token or state["state"] not in ("submitting", "pending"):
        raise ValueError("Stale or cancelled continuation allocation")
    scheduler = attempt["scheduler"]
    job_id = os.environ["SLURM_JOB_ID" if scheduler == "slurm" else "PBS_JOBID"]
    if attempt["job_id"] not in (None, job_id):
        raise ValueError("Allocation ID does not match submission intent")
    # Handlers only set flags: filesystem locking inside a signal handler can
    # deadlock if a signal interrupts our own manifest transaction.
    signals = {"pause": False, "complete": False, "terminate": None}

    def receive(signum, _):
        """Record a signal without performing I/O or acquiring locks."""
        if signum == signal.SIGUSR2:
            signals["pause"] = True
        elif signum == signal.SIGUSR1:
            signals["complete"] = True
        else:
            signals["terminate"] = signum

    for signum in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, receive)
    preflight_storage(
        run_dir,
        attempt.get("site") or ("nersc" if os.environ.get("NERSC_HOST") else None),
        continuation_dir=continuation_dir,
    )
    error = None
    for retry in range(3):
        try:
            start, end = allocation(scheduler, job_id)
            break
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            error = exc
            if retry < 2:
                time.sleep(2)
    else:
        raise RuntimeError(f"Cannot establish actual allocation deadline: {error}")
    directory = Path(attempt["directory"])
    pause_file = directory / "pause"
    complete_file = directory / "graceful_stop"
    # A unique attempt is never reused or cleared while another process is live.
    if pause_file.exists() or (directory / "outcome.json").exists():
        raise ValueError("Attempt has stale control/output files; use a new attempt")
    with locked(path):
        state = read(path)
        if state["attempts"][-1]["token"] != token or state["state"] not in (
            "pending",
            "submitting",
        ):
            raise ValueError("Allocation cancelled during deadline discovery")
        state["attempts"][-1].update(
            job_id=job_id, start=start, deadline=end, assigned_seconds=end - start
        )
        state.update(state="running", generation=state["generation"] + 1)
        atomic_json(path, state)
    deadline = min(
        end, start + state["policy"]["runtime_budget"] - state["used_seconds"]
    )
    pause_at = deadline - state["policy"]["checkpoint_margin"]
    if pause_at - time.time() < state["policy"]["startup_allowance"]:
        raise ValueError(
            "Assigned allocation cannot fit remaining startup_allowance plus checkpoint_margin"
        )
    requested = False

    def pause(*_):
        """Publish the pause request once and record its timestamp."""
        nonlocal requested
        if not requested:
            pause_file.touch(exist_ok=True)
            requested = True
            with locked(path):
                current = read(path)
                current["attempts"][-1]["pause_requested_at"] = time.time()
                if current["state"] == "running":
                    current.update(
                        state="pausing", generation=current["generation"] + 1
                    )
                    atomic_json(path, current)

    if signals["terminate"] is not None:
        raise RuntimeError("Allocation terminated during startup")
    child = subprocess.Popen(argv, start_new_session=True)
    try:
        while child.poll() is None:
            if signals["terminate"] is not None:
                os.killpg(child.pid, signals["terminate"])
                signals["terminate"] = None
            if signals["complete"]:
                complete_file.touch(exist_ok=True)
                signals["complete"] = False
            if signals["pause"] or time.time() >= pause_at:
                pause()
            time.sleep(0.2)
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        with locked(path):
            state = read(path)
            attempt = state["attempts"][-1]
            if attempt["token"] == token:
                attempt.update(
                    end=time.time(),
                    provisional_seconds=time.time() - start,
                    launcher_exit=child.returncode,
                )
                state["generation"] += 1
                atomic_json(path, state)


def main():
    """Parse the runtime CLI and report supervisor failures durably."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-dir")
    parser.add_argument("run_dir")
    parser.add_argument("token")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        sys.exit(
            run(
                args.run_dir,
                args.token,
                args.command[1:] if args.command[0] == "--" else args.command,
                continuation_dir=args.continuation_dir,
            )
        )
    except Exception as error:
        trace = traceback.format_exc()
        print(f"Continuation runtime failed: {error}\n{trace}", file=sys.stderr)
        # Diagnostic publication does not require the manifest lock that may
        # itself have failed. Never rewrite scientific/controller state here.
        try:
            state = read(manifest_path(args.run_dir, args.continuation_dir))
            attempt = state["attempts"][-1]
            if attempt["token"] == args.token:
                atomic_json(
                    Path(attempt["directory"]) / "failure.json",
                    {
                        "token": args.token,
                        "reason": "runtime_supervisor_failure",
                        "error": str(error),
                        "traceback": trace,
                        "time": time.time(),
                    },
                )
        except Exception as diagnostic_error:
            print(
                f"Could not publish failure diagnostics: {diagnostic_error}",
                file=sys.stderr,
            )
        sys.exit(1)

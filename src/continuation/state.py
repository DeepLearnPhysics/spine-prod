"""Durable manifests and serialized submission intent on shared storage."""

import contextlib
import errno
import fcntl
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from .policy import next_request

CONTRACT = 1


def validate_storage(path, site):
    """Require explicit Scratch continuation storage on NERSC before submission.

    Check canonical paths as well as the provided spelling (symlinks included).
    This is not a substitute for compute-host preflight on other filesystems.
    """
    if site != "nersc":
        return
    if not path:
        raise ValueError(
            "NERSC continuation requires an explicit --continuation-dir on $SCRATCH"
        )
    resolved = Path(path).expanduser().resolve()
    roots = ("/pscratch", "/global/cscratch1", "/global/cscratch2")
    if not any(str(resolved).startswith(root + "/") for root in roots):
        raise ValueError(
            f"NERSC --continuation-dir {resolved} must be on Scratch ($SCRATCH); "
            "CFS and other DVS paths cannot hold continuation locks. "
            "The workspace, training logs and checkpoints may remain on CFS."
        )


def preflight_storage(run_dir, site=None, continuation_dir=None):
    """Verify locking and atomic publication without modifying run state.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Training output directory.
    site : str, optional
        Site whose filesystem restrictions apply.
    continuation_dir : str or pathlib.Path, optional
        Separate state directory; defaults to the training directory.

    Notes
    -----
    The probe runs on the calling host. Login-node success alone does not
    establish compute-node support for the same filesystem.
    """
    validate_storage(
        continuation_dir if continuation_dir is not None else run_dir, site
    )
    # Use the actual run's lock. Test publication on a unique probe without
    # modifying the manifest or deleting another process's probe.
    path = manifest_path(run_dir, continuation_dir)
    with locked(path):
        probe = path.parent / ("preflight-" + uuid.uuid4().hex + ".json")
        try:
            atomic_json(probe, {"probe": True})
            if read(probe) != {"probe": True}:
                raise RuntimeError(
                    "Continuation filesystem publication verification failed"
                )
        finally:
            probe.unlink(missing_ok=True)


def atomic_json(path, data):
    """Publish a JSON document atomically and synchronize it to storage.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination; its parent directory is created when necessary.
    data : object
        JSON-serializable payload.

    Notes
    -----
    The temporary file and its parent directory are both synchronized. Callers
    must hold the manifest lock when updating an existing state document.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def locked(path):
    """Serialize readers that mutate state; replacement keeps readers atomic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno in (524, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise RuntimeError(
                    f"Continuation cannot lock {path}: this filesystem does not support "
                    "file locking from this host. On NERSC set --continuation-dir on "
                    "$SCRATCH, not CFS or Global Homes. Do not disable locking."
                ) from exc
            raise
        yield


def read(path):
    """Read a JSON state document without acquiring a mutation lock."""
    return json.loads(Path(path).read_text())


def manifest_path(run_dir, continuation_dir=None):
    """Locate run state in its configured continuation directory."""
    return (
        Path(continuation_dir if continuation_dir is not None else run_dir)
        / ".spine-prod"
        / "continuation.json"
    )


def reserve(
    run_dir, identity, config, scheduler, attempt_dir, site=None, continuation_dir=None
):
    """Persist an allocation intent before contacting the scheduler.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Logical training output directory.
    identity : str
        Digest of the scientific configuration and runtime topology.
    config : dict
        Normalized allocation policy.
    scheduler : {"slurm", "pbs"}
        Scheduler used for the new attempt.
    attempt_dir : str or pathlib.Path
        Unique directory for this allocation's control and outcome files.
    site : str, optional
        Site recorded for compute-side storage validation.
    continuation_dir : str or pathlib.Path, optional
        Separate directory containing the run manifest.

    Returns
    -------
    dict
        Durable attempt record, including token and requested allocation bounds.

    Raises
    ------
    ValueError
        The run identity changed, an attempt is unresolved, or a limit is reached.
    """
    path = manifest_path(run_dir, continuation_dir)
    with locked(path):
        state = (
            read(path)
            if path.exists()
            else {
                "contract": CONTRACT,
                "identity": identity,
                "generation": 0,
                "state": "pending",
                "used_seconds": 0,
                "attempts": [],
            }
        )
        if state["identity"] != identity:
            raise ValueError("Continuation scientific configuration/topology changed")
        if state["state"] not in ("pending", "paused") or (
            state["attempts"] and not state["attempts"][-1].get("accounted")
        ):
            raise ValueError(
                f"Cannot submit continuation in state {state['state']}; reconcile first"
            )
        config = dict(config)
        config["runtime_budget"] = state.get(
            "budget_override", config["runtime_budget"]
        )
        state["policy"] = config
        request = next_request(config, state["used_seconds"])
        if len(state["attempts"]) >= config["max_attempts"]:
            state.update(
                state="failed",
                reason="max_attempts reached",
                generation=state["generation"] + 1,
            )
            atomic_json(path, state)
            raise ValueError("Continuation max_attempts reached")
        if request is None:
            state.update(
                state="budget_exhausted",
                reason="runtime_budget_exhausted",
                generation=state["generation"] + 1,
            )
            atomic_json(path, state)
            raise ValueError("Continuation runtime budget exhausted")
        token = "spc-" + uuid.uuid4().hex[:20]
        attempt = {
            "token": token,
            "directory": str(attempt_dir),
            "scheduler": scheduler,
            "number": len(state["attempts"]) + 1,
            "site": site,
            "start_progress": state.get("progress", -1),
            "request": request,
            "intent_time": time.time(),
            "job_id": None,
        }
        state["attempts"].append(attempt)
        state.update(state="submitting", generation=state["generation"] + 1)
        atomic_json(path, state)
        return attempt


def submitted(run_dir, token, job_id, continuation_dir=None):
    """Record the scheduler response for an existing submission intent.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Training output directory identifying the logical run.
    token : str
        Unique token assigned before submission.
    job_id : str or None
        Scheduler identifier; None preserves the unresolved intent.
    continuation_dir : str or pathlib.Path, optional
        Separate directory containing the run manifest.

    Raises
    ------
    ValueError
        The token is stale or already belongs to another scheduler job.
    """
    path = manifest_path(run_dir, continuation_dir)
    with locked(path):
        state = read(path)
        attempt = state["attempts"][-1]
        if attempt["token"] != token:
            raise ValueError("Stale submission token")
        # An unknown response is deliberately left submitting for reconciliation.
        if job_id:
            if attempt.get("job_id") not in (None, job_id):
                raise ValueError("Submission token already belongs to another job")
            attempt["job_id"] = job_id
            if state["state"] == "submitting":
                state["state"] = "pending"
        state["generation"] += 1
        atomic_json(path, state)

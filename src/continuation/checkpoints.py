"""Checkpoint integrity checks and recovery artifact selection."""

from pathlib import Path

from ..runs import RunManager
from .state import CONTRACT, read


def verify_outcome(run_dir, attempt):
    """Validate an attempt's outcome and its committed checkpoint.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Training directory whose weights directory must contain the checkpoint.
    attempt : dict
        Attempt record carrying the expected token and outcome directory.

    Returns
    -------
    dict
        Outcome with a matching contract, token, checksum and progress counter.

    Raises
    ------
    ValueError
        The outcome is stale, malformed or references an invalid checkpoint.
    """
    outcome = read(Path(attempt["directory"]) / "outcome.json")
    if outcome.get("contract") != CONTRACT or outcome.get("token") != attempt["token"]:
        raise ValueError("Missing or stale SPINE continuation outcome")
    if outcome.get("state") not in ("paused", "completed"):
        raise ValueError("Invalid SPINE continuation outcome")
    checkpoint = Path(outcome["checkpoint"]).resolve()
    if checkpoint.parent != (Path(run_dir) / "weights").resolve():
        raise ValueError("Continuation checkpoint is outside the run weights directory")
    digest = RunManager.checkpoint_digest(checkpoint)
    if not digest or digest != outcome["sha256"]:
        raise ValueError(
            "Continuation requires a committed checkpoint and matching SHA256"
        )
    if type(outcome.get("iteration")) is not int or outcome["iteration"] < -1:
        raise ValueError("Invalid checkpoint progress")
    return outcome


def recovery_checkpoint(run_dir, attempt):
    """Only use integrity-checked snapshots from the interrupted allocation.

    Native SPINE enforces the full-state contract on resume. Never recover from
    best.ckpt or infer terminal success from a periodic checkpoint.
    """
    failures = []
    for iteration, path in reversed(RunManager.checkpoints(Path(run_dir))):
        if iteration <= attempt.get("start_progress", -1):
            continue
        try:
            digest = RunManager.checkpoint_digest(path)
            if digest:
                return dict(
                    checkpoint=str(path.resolve()), sha256=digest, progress=iteration
                )
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    raise ValueError(
        "No new verified periodic checkpoint for recovery"
        + (": " + "; ".join(failures) if failures else "")
    )

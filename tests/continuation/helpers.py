"""Small manifests and checksummed checkpoints for continuation tests."""

import hashlib
from pathlib import Path

from src.continuation import policy as policies
from src.continuation import state as storage


def config(**extra):
    return policies.policy(
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


def allocation(run, number, outcome="paused", iteration=None, continuation_dir=None):
    attempt_dir = (continuation_dir or run) / "attempts" / str(number)
    attempt_dir.mkdir(parents=True)
    attempt = storage.reserve(
        run,
        "scientific-identity",
        config(),
        "slurm",
        attempt_dir,
        continuation_dir=continuation_dir,
    )
    storage.submitted(run, attempt["token"], str(number), continuation_dir)
    cp, sha = checkpoint(run, iteration if iteration is not None else number * 10)
    storage.atomic_json(
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

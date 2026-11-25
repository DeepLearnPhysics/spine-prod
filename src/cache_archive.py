"""Verified immutable cache snapshots; archives never require filesystem locks."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .continuation.state import atomic_json


def digest(path):
    """Hash a file without loading shards into memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sync_directory(path):
    """Persist rename entries before publishing references to their files."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fingerprint(value):
    """Stable identity for a JSON snapshot."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def contained(root, relative):
    """Reject absolute paths, traversal, and symlinks outside the repository."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or path == root or root not in path.parents:
        raise ValueError("Cache shard escapes repository: " + str(relative))
    return path


def shard_paths(manifest, stage):
    """Only archive a fully published cache repository."""
    if manifest.get("format") != "spine_cache" or manifest.get("version") != 1:
        raise ValueError("Unsupported cache manifest")
    stages = manifest["stages"]
    if stage not in stages or manifest.get("replacements"):
        raise ValueError("Cache publication is missing or still in progress")
    expected = {source["id"] for source in manifest["sources"]}
    paths = set()
    for record in stages.values():
        if not record.get("complete", True) or set(record["shards"]) != expected:
            raise ValueError("Cache stage is incomplete")
        paths.update(record["shards"].values())
    return sorted(paths)


def archive(source, destination, stage, publication_id=None):
    """Copy committed shards under the working repository's publication lock.

    Shards are content-addressed in the archive, so snapshots share unchanged
    data. Publish the snapshot only after every copied object verifies. No lock
    is taken on the archive filesystem (which may be NERSC CFS).
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("Working cache and archive must be separate directories")
    objects = destination / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    with (source / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((source / "manifest.json").read_text())
        if (
            publication_id is not None
            and manifest.get("stages", {}).get(stage, {}).get("publication_id")
            != publication_id
        ):
            raise ValueError("Cache publication does not match the completed producer")
        files = {}
        for relative in shard_paths(manifest, stage):
            original = contained(source, relative)
            checksum = digest(original)
            target = objects / checksum
            if not target.exists():
                fd, temporary = tempfile.mkstemp(dir=objects, prefix=".copy-")
                os.close(fd)
                try:
                    shutil.copyfile(original, temporary)
                    if digest(temporary) != checksum:
                        raise ValueError("Cache archive copy checksum mismatch")
                    with open(temporary, "rb") as stream:
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            if digest(target) != checksum:
                raise ValueError("Cache archive object checksum mismatch")
            files[relative] = checksum
        sync_directory(objects)
        snapshot = {
            "format": "spine-prod-cache-archive",
            "version": 1,
            "manifest": manifest,
            "files": files,
            "stage": stage,
        }
        path = destination / "snapshots" / (fingerprint(snapshot) + ".json")
        atomic_json(path, snapshot)
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("Cache archive snapshot verification failed")
        return path


def restore(snapshot_path, destination):
    """Restore an explicitly selected archive into a fresh working repository.

    This restores cached data only, never continuation state or scheduler jobs.
    """
    snapshot_path, destination = (
        Path(snapshot_path).resolve(),
        Path(destination).resolve(),
    )
    snapshot = json.loads(snapshot_path.read_text())
    if (
        snapshot.get("format") != "spine-prod-cache-archive"
        or snapshot.get("version") != 1
        or snapshot_path.stem != fingerprint(snapshot)
    ):
        raise ValueError("Invalid cache archive snapshot")
    expected = shard_paths(snapshot["manifest"], snapshot["stage"])
    if set(expected) != set(snapshot["files"]):
        raise ValueError("Cache archive shard inventory mismatch")
    if destination.exists():
        raise ValueError("Restore destination must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".restore-"))
    try:
        for relative, checksum in snapshot["files"].items():
            if len(checksum) != 64 or any(
                c not in "0123456789abcdef" for c in checksum
            ):
                raise ValueError("Invalid cache object checksum")
            source = snapshot_path.parent.parent / "objects" / checksum
            target = contained(temporary, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if digest(target) != checksum:
                raise ValueError("Restored cache checksum mismatch")
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            parent = target.parent
            while parent != temporary:
                sync_directory(parent)
                parent = parent.parent
        # Active publication fences belong to the old run, not the restored copy.
        manifest = dict(snapshot["manifest"], publications={}, replacements={})
        atomic_json(temporary / "manifest.json", manifest)
        os.rename(temporary, destination)
        sync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

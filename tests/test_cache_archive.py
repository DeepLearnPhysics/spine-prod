"""Exercise durable cache publication and explicit data-only restoration."""

import json
from pathlib import Path

import pytest

from src import cache_archive as cache
from src.continuation.state import atomic_json


@pytest.fixture
def repository(tmp_path):
    source = tmp_path / "work"
    source.mkdir()
    (source / "shard.h5").write_bytes(b"cached scientific data")
    manifest = dict(
        format="spine_cache",
        version=1,
        sources=[{"id": "a"}],
        stages={
            "encoder": dict(
                complete=True, shards={"a": "shard.h5"}, publication_id="producer"
            )
        },
        publications={"encoder": "old fence"},
        replacements={},
    )
    atomic_json(source / "manifest.json", manifest)
    return source, tmp_path / "archive", manifest


def test_incremental_archive_restore(repository, tmp_path):
    source, destination, manifest = repository
    snapshot = cache.archive(source, destination, "encoder", "producer")
    assert cache.archive(source, destination, "encoder") == snapshot
    assert len(list((destination / "objects").iterdir())) == 1
    restored = tmp_path / "restored"
    cache.restore(snapshot, restored)
    assert (restored / "shard.h5").read_bytes() == (source / "shard.h5").read_bytes()
    assert json.loads((restored / "manifest.json").read_text()) == dict(
        manifest, publications={}
    )
    with pytest.raises(ValueError, match="must not exist"):
        cache.restore(snapshot, restored)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"format": "wrong"}, "Unsupported"),
        ({"version": 2}, "Unsupported"),
        ({"replacements": {"encoder": "pending"}}, "in progress"),
        ({"stages": {}}, "missing"),
        (
            {"stages": {"encoder": {"complete": False, "shards": {"a": "shard.h5"}}}},
            "incomplete",
        ),
        ({"stages": {"encoder": {"shards": {}}}}, "incomplete"),
    ],
)
def test_incomplete_publications_rejected(repository, mutation, message):
    source, destination, manifest = repository
    atomic_json(source / "manifest.json", dict(manifest, **mutation))
    with pytest.raises(ValueError, match=message):
        cache.archive(source, destination, "encoder")
    assert not (destination / "snapshots").exists()


@pytest.mark.parametrize("relative", ["/outside", "../outside", "."])
def test_containment(tmp_path, relative):
    with pytest.raises(ValueError, match="escapes"):
        cache.contained(tmp_path, relative)


def test_wrong_producer_and_overlap(repository):
    source, destination, _ = repository
    with pytest.raises(ValueError, match="producer"):
        cache.archive(source, destination, "encoder", "another-run")
    for target in (source, source / "nested", source.parent):
        with pytest.raises(ValueError, match="separate"):
            cache.archive(source, target, "encoder")


def test_interrupted_copy_does_not_publish(repository, monkeypatch):
    source, destination, _ = repository

    def partial(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(cache.shutil, "copyfile", partial)
    with pytest.raises(OSError, match="interrupted"):
        cache.archive(source, destination, "encoder")
    assert not list((destination / "objects").iterdir())
    assert not (destination / "snapshots").exists()


def test_corrupt_copy_and_existing_object(repository, monkeypatch):
    source, destination, _ = repository
    with monkeypatch.context() as patch:
        patch.setattr(
            cache.shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"bad")
        )
        with pytest.raises(ValueError, match="copy checksum"):
            cache.archive(source, destination, "encoder")
    snapshot = cache.archive(source, destination, "encoder")
    next((destination / "objects").iterdir()).write_bytes(b"bad")
    with pytest.raises(ValueError, match="object checksum"):
        cache.archive(source, destination, "encoder")
    restored = source.parent / "restored"
    with pytest.raises(ValueError, match="Restored cache checksum"):
        cache.restore(snapshot, restored)
    assert not restored.exists()
    assert not list(source.parent.glob(".restore-*"))


def test_snapshot_readback(repository, monkeypatch):
    source, destination, _ = repository
    monkeypatch.setattr(cache, "atomic_json", lambda p, d: atomic_json(p, {}))
    with pytest.raises(ValueError, match="snapshot verification"):
        cache.archive(source, destination, "encoder")


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"format": "bad"}, "Invalid cache archive"),
        ({"version": 2}, "Invalid cache archive"),
        ({"files": {}}, "inventory"),
        ({"files": {"shard.h5": "bad"}}, "Invalid cache object"),
        ({"files": {"shard.h5": "z" * 64}}, "Invalid cache object"),
    ],
)
def test_restore_validates_snapshot(repository, mutation, message):
    source, destination, _ = repository
    path = cache.archive(source, destination, "encoder")
    payload = dict(json.loads(path.read_text()), **mutation)
    changed = path.parent / (cache.fingerprint(payload) + ".json")
    atomic_json(changed, payload)
    with pytest.raises(ValueError, match=message):
        cache.restore(changed, source.parent / "new")


def test_restore_snapshot_identity(repository):
    source, destination, _ = repository
    path = cache.archive(source, destination, "encoder")
    payload = json.loads(path.read_text())
    payload["stage"] = "tampered"
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="Invalid cache archive"):
        cache.restore(path, source.parent / "new")


def test_script_archive_and_restore(repository, tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    source, destination, _ = repository
    script = Path(__file__).resolve().parents[1] / "scripts/cache_archive.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), "archive", str(source), str(destination)]
    )
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "archive", str(source), str(destination), "--stage", "encoder"],
    )
    runpy.run_path(str(script), run_name="__main__")
    snapshot = next((destination / "snapshots").iterdir())
    monkeypatch.setattr(
        sys, "argv", [str(script), "restore", str(snapshot), str(tmp_path / "restore")]
    )
    runpy.run_path(str(script), run_name="__main__")
    assert "Restored cache" in capsys.readouterr().out


def test_nested_shards_restore_durably(repository, tmp_path):
    source, destination, manifest = repository
    (source / "nested").mkdir()
    (source / "shard.h5").rename(source / "nested/shard.h5")
    manifest["stages"]["encoder"]["shards"]["a"] = "nested/shard.h5"
    atomic_json(source / "manifest.json", manifest)
    snapshot = cache.archive(source, destination, "encoder")
    cache.restore(snapshot, tmp_path / "restored")
    assert (
        tmp_path / "restored/nested/shard.h5"
    ).read_bytes() == b"cached scientific data"

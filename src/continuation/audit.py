"""Historical monitoring snapshots, deliberately without restore semantics."""

from pathlib import Path

from ..cache_archive import fingerprint
from .state import atomic_json, manifest_path, read


def archive_audit(manifest):
    """Verify a self-contained audit record before the monitor may retire.

    The caller holds the pipeline or registration lock. Account registries,
    executable tick scripts, checkpoints and cache payloads are not copied.
    Older campaigns without audit_dir retain their original behavior.
    """
    manifest = Path(manifest).resolve()
    state = read(manifest)
    if not state.get("audit_dir"):
        return None
    run_id = fingerprint(str(manifest))[:16]
    payload = {
        "format": "spine-prod-monitoring-audit",
        "version": 1,
        "historical_only": True,
        "pipeline": state,
        "stages": {},
        "controller_log": "",
        "summary": [],
    }
    log = manifest.parent / "controller.log"
    if log.exists():
        payload["controller_log"] = log.read_text(errors="replace")
    for record in state["records"]:
        options = record["options"]
        detail = {"record": record, "attempt_files": {}}
        status, reason, used = record["status"], None, None
        if options.get("continuation") is not None:
            logical_path = manifest_path(
                options["run_dir"], options.get("continuation_dir")
            )
            if logical_path.exists():
                logical = read(logical_path)
                detail["continuation"] = logical
                status, reason, used = (
                    logical["state"],
                    logical.get("reason"),
                    logical["used_seconds"],
                )
                for attempt in logical["attempts"]:
                    for filename in (
                        "outcome.json",
                        "failure.json",
                        "job_metadata.json",
                    ):
                        path = Path(attempt["directory"]) / filename
                        if path.exists():
                            detail["attempt_files"][
                                f"{attempt['number']}/{filename}"
                            ] = read(path)
        payload["stages"][record["name"]] = detail
        payload["summary"].append(
            {
                "stage": record["name"],
                "state": status,
                "reason": reason,
                "runtime_seconds": used,
                "jobs": record.get("jobs", []),
            }
        )
    directory = Path(state["audit_dir"]) / run_id
    checksum = fingerprint(payload)
    path = directory / (checksum + ".json")
    atomic_json(path, payload)
    if fingerprint(read(path)) != checksum:
        raise ValueError("Monitoring audit verification failed; timer retained")
    # A convenient index is published only after the immutable audit verifies.
    atomic_json(directory / "latest.json", {"snapshot": path.name, "sha256": checksum})
    return path

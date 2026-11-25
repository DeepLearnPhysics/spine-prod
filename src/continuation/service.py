"""Site-managed periodic continuation with shared registration and reconciliation."""

import contextlib
import hashlib
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .state import atomic_json, locked, manifest_path, read, validate_storage


def clean_environment():
    """Do not export the cron allocation's resource settings into training jobs."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "SBATCH_", "SRUN_"))
    }


def service_paths(account, site="nersc"):
    """Resolve shared registration and account state for the hosting site."""
    if site == "s3df":
        if not re.fullmatch(r"[A-Za-z0-9_:-]+", account or ""):
            raise ValueError("Automatic S3DF continuation requires a valid account")
        root = Path.home() / ".spine-prod" / "automation" / "s3df"
        return root, root / account / "registry.json"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", account or ""):
        raise ValueError("Automatic NERSC continuation requires a valid account")
    scratch = os.environ.get("SCRATCH")
    validate_storage(scratch, "nersc")
    if re.search(r"[\s$%]", scratch):
        raise ValueError("Unsupported character in Scratch automation root")
    root = Path(scratch).resolve() / ".spine-prod" / "automation"
    return root, root / account / "registry.json"


def schedule_command(site, option):
    """Resolve the site's schedule command, remotely when S3DF requires it."""
    if site == "s3df":
        from .s3df import command

        return command(["/usr/bin/crontab", option])
    return ["scrontab", option]


def read_schedule(site="nersc"):
    """Read the schedule; only the first-use diagnostic means an empty schedule."""
    result = subprocess.run(
        schedule_command(site, "-l"),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        # Slurm versions emit the first-use diagnostic on either stream.
        # Only that complete diagnostic means an empty schedule: do not hide
        # permission/network errors or overwrite a partially returned schedule.
        message = "\n".join(
            value.strip() for value in (result.stdout, result.stderr) if value.strip()
        )
        if re.fullmatch(
            r"(?:(?:s?crontab):\s*)?no s?crontab for \S+", message, re.IGNORECASE
        ):
            return ""
        raise RuntimeError(
            "Cannot read scrontab: "
            + (message or f"exit status {result.returncode}, no diagnostic output")
        )
    return result.stdout


def schedule_block(account, registry, basedir):
    """Build the uniquely marked cron entry for one account registry."""
    key = hashlib.sha256(account.encode()).hexdigest()[:16]
    command = shlex.join(
        [
            sys.executable,
            str(Path(basedir).resolve() / "scripts/continuation_service.py"),
            str(registry),
        ]
    )
    # scrontab expands $ itself, before the shell sees quotes.
    if any(char in command for char in ("\n", "\r", "$", "%")):
        raise ValueError("Unsupported character in automatic controller path")
    return (
        f"# BEGIN spine-prod continuation {key}\n"
        "#SCRON -q cron\n"
        "#SCRON -C cron\n"
        f"#SCRON -A {account}\n"
        "#SCRON -t 00:30:00\n"
        f"#SCRON --job-name=spc-controller-{key}\n"
        f"#SCRON --output={registry.parent}/controller.log\n"
        "#SCRON --open-mode=append\n"
        f"* * * * * {command}\n"
        f"# END spine-prod continuation {key}\n"
    )


def merge_schedule(existing, block):
    """Replace one managed block while preserving unrelated scrontab entries."""
    start, end = block.splitlines()[0], block.splitlines()[-1]
    # Only replace our marked block; malformed/duplicated markers require review.
    if existing.count(start) != existing.count(end) or existing.count(start) > 1:
        raise RuntimeError("Ambiguous managed scrontab block; inspect scrontab -l")
    if start in existing:
        left, rest = existing.split(start, 1)
        _, right = rest.split(end, 1)
        if right.startswith("\n"):
            right = right[1:]
        return left + block + right
    return existing + ("" if not existing or existing.endswith("\n") else "\n") + block


def register(manifest, account, basedir, site="nersc"):
    """Register and verify automatic hosting before enabling a pipeline.

    Parameters
    ----------
    manifest : str or pathlib.Path
        Existing pipeline manifest to register.
    account : str
        Allocation account used to group registered pipelines.
    basedir : str or pathlib.Path
        Repository containing the controller entry point.
    site : {"nersc", "s3df"}
        Site providing automatic timer hosting.

    Returns
    -------
    pathlib.Path
        Account registry used by the shared timer.

    Notes
    -----
    All accounts share the registration lock because the schedule is user-wide.
    The lock is always acquired before any pipeline manifest lock.
    """
    manifest = Path(manifest).resolve()
    root, registry = service_paths(account, site)
    if site == "s3df":
        from .s3df import preflight

        preflight(manifest, basedir)
    # All accounts share a lock because the schedule belongs to the user.
    with locked(root / "registration"):
        _register_locked(manifest, account, basedir, registry, site)

    return registry


def _register_locked(manifest, account, basedir, registry, site):
    """Caller holds the user-wide registration lock."""
    if site == "s3df":
        from .s3df import schedule_block as cron_block

        block = cron_block(account, registry, basedir)
    else:
        block = schedule_block(account, registry, basedir)
    data = read(registry) if registry.exists() else {"pipelines": {}}
    data["site"] = site
    entry = data["pipelines"].setdefault(str(manifest), {})
    entry.setdefault("enabled", False)
    atomic_json(registry, data)
    existing = read_schedule(site)
    desired = merge_schedule(existing, block)
    if desired != existing:
        backup = registry.parent / ("scrontab-before-" + str(time.time_ns()) + ".txt")
        backup.write_text(existing)
        # Detect edits made while preparing the replacement.
        if read_schedule(site) != existing:
            raise RuntimeError("scrontab changed during registration; retry submission")
        subprocess.run(
            schedule_command(site, "-"),
            input=desired,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
    if read_schedule(site) != desired:
        raise RuntimeError(
            "scrontab registration could not be verified; training not submitted"
        )
    entry.update(enabled=True, registered_at=time.time())
    data.update(schedule_state="active", cleanup_error=None)
    with locked(manifest):
        state = read(manifest)
        state["automation"] = {
            "backend": "s3df-crontab" if site == "s3df" else "nersc-scrontab",
            "registry": str(registry),
            "manifest": str(manifest),
            "registered_at": entry["registered_at"],
        }
        atomic_json(manifest, state)
    atomic_json(registry, data)


@contextlib.contextmanager
def reactivating(state):
    """Serialize reactivation and the user operation against idle retirement."""
    automation = state.get("automation", {})
    if automation.get("backend") not in ("nersc-scrontab", "s3df-crontab"):
        yield
        return
    registry = Path(automation["registry"])
    manifest = Path(automation["manifest"])
    with locked(registry.parent.parent / "registration"):
        _register_locked(
            manifest,
            registry.parent.name,
            state["basedir"],
            registry,
            "s3df" if automation["backend"] == "s3df-crontab" else "nersc",
        )
        yield


def remove_schedule(existing, account):
    """Remove only the selected account block, rejecting ambiguous markers."""
    key = hashlib.sha256(account.encode()).hexdigest()[:16]
    start = f"# BEGIN spine-prod continuation {key}"
    end = f"# END spine-prod continuation {key}"
    if existing.count(start) != existing.count(end) or existing.count(start) > 1:
        raise RuntimeError("Ambiguous managed scrontab block; inspect scrontab -l")
    if start not in existing:
        return existing
    left, rest = existing.split(start, 1)
    _, right = rest.split(end, 1)
    if right.startswith("\n"):
        right = right[1:]
    return left + right


def retire_if_idle(registry):
    """Remove the account timer only when every registered pipeline is inactive.

    Parameters
    ----------
    registry : str or pathlib.Path
        Account registry to recheck under the registration lock.

    Returns
    -------
    bool
        True if the managed block is absent; False if monitoring is still needed.

    Notes
    -----
    Missing or unreadable manifests are uncertainty, not evidence of completion.
    The registration lock prevents removal during a concurrent reactivation.
    """
    registry = Path(registry)
    with locked(registry.parent.parent / "registration"):
        data = read(registry)
        site = data.get("site", "nersc")
        # Re-read while holding the same lock as registrations/reactivations.
        # Missing/corrupt state is uncertainty, never evidence of completion.
        for name, entry in data["pipelines"].items():
            if entry.get("enabled") and needs_tick(Path(name)):
                return False
        from .audit import archive_audit

        for name, entry in data["pipelines"].items():
            if entry.get("enabled"):
                archive_audit(Path(name))
        existing = read_schedule(site)
        desired = remove_schedule(existing, registry.parent.name)
        if desired != existing:
            backup = registry.parent / (
                "scrontab-before-retirement-" + str(time.time_ns()) + ".txt"
            )
            backup.write_text(existing)
            if read_schedule(site) != existing:
                raise RuntimeError(
                    "scrontab changed during retirement; leaving controller registered"
                )
            # Removing our entry may terminate this cron invocation. Persist
            # intent first, so its status remains intelligible in that case.
            data.update(schedule_state="retiring", cleanup_error=None)
            atomic_json(registry, data)
            subprocess.run(
                schedule_command(site, "-"),
                input=desired,
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            )
            if read_schedule(site) != desired:
                raise RuntimeError("Could not verify automatic controller retirement")
        data.update(
            schedule_state="stopped", stopped_at=time.time(), cleanup_error=None
        )
        atomic_json(registry, data)
        return True


def needs_tick(manifest):
    """Check whether a pipeline still needs reconciliation or submission work."""
    state = read(manifest)
    if state["state"] != "active":
        return False
    for record in state["records"]:
        if record["status"] == "completed":
            continue
        if record["status"] == "failed":
            return False
        options = record["options"]
        if options.get("continuation") is not None:
            path = manifest_path(options["run_dir"], options.get("continuation_dir"))
            if path.exists() and read(path)["state"] in (
                "failed",
                "cancelled",
                "budget_exhausted",
            ):
                return False
        return True
    # One last tick marks the pipeline completed.
    return True


def tick_registry(registry):
    """Bounded scan; each pipeline retains its own interpreter and environment."""
    registry = Path(registry).resolve()
    root = registry.parent.parent
    with locked(root / "registration"):
        data = read(registry)
    # Scrontab does not overlap its own occurrences. This also serializes manual
    # invocations without holding the registration lock during scheduler calls.
    with locked(registry.parent / "tick"):
        deadline = time.monotonic() + 25 * 60
        entries = sorted(
            data["pipelines"].items(), key=lambda item: item[1].get("last_checked", 0)
        )
        for name, entry in entries:
            if time.monotonic() >= deadline:
                break
            if not entry.get("enabled"):
                continue
            manifest = Path(name)
            error = None
            checked = time.time()
            try:
                if needs_tick(manifest):
                    with open(manifest.parent / "controller.log", "a") as stream:
                        subprocess.run(
                            ["bash", str(manifest.parent / "advance.sh")],
                            env=clean_environment(),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=300,
                        )
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                error = str(exc)
                print(f"{manifest}: {error}", file=sys.stderr)
            with locked(root / "registration"):
                latest = read(registry)
                if name in latest["pipelines"]:
                    latest["pipelines"][name].update(
                        last_checked=checked, last_error=error
                    )
                    atomic_json(registry, latest)

        try:
            retire_if_idle(registry)
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            with locked(root / "registration"):
                latest = read(registry)
                latest["cleanup_error"] = str(exc)
                atomic_json(registry, latest)
            print(f"Controller retirement deferred: {exc}", file=sys.stderr)


def health(state):
    """Describe timer registration, retirement, errors and heartbeat freshness."""
    automation = state.get("automation")
    if not automation:
        return "manual"
    try:
        registry = read(automation["registry"])
        if registry.get("cleanup_error"):
            return "controller retirement error: " + registry["cleanup_error"]
        if registry.get("schedule_state") in ("stopped", "retiring"):
            return registry["schedule_state"] + "; no pipelines need monitoring"
        entry = registry["pipelines"][automation["manifest"]]
        if not entry.get("enabled"):
            return "disabled"
        if entry.get("last_error"):
            return "controller error: " + entry["last_error"]
        last = entry.get("last_checked")
        if last is None:
            age = int(time.time() - entry["registered_at"])
            return "registered; awaiting first scheduled check" + (
                " (stale)" if age > 600 else ""
            )
        age = int(time.time() - last)
        return f"last checked {age}s ago" + (" (stale)" if age > 600 else "")
    except (OSError, ValueError, KeyError):
        return "registry unavailable"

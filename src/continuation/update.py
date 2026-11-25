"""Update scheduling and unstarted cache locations without restarting a campaign."""

import copy
import time
import uuid
from pathlib import Path

from ..pipeline.fields import PROFILE_FIELDS
from ..pipeline.storage import validate_cache_storage, working_stages
from .policy import profile_policy, seconds
from .state import atomic_json, locked, read


def update_campaign(
    path, workspace, minimum=None, cache_dir=None, apply=False, context=None
):
    """Change future submission options under the controller lock.

    Existing allocations, checkpoint identities, accounting, and completed stages
    are untouched. Cache relocation is supported only before any cache producer
    starts and only when the old repositories contain no data. The original
    definition digest remains the identity of the launch document; operational
    changes are recorded separately and are not undone by a repeated start.
    """
    from ..submitter import Submitter

    path = Path(path).expanduser().resolve()
    workspace = Path(workspace).expanduser().resolve()
    if minimum is None and cache_dir is None:
        raise ValueError("Provide --min-time and/or --cache-dir")
    if minimum is not None:
        seconds(minimum)
    with locked(path):
        original = read(path)
        if original["state"] != "active":
            raise ValueError("Only active campaigns can be updated")
        state = copy.deepcopy(original)
        context = context or Submitter(basedir=Path(state["basedir"]))
        records = state["records"]
        # Verify the workspace against actual stage paths, not a guessed root.
        for record in records:
            run_dir = record["options"].get("run_dir")
            if run_dir and workspace not in Path(run_dir).resolve().parents:
                raise ValueError(
                    "Stage run directory is outside --workspace: " + run_dir
                )
        waiting = [record for record in records if record["status"] == "waiting"]
        if cache_dir is not None:
            producers = [r for r in records if r["options"].get("cache_repository")]
            if not producers:
                raise ValueError("No declared cache producers in this campaign")
            for record in producers:
                source = Path(record["options"]["cache_repository"]).expanduser()
                if record["status"] != "waiting":
                    raise ValueError(
                        "Cache relocation requires all producers to be waiting"
                    )
                if source.exists() and (not source.is_dir() or any(source.iterdir())):
                    raise ValueError(
                        "Existing cache data requires a separate migration: "
                        + str(source)
                    )
            mapped = working_stages(
                [r["options"] for r in waiting], workspace, cache_dir
            )
            for record, options in zip(waiting, mapped):
                if options.get("cache_repository"):
                    target = Path(options["cache_repository"])
                    if target.exists() and (
                        not target.is_dir() or any(target.iterdir())
                    ):
                        raise ValueError(
                            "Destination cache repository is not empty: " + str(target)
                        )
                record["options"] = options
        changed = []
        for before, record in zip(original["records"], records):
            options = record["options"]
            if (
                minimum is not None
                and record["status"] != "completed"
                and options.get("continuation") is not None
            ):
                options["continuation"]["min_time"] = minimum
            profile = context.config_mgr.get_profile(options.get("profile", "auto"))
            profile.update(
                {key: options[key] for key in PROFILE_FIELDS if key in options}
            )
            if options.get("continuation") is not None:
                profile_policy(context.profiles, profile, options["continuation"])
            # Completed filter caches remain in their original location.
            if cache_dir is not None and record["status"] == "waiting":
                validate_cache_storage(options, profile.get("site"))
            if options != before["options"]:
                changed.append(record["name"])
        print("Stages with changed future submission options: " + ", ".join(changed))
        if not changed:
            return None
        if not apply:
            print("Dry run: no state or scheduler jobs changed. Use --apply to save.")
            return None
        backup = path.with_name(
            path.stem + ".before-update-" + uuid.uuid4().hex + ".json"
        )
        atomic_json(backup, original)
        if read(backup) != original:
            raise ValueError(
                "Controller backup verification failed; update not applied"
            )
        state.setdefault("configuration_updates", []).append(
            dict(
                timestamp=time.time(),
                backup=str(backup),
                stages=changed,
                min_time=minimum,
                cache_dir=(
                    str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
                ),
            )
        )
        state["generation"] = state.get("generation", 0) + 1
        atomic_json(path, state)
        print("Updated future submissions. Backup: " + str(backup))
        print("Existing scheduler jobs and allocation request history are unchanged.")
        return backup

"""Separate writable working caches from scientific outputs and archives."""

import re
from pathlib import Path

from ..sites import validate_allocation
from .fields import PROFILE_FIELDS


def working_stages(stages, workspace, cache_dir):
    """Relocate declared cache roots and every reference to them consistently."""
    if cache_dir is None:
        return stages
    if not workspace:
        raise ValueError("--cache-dir requires a pipeline workspace")
    workspace = Path(workspace).expanduser().resolve()
    root = Path(cache_dir).expanduser().resolve()
    replacements = {}
    destinations = {}
    for stage in stages:
        for field in ("cache_repository", "cache_dir"):
            if not stage.get(field):
                continue
            original = str(stage[field])
            path = Path(original).expanduser().resolve()
            try:
                relative = path.relative_to(workspace)
            except ValueError as exc:
                raise ValueError(
                    "--cache-dir requires declared caches under workspace: " + original
                ) from exc
            if relative.parts and relative.parts[0] == "cache":
                relative = Path(*relative.parts[1:])
            target = str(root / relative)
            if target in destinations and destinations[target] != str(path):
                raise ValueError("Relocated cache paths collide: " + target)
            destinations[target] = str(path)
            replacements[original] = target

    pattern = (
        re.compile(
            r"(?<![A-Za-z0-9_./-])("
            + "|".join(
                re.escape(path) for path in sorted(replacements, key=len, reverse=True)
            )
            + r")(?=$|[/\s,'\"\]])"
        )
        if replacements
        else None
    )

    def replace(value):
        if isinstance(value, str):
            return (
                pattern.sub(lambda match: replacements[match[0]], value)
                if pattern
                else value
            )
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [replace(item) for item in value]
        return value

    return [replace(stage) for stage in stages]


def validate_cache_storage(stage, site):
    """NERSC cache publication must not lock DVS-mounted filesystems."""
    if site != "nersc":
        return
    for field in ("cache_repository", "cache_dir"):
        if stage.get(field):
            path = str(Path(stage[field]).expanduser().resolve())
            if not any(
                path.startswith(prefix)
                for prefix in ("/pscratch/", "/global/cscratch1/", "/global/cscratch2/")
            ):
                raise ValueError(
                    f"{field} requires NERSC Scratch for file locking: {path}. "
                    "Set --cache-dir to a Scratch directory; --continuation-dir only moves controller state."
                )


def archive_plans(stages, workspace, destination, context):
    """Describe CPU archive jobs which gate consumers of each cache publisher."""
    if destination is None:
        return {}
    if not workspace:
        raise ValueError("--cache-archive-dir requires a pipeline workspace")
    plans = {}
    for stage in stages:
        source = stage.get("cache_repository")
        if not source:
            continue
        profile = context.config_mgr.get_profile(stage.get("profile", "auto"))
        profile.update({key: stage[key] for key in PROFILE_FIELDS if key in stage})
        site = profile.get("site", "s3df")
        cpu = context.profiles["sites"][site]["roles"]["cpu"]
        resources = context.config_mgr.get_profile(cpu)
        resources["account"] = profile.get("account")
        validate_allocation(context.profiles, resources)
        # A stable repository name preserves train/validation separation; reject
        # collisions rather than archiving unrelated repositories together.
        target = str(Path(destination).expanduser().resolve() / Path(source).name)
        for previous in plans.values():
            if previous["destination"] == target and previous["source"] != source:
                raise ValueError("Cache archive names collide: " + target)
        if (
            Path(source).resolve() == Path(target)
            or Path(source).resolve() in Path(target).parents
            or Path(target) in Path(source).resolve().parents
        ):
            raise ValueError("Cache archive must be separate from working storage")
        name = stage["name"] + "__archive"
        if any(item["name"] == name for item in stages):
            raise ValueError("Reserved archive stage name: " + name)
        plans[stage["name"]] = dict(
            producer_run_dir=stage["run_dir"],
            source=source,
            destination=target,
            cache_stage=stage["cache_stage"],
            profile=cpu,
            account=profile.get("account"),
            run_dir=str(Path(workspace) / "archive" / stage["name"]),
        )
    if not plans:
        raise ValueError("--cache-archive-dir requires a cache-producing stage")
    return plans

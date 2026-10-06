"""Normalize the pre-1.4 schema for behavioral migration fingerprints.

Only descriptor syntax, ordering representations and relocatable paths are
normalized. Model/loss inputs, output options and constructor values are retained.
"""

import hashlib
import json

CAL = {
    "gain",
    "lifetime",
    "recombination",
    "response",
    "smearing",
    "field",
    "transparency",
    "gain_scale",
}


def norm(x, path=()):
    if isinstance(x, list):
        return [norm(v, path + ("[]",)) for v in x]
    if not isinstance(x, dict):
        return x
    if "__tag__" in x:
        return x
    x = dict(x)
    post_cal = (
        path
        and path[-1] in ("apply_calibrations", "calibration")
        and len(path) > 1
        and path[-2] == "post"
    )
    iscal = bool(path and path[-1] == "calibration" and not post_cal)
    rootcal = not path and (
        (set(x) - {"__meta__"} <= CAL and bool(set(x) & CAL))
        or (set(x) == {"stages"} and all(e["name"] in CAL for e in x["stages"]))
    )
    manager = (
        path in [("post",), ("ana",)]
        or (path and path[-1] == "augment")
        or iscal
        or rootcal
    )
    if manager or post_cal:
        settings = (
            {"gain_applied"}
            if iscal or rootcal
            else (
                {
                    "overwrite",
                    "prefix_output",
                    "log_dir",
                    "parent_path",
                    "run_mode",
                    "use_objects",
                    "train",
                    "iteration",
                }
                if path == ("ana",)
                else set()
            )
        )
        if post_cal:
            settings = set(x) - CAL - {"stages"}
        out = {k: norm(v, path + (k,)) for k, v in x.items() if k in settings}
        if "stages" in x:
            entries = x["stages"]
        else:
            pairs = [
                (k, v)
                for k, v in x.items()
                if k not in settings and k != "__meta__" and v is not None
            ]
            if not (path and path[-1] == "augment"):
                pairs.sort(
                    key=lambda a: (
                        a[1].get("priority") is None,
                        -a[1].get("priority", 0),
                    )
                )
            entries = [
                {
                    "name": k,
                    "provider": v.get("provider", v.get("name", k)),
                    "config": {
                        a: b
                        for a, b in v.items()
                        if a not in ("provider", "name", "priority")
                    },
                }
                for k, v in pairs
            ]
        result = []
        for ent in entries:
            name = ent["name"]
            provider = ent.get("provider", name)
            params = ent.get(
                "config",
                {k: v for k, v in ent.items() if k not in ("name", "provider")},
            )
            params = norm(params, path + (name,))
            result.append({"name": name, "provider": provider, "config": params})
        out["stages"] = result
        return out
    # Component selectors, except list instance names and metadata.
    if "name" in x and (
        not path or path[-1] not in ("[]", "__meta__", "update", "remove", "value")
    ):
        x["provider"] = x.pop("name")
    if "parser" in x:
        x["provider"] = x.pop("parser")
    if "provider" in x and path and path[-1] == "[]" and x.get("name") == x["provider"]:
        x.pop("provider")
    return {k: norm(v, path + (k,)) for k, v in x.items()}


def portable(value, root):
    """Serialize unresolved download tags without downloading external assets."""
    if isinstance(value, dict):
        return {key: portable(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, root) for item in value]
    if hasattr(value, "__dict__"):
        return {"__tag__": type(value).__name__, **portable(vars(value), root)}
    if isinstance(value, str):
        return value.replace(str(root), "$CONFIG_ROOT")
    return value


def stable(value):
    """Sort mapping keys without conflating numeric IDs with string labels."""
    if isinstance(value, dict):
        return {
            "mapping": [
                [type(key).__name__, key, stable(item)]
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: (type(pair[0]).__name__, str(pair[0])),
                )
            ]
        }
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def fingerprint(config, root):
    canonical = norm(portable(config, root))
    return hashlib.sha256(
        json.dumps(stable(canonical), sort_keys=True).encode()
    ).hexdigest()

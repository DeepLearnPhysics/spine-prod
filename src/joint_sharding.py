"""Entry-count surveys and disjoint, file-bounded joint inference shards."""

import json
import os
import tempfile
from pathlib import Path


def count_root_entries(path, validate_trees=False):
    """Read an entry count from ROOT metadata without loading events.

    Parameters
    ----------
    path : str or pathlib.Path
        Local LArCV ROOT input file.
    validate_trees : bool, default False
        Check counts across all LArCV trees. Otherwise stop after the first
        readable ``*_tree`` in sorted name order, used as the reference tree.

    Returns
    -------
    int
        Number of entries in the reference tree.

    Raises
    ------
    ValueError
        If no readable LArCV tree exists, or validation finds unequal counts.
    """
    try:
        import uproot
    except ImportError:
        try:
            import ROOT
        except ImportError as error:
            raise RuntimeError(
                "Balanced joint sharding needs uproot or PyROOT on the submission host"
            ) from error
        source = ROOT.TFile.Open(str(path), "READ")
        if not source or source.IsZombie():
            raise ValueError(f"Cannot open ROOT file: {path}")
        try:
            counts = []
            for key in sorted(source.GetListOfKeys(), key=lambda key: key.GetName()):
                if not key.GetName().endswith("_tree"):
                    continue
                tree = source.Get(key.GetName())
                if tree.InheritsFrom("TTree"):
                    counts.append(int(tree.GetEntries()))
                    if not validate_trees:
                        break
        finally:
            source.Close()
    else:
        with uproot.open(path) as source:
            counts = []
            for key in sorted(source.keys(cycle=False)):
                if not key.endswith("_tree"):
                    continue
                tree = source[key]
                if hasattr(tree, "num_entries"):
                    counts.append(int(tree.num_entries))
                    if not validate_trees:
                        break
    if not counts or len(set(counts)) != 1:
        raise ValueError(f"Expected equally sized LArCV *_tree products in {path}")
    return counts[0]


def survey_sources(sources, cache_path, validate_trees=False):
    """Cache counts by path, size and modification time, recording validation.

    Validated counts can serve reference-only surveys. A request to validate
    all trees must resurvey counts cached without full validation.
    """
    cache_path = Path(cache_path)
    try:
        cache = json.loads(cache_path.read_text())
    except (FileNotFoundError, ValueError):
        cache = {}
    counts = {}
    for target, paths in sources.items():
        counts[target] = []
        for path in paths:
            absolute = str(Path(path).resolve())
            stat = Path(path).stat()
            signature = [stat.st_size, stat.st_mtime_ns]
            record = cache.get(absolute, {})
            count = record.get("entries")
            if (
                record.get("signature") != signature
                or type(count) is not int
                or count < 0
                or (validate_trees and not record.get("validated_trees", False))
            ):
                count = count_root_entries(path, validate_trees=validate_trees)
                cache[absolute] = {
                    "signature": signature,
                    "entries": count,
                    "validated_trees": validate_trees,
                }
            counts[target].append(count)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement also keeps concurrent submissions from reading partial JSON.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=cache_path.parent, delete=False
    ) as stream:
        json.dump(cache, stream, indent=2)
        temporary = stream.name
    os.replace(temporary, cache_path)
    return counts


def plan_shards(sources, counts, files_per_task):
    """Keep primary file groups intact and slice consecutive secondary entries.

    Only the final primary group can be cut short, when secondary is exhausted.
    Empty primary files do not generate empty jobs.
    """
    total = min(sum(counts["primary"]), sum(counts["secondary"]))
    if total < 1:
        raise ValueError("Balanced joint sources have no usable entry pairs")
    secondary_offsets = [0]
    for count in counts["secondary"]:
        secondary_offsets.append(secondary_offsets[-1] + count)
    shards = []
    cursor = 0
    secondary_index = 0
    for first in range(0, len(sources["primary"]), files_per_task):
        stop = first + files_per_task
        size = min(sum(counts["primary"][first:stop]), total - cursor)
        if size <= 0:
            if cursor == total:
                break
            continue
        while secondary_offsets[secondary_index + 1] <= cursor:
            secondary_index += 1
        last = secondary_index
        end = cursor + size
        while secondary_offsets[last + 1] < end:
            last += 1
        shards.append(
            {
                "start": cursor,
                "entries": size,
                "primary": {"files": sources["primary"][first:stop], "n_skip": 0},
                "secondary": {
                    "files": sources["secondary"][secondary_index : last + 1],
                    "n_skip": cursor - secondary_offsets[secondary_index],
                },
            }
        )
        cursor = end
    return shards

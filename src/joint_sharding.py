"""Entry-count surveys and disjoint, file-bounded joint inference shards."""

import json
import os
import tempfile
from pathlib import Path


def count_root_entries(path):
    """Read LArCV tree metadata without loading events (uproot or PyROOT)."""
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
            counts = [
                int(source.Get(key.GetName()).GetEntries())
                for key in source.GetListOfKeys()
                if key.GetName().endswith("_tree")
                and source.Get(key.GetName()).InheritsFrom("TTree")
            ]
        finally:
            source.Close()
    else:
        with uproot.open(path) as source:
            counts = [
                int(source[key].num_entries)
                for key in source.keys(cycle=False)
                if key.endswith("_tree") and hasattr(source[key], "num_entries")
            ]
    if not counts or len(set(counts)) != 1:
        raise ValueError(f"Expected equally sized LArCV *_tree products in {path}")
    return counts[0]


def survey_sources(sources, cache_path):
    """Cache counts by absolute path, file size and nanosecond modification time."""
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
            ):
                count = count_root_entries(path)
                cache[absolute] = {"signature": signature, "entries": count}
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

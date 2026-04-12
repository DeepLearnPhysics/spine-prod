#!/usr/bin/env python3
"""Preload SPINE !download assets into the shared cache."""

import argparse
import fcntl
import hashlib
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class DownloadSpec:
    """A file referenced by a SPINE !download YAML tag."""

    url: str
    expected_hash: Optional[str]
    source: Path


def get_project_root() -> Path:
    """Return the spine-prod repository root."""
    return Path(__file__).resolve().parents[1]


def resolve_config_path(config: str, project_root: Path) -> Path:
    """Resolve a config path from cwd or the repository config directory."""
    path = Path(config).expanduser()
    if path.exists():
        return path.resolve()

    repo_config_path = project_root / "config" / config
    if repo_config_path.exists():
        return repo_config_path.resolve()

    raise FileNotFoundError(f"Config not found: {config}")


def resolve_include_path(include: str, root_dir: Path, project_root: Path) -> Path:
    """Resolve an included config path."""
    include_path = Path(os.path.expandvars(include)).expanduser()
    if include_path.is_absolute() and include_path.exists():
        return include_path.resolve()

    candidates = [root_dir / include_path]

    config_path = os.environ.get("SPINE_CONFIG_PATH")
    if config_path:
        candidates.extend(Path(p) / include_path for p in config_path.split(":") if p)

    candidates.append(project_root / "config" / include_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Included config not found: {include}")


def strip_inline_comment(value: str) -> str:
    """Remove simple inline comments from an unquoted YAML scalar."""
    if "#" not in value:
        return value
    return value.split("#", 1)[0]


def clean_scalar(value: str) -> str:
    """Clean a simple YAML scalar."""
    value = strip_inline_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def line_indent(line: str) -> int:
    """Return leading-space indentation for a line."""
    return len(line) - len(line.lstrip(" "))


def parse_includes(lines: List[str]) -> List[str]:
    """Parse top-level SPINE include directives from YAML text."""
    includes = []
    for idx, line in enumerate(lines):
        if not line.startswith("include:"):
            continue

        value = clean_scalar(line.split(":", 1)[1])
        if value:
            includes.append(value)
            continue

        for next_line in lines[idx + 1 :]:
            stripped = next_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line_indent(next_line) == 0:
                break
            if stripped.startswith("- "):
                includes.append(clean_scalar(stripped[2:]))
        break

    return includes


def parse_downloads(lines: List[str], source: Path) -> List[DownloadSpec]:
    """Parse !download tags from YAML text."""
    downloads = []
    for idx, line in enumerate(lines):
        if "!download" not in line:
            continue

        indent = line_indent(line)
        after_tag = line.split("!download", 1)[1].strip()
        if after_tag:
            downloads.append(DownloadSpec(clean_scalar(after_tag), None, source))
            continue

        url = None
        expected_hash = None
        for next_line in lines[idx + 1 :]:
            stripped = next_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line_indent(next_line) <= indent:
                break
            if stripped.startswith("url:"):
                url = clean_scalar(stripped.split(":", 1)[1])
            elif stripped.startswith("hash:"):
                expected_hash = clean_scalar(stripped.split(":", 1)[1])

        if not url:
            raise ValueError(f"!download in {source} is missing a url")
        downloads.append(DownloadSpec(url, expected_hash, source))

    return downloads


def get_cache_dir(project_root: Path) -> Path:
    """Return the SPINE download cache directory."""
    if os.environ.get("SPINE_CACHE_DIR"):
        return Path(os.environ["SPINE_CACHE_DIR"]).expanduser()
    if os.environ.get("SPINE_PROD_BASEDIR"):
        return Path(os.environ["SPINE_PROD_BASEDIR"]) / ".cache" / "weights"
    if os.environ.get("SPINE_BASEDIR"):
        return Path(os.environ["SPINE_BASEDIR"]) / ".cache" / "weights"
    return project_root / ".cache" / "weights"


def url_to_filename(url: str) -> str:
    """Convert a URL to SPINE's cache filename convention."""
    suffix = Path(urllib.parse.urlparse(url).path).suffix
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{url_hash}{suffix}"


def compute_file_hash(path: Path) -> str:
    """Compute the SHA256 digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_cached_file(path: Path, expected_hash: Optional[str]) -> bool:
    """Check that a cached file exists and, if requested, matches its hash."""
    if not path.exists():
        return False
    if expected_hash is None:
        print(f"Using cached file: {path}")
        return True

    actual_hash = compute_file_hash(path)
    if actual_hash == expected_hash:
        print(f"Using cached file: {path}")
        return True

    print(f"Cached file hash mismatch, will re-download: {path}")
    return False


def download_file(url: str, output_path: Path, expected_hash: Optional[str]) -> None:
    """Download a URL to a path and validate its hash."""
    print(f"Downloading: {url}")
    print(f"Destination: {output_path}")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            sys.stdout.write(f"\rProgress: {percent}% ")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()
    except HTTPError as exc:
        raise HTTPError(
            url, exc.code, f"HTTP Error {exc.code}: {exc.reason}", exc.hdrs, exc.fp
        ) from exc
    except URLError as exc:
        raise URLError(f"Failed to download {url}: {exc.reason}") from exc

    if expected_hash is None:
        return

    actual_hash = compute_file_hash(output_path)
    if actual_hash != expected_hash:
        output_path.unlink()
        raise ValueError(
            "Downloaded file hash mismatch!\n"
            f"Expected: {expected_hash}\n"
            f"Got:      {actual_hash}\n"
            "File has been removed. Please try again."
        )
    print(f"Hash validated: {actual_hash[:16]}...")


def download_to_cache(
    spec: DownloadSpec,
    cache_dir: Path,
    max_wait_seconds: int = 3600,
) -> Path:
    """Download a spec into the cache if it is not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / url_to_filename(spec.url)
    lock_path = cache_dir / f".{output_path.name}.lock"

    if validate_cached_file(output_path, spec.expected_hash):
        return output_path

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.time() - start_time > max_wait_seconds:
                    raise TimeoutError(
                        f"Timeout waiting for download lock: {lock_path}"
                    ) from exc
                time.sleep(1)

        try:
            if validate_cached_file(output_path, spec.expected_hash):
                return output_path

            temp_fd, temp_path = tempfile.mkstemp(
                dir=cache_dir, prefix=f".{output_path.name}.", suffix=".tmp"
            )
            temp_path = Path(temp_path)
            os.close(temp_fd)
            try:
                download_file(spec.url, temp_path, spec.expected_hash)
                temp_path.replace(output_path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    try:
        lock_path.unlink()
    except OSError:
        pass

    print(f"Download complete: {output_path}")
    return output_path


def collect_downloads(
    config_path: Path,
    project_root: Path,
    seen: Optional[Set[Path]] = None,
) -> List[DownloadSpec]:
    """Collect all !download specs from a config and its includes."""
    if seen is None:
        seen = set()

    config_path = config_path.resolve()
    if config_path in seen:
        return []
    seen.add(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    downloads = parse_downloads(lines, config_path)
    for include in parse_includes(lines):
        include_path = resolve_include_path(include, config_path.parent, project_root)
        downloads.extend(collect_downloads(include_path, project_root, seen))

    return downloads


def deduplicate_downloads(downloads: List[DownloadSpec]) -> List[DownloadSpec]:
    """Deduplicate downloads by URL and expected hash while preserving order."""
    deduped = []
    seen = set()
    for spec in downloads:
        key = (spec.url, spec.expected_hash)
        if key not in seen:
            seen.add(key)
            deduped.append(spec)
    return deduped


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Preload files referenced by SPINE !download tags. Run this on a "
            "login node before submitting jobs on systems where worker nodes "
            "do not have outbound network access."
        )
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="SPINE config files to scan, e.g. infer/2x2/full_chain_240819.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        help=(
            "Override SPINE_CACHE_DIR. Defaults to SPINE's normal cache "
            "resolution, typically $SPINE_PROD_BASEDIR/.cache/weights."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Preload all downloads needed by the requested configs."""
    args = parse_args()
    project_root = get_project_root()

    os.environ.setdefault("SPINE_PROD_BASEDIR", str(project_root))
    if args.cache_dir:
        os.environ["SPINE_CACHE_DIR"] = str(Path(args.cache_dir).expanduser())

    cache_dir = get_cache_dir(project_root)
    print(f"Download cache: {cache_dir}")

    try:
        downloads = []
        for config in args.configs:
            config_path = resolve_config_path(config, project_root)
            print(f"\nScanning config: {config_path}")
            downloads.extend(collect_downloads(config_path, project_root))

        downloads = deduplicate_downloads(downloads)
        if not downloads:
            print("\nNo !download assets found.")
            return 0

        print(f"\nFound {len(downloads)} download(s).")
        for spec in downloads:
            print(f"\nSource: {spec.source}")
            download_to_cache(spec, cache_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nPreload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

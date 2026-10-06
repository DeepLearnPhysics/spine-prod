#!/usr/bin/env python3
"""Reject runtimes older than the maintained production configuration schema."""

import re
import subprocess
import sys

MINIMUM_VERSION = (1, 4, 0)


def validate_version(version):
    """Require the released 1.4 configuration API, including for source installs."""
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(.*)", version)
    required = ".".join(map(str, MINIMUM_VERSION))
    if match is None:
        raise RuntimeError(
            f"Cannot verify SPINE version {version!r}; require >= {required}."
        )
    release = tuple(int(match.group(i)) for i in range(1, 4))
    prerelease = re.search(r"(?:a|b|rc|dev)\d", match.group(4)) is not None
    if release < MINIMUM_VERSION or (release == MINIMUM_VERSION and prerelease):
        raise RuntimeError(
            f"Maintained spine-prod configurations require SPINE >= {required}; "
            f"this runtime provides {version}. Update the active environment or container."
        )


def main():
    """Check inside the selected runtime, after any environment setup."""
    try:
        # Probe the actual executable (which may use a different Python or a
        # source checkout), not whichever package this helper can import.
        command = sys.argv[1:] or ["spine"]
        result = subprocess.run(
            [*command, "--version"], capture_output=True, text=True, check=True
        )
        match = re.search(r"^SPINE\s+(\S+)", result.stdout, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Unrecognized SPINE version output: {result.stdout!r}")
        validate_version(match.group(1))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"SPINE runtime check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

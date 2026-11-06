"""SPINE Production submission system.

Keep scheduler-side runtime helpers usable with the Python standard library;
submission-only YAML/template dependencies are loaded when Submitter is used.
"""

import sys

if sys.version_info < (3, 9):
    raise ImportError(
        "spine-prod requires Python 3.9 or newer; found {}.{}.{}. "
        "On NERSC, run `module load python`, verify `python3 --version`, "
        "and retry using that environment.".format(*sys.version_info[:3])
    )

__all__ = ["Submitter"]


def __getattr__(name):
    if name == "Submitter":
        from .submitter import Submitter

        return Submitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

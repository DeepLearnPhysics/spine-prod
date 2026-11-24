"""SPINE Production submission system.

Keep scheduler-side runtime helpers usable with the Python standard library;
submission-only YAML/template dependencies are loaded when Submitter is used.
"""

import sys

if sys.version_info < (3, 10):
    raise ImportError(
        "spine-prod requires Python 3.10 or newer; found {}.{}.{}. "
        "Activate a Python 3.10+ environment on the submission host, install "
        "requirements.txt there, and retry with that environment's interpreter. "
        "For scripts/smoke_multinode.sh, select it with SPINE_PROD_PYTHON.".format(
            *sys.version_info[:3]
        )
    )

__all__ = ["Submitter"]


def __getattr__(name):
    """Load submission dependencies only when the public Submitter is requested."""
    if name == "Submitter":
        from .submitter import Submitter

        return Submitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

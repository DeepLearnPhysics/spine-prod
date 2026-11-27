"""SPINE Production submission system.

Keep scheduler-side runtime helpers usable with the Python standard library;
submission-only YAML/template dependencies are loaded when Submitter is used.
"""

import sys


def require_python(minimum=(3, 10)):
    """Check host Python; only the standard-library worker accepts Python 3.9."""
    if sys.version_info < minimum:
        raise ImportError(
            "spine-prod requires Python {}.{} or newer; found {}.{}.{}. "
            "Activate a supported environment on the submission host and install "
            "requirements.txt there. For test helpers, select it "
            "with SPINE_PROD_PYTHON.".format(*minimum, *sys.version_info[:3])
        )


__all__ = ["Submitter"]


def __getattr__(name):
    """Load submission dependencies only when the public Submitter is requested."""
    if name == "Submitter":
        require_python()
        from .submitter import Submitter

        return Submitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""SPINE Production submission system.

Keep scheduler-side runtime helpers usable with the Python standard library;
submission-only YAML/template dependencies are loaded when Submitter is used.
"""

__all__ = ["Submitter"]


def __getattr__(name):
    if name == "Submitter":
        from .submitter import Submitter

        return Submitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

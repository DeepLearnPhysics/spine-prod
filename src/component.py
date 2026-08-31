"""Shared contract for services composed by the public submitter façade."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .submitter import Submitter


class SubmissionComponent:
    """Base class for a service that uses shared submission state.

    Components own operational behavior while the façade owns shared managers,
    paths, and scheduler clients. Attribute forwarding keeps those dependencies
    explicit at construction without duplicating mutable state.
    """

    def __init__(self, context: "Submitter"):
        """Attach the component to its shared submission context."""
        self.context = context

    @property
    def basedir(self):
        """Return the repository base directory owned by the façade."""
        return self.context.basedir

    @property
    def jobs_dir(self):
        """Return the automatic job directory owned by the façade."""
        return self.context.jobs_dir

    @property
    def config_mgr(self):
        """Return the shared configuration manager."""
        return self.context.config_mgr

    @property
    def file_handler(self):
        """Return the shared input-file handler."""
        return self.context.file_handler

    @property
    def batch_client(self):
        """Return the façade's default scheduler client."""
        return self.context.batch_client

    @property
    def profiles(self):
        """Return scheduler and detector profiles."""
        return self.context.profiles

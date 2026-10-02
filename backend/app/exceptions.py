"""Application-level exception hierarchy.

Agents and services raise these instead of leaking library-specific errors,
so the API layer can turn them into clear, user-facing messages.
"""


class ResearchPilotError(Exception):
    """Base class for all expected application errors."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(ResearchPilotError):
    """A required setting (e.g. an API key) is missing or invalid."""

    status_code = 503


class DatabaseError(ResearchPilotError):
    """A persistence operation failed."""

    status_code = 500


class NotFoundError(ResearchPilotError):
    """A requested record does not exist."""

    status_code = 404

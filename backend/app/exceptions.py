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


class InvalidDocumentError(ResearchPilotError):
    """An uploaded file is unsupported, empty, too large or unreadable."""

    status_code = 422


class ExternalServiceError(ResearchPilotError):
    """An external API (Gemini, Tavily) failed or returned an unusable response."""

    status_code = 502

    def __init__(self, service: str, message: str) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service


class RateLimitError(ExternalServiceError):
    """An external API rejected the request due to rate limits or exhausted quota."""

    status_code = 429


class MalformedResponseError(ExternalServiceError):
    """The LLM returned output that could not be parsed into the expected schema."""


class WorkflowError(ResearchPilotError):
    """A research run cannot be started/resumed in its current state."""

    status_code = 409

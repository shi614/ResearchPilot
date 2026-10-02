"""Classify Gemini API failures, including ones wrapped by LangChain.

langchain-google-genai wraps `google.genai` errors in `GoogleGenerativeAIError`,
so the HTTP status is recovered by walking the exception's cause chain.
"""

from __future__ import annotations

import re

import httpx
from google.genai import errors as genai_errors

from app.exceptions import ExternalServiceError, RateLimitError

_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _cause_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def gemini_status_code(exc: BaseException) -> int | None:
    for error in _cause_chain(exc):
        if isinstance(error, genai_errors.APIError):
            return error.code
    return None


def is_network_error(exc: BaseException) -> bool:
    return any(isinstance(error, httpx.TransportError) for error in _cause_chain(exc))


def is_retryable_gemini_error(exc: Exception) -> bool:
    return gemini_status_code(exc) in RETRYABLE_STATUS_CODES or is_network_error(exc)


def gemini_retry_after(exc: Exception) -> float | None:
    """Seconds to wait, if the API suggested a delay (RetryInfo.retryDelay)."""
    for error in _cause_chain(exc):
        match = _RETRY_DELAY_PATTERN.search(str(error))
        if match:
            return float(match.group(1))
    return None


def is_daily_quota_error(exc: BaseException) -> bool:
    """True when a 429 refers to a per-day quota (retrying minutes later won't help)."""
    return gemini_status_code(exc) == 429 and any("PerDay" in str(e) for e in _cause_chain(exc))


def to_service_error(exc: Exception, operation: str) -> ExternalServiceError:
    """Map any Gemini failure to an application error with a helpful message."""
    code = gemini_status_code(exc)
    if code == 429 and is_daily_quota_error(exc):
        return RateLimitError(
            "Gemini",
            f"{operation} failed: the free-tier DAILY quota for this model is exhausted. "
            "The run is saved; resume it after the quota resets.",
        )
    if code == 429:
        return RateLimitError(
            "Gemini",
            f"{operation} hit the free-tier rate limit or daily quota. "
            "Wait a minute (or until the daily quota resets) and try again.",
        )
    if code in (401, 403) or (code == 400 and "api key" in str(exc).lower()):
        return ExternalServiceError("Gemini", f"{operation} failed: the API key was rejected.")
    if code == 404:
        return ExternalServiceError(
            "Gemini", f"{operation} failed: the configured model is not available for this key."
        )
    if code is not None and code >= 500:
        return ExternalServiceError("Gemini", f"{operation} failed: Gemini service error ({code}).")
    if is_network_error(exc):
        return ExternalServiceError("Gemini", f"{operation} failed: could not reach the Gemini API.")
    return ExternalServiceError("Gemini", f"{operation} failed ({type(exc).__name__}).")

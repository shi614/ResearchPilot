from __future__ import annotations

import httpx
import pytest
from google.genai import errors as genai_errors
from langchain_google_genai._common import GoogleGenerativeAIError

from app.exceptions import ExternalServiceError, RateLimitError
from app.utils.gemini_errors import (
    gemini_retry_after,
    gemini_status_code,
    is_retryable_gemini_error,
    to_service_error,
)
from app.utils.retry import call_with_retry


class Flaky:
    def __init__(self, failures: list[Exception], result: str = "ok") -> None:
        self.failures = list(failures)
        self.result = result
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


def gemini_429(retry_delay: str | None = None) -> genai_errors.ClientError:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    body = {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    if retry_delay:
        body["error"]["details"] = details
    return genai_errors.ClientError(429, body)


def wrapped(exc: Exception) -> GoogleGenerativeAIError:
    """Mimic langchain-google-genai, which wraps google.genai errors."""
    try:
        raise GoogleGenerativeAIError("Error embedding content") from exc
    except GoogleGenerativeAIError as wrapper:
        return wrapper


def test_retry_succeeds_after_transient_failures() -> None:
    delays: list[float] = []
    fn = Flaky([ValueError("boom"), ValueError("boom")])
    result = call_with_retry(fn, should_retry=lambda _e: True, base_delay=1, sleep=delays.append)
    assert result == "ok" and fn.calls == 3
    assert delays == [1, 2]  # exponential backoff


def test_retry_honours_server_delay_but_caps_it() -> None:
    delays: list[float] = []
    fn = Flaky([ValueError("x"), ValueError("y")])
    call_with_retry(
        fn,
        should_retry=lambda _e: True,
        retry_after=lambda e: 120.0 if str(e) == "y" else 7.0,
        max_delay=60,
        sleep=delays.append,
    )
    assert delays == [7.0, 60]


def test_retry_gives_up_and_reraises_original_error() -> None:
    fn = Flaky([ValueError("1"), ValueError("2"), ValueError("3")])
    with pytest.raises(ValueError, match="3"):
        call_with_retry(fn, should_retry=lambda _e: True, max_attempts=3, sleep=lambda _s: None)


def test_non_retryable_errors_fail_immediately() -> None:
    fn = Flaky([KeyError("fatal")])
    with pytest.raises(KeyError):
        call_with_retry(fn, should_retry=lambda e: isinstance(e, ValueError), sleep=lambda _s: None)
    assert fn.calls == 1


def test_status_code_and_retry_delay_are_found_through_langchain_wrapper() -> None:
    error = wrapped(gemini_429(retry_delay="37s"))
    assert gemini_status_code(error) == 429
    assert is_retryable_gemini_error(error)
    assert gemini_retry_after(error) == 37.0


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad"}}), False),
        (genai_errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}}), True),
        (httpx.ConnectError("offline"), True),
        (ValueError("parse error"), False),
    ],
)
def test_retryable_classification(error: Exception, retryable: bool) -> None:
    assert is_retryable_gemini_error(wrapped(error)) is retryable


def test_service_error_mapping() -> None:
    assert isinstance(to_service_error(wrapped(gemini_429()), "Embedding"), RateLimitError)
    server = to_service_error(
        genai_errors.ServerError(500, {"error": {"code": 500, "message": "x"}}), "Embedding"
    )
    assert type(server) is ExternalServiceError and "service error" in server.message
    assert "could not reach" in to_service_error(httpx.ConnectError("x"), "Embedding").message

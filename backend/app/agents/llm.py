"""Gemini gateway shared by all agents.

Free-tier protection lives here, in one place:
- every request passes a shared requests-per-minute throttle (GEMINI_MAX_RPM);
- every request (including retries) is counted, so runs report real usage;
- the SDK's hidden retries are disabled; we retry at most once, and only for
  transient errors or a short per-minute 429. Daily-quota 429s fail fast.
- structured outputs are validated with Pydantic; one repair attempt is made
  if the model returns malformed JSON.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app.config import Settings
from app.exceptions import ConfigurationError, MalformedResponseError
from app.utils.gemini_errors import (
    gemini_retry_after,
    gemini_status_code,
    is_daily_quota_error,
    is_retryable_gemini_error,
    to_service_error,
)
from app.utils.retry import call_with_retry

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

MAX_PER_MINUTE_RETRY_WAIT = 60.0


class LLMGateway(Protocol):
    """What agents need from an LLM. Tests provide a scripted implementation."""

    @property
    def call_count(self) -> int: ...

    def structured(self, schema: type[M], system: str, user: str, *, operation: str) -> M: ...

    def invoke_with_tools(
        self, messages: list[BaseMessage], tools: list[BaseTool], *, operation: str
    ) -> AIMessage: ...


def should_retry_llm_error(exc: Exception) -> bool:
    """Retry transient failures and short per-minute rate limits only."""
    if gemini_status_code(exc) == 429:
        if is_daily_quota_error(exc):
            return False
        delay = gemini_retry_after(exc)
        return delay is not None and delay <= MAX_PER_MINUTE_RETRY_WAIT
    return is_retryable_gemini_error(exc)


class GeminiLLM:
    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        max_rpm: int,
        max_attempts: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._chat = chat_model
        self._limiter = InMemoryRateLimiter(
            requests_per_second=max_rpm / 60, check_every_n_seconds=0.1, max_bucket_size=1
        )
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        return self._calls

    def _request(self, fn: Callable[[], Any], operation: str) -> Any:
        def counted() -> Any:
            self._limiter.acquire(blocking=True)
            with self._lock:
                self._calls += 1
            return fn()

        try:
            return call_with_retry(
                counted,
                should_retry=should_retry_llm_error,
                retry_after=gemini_retry_after,
                max_attempts=self._max_attempts,
                sleep=self._sleep,
                description=f"Gemini {operation}",
            )
        except Exception as exc:
            raise to_service_error(exc, operation) from exc

    def structured(self, schema: type[M], system: str, user: str, *, operation: str) -> M:
        runnable = self._chat.with_structured_output(schema, method="json_schema", include_raw=True)
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        for attempt in (1, 2):
            raw = self._request(lambda: runnable.invoke(messages), operation)
            parsed = raw.get("parsed") if isinstance(raw, dict) else None
            if isinstance(parsed, schema):
                return parsed
            error = raw.get("parsing_error") if isinstance(raw, dict) else None
            logger.warning("Malformed %s response (attempt %d): %s", operation, attempt, error)
            messages = [
                *messages[:2],
                HumanMessage(
                    "Your previous answer did not match the required JSON schema "
                    f"({str(error)[:300]}). Answer again with valid JSON only."
                ),
            ]
        raise MalformedResponseError("Gemini", f"{operation} returned malformed output twice.")

    def invoke_with_tools(
        self, messages: list[BaseMessage], tools: list[BaseTool], *, operation: str
    ) -> AIMessage:
        bound = self._chat.bind_tools(tools)
        result = self._request(lambda: bound.invoke(messages), operation)
        if not isinstance(result, AIMessage):
            raise MalformedResponseError("Gemini", f"{operation} returned an unexpected message type.")
        return result


def create_llm(settings: Settings) -> GeminiLLM:
    if settings.gemini_api_key is None:
        raise ConfigurationError("GEMINI_API_KEY is not set; the research agents are unavailable.")
    chat = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.gemini_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,  # 1 = no SDK retries (0 means "SDK default of 5"); we retry ourselves
    )
    return GeminiLLM(chat, max_rpm=settings.gemini_max_rpm)

"""Gemini gateway shared by all agents.

Free-tier protection lives here, in one place:
- every request passes a shared requests-per-minute throttle (GEMINI_MAX_RPM);
- every request (including retries) is counted, so runs report real usage;
- the SDK's hidden retries are disabled; we retry at most once, and only for
  transient errors or a short per-minute 429. Daily-quota 429s fail fast.
- structured outputs are validated with Pydantic; one repair attempt is made
  if the model returns malformed JSON.
- optionally, a pool of Gemini keys (GEMINI_API_KEYS) is rotated through: a
  daily-quota 429 on the current key rotates to the next one and retries,
  rather than failing the run. With a single key, behaviour is unchanged.
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
        chat_model: BaseChatModel | list[BaseChatModel],
        *,
        max_rpm: int,
        max_attempts: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._chats: list[BaseChatModel] = (
            list(chat_model) if isinstance(chat_model, list) else [chat_model]
        )
        if not self._chats:
            raise ValueError("GeminiLLM requires at least one chat model")
        self._chat_index = 0
        self._chat = self._chats[0]
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

    @property
    def active_key_index(self) -> int:
        """Index (0-based) of the Gemini key currently in use, for diagnostics."""
        return self._chat_index

    def _rotate_key(self) -> bool:
        """Advance to the next configured key. Returns False if none remain."""
        if self._chat_index + 1 >= len(self._chats):
            return False
        self._chat_index += 1
        self._chat = self._chats[self._chat_index]
        logger.warning(
            "Gemini key %d/%d exhausted its daily quota; rotating to key %d/%d",
            self._chat_index,
            len(self._chats),
            self._chat_index + 1,
            len(self._chats),
        )
        return True

    def _request(self, make_call: Callable[[], Any], operation: str) -> Any:
        """Run `make_call` (which must read `self._chat` fresh, not a captured copy).

        Rotates to the next configured key and retries the whole call when
        the current key's daily quota is exhausted and another key remains;
        otherwise behaves exactly as before (single retry on transient
        errors, fail-fast on a daily-quota 429).
        """

        def counted() -> Any:
            self._limiter.acquire(blocking=True)
            with self._lock:
                self._calls += 1
            return make_call()

        while True:
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
                if is_daily_quota_error(exc) and self._rotate_key():
                    continue
                raise to_service_error(exc, operation) from exc

    def structured(self, schema: type[M], system: str, user: str, *, operation: str) -> M:
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        for attempt in (1, 2):
            def call_once(messages: list[BaseMessage] = messages, schema: type[M] = schema) -> Any:
                runnable = self._chat.with_structured_output(
                    schema, method="json_schema", include_raw=True
                )
                return runnable.invoke(messages)

            raw = self._request(call_once, operation)
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
        def call_once() -> Any:
            bound = self._chat.bind_tools(tools)
            return bound.invoke(messages)

        result = self._request(call_once, operation)
        if not isinstance(result, AIMessage):
            raise MalformedResponseError("Gemini", f"{operation} returned an unexpected message type.")
        return result


def create_llm(settings: Settings) -> GeminiLLM:
    keys = settings.gemini_keys
    if not keys:
        raise ConfigurationError("GEMINI_API_KEY is not set; the research agents are unavailable.")
    chats = [
        ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=key,
            temperature=settings.gemini_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,  # 1 = no SDK retries (0 means "SDK default of 5"); we retry ourselves
        )
        for key in keys
    ]
    return GeminiLLM(chats, max_rpm=settings.gemini_max_rpm)

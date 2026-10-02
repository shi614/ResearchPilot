"""Retry with exponential backoff, honouring server-suggested retry delays."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    retry_after: Callable[[Exception], float | None] = lambda _exc: None,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    description: str = "operation",
) -> T:
    """Call `fn`, retrying transient failures.

    The last exception is re-raised unchanged when attempts run out or the
    error is not retryable, so callers keep full control of error mapping.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_attempts or not should_retry(exc):
                raise
            suggested = retry_after(exc)
            delay = min(max_delay, suggested if suggested else base_delay * 2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d, %s); retrying in %.1fs",
                description,
                attempt,
                max_attempts,
                type(exc).__name__,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover

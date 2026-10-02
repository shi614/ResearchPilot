"""Gemini embeddings with free-tier quota protection.

`QuotaAwareEmbeddings` wraps any LangChain `Embeddings` and adds:
- explicit batching (one API request per batch),
- a client-side requests-per-minute throttle,
- retries with backoff on 429 / 5xx (honouring the API's suggested delay),
- mapping of failures to `RateLimitError` / `ExternalServiceError`.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings
from app.exceptions import ConfigurationError
from app.utils.gemini_errors import gemini_retry_after, is_retryable_gemini_error, to_service_error
from app.utils.retry import call_with_retry


class QuotaAwareEmbeddings(Embeddings):
    def __init__(
        self,
        inner: Embeddings,
        *,
        batch_size: int,
        max_rpm: int,
        max_attempts: int = 4,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._inner = inner
        self._batch_size = batch_size
        self._limiter = InMemoryRateLimiter(
            requests_per_second=max_rpm / 60, check_every_n_seconds=0.05, max_bucket_size=1
        )
        self._retry_kwargs: dict = {"max_attempts": max_attempts}
        if sleep is not None:
            self._retry_kwargs["sleep"] = sleep

    def _call(self, fn: Callable[[], list], operation: str) -> list:
        def throttled() -> list:
            self._limiter.acquire(blocking=True)
            return fn()

        try:
            return call_with_retry(
                throttled,
                should_retry=is_retryable_gemini_error,
                retry_after=gemini_retry_after,
                description=f"Gemini {operation}",
                **self._retry_kwargs,
            )
        except Exception as exc:
            raise to_service_error(exc, operation) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(
                self._call(
                    lambda batch=batch: self._inner.embed_documents(batch),
                    "document embedding",
                )
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._call(lambda: self._inner.embed_query(text), "query embedding")


def create_embeddings(settings: Settings) -> QuotaAwareEmbeddings:
    if settings.gemini_api_key is None:
        raise ConfigurationError("GEMINI_API_KEY is not set; embeddings are unavailable.")
    inner = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=settings.embedding_dimensions,
    )
    return QuotaAwareEmbeddings(
        inner, batch_size=settings.embedding_batch_size, max_rpm=settings.embedding_max_rpm
    )

"""Verifies that the configured Gemini models are usable with the user's API key.

Listing models does not consume generation quota, so the default check is
free. An optional *probe* sends one tiny request to confirm the model actually
has free-tier quota (some models are listed but have zero free quota).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.models.schemas import ModelCheckResponse, ModelCheckStatus

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 300
MAX_SUGGESTIONS = 5
# Flash variants that are not general-purpose text models.
_EXCLUDED_FLASH_MARKERS = ("image", "tts", "audio", "live", "exp", "thinking", "computer")
_VERSION_PATTERN = re.compile(r"gemini-(\d+(?:\.\d+)?)")


class GenAIClient(Protocol):
    """The subset of `google.genai.Client` this service relies on."""

    @property
    def models(self) -> Any: ...


ClientFactory = Callable[[str], GenAIClient]


def _default_client_factory(api_key: str) -> GenAIClient:
    return genai.Client(api_key=api_key)


def _flash_sort_key(name: str) -> tuple[bool, float, bool]:
    """Stable before preview, newer before older, full Flash before Flash-Lite."""
    match = _VERSION_PATTERN.search(name)
    version = float(match.group(1)) if match else 0.0
    is_stable = "preview" not in name and "latest" not in name
    return (is_stable, version, "lite" not in name)


def suggest_flash_models(available_generation_models: set[str]) -> list[str]:
    """Pick general-purpose Flash text models, best candidates first."""
    candidates = [
        name
        for name in available_generation_models
        if "flash" in name and not any(marker in name for marker in _EXCLUDED_FLASH_MARKERS)
    ]
    return sorted(candidates, key=_flash_sort_key, reverse=True)[:MAX_SUGGESTIONS]


class ModelCheckService:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        embedding_model: str,
        client_factory: ClientFactory = _default_client_factory,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._embedding_model = embedding_model
        self._client_factory = client_factory
        self._cache: dict[bool, tuple[float, ModelCheckResponse]] = {}
        self._lock = threading.Lock()

    def check(self, *, probe: bool = False, refresh: bool = False) -> ModelCheckResponse:
        with self._lock:
            cached = self._cache.get(probe)
            if cached and not refresh and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
                return cached[1]
            result = self._run_check(probe)
            self._cache[probe] = (time.monotonic(), result)
            return result

    # ------------------------------------------------------------------ internals

    def _result(self, status: ModelCheckStatus, message: str, **fields: Any) -> ModelCheckResponse:
        return ModelCheckResponse(
            status=status,
            configured_model=self._model,
            embedding_model=self._embedding_model,
            message=self._redact(message),
            checked_at=datetime.now(UTC),
            **fields,
        )

    def _redact(self, text: str) -> str:
        """Defence in depth: never let the API key leak through an error message."""
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return text

    def _run_check(self, probe: bool) -> ModelCheckResponse:
        if not self._api_key:
            return self._result(
                ModelCheckStatus.MISSING_API_KEY,
                "GEMINI_API_KEY is not set. Add it to your .env file and restart the backend.",
            )
        try:
            client = self._client_factory(self._api_key)
            generation_models, embedding_models = self._list_models(client)
        except Exception as exc:  # noqa: BLE001 - mapped to a user-facing status
            return self._error_result(exc)

        suggestions = suggest_flash_models(generation_models)
        model_available = self._model in generation_models
        embedding_available = self._embedding_model in embedding_models
        common = {
            "model_available": model_available,
            "embedding_available": embedding_available,
            "suggested_models": suggestions,
        }

        if not model_available:
            hint = f" Try setting GEMINI_MODEL={suggestions[0]} in .env." if suggestions else ""
            return self._result(
                ModelCheckStatus.MODEL_UNAVAILABLE,
                f"Model '{self._model}' is not available for this API key.{hint}",
                **common,
            )
        if not embedding_available:
            return self._result(
                ModelCheckStatus.MODEL_UNAVAILABLE,
                f"Embedding model '{self._embedding_model}' is not available for this API key.",
                **common,
            )
        if probe:
            try:
                client.models.generate_content(
                    model=self._model,
                    contents="Reply with OK.",
                    config=genai_types.GenerateContentConfig(max_output_tokens=8),
                )
            except Exception as exc:  # noqa: BLE001
                return self._error_result(exc, **common, probed=True)
            return self._result(
                ModelCheckStatus.OK,
                f"Model '{self._model}' is available and responded to a test request.",
                probed=True,
                **common,
            )
        return self._result(
            ModelCheckStatus.OK,
            f"Model '{self._model}' and embedding model '{self._embedding_model}' are available.",
            **common,
        )

    @staticmethod
    def _list_models(client: GenAIClient) -> tuple[set[str], set[str]]:
        generation: set[str] = set()
        embedding: set[str] = set()
        for model in client.models.list():
            name = (model.name or "").removeprefix("models/")
            actions = set(model.supported_actions or [])
            if "generateContent" in actions:
                generation.add(name)
            if "embedContent" in actions:
                embedding.add(name)
        return generation, embedding

    def _error_result(self, exc: Exception, **fields: Any) -> ModelCheckResponse:
        logger.warning("Gemini model check failed: %s", self._redact(type(exc).__name__))
        if isinstance(exc, genai_errors.ClientError):
            if exc.code == 429:
                return self._result(
                    ModelCheckStatus.QUOTA_EXHAUSTED,
                    f"Free-tier quota for '{self._model}' is exhausted or zero. "
                    "Wait for the quota to reset, or switch GEMINI_MODEL to another "
                    "suggested Flash model.",
                    **fields,
                )
            if exc.code in (400, 401, 403) and "api key" in (exc.message or "").lower():
                return self._result(
                    ModelCheckStatus.INVALID_API_KEY,
                    "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file.",
                    **fields,
                )
            if exc.code in (401, 403):
                return self._result(
                    ModelCheckStatus.INVALID_API_KEY,
                    "Gemini denied access with this API key (permission denied).",
                    **fields,
                )
            if exc.code == 404:
                listed = fields.get("model_available") is True
                detail = (
                    f"Model '{self._model}' is listed for this API key but is not available "
                    "for text generation (it may be retired or restricted for this key)."
                    if listed
                    else f"Model '{self._model}' was not found for this API key."
                )
                suggestions = [m for m in fields.get("suggested_models", []) if m != self._model]
                hint = f" Try setting GEMINI_MODEL={suggestions[0]} in .env." if suggestions else ""
                return self._result(ModelCheckStatus.MODEL_UNAVAILABLE, detail + hint, **fields)
            return self._result(
                ModelCheckStatus.ERROR, f"Gemini request failed ({exc.code}): {exc.message}", **fields
            )
        if isinstance(exc, genai_errors.ServerError):
            return self._result(
                ModelCheckStatus.ERROR,
                f"Gemini service error ({exc.code}). Try again shortly.",
                **fields,
            )
        if isinstance(exc, httpx.HTTPError):
            return self._result(
                ModelCheckStatus.ERROR,
                "Could not reach the Gemini API. Check your internet connection.",
                **fields,
            )
        return self._result(
            ModelCheckStatus.ERROR, f"Unexpected error during model check: {type(exc).__name__}", **fields
        )

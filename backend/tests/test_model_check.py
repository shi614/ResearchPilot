"""Model-check tests use a fake Gemini client built from real `google.genai` types."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.models.schemas import ModelCheckStatus
from app.services.model_check import ModelCheckService, suggest_flash_models

API_KEY = "AIza-test-key-should-never-leak"

AVAILABLE_MODELS = [
    ("models/gemini-2.5-flash", ["generateContent", "countTokens"]),
    ("models/gemini-2.5-flash-lite", ["generateContent"]),
    ("models/gemini-2.0-flash", ["generateContent"]),
    ("models/gemini-3-flash-preview", ["generateContent"]),
    ("models/gemini-2.5-flash-image", ["generateContent"]),
    ("models/gemini-2.5-flash-preview-tts", ["generateContent"]),
    ("models/gemini-2.5-pro", ["generateContent"]),
    ("models/gemini-embedding-001", ["embedContent"]),
]


class FakeModels:
    def __init__(self, models: list[tuple[str, list[str]]], generate_error: Exception | None) -> None:
        self._models = models
        self._generate_error = generate_error
        self.generate_calls = 0

    def list(self) -> list[genai_types.Model]:
        return [genai_types.Model(name=n, supported_actions=a) for n, a in self._models]

    def generate_content(self, **_kwargs: Any) -> None:
        self.generate_calls += 1
        if self._generate_error:
            raise self._generate_error


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


def make_service(
    model: str = "gemini-2.5-flash",
    *,
    models: list[tuple[str, list[str]]] | None = None,
    list_error: Exception | None = None,
    generate_error: Exception | None = None,
    api_key: str | None = API_KEY,
) -> tuple[ModelCheckService, FakeModels]:
    fake_models = FakeModels(models if models is not None else AVAILABLE_MODELS, generate_error)

    def factory(key: str) -> FakeClient:
        assert key == API_KEY
        if list_error:
            raise list_error
        return FakeClient(fake_models)

    service = ModelCheckService(api_key, model, "gemini-embedding-001", client_factory=factory)
    return service, fake_models


def client_error(code: int, message: str, status: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}})


def test_suggestions_exclude_non_text_variants_and_prefer_stable_newer_models() -> None:
    names = {name.removeprefix("models/") for name, _ in AVAILABLE_MODELS}
    suggestions = suggest_flash_models(names)
    assert suggestions == [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-3-flash-preview",
    ]
    assert not any("pro" in s or "image" in s or "tts" in s for s in suggestions)


def test_available_model_returns_ok_without_spending_quota() -> None:
    service, fake_models = make_service()
    result = service.check()
    assert result.status is ModelCheckStatus.OK
    assert result.model_available and result.embedding_available
    assert fake_models.generate_calls == 0


def test_unavailable_model_suggests_a_free_tier_flash_alternative() -> None:
    service, _ = make_service("gemini-1.5-flash")
    result = service.check()
    assert result.status is ModelCheckStatus.MODEL_UNAVAILABLE
    assert result.model_available is False
    assert result.suggested_models[0] == "gemini-2.5-flash"
    assert "GEMINI_MODEL=gemini-2.5-flash" in result.message


def test_missing_embedding_model_is_reported() -> None:
    service, _ = make_service(models=[("models/gemini-2.5-flash", ["generateContent"])])
    result = service.check()
    assert result.status is ModelCheckStatus.MODEL_UNAVAILABLE
    assert result.embedding_available is False


def test_missing_api_key() -> None:
    service, _ = make_service(api_key=None)
    assert service.check().status is ModelCheckStatus.MISSING_API_KEY


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (client_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"),
         ModelCheckStatus.INVALID_API_KEY),
        (client_error(403, "Permission denied", "PERMISSION_DENIED"), ModelCheckStatus.INVALID_API_KEY),
        (client_error(429, "Resource exhausted", "RESOURCE_EXHAUSTED"), ModelCheckStatus.QUOTA_EXHAUSTED),
        (genai_errors.ServerError(503, {"error": {"code": 503, "message": "unavailable"}}),
         ModelCheckStatus.ERROR),
        (httpx.ConnectError("offline"), ModelCheckStatus.ERROR),
    ],
)
def test_api_errors_map_to_clear_statuses(error: Exception, expected: ModelCheckStatus) -> None:
    service, _ = make_service(list_error=error)
    assert service.check().status is expected


def test_probe_detects_zero_free_tier_quota() -> None:
    service, fake_models = make_service(
        generate_error=client_error(429, "Quota exceeded", "RESOURCE_EXHAUSTED")
    )
    result = service.check(probe=True)
    assert fake_models.generate_calls == 1
    assert result.status is ModelCheckStatus.QUOTA_EXHAUSTED
    assert result.probed is True
    assert result.suggested_models  # still offers alternatives


def test_probe_success() -> None:
    service, fake_models = make_service()
    result = service.check(probe=True)
    assert result.status is ModelCheckStatus.OK and result.probed
    assert fake_models.generate_calls == 1


def test_results_are_cached_until_refresh() -> None:
    service, fake_models = make_service()
    service.check(probe=True)
    service.check(probe=True)
    assert fake_models.generate_calls == 1
    service.check(probe=True, refresh=True)
    assert fake_models.generate_calls == 2


def test_api_key_is_redacted_from_error_messages() -> None:
    service, _ = make_service(list_error=client_error(400, f"bad request key={API_KEY}", "INVALID_ARGUMENT"))
    result = service.check()
    assert API_KEY not in result.model_dump_json()
    assert "***" in result.message

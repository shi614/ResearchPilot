from __future__ import annotations

from app.models.schemas import ModelCheckResponse, ModelCheckStatus
from app.services.model_check import ModelCheckService

GEMINI_KEY = "AIza-api-test-secret"
TAVILY_KEY = "tvly-api-test-secret"


def test_health_is_degraded_when_keys_are_missing(client_factory) -> None:
    response = client_factory().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "ok"
    assert body["missing_keys"] == ["GEMINI_API_KEY", "TAVILY_API_KEY"]


def test_health_is_ok_and_never_exposes_keys(client_factory) -> None:
    client = client_factory(gemini_api_key=GEMINI_KEY, tavily_api_key=TAVILY_KEY)
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert body["gemini_api_key_configured"] is True
    assert GEMINI_KEY not in response.text and TAVILY_KEY not in response.text


def test_model_check_without_key_reports_missing_key(client_factory) -> None:
    response = client_factory().get("/health/models")
    assert response.status_code == 200
    assert response.json()["status"] == ModelCheckStatus.MISSING_API_KEY.value


def test_model_check_endpoint_passes_probe_flag_and_hides_key(client_factory) -> None:
    client = client_factory(gemini_api_key=GEMINI_KEY)
    calls: list[dict] = []

    class StubService(ModelCheckService):
        def check(self, *, probe: bool = False, refresh: bool = False) -> ModelCheckResponse:
            calls.append({"probe": probe, "refresh": refresh})
            return self._result(ModelCheckStatus.OK, f"fine {GEMINI_KEY}", model_available=True)

    client.app.state.model_check = StubService(GEMINI_KEY, "gemini-2.5-flash", "gemini-embedding-001")
    response = client.get("/health/models", params={"probe": "true"})
    assert response.status_code == 200
    assert calls == [{"probe": True, "refresh": False}]
    assert GEMINI_KEY not in response.text


def test_openapi_docs_available(client_factory) -> None:
    assert client_factory().get("/openapi.json").status_code == 200

"""End-to-end knowledge-base API tests: upload → process → search → delete."""

from __future__ import annotations

from pathlib import Path

from app.exceptions import RateLimitError
from tests.conftest import make_pdf

SOLAR_TEXT = (
    "# Solar energy\n\nPhotovoltaic solar panels convert sunlight into electricity. "
    "Panel efficiency has improved steadily over the last decade."
)


def upload(client, name: str, data: bytes, **params):
    return client.post("/knowledge/documents", files={"file": (name, data)}, params=params)


def test_upload_processes_and_makes_document_searchable(client_factory) -> None:
    client = client_factory()
    response = upload(client, "solar.md", SOLAR_TEXT.encode())
    assert response.status_code == 201
    doc = response.json()
    assert doc["status"] == "processed" and doc["chunk_count"] >= 1
    assert doc["content_type"] == "text/markdown"
    assert "stored_path" not in doc  # server paths are not exposed

    search = client.post("/knowledge/search", json={"query": "photovoltaic solar panel efficiency"})
    assert search.status_code == 200
    [first, *_] = search.json()["results"]
    assert first["document_id"] == doc["id"] and first["filename"] == "solar.md"


def test_pdf_upload_keeps_page_numbers(client_factory) -> None:
    client = client_factory()
    pdf = make_pdf(["Introduction to wind power.", "Offshore wind turbines generate more electricity."])
    doc = upload(client, "wind.pdf", pdf).json()
    assert doc["status"] == "processed"
    results = client.post("/knowledge/search", json={"query": "offshore wind turbines", "k": 1}).json()
    assert results["results"][0]["page"] == 2


def test_invalid_upload_returns_422_and_stores_nothing(client_factory, tmp_path: Path) -> None:
    client = client_factory()
    response = upload(client, "malware.exe", b"MZ...")
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]
    assert client.get("/knowledge/documents").json() == []
    assert list((tmp_path / "uploads").iterdir()) == []


def test_oversized_upload_is_rejected(client_factory) -> None:
    client = client_factory(max_upload_mb=1)
    assert upload(client, "big.txt", b"a" * (1_048_576 + 10)).status_code == 422


def test_unprocessable_document_is_saved_as_failed_and_can_be_retried(client_factory) -> None:
    client = client_factory()
    doc = upload(client, "blank.pdf", make_pdf([""])).json()
    assert doc["status"] == "failed" and "No usable text" in doc["error"]

    retried = client.post(f"/knowledge/documents/{doc['id']}/process")
    assert retried.status_code == 200 and retried.json()["status"] == "failed"


def test_quota_error_during_processing_is_recorded_not_crashing(client_factory, monkeypatch) -> None:
    client = client_factory()
    store = client.app.state.knowledge._store

    def exhausted(*_args, **_kwargs):
        raise RateLimitError("Gemini", "document embedding hit the free-tier rate limit.")

    monkeypatch.setattr(store, "add_document_chunks", exhausted)
    doc = upload(client, "notes.txt", b"Some perfectly valid text about batteries.").json()
    assert doc["status"] == "failed" and "rate limit" in doc["error"]


def test_upload_without_processing_then_process(client_factory) -> None:
    client = client_factory()
    doc = upload(client, "notes.txt", b"Grid batteries store renewable energy.", process="false").json()
    assert doc["status"] == "uploaded" and doc["chunk_count"] == 0
    processed = client.post(f"/knowledge/documents/{doc['id']}/process").json()
    assert processed["status"] == "processed"


def test_list_get_and_delete_document(client_factory, tmp_path: Path) -> None:
    client = client_factory()
    doc = upload(client, "solar.md", SOLAR_TEXT.encode()).json()
    assert [d["id"] for d in client.get("/knowledge/documents").json()] == [doc["id"]]
    assert client.get(f"/knowledge/documents/{doc['id']}").json()["filename"] == "solar.md"

    assert client.delete(f"/knowledge/documents/{doc['id']}").status_code == 204
    assert client.get("/knowledge/documents").json() == []
    assert list((tmp_path / "uploads").iterdir()) == []
    assert client.post("/knowledge/search", json={"query": "solar panels"}).json()["results"] == []


def test_unknown_document_returns_404(client_factory) -> None:
    client = client_factory()
    assert client.get("/knowledge/documents/missing").status_code == 404
    assert client.delete("/knowledge/documents/missing").status_code == 404
    assert client.post("/knowledge/documents/missing/process").status_code == 404


def test_search_validates_input(client_factory) -> None:
    client = client_factory()
    assert client.post("/knowledge/search", json={"query": ""}).status_code == 422
    assert client.post("/knowledge/search", json={"query": "x", "k": 100}).status_code == 422

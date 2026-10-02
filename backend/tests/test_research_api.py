"""Research API end-to-end over HTTP (real app, graph, SQLite, thread pool; scripted LLM)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.base import AgentDependencies
from app.graph.builder import build_research_graph
from app.graph.checkpoint import create_checkpointer
from app.graph.runner import ResearchRunner
from app.main import create_app
from tests.conftest import KeywordEmbeddings, make_settings
from tests.fakes import ScriptedLLM, fake_web, happy_llm


def build_client(tmp_path: Path, llm: ScriptedLLM, **overrides) -> TestClient:
    settings = make_settings(tmp_path, **({"gemini_api_key": "test-key"} | overrides))

    def runner_factory(app_settings, knowledge):
        def build(on_update):
            deps = AgentDependencies(llm=llm, settings=app_settings, web=fake_web(), knowledge=knowledge)
            graph = build_research_graph(deps, create_checkpointer(app_settings.checkpoint_db))
            return ResearchRunner(graph, on_update=on_update)

        return build

    app = create_app(settings, embeddings_factory=lambda _s: KeywordEmbeddings(), runner_factory=runner_factory)
    client = TestClient(app)
    client.__enter__()
    return client


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    c = build_client(tmp_path, happy_llm())
    yield c
    c.__exit__(None, None, None)


def settle(client: TestClient, session_id: str) -> dict:
    """Wait for the background operation, then fetch the session."""
    client.app.state.research.wait(session_id, timeout=30)
    response = client.get(f"/research/{session_id}")
    assert response.status_code == 200
    return response.json()


def start(client: TestClient, query: str = "How have solar costs changed?", **extra) -> dict:
    response = client.post("/research", json={"query": query, **extra})
    assert response.status_code == 202, response.text
    return settle(client, response.json()["id"])


def test_full_flow_over_http(client: TestClient) -> None:
    session = start(client, instructions="Focus on the last decade")
    assert session["status"] == "awaiting_approval"
    assert session["instructions"] == "Focus on the last decade"
    request = session["approval_request"]
    assert request["sources"]["total"] == 4  # empty knowledge base → web only
    assert request["research_questions"] and request["can_modify"] is True
    assert session["stats"]["llm_calls"] == 5 and session["stats"]["web_searches"] == 2
    assert session["events"][0]["node"] == "planner"
    assert client.get(f"/research/{session['id']}/report").status_code == 404  # not yet

    response = client.post(f"/research/{session['id']}/decision", json={"action": "approve"})
    assert response.status_code == 202
    done = settle(client, session["id"])
    assert done["status"] == "completed" and done["has_report"] and done["title"]

    report = client.get(f"/research/{session['id']}/report").json()
    assert report["title"] == done["title"]
    assert report["citations"] and "## References" in report["markdown"]
    assert {c["id"] for c in report["citations"]} <= {"W1", "W2", "W3", "W4"}


def test_events_endpoint_supports_incremental_polling(client: TestClient) -> None:
    session = start(client)
    events = session["events"]
    newer = client.get(f"/research/{session['id']}", params={"after_event": events[1]["id"]}).json()["events"]
    assert [e["id"] for e in newer] == [e["id"] for e in events[2:]]


def test_uploaded_document_feeds_rag_evidence(tmp_path: Path) -> None:
    c = build_client(tmp_path, happy_llm(), rag_min_relevance=0.0)
    upload = c.post("/knowledge/documents",
                    files={"file": ("solar.md", b"# Solar\n\nSolar costs changed and fell; adoption drivers.")})
    assert upload.json()["status"] == "processed"
    session = start(c)
    assert session["approval_request"]["sources"]["knowledge_base"] >= 1
    assert session["stats"]["evidence_knowledge_base"] >= 1
    assert any(e["node"] == "rag_research" for e in session["events"])
    c.__exit__(None, None, None)


def test_modify_and_cancel_decisions(client: TestClient) -> None:
    session = start(client)
    modify = client.post(f"/research/{session['id']}/decision",
                         json={"action": "modify", "feedback": "Include policy incentives"})
    assert modify.status_code == 202
    again = settle(client, session["id"])
    assert again["status"] == "awaiting_approval" and again["approval_request"]["iteration"] == 2

    client.post(f"/research/{session['id']}/decision", json={"action": "cancel"})
    assert settle(client, session["id"])["status"] == "cancelled"


def test_invalid_transitions_return_409(client: TestClient) -> None:
    session = start(client)
    retry = client.post(f"/research/{session['id']}/retry")
    assert retry.status_code == 409 and retry.json()["error"] == "WorkflowError"
    client.post(f"/research/{session['id']}/decision", json={"action": "approve"})
    settle(client, session["id"])
    again = client.post(f"/research/{session['id']}/decision", json={"action": "approve"})
    assert again.status_code == 409 and "not waiting for approval" in again.json()["detail"]


def test_history_list_and_delete(client: TestClient) -> None:
    first = start(client, query="First topic")
    second = start(client, query="Second topic")
    history = client.get("/research").json()
    assert [h["id"] for h in history] == [second["id"], first["id"]]
    assert client.get("/research", params={"limit": 1}).json()[0]["id"] == second["id"]

    assert client.delete(f"/research/{first['id']}").status_code == 204
    assert client.get(f"/research/{first['id']}").status_code == 404
    assert [h["id"] for h in client.get("/research").json()] == [second["id"]]


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/research/missing", None),
        ("get", "/research/missing/report", None),
        ("post", "/research/missing/decision", {"action": "approve"}),
        ("post", "/research/missing/retry", None),
        ("delete", "/research/missing", None),
    ],
)
def test_unknown_session_returns_404(client: TestClient, method: str, path: str, body) -> None:
    response = getattr(client, method)(path, **({"json": body} if body else {}))
    assert response.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "ab"}, {"query": "x" * 2001}, {"query": "valid query", "instructions": "x" * 2001}, {}],
)
def test_start_validates_input(client: TestClient, payload: dict) -> None:
    assert client.post("/research", json=payload).status_code == 422


def test_decision_validates_action(client: TestClient) -> None:
    session = start(client)
    assert client.post(f"/research/{session['id']}/decision", json={"action": "maybe"}).status_code == 422
    assert client.get("/research", params={"limit": 0}).status_code == 422


def test_missing_gemini_key_returns_503_but_history_works(tmp_path: Path) -> None:
    c = build_client(tmp_path, happy_llm(), gemini_api_key=None)
    response = c.post("/research", json={"query": "valid query"})
    assert response.status_code == 503 and "GEMINI_API_KEY" in response.json()["detail"]
    assert c.get("/research").json() == []
    c.__exit__(None, None, None)


def test_startup_recovers_sessions_left_running(tmp_path: Path) -> None:
    from app.database import Database, ResearchRepository
    from app.database.orm import SessionStatus

    settings = make_settings(tmp_path)
    settings.ensure_directories()
    db = Database(settings.database_url)
    db.create_tables()
    with db.session() as s:
        repo = ResearchRepository(s)
        repo.update_status(repo.create("left running").id, SessionStatus.RUNNING)
    db.dispose()

    c = build_client(tmp_path, happy_llm())
    [session] = c.get("/research").json()
    assert session["status"] == "interrupted"
    c.__exit__(None, None, None)

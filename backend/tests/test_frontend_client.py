"""Frontend API client against the real FastAPI app (scripted LLM), plus stage derivation."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from frontend.api_client import ApiError, BackendUnavailable, ResearchPilotClient
from frontend.workflow import progress_fraction, stage_views
from tests.test_research_api import build_client
from tests.fakes import happy_llm

# The real client sets per-request timeouts (long uploads); Starlette's TestClient warns about them.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


@pytest.fixture
def api(tmp_path: Path):
    test_client = build_client(tmp_path, happy_llm())
    yield ResearchPilotClient("http://testserver", http=test_client), test_client
    test_client.__exit__(None, None, None)


def settle(test_client, session_id: str) -> None:
    test_client.app.state.research.wait(session_id, timeout=30)


def test_client_drives_the_full_research_flow(api) -> None:
    client, test_client = api
    started = client.start_research("How have solar costs changed?", "")
    settle(test_client, started["id"])
    session = client.get_research(started["id"])
    assert session["status"] == "awaiting_approval" and session["approval_request"]

    client.decide(started["id"], "approve")
    settle(test_client, started["id"])
    assert client.get_research(started["id"])["status"] == "completed"
    assert "## References" in client.get_report(started["id"])["markdown"]
    assert client.get_report_pdf(started["id"]).startswith(b"%PDF-")
    assert [s["id"] for s in client.list_research()] == [started["id"]]

    client.delete_research(started["id"])
    assert client.list_research() == []


def test_client_knowledge_base_and_settings(api) -> None:
    client, _ = api
    doc = client.upload_document("notes.md", b"# Notes\n\nSolar panels convert sunlight into electricity.")
    assert doc["status"] == "processed"
    assert [d["id"] for d in client.list_documents()] == [doc["id"]]
    assert client.process_document(doc["id"])["status"] == "processed"
    client.delete_document(doc["id"])
    assert client.list_documents() == []
    assert client.health()["database"] == "ok"
    assert client.config()["max_web_searches"] == 3


def test_api_errors_carry_backend_messages(api) -> None:
    client, _ = api
    with pytest.raises(ApiError) as missing:
        client.get_research("missing")
    assert missing.value.status_code == 404 and "not found" in missing.value.message
    with pytest.raises(ApiError) as invalid:
        client.start_research("ab")
    assert invalid.value.status_code == 422 and "at least 3 characters" in invalid.value.message
    with pytest.raises(ApiError) as bad_file:
        client.upload_document("virus.exe", b"MZ")
    assert "Unsupported file type" in bad_file.value.message


def test_unreachable_backend_raises_friendly_error() -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = ResearchPilotClient("http://127.0.0.1:9", http=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(BackendUnavailable, match="python run.py backend"):
        client.health()


# --------------------------------------------------------------------------- stage derivation


def event(node: str, status: str = "completed", message: str = "ok") -> dict:
    return {"id": 1, "node": node, "status": status, "message": message, "created_at": "2026-10-02T00:00:00"}


def states(session: dict) -> dict[str, str]:
    return {v.node: v.state for v in stage_views(session)}


def test_stages_reflect_running_backend_state() -> None:
    session = {"status": "running", "pending_nodes": ["source_verification"],
               "events": [event("planner"), event("web_research"), event("rag_research")]}
    s = states(session)
    assert s["planner"] == s["web_research"] == s["rag_research"] == "done"
    assert s["source_verification"] == "active" and s["analysis"] == "pending"


def test_new_run_shows_planning_active_before_first_checkpoint() -> None:
    assert states({"status": "running", "pending_nodes": [], "events": []})["planner"] == "active"


def test_awaiting_approval_and_degraded_steps() -> None:
    session = {"status": "awaiting_approval", "pending_nodes": ["human_review"],
               "events": [event("planner"), event("web_research"), event("source_verification"),
                          event("analysis", "failed", "Analysis failed (503)"), event("analysis", message="unavailable")]}
    s = states(session)
    assert s["human_review"] == "waiting" and s["analysis"] == "warning"
    assert s["rag_research"] == "skipped"  # knowledge base not used in this run


def test_stopped_run_marks_failed_stage_and_completed_run_skips_revision() -> None:
    stopped = {"status": "interrupted", "pending_nodes": ["writer"], "events": [event("human_review")]}
    assert states(stopped)["writer"] == "failed"
    done = {"status": "completed", "pending_nodes": [],
            "events": [event(n) for n in ("planner", "web_research", "rag_research", "source_verification",
                                          "analysis", "human_review", "writer", "critic", "finalize")]}
    views = stage_views(done)
    assert {v.node: v.state for v in views}["revision"] == "skipped"
    assert next(v for v in views if v.node == "revision").detail == "Not needed"
    assert progress_fraction(views, "completed") == 1.0

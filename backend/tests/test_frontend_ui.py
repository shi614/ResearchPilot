"""Headless Streamlit UI tests (AppTest) with a fake API client in session state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import BackendUnavailable

APP_FILE = str(Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py")
NOW = "2026-10-02T18:30:00"

CONFIG = {
    "gemini_model": "gemini-3.5-flash", "gemini_embedding_model": "gemini-embedding-001", "gemini_max_rpm": 8,
    "max_revisions": 1, "max_research_iterations": 2, "max_web_searches": 3, "max_research_tool_rounds": 2,
    "writer_max_sources": 14, "tavily_search_depth": "basic", "rag_top_k": 4, "rag_min_relevance": 0.6,
    "chunk_size": 1000, "chunk_overlap": 150, "max_upload_mb": 20, "web_research_enabled": True,
    "research_enabled": True,
}
HEALTH = {"status": "ok", "version": "0.1.0", "database": "ok", "gemini_api_key_configured": True,
          "tavily_api_key_configured": True, "missing_keys": [], "gemini_model": "gemini-3.5-flash"}


def event(node: str, status: str = "completed", message: str = "done") -> dict:
    return {"id": 1, "node": node, "status": status, "message": message, "created_at": NOW}


def session(status: str, **extra: Any) -> dict:
    return {
        "id": "s1", "query": "What are the benefits of RAG?", "status": status, "title": None,
        "created_at": NOW, "updated_at": NOW, "has_report": status == "completed", "instructions": None,
        "error": None, "retryable": False, "pending_nodes": [], "approval_request": None, "errors": [],
        "stats": {"llm_calls": 5, "web_searches": 3, "web_sources": 9, "knowledge_base_sources": 1, "evidence": 10,
                  "evidence_web": 9, "evidence_knowledge_base": 1, "iteration": 1, "revisions": 0},
        "events": [event("planner"), event("web_research"), event("rag_research")],
    } | extra


APPROVAL = {
    "type": "approval_request", "iteration": 1, "can_modify": True,
    "research_questions": ["What are RAG's benefits?", "What are its limitations?"],
    "sources": {"web": 9, "knowledge_base": 1, "total": 10}, "answer_summary": "RAG grounds answers.",
    "key_findings": ["RAG reduces unsupported answers"], "conflicts": [], "errors": [],
}
REPORT = {
    "session_id": "s1", "title": "Benefits and Limitations of RAG", "markdown": "# Benefits and Limitations of RAG\n\nBody [W1].",
    "report": {}, "quality_notes": [], "created_at": NOW,
    "citations": [{"id": "W1", "title": "RAG survey", "reference": "https://example.org", "source_type": "web",
                   "verification_status": "corroborated", "excerpt": ""}],
}


class FakeClient:
    def __init__(self, research: dict | None = None, *, offline: bool = False, **overrides: Any) -> None:
        self.research = research
        self.offline = offline
        self.calls: list[tuple] = []
        self.overrides = overrides

    def _respond(self, name: str, default: Any, *args: Any) -> Any:
        self.calls.append((name, *args))
        if self.offline:
            raise BackendUnavailable("Cannot reach the ResearchPilot backend at http://127.0.0.1:8000.")
        return self.overrides.get(name, default)

    def health(self): return self._respond("health", HEALTH)
    def config(self): return self._respond("config", CONFIG)
    def model_check(self, *, probe=False): return self._respond("model_check", {"status": "ok", "message": "fine"}, probe)
    def list_research(self, limit=100): return self._respond("list_research", [])
    def get_research(self, session_id, after_event=0): return self._respond("get_research", self.research, session_id)
    def start_research(self, query, instructions=None):
        return self._respond("start_research", session("running", id="new1"), query, instructions)
    def decide(self, session_id, action, feedback=None):
        return self._respond("decide", session("running"), session_id, action, feedback)
    def retry(self, session_id): return self._respond("retry", session("running"), session_id)
    def delete_research(self, session_id): return self._respond("delete_research", None, session_id)
    def get_report(self, session_id): return self._respond("get_report", REPORT, session_id)
    def get_report_pdf(self, session_id): return self._respond("get_report_pdf", b"%PDF-1.4 test", session_id)
    def list_documents(self): return self._respond("list_documents", [])
    def upload_document(self, filename, data): return self._respond("upload_document", {}, filename)
    def process_document(self, document_id): return self._respond("process_document", {}, document_id)
    def delete_document(self, document_id): return self._respond("delete_document", None, document_id)

    def called(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]


def run_page(page: str, fake: FakeClient, active_session: str | None = None) -> AppTest:
    def script(page_name: str) -> None:
        from frontend.views import history, knowledge_base, new_research, settings_page

        {"research": new_research, "history": history, "knowledge": knowledge_base,
         "settings": settings_page}[page_name].render()

    app = AppTest.from_function(script, args=(page,), default_timeout=90)  # first run may load pyarrow natively (slow on cold Windows)
    app.session_state["api_client"] = fake
    if active_session:
        app.session_state["active_session_id"] = active_session
    return app.run()


def texts(app: AppTest) -> str:
    parts = [m.value for m in app.markdown] + [e.value for e in app.caption] + [t.value for t in app.title]
    parts += [h.value for h in app.header]
    parts += [x.value for kind in (app.info, app.warning, app.error, app.success, app.subheader) for x in kind]
    return "\n".join(str(p) for p in parts)


def button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


# --------------------------------------------------------------------------- new research


def test_start_form_renders_user_facing_content() -> None:
    app = run_page("research", FakeClient())
    assert not app.exception
    page = texts(app)
    assert "ResearchPilot" in page and "Multi-Agent AI Research" in page
    assert len(app.text_area) == 2 and "pauses for your review" in page and "How it works" in page
    assert "web searches" not in page and "gemini" not in page.lower()  # no technical limits on the main page


def test_start_requires_a_topic_then_starts_research() -> None:
    fake = FakeClient(research=session("running", id="new1"))
    app = run_page("research", fake)
    button(app, "Start Research").click().run()
    assert "at least 3 characters" in texts(app) and not fake.called("start_research")

    app.text_area[0].input("What are the benefits of RAG?")
    app.text_area[1].input("Focus on enterprises")
    button(app, "Start Research").click().run()
    assert fake.called("start_research") == [("start_research", "What are the benefits of RAG?", "Focus on enterprises")]
    assert app.session_state["active_session_id"] == "new1"


def test_research_disabled_without_gemini_key() -> None:
    app = run_page("research", FakeClient(config=CONFIG | {"research_enabled": False}))
    assert "AI service isn't configured" in texts(app) and "GEMINI_API_KEY" not in texts(app)
    assert button(app, "Start Research").disabled


def test_running_session_shows_live_stages() -> None:
    fake = FakeClient(research=session("running", pending_nodes=["source_verification"]))
    app = run_page("research", fake, active_session="s1")
    assert not app.exception
    page = texts(app)
    assert "agents are working" in page and "Source Verification" in page and "In progress" in page
    assert app.metric and fake.called("get_research")


def test_approval_panel_approve_modify_cancel() -> None:
    fake = FakeClient(research=session("awaiting_approval", pending_nodes=["human_review"],
                                       approval_request=APPROVAL))
    app = run_page("research", fake, active_session="s1")
    page = texts(app)
    assert "Review the research before the report is written" in page and "RAG reduces unsupported answers" in page
    assert "What are its limitations?" in page

    button(app, "Generate Final Report").click().run()
    assert fake.called("decide")[-1] == ("decide", "s1", "approve", None)

    button(app, "Run another research iteration").click().run()
    assert "Please describe what should change." in texts(app)
    next(t for t in app.text_area if t.key == "modify_feedback").input("Add cost data")
    button(app, "Run another research iteration").click().run()
    assert fake.called("decide")[-1] == ("decide", "s1", "modify", "Add cost data")

    button(app, "Cancel research").click().run()
    assert fake.called("decide")[-1] == ("decide", "s1", "cancel", None)


def test_modify_disabled_at_iteration_limit() -> None:
    fake = FakeClient(research=session("awaiting_approval", pending_nodes=["human_review"],
                                       approval_request=APPROVAL | {"can_modify": False}))
    app = run_page("research", fake, active_session="s1")
    assert button(app, "Run another research iteration").disabled
    assert "maximum number of times" in texts(app)


def test_completed_session_shows_report_sources_and_pdf() -> None:
    fake = FakeClient(research=session("completed", title="Benefits and Limitations of RAG",
                                       events=[event(n) for n in ("planner", "web_research", "source_verification",
                                                                  "analysis", "human_review", "writer", "critic",
                                                                  "finalize")]))
    app = run_page("research", fake, active_session="s1")
    assert not app.exception
    assert "Body [W1]." in texts(app)
    assert not any(m.value.lstrip().startswith("# ") for m in app.markdown)  # title not repeated in the body
    assert fake.called("get_report") and fake.called("get_report_pdf")
    assert app.get("download_button")  # PDF download offered
    assert app.dataframe  # sources table


def test_stopped_session_offers_retry() -> None:
    fake = FakeClient(research=session("interrupted", retryable=True, pending_nodes=["writer"],
                                       error="Gemini service error (503)."))
    app = run_page("research", fake, active_session="s1")
    assert "stopped before finishing" in texts(app) and "503" in texts(app)
    button(app, "Retry from last checkpoint").click().run()
    assert fake.called("retry") == [("retry", "s1")]


def test_non_retryable_failure_has_no_retry_button() -> None:
    app = run_page("research", FakeClient(research=session("failed", error="No sources found.")), active_session="s1")
    assert "No sources found." in texts(app)
    assert not any(b.label == "Retry from last checkpoint" for b in app.button)


def test_cancelled_session_message() -> None:
    app = run_page("research", FakeClient(research=session("cancelled")), active_session="s1")
    assert "cancelled" in texts(app)


def test_new_research_button_clears_active_session() -> None:
    app = run_page("research", FakeClient(research=session("cancelled")), active_session="s1")
    button(app, "New research").click().run()
    assert app.session_state["active_session_id"] is None


# --------------------------------------------------------------------------- other pages


@pytest.mark.parametrize(
    ("page", "message"),
    [("history", "No research yet"), ("knowledge", "No documents yet")],
)
def test_empty_states(page: str, message: str) -> None:
    app = run_page(page, FakeClient())
    assert not app.exception and message in texts(app)


def test_history_lists_sessions() -> None:
    rows = [session("completed", id="a", title="Report A"), session("failed", id="b")]
    app = run_page("history", FakeClient(list_research=rows))
    assert app.dataframe and "2 research sessions" in texts(app)


def test_knowledge_base_lists_documents() -> None:
    docs = [{"id": "d1", "filename": "notes.pdf", "content_type": "application/pdf", "size_bytes": 2048,
             "chunk_count": 7, "status": "processed", "error": None, "created_at": NOW}]
    app = run_page("knowledge", FakeClient(list_documents=docs))
    assert app.dataframe and "1 document." in texts(app)


def test_settings_is_a_user_facing_overview() -> None:
    docs = [{"id": "d1", "filename": "a.pdf", "content_type": "application/pdf", "size_bytes": 1, "chunk_count": 7,
             "status": "processed", "error": None, "created_at": NOW},
            {"id": "d2", "filename": "b.pdf", "content_type": "application/pdf", "size_bytes": 1, "chunk_count": 0,
             "status": "failed", "error": "No text", "created_at": NOW}]
    fake = FakeClient(list_documents=docs)
    app = run_page("settings", fake)
    assert not app.exception
    page = texts(app)
    assert "All systems operational" in page and "Gemini 3.5 Flash" in page
    assert "1 document ready" in page and "7 searchable passages" in page and "1 document needs attention" in page
    assert "About ResearchPilot" in page
    for technical in ("requests / minute", "Tool-calling", "Chunk", "Minimum relevance", "Max upload",
                      "Database", "API key", "Tavily key", "Critic revisions", "Research iterations"):
        assert technical.lower() not in page.lower(), technical
    assert not any(b.label in ("Test with one request", "Check availability") for b in app.button)
    assert not fake.called("model_check")  # never spends Gemini quota


def test_settings_reports_limited_features_simply() -> None:
    app = run_page("settings", FakeClient(config=CONFIG | {"web_research_enabled": False}))
    assert "web search is unavailable" in texts(app)
    app = run_page("settings", FakeClient(config=CONFIG | {"research_enabled": False}))
    assert any("Research is unavailable" in e.value for e in app.error)


@pytest.mark.parametrize("page", ["research", "history", "knowledge", "settings"])
def test_backend_offline_shows_friendly_error(page: str) -> None:
    app = run_page(page, FakeClient(offline=True), active_session="s1" if page == "research" else None)
    assert not app.exception
    assert any("Cannot reach the ResearchPilot backend" in e.value for e in app.error)


def test_full_app_boots_with_navigation_and_sidebar_status() -> None:
    app = AppTest.from_file(APP_FILE, default_timeout=90)  # first run may load pyarrow natively (slow on cold Windows)
    app.session_state["api_client"] = FakeClient()
    app.run()
    assert not app.exception
    assert "ResearchPilot" in texts(app)
    assert any("Online" in m.value for m in app.sidebar.markdown)  # simple status badge


def test_timestamps_are_shown_in_local_time_and_counts_pluralised() -> None:
    from datetime import UTC, datetime

    from frontend.components import local_time, plural

    expected = datetime(2026, 10, 2, 18, 30, tzinfo=UTC).astimezone().strftime("%d %b %Y, %H:%M")
    assert local_time("2026-10-02T18:30:00") == expected  # naive backend value treated as UTC
    assert local_time("2026-10-02T18:30:00+00:00") == expected
    assert local_time("not a date") == "not a date"
    assert plural(1, "passage") == "1 passage" and plural(7, "passage") == "7 passages"

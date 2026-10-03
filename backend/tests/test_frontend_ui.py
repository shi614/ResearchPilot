"""Headless Streamlit UI tests (AppTest) with a fake API client in session state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import BackendUnavailable

APP_FILE = str(Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py")
NOW = "2026-10-02T18:30:00"
TIMEOUT = 90  # first run may load pyarrow natively (slow on a cold Windows machine)

CONFIG = {
    "gemini_model": "gemini-3.5-flash", "gemini_embedding_model": "gemini-embedding-001", "gemini_max_rpm": 8,
    "max_revisions": 1, "max_research_iterations": 2, "max_web_searches": 3, "max_research_tool_rounds": 2,
    "writer_max_sources": 14, "tavily_search_depth": "basic", "rag_top_k": 4, "rag_min_relevance": 0.6,
    "chunk_size": 1000, "chunk_overlap": 150, "max_upload_mb": 20, "web_research_enabled": True,
    "research_enabled": True,
}
HEALTH = {"status": "ok", "version": "0.1.0", "database": "ok", "gemini_api_key_configured": True,
          "tavily_api_key_configured": True, "missing_keys": [], "gemini_model": "gemini-3.5-flash"}
TECHNICAL_TERMS = ("requests / minute", "requests per minute", "Tool-calling", "Chunk size", "Chunk overlap",
                   "Minimum relevance", "Max upload", "Database", "API key", "Tavily key", "Critic revisions",
                   "Research iterations", "RPM", "backend URL")


def event(node: str, status: str = "completed", message: str = "done") -> dict:
    return {"id": 1, "node": node, "status": status, "message": message, "created_at": NOW}


def session(status: str, **extra: Any) -> dict:
    return {
        "id": "s1", "query": "What are the benefits of RAG?", "status": status, "title": None,
        "created_at": NOW, "updated_at": NOW, "has_report": status == "completed", "instructions": None,
        "error": None, "retryable": False, "pending_nodes": [], "approval_request": None, "errors": [],
        "stats": {"llm_calls": 5, "web_searches": 3, "web_sources": 9, "knowledge_base_sources": 1, "evidence": 10,
                  "evidence_web": 9, "evidence_knowledge_base": 1, "iteration": 1, "revisions": 1},
        "events": [event("planner"), event("web_research"), event("rag_research")],
    } | extra


APPROVAL = {
    "type": "approval_request", "iteration": 1, "can_modify": True,
    "research_questions": ["What are RAG's benefits?", "What are its limitations?"],
    "sources": {"web": 9, "knowledge_base": 1, "total": 10}, "answer_summary": "RAG grounds answers.",
    "key_findings": ["RAG reduces unsupported answers"], "conflicts": [],
    "errors": ["Analysis failed: Gemini service error (503)."],
}
REPORT = {
    "session_id": "s1", "title": "Benefits and Limitations of RAG",
    "markdown": "# Benefits and Limitations of RAG\n\n## Executive Summary\n\nBody [W1].",
    "report": {"executive_summary": "RAG grounds answers in retrieved text [W1].",
               "key_findings": ["Fewer unsupported answers [W1]", "Retrieval quality matters [K1]"]},
    "quality_notes": ["1 of 2 cited sources are not corroborated by a second source."], "created_at": NOW,
    "citations": [
        {"id": "W1", "title": "RAG survey", "reference": "https://example.org/rag", "source_type": "web",
         "verification_status": "corroborated", "excerpt": ""},
        {"id": "K1", "title": "team_notes.md", "reference": "team_notes.md", "source_type": "knowledge_base",
         "verification_status": "single_source", "excerpt": ""},
    ],
}
DOCS = [
    {"id": "d1", "filename": "notes.pdf", "content_type": "application/pdf", "size_bytes": 2048, "chunk_count": 7,
     "status": "processed", "error": None, "created_at": NOW},
    {"id": "d2", "filename": "scan.pdf", "content_type": "application/pdf", "size_bytes": 4096, "chunk_count": 0,
     "status": "failed", "error": "No usable text could be extracted.", "created_at": NOW},
]


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
        from frontend.ui.theme import inject_theme
        from frontend.views import history, knowledge_base, new_research, settings_page

        inject_theme()
        {"research": new_research, "history": history, "knowledge": knowledge_base,
         "settings": settings_page}[page_name].render()

    app = AppTest.from_function(script, args=(page,), default_timeout=TIMEOUT)
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


def labels(app: AppTest) -> set[str]:
    return {b.label for b in app.button}


# --------------------------------------------------------------------------- new research


def test_landing_explains_the_product_within_seconds() -> None:
    app = run_page("research", FakeClient())
    assert not app.exception
    page = texts(app)
    for expected in ("ResearchPilot", "Multi-Agent AI Research", "Report Generation", "System Ready",
                     "What would you like to research?", "How ResearchPilot works", "8 specialised agents",
                     "LangGraph"):
        assert expected in page, expected
    for step in ("01", "Plan", "02", "Research", "03", "Verify", "04", "Review", "05", "Report"):
        assert step in page
    assert "Start Research →" in labels(app)
    assert "gemini" not in page.lower() and "web searches" not in page  # no technical limits


def test_composer_requires_a_topic_then_starts_research() -> None:
    fake = FakeClient(research=session("running", id="new1"))
    app = run_page("research", fake)
    button(app, "Start Research →").click().run()
    assert "at least 3 characters" in texts(app) and not fake.called("start_research")

    app.text_area(key="composer_query").input("What are the benefits of RAG?")
    app.text_area(key="composer_instructions").input("Focus on enterprises")
    button(app, "Start Research →").click().run()
    assert fake.called("start_research") == [("start_research", "What are the benefits of RAG?", "Focus on enterprises")]
    assert app.session_state["active_session_id"] == "new1"
    assert "composer_query" not in app.session_state  # composer cleared for the next research


def test_research_disabled_without_ai_service() -> None:
    app = run_page("research", FakeClient(config=CONFIG | {"research_enabled": False}))
    assert "AI service isn't configured" in texts(app) and "GEMINI_API_KEY" not in texts(app)
    assert button(app, "Start Research →").disabled
    assert "Limited Mode" in texts(app)


# --------------------------------------------------------------------------- research session


def test_running_session_shows_agent_workflow_timeline() -> None:
    fake = FakeClient(research=session("running", pending_nodes=["source_verification"]))
    app = run_page("research", fake, active_session="s1")
    assert not app.exception
    page = texts(app)
    assert "Research in progress" in page and "Agent working..." in page and "Agent workflow" in page
    assert "rp-tl-item active" in page and "Source Verification" in page
    assert page.count("rp-tl-item done") == 3 and page.count("rp-tl-item pending") == 6
    assert "Cross-checking sources" in page  # active step explained in plain language


def test_approval_is_a_clear_decision_point() -> None:
    fake = FakeClient(research=session("awaiting_approval", pending_nodes=["human_review"], approval_request=APPROVAL))
    app = run_page("research", fake, active_session="s1")
    page = texts(app)
    assert "Your Review Is Needed" in page and "ready to generate the final report" in page
    assert "Sources found" in page and "RAG reduces unsupported answers" in page and "What are its limitations?" in page
    assert "Some steps completed with limitations" in page
    assert "503" not in page and "503" in app.code[0].value  # raw errors only inside "Technical details"
    assert {"Approve & Generate Report", "Modify Research", "Cancel"} <= labels(app)

    button(app, "Approve & Generate Report").click().run()
    assert fake.called("decide")[-1] == ("decide", "s1", "approve", None)
    button(app, "Cancel").click().run()
    assert fake.called("decide")[-1] == ("decide", "s1", "cancel", None)


def test_modify_research_panel() -> None:
    fake = FakeClient(research=session("awaiting_approval", pending_nodes=["human_review"], approval_request=APPROVAL))
    app = run_page("research", fake, active_session="s1")
    assert "Run another research iteration" not in labels(app)
    button(app, "Modify Research").click().run()
    button(app, "Run another research iteration").click().run()
    assert "Please describe what should change." in texts(app)
    app.text_area(key="modify_feedback").input("Add cost data")
    button(app, "Run another research iteration").click().run()
    assert fake.called("decide")[-1] == ("decide", "s1", "modify", "Add cost data")


def test_modify_disabled_at_iteration_limit() -> None:
    fake = FakeClient(research=session("awaiting_approval", pending_nodes=["human_review"],
                                       approval_request=APPROVAL | {"can_modify": False}))
    app = run_page("research", fake, active_session="s1")
    button(app, "Modify Research").click().run()
    assert button(app, "Run another research iteration").disabled
    assert "maximum number of times" in texts(app)


def completed_app() -> tuple[AppTest, FakeClient]:
    fake = FakeClient(research=session("completed", title="Benefits and Limitations of RAG",
                                       events=[event(n) for n in ("planner", "web_research", "source_verification",
                                                                  "analysis", "human_review", "writer", "critic",
                                                                  "finalize")]))
    return run_page("research", fake, active_session="s1"), fake


def test_completed_report_feels_finished() -> None:
    app, fake = completed_app()
    assert not app.exception
    page = texts(app)
    assert "Research Complete ✓" in page and "Benefits and Limitations of RAG" in page and "2 cited sources" in page
    assert "Executive Summary" in page and "RAG grounds answers in retrieved text" in page
    assert "Key Findings" in page and "Retrieval quality matters" in page
    assert {"View Report", "Back to History", "New Research"} <= labels(app)
    assert app.get("download_button")  # PDF download
    assert fake.called("get_report") and fake.called("get_report_pdf")


def test_view_report_and_sources() -> None:
    app, _ = completed_app()
    button(app, "View Report").click().run()
    assert app.session_state["view:s1"] == "Full report"
    assert "Body [W1]." in texts(app)
    assert not any(m.value.lstrip().startswith("# ") for m in app.markdown)  # title not repeated

    app.session_state["view:s1"] = "Sources"
    app.run()
    page = texts(app)
    assert "RAG survey" in page and "Corroborated" in page and "Single source" in page
    assert "https://example.org/rag" in page


def test_stopped_session_explains_and_offers_retry() -> None:
    fake = FakeClient(research=session("interrupted", retryable=True, pending_nodes=["writer"],
                                       error="Gemini: report writing failed: Gemini service error (503)."))
    app = run_page("research", fake, active_session="s1")
    page = texts(app)
    assert "What happened?" in page and "Stopped at Report Generation" in page
    assert "safely saved" in page and "retry from the last checkpoint" in page
    assert "503" not in page and "503" in app.code[0].value  # raw error only in Technical details
    button(app, "Retry from checkpoint").click().run()
    assert fake.called("retry") == [("retry", "s1")]


def test_non_retryable_failure_has_no_retry_button() -> None:
    app = run_page("research", FakeClient(research=session("failed", error="No sources found.")), active_session="s1")
    assert "be completed. Try rephrasing the question" in texts(app)
    assert "Retry from checkpoint" not in labels(app)
    assert "No sources found." in app.code[0].value


def test_cancelled_session_message() -> None:
    app = run_page("research", FakeClient(research=session("cancelled")), active_session="s1")
    assert "This research was cancelled" in texts(app)


def test_new_research_button_clears_active_session() -> None:
    app = run_page("research", FakeClient(research=session("cancelled")), active_session="s1")
    button(app, "New Research").click().run()
    assert app.session_state["active_session_id"] is None


# --------------------------------------------------------------------------- history


def test_history_empty_state_starts_new_research() -> None:
    app = run_page("history", FakeClient())
    assert not app.exception
    assert "Your research history is empty" in texts(app) and "Start your first research task" in texts(app)
    app.session_state["active_session_id"] = "old"
    button(app, "Start New Research").click().run()
    assert app.session_state["active_session_id"] is None


def test_history_shows_research_cards() -> None:
    rows = [session("completed", id="a", title="Report A"),
            session("awaiting_approval", id="b"), session("failed", id="c")]
    app = run_page("history", FakeClient(research=session("completed"), list_research=rows))
    assert not app.exception
    page = texts(app)
    assert "Total research" in page and "Waiting for approval" in page
    assert "Report A" in page and "Findings are ready for your review." in page
    assert "Completed" in page and "Waiting for Approval" in page and "Failed" in page
    assert "10 sources" in page and "1 revision" in page
    assert [b.label for b in app.button].count("Open Research") == 3


def test_history_open_and_delete() -> None:
    rows = [session("completed", id="a", title="Report A"), session("failed", id="b")]
    fake = FakeClient(research=session("completed"), list_research=rows)
    app = run_page("history", fake)
    app.button(key="confirm_delete_b").click().run()
    assert fake.called("delete_research") == [("delete_research", "b")]
    app.button(key="open_a").click().run()
    assert app.session_state["active_session_id"] == "a"


# --------------------------------------------------------------------------- knowledge base


def test_knowledge_base_empty_state() -> None:
    app = run_page("knowledge", FakeClient())
    assert "No documents yet" in texts(app)
    assert button(app, "Add to Knowledge Base").disabled  # nothing selected yet


def test_knowledge_base_document_cards() -> None:
    fake = FakeClient(list_documents=DOCS)
    app = run_page("knowledge", fake)
    assert not app.exception
    page = texts(app)
    assert "notes.pdf" in page and "scan.pdf" in page and "Ready" in page and "Needs attention" in page
    assert "Searchable passages" in page and "7 passages" in page
    assert "chroma" not in page.lower() and "embedding" not in page.lower()
    assert not any(b.key == "proc_d1" for b in app.button)  # only unprocessed docs can be re-processed
    app.button(key="proc_d2").click().run()
    assert fake.called("process_document") == [("process_document", "d2")]
    app.button(key="del_d1").click().run()
    assert fake.called("delete_document") == [("delete_document", "d1")]


# --------------------------------------------------------------------------- settings


def test_settings_is_user_facing() -> None:
    fake = FakeClient(list_documents=DOCS)
    app = run_page("settings", fake)
    assert not app.exception
    page = texts(app)
    for expected in ("AI Model", "Gemini 3.5 Flash", "Knowledge Base", "1 document available to search",
                     "System Status", "All research features are available", "About ResearchPilot",
                     "Multi-agent AI research and report generation system.", "Version 1.0"):
        assert expected in page, expected
    for technical in TECHNICAL_TERMS:
        assert technical.lower() not in page.lower(), technical
    assert not any(b.label in ("Test with one request", "Check availability") for b in app.button)
    assert not fake.called("model_check")  # never spends Gemini quota


@pytest.mark.parametrize("config", [CONFIG | {"web_research_enabled": False}, CONFIG | {"research_enabled": False}])
def test_settings_status_is_simple(config: dict) -> None:
    assert "Limited mode" in texts(run_page("settings", FakeClient(config=config)))


def test_settings_offline_does_not_crash() -> None:
    app = run_page("settings", FakeClient(offline=True))
    assert not app.exception and "Offline" in texts(app)


# --------------------------------------------------------------------------- whole app & offline


@pytest.mark.parametrize("page", ["research", "history", "knowledge"])
def test_backend_offline_shows_friendly_error(page: str) -> None:
    app = run_page(page, FakeClient(offline=True), active_session="s1" if page == "research" else None)
    assert not app.exception
    assert any("research service isn't reachable" in e.value or "Cannot reach the ResearchPilot backend" in e.value
               for e in app.error)


def test_full_app_boots_with_navigation_and_status_card() -> None:
    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT)
    app.session_state["api_client"] = FakeClient()
    app.run()
    assert not app.exception
    assert "ResearchPilot" in texts(app)
    sidebar = "\n".join(m.value for m in app.sidebar.markdown)
    assert "System Ready" in sidebar and "AI research workspace is online" in sidebar


def test_full_app_shows_offline_status() -> None:
    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT)
    app.session_state["api_client"] = FakeClient(offline=True)
    app.run()
    assert not app.exception
    assert "Offline" in "\n".join(m.value for m in app.sidebar.markdown)


def test_timestamps_are_shown_in_local_time_and_counts_pluralised() -> None:
    from datetime import UTC, datetime

    from frontend.components import local_time, plural

    expected = datetime(2026, 10, 2, 18, 30, tzinfo=UTC).astimezone().strftime("%d %b %Y, %H:%M")
    assert local_time("2026-10-02T18:30:00") == expected  # naive backend value treated as UTC
    assert local_time("2026-10-02T18:30:00+00:00") == expected
    assert local_time("not a date") == "not a date"
    assert plural(1, "passage") == "1 passage" and plural(7, "passage") == "7 passages"


def test_user_text_is_escaped_in_html_components() -> None:
    from frontend.ui.components import research_card_html, source_item_html

    card = research_card_html("<script>x</script>", "a & b", "today", "completed", 1, 0)
    assert "<script>" not in card and "&lt;script&gt;" in card and "a &amp; b" in card
    source = source_item_html({"id": "W1", "title": "<b>t</b>", "reference": "javascript:alert(1)",
                               "source_type": "web", "verification_status": "unverified"})
    assert "<b>t</b>" not in source and "href" not in source  # non-http references are not links

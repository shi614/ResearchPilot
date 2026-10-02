"""ResearchService with the real graph, real thread pool and SQLite; scripted LLM."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.agents.base import AgentDependencies
from app.database import Database, ResearchRepository
from app.database.orm import EventStatus, SessionStatus
from app.exceptions import ConfigurationError, NotFoundError, RateLimitError, WorkflowError
from app.graph.builder import build_research_graph
from app.graph.checkpoint import create_checkpointer
from app.graph.runner import ResearchRunner
from app.models.agent_outputs import HumanDecision
from app.services.research_service import RESTART_MESSAGE, ResearchService
from tests.conftest import make_settings
from tests.fakes import FakeKnowledge, ScriptedLLM, analysis, fake_web, happy_llm, kb_chunk, plan

APPROVE = HumanDecision(action="approve")


class Harness:
    """A service wired to the real graph; `llm` is the scripted Gemini stand-in."""

    def __init__(self, tmp_path: Path, llm: ScriptedLLM, **settings_overrides) -> None:
        self.llm = llm
        self.settings = make_settings(tmp_path, gemini_api_key="test-key", **settings_overrides)
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_url)
        self.database.create_tables()
        self.service = self.new_service()

    def new_service(self) -> ResearchService:
        def runner_factory(on_update):
            deps = AgentDependencies(llm=self.llm, settings=self.settings, web=fake_web(),
                                     knowledge=FakeKnowledge([kb_chunk()]))
            graph = build_research_graph(deps, create_checkpointer(self.settings.checkpoint_db))
            return ResearchRunner(graph, on_update=on_update)

        return ResearchService(self.settings, self.database, runner_factory)

    def run(self, method: str, *args):
        response = getattr(self.service, method)(*args)
        self.service.wait(response.id, timeout=30)
        return self.service.get(response.id)


@pytest.fixture
def harness(tmp_path: Path):
    h = Harness(tmp_path, happy_llm())
    yield h
    h.service.shutdown()
    h.database.dispose()


def test_start_runs_in_background_until_approval(harness: Harness) -> None:
    started = harness.service.start("  How have solar costs changed?  ", "  ")
    assert started.status in (SessionStatus.RUNNING, SessionStatus.AWAITING_APPROVAL)
    assert started.query == "How have solar costs changed?" and started.instructions is None
    harness.service.wait(started.id, timeout=30)

    session = harness.service.get(started.id)
    assert session.status is SessionStatus.AWAITING_APPROVAL
    assert session.approval_request["sources"]["total"] == 5
    assert session.pending_nodes == ["human_review"]
    assert session.stats.llm_calls == 5 and session.stats.web_searches == 2
    assert session.stats.evidence_web == 4 and session.stats.evidence_knowledge_base == 1
    nodes = [e.node for e in session.events]
    assert nodes[0] == "planner" and {"web_research", "rag_research", "analysis"} <= set(nodes)
    assert all(e.status is EventStatus.COMPLETED for e in session.events)


def test_events_can_be_polled_incrementally(harness: Harness) -> None:
    session = harness.run("start", "q")
    third = session.events[2].id
    newer = harness.service.get(session.id, after_event=third).events
    assert [e.id for e in newer] == [e.id for e in session.events[3:]]


def test_approval_completes_and_saves_report(harness: Harness) -> None:
    session = harness.run("start", "q")
    done = harness.run("decide", session.id, APPROVE)
    assert done.status is SessionStatus.COMPLETED and done.has_report
    assert done.title == "Solar Energy Cost Trends" and done.approval_request is None
    assert done.events[-1].node == "finalize"

    report = harness.service.report(session.id)
    assert report.title == "Solar Energy Cost Trends"
    assert report.citations and "## References" in report.markdown
    assert harness.service.list_sessions()[0].has_report is True


def test_modify_loops_back_to_research(tmp_path: Path) -> None:
    llm = happy_llm(ResearchPlan=[plan(), plan(queries=["solar india"])])
    h = Harness(tmp_path, llm)
    session = h.run("start", "q")
    again = h.run("decide", session.id, HumanDecision(action="modify", feedback="Focus on India"))
    assert again.status is SessionStatus.AWAITING_APPROVAL
    assert again.approval_request["iteration"] == 2 and again.stats.iteration == 2
    h.service.shutdown()


def test_cancel_ends_without_report(harness: Harness) -> None:
    session = harness.run("start", "q")
    cancelled = harness.run("decide", session.id, HumanDecision(action="cancel"))
    assert cancelled.status is SessionStatus.CANCELLED and not cancelled.has_report
    with pytest.raises(NotFoundError):
        harness.service.report(session.id)


def test_quota_stop_is_retryable(tmp_path: Path) -> None:
    llm = happy_llm(AnalysisNotes=[RateLimitError("Gemini", "daily quota exhausted"), analysis])
    h = Harness(tmp_path, llm)
    stopped = h.run("start", "q")
    assert stopped.status is SessionStatus.QUOTA_EXHAUSTED and stopped.retryable
    assert "quota" in stopped.error and stopped.pending_nodes == ["analysis"]

    resumed = h.run("retry", stopped.id)
    assert resumed.status is SessionStatus.AWAITING_APPROVAL and resumed.error is None
    assert llm.calls_for("planning") == 1
    h.service.shutdown()


def test_restart_marks_running_sessions_interrupted_and_they_resume(tmp_path: Path) -> None:
    llm = happy_llm(ResearchPlan=[RateLimitError("Gemini", "quota"), plan()])
    h = Harness(tmp_path, llm)
    session = h.run("start", "q")  # stops at the planner, checkpoint saved
    with h.database.session() as db:  # simulate the process dying mid-run
        ResearchRepository(db).update_status(session.id, SessionStatus.RUNNING)

    h.service.shutdown()
    h.service = h.new_service()  # "backend restart"
    assert h.service.recover_interrupted() == 1
    recovered = h.service.get(session.id)
    assert recovered.status is SessionStatus.INTERRUPTED and recovered.error == RESTART_MESSAGE
    assert recovered.retryable

    assert h.run("retry", session.id).status is SessionStatus.AWAITING_APPROVAL
    h.service.shutdown()


def test_invalid_transitions_are_rejected(harness: Harness) -> None:
    session = harness.run("start", "q")
    with pytest.raises(WorkflowError, match="cannot be retried"):
        harness.service.retry(session.id)
    harness.run("decide", session.id, APPROVE)
    with pytest.raises(WorkflowError, match="not waiting for approval"):
        harness.service.decide(session.id, APPROVE)
    with pytest.raises(WorkflowError, match="cannot be retried"):
        harness.service.retry(session.id)
    with pytest.raises(NotFoundError):
        harness.service.get("missing")


def test_one_operation_per_session_at_a_time(tmp_path: Path) -> None:
    gate, entered = threading.Event(), threading.Event()

    def slow_plan(_prompt: str):
        entered.set()
        gate.wait(timeout=10)
        return plan()

    h = Harness(tmp_path, happy_llm(ResearchPlan=[slow_plan]))
    session = h.service.start("q")
    assert entered.wait(timeout=10)
    assert h.service.get(session.id).status is SessionStatus.RUNNING
    with pytest.raises(WorkflowError, match="already running"):
        h.service.decide(session.id, APPROVE)
    with pytest.raises(WorkflowError, match="still running"):
        h.service.delete(session.id)
    gate.set()
    h.service.wait(session.id, timeout=30)
    assert h.service.get(session.id).status is SessionStatus.AWAITING_APPROVAL
    h.service.shutdown()


def test_unexpected_background_error_marks_session_failed(tmp_path: Path) -> None:
    h = Harness(tmp_path, happy_llm())

    class BrokenRunner:
        def start(self, *_args):
            raise RuntimeError("boom")

        def outcome(self, *_args):
            raise ConfigurationError("unavailable")

    h.service = ResearchService(h.settings, h.database, lambda _cb: BrokenRunner())
    session = h.run("start", "q")
    assert session.status is SessionStatus.FAILED and session.error == "Unexpected error: RuntimeError"
    h.service.shutdown()


def test_missing_gemini_key_refuses_to_start(tmp_path: Path) -> None:
    h = Harness(tmp_path, happy_llm())
    h.settings = make_settings(tmp_path)  # no keys
    h.service = h.new_service()
    with pytest.raises(ConfigurationError):
        h.service.start("q")
    assert h.service.list_sessions() == []


def test_delete_removes_session_events_report_and_checkpoint(harness: Harness) -> None:
    session = harness.run("start", "q")
    harness.run("decide", session.id, APPROVE)
    harness.service.delete(session.id)
    with pytest.raises(NotFoundError):
        harness.service.get(session.id)
    assert harness.service.runner().outcome(session.id).values == {}

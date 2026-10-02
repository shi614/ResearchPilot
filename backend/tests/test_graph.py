"""Workflow tests: the real compiled LangGraph with scripted LLM responses."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.base import AgentDependencies
from app.exceptions import RateLimitError, WorkflowError
from app.graph.builder import build_research_graph
from app.graph.checkpoint import create_checkpointer
from app.graph.runner import ResearchRunner
from app.models.agent_outputs import HumanDecision
from tests.fakes import (
    FakeKnowledge,
    FakeSearchTool,
    ScriptedLLM,
    agent_settings,
    analysis,
    clean_critique,
    critique_with,
    fake_web,
    happy_llm,
    kb_chunk,
    plan,
    tool_call_turn,
)

APPROVE = HumanDecision(action="approve")


def make_runner(llm: ScriptedLLM, *, checkpointer=None, search: FakeSearchTool | None = None,
                knowledge: FakeKnowledge | None = None, events: list | None = None, **settings):
    deps = AgentDependencies(
        llm=llm,
        settings=agent_settings(**settings),
        web=fake_web(search),
        knowledge=knowledge if knowledge is not None else FakeKnowledge([kb_chunk()]),
    )
    graph = build_research_graph(deps, checkpointer or InMemorySaver())
    on_update = (lambda _t, node, _u: events.append(node)) if events is not None else None
    return ResearchRunner(graph, on_update=on_update)


def test_full_run_pauses_for_approval_then_completes() -> None:
    llm, events = happy_llm(), []
    runner = make_runner(llm, events=events)

    paused = runner.start("t1", "How have solar costs changed?")
    assert paused.status == "awaiting_approval"
    request = paused.approval_request
    assert request["sources"] == {"web": 4, "knowledge_base": 1, "total": 5}
    assert request["research_questions"] == ["How have solar costs changed?", "What drives adoption?"]
    assert request["key_findings"] == ["Costs fell"]  # finding citing unknown W999 was dropped
    assert request["can_modify"] is True
    # planner 1 + web tool-calling 2 + verification 1 + analysis 1; RAG uses no LLM calls
    assert llm.call_count == 5 and paused.values["llm_calls"] == 5
    assert {"web_research", "rag_research"} <= set(events)
    assert llm.calls_for("source verification") == 1  # fan-in: verified once after both branches

    done = runner.resume("t1", APPROVE)
    assert done.status == "completed"
    assert llm.call_count == 7  # + writer + critic; clean critique → no revision
    values = done.values
    assert values["revision_count"] == 0
    cited = {c.id for c in values["citations"]}
    evidence_ids = {e.source.id for e in values["evidence"]}
    assert cited and cited <= evidence_ids
    assert values["final_report"].title == "Solar Energy Cost Trends"
    assert events[-1] == "finalize"


def test_critic_triggers_exactly_one_revision_then_finalizes() -> None:
    llm = happy_llm(Critique=[critique_with("high")])  # critic keeps complaining
    runner = make_runner(llm, max_revisions=1)
    runner.start("t1", "q")
    done = runner.resume("t1", APPROVE)
    assert done.status == "completed"
    assert llm.calls_for("critique") == 2 and llm.calls_for("revision") == 1
    assert done.values["revision_count"] == 1
    assert any("Claim not supported" in n for n in done.values["quality_notes"])


def test_revision_loop_resolves_when_critique_is_clean_the_second_time() -> None:
    llm = happy_llm(Critique=[critique_with("high"), clean_critique()])
    runner = make_runner(llm, max_revisions=1)
    runner.start("t1", "q")
    done = runner.resume("t1", APPROVE)
    assert llm.calls_for("revision") == 1 and done.values["needs_revision"] is False


def test_zero_revision_budget_never_revises() -> None:
    llm = happy_llm(Critique=[critique_with("high")])
    runner = make_runner(llm, max_revisions=0)
    runner.start("t1", "q")
    assert runner.resume("t1", APPROVE).status == "completed"
    assert llm.calls_for("revision") == 0 and llm.calls_for("critique") == 1


def test_low_severity_issues_do_not_trigger_revision() -> None:
    llm = happy_llm(Critique=[critique_with("low")])
    runner = make_runner(llm)
    runner.start("t1", "q")
    runner.resume("t1", APPROVE)
    assert llm.calls_for("revision") == 0


def test_modify_research_replans_and_limits_iterations() -> None:
    llm = happy_llm(ResearchPlan=[plan(), plan(queries=["solar india"])])
    llm.tool_turns.append(tool_call_turn("solar india"))
    search = FakeSearchTool()
    runner = make_runner(llm, search=search, max_research_iterations=2)
    runner.start("t1", "q")

    second = runner.resume("t1", HumanDecision(action="modify", feedback="Focus on India"))
    assert second.status == "awaiting_approval"
    assert second.approval_request["iteration"] == 2 and second.approval_request["can_modify"] is False
    assert "Focus on India" in llm.prompts["planning"][1]
    assert search.queries == ["solar cost trends", "solar adoption drivers", "solar india"]
    assert second.approval_request["sources"]["web"] == 6  # earlier sources are kept

    # A further modify request beyond the limit is treated as approval.
    done = runner.resume("t1", HumanDecision(action="modify", feedback="more"))
    assert done.status == "completed" and llm.calls_for("planning") == 2


def test_cancel_stops_without_writing() -> None:
    llm = happy_llm()
    runner = make_runner(llm)
    runner.start("t1", "q")
    assert runner.resume("t1", HumanDecision(action="cancel")).status == "cancelled"
    assert llm.calls_for("report writing") == 0


def test_no_sources_anywhere_fails_cleanly_without_wasting_calls() -> None:
    llm = happy_llm()
    runner = make_runner(llm, search=FakeSearchTool(empty=True), knowledge=FakeKnowledge([]))
    outcome = runner.start("t1", "q")
    assert outcome.status == "failed"
    assert "No sources" in outcome.error
    assert llm.calls_for("analysis") == 0 and llm.calls_for("source verification") == 0


def test_runs_without_tavily_key_use_knowledge_base_only() -> None:
    llm = happy_llm()
    deps = AgentDependencies(llm=llm, settings=agent_settings(), web=None, knowledge=FakeKnowledge([kb_chunk()]))
    runner = ResearchRunner(build_research_graph(deps, InMemorySaver()))
    outcome = runner.start("t1", "q")
    assert outcome.status == "awaiting_approval"
    assert outcome.approval_request["sources"] == {"web": 0, "knowledge_base": 1, "total": 1}
    assert llm.calls_for("web research") == 0


def test_quota_error_pauses_run_and_retry_resumes_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.db"
    llm = happy_llm(AnalysisNotes=[RateLimitError("Gemini", "daily quota exhausted"), analysis])
    first = make_runner(llm, checkpointer=create_checkpointer(db))
    stopped = first.start("t1", "q")
    assert stopped.status == "quota_exhausted" and "quota" in stopped.error
    assert stopped.pending_nodes == ("analysis",)
    calls_before = llm.call_count

    # Simulate a new process: fresh graph + checkpointer on the same SQLite file.
    second = make_runner(llm, checkpointer=create_checkpointer(db))
    resumed = second.retry("t1")
    assert resumed.status == "awaiting_approval"
    assert llm.calls_for("planning") == 1  # completed steps were not repeated
    assert llm.call_count == calls_before + 1  # only the analysis call was retried
    assert second.resume("t1", APPROVE).status == "completed"


def test_parallel_branch_failure_keeps_other_branch_results() -> None:
    knowledge = FakeKnowledge([kb_chunk()], error=RuntimeError("chroma crashed"))
    outcome = make_runner(happy_llm(), knowledge=knowledge).start("t1", "q")
    # The KB branch failed unexpectedly; the web branch's results carry the run forward.
    assert outcome.status == "awaiting_approval"
    assert outcome.approval_request["sources"] == {"web": 4, "knowledge_base": 0, "total": 4}
    assert any("RuntimeError" in e for e in outcome.approval_request["errors"])


def test_unexpected_error_in_core_agent_fails_safely() -> None:
    llm = happy_llm(AnalysisNotes=[RuntimeError("bug")])
    outcome = make_runner(llm).start("t1", "q")
    assert outcome.status == "failed" and outcome.error == "Unexpected error: RuntimeError"


def test_runner_guards_invalid_transitions() -> None:
    runner = make_runner(happy_llm())
    with pytest.raises(WorkflowError):
        runner.resume("missing", APPROVE)
    runner.start("t1", "q")
    with pytest.raises(WorkflowError):
        runner.start("t1", "q")
    with pytest.raises(WorkflowError):
        runner.retry("t1")  # waiting for approval, not failed
    runner.resume("t1", APPROVE)
    with pytest.raises(WorkflowError):
        runner.retry("t1")  # finished


def test_sqlite_checkpoints_restore_typed_state_without_warnings(tmp_path: Path, caplog, recwarn) -> None:
    db = tmp_path / "cp.db"
    make_runner(happy_llm(), checkpointer=create_checkpointer(db)).start("t1", "q")
    caplog.set_level(logging.WARNING)
    restored = make_runner(happy_llm(), checkpointer=create_checkpointer(db)).outcome("t1")
    assert restored.status == "awaiting_approval"
    assert type(restored.values["research_plan"]).__name__ == "ResearchPlan"
    assert type(restored.values["evidence"][0]).__name__ == "Evidence"
    messages = [r.getMessage() for r in caplog.records] + [str(w.message) for w in recwarn]
    assert not any("unregistered" in m.lower() for m in messages)


def test_writer_outage_stops_resumably_and_retry_finishes(tmp_path: Path) -> None:
    from app.exceptions import ExternalServiceError
    from tests.fakes import report

    db = tmp_path / "cp.db"
    llm = happy_llm(ResearchReport=[ExternalServiceError("Gemini", "Gemini service error (503)."), report])
    runner = make_runner(llm, checkpointer=create_checkpointer(db))
    runner.start("t1", "q")
    stopped = runner.resume("t1", APPROVE)
    assert stopped.status == "failed" and "503" in stopped.error
    assert stopped.pending_nodes == ("writer",) and stopped.retryable

    resumed = make_runner(llm, checkpointer=create_checkpointer(db)).retry("t1")
    assert resumed.status == "completed" and not resumed.retryable
    assert llm.calls_for("planning") == 1 and llm.calls_for("report writing") == 2

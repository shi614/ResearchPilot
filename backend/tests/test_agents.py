"""Unit tests for individual agent nodes (scripted LLM, stubbed Tavily, fake KB)."""

from __future__ import annotations

import pytest

from app.agents.base import AgentDependencies
from app.agents.critic import check_citations, needs_revision
from app.agents.formatting import extract_citation_ids
from app.agents.planner import PlannerAgent
from app.agents.rag_researcher import KnowledgeResearchAgent
from app.agents.source_verifier import SourceVerificationAgent, build_evidence
from app.agents.web_researcher import WebResearchAgent
from app.exceptions import ExternalServiceError, RateLimitError
from app.graph.nodes import finalize
from app.models.agent_outputs import (
    ClaimCheck,
    CritiqueIssue,
    ResearchPlan,
    SourceAssessment,
    VerificationResult,
)
from app.models.domain import Evidence, Source, SourceType, VerificationStatus
from tests.fakes import (
    FakeExtractTool,
    FakeKnowledge,
    FakeSearchTool,
    ScriptedLLM,
    agent_settings,
    fake_web,
    kb_chunk,
    plan,
    report,
    tool_call_turn,
)


def deps(llm: ScriptedLLM, **kwargs) -> AgentDependencies:
    return AgentDependencies(llm=llm, settings=kwargs.pop("settings", agent_settings()), **kwargs)


def web_source(n: int, url: str | None = None) -> Source:
    return Source(id=f"W{n}", source_type=SourceType.WEB, title=f"T{n}", content=f"content {n}",
                  query="q", url=url or f"https://s{n}.example")


# --------------------------------------------------------------------------- planner


def test_planner_dedupes_caps_and_respects_kb_availability() -> None:
    raw = plan(use_kb=True, queries=["a", "A ", "b", "c", "d", "e"])
    raw.research_questions = ["q1", "q1", "q2", "q3", "q4", "q5"]
    llm = ScriptedLLM({ResearchPlan: [raw]})
    update = PlannerAgent(deps(llm, knowledge=FakeKnowledge([]), web=fake_web()))({"user_query": "x"})
    assert update["research_plan"].search_queries == ["a", "b", "c", "d"]
    assert update["research_questions"] == ["q1", "q2", "q3", "q4"]
    assert update["research_plan"].use_knowledge_base is False  # KB empty → overridden
    assert update["iteration_count"] == 1 and update["llm_calls"] == 1


def test_planner_falls_back_when_llm_fails() -> None:
    llm = ScriptedLLM({ResearchPlan: [ExternalServiceError("Gemini", "malformed")]})
    update = PlannerAgent(deps(llm, knowledge=FakeKnowledge([kb_chunk()])))({"user_query": "EV batteries"})
    assert update["research_plan"].search_queries == ["EV batteries"]
    assert update["research_plan"].use_knowledge_base is True
    assert "fallback" in update["errors"][0].message


def test_planner_lets_quota_errors_stop_the_run() -> None:
    llm = ScriptedLLM({ResearchPlan: [RateLimitError("Gemini", "quota")]})
    with pytest.raises(RateLimitError):
        PlannerAgent(deps(llm))({"user_query": "x"})


def test_replanning_prompt_includes_feedback_and_previous_work() -> None:
    llm = ScriptedLLM({ResearchPlan: [plan()]})
    state = {"user_query": "x", "human_feedback": "Focus on India", "iteration_count": 1,
             "research_questions": ["old question"], "searched_queries": ["old query"]}
    update = PlannerAgent(deps(llm))(state)
    prompt = llm.prompts["planning"][0]
    assert "Focus on India" in prompt and "old question" in prompt and "old query" in prompt
    assert update["iteration_count"] == 2


# --------------------------------------------------------------------------- web research


def test_web_agent_executes_llm_tool_calls_and_numbers_sources() -> None:
    search, extract = FakeSearchTool(), FakeExtractTool()
    llm = ScriptedLLM(tool_turns=[
        tool_call_turn("solar cost trends", "solar adoption drivers"),
        tool_call_turn(extract=["https://example.org/solar-cost-trends"]),
    ])
    state = {"user_query": "x", "research_plan": plan(), "web_sources": [web_source(1)]}
    update = WebResearchAgent(deps(llm, web=fake_web(search, extract)))(state)

    assert search.queries == ["solar cost trends", "solar adoption drivers"]
    assert [s.id for s in update["web_sources"]] == ["W2", "W3", "W4", "W5"]  # continues numbering
    assert extract.calls == [["https://example.org/solar-cost-trends"]]
    assert "Full text of" in update["web_sources"][0].content
    assert update["web_searches"] == 2 and update["llm_calls"] == 2
    assert update["searched_queries"] == ["solar cost trends", "solar adoption drivers"]


def test_web_agent_enforces_search_budget_and_skips_repeats() -> None:
    search = FakeSearchTool()
    llm = ScriptedLLM(tool_turns=[tool_call_turn("a", "old", "b", "c")])
    state = {"user_query": "x", "research_plan": plan(), "searched_queries": ["old"]}
    update = WebResearchAgent(deps(llm, web=fake_web(search), settings=agent_settings(max_web_searches=2)))(state)
    assert search.queries == ["a", "b"]
    assert update["web_searches"] == 2


def test_web_agent_falls_back_to_planned_queries_without_tool_calls() -> None:
    search = FakeSearchTool()
    llm = ScriptedLLM(tool_turns=[ExternalServiceError("Gemini", "tool calling unsupported")])
    update = WebResearchAgent(deps(llm, web=fake_web(search)))({"user_query": "x", "research_plan": plan()})
    assert search.queries == ["solar cost trends", "solar adoption drivers"]
    assert len(update["web_sources"]) == 4
    assert "planned queries" in update["progress"][-1].message


def test_tavily_failure_degrades_instead_of_stopping() -> None:
    search = FakeSearchTool(fail_with={"error": ValueError("Error 432: credit limit")})
    llm = ScriptedLLM(tool_turns=[tool_call_turn("solar")])
    update = WebResearchAgent(deps(llm, web=fake_web(search)))({"user_query": "x", "research_plan": plan()})
    assert update["web_sources"] == []
    assert any("credit limit" in e.message or "rate limit" in e.message for e in update["errors"])


# --------------------------------------------------------------------------- RAG


def test_rag_agent_creates_cited_kb_sources_without_duplicates() -> None:
    knowledge = FakeKnowledge([kb_chunk("chunk A"), kb_chunk("chunk B", page=None)])
    update = KnowledgeResearchAgent(deps(ScriptedLLM(), knowledge=knowledge))(
        {"user_query": "x", "research_questions": ["q1", "q2"]}
    )
    sources = update["retrieved_documents"]
    assert [s.id for s in sources] == ["K1", "K2"]  # same chunks for q2 are de-duplicated
    assert sources[0].reference == "internal.pdf, p. 3" and sources[1].reference == "internal.pdf"
    assert knowledge.queries == ["q1", "q2"]


def test_rag_failure_is_recorded_not_raised() -> None:
    knowledge = FakeKnowledge([kb_chunk()], error=RateLimitError("Gemini", "embedding quota"))
    update = KnowledgeResearchAgent(deps(ScriptedLLM(), knowledge=knowledge))(
        {"user_query": "x", "research_questions": ["q1"]}
    )
    assert update["retrieved_documents"] == [] and "embedding quota" in update["errors"][0].message


# --------------------------------------------------------------------------- verification


def test_build_evidence_statuses_and_exclusions() -> None:
    sources = [web_source(i) for i in range(1, 5)]
    result = VerificationResult(
        assessments=[
            SourceAssessment(source_id="W1", relevance="high", reliability="high", reason=""),
            SourceAssessment(source_id="W2", relevance="high", reliability="medium", reason=""),
            SourceAssessment(source_id="W3", relevance="low", reliability="high", reason="off-topic"),
            SourceAssessment(source_id="W99", relevance="high", reliability="high", reason="hallucinated"),
        ],
        claims=[
            ClaimCheck(claim="A", source_ids=["W1", "W2"], status="corroborated"),
            ClaimCheck(claim="B", source_ids=["W2", "W4"], status="conflicting"),
            ClaimCheck(claim="C", source_ids=["W4", "W99"], status="corroborated"),  # only 1 real source
        ],
        conflicts=["W2 vs W4"],
        summary="",
    )
    evidence = {e.source.id: e for e in build_evidence(sources, result)}
    assert set(evidence) == {"W1", "W2", "W4"}  # W3 excluded (low relevance)
    assert evidence["W1"].status is VerificationStatus.CORROBORATED
    assert evidence["W2"].status is VerificationStatus.CONFLICTING  # conflict outweighs support
    assert evidence["W4"].status is VerificationStatus.CONFLICTING
    assert evidence["W4"].reliability == "unknown"  # not assessed


def test_verifier_without_sources_fails_the_run() -> None:
    update = SourceVerificationAgent(deps(ScriptedLLM()))({"user_query": "x"})
    assert update["status"] == "failed" and update["evidence"] == []


def test_verifier_failure_marks_sources_unverified() -> None:
    llm = ScriptedLLM({VerificationResult: [ExternalServiceError("Gemini", "malformed")]})
    update = SourceVerificationAgent(deps(llm))({"user_query": "x", "web_sources": [web_source(1)]})
    assert [e.status for e in update["evidence"]] == [VerificationStatus.UNVERIFIED]


# --------------------------------------------------------------------------- critic & finalize


def test_citation_parsing_supports_grouped_ids() -> None:
    assert extract_citation_ids("A [W1]. B [W2, K1]; C [W1; W3]. D [note]") == ["W1", "W2", "K1", "W3"]


def test_deterministic_citation_checks() -> None:
    draft = report("Evidence [W1] [W2]", extra_citation="[W9]")
    draft.key_findings.append("Uncited finding")
    issues = check_citations(draft, {"W1", "W2"})
    categories = {(i.category, i.severity) for i in issues}
    assert ("invalid_citation", "high") in categories
    assert ("missing_citation", "medium") in categories
    assert needs_revision(issues)


def test_revision_rule() -> None:
    def issue(severity: str) -> CritiqueIssue:
        return CritiqueIssue(severity=severity, category="other", location="x", description="d", suggestion="s")

    assert not needs_revision([issue("medium"), issue("medium"), issue("low")])
    assert needs_revision([issue("medium")] * 3)
    assert needs_revision([issue("high")])


def test_finalize_removes_unknown_citations_and_builds_references() -> None:
    evidence = [
        Evidence(source=web_source(1), relevance="high", reliability="high", status=VerificationStatus.CORROBORATED),
        Evidence(source=web_source(2), relevance="high", reliability="low", status=VerificationStatus.SINGLE_SOURCE),
    ]
    draft = report("Evidence [W1] [W2]", extra_citation="[W7]")
    draft.conclusion = "Grouped [W2, W8]."
    result = finalize({"draft_report": draft, "evidence": evidence})
    final = result["final_report"]
    assert "[W7]" not in final.key_findings[0] and "citation removed" in final.key_findings[0]
    assert final.conclusion == "Grouped [W2]."
    assert [c.id for c in result["citations"]] == ["W1", "W2"]
    assert result["citations"][1].verification_status is VerificationStatus.SINGLE_SOURCE
    assert any("not corroborated" in note for note in result["quality_notes"])
    assert result["status"] == "completed"

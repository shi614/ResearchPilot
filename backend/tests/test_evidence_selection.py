"""Selection of the strongest evidence for the writer/critic/reviser."""

from __future__ import annotations

from app.agents.base import AgentDependencies
from app.agents.critic import CriticAgent
from app.agents.evidence_selection import report_evidence, select_evidence
from app.agents.writer import WriterAgent
from app.graph.nodes import finalize
from app.models.agent_outputs import Critique, ResearchReport
from app.models.domain import Evidence, Source, SourceType, VerificationStatus
from tests.fakes import ScriptedLLM, agent_settings, ids_in, report

C, S, X, U = (
    VerificationStatus.CORROBORATED,
    VerificationStatus.SINGLE_SOURCE,
    VerificationStatus.CONFLICTING,
    VerificationStatus.UNVERIFIED,
)


def web(n: int, status=C, relevance="high", reliability="high", score=0.5, domain: str | None = None) -> Evidence:
    source = Source(id=f"W{n}", source_type=SourceType.WEB, title=f"Web {n}", content=f"web content {n}",
                    query="q", url=f"https://{domain or f'site{n}.org'}/page{n}", score=score)
    return Evidence(source=source, relevance=relevance, reliability=reliability, status=status)


def kb(n: int, status=S, relevance="medium") -> Evidence:
    source = Source(id=f"K{n}", source_type=SourceType.KNOWLEDGE_BASE, title=f"notes.pdf, p. {n}",
                    content=f"kb content {n}", query="q", document_id="doc1", filename="notes.pdf", page=n, score=0.7)
    return Evidence(source=source, relevance=relevance, reliability="unknown", status=status)


def ids(items: list[Evidence]) -> list[str]:
    return [e.source.id for e in items]


def test_within_limit_keeps_everything_ranked() -> None:
    items = [web(1, status=U), web(2, status=C)]
    assert ids(select_evidence(items, limit=14)) == ["W2", "W1"]


def test_prefers_corroborated_over_order_of_collection() -> None:
    items = [web(i, status=U) for i in range(1, 11)] + [web(i, status=C) for i in range(11, 21)]
    selected = select_evidence(items, limit=10)
    assert set(ids(selected)) == {f"W{i}" for i in range(11, 21)}  # not simply the first ten


def test_ranks_by_status_then_relevance_then_reliability_then_score() -> None:
    items = [
        web(1, status=S, relevance="high", reliability="high", score=0.99),
        web(2, status=C, relevance="medium", reliability="high"),
        web(3, status=C, relevance="high", reliability="medium"),
        web(4, status=C, relevance="high", reliability="high", score=0.2),
        web(5, status=C, relevance="high", reliability="high", score=0.9),
        web(6, status=U, relevance="high", reliability="high", score=1.0),
    ]
    assert ids(select_evidence(items, limit=5)) == ["W5", "W4", "W3", "W2", "W1"]


def test_relevant_knowledge_base_evidence_is_kept_even_when_outranked() -> None:
    items = [web(i) for i in range(1, 21)] + [kb(1), kb(2), kb(3), kb(4)]
    selected = select_evidence(items, limit=12)
    assert len(selected) == 12
    assert sum(e.source.source_type is SourceType.KNOWLEDGE_BASE for e in selected) == 3  # reserved slots


def test_no_single_website_dominates() -> None:
    same_site = [web(i, score=0.9, domain="bigblog.com") for i in range(1, 9)]
    others = [web(i, score=0.1) for i in range(9, 15)]
    selected = select_evidence(same_site + others, limit=8)
    domains = [e.source.url.split("/")[2] for e in selected]
    assert domains.count("bigblog.com") == 3


def test_domain_cap_does_not_leave_slots_empty() -> None:
    items = [web(i, domain="onlysite.com") for i in range(1, 21)]
    assert len(select_evidence(items, limit=12)) == 12


def test_selection_never_changes_source_ids() -> None:
    items = [web(i, status=[C, S, X, U][i % 4]) for i in range(1, 31)] + [kb(1)]
    selected = select_evidence(items, limit=14)
    assert len(selected) == 14 and set(ids(selected)) <= set(ids(items))
    assert len(set(ids(selected))) == 14  # no duplicates


def test_report_evidence_filters_by_selected_ids() -> None:
    items = [web(1), web(2), kb(1)]
    assert ids(report_evidence(items, ["K1", "W1"])) == ["W1", "K1"]
    assert report_evidence(items, None) == items  # runs written before selection existed


def test_writer_critic_and_finalize_use_only_selected_sources() -> None:
    evidence = [web(i, status=U, score=0.1) for i in range(1, 21)] + [web(i) for i in range(21, 32)] + [kb(1)]
    llm = ScriptedLLM({ResearchReport: [report], Critique: [Critique(issues=[], overall_assessment="ok")]})
    deps = AgentDependencies(llm=llm, settings=agent_settings(writer_max_sources=12))
    state = {"user_query": "q", "research_questions": ["q"], "evidence": evidence, "web_searches": 3}

    update = WriterAgent(deps)(state)
    selected_ids = update["report_evidence_ids"]
    assert len(selected_ids) == 12 and "K1" in selected_ids
    assert not any(f"W{i}" in selected_ids for i in range(1, 21))  # unverified sources left out
    writer_prompt = llm.prompts["report writing"][0]
    assert set(ids_in(writer_prompt.split("Evidence (cite")[-1])) == set(selected_ids)
    assert "the 12 strongest were used" in writer_prompt

    # A citation to a real but unselected source is invalid: the writer never saw it.
    draft = update["draft_report"]
    draft.conclusion = "Unselected [W1]."
    state = {**state, "draft_report": draft, "report_evidence_ids": selected_ids}
    critique = CriticAgent(deps)(state)["critique"]
    assert any(i.category == "invalid_citation" and "W1" in i.description for i in critique.issues)
    final = finalize(state)
    assert "W1" not in {c.id for c in final["citations"]}
    assert {c.id for c in final["citations"]} <= set(selected_ids)

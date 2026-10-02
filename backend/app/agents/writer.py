"""Writer agent: produces the structured research report from verified evidence."""

from __future__ import annotations

from typing import Any

from app.agents import prompts
from app.agents.base import AgentDependencies, CallCounter, merge, progress
from app.agents.evidence_selection import select_evidence
from app.agents.formatting import cited_ids, format_analysis, format_evidence
from app.graph.state import ResearchState
from app.models.agent_outputs import ResearchReport
from app.models.domain import Evidence, SourceType

NODE = "writer"


def methodology_facts(state: ResearchState, selected: list[Evidence]) -> str:
    """Facts about how the research was actually performed (so the methodology is truthful)."""
    collected = state.get("evidence", [])
    web = sum(e.source.source_type is SourceType.WEB for e in selected)
    kb = sum(e.source.source_type is SourceType.KNOWLEDGE_BASE for e in selected)
    statuses: dict[str, int] = {}
    for item in selected:
        statuses[item.status.value] = statuses.get(item.status.value, 0) + 1
    facts = [
        f"- Research iterations: {state.get('iteration_count', 1)}",
        f"- Web searches run: {state.get('web_searches', 0)} (Tavily)",
        f"- Verified sources collected: {len(collected)}; the {len(selected)} strongest were used "
        "(ranked by verification status, relevance and reliability)",
        f"- Evidence used: {web} web sources, {kb} knowledge-base excerpts",
        "- Verification outcome: " + ", ".join(f"{n} {s}" for s, n in sorted(statuses.items())),
        "- Steps: planning, parallel web + knowledge-base research, source verification, "
        "analysis, human approval, drafting, automated critique",
    ]
    if state.get("human_feedback"):
        facts.append(f"- The user asked for changes during review: {state['human_feedback']}")
    return "\n".join(facts)


class WriterAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        counter = CallCounter(self._deps.llm)
        evidence = state.get("evidence", [])
        selected = select_evidence(evidence, self._deps.settings.writer_max_sources)
        questions = "\n".join(f"- {q}" for q in state.get("research_questions", []))
        instructions = state.get("instructions") or "none"
        user = (
            f"User question: {state['user_query']}\nUser instructions: {instructions}\n"
            f"Research questions:\n{questions}\n\n"
            f"Methodology facts:\n{methodology_facts(state, selected)}\n\n"
            f"Analysis notes:\n{format_analysis(state.get('analysis'))}\n\n"
            f"Conflicts between sources: {'; '.join(state.get('conflicts', [])) or 'none reported'}\n\n"
            f"Evidence (cite only these IDs):\n{format_evidence(selected)}"
        )
        # No fallback exists for the report itself, so errors propagate: the run stops at this
        # node with its checkpoint intact and `retry()` re-runs only the writer.
        report = self._deps.llm.structured(ResearchReport, prompts.WRITER, user, operation="report writing")
        return merge(
            {
                "draft_report": report,
                "report_evidence_ids": [e.source.id for e in selected],
                "revision_count": 0,
                "llm_calls": counter.used,
            },
            progress(
                NODE,
                f"Draft '{report.title}' from the {len(selected)} strongest of {len(evidence)} sources; "
                f"{len(cited_ids(report))} cited",
            ),
        )

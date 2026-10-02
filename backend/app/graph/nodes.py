"""Workflow nodes that are not LLM agents: the human approval pause and finalisation."""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import interrupt

from app.agents.base import merge, progress
from app.agents.formatting import CITATION_PATTERN, cited_ids
from app.config import Settings
from app.graph.state import ResearchState
from app.models.agent_outputs import HumanDecision, ResearchReport
from app.models.domain import Citation, SourceType, VerificationStatus

_ID_PATTERN = re.compile(r"[WK]\d+")


def approval_request(state: ResearchState, max_iterations: int) -> dict[str, Any]:
    """The payload shown to the user while the graph is paused."""
    evidence = state.get("evidence", [])
    analysis = state.get("analysis")
    return {
        "type": "approval_request",
        "iteration": state.get("iteration_count", 1),
        "can_modify": state.get("iteration_count", 1) < max_iterations,
        "research_questions": state.get("research_questions", []),
        "sources": {
            "web": sum(e.source.source_type is SourceType.WEB for e in evidence),
            "knowledge_base": sum(e.source.source_type is SourceType.KNOWLEDGE_BASE for e in evidence),
            "total": len(evidence),
        },
        "answer_summary": analysis.answer_summary if analysis else None,
        "key_findings": [f.statement for f in analysis.key_findings] if analysis else [],
        "conflicts": state.get("conflicts", []),
        "errors": [e.message for e in state.get("errors", [])],
    }


def make_human_review(settings: Settings):  # noqa: ANN201 - returns a LangGraph node
    max_iterations = settings.max_research_iterations

    def human_review(state: ResearchState) -> dict[str, Any]:
        # Pauses the graph; the run is checkpointed and resumed with Command(resume=...).
        # Code before interrupt() re-runs on resume, so it must stay side-effect free.
        raw = interrupt(approval_request(state, max_iterations))
        decision = HumanDecision.model_validate(raw)

        if decision.action == "cancel":
            return merge({"human_decision": "cancel", "status": "cancelled"}, progress("human_review", "Cancelled by user"))
        if decision.action == "modify":
            if state.get("iteration_count", 1) >= max_iterations:
                return merge(
                    {"human_decision": "approve", "status": "writing"},
                    progress("human_review", f"Modification limit ({max_iterations}) reached; continuing to report"),
                )
            feedback = (decision.feedback or "").strip() or "Research the topic more thoroughly."
            return merge(
                {"human_decision": "modify", "human_feedback": feedback, "status": "planning"},
                progress("human_review", f"User requested changes: {feedback[:120]}"),
            )
        return merge({"human_decision": "approve", "status": "writing"}, progress("human_review", "Approved by user"))

    return human_review


def _sanitize(text: str, valid_ids: set[str]) -> str:
    """Remove citation IDs that do not correspond to collected evidence."""

    def replace(match: re.Match[str]) -> str:
        kept = [i for i in _ID_PATTERN.findall(match.group(1)) if i in valid_ids]
        return f"[{', '.join(kept)}]" if kept else "[citation removed: unknown source]"

    return CITATION_PATTERN.sub(replace, text)


def sanitize_report(report: ResearchReport, valid_ids: set[str]) -> ResearchReport:
    clean = report.model_copy(deep=True)
    for field in ("executive_summary", "introduction", "methodology", "limitations", "conclusion"):
        setattr(clean, field, _sanitize(getattr(clean, field), valid_ids))
    clean.key_findings = [_sanitize(f, valid_ids) for f in clean.key_findings]
    for section in clean.detailed_analysis:
        section.content = _sanitize(section.content, valid_ids)
    return clean


def finalize(state: ResearchState) -> dict[str, Any]:
    """Build the final report and its citations strictly from collected evidence."""
    evidence = {e.source.id: e for e in state.get("evidence", [])}
    report = sanitize_report(state["draft_report"], set(evidence))
    citations = [
        Citation(
            id=source_id,
            title=evidence[source_id].source.title,
            reference=evidence[source_id].source.reference,
            source_type=evidence[source_id].source.source_type,
            verification_status=evidence[source_id].status,
            excerpt=evidence[source_id].source.content[:300],
        )
        for source_id in cited_ids(report)
        if source_id in evidence
    ]

    notes = []
    critique = state.get("critique")
    if critique is not None:
        notes = [
            f"[{i.severity}] {i.location}: {i.description}"
            for i in critique.issues
            if i.severity in ("high", "medium")
        ]
    weak = sum(c.verification_status is not VerificationStatus.CORROBORATED for c in citations)
    if weak:
        notes.append(f"{weak} of {len(citations)} cited sources are not corroborated by a second source.")

    return merge(
        {"final_report": report, "citations": citations, "quality_notes": notes, "status": "completed"},
        progress("finalize", f"Final report with {len(citations)} citations; {len(notes)} quality note(s)"),
    )

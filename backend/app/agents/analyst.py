"""Analysis agent: combines verified web and knowledge-base evidence into analytical notes."""

from __future__ import annotations

from typing import Any

from app.agents import prompts
from app.agents.base import AgentDependencies, CallCounter, error, merge, progress
from app.agents.formatting import format_evidence
from app.exceptions import ExternalServiceError, RateLimitError
from app.graph.state import ResearchState
from app.models.agent_outputs import AnalysisNotes

NODE = "analysis"
MAX_FINDINGS = 7


class AnalysisAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        valid_ids = {e.source.id for e in evidence}
        counter = CallCounter(self._deps.llm)
        questions = "\n".join(f"- {q}" for q in state.get("research_questions", []))
        conflicts = "\n".join(f"- {c}" for c in state.get("conflicts", [])) or "none reported"
        user = (
            f"User question: {state['user_query']}\nResearch questions:\n{questions}\n\n"
            f"Verification summary: {state.get('verification_summary', '')}\n"
            f"Conflicts:\n{conflicts}\n\nEvidence:\n{format_evidence(evidence)}"
        )
        try:
            analysis = self._deps.llm.structured(AnalysisNotes, prompts.ANALYST, user, operation="analysis")
        except RateLimitError:
            raise
        except ExternalServiceError as exc:
            return merge(
                {"analysis": None, "status": "awaiting_approval", "llm_calls": counter.used},
                error(NODE, f"Analysis failed; the writer will work from raw evidence: {exc.message}"),
                progress(NODE, "Analysis unavailable"),
            )

        # Drop citations to unknown sources; keep findings that still have support.
        for finding in analysis.key_findings:
            finding.source_ids = [i for i in finding.source_ids if i in valid_ids]
        analysis.key_findings = [f for f in analysis.key_findings if f.source_ids][:MAX_FINDINGS]
        return merge(
            {"analysis": analysis, "status": "awaiting_approval", "llm_calls": counter.used},
            progress(
                NODE,
                f"{len(analysis.key_findings)} key findings, {len(analysis.patterns)} patterns, "
                f"{len(analysis.gaps)} gaps",
            ),
        )

"""Critic / fact-checker agent.

Combines deterministic citation checks (cheap, exact) with an LLM review of
whether claims are supported by the evidence. The decision to revise is made
by a fixed rule, not by the LLM, so routing is predictable.
"""

from __future__ import annotations

from typing import Any

from app.agents import prompts
from app.agents.base import AgentDependencies, CallCounter, error, merge, progress
from app.agents.formatting import CITATION_PATTERN, extract_citation_ids, format_evidence, report_to_markdown
from app.exceptions import ExternalServiceError, RateLimitError
from app.graph.state import ResearchState
from app.models.agent_outputs import Critique, CritiqueIssue, ResearchReport

NODE = "critic"
MEDIUM_ISSUES_FOR_REVISION = 3


def check_citations(report: ResearchReport, valid_ids: set[str]) -> list[CritiqueIssue]:
    """Exact checks the LLM might miss: unknown IDs and uncited findings."""
    issues: list[CritiqueIssue] = []
    used = extract_citation_ids(report_to_markdown(report))
    if not used:
        issues.append(
            CritiqueIssue(
                severity="high",
                category="missing_citation",
                location="whole report",
                description="The report contains no citations.",
                suggestion="Cite the supporting source ID for every factual claim.",
            )
        )
    invalid = [i for i in used if i not in valid_ids]
    if invalid:
        issues.append(
            CritiqueIssue(
                severity="high",
                category="invalid_citation",
                location="whole report",
                description=f"Citations to non-existent sources: {', '.join(invalid)}.",
                suggestion="Use only the listed source IDs or remove the claims.",
            )
        )
    uncited = [f for f in report.key_findings if not CITATION_PATTERN.search(f)]
    if uncited:
        issues.append(
            CritiqueIssue(
                severity="medium",
                category="missing_citation",
                location="Key Findings",
                description=f"{len(uncited)} key finding(s) have no citation, e.g. '{uncited[0][:100]}'.",
                suggestion="Add the supporting source IDs or remove unsupported findings.",
            )
        )
    return issues


def needs_revision(issues: list[CritiqueIssue]) -> bool:
    high = sum(i.severity == "high" for i in issues)
    medium = sum(i.severity == "medium" for i in issues)
    return high > 0 or medium >= MEDIUM_ISSUES_FOR_REVISION


class CriticAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        report = state["draft_report"]
        evidence = state.get("evidence", [])
        deterministic = check_citations(report, {e.source.id for e in evidence})
        counter = CallCounter(self._deps.llm)
        updates: list[dict[str, Any]] = []

        user = (
            f"Report:\n{report_to_markdown(report)}\n\n"
            f"Evidence available:\n{format_evidence(evidence, max_chars=400)}"
        )
        try:
            llm_critique = self._deps.llm.structured(Critique, prompts.CRITIC, user, operation="critique")
            issues = deterministic + llm_critique.issues
            assessment = llm_critique.overall_assessment
        except RateLimitError:
            raise
        except ExternalServiceError as exc:
            issues = deterministic
            assessment = "LLM review unavailable; only automated citation checks were run."
            updates.append(error(NODE, f"LLM critique failed: {exc.message}"))

        critique = Critique(issues=issues, overall_assessment=assessment)
        revise = needs_revision(issues)
        by_severity = {s: sum(i.severity == s for i in issues) for s in ("high", "medium", "low")}
        round_label = "re-check after revision" if state.get("revision_count", 0) else "first review"
        return merge(
            {"critique": critique, "needs_revision": revise, "llm_calls": counter.used},
            *updates,
            progress(
                NODE,
                f"{round_label}: {by_severity['high']} high, {by_severity['medium']} medium, "
                f"{by_severity['low']} low issues → {'revision needed' if revise else 'acceptable'}",
            ),
        )

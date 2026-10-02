"""Source verification agent.

Rates each source's relevance and publisher reliability and checks which key
claims are corroborated, single-source or conflicting. It never labels a claim
"true" — only how well the collected sources support it. Low-relevance sources
are excluded from the evidence used for writing.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents import prompts
from app.agents.base import AgentDependencies, CallCounter, error, merge, progress
from app.agents.formatting import format_sources
from app.exceptions import ExternalServiceError, RateLimitError
from app.graph.state import ResearchState
from app.models.agent_outputs import VerificationResult
from app.models.domain import Evidence, Source, VerificationStatus

logger = logging.getLogger(__name__)

NODE = "source_verification"
# Status precedence when a source appears in several claims: a conflict outweighs support.
_PRECEDENCE = {
    VerificationStatus.CONFLICTING: 3,
    VerificationStatus.CORROBORATED: 2,
    VerificationStatus.SINGLE_SOURCE: 1,
    VerificationStatus.UNVERIFIED: 0,
}


def _source_statuses(result: VerificationResult, valid_ids: set[str]) -> dict[str, VerificationStatus]:
    statuses: dict[str, VerificationStatus] = {}
    for claim in result.claims:
        ids = [i for i in claim.source_ids if i in valid_ids]
        status = VerificationStatus(claim.status)
        if status is VerificationStatus.CORROBORATED and len(set(ids)) < 2:
            status = VerificationStatus.SINGLE_SOURCE  # "corroborated" needs 2+ sources
        for source_id in ids:
            current = statuses.get(source_id, VerificationStatus.UNVERIFIED)
            if _PRECEDENCE[status] > _PRECEDENCE[current]:
                statuses[source_id] = status
    return statuses


def build_evidence(sources: list[Source], result: VerificationResult) -> list[Evidence]:
    valid_ids = {s.id for s in sources}
    assessments = {a.source_id: a for a in result.assessments if a.source_id in valid_ids}
    statuses = _source_statuses(result, valid_ids)
    evidence = []
    for source in sources:
        assessment = assessments.get(source.id)
        if assessment is not None and assessment.relevance == "low":
            continue
        evidence.append(
            Evidence(
                source=source,
                relevance=assessment.relevance if assessment else "medium",
                reliability=assessment.reliability if assessment else "unknown",
                status=statuses.get(source.id, VerificationStatus.UNVERIFIED),
                notes=assessment.reason if assessment else "Not assessed by the verifier.",
            )
        )
    return evidence


def unverified_evidence(sources: list[Source]) -> list[Evidence]:
    return [
        Evidence(
            source=s,
            relevance="medium",
            reliability="unknown",
            status=VerificationStatus.UNVERIFIED,
            notes="Verification unavailable.",
        )
        for s in sources
    ]


class SourceVerificationAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        sources = [
            s
            for s in [*state.get("web_sources", []), *state.get("retrieved_documents", [])]
            if s.content.strip()
        ]
        if not sources:
            return merge(
                {"evidence": [], "status": "failed"},
                error(NODE, "No sources were found on the web or in the knowledge base."),
                progress(NODE, "No sources to verify"),
            )

        counter = CallCounter(self._deps.llm)
        questions = "\n".join(f"- {q}" for q in state.get("research_questions", []))
        user = f"Research questions:\n{questions}\n\nSources:\n{format_sources(sources, max_chars=500)}"
        try:
            result = self._deps.llm.structured(
                VerificationResult, prompts.VERIFIER, user, operation="source verification"
            )
        except RateLimitError:
            raise
        except ExternalServiceError as exc:
            return merge(
                {
                    "evidence": unverified_evidence(sources),
                    "verification_summary": "Automatic verification was unavailable.",
                    "conflicts": [],
                    "llm_calls": counter.used,
                },
                error(NODE, f"Verification failed; all sources marked unverified: {exc.message}"),
                progress(NODE, f"{len(sources)} sources kept as unverified"),
            )

        evidence = build_evidence(sources, result)
        counts: dict[str, int] = {}
        for item in evidence:
            counts[item.status.value] = counts.get(item.status.value, 0) + 1
        excluded = len(sources) - len(evidence)
        breakdown = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        return merge(
            {
                "evidence": evidence,
                "verification_summary": result.summary,
                "conflicts": result.conflicts,
                "llm_calls": counter.used,
            },
            progress(
                NODE,
                f"{len(evidence)} sources kept ({breakdown}); {excluded} excluded as low relevance; "
                f"{len(result.conflicts)} conflict(s)",
            ),
        )

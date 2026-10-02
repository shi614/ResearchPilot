"""Choose the best verified evidence for the writer (and the critic/reviser that check it).

Passing every collected source to the writer makes the largest prompt of the run
even larger. Instead, sources are ranked by:
  1. verification status — corroborated first, unverified last;
  2. relevance, then publisher reliability, then the retrieval/search score.
Relevant knowledge-base excerpts get reserved slots (the user's own documents
should not be crowded out by web results), and no single website may dominate.
Source IDs are never changed, so every citation stays traceable.
"""

from __future__ import annotations

from urllib.parse import urlparse

from app.models.domain import Evidence, SourceType, VerificationStatus

_STATUS_RANK = {
    VerificationStatus.CORROBORATED: 3,
    VerificationStatus.SINGLE_SOURCE: 2,
    VerificationStatus.CONFLICTING: 2,  # disagreements are worth reporting, like single-source claims
    VerificationStatus.UNVERIFIED: 0,
}
_RELEVANCE_RANK = {"high": 2, "medium": 1, "low": 0}
_RELIABILITY_RANK = {"high": 3, "medium": 2, "unknown": 1, "low": 0}

MAX_RESERVED_KB = 3
MAX_PER_DOMAIN = 3


def evidence_rank(item: Evidence) -> tuple[int, int, int, float]:
    return (
        _STATUS_RANK[item.status],
        _RELEVANCE_RANK[item.relevance],
        _RELIABILITY_RANK[item.reliability],
        item.source.score or 0.0,
    )


def _domain(item: Evidence) -> str:
    if item.source.source_type is SourceType.KNOWLEDGE_BASE or not item.source.url:
        return f"kb:{item.source.document_id}"
    return urlparse(item.source.url).netloc.removeprefix("www.").lower()


def select_evidence(evidence: list[Evidence], limit: int) -> list[Evidence]:
    """Return at most `limit` items, best first. Returns everything if already within the limit."""
    ranked = sorted(evidence, key=evidence_rank, reverse=True)  # stable: ties keep original order
    if len(ranked) <= limit:
        return ranked

    selected: list[Evidence] = []
    chosen: set[str] = set()
    per_domain: dict[str, int] = {}

    def take(item: Evidence) -> None:
        selected.append(item)
        chosen.add(item.source.id)
        per_domain[_domain(item)] = per_domain.get(_domain(item), 0) + 1

    # 1. Reserve slots for the best relevant knowledge-base excerpts.
    kb_items = [e for e in ranked if e.source.source_type is SourceType.KNOWLEDGE_BASE]
    for item in kb_items[: min(MAX_RESERVED_KB, limit // 4 or 1)]:
        take(item)

    # 2. Fill by rank, at most MAX_PER_DOMAIN sources per website.
    for item in ranked:
        if len(selected) >= limit:
            break
        if item.source.id not in chosen and per_domain.get(_domain(item), 0) < MAX_PER_DOMAIN:
            take(item)

    # 3. If the domain cap left slots empty, fill them by rank anyway.
    for item in ranked:
        if len(selected) >= limit:
            break
        if item.source.id not in chosen:
            take(item)

    return sorted(selected, key=evidence_rank, reverse=True)


def report_evidence(evidence: list[Evidence], selected_ids: list[str] | None) -> list[Evidence]:
    """The evidence the report was written from (all evidence for runs predating selection)."""
    if selected_ids is None:
        return evidence
    wanted = set(selected_ids)
    return [e for e in evidence if e.source.id in wanted]

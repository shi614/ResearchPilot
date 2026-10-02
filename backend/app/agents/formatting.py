"""Compact text renderings of state objects for prompts (token-conscious)."""

from __future__ import annotations

import re

from app.models.agent_outputs import AnalysisNotes, ResearchReport
from app.models.domain import Evidence, Source

# Matches [W1] as well as grouped citations such as [W1, K2] or [W1; W3].
CITATION_PATTERN = re.compile(r"\[((?:[WK]\d+)(?:\s*[,;]\s*[WK]\d+)*)\]")
_ID_PATTERN = re.compile(r"[WK]\d+")


def extract_citation_ids(text: str) -> list[str]:
    """Citation IDs in order of first appearance."""
    ids: list[str] = []
    for group in CITATION_PATTERN.findall(text):
        ids.extend(_ID_PATTERN.findall(group))
    return list(dict.fromkeys(ids))


def format_sources(sources: list[Source], max_chars: int = 600) -> str:
    lines = []
    for source in sources:
        origin = "web" if source.url else "uploaded document"
        lines.append(
            f"[{source.id}] ({origin}) {source.title}\n"
            f"    ref: {source.reference}\n"
            f"    {source.content[:max_chars].strip()}"
        )
    return "\n".join(lines)


def format_evidence(evidence: list[Evidence], max_chars: int = 600) -> str:
    lines = []
    for item in evidence:
        source = item.source
        lines.append(
            f"[{source.id}] {source.title} | relevance={item.relevance} "
            f"reliability={item.reliability} status={item.status.value}\n"
            f"    ref: {source.reference}\n"
            f"    {source.content[:max_chars].strip()}"
        )
    return "\n".join(lines)


def format_analysis(analysis: AnalysisNotes | None) -> str:
    if analysis is None:
        return "(analysis unavailable — rely on the evidence directly)"
    findings = "\n".join(
        f"- {f.statement} [{', '.join(f.source_ids)}] (confidence: {f.confidence})"
        for f in analysis.key_findings
    )
    parts = [
        f"Answer summary: {analysis.answer_summary}",
        f"Key findings:\n{findings}",
        "Patterns: " + "; ".join(analysis.patterns),
        "Comparisons: " + "; ".join(analysis.comparisons),
        "Gaps: " + "; ".join(analysis.gaps),
    ]
    return "\n".join(parts)


def report_to_markdown(report: ResearchReport) -> str:
    """Render the report body (references are rendered separately from citations)."""
    parts = [
        f"# {report.title}",
        "## Executive Summary",
        report.executive_summary,
        "## Introduction",
        report.introduction,
        "## Research Questions",
        "\n".join(f"{i}. {q}" for i, q in enumerate(report.research_questions, 1)),
        "## Methodology",
        report.methodology,
        "## Key Findings",
        "\n".join(f"- {finding}" for finding in report.key_findings),
        "## Detailed Analysis",
        *[f"### {section.heading}\n\n{section.content}" for section in report.detailed_analysis],
        "## Limitations",
        report.limitations,
        "## Conclusion",
        report.conclusion,
    ]
    return "\n\n".join(parts)


def cited_ids(report: ResearchReport) -> list[str]:
    return extract_citation_ids(report_to_markdown(report))

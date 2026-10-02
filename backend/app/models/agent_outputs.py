"""Structured-output schemas the agents request from Gemini.

Field descriptions are sent to the model as part of the JSON schema, so they
double as compact instructions. List lengths are capped in code rather than
in the schema, so a slightly-too-long answer is trimmed instead of rejected.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.domain import Rating


class ResearchPlan(BaseModel):
    objective: str = Field(description="One-sentence statement of what the research must answer")
    research_questions: list[str] = Field(description="2-4 focused sub-questions")
    search_queries: list[str] = Field(description="2-4 concise web search queries")
    use_knowledge_base: bool = Field(
        description="Whether the user's uploaded documents are likely relevant"
    )
    scope_notes: str = Field(description="Scope, assumptions and what is out of scope")


class SourceAssessment(BaseModel):
    source_id: str
    relevance: Rating = Field(description="How directly the source addresses the research questions")
    reliability: Literal["high", "medium", "low", "unknown"] = Field(
        description="Credibility of the publisher/document type (not whether claims are true)"
    )
    reason: str = Field(description="One short sentence")


class ClaimCheck(BaseModel):
    claim: str
    source_ids: list[str] = Field(description="IDs of sources making or disputing this claim")
    status: Literal["corroborated", "single_source", "conflicting"]
    note: str = ""


class VerificationResult(BaseModel):
    assessments: list[SourceAssessment]
    claims: list[ClaimCheck] = Field(description="The 3-8 most important factual claims")
    conflicts: list[str] = Field(description="Disagreements between sources, citing source IDs")
    summary: str


class Finding(BaseModel):
    statement: str
    source_ids: list[str]
    confidence: Rating = Field(description="Based on corroboration and source reliability")


class AnalysisNotes(BaseModel):
    answer_summary: str = Field(description="Direct answer to the user's question in 2-4 sentences")
    key_findings: list[Finding] = Field(description="3-7 findings, each backed by source IDs")
    patterns: list[str] = Field(description="Trends or recurring themes across sources")
    comparisons: list[str] = Field(description="Where sources agree, differ or use different scopes")
    gaps: list[str] = Field(description="Questions the evidence does not answer well")


class ReportSection(BaseModel):
    heading: str
    content: str = Field(description="Markdown paragraphs with inline citations like [W1] or [K2]")


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    introduction: str
    research_questions: list[str]
    methodology: str
    key_findings: list[str] = Field(description="Bullet findings, each with inline citations")
    detailed_analysis: list[ReportSection]
    limitations: str
    conclusion: str


class CritiqueIssue(BaseModel):
    severity: Literal["high", "medium", "low"] = Field(
        description="high = unsupported/incorrect key claim or invalid citation"
    )
    category: Literal[
        "unsupported_claim",
        "missing_citation",
        "invalid_citation",
        "contradiction",
        "incomplete",
        "other",
    ]
    location: str = Field(description="Report section the issue is in")
    description: str
    suggestion: str


class Critique(BaseModel):
    issues: list[CritiqueIssue]
    overall_assessment: str


class HumanDecision(BaseModel):
    """Resume value for the human-approval interrupt."""

    action: Literal["approve", "modify", "cancel"]
    feedback: str | None = None

"""Typed LangGraph state for a research run.

Fields written by the parallel research branches (web + knowledge base) use
reducers, so concurrent updates are merged instead of overwriting each other.
Counters (`llm_calls`, `web_searches`) are summed across all nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from app.models.agent_outputs import AnalysisNotes, Critique, ResearchPlan, ResearchReport
from app.models.domain import AgentError, Citation, Evidence, ProgressEvent, Source

RunStatus = Literal[
    "planning",
    "researching",
    "awaiting_approval",
    "writing",
    "completed",
    "cancelled",
    "failed",
]


class ResearchState(TypedDict, total=False):
    # --- input ---
    user_query: str
    instructions: str | None

    # --- planning / human-in-the-loop ---
    research_plan: ResearchPlan
    research_questions: list[str]
    iteration_count: int  # research iterations (1 + number of "modify" decisions)
    human_decision: Literal["approve", "modify", "cancel"] | None
    human_feedback: str | None

    # --- research (written in parallel → reducers) ---
    web_sources: Annotated[list[Source], operator.add]
    retrieved_documents: Annotated[list[Source], operator.add]
    searched_queries: Annotated[list[str], operator.add]

    # --- verification / analysis ---
    evidence: list[Evidence]
    verification_summary: str
    conflicts: list[str]
    analysis: AnalysisNotes | None

    # --- writing / critique loop ---
    report_evidence_ids: list[str]  # best sources selected for the writer
    draft_report: ResearchReport
    critique: Critique
    needs_revision: bool
    revision_count: int
    final_report: ResearchReport
    citations: list[Citation]
    quality_notes: list[str]

    # --- bookkeeping ---
    status: RunStatus
    errors: Annotated[list[AgentError], operator.add]
    progress: Annotated[list[ProgressEvent], operator.add]
    llm_calls: Annotated[int, operator.add]
    web_searches: Annotated[int, operator.add]

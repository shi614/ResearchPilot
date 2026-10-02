"""Conditional-edge functions. Pure functions of state → easy to unit test."""

from __future__ import annotations

from langgraph.graph import END

from app.graph.state import ResearchState

WEB_RESEARCH = "web_research"
RAG_RESEARCH = "rag_research"
VERIFY = "source_verification"
ANALYSIS = "analysis"
HUMAN_REVIEW = "human_review"
PLANNER = "planner"
WRITER = "writer"
CRITIC = "critic"
REVISION = "revision"
FINALIZE = "finalize"


def route_after_planning(state: ResearchState, *, web_enabled: bool) -> list[str]:
    """Fan out to the research branches the plan needs (run in parallel)."""
    plan = state["research_plan"]
    branches = []
    if web_enabled and plan.search_queries:
        branches.append(WEB_RESEARCH)
    if plan.use_knowledge_base:
        branches.append(RAG_RESEARCH)
    return branches or [VERIFY]  # nothing to search: verification reports "no sources"


def route_after_verification(state: ResearchState) -> str:
    return ANALYSIS if state.get("evidence") else END


def route_after_review(state: ResearchState) -> str:
    decision = state.get("human_decision")
    if decision == "modify":
        return PLANNER
    if decision == "cancel":
        return END
    return WRITER


def route_after_critic(state: ResearchState, *, max_revisions: int) -> str:
    """Revise only while issues are significant AND the revision budget allows it."""
    if state.get("needs_revision") and state.get("revision_count", 0) < max_revisions:
        return REVISION
    return FINALIZE

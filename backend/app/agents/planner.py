"""Planner / orchestrator agent: turns the query into a research plan and decides which
research branches (web, knowledge base) the graph should run."""

from __future__ import annotations

import logging
from typing import Any

from app.agents import prompts
from app.agents.base import AgentDependencies, CallCounter, error, merge, progress
from app.exceptions import ExternalServiceError, RateLimitError
from app.graph.state import ResearchState
from app.models.agent_outputs import ResearchPlan

logger = logging.getLogger(__name__)

NODE = "planner"
MAX_QUESTIONS = 4
MAX_QUERIES = 4


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: dict[str, str] = {}
    for item in items:
        cleaned = " ".join(item.split())
        if cleaned and cleaned.lower() not in seen:
            seen[cleaned.lower()] = cleaned
    return list(seen.values())[:limit]


class PlannerAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    def _knowledge_base_available(self) -> bool:
        knowledge = self._deps.knowledge
        try:
            return knowledge is not None and knowledge.has_documents()
        except Exception:  # noqa: BLE001 - KB problems must not block planning
            logger.exception("Could not check knowledge base")
            return False

    def _prompt(self, state: ResearchState, kb_available: bool) -> str:
        lines = [f"Research request: {state['user_query']}"]
        if state.get("instructions"):
            lines.append(f"Additional instructions: {state['instructions']}")
        lines.append(
            "The user has uploaded documents to the knowledge base."
            if kb_available
            else "The knowledge base is empty."
        )
        if state.get("human_feedback"):
            lines.append(f"\nThe user reviewed the previous research and asked: {state['human_feedback']}")
            lines.append("Previously researched questions:")
            lines.extend(f"- {q}" for q in state.get("research_questions", []))
            lines.append("Already-used search queries: " + "; ".join(state.get("searched_queries", [])))
        return "\n".join(lines)

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0) + 1
        kb_available = self._knowledge_base_available()
        counter = CallCounter(self._deps.llm)
        updates: list[dict[str, Any]] = []

        try:
            plan = self._deps.llm.structured(
                ResearchPlan, prompts.PLANNER, self._prompt(state, kb_available), operation="planning"
            )
        except RateLimitError:
            raise  # stop the run; it can be resumed from the checkpoint
        except ExternalServiceError as exc:
            # Degrade gracefully: research the query as-is rather than failing the run.
            query = state["user_query"]
            plan = ResearchPlan(
                objective=query,
                research_questions=[query],
                search_queries=[query],
                use_knowledge_base=True,
                scope_notes="Fallback plan (the planning model was unavailable).",
            )
            updates.append(error(NODE, f"Planning failed, using a fallback plan: {exc.message}"))

        plan.research_questions = _dedupe(plan.research_questions, MAX_QUESTIONS) or [state["user_query"]]
        plan.search_queries = _dedupe(plan.search_queries, MAX_QUERIES)
        # The LLM suggests; the orchestrator decides based on what is actually available.
        plan.use_knowledge_base = plan.use_knowledge_base and kb_available

        branches = []
        if plan.search_queries and self._deps.web is not None:
            branches.append("web")
        if plan.use_knowledge_base:
            branches.append("knowledge base")
        updates.append(
            progress(
                NODE,
                f"Iteration {iteration}: {len(plan.research_questions)} research questions, "
                f"{len(plan.search_queries)} search queries; branches: {', '.join(branches) or 'none'}",
            )
        )
        return merge(
            {
                "research_plan": plan,
                "research_questions": plan.research_questions,
                "iteration_count": iteration,
                "human_decision": None,
                "status": "researching",
                "llm_calls": counter.used,
            },
            *updates,
        )

"""Builds the ResearchPilot LangGraph workflow.

    START → planner ─┬→ web_research ─┐
                     └→ rag_research ─┴→ source_verification → analysis → human_review ⏸
    human_review ─approve→ writer → critic ─(issues & budget left)→ revision → critic
                 ─modify──→ planner                 └─(otherwise)→ finalize → END
                 ─cancel──→ END
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.analyst import AnalysisAgent
from app.agents.base import AgentDependencies
from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.rag_researcher import KnowledgeResearchAgent
from app.agents.reviser import RevisionAgent
from app.agents.source_verifier import SourceVerificationAgent
from app.agents.web_researcher import WebResearchAgent
from app.agents.writer import WriterAgent
from app.graph import routing as r
from app.graph.nodes import finalize, make_human_review
from app.graph.state import ResearchState


def build_research_graph(
    deps: AgentDependencies, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node(r.PLANNER, PlannerAgent(deps))
    graph.add_node(r.WEB_RESEARCH, WebResearchAgent(deps))
    graph.add_node(r.RAG_RESEARCH, KnowledgeResearchAgent(deps))
    graph.add_node(r.VERIFY, SourceVerificationAgent(deps))
    graph.add_node(r.ANALYSIS, AnalysisAgent(deps))
    graph.add_node(r.HUMAN_REVIEW, make_human_review(deps.settings))
    graph.add_node(r.WRITER, WriterAgent(deps))
    graph.add_node(r.CRITIC, CriticAgent(deps))
    graph.add_node(r.REVISION, RevisionAgent(deps))
    graph.add_node(r.FINALIZE, finalize)

    graph.add_edge(START, r.PLANNER)
    graph.add_conditional_edges(
        r.PLANNER,
        partial(r.route_after_planning, web_enabled=deps.web is not None),
        [r.WEB_RESEARCH, r.RAG_RESEARCH, r.VERIFY],
    )
    # Both branches run in the same super-step, so verification runs once, after both finish.
    graph.add_edge(r.WEB_RESEARCH, r.VERIFY)
    graph.add_edge(r.RAG_RESEARCH, r.VERIFY)
    graph.add_conditional_edges(r.VERIFY, r.route_after_verification, [r.ANALYSIS, END])
    graph.add_edge(r.ANALYSIS, r.HUMAN_REVIEW)
    graph.add_conditional_edges(r.HUMAN_REVIEW, r.route_after_review, [r.PLANNER, r.WRITER, END])
    graph.add_edge(r.WRITER, r.CRITIC)
    graph.add_conditional_edges(
        r.CRITIC,
        partial(r.route_after_critic, max_revisions=deps.settings.max_revisions),
        [r.REVISION, r.FINALIZE],
    )
    graph.add_edge(r.REVISION, r.CRITIC)
    graph.add_edge(r.FINALIZE, END)

    return graph.compile(checkpointer=checkpointer, name="researchpilot")

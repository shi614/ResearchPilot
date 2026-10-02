"""Wires real dependencies (Gemini, Tavily, knowledge base, SQLite checkpoints) into a runner."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.base import AgentDependencies, KnowledgeSearcher
from app.agents.llm import GeminiLLM, create_llm
from app.config import Settings
from app.graph.builder import build_research_graph
from app.graph.checkpoint import create_checkpointer
from app.graph.runner import ResearchRunner, UpdateCallback
from app.tools.web_search import WebSearchTools


@dataclass
class ResearchEngine:
    runner: ResearchRunner
    llm: GeminiLLM
    web_enabled: bool


def create_research_engine(
    settings: Settings,
    knowledge: KnowledgeSearcher | None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    on_update: UpdateCallback | None = None,
) -> ResearchEngine:
    llm = create_llm(settings)  # raises ConfigurationError without GEMINI_API_KEY
    web = WebSearchTools.from_settings(settings) if settings.tavily_api_key else None
    deps = AgentDependencies(llm=llm, settings=settings, web=web, knowledge=knowledge)
    graph = build_research_graph(deps, checkpointer or create_checkpointer(settings.checkpoint_db))
    return ResearchEngine(ResearchRunner(graph, on_update=on_update), llm, web is not None)

"""Knowledge-base (RAG) agent: semantic retrieval from the user's uploaded documents.

Uses embeddings only (no generation calls), so it adds no LLM quota cost.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import AgentDependencies, error, merge, progress
from app.exceptions import ResearchPilotError
from app.graph.state import ResearchState
from app.models.domain import Source, SourceType

logger = logging.getLogger(__name__)

NODE = "rag_research"


class KnowledgeResearchAgent:
    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        knowledge = self._deps.knowledge
        if knowledge is None:
            return progress(NODE, "Knowledge base unavailable; skipped")

        existing = state.get("retrieved_documents", [])
        seen = {(s.document_id, s.content) for s in existing}
        next_id = len(existing) + 1
        sources: list[Source] = []
        failures: list[dict[str, Any]] = []

        for question in state.get("research_questions") or [state["user_query"]]:
            try:
                chunks = knowledge.search(question)
            except ResearchPilotError as exc:
                # Includes embedding rate limits: the KB is optional, so degrade, don't stop.
                failures.append(error(NODE, f"Retrieval for '{question[:60]}' failed: {exc.message}"))
                continue
            except Exception as exc:  # noqa: BLE001 - an optional branch must not kill the run
                logger.exception("Unexpected knowledge-base error")
                failures.append(error(NODE, f"Knowledge-base retrieval error ({type(exc).__name__})"))
                break
            for chunk in chunks:
                key = (chunk.document_id, chunk.content)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(
                    Source(
                        id=f"K{next_id}",
                        source_type=SourceType.KNOWLEDGE_BASE,
                        title=chunk.reference,
                        content=chunk.content,
                        query=question,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        page=chunk.page,
                        score=chunk.relevance,
                    )
                )
                next_id += 1

        documents = len({s.document_id for s in sources})
        return merge(
            {"retrieved_documents": sources},
            *failures,
            progress(NODE, f"{len(sources)} relevant chunks from {documents} document(s)"),
        )

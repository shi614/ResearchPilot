"""Shared agent dependencies and small helpers for building node updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.agents.llm import LLMGateway
from app.config import Settings
from app.models.domain import AgentError, ProgressEvent, RetrievedChunk
from app.tools.web_search import WebSearchTools


class KnowledgeSearcher(Protocol):
    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]: ...

    def has_documents(self) -> bool: ...


@dataclass
class AgentDependencies:
    llm: LLMGateway
    settings: Settings
    web: WebSearchTools | None = None  # None → web research disabled (no Tavily key)
    knowledge: KnowledgeSearcher | None = None


def progress(node: str, message: str) -> dict[str, Any]:
    return {"progress": [ProgressEvent(node=node, message=message)]}


def error(node: str, message: str) -> dict[str, Any]:
    return {"errors": [AgentError(node=node, message=message)]}


def merge(*updates: dict[str, Any]) -> dict[str, Any]:
    """Merge partial node updates; list values for the same key are concatenated."""
    merged: dict[str, Any] = {}
    for update in updates:
        for key, value in update.items():
            if key in merged and isinstance(value, list) and isinstance(merged[key], list):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
    return merged


class CallCounter:
    """Measures how many LLM calls a node made (for the `llm_calls` state counter)."""

    def __init__(self, llm: LLMGateway) -> None:
        self._llm = llm
        self._start = llm.call_count

    @property
    def used(self) -> int:
        return self._llm.call_count - self._start

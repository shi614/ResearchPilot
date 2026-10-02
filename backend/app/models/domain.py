"""Domain models for evidence gathered by the research tools.

Every piece of evidence keeps enough provenance (URL or document + page)
to be cited later; nothing here asserts that the content is true.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    WEB = "web"
    KNOWLEDGE_BASE = "knowledge_base"


class WebSearchResult(BaseModel):
    title: str
    url: str
    content: str = Field(description="Relevant snippet returned by the search engine")
    score: float | None = Field(default=None, description="Search engine relevance score (0-1)")
    published_date: str | None = None
    query: str = Field(description="The search query that produced this result")
    source_type: SourceType = SourceType.WEB


class WebPageExtract(BaseModel):
    url: str
    content: str


class RetrievedChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    page: int | None = Field(default=None, description="1-based page number for PDFs")
    content: str
    relevance: float = Field(ge=0.0, le=1.0)
    source_type: SourceType = SourceType.KNOWLEDGE_BASE

    @property
    def reference(self) -> str:
        return f"{self.filename}, p. {self.page}" if self.page else self.filename

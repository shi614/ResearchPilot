"""Domain models for evidence gathered by the research tools and agents.

Every piece of evidence keeps enough provenance (URL or document + page)
to be cited later; nothing here asserts that the content is true.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

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


class Source(BaseModel):
    """A citable source collected during research. IDs: W1, W2… (web), K1, K2… (knowledge base)."""

    id: str
    source_type: SourceType
    title: str
    content: str
    query: str = Field(description="Search query or research question that found this source")
    url: str | None = None
    document_id: str | None = None
    filename: str | None = None
    page: int | None = None
    score: float | None = None

    @property
    def reference(self) -> str:
        if self.url:
            return self.url
        return f"{self.filename}, p. {self.page}" if self.page else (self.filename or self.title)

    @property
    def dedupe_key(self) -> str:
        return self.url or f"{self.document_id}:{self.content[:80]}"


class VerificationStatus(str, Enum):
    CORROBORATED = "corroborated"  # supported by 2+ independent sources
    SINGLE_SOURCE = "single_source"  # only one source makes this claim
    CONFLICTING = "conflicting"  # other sources disagree
    UNVERIFIED = "unverified"  # could not be assessed


Rating = Literal["high", "medium", "low"]


class Evidence(BaseModel):
    """A source after verification. Status describes corroboration, not truth."""

    source: Source
    relevance: Rating
    reliability: Literal["high", "medium", "low", "unknown"]
    status: VerificationStatus
    notes: str = ""


class Citation(BaseModel):
    id: str
    title: str
    reference: str
    source_type: SourceType
    verification_status: VerificationStatus
    excerpt: str


class AgentError(BaseModel):
    node: str
    message: str


class ProgressEvent(BaseModel):
    node: str
    message: str

"""Request/response schemas for the HTTP API.

These deliberately expose only non-sensitive information: whether a key is
configured, never the key itself.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.database.orm import DocumentStatus, EventStatus, SessionStatus
from app.models.agent_outputs import ResearchReport
from app.models.domain import Citation, RetrievedChunk


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: Literal["ok", "unavailable"]
    gemini_api_key_configured: bool
    tavily_api_key_configured: bool
    missing_keys: list[str] = Field(default_factory=list)
    gemini_model: str


class ModelCheckStatus(str, Enum):
    OK = "ok"
    MODEL_UNAVAILABLE = "model_unavailable"
    MISSING_API_KEY = "missing_api_key"
    INVALID_API_KEY = "invalid_api_key"
    QUOTA_EXHAUSTED = "quota_exhausted"
    ERROR = "error"


class ModelCheckResponse(BaseModel):
    status: ModelCheckStatus
    configured_model: str
    model_available: bool | None = None
    embedding_model: str
    embedding_available: bool | None = None
    probed: bool = False
    suggested_models: list[str] = Field(default_factory=list)
    message: str
    checked_at: datetime


class DocumentResponse(BaseModel):
    """A knowledge-base document (the server-side storage path is not exposed)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    status: DocumentStatus
    error: str | None = None
    created_at: datetime


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=20)


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class StartResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000, description="Research topic or question")
    instructions: str | None = Field(default=None, max_length=2000, description="Optional guidance")


class ResearchEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    node: str
    status: EventStatus
    message: str | None = None
    created_at: datetime


class ResearchStats(BaseModel):
    llm_calls: int = 0
    web_searches: int = 0
    web_sources: int = 0
    knowledge_base_sources: int = 0
    evidence: int = 0
    evidence_web: int = 0
    evidence_knowledge_base: int = 0
    iteration: int = 0
    revisions: int = 0


class ResearchSummary(BaseModel):
    id: str
    query: str
    status: SessionStatus
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    has_report: bool = False


class ResearchSessionResponse(ResearchSummary):
    instructions: str | None = None
    error: str | None = None
    retryable: bool = False
    pending_nodes: list[str] = Field(default_factory=list)
    approval_request: dict | None = Field(
        default=None, description="Present while status is awaiting_approval"
    )
    errors: list[str] = Field(default_factory=list, description="Non-fatal agent errors")
    stats: ResearchStats | None = None
    events: list[ResearchEventResponse] = Field(default_factory=list)


class ReportResponse(BaseModel):
    session_id: str
    title: str
    markdown: str
    report: ResearchReport
    citations: list[Citation]
    quality_notes: list[str]
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: str

"""Request/response schemas for the HTTP API.

These deliberately expose only non-sensitive information: whether a key is
configured, never the key itself.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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


class ErrorResponse(BaseModel):
    error: str
    detail: str

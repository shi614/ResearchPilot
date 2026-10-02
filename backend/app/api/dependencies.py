"""FastAPI dependencies resolved from per-app state (overridable in tests)."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.database import Database
from app.services.knowledge_service import KnowledgeService
from app.services.model_check import ModelCheckService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_model_check_service(request: Request) -> ModelCheckService:
    return request.app.state.model_check


def get_knowledge_service(request: Request) -> KnowledgeService:
    return request.app.state.knowledge

"""Health and configuration-check endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

from app import __version__
from app.api.dependencies import get_app_settings, get_database, get_model_check_service
from app.config import Settings
from app.database import Database
from app.models.schemas import HealthResponse, ModelCheckResponse
from app.services.model_check import ModelCheckService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Service health and configuration status")
def health(
    settings: Annotated[Settings, Depends(get_app_settings)],
    database: Annotated[Database, Depends(get_database)],
) -> HealthResponse:
    database_ok = database.is_healthy()
    missing = settings.missing_required_keys()
    return HealthResponse(
        status="ok" if database_ok and not missing else "degraded",
        version=__version__,
        database="ok" if database_ok else "unavailable",
        gemini_api_key_configured=settings.gemini_api_key is not None,
        tavily_api_key_configured=settings.tavily_api_key is not None,
        missing_keys=missing,
        gemini_model=settings.gemini_model,
    )


@router.get(
    "/models",
    response_model=ModelCheckResponse,
    summary="Check that the configured Gemini models are available for this API key",
)
async def check_models(
    service: Annotated[ModelCheckService, Depends(get_model_check_service)],
    probe: Annotated[
        bool, Query(description="Send one tiny generation request to confirm free-tier quota.")
    ] = False,
    refresh: Annotated[bool, Query(description="Bypass the 5-minute result cache.")] = False,
) -> ModelCheckResponse:
    return await run_in_threadpool(service.check, probe=probe, refresh=refresh)

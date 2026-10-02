"""FastAPI application factory and entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import routes_health
from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import ResearchPilotError
from app.services.model_check import ModelCheckService

logger = logging.getLogger("researchpilot")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.ensure_directories()
        database = Database(settings.database_url)
        database.create_tables()
        app.state.settings = settings
        app.state.database = database
        app.state.model_check = ModelCheckService(
            api_key=settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None,
            model=settings.gemini_model,
            embedding_model=settings.gemini_embedding_model,
        )
        if missing := settings.missing_required_keys():
            logger.warning("Missing API keys: %s — research features are disabled.", ", ".join(missing))
        logger.info("ResearchPilot backend v%s started (model: %s)", __version__, settings.gemini_model)
        yield
        database.dispose()

    app = FastAPI(
        title="ResearchPilot API",
        description="Multi-Agent AI Research & Report Generation System",
        version=__version__,
        lifespan=lifespan,
    )

    @app.exception_handler(ResearchPilotError)
    async def handle_app_error(_request: Request, exc: ResearchPilotError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "detail": exc.message},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": "InternalServerError", "detail": "An unexpected error occurred."},
        )

    app.include_router(routes_health.router)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

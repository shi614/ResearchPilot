"""Shared fixtures. Tests never read the real `.env` or touch the real database."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.database import Database
from app.main import create_app

_ENV_VARS = [
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "DATABASE_URL",
    "MAX_REVISIONS",
]


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        "chroma_dir": tmp_path / "chroma",
        "upload_dir": tmp_path / "uploads",
        "reports_dir": tmp_path / "reports",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def database(settings: Settings) -> Iterator[Database]:
    settings.ensure_directories()
    db = Database(settings.database_url)
    db.create_tables()
    yield db
    db.dispose()


@pytest.fixture
def client_factory(tmp_path: Path) -> Iterator:
    """Build a TestClient for an app configured with the given setting overrides."""
    clients: list[TestClient] = []

    def _build(**overrides: object) -> TestClient:
        client = TestClient(create_app(make_settings(tmp_path, **overrides)))
        client.__enter__()  # run lifespan startup
        clients.append(client)
        return client

    yield _build
    for client in clients:
        client.__exit__(None, None, None)

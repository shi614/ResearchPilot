"""Shared fixtures. Tests never read the real `.env` or touch the real database."""

from __future__ import annotations

import io
import math
import re
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.config import Settings
from app.database import Database
from app.main import create_app


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words are similar.

    Lets vector-store tests exercise real ChromaDB retrieval without calling Gemini.
    """

    DIMENSIONS = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            vector[zlib.crc32(word.encode()) % self.DIMENSIONS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_pdf(pages: list[str]) -> bytes:
    """Build a real multi-page PDF (one text line per page) with ReportLab."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for text in pages:
        if text:
            pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()

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
        app = create_app(
            make_settings(tmp_path, **overrides),
            embeddings_factory=lambda _settings: KeywordEmbeddings(),
        )
        client = TestClient(app)
        client.__enter__()  # run lifespan startup
        clients.append(client)
        return client

    yield _build
    for client in clients:
        client.__exit__(None, None, None)

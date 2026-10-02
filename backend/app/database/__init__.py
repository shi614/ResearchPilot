"""Persistence layer (SQLAlchemy; SQLite by default, PostgreSQL-ready)."""

from app.database.engine import Database
from app.database.repository import DocumentRepository, ResearchRepository

__all__ = ["Database", "DocumentRepository", "ResearchRepository"]

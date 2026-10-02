"""SQLAlchemy ORM models for research sessions, run events, reports and documents."""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class SessionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    QUOTA_EXHAUSTED = "quota_exhausted"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"  # stopped mid-workflow (e.g. backend restart); retryable


RETRYABLE_STATUSES = {SessionStatus.FAILED, SessionStatus.QUOTA_EXHAUSTED, SessionStatus.INTERRUPTED}


class EventStatus(str, enum.Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSED = "processed"
    FAILED = "failed"


def _enum(enum_cls: type[enum.Enum]) -> Enum:
    """Store enum *values* as plain strings (portable across SQLite/PostgreSQL)."""
    return Enum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda members: [member.value for member in members],
    )


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    query: Mapped[str] = mapped_column(Text)
    instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[SessionStatus] = mapped_column(
        _enum(SessionStatus), default=SessionStatus.PENDING, index=True
    )
    title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    events: Mapped[list[RunEvent]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="RunEvent.id"
    )
    report: Mapped[Report | None] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )


class RunEvent(Base):
    """One workflow step (agent node) transition — drives the live UI progress."""

    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    node: Mapped[str] = mapped_column(String(64))
    status: Mapped[EventStatus] = mapped_column(_enum(EventStatus))
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[ResearchSession] = relationship(back_populates="events")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), unique=True
    )
    title: Mapped[str] = mapped_column(String(300))
    content_json: Mapped[str] = mapped_column(Text)
    pdf_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[ResearchSession] = relationship(back_populates="report")


class Document(Base):
    """A file uploaded to the knowledge base."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus), default=DocumentStatus.UPLOADED
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

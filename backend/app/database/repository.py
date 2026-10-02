"""Repository classes: the only place that issues database queries.

Services depend on these instead of raw sessions, which keeps persistence
swappable (SQLite now, PostgreSQL later) and easy to test.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.orm import (
    Document,
    DocumentStatus,
    EventStatus,
    Report,
    ResearchSession,
    RunEvent,
    SessionStatus,
    new_id,
)
from app.exceptions import NotFoundError


class ResearchRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, query: str, instructions: str | None = None) -> ResearchSession:
        record = ResearchSession(query=query, instructions=instructions)
        self._session.add(record)
        self._session.flush()
        return record

    def get(self, session_id: str) -> ResearchSession:
        record = self._session.get(ResearchSession, session_id)
        if record is None:
            raise NotFoundError(f"Research session '{session_id}' not found")
        return record

    def list_recent(
        self, limit: int = 50, statuses: set[SessionStatus] | None = None
    ) -> list[ResearchSession]:
        stmt = select(ResearchSession)
        if statuses:
            stmt = stmt.where(ResearchSession.status.in_(statuses))
        stmt = stmt.order_by(ResearchSession.created_at.desc(), ResearchSession.id).limit(limit)
        return list(self._session.scalars(stmt))

    def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        *,
        error: str | None = None,
        title: str | None = None,
    ) -> ResearchSession:
        record = self.get(session_id)
        record.status = status
        record.error = error
        if title is not None:
            record.title = title
        self._session.flush()
        return record

    def add_event(
        self,
        session_id: str,
        node: str,
        status: EventStatus,
        message: str | None = None,
    ) -> RunEvent:
        self.get(session_id)  # fail fast with NotFoundError
        event = RunEvent(session_id=session_id, node=node, status=status, message=message)
        self._session.add(event)
        self._session.flush()
        return event

    def list_events(self, session_id: str, after_id: int = 0) -> list[RunEvent]:
        stmt = (
            select(RunEvent)
            .where(RunEvent.session_id == session_id, RunEvent.id > after_id)
            .order_by(RunEvent.id)
        )
        return list(self._session.scalars(stmt))

    def save_report(
        self, session_id: str, title: str, content_json: str, pdf_path: str | None = None
    ) -> Report:
        record = self.get(session_id)
        if record.report is None:
            record.report = Report(title=title, content_json=content_json, pdf_path=pdf_path)
        else:
            record.report.title = title
            record.report.content_json = content_json
            record.report.pdf_path = pdf_path
        record.title = title
        self._session.flush()
        return record.report

    def set_pdf_path(self, session_id: str, pdf_path: str) -> None:
        record = self.get(session_id)
        if record.report is not None:
            record.report.pdf_path = pdf_path
            self._session.flush()

    def delete(self, session_id: str) -> None:
        self._session.delete(self.get(session_id))
        self._session.flush()


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        filename: str,
        stored_path: str,
        content_type: str,
        size_bytes: int,
        document_id: str | None = None,
    ) -> Document:
        record = Document(
            id=document_id or new_id(),
            filename=filename,
            stored_path=stored_path,
            content_type=content_type,
            size_bytes=size_bytes,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get(self, document_id: str) -> Document:
        record = self._session.get(Document, document_id)
        if record is None:
            raise NotFoundError(f"Document '{document_id}' not found")
        return record

    def list_all(self) -> list[Document]:
        stmt = select(Document).order_by(Document.created_at.desc(), Document.id)
        return list(self._session.scalars(stmt))

    def mark_processed(self, document_id: str, chunk_count: int) -> Document:
        record = self.get(document_id)
        record.status = DocumentStatus.PROCESSED
        record.chunk_count = chunk_count
        record.error = None
        self._session.flush()
        return record

    def mark_failed(self, document_id: str, error: str) -> Document:
        record = self.get(document_id)
        record.status = DocumentStatus.FAILED
        record.error = error
        self._session.flush()
        return record

    def delete(self, document_id: str) -> None:
        self._session.delete(self.get(document_id))
        self._session.flush()

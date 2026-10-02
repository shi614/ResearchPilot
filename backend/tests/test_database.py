from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.database import Database, DocumentRepository, ResearchRepository
from app.database.orm import DocumentStatus, EventStatus, RunEvent, SessionStatus
from app.exceptions import DatabaseError, NotFoundError


def test_research_session_lifecycle(database: Database) -> None:
    with database.session() as session:
        record = ResearchRepository(session).create("Impact of AI on healthcare", "Focus on 2024+")
        session_id = record.id

    with database.session() as session:
        repo = ResearchRepository(session)
        loaded = repo.get(session_id)
        assert loaded.status is SessionStatus.PENDING
        assert loaded.instructions == "Focus on 2024+"

        repo.update_status(session_id, SessionStatus.FAILED, error="Gemini quota exhausted")
        assert repo.get(session_id).error == "Gemini quota exhausted"


def test_events_are_ordered_and_support_incremental_polling(database: Database) -> None:
    with database.session() as session:
        repo = ResearchRepository(session)
        session_id = repo.create("q").id
        first = repo.add_event(session_id, "planner", EventStatus.STARTED)
        repo.add_event(session_id, "planner", EventStatus.COMPLETED, "3 research questions")
        repo.add_event(session_id, "web_researcher", EventStatus.STARTED)

    with database.session() as session:
        repo = ResearchRepository(session)
        assert [e.node for e in repo.list_events(session_id)] == [
            "planner",
            "planner",
            "web_researcher",
        ]
        newer = repo.list_events(session_id, after_id=first.id)
        assert len(newer) == 2
        assert newer[0].message == "3 research questions"


def test_save_report_creates_then_updates_and_sets_title(database: Database) -> None:
    with database.session() as session:
        repo = ResearchRepository(session)
        session_id = repo.create("q").id
        repo.save_report(session_id, "Draft title", '{"v": 1}')
        repo.save_report(session_id, "Final title", '{"v": 2}', pdf_path="reports/x.pdf")

    with database.session() as session:
        record = ResearchRepository(session).get(session_id)
        assert record.title == "Final title"
        assert record.report is not None
        assert record.report.content_json == '{"v": 2}'
        assert record.report.pdf_path == "reports/x.pdf"


def test_list_recent_returns_newest_first(database: Database) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with database.session() as session:
        repo = ResearchRepository(session)
        for minutes, query in enumerate(("first", "second", "third")):
            repo.create(query).created_at = start + timedelta(minutes=minutes)

    with database.session() as session:
        assert [r.query for r in ResearchRepository(session).list_recent(limit=2)] == [
            "third",
            "second",
        ]


def test_deleting_session_cascades_to_events_and_report(database: Database) -> None:
    with database.session() as session:
        repo = ResearchRepository(session)
        session_id = repo.create("q").id
        repo.add_event(session_id, "planner", EventStatus.COMPLETED)
        repo.save_report(session_id, "t", "{}")

    with database.session() as session:
        ResearchRepository(session).delete(session_id)

    with database.session() as session:
        assert session.query(RunEvent).count() == 0
        with pytest.raises(NotFoundError):
            ResearchRepository(session).get(session_id)


def test_document_status_transitions(database: Database) -> None:
    with database.session() as session:
        repo = DocumentRepository(session)
        doc_id = repo.create("paper.pdf", "data/uploads/paper.pdf", "application/pdf", 1024).id
        repo.mark_processed(doc_id, chunk_count=12)

    with database.session() as session:
        repo = DocumentRepository(session)
        doc = repo.get(doc_id)
        assert doc.status is DocumentStatus.PROCESSED
        assert doc.chunk_count == 12
        repo.mark_failed(doc_id, "Could not parse PDF")
        assert repo.get(doc_id).status is DocumentStatus.FAILED
        repo.delete(doc_id)
        assert repo.list_all() == []


def test_missing_records_raise_not_found(database: Database) -> None:
    with database.session() as session:
        with pytest.raises(NotFoundError):
            ResearchRepository(session).add_event("does-not-exist", "planner", EventStatus.STARTED)
        with pytest.raises(NotFoundError):
            DocumentRepository(session).get("does-not-exist")


def test_sqlalchemy_errors_are_wrapped_and_rolled_back(database: Database) -> None:
    with pytest.raises(DatabaseError):
        with database.session() as session:
            ResearchRepository(session).create(None)  # type: ignore[arg-type]  # NOT NULL violation

    with database.session() as session:
        assert ResearchRepository(session).list_recent() == []


def test_health_check_reports_status(database: Database) -> None:
    assert database.is_healthy() is True

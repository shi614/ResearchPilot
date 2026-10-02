from __future__ import annotations

from app.database import Database, ResearchRepository
from app.database.orm import SessionStatus
from app.models.domain import Citation, SourceType, VerificationStatus
from app.services.report_render import StoredReport, render_markdown
from tests.fakes import report


def stored() -> StoredReport:
    return StoredReport(
        report=report("Evidence [W1] [K1]"),
        citations=[
            Citation(id="W1", title="IEA study", reference="https://iea.org/x", source_type=SourceType.WEB,
                     verification_status=VerificationStatus.CORROBORATED, excerpt="..."),
            Citation(id="K1", title="notes.pdf, p. 2", reference="notes.pdf, p. 2",
                     source_type=SourceType.KNOWLEDGE_BASE, verification_status=VerificationStatus.UNVERIFIED,
                     excerpt="..."),
        ],
        quality_notes=["1 of 2 cited sources are not corroborated by a second source."],
    )


def test_stored_report_round_trips_through_json() -> None:
    original = stored()
    restored = StoredReport.from_json(original.to_json())
    assert restored == original


def test_markdown_contains_sections_references_and_verification_labels() -> None:
    markdown = render_markdown(stored())
    for heading in ("# Solar Energy Cost Trends", "## Executive Summary", "## Methodology",
                    "## Limitations", "## Conclusion", "## References", "## Quality Notes"):
        assert heading in markdown
    assert "**[W1]** IEA study. Web: https://iea.org/x _(corroborated by multiple sources)_" in markdown
    assert "**[K1]** notes.pdf, p. 2. Knowledge base: notes.pdf, p. 2 _(unverified)_" in markdown


def test_markdown_without_citations_says_so() -> None:
    empty = StoredReport(report=report("no evidence"), citations=[])
    assert "_No sources were cited._" in render_markdown(empty)


def test_new_statuses_persist_and_history_can_filter(database: Database) -> None:
    with database.session() as session:
        repo = ResearchRepository(session)
        for status in (SessionStatus.CANCELLED, SessionStatus.INTERRUPTED, SessionStatus.COMPLETED):
            repo.update_status(repo.create(f"q-{status.value}").id, status)

    with database.session() as session:
        repo = ResearchRepository(session)
        assert {r.status for r in repo.list_recent()} == {
            SessionStatus.CANCELLED, SessionStatus.INTERRUPTED, SessionStatus.COMPLETED,
        }
        interrupted = repo.list_recent(statuses={SessionStatus.INTERRUPTED})
        assert [r.query for r in interrupted] == ["q-interrupted"]


def test_new_statuses_work_on_a_database_created_before_they_existed(tmp_path) -> None:
    # Enum values are stored as plain strings, so existing databases need no migration.
    import sqlite3

    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE research_sessions (id VARCHAR(32) PRIMARY KEY, query TEXT, instructions TEXT, "
        "status VARCHAR(32), title VARCHAR(300), error TEXT, created_at DATETIME, updated_at DATETIME)"
    )
    connection.commit()
    connection.close()
    db = Database(f"sqlite:///{path.as_posix()}")
    db.create_tables()
    with db.session() as session:
        repo = ResearchRepository(session)
        repo.update_status(repo.create("q").id, SessionStatus.INTERRUPTED)
    with db.session() as session:
        assert ResearchRepository(session).list_recent()[0].status is SessionStatus.INTERRUPTED
    db.dispose()

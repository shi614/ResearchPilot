"""Research sessions: runs the LangGraph workflow in the background and keeps the
database (sessions, live events, reports) in sync with it.

- Each session is a LangGraph thread (thread_id == session id).
- Work runs on a small thread pool; API calls return immediately.
- Only one operation per session at a time (start / decision / retry).
- Every agent step is recorded as a RunEvent, which the UI polls for live progress.
- At startup, sessions left "running" by a previous process become "interrupted"
  (retryable from their checkpoint).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

from app.config import Settings
from app.database import Database, ResearchRepository
from app.database.orm import RETRYABLE_STATUSES, EventStatus, ResearchSession, SessionStatus
from app.exceptions import ConfigurationError, NotFoundError, WorkflowError
from app.graph.runner import ResearchRunner, RunOutcome, UpdateCallback
from app.models.agent_outputs import HumanDecision
from app.models.domain import SourceType
from app.models.schemas import (
    ReportResponse,
    ResearchEventResponse,
    ResearchSessionResponse,
    ResearchStats,
    ResearchSummary,
)
from app.services.report_render import StoredReport, render_markdown

logger = logging.getLogger(__name__)

RunnerFactory = Callable[[UpdateCallback], ResearchRunner]

_OUTCOME_TO_STATUS = {
    "awaiting_approval": SessionStatus.AWAITING_APPROVAL,
    "completed": SessionStatus.COMPLETED,
    "cancelled": SessionStatus.CANCELLED,
    "failed": SessionStatus.FAILED,
    "quota_exhausted": SessionStatus.QUOTA_EXHAUSTED,
    "interrupted": SessionStatus.INTERRUPTED,
}
RESTART_MESSAGE = "The backend stopped while this research was running. Retry to continue."


class ResearchService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        runner_factory: RunnerFactory,
        executor: Executor | None = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._runner_factory = runner_factory
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="research")
        self._runner: ResearchRunner | None = None
        self._runner_lock = threading.Lock()
        self._active: set[str] = set()
        self._active_lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    # ------------------------------------------------------------------ lifecycle

    def runner(self) -> ResearchRunner:
        """The shared workflow runner (one Gemini gateway → one RPM budget for all runs)."""
        with self._runner_lock:
            if self._runner is None:
                missing = self._settings.missing_required_keys()
                if "GEMINI_API_KEY" in missing:
                    raise ConfigurationError("GEMINI_API_KEY is not set; research is unavailable.")
                self._runner = self._runner_factory(self._record_update)
            return self._runner

    def recover_interrupted(self) -> int:
        """Mark sessions left running by a previous process as interrupted."""
        with self._database.session() as session:
            repo = ResearchRepository(session)
            stale = repo.list_recent(limit=1000, statuses={SessionStatus.RUNNING})
            for record in stale:
                repo.update_status(record.id, SessionStatus.INTERRUPTED, error=RESTART_MESSAGE)
        if stale:
            logger.warning("Marked %d interrupted research session(s) for retry", len(stale))
        return len(stale)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def wait(self, session_id: str, timeout: float | None = None) -> None:
        """Block until the session's current background operation finishes (tests/CLI)."""
        future = self._futures.get(session_id)
        if future is not None:
            future.result(timeout=timeout)

    # ------------------------------------------------------------------ commands

    def start(self, query: str, instructions: str | None = None) -> ResearchSessionResponse:
        runner = self.runner()  # fail fast (503) before creating a session if keys are missing
        query = query.strip()
        instructions = (instructions or "").strip() or None
        with self._database.session() as session:
            record = ResearchRepository(session).create(query, instructions)
            ResearchRepository(session).update_status(record.id, SessionStatus.RUNNING)
            session_id = record.id
        self._reserve(session_id)
        self._launch(session_id, lambda: runner.start(session_id, query, instructions))
        return self.get(session_id)

    def decide(self, session_id: str, decision: HumanDecision) -> ResearchSessionResponse:
        runner = self.runner()
        self._reserve(session_id)  # claim first, then validate: no double-submit race
        try:
            self._require_status(session_id, {SessionStatus.AWAITING_APPROVAL}, "is not waiting for approval")
        except Exception:
            self._release(session_id)
            raise
        self._launch(session_id, lambda: runner.resume(session_id, decision))
        return self.get(session_id)

    def retry(self, session_id: str) -> ResearchSessionResponse:
        runner = self.runner()
        self._reserve(session_id)
        try:
            self._require_status(session_id, RETRYABLE_STATUSES, "cannot be retried")
            if not runner.outcome(session_id).pending_nodes:
                raise WorkflowError("This research has no unfinished steps to retry.")
        except Exception:
            self._release(session_id)
            raise
        self._launch(session_id, lambda: runner.retry(session_id))
        return self.get(session_id)

    def delete(self, session_id: str) -> None:
        if self._is_active(session_id):
            raise WorkflowError("This research is still running; wait for it to pause or finish.")
        with self._database.session() as session:
            ResearchRepository(session).delete(session_id)
        try:
            self.runner().delete_thread(session_id)
        except ConfigurationError:
            pass  # without keys no checkpoints can exist for new runs; nothing to clean up

    # ------------------------------------------------------------------ queries

    def get(self, session_id: str, after_event: int = 0) -> ResearchSessionResponse:
        with self._database.session() as session:
            repo = ResearchRepository(session)
            record = repo.get(session_id)
            events = [ResearchEventResponse.model_validate(e) for e in repo.list_events(session_id, after_event)]
            response = ResearchSessionResponse(
                id=record.id,
                query=record.query,
                instructions=record.instructions,
                status=record.status,
                title=record.title,
                error=record.error,
                created_at=record.created_at,
                updated_at=record.updated_at,
                has_report=record.report is not None,
                events=events,
            )
        self._add_workflow_details(response, record_status=response.status)
        return response

    def list_sessions(self, limit: int = 50) -> list[ResearchSummary]:
        with self._database.session() as session:
            return [self._summary(r) for r in ResearchRepository(session).list_recent(limit)]

    def report(self, session_id: str) -> ReportResponse:
        with self._database.session() as session:
            record = ResearchRepository(session).get(session_id)
            if record.report is None:
                raise NotFoundError("This research has no report yet.")
            stored = StoredReport.from_json(record.report.content_json)
            created_at = record.report.created_at
        return ReportResponse(
            session_id=session_id,
            title=stored.report.title,
            markdown=render_markdown(stored),
            report=stored.report,
            citations=stored.citations,
            quality_notes=stored.quality_notes,
            created_at=created_at,
        )

    # ------------------------------------------------------------------ internals

    def _reserve(self, session_id: str) -> None:
        """Atomically claim a session for one operation."""
        with self._active_lock:
            if session_id in self._active:
                raise WorkflowError("Another operation is already running for this research.")
            self._active.add(session_id)

    def _release(self, session_id: str) -> None:
        with self._active_lock:
            self._active.discard(session_id)

    def _launch(self, session_id: str, operation: Callable[[], RunOutcome]) -> None:
        """Run a reserved operation in the background (releases the claim when done)."""
        try:
            self._set_status(session_id, SessionStatus.RUNNING)
            self._futures[session_id] = self._executor.submit(self._execute, session_id, operation)
        except Exception:
            self._release(session_id)
            raise

    def _execute(self, session_id: str, operation: Callable[[], RunOutcome]) -> None:
        try:
            outcome = operation()
            self._apply_outcome(session_id, outcome)
        except Exception as exc:  # noqa: BLE001 - background work must always settle the session
            logger.exception("Research %s failed in the background", session_id)
            message = getattr(exc, "message", None) or f"Unexpected error: {type(exc).__name__}"
            self._set_status(session_id, SessionStatus.FAILED, error=message)
        finally:
            self._release(session_id)

    def _apply_outcome(self, session_id: str, outcome: RunOutcome) -> None:
        status = _OUTCOME_TO_STATUS.get(outcome.status, SessionStatus.FAILED)
        with self._database.session() as session:
            repo = ResearchRepository(session)
            if status is SessionStatus.COMPLETED and outcome.values.get("final_report") is not None:
                stored = StoredReport.from_state(outcome.values)
                repo.save_report(session_id, stored.report.title, stored.to_json())
            repo.update_status(session_id, status, error=outcome.error)
        logger.info("Research %s → %s", session_id, status.value)

    def _record_update(self, session_id: str, node: str, update: dict[str, Any]) -> None:
        """Runner callback: persist each agent step as live progress events."""
        try:
            with self._database.session() as session:
                repo = ResearchRepository(session)
                for event in update.get("progress", []):
                    repo.add_event(session_id, event.node, EventStatus.COMPLETED, event.message)
                for error in update.get("errors", []):
                    repo.add_event(session_id, error.node, EventStatus.FAILED, error.message)
        except Exception:  # noqa: BLE001 - progress logging must never break the run
            logger.exception("Could not record progress for %s/%s", session_id, node)

    def _add_workflow_details(self, response: ResearchSessionResponse, record_status: SessionStatus) -> None:
        """Enrich with live checkpoint data (approval request, counters, pending steps)."""
        try:
            outcome = self.runner().outcome(response.id)
        except ConfigurationError:
            return
        values = outcome.values
        if not values:
            return
        if record_status is SessionStatus.AWAITING_APPROVAL:
            response.approval_request = outcome.approval_request
        response.pending_nodes = list(outcome.pending_nodes)
        response.retryable = record_status in RETRYABLE_STATUSES and bool(outcome.pending_nodes)
        response.errors = [e.message for e in values.get("errors", [])]
        evidence = values.get("evidence", [])
        response.stats = ResearchStats(
            llm_calls=values.get("llm_calls", 0),
            web_searches=values.get("web_searches", 0),
            web_sources=len(values.get("web_sources", [])),
            knowledge_base_sources=len(values.get("retrieved_documents", [])),
            evidence=len(evidence),
            evidence_web=sum(e.source.source_type is SourceType.WEB for e in evidence),
            evidence_knowledge_base=sum(e.source.source_type is SourceType.KNOWLEDGE_BASE for e in evidence),
            iteration=values.get("iteration_count", 0),
            revisions=values.get("revision_count", 0),
        )

    def _require_status(self, session_id: str, allowed: set[SessionStatus], problem: str) -> None:
        with self._database.session() as session:
            status = ResearchRepository(session).get(session_id).status
        if status not in allowed:
            raise WorkflowError(f"This research {problem} (status: {status.value}).")

    def _set_status(self, session_id: str, status: SessionStatus, error: str | None = None) -> None:
        with self._database.session() as session:
            ResearchRepository(session).update_status(session_id, status, error=error)

    def _is_active(self, session_id: str) -> bool:
        with self._active_lock:
            return session_id in self._active

    @staticmethod
    def _summary(record: ResearchSession) -> ResearchSummary:
        return ResearchSummary(
            id=record.id,
            query=record.query,
            status=record.status,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
            has_report=record.report is not None,
        )

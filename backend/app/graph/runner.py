"""Runs, pauses and resumes research workflows on top of the checkpointed graph.

Each research session is a LangGraph thread. The runner never loses work:
- human approval pauses the run (interrupt) until `resume()` is called;
- a Gemini quota error stops the run with status `quota_exhausted`; `retry()`
  continues from the last checkpoint (completed steps are not re-run);
- every node update is reported through `on_update` (used for live progress).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.exceptions import RateLimitError, ResearchPilotError, WorkflowError
from app.models.agent_outputs import HumanDecision

logger = logging.getLogger(__name__)

UpdateCallback = Callable[[str, str, dict[str, Any]], None]  # (thread_id, node, update)
TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


@dataclass
class RunOutcome:
    thread_id: str
    status: str
    values: dict[str, Any] = field(default_factory=dict)
    approval_request: dict[str, Any] | None = None
    error: str | None = None
    pending_nodes: tuple[str, ...] = ()

    @property
    def retryable(self) -> bool:
        """Stopped mid-workflow (quota, service outage): `retry()` continues from the checkpoint."""
        return self.status in ("failed", "quota_exhausted") and bool(self.pending_nodes)


class ResearchRunner:
    def __init__(self, graph: CompiledStateGraph, on_update: UpdateCallback | None = None) -> None:
        self._graph = graph
        self._on_update = on_update

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    # ------------------------------------------------------------------ public API

    def start(self, thread_id: str, query: str, instructions: str | None = None) -> RunOutcome:
        if self._graph.get_state(self._config(thread_id)).values:
            raise WorkflowError(f"Research run '{thread_id}' already exists.")
        initial = {
            "user_query": query,
            "instructions": instructions,
            "iteration_count": 0,
            "revision_count": 0,
            "status": "planning",
        }
        return self._run(thread_id, initial)

    def resume(self, thread_id: str, decision: HumanDecision) -> RunOutcome:
        """Answer the human-approval pause (approve / modify / cancel)."""
        if not self._graph.get_state(self._config(thread_id)).interrupts:
            raise WorkflowError("This run is not waiting for approval.")
        return self._run(thread_id, Command(resume=decision.model_dump()))

    def retry(self, thread_id: str) -> RunOutcome:
        """Continue a run that stopped on an error (e.g. quota), from its last checkpoint."""
        snapshot = self._graph.get_state(self._config(thread_id))
        if not snapshot.values:
            raise WorkflowError(f"Research run '{thread_id}' does not exist.")
        if snapshot.interrupts:
            raise WorkflowError("This run is waiting for approval; use resume instead.")
        if not snapshot.next:
            raise WorkflowError("This run has already finished.")
        return self._run(thread_id, None)

    def outcome(self, thread_id: str) -> RunOutcome:
        return self._outcome(thread_id)

    # ------------------------------------------------------------------ internals

    def _run(self, thread_id: str, graph_input: Any) -> RunOutcome:
        try:
            for chunk in self._graph.stream(graph_input, self._config(thread_id), stream_mode="updates"):
                for node, update in chunk.items():
                    if node != "__interrupt__" and self._on_update is not None:
                        self._on_update(thread_id, node, update or {})
        except RateLimitError as exc:
            logger.warning("Run %s paused by quota: %s", thread_id, exc.message)
            return self._outcome(thread_id, status="quota_exhausted", error=exc.message)
        except ResearchPilotError as exc:
            logger.error("Run %s failed: %s", thread_id, exc.message)
            return self._outcome(thread_id, status="failed", error=exc.message)
        except Exception as exc:  # noqa: BLE001 - never crash the caller; the checkpoint survives
            logger.exception("Run %s crashed", thread_id)
            return self._outcome(thread_id, status="failed", error=f"Unexpected error: {type(exc).__name__}")
        return self._outcome(thread_id)

    def _outcome(self, thread_id: str, status: str | None = None, error: str | None = None) -> RunOutcome:
        snapshot = self._graph.get_state(self._config(thread_id))
        values = dict(snapshot.values)
        approval = snapshot.interrupts[0].value if snapshot.interrupts else None
        if approval is not None:
            status = "awaiting_approval"
        elif status is None:
            status = values.get("status", "failed")
            if not snapshot.next and status not in TERMINAL_STATUSES:
                status = "failed"  # the graph ended early (e.g. no sources were found)
        if error is None and status == "failed" and values.get("errors"):
            error = values["errors"][-1].message
        return RunOutcome(
            thread_id=thread_id,
            status=status,
            values=values,
            approval_request=approval,
            error=error,
            pending_nodes=tuple(snapshot.next),
        )

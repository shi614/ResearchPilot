"""Derive the workflow stage display from real backend state (events, pending nodes, status).

Nothing here is simulated: a stage is "done" only if the backend recorded an event
for it, and "active" only if the checkpoint says it is the next step to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

StageState = Literal["done", "active", "waiting", "warning", "failed", "skipped", "pending"]

STAGES: list[tuple[str, str]] = [
    ("planner", "Planner"),
    ("web_research", "Web Research"),
    ("rag_research", "Knowledge Retrieval"),
    ("source_verification", "Source Verification"),
    ("analysis", "Analysis"),
    ("human_review", "Human Approval"),
    ("writer", "Report Generation"),
    ("critic", "Quality Review"),
    ("revision", "Revision"),
    ("finalize", "Final Report"),
]
STAGE_INDEX = {node: i for i, (node, _) in enumerate(STAGES)}

STOPPED = {"failed", "quota_exhausted", "interrupted"}
ACTIVE = {"running", "pending"}

STATUS_BADGES: dict[str, tuple[str, str, str]] = {
    # status: (label, colour, material icon)
    "pending": ("Queued", "gray", ":material/schedule:"),
    "running": ("Running", "blue", ":material/progress_activity:"),
    "awaiting_approval": ("Awaiting your approval", "orange", ":material/front_hand:"),
    "completed": ("Completed", "green", ":material/check_circle:"),
    "failed": ("Failed", "red", ":material/error:"),
    "quota_exhausted": ("Quota exhausted", "red", ":material/hourglass_disabled:"),
    "interrupted": ("Interrupted", "orange", ":material/pause_circle:"),
    "cancelled": ("Cancelled", "gray", ":material/cancel:"),
}

STATE_ICONS: dict[StageState, str] = {
    "done": ":material/check_circle:",
    "active": ":material/progress_activity:",
    "waiting": ":material/front_hand:",
    "warning": ":material/warning:",
    "failed": ":material/error:",
    "skipped": ":material/remove_circle_outline:",
    "pending": ":material/radio_button_unchecked:",
}


@dataclass(frozen=True)
class StageView:
    node: str
    label: str
    state: StageState
    detail: str = ""


def stage_views(session: dict[str, Any]) -> list[StageView]:
    status = session.get("status", "pending")
    events = session.get("events", [])
    pending = set(session.get("pending_nodes") or [])

    completed: dict[str, str] = {}
    warnings: dict[str, str] = {}
    for event in events:
        if event["status"] == "failed":
            warnings[event["node"]] = event.get("message") or "degraded"
        else:
            completed[event["node"]] = event.get("message") or ""

    furthest = max((STAGE_INDEX[n] for n in completed if n in STAGE_INDEX), default=-1)
    if status in ACTIVE and not pending and not completed:
        pending = {"planner"}  # just started; the first checkpoint is not written yet

    views = []
    for node, label in STAGES:
        index = STAGE_INDEX[node]
        if node in pending:
            if status == "awaiting_approval":
                views.append(StageView(node, label, "waiting", "Review the findings below"))
            elif status in STOPPED:
                views.append(StageView(node, label, "failed", "Stopped here - retry to continue"))
            elif status in ACTIVE:
                views.append(StageView(node, label, "active", "In progress"))
            else:
                views.append(StageView(node, label, "pending"))
        elif node in completed:
            state: StageState = "warning" if node in warnings else "done"
            views.append(StageView(node, label, state, warnings.get(node) or completed[node]))
        elif node in warnings:
            views.append(StageView(node, label, "warning", warnings[node]))
        elif index < furthest:
            reason = "Not needed" if node == "revision" else "Skipped"
            views.append(StageView(node, label, "skipped", reason))
        else:
            views.append(StageView(node, label, "pending"))
    return views


def progress_fraction(views: list[StageView], status: str) -> float:
    if status == "completed":
        return 1.0
    finished = sum(v.state in ("done", "warning", "skipped") for v in views)
    return min(finished / len(views), 0.99)


def is_active(status: str) -> bool:
    return status in ACTIVE

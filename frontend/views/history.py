"""Research History: a dashboard of research cards (open, delete) with summary metrics."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components import call, client, go_to, local_time, open_session, quietly
from frontend.ui.components import empty_state, html_block, metric_row, page_header, research_card_html

STATS_CACHE_KEY = "_history_stats"
STATUS_DESCRIPTIONS = {
    "pending": "The agents are researching this question.",
    "running": "The agents are researching this question.",
    "awaiting_approval": "Findings are ready for your review.",
    "failed": "Stopped before the report was finished.",
    "interrupted": "Paused - you can continue from the last checkpoint.",
    "quota_exhausted": "Paused by the AI usage limit - continue later.",
    "cancelled": "This research was cancelled.",
    "completed": "Report ready to read and download.",
}


def render() -> None:
    page_header("Research History", "Revisit earlier research, continue paused runs or download reports.")
    st.write("")
    sessions = call(lambda: client().list_research(), failure="Could not load research history")
    if sessions is None:
        return
    if not sessions:
        empty_state("history", "Your research history is empty", "Start your first research task to see it here.")
        _, middle, _ = st.columns([2, 1.2, 2])
        if middle.button("Start New Research", type="primary", icon=":material/add:", width="stretch",
                         key="empty_new"):
            open_session(None)
            go_to("research")
        return

    statuses = [s["status"] for s in sessions]
    metric_row([
        ("Total research", len(sessions), "folder_open"),
        ("Completed", statuses.count("completed"), "task_alt"),
        ("Waiting for approval", statuses.count("awaiting_approval"), "front_hand"),
        ("In progress", sum(s in ("running", "pending") for s in statuses), "progress_activity"),
    ])
    st.write("")
    for start in range(0, len(sessions), 2):
        columns = st.columns(2, gap="medium")
        for column, session in zip(columns, sessions[start:start + 2], strict=False):
            with column:
                _research_card(session)


def _stats(session: dict[str, Any]) -> dict[str, Any]:
    """Source/revision counts from the session detail endpoint, cached until the session changes."""
    cache = st.session_state.setdefault(STATS_CACHE_KEY, {})
    cached = cache.get(session["id"])
    if cached and cached[0] == session["updated_at"]:
        return cached[1]
    detail = quietly(lambda: client().get_research(session["id"])) or {}
    stats = detail.get("stats") or {}
    cache[session["id"]] = (session["updated_at"], stats)
    return stats


def _research_card(session: dict[str, Any]) -> None:
    stats = _stats(session)
    title = session.get("title") or session["query"]
    description = session["query"] if session.get("title") else STATUS_DESCRIPTIONS.get(session["status"], "")
    session_id = session["id"]
    with st.container(key=f"card_history_{session_id}", height="stretch"):
        html_block(research_card_html(
            title=title,
            description=description,
            when=local_time(session["created_at"]),
            status=session["status"],
            sources=stats.get("evidence") if stats else None,
            revisions=stats.get("revisions") if stats else None,
        ))
        st.write("")
        open_col, delete_col = st.columns([2.2, 1])
        primary = session["status"] in ("awaiting_approval", "interrupted", "quota_exhausted")
        if open_col.button("Open Research", icon=":material/arrow_forward:", width="stretch",
                           key=f"open_{session_id}", type="primary" if primary else "secondary"):
            open_session(session_id)
            go_to("research")
        with delete_col, st.container(key=f"danger_delete_{session_id}"):
            with st.popover("Delete", icon=":material/delete:", width="stretch"):
                st.markdown("Delete this research permanently?")
                st.caption("Its report, PDF and progress will be removed.")
                if st.button("Delete research", key=f"confirm_delete_{session_id}", type="primary"):
                    if call(lambda: client().delete_research(session_id) or True, failure="Could not delete"):
                        if st.session_state.get("active_session_id") == session_id:
                            open_session(None)
                        st.toast("Research deleted", icon=":material/delete:")
                        st.rerun()

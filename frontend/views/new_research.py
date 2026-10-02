"""Main page: start a research run and follow it live through approval to the final report."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components import (
    ACTIVE_SESSION_KEY,
    activity_log,
    approval_panel,
    call,
    client,
    open_session,
    report_view,
    stage_tracker,
    stats_row,
    status_badge,
)
from frontend.workflow import is_active

POLL_SECONDS = 2


def render() -> None:
    st.title("ResearchPilot")
    st.caption("Multi-Agent AI Research & Report Generation System")
    session_id = st.session_state.get(ACTIVE_SESSION_KEY)
    if session_id:
        _session_view(session_id)
    else:
        _start_form()


# --------------------------------------------------------------------------- start


def _start_form() -> None:
    config = call(lambda: client().config(), failure="Could not load backend settings")
    if config is not None and not config["research_enabled"]:
        st.error("Research is disabled: GEMINI_API_KEY is not configured on the backend. "
                 "Add it to `.env` and restart the backend.", icon=":material/key_off:")
    elif config is not None and not config["web_research_enabled"]:
        st.warning("TAVILY_API_KEY is not configured: research will use the knowledge base only.",
                   icon=":material/travel_explore:")

    with st.form("start_research", border=True):
        query = st.text_area(
            "Research topic or question",
            placeholder="e.g. What are the main benefits and limitations of Retrieval-Augmented Generation?",
            max_chars=2000,
        )
        instructions = st.text_area(
            "Optional instructions",
            placeholder="e.g. Focus on enterprise use cases and research published since 2023",
            max_chars=2000,
            height=80,
        )
        files = st.file_uploader(
            "Add documents to the knowledge base (optional)",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            help="Uploaded files are chunked, embedded and searched by the knowledge-base agent.",
        )
        submitted = st.form_submit_button("Start Research", type="primary", icon=":material/rocket_launch:",
                                          disabled=config is not None and not config["research_enabled"])
    if limits := config:
        st.caption(
            f"Model `{limits['gemini_model']}` · up to {limits['max_web_searches']} web searches per iteration · "
            f"{limits['max_revisions']} critic revision(s) · approval required before the final report"
        )
    if submitted:
        _start(query.strip(), instructions.strip(), files or [])


def _start(query: str, instructions: str, files: list[Any]) -> None:
    if len(query) < 3:
        st.warning("Please enter a research topic (at least 3 characters).")
        return
    for file in files:
        with st.spinner(f"Uploading and indexing {file.name}..."):
            document = call(lambda f=file: client().upload_document(f.name, f.getvalue()),
                            failure=f"Could not upload {file.name}")
        if document and document["status"] == "failed":
            st.warning(f"{file.name} was saved but could not be indexed: {document['error']}")
    session = call(lambda: client().start_research(query, instructions or None), failure="Could not start research")
    if session:
        open_session(session["id"])
        st.rerun()


# --------------------------------------------------------------------------- live session


def _session_view(session_id: str) -> None:
    session = call(lambda: client().get_research(session_id), failure="Could not load this research")
    if session is None:
        if st.button("Start a new research", icon=":material/add:"):
            open_session(None)
            st.rerun()
        return
    running = is_active(session["status"])
    # While agents run, only this fragment re-renders every few seconds (real backend polling).
    st.fragment(_live_panel, run_every=POLL_SECONDS if running else None)(session_id, session["status"])


def _live_panel(session_id: str, rendered_status: str) -> None:
    session = call(lambda: client().get_research(session_id), failure="Could not refresh progress")
    if session is None:
        return
    status = session["status"]
    if status != rendered_status:
        st.rerun(scope="app")  # status changed: re-render the page (stops/starts polling)

    header, actions = st.columns([4, 1])
    with header:
        st.subheader(session.get("title") or session["query"])
        status_badge(status)
        if session.get("instructions"):
            st.caption(f"Instructions: {session['instructions']}")
    with actions:
        if st.button("New research", icon=":material/add:", width="stretch", key="new_research"):
            open_session(None)
            st.rerun(scope="app")

    stage_tracker(session)
    stats_row(session)

    if status in ("running", "pending"):
        st.info("The agents are working. This page updates automatically.", icon=":material/autorenew:")
    elif status == "awaiting_approval":
        approval_panel(session)
    elif status == "completed":
        report_view(session_id)
    elif status == "cancelled":
        st.info("This research was cancelled. No report was generated.", icon=":material/cancel:")
    else:
        _stopped_panel(session)

    activity_log(session)


def _stopped_panel(session: dict[str, Any]) -> None:
    messages = {
        "quota_exhausted": "The Gemini free-tier quota was exhausted.",
        "interrupted": "The research stopped before finishing.",
        "failed": "The research stopped because of an error.",
    }
    st.error(f"{messages.get(session['status'], 'The research stopped.')} {session.get('error') or ''}",
             icon=":material/error:")
    if session.get("retryable"):
        st.caption("All completed steps are saved. Retrying continues from the step that failed.")
        if st.button("Retry from last checkpoint", type="primary", icon=":material/replay:", key="retry"):
            if call(lambda: client().retry(session["id"]), failure="Could not retry"):
                st.rerun(scope="app")

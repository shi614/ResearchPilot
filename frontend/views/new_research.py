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
    page_header,
    report_view,
    stage_tracker,
    stats_row,
    status_badge,
)
from frontend.workflow import is_active

POLL_SECONDS = 2

HOW_IT_WORKS = [
    (":material/travel_explore:", "Plan & search", "Agents plan the research and search the web and your documents."),
    (":material/fact_check:", "Verify sources", "Sources are cross-checked; weak or conflicting claims are flagged."),
    (":material/front_hand:", "You review", "Approve the findings or ask for more research before writing."),
    (":material/description:", "Cited report", "A structured, fact-checked report with references and a PDF."),
]


def render() -> None:
    session_id = st.session_state.get(ACTIVE_SESSION_KEY)
    if session_id:
        _session_view(session_id)
    else:
        page_header("ResearchPilot", "Multi-Agent AI Research & Report Generation System")
        _start_form()
        _how_it_works()


# --------------------------------------------------------------------------- start


def _start_form() -> None:
    config = call(lambda: client().config(), failure="Could not load the app settings")
    research_enabled = config is None or config["research_enabled"]
    if config is not None and not config["research_enabled"]:
        st.error("Research is unavailable because the AI service isn't configured yet. "
                 "See the setup guide in the README.", icon=":material/key_off:")
    elif config is not None and not config["web_research_enabled"]:
        st.warning("Web search is unavailable right now, so research will use your knowledge base only.",
                   icon=":material/travel_explore:")

    with st.form("start_research", border=True):
        query = st.text_area(
            "What would you like to research?",
            placeholder="e.g. What are the main benefits and limitations of Retrieval-Augmented Generation?",
            max_chars=2000,
        )
        with st.expander("Add instructions or documents (optional)", icon=":material/tune:"):
            instructions = st.text_area(
                "Instructions",
                placeholder="e.g. Focus on enterprise use cases and research published since 2023",
                max_chars=2000,
                height=80,
            )
            files = st.file_uploader(
                "Documents to search alongside the web",
                type=["pdf", "txt", "md", "markdown"],
                accept_multiple_files=True,
                help="Added to your knowledge base so the agents can cite them.",
            )
        submitted = st.form_submit_button("Start Research", type="primary", icon=":material/rocket_launch:",
                                          disabled=not research_enabled)
    st.caption(":material/info: Research pauses for your review before the final report is written.")
    if submitted:
        _start(query.strip(), instructions.strip(), files or [])


def _how_it_works() -> None:
    st.markdown('<p class="rp-section">How it works</p>', unsafe_allow_html=True)
    for column, (icon, title, text) in zip(st.columns(4), HOW_IT_WORKS, strict=True):
        with column, st.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)


def _start(query: str, instructions: str, files: list[Any]) -> None:
    if len(query) < 3:
        st.warning("Please enter a research topic (at least 3 characters).")
        return
    for file in files:
        with st.spinner(f"Adding {file.name} to your knowledge base..."):
            document = call(lambda f=file: client().upload_document(f.name, f.getvalue()),
                            failure=f"Could not add {file.name}")
        if document and document["status"] == "failed":
            st.warning(f"{file.name} was saved but could not be read: {document['error']}")
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

    header, actions = st.columns([4, 1], vertical_alignment="center")
    with header:
        st.markdown('<p class="rp-eyebrow">Research</p>', unsafe_allow_html=True)
        st.header(session.get("title") or session["query"], anchor=False)
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
        "quota_exhausted": "The AI service's usage limit has been reached for now.",
        "interrupted": "The research stopped before finishing.",
        "failed": "The research stopped because of an error.",
    }
    st.error(f"{messages.get(session['status'], 'The research stopped.')} {session.get('error') or ''}",
             icon=":material/error:")
    if session.get("retryable"):
        st.caption("Completed steps are saved. Retrying continues from the step that stopped.")
        if st.button("Retry from last checkpoint", type="primary", icon=":material/replay:", key="retry"):
            if call(lambda: client().retry(session["id"]), failure="Could not retry"):
                st.rerun(scope="app")

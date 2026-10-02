"""Reusable UI pieces: client access, error display, status badges, stage tracker,
approval panel and report view."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

from frontend.api_client import ApiError, BackendUnavailable, ResearchPilotClient
from frontend.workflow import STATE_ICONS, STATUS_BADGES, progress_fraction, stage_views

T = TypeVar("T")

CLIENT_KEY = "api_client"
ACTIVE_SESSION_KEY = "active_session_id"
PAGES_KEY = "_pages"


def client() -> ResearchPilotClient:
    if CLIENT_KEY not in st.session_state:
        st.session_state[CLIENT_KEY] = ResearchPilotClient()
    return st.session_state[CLIENT_KEY]


def call(action: Callable[[], T], *, failure: str = "Request failed") -> T | None:
    """Run an API call; show a clear error instead of a stack trace on failure."""
    try:
        return action()
    except BackendUnavailable as exc:
        st.error(exc.message, icon=":material/cloud_off:")
    except ApiError as exc:
        st.error(f"{failure}: {exc.message}", icon=":material/error:")
    return None


def open_session(session_id: str | None) -> None:
    st.session_state[ACTIVE_SESSION_KEY] = session_id
    for key in [k for k in st.session_state if str(k).startswith(("report:", "pdf:"))]:
        del st.session_state[key]


def go_to(page: str) -> None:
    pages = st.session_state.get(PAGES_KEY)
    if pages and page in pages:
        st.switch_page(pages[page])
    else:
        st.rerun()


def status_badge(status: str) -> None:
    label, color, icon = STATUS_BADGES.get(status, (status, "gray", None))
    st.badge(label, color=color, icon=icon)


def page_header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


# --------------------------------------------------------------------------- research session


def stage_tracker(session: dict[str, Any]) -> None:
    views = stage_views(session)
    st.progress(progress_fraction(views, session["status"]))
    columns = st.columns(5)
    for i, view in enumerate(views):
        with columns[i % 5]:
            with st.container(border=True):
                st.markdown(f"{STATE_ICONS[view.state]} **{view.label}**")
                st.caption(view.detail or view.state.capitalize())


def stats_row(session: dict[str, Any]) -> None:
    stats = session.get("stats")
    if not stats:
        return
    a, b, c, d, e = st.columns(5)
    a.metric("Sources kept", stats["evidence"], help="Sources that passed verification")
    b.metric("Web / KB", f"{stats['evidence_web']} / {stats['evidence_knowledge_base']}")
    c.metric("Web searches", stats["web_searches"])
    d.metric("Gemini calls", stats["llm_calls"], help="Successful LLM steps recorded in the workflow")
    e.metric("Revisions", stats["revisions"])


def activity_log(session: dict[str, Any]) -> None:
    events = session.get("events", [])
    with st.expander(f"Agent activity log ({len(events)} events)", icon=":material/list_alt:"):
        if not events:
            st.caption("No agent has finished a step yet.")
        for event in events:
            time = str(event["created_at"])[11:19]
            text = f"`{time}` **{event['node']}** - {event.get('message') or ''}"
            if event["status"] == "failed":
                st.warning(text, icon=":material/warning:")
            else:
                st.markdown(text)


def approval_panel(session: dict[str, Any]) -> None:
    request = session.get("approval_request") or {}
    sources = request.get("sources", {})
    with st.container(border=True):
        st.subheader(":material/front_hand: Research complete - your approval is needed")
        st.write(
            "Review what the agents found. Generating the final report runs the writer, "
            "critic and (if needed) revision agents."
        )
        a, b, c = st.columns(3)
        a.metric("Sources", sources.get("total", 0))
        b.metric("Web", sources.get("web", 0))
        c.metric("Knowledge base", sources.get("knowledge_base", 0))

        st.markdown("**Research questions**")
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(request.get("research_questions", []), 1)))
        if request.get("answer_summary"):
            st.markdown(f"**Preliminary answer:** {request['answer_summary']}")
        findings = request.get("key_findings", [])
        st.markdown("**Key findings**")
        st.markdown("\n".join(f"- {f}" for f in findings) if findings else "_The analysis step was unavailable; "
                    "the report will be written directly from the verified sources._")
        if request.get("conflicts"):
            st.markdown("**Conflicts between sources**")
            st.markdown("\n".join(f"- {c}" for c in request["conflicts"]))
        for message in request.get("errors", []):
            st.warning(message, icon=":material/warning:")

        session_id = session["id"]
        approve_col, cancel_col = st.columns([3, 1])
        if approve_col.button("Generate Final Report", type="primary", icon=":material/description:",
                              width="stretch", key="approve"):
            if call(lambda: client().decide(session_id, "approve"), failure="Could not approve"):
                st.rerun()
        if cancel_col.button("Cancel research", icon=":material/close:", width="stretch", key="cancel"):
            if call(lambda: client().decide(session_id, "cancel"), failure="Could not cancel"):
                st.rerun()

        can_modify = request.get("can_modify", False)
        with st.expander("Modify Research", icon=":material/edit_note:", expanded=False):
            if not can_modify:
                st.info("The modification limit for this run has been reached.")
            feedback = st.text_area("What should the agents change or investigate further?",
                                    key="modify_feedback", disabled=not can_modify,
                                    placeholder="e.g. Focus on enterprise deployments and cost comparisons")
            if st.button("Run another research iteration", disabled=not can_modify, key="modify"):
                if not feedback.strip():
                    st.warning("Please describe what should change.")
                elif call(lambda: client().decide(session_id, "modify", feedback.strip()),
                          failure="Could not modify research"):
                    st.rerun()


def report_view(session_id: str) -> None:
    report_key, pdf_key = f"report:{session_id}", f"pdf:{session_id}"
    if report_key not in st.session_state:
        report = call(lambda: client().get_report(session_id), failure="Could not load the report")
        if report is None:
            return
        st.session_state[report_key] = report
    report = st.session_state[report_key]

    with st.container(border=True):
        title_col, download_col = st.columns([3, 1])
        title_col.subheader(f":material/article: {report['title']}")
        if pdf_key not in st.session_state:
            st.session_state[pdf_key] = call(lambda: client().get_report_pdf(session_id),
                                             failure="Could not generate the PDF")
        if st.session_state[pdf_key]:
            download_col.download_button(
                "Download PDF", data=st.session_state[pdf_key], file_name=f"{report['title'][:80]}.pdf",
                mime="application/pdf", icon=":material/download:", type="primary", width="stretch",
            )

        report_tab, sources_tab = st.tabs(["Report", f"Sources ({len(report['citations'])})"])
        with report_tab:
            st.markdown(report["markdown"])
        with sources_tab:
            if not report["citations"]:
                st.info("The report does not cite any sources.")
            else:
                st.dataframe(
                    [
                        {
                            "ID": c["id"],
                            "Title": c["title"],
                            "Type": "Web" if c["source_type"] == "web" else "Knowledge base",
                            "Verification": c["verification_status"].replace("_", " "),
                            "Reference": c["reference"],
                        }
                        for c in report["citations"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.caption("Verification describes how well other collected sources support a source's "
                           "claims - not whether the claims are true.")

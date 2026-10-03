"""Reusable UI pieces: client access, error display, headers, status badges, stage tracker,
approval panel and report view."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

import streamlit as st

from frontend.api_client import ApiError, BackendUnavailable, ResearchPilotClient
from frontend.workflow import STAGES, STATE_ICONS, STATUS_BADGES, progress_fraction, stage_views

T = TypeVar("T")

CLIENT_KEY = "api_client"
ACTIVE_SESSION_KEY = "active_session_id"
PAGES_KEY = "_pages"
STAGE_LABELS = dict(STAGES)


def client() -> ResearchPilotClient:
    if CLIENT_KEY not in st.session_state:
        st.session_state[CLIENT_KEY] = ResearchPilotClient()
    return st.session_state[CLIENT_KEY]


def call(action: Callable[[], T], *, failure: str = "Something went wrong") -> T | None:
    """Run an API call; show a clear message instead of a stack trace on failure."""
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


def local_time(value: str, fmt: str = "%d %b %Y, %H:%M") -> str:
    """Backend timestamps are UTC (naive ones included); show them in the viewer's local time."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone().strftime(fmt)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def status_badge(status: str) -> None:
    label, color, icon = STATUS_BADGES.get(status, (status, "gray", None))
    st.badge(label, color=color, icon=icon)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="rp-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<p class="rp-section">{title}</p>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


# --------------------------------------------------------------------------- research session


def stage_tracker(session: dict[str, Any]) -> None:
    views = stage_views(session)
    st.progress(progress_fraction(views, session["status"]))
    for row in (views[:5], views[5:]):
        columns = st.columns(5, gap="small")
        for column, view in zip(columns, row, strict=False):
            with column, st.container(border=True, height="stretch"):
                st.markdown(f"{STATE_ICONS[view.state]} **{view.label}**")
                st.caption(STATE_TEXT[view.state] if view.node != "revision" or view.state != "skipped"
                           else "Not needed")


# Short, user-facing stage states; detailed agent messages live in the activity log.
STATE_TEXT = {
    "done": "Done",
    "active": "In progress",
    "waiting": "Waiting for you",
    "warning": "Done, with limitations",
    "failed": "Stopped here",
    "skipped": "Skipped",
    "pending": "Up next",
}


def stats_row(session: dict[str, Any]) -> None:
    stats = session.get("stats")
    if not stats:
        return
    a, b, c, d = st.columns(4)
    a.metric("Sources kept", stats["evidence"], help="Sources that passed verification")
    b.metric("From the web", stats["evidence_web"])
    c.metric("From your documents", stats["evidence_knowledge_base"])
    d.metric("Web searches", stats["web_searches"])


def activity_log(session: dict[str, Any]) -> None:
    events = session.get("events", [])
    with st.expander(f"Activity log · {len(events)} updates", icon=":material/list_alt:"):
        if not events:
            st.caption("No step has finished yet.")
        for event in events:
            time = local_time(event["created_at"], "%H:%M")
            stage = STAGE_LABELS.get(event["node"], event["node"].replace("_", " ").capitalize())
            text = f"**{stage}** · {event.get('message') or ''}"
            if event["status"] == "failed":
                st.warning(f"{text}  \n`{time}`", icon=":material/warning:")
            else:
                st.markdown(f"{text} <span class='rp-muted'>{time}</span>", unsafe_allow_html=True)


def approval_panel(session: dict[str, Any]) -> None:
    request = session.get("approval_request") or {}
    sources = request.get("sources", {})
    with st.container(border=True):
        st.subheader(":material/front_hand: Review the research before the report is written")
        st.caption(f"The agents verified {sources.get('total', 0)} sources. Generate the report if the findings "
                   "look right, or ask them to dig further.")

        left, right = st.columns(2, gap="large")
        with left:
            section("Research questions")
            st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(request.get("research_questions", []), 1)))
        with right:
            section("Key findings")
            findings = request.get("key_findings", [])
            st.markdown("\n".join(f"- {f}" for f in findings) if findings else
                        "_A summary isn't available for this run; the report will be written directly from "
                        "the verified sources._")
        if request.get("answer_summary"):
            st.info(request["answer_summary"], icon=":material/lightbulb:")
        if request.get("conflicts"):
            section("Where sources disagree")
            st.markdown("\n".join(f"- {c}" for c in request["conflicts"]))
        for message in request.get("errors", []):
            st.warning(message, icon=":material/warning:")

        session_id = session["id"]
        approve_col, cancel_col = st.columns([3, 1])
        if approve_col.button("Generate Final Report", type="primary", icon=":material/description:",
                              width="stretch", key="approve"):
            if call(lambda: client().decide(session_id, "approve"), failure="Could not continue"):
                st.rerun()
        if cancel_col.button("Cancel research", icon=":material/close:", width="stretch", key="cancel"):
            if call(lambda: client().decide(session_id, "cancel"), failure="Could not cancel"):
                st.rerun()

        can_modify = request.get("can_modify", False)
        with st.expander("Modify Research", icon=":material/edit_note:"):
            if not can_modify:
                st.info("This run has already been refined the maximum number of times.")
            feedback = st.text_area("What should the agents change or investigate further?",
                                    key="modify_feedback", disabled=not can_modify,
                                    placeholder="e.g. Focus on enterprise deployments and cost comparisons")
            if st.button("Run another research iteration", disabled=not can_modify, key="modify"):
                if not feedback.strip():
                    st.warning("Please describe what should change.")
                elif call(lambda: client().decide(session_id, "modify", feedback.strip()),
                          failure="Could not update the research"):
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
        title_col, download_col = st.columns([3, 1], vertical_alignment="center")
        title_col.subheader(f":material/article: {report['title']}")
        if pdf_key not in st.session_state:
            st.session_state[pdf_key] = call(lambda: client().get_report_pdf(session_id),
                                             failure="Could not prepare the PDF")
        if st.session_state[pdf_key]:
            download_col.download_button(
                "Download PDF", data=st.session_state[pdf_key], file_name=f"{report['title'][:80]}.pdf",
                mime="application/pdf", icon=":material/download:", type="primary", width="stretch",
            )

        report_tab, sources_tab = st.tabs(["Report", f"Sources ({len(report['citations'])})"])
        with report_tab:
            st.markdown(_without_title(report["markdown"]))  # title is already shown above
        with sources_tab:
            if not report["citations"]:
                st.info("The report does not cite any sources.")
            else:
                st.dataframe(
                    [
                        {
                            "ID": c["id"],
                            "Source": c["title"],
                            "Type": "Web" if c["source_type"] == "web" else "Your documents",
                            "Verification": c["verification_status"].replace("_", " ").capitalize(),
                            "Link / location": c["reference"],
                        }
                        for c in report["citations"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.caption("Verification shows how well other collected sources support each source's "
                           "claims — not whether the claims are true.")


def _without_title(markdown: str) -> str:
    lines = markdown.lstrip().splitlines()
    return "\n".join(lines[1:]).lstrip() if lines and lines[0].startswith("# ") else markdown

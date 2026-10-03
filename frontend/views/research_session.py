"""A research session: live progress, human approval, completed report, stopped/cancelled states.

Everything shown is read from the backend (events, pending steps, status); while the
agents run, only the session fragment re-renders every few seconds.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components import call, client, go_to, local_time, open_session, plural
from frontend.ui.components import (
    badge_html,
    esc,
    html_block,
    metric_card,
    metric_row,
    section_header,
    source_item_html,
    workflow_timeline,
)
from frontend.workflow import STAGES, is_active, stage_views

POLL_SECONDS = 2
STAGE_LABELS = dict(STAGES)
STAGE_DESCRIPTIONS = {
    "planner": "Breaking your question into focused research questions and searches.",
    "web_research": "Searching the web and reading the most relevant pages.",
    "rag_research": "Finding relevant passages in your knowledge base.",
    "source_verification": "Cross-checking sources and flagging weak or conflicting claims.",
    "analysis": "Identifying key findings, patterns and gaps across the evidence.",
    "writer": "Writing the structured report with inline citations.",
    "critic": "Fact-checking the draft against the collected evidence.",
    "revision": "Fixing the issues found by the fact checker.",
    "finalize": "Building the reference list and finalising the report.",
}
STOPPED_MESSAGES = {
    "quota_exhausted": "The AI service's usage limit was reached, so the agents paused their work.",
    "interrupted": "The research stopped before it could finish - for example, the AI service was "
                   "temporarily busy or the app was restarted.",
    "failed": "One of the agents couldn't complete its step.",
}
VIEW_OPTIONS = ["Overview", "Full report", "Sources"]


def render(session_id: str) -> None:
    session = call(lambda: client().get_research(session_id), failure="Could not load this research")
    if session is None:
        if st.button("Start a new research", icon=":material/add:"):
            open_session(None)
            st.rerun()
        return
    running = is_active(session["status"])
    # While agents run, only this fragment re-renders (real backend polling).
    st.fragment(_live, run_every=POLL_SECONDS if running else None)(session_id, session["status"])


def _live(session_id: str, rendered_status: str) -> None:
    session = call(lambda: client().get_research(session_id), failure="Could not refresh progress")
    if session is None:
        return
    status = session["status"]
    if status != rendered_status:
        st.rerun(scope="app")  # status changed: re-render the page (stops/starts polling)

    if status == "completed":
        _completed(session)
        return

    titles = {
        "running": "Research in progress", "pending": "Research in progress",
        "awaiting_approval": "Research ready for review", "cancelled": "Research cancelled",
    }
    _header(session, titles.get(status, "Research paused"))
    main, side = st.columns([1.85, 1], gap="large")
    with side, st.container(key="card_timeline"):
        section_header("Agent workflow", "Live status of each step")
        st.write("")
        workflow_timeline(stage_views(session))
    with main:
        if is_active(status):
            _working(session)
        elif status == "awaiting_approval":
            _approval(session)
        elif status == "cancelled":
            _cancelled()
        else:
            _stopped(session)
        _activity_log(session)


# --------------------------------------------------------------------------- shared pieces


def _header(session: dict[str, Any], eyebrow: str) -> None:
    left, right = st.columns([4, 1.2], vertical_alignment="center")
    with left:
        html_block(f'<div class="rp-eyebrow">{esc(eyebrow)}</div>'
                   f'<div class="rp-session-title">{esc(session.get("title") or session["query"])}</div>')
        meta = [badge_html(session["status"]), f'<span class="rp-small">Started {esc(local_time(session["created_at"]))}'
                                               "</span>"]
        if session.get("instructions"):
            meta.append(f'<span class="rp-small">· {esc(session["instructions"])}</span>')
        html_block('<div style="display:flex;gap:.7rem;align-items:center;flex-wrap:wrap;margin-top:.45rem">'
                   + "".join(meta) + "</div>")
    with right:
        if st.button("New Research", icon=":material/add:", width="stretch", key="new_research"):
            open_session(None)
            st.rerun(scope="app")
    st.write("")


def _stats(session: dict[str, Any]) -> None:
    stats = session.get("stats")
    if stats:
        metric_row([
            ("Sources", stats["evidence"], "library_books"),
            ("Web", stats["evidence_web"], "public"),
            ("Documents", stats["evidence_knowledge_base"], "folder"),
            ("Searches", stats["web_searches"], "search"),
        ])


def _activity_log(session: dict[str, Any]) -> None:
    events = session.get("events", [])
    st.write("")
    with st.expander(f"Activity log · {plural(len(events), 'update')}", icon=":material/list_alt:"):
        if not events:
            st.caption("No step has finished yet.")
        for event in events:
            stage = STAGE_LABELS.get(event["node"], event["node"].replace("_", " ").capitalize())
            line = f"**{stage}** · {event.get('message') or ''}"
            when = local_time(event["created_at"], "%H:%M")
            if event["status"] == "failed":
                st.warning(f"{line}  \n`{when}`", icon=":material/warning:")
            else:
                st.markdown(f"{line} <span class='rp-small'>{when}</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- running


def _working(session: dict[str, Any]) -> None:
    views = stage_views(session)
    active = next((v for v in views if v.state == "active"), None)
    done = sum(v.state in ("done", "warning", "skipped") for v in views)
    stage = active.label if active else "Preparing"
    description = STAGE_DESCRIPTIONS.get(active.node, "") if active else "Starting the agents."
    with st.container(key="accentcard_working"):
        html_block(f'<div class="rp-eyebrow"><span class="rp-spinner"></span>&nbsp; Agent working...</div>'
                   f'<div class="rp-page-title" style="font-size:1.35rem">{esc(stage)}</div>'
                   f'<div class="rp-page-subtitle">{esc(description)}</div>')
        st.progress(done / len(views), text=f"{done} of {len(views)} steps complete")
        st.caption("This page updates automatically. You'll be asked to review the findings before "
                   "the final report is written.")
    st.write("")
    _stats(session)


# --------------------------------------------------------------------------- approval


def _approval(session: dict[str, Any]) -> None:
    request = session.get("approval_request") or {}
    sources = request.get("sources", {})
    session_id = session["id"]
    with st.container(key="accentcard_approval"):
        html_block('<div class="rp-eyebrow">Decision point</div>'
                   '<div class="rp-page-title" style="font-size:1.45rem">Your Review Is Needed</div>'
                   '<div class="rp-page-subtitle">ResearchPilot has completed the research and is ready to '
                   "generate the final report.</div>")
        st.write("")
        metric_row([
            ("Sources found", sources.get("total", 0), "library_books"),
            ("Web sources", sources.get("web", 0), "public"),
            ("From documents", sources.get("knowledge_base", 0), "folder"),
        ])
        st.write("")
        questions, findings = st.columns(2, gap="large")
        with questions:
            section_header("Research questions")
            st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(request.get("research_questions", []), 1)))
        with findings:
            section_header("Key findings")
            items = request.get("key_findings", [])
            st.markdown("\n".join(f"- {f}" for f in items) if items else
                        "_A summary isn't available for this run; the report will be written directly from "
                        "the verified sources._")
        if request.get("answer_summary"):
            st.info(request["answer_summary"], icon=":material/lightbulb:")
        if request.get("conflicts"):
            section_header("Where sources disagree")
            st.markdown("\n".join(f"- {c}" for c in request["conflicts"]))
        if request.get("errors"):
            st.caption(":material/info: Some steps completed with limitations.")
            with st.expander("Technical details"):
                st.code("\n".join(request["errors"]), language=None, wrap_lines=True)

        st.write("")
        approve_col, modify_col, cancel_col = st.columns([2.2, 1.6, 1.1])
        if approve_col.button("Approve & Generate Report", type="primary", icon=":material/check:",
                              width="stretch", key="approve"):
            if call(lambda: client().decide(session_id, "approve"), failure="Could not continue"):
                st.rerun()
        if modify_col.button("Modify Research", icon=":material/edit:", width="stretch", key="modify_toggle"):
            st.session_state["modify_open"] = not st.session_state.get("modify_open", False)
        with cancel_col, st.container(key="danger_cancel"):
            if st.button("Cancel", icon=":material/close:", width="stretch", key="cancel"):
                if call(lambda: client().decide(session_id, "cancel"), failure="Could not cancel"):
                    st.rerun()
        if st.session_state.get("modify_open"):
            _modify_panel(session_id, request.get("can_modify", False))


def _modify_panel(session_id: str, can_modify: bool) -> None:
    with st.container(key="card_modify"):
        section_header("Modify research", "Tell the agents what to change or investigate further.")
        if not can_modify:
            st.info("This research has already been refined the maximum number of times.")
        feedback = st.text_area("What should change?", key="modify_feedback", disabled=not can_modify,
                                label_visibility="collapsed",
                                placeholder="e.g. Focus on enterprise deployments and cost comparisons")
        if st.button("Run another research iteration", type="primary", disabled=not can_modify, key="modify"):
            if not feedback.strip():
                st.warning("Please describe what should change.")
            elif call(lambda: client().decide(session_id, "modify", feedback.strip()),
                      failure="Could not update the research"):
                st.session_state["modify_open"] = False
                st.rerun()


# --------------------------------------------------------------------------- stopped / cancelled


def _stopped(session: dict[str, Any]) -> None:
    retryable = session.get("retryable", False)
    message = STOPPED_MESSAGES.get(session["status"], "The research stopped.")
    if session["status"] == "failed" and not retryable:
        message = "The research couldn't be completed. Try rephrasing the question or adding documents."
    stopped_at = ", ".join(STAGE_LABELS.get(n, n) for n in session.get("pending_nodes") or [])
    with st.container(key="card_stopped"):
        eyebrow = f"Stopped at {stopped_at}" if stopped_at else "Stopped"
        html_block(f'<div class="rp-eyebrow" style="color:#b42318">{esc(eyebrow)}</div>'
                   '<div class="rp-page-title" style="font-size:1.35rem">What happened?</div>'
                   f'<div class="rp-page-subtitle">{esc(message)}</div>')
        st.write("")
        if retryable:
            st.success("Your research is safely saved. You can retry from the last checkpoint.",
                       icon=":material/verified_user:")
            if st.button("Retry from checkpoint", type="primary", icon=":material/replay:", key="retry"):
                if call(lambda: client().retry(session["id"]), failure="Could not retry"):
                    st.rerun(scope="app")
        if session.get("error"):
            with st.expander("Technical details"):
                st.code(session["error"], language=None, wrap_lines=True)
    st.write("")
    _stats(session)


def _cancelled() -> None:
    with st.container(key="card_cancelled"):
        section_header("This research was cancelled", "No report was generated. You can start a new research "
                                                       "at any time.")


# --------------------------------------------------------------------------- completed


def _completed(session: dict[str, Any]) -> None:
    session_id = session["id"]
    report_key, pdf_key, view_key = f"report:{session_id}", f"pdf:{session_id}", f"view:{session_id}"
    if report_key not in st.session_state:
        report = call(lambda: client().get_report(session_id), failure="Could not load the report")
        if report is None:
            return
        st.session_state[report_key] = report
    report = st.session_state[report_key]
    if pdf_key not in st.session_state:
        st.session_state[pdf_key] = call(lambda: client().get_report_pdf(session_id),
                                         failure="Could not prepare the PDF")

    sources = len(report["citations"])
    html_block('<div class="rp-eyebrow" style="color:#067647">Research Complete ✓</div>'
               f'<div class="rp-session-title">{esc(report["title"])}</div>'
               '<div style="display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;margin:.5rem 0 1rem">'
               f'{badge_html("completed")}<span class="rp-small">{esc(local_time(report["created_at"]))}</span>'
               f'<span class="rp-small">· {plural(sources, "cited source")}</span>'
               f'<span class="rp-small">· {esc(session["query"])}</span></div>')

    pdf_col, view_col, history_col, new_col = st.columns([1.3, 1.1, 1.2, 1.1])
    if st.session_state[pdf_key]:
        pdf_col.download_button("Download PDF", data=st.session_state[pdf_key], file_name=f"{report['title'][:80]}.pdf",
                                mime="application/pdf", icon=":material/download:", type="primary",
                                width="stretch", key="download_pdf")
    if view_col.button("View Report", icon=":material/article:", width="stretch", key="view_report"):
        st.session_state[view_key] = "Full report"
    if history_col.button("Back to History", icon=":material/history:", width="stretch", key="back_history"):
        go_to("history")
    if new_col.button("New Research", icon=":material/add:", width="stretch", key="new_research"):
        open_session(None)
        st.rerun()

    st.write("")
    st.session_state.setdefault(view_key, "Overview")  # value managed via state ("View Report" sets it)
    view = st.segmented_control("Report view", VIEW_OPTIONS, key=view_key, label_visibility="collapsed") or "Overview"
    if view == "Overview":
        _overview(session, report)
    elif view == "Full report":
        with st.container(key="card_report"):
            st.markdown(_without_title(report["markdown"]))
    else:
        with st.container(key="card_sources"):
            section_header("Sources", "Verification shows how well other collected sources support each source - "
                                      "not whether its claims are true.")
            st.write("")
            if report["citations"]:
                html_block("".join(source_item_html(c) for c in report["citations"]))
            else:
                st.caption("The report does not cite any sources.")


def _overview(session: dict[str, Any], report: dict[str, Any]) -> None:
    body = report.get("report") or {}
    main, side = st.columns([1.85, 1], gap="large")
    with main:
        with st.container(key="card_summary"):
            section_header("Executive Summary")
            st.markdown(body.get("executive_summary", ""))
        st.write("")
        with st.container(key="card_findings"):
            section_header("Key Findings")
            st.markdown("\n".join(f"- {f}" for f in body.get("key_findings", [])) or "_No key findings._")
        if report.get("quality_notes"):
            st.write("")
            with st.expander("Quality notes", icon=":material/rule:"):
                for note in report["quality_notes"]:
                    st.caption(note)
    with side:
        stats = session.get("stats") or {}
        metric_card("Cited sources", len(report["citations"]), "library_books")
        st.write("")
        metric_card("Sources verified", stats.get("evidence", 0), "verified")
        st.write("")
        with st.container(key="card_timeline_done"):
            section_header("Agent workflow")
            st.write("")
            workflow_timeline(stage_views(session))


def _without_title(markdown: str) -> str:
    lines = markdown.lstrip().splitlines()
    return "\n".join(lines[1:]).lstrip() if lines and lines[0].startswith("# ") else markdown

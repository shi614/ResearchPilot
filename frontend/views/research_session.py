"""A research session: live progress, human approval, research report, stopped/cancelled states.

Everything shown is read from the backend (events, pending steps, status); while the
agents run, only the session fragment re-renders every few seconds.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components import (
    PREF_SHOW_ACTIVITY,
    call,
    client,
    go_to,
    local_time,
    open_session,
    plural,
)
from frontend.ui.components import (
    badge_html,
    esc,
    html_block,
    metric_row,
    primary_button,
    secondary_button,
    section_header,
    source_item_html,
    timeline,
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
    "critic": "Reviewing the draft against the collected evidence.",
    "revision": "Fixing the issues found during quality review.",
    "finalize": "Building the reference list and finalising the report.",
}
STOPPED_MESSAGES = {
    "quota_exhausted": "The AI service's usage limit was reached, so the agents paused their work.",
    "interrupted": "The research stopped before it could finish - for example, the AI service was "
                   "temporarily busy or the app was restarted.",
    "failed": "One of the agents couldn't complete its step.",
}


def render(session_id: str) -> None:
    session = call(lambda: client().get_research(session_id), failure="Could not load this research")
    if session is None:
        if secondary_button("Start a new research", key="fallback_new", icon_name="add", stretch=False):
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
        _report(session)
        return

    headings = {
        "running": "Research in progress", "pending": "Research in progress",
        "awaiting_approval": "Review Before Report Generation", "cancelled": "Research cancelled",
    }
    _header(session, headings.get(status, "Research paused"))
    main, side = st.columns([1.85, 1], gap="large")
    with side, st.container(key="card_timeline"):
        section_header("Agent workflow", "Live status from the research agents")
        st.write("")
        timeline(stage_views(session))
    with main:
        if is_active(status):
            _working(session)
        elif status == "awaiting_approval":
            _approval(session)
        elif status == "cancelled":
            _cancelled()
        else:
            _stopped(session)
        if st.session_state.get(PREF_SHOW_ACTIVITY, True):
            _activity_log(session)


# --------------------------------------------------------------------------- shared pieces


def _header(session: dict[str, Any], heading: str) -> None:
    left, right = st.columns([4, 1.2], vertical_alignment="center")
    with left:
        html_block(f'<div class="rp-page-title">{esc(heading)}</div>'
                   f'<div class="rp-page-subtitle">{esc(session.get("title") or session["query"])}</div>')
        meta = [badge_html(session["status"]),
                f'<span class="rp-meta">Started <b>{esc(local_time(session["created_at"]))}</b></span>']
        if session.get("instructions"):
            meta.append(f'<span class="rp-meta">Instructions <b>{esc(session["instructions"])}</b></span>')
        html_block('<div class="rp-meta-row" style="align-items:center">' + "".join(meta) + "</div>")
    with right:
        if secondary_button("New Research", key="new_research", icon_name="add"):
            open_session(None)
            st.rerun(scope="app")


def _stats(session: dict[str, Any]) -> None:
    stats = session.get("stats")
    if stats:
        metric_row([
            ("Sources", stats["evidence"], "library_books"),
            ("Web", stats["evidence_web"], "public"),
            ("Documents", stats["evidence_knowledge_base"], "menu_book"),
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
    with st.container(key="tealcard_working"):
        html_block(f'<div class="rp-eyebrow"><span class="rp-spinner"></span>&nbsp; Agent working...</div>'
                   f'<div class="rp-session-title">{esc(stage)}</div>'
                   f'<div class="rp-page-subtitle">{esc(description)}</div>')
        st.write("")
        st.progress(done / len(views), text=f"{done} of {len(views)} steps complete")
        st.caption("This page updates automatically. You'll review the findings before the report is written.")
    st.write("")
    _stats(session)


# --------------------------------------------------------------------------- approval


def _approval(session: dict[str, Any]) -> None:
    request = session.get("approval_request") or {}
    sources = request.get("sources", {})
    session_id = session["id"]
    with st.container(key="tealcard_approval"):
        html_block('<span class="rp-human"><span class="rp-ms">person</span> Human-in-the-loop checkpoint</span>'
                   '<div class="rp-session-title" style="margin-top:.6rem">Your decision is needed</div>'
                   '<div class="rp-page-subtitle">The agents have finished researching. Review what they found '
                   "before ResearchPilot writes the final report.</div>")
        st.write("")
        metric_row([
            ("Sources", sources.get("total", 0), "library_books"),
            ("Web sources", sources.get("web", 0), "public"),
            ("Knowledge base", sources.get("knowledge_base", 0), "menu_book"),
        ])
        st.write("")
        if request.get("answer_summary"):
            section_header("Research summary")
            st.markdown(request["answer_summary"])
        findings, questions = st.columns(2, gap="large")
        with findings:
            section_header("Important findings")
            items = request.get("key_findings", [])
            st.markdown("\n".join(f"- {f}" for f in items) if items else
                        "_A summary isn't available for this run; the report will be written directly from "
                        "the verified sources._")
        with questions:
            section_header("Research questions")
            st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(request.get("research_questions", []), 1)))
        if request.get("conflicts"):
            section_header("Analysis: where sources disagree")
            st.markdown("\n".join(f"- {c}" for c in request["conflicts"]))
        if request.get("errors"):
            st.caption(":material/info: Some steps completed with limitations.")
            with st.expander("Technical details"):
                st.code("\n".join(request["errors"]), language=None, wrap_lines=True)

        st.divider()
        approve_col, changes_col, cancel_col = st.columns([1.8, 1.8, 1], vertical_alignment="center")
        with approve_col:
            if primary_button("Approve & Continue", key="approve", icon_name="check"):
                if call(lambda: client().decide(session_id, "approve"), failure="Could not continue"):
                    st.rerun()
        with changes_col:
            if secondary_button("Request Changes", key="modify_toggle", icon_name="edit"):
                st.session_state["modify_open"] = not st.session_state.get("modify_open", False)
        with cancel_col, st.container(key="danger_cancel"):
            if st.button("Cancel", icon=":material/close:", width="stretch", key="cancel", type="tertiary"):
                if call(lambda: client().decide(session_id, "cancel"), failure="Could not cancel"):
                    st.rerun()
        if st.session_state.get("modify_open"):
            _changes_panel(session_id, request.get("can_modify", False))


def _changes_panel(session_id: str, can_modify: bool) -> None:
    with st.container(key="card_modify"):
        section_header("Request changes", "Tell the agents what to change or investigate further.")
        if not can_modify:
            st.info("This research has already been refined the maximum number of times.")
        feedback = st.text_area("What should change?", key="modify_feedback", disabled=not can_modify,
                                label_visibility="collapsed",
                                placeholder="e.g. Focus on enterprise deployments and cost comparisons")
        if primary_button("Run another research iteration", key="modify", disabled=not can_modify, stretch=False):
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
        html_block(f'<div class="rp-eyebrow" style="color:#C75C5C">{esc(eyebrow)}</div>'
                   '<div class="rp-session-title">What happened?</div>'
                   f'<div class="rp-page-subtitle">{esc(message)}</div>')
        st.write("")
        if retryable:
            st.success("Your research is safely saved. You can retry from the last checkpoint.",
                       icon=":material/verified_user:")
            if primary_button("Retry from checkpoint", key="retry", icon_name="replay", stretch=False):
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


# --------------------------------------------------------------------------- research report


def _report(session: dict[str, Any]) -> None:
    session_id = session["id"]
    report_key, pdf_key, sources_key = f"report:{session_id}", f"pdf:{session_id}", f"view:{session_id}:sources"
    if report_key not in st.session_state:
        report = call(lambda: client().get_report(session_id), failure="Could not load the report")
        if report is None:
            return
        st.session_state[report_key] = report
    report = st.session_state[report_key]
    if pdf_key not in st.session_state:
        st.session_state[pdf_key] = call(lambda: client().get_report_pdf(session_id),
                                         failure="Could not prepare the PDF")

    head, back = st.columns([4, 1.2], vertical_alignment="center")
    with head:
        html_block('<div class="rp-eyebrow" style="color:#4F8A70">Research Complete ✓</div>'
                   '<div class="rp-page-title">Research Report</div>')
    with back:
        if st.button("Back to History", icon=":material/arrow_back:", type="tertiary", key="back_history",
                     width="stretch"):
            go_to("history")
    html_block('<div class="rp-meta-row">'
               f'<span class="rp-meta">Topic <b>{esc(session["query"])}</b></span>'
               f'<span class="rp-meta">Date <b>{esc(local_time(report["created_at"]))}</b></span>'
               f'<span class="rp-meta">Sources <b>{len(report["citations"])}</b></span>'
               f'<span class="rp-meta">Status {badge_html("completed")}</span></div>')

    pdf_col, sources_col, new_col, _ = st.columns([1.3, 1.2, 1.5, 1.2])
    if st.session_state[pdf_key]:
        pdf_col.download_button("Download PDF", data=st.session_state[pdf_key], file_name=f"{report['title'][:80]}.pdf",
                                mime="application/pdf", icon=":material/download:", type="primary",
                                width="stretch", key="download_pdf")
    with sources_col:
        if secondary_button("View Sources", key="view_sources", icon_name="library_books"):
            st.session_state[sources_key] = not st.session_state.get(sources_key, False)
    with new_col:
        if secondary_button("Start New Research", key="new_research", icon_name="add"):
            open_session(None)
            st.rerun()

    st.write("")
    with st.expander(f"Sources ({len(report['citations'])})", icon=":material/library_books:",
                     expanded=st.session_state.get(sources_key, False)):
        st.caption("Verification shows how well other collected sources support each source - not whether "
                   "its claims are true.")
        if report["citations"]:
            html_block("".join(source_item_html(c) for c in report["citations"]))
        else:
            st.caption("The report does not cite any sources.")

    document, side = st.columns([2.6, 1], gap="large")
    with document, st.container(key="doccard_report"):
        html_block(f'<div class="rp-eyebrow">Research report</div><div class="rp-page-title" '
                   f'style="font-size:1.6rem;margin-bottom:.4rem">{esc(report["title"])}</div>')
        st.markdown(_without_title(report["markdown"]))
    with side:
        with st.container(key="card_report_meta"):
            section_header("At a glance")
            st.write("")
            stats = session.get("stats") or {}
            html_block(f'<div class="rp-meta">Cited sources <b>{len(report["citations"])}</b></div>'
                       f'<div class="rp-meta">Sources verified <b>{stats.get("evidence", 0)}</b></div>'
                       f'<div class="rp-meta">Revisions <b>{stats.get("revisions", 0)}</b></div>')
        st.write("")
        with st.expander("Agent workflow", icon=":material/account_tree:"):
            timeline(stage_views(session))
        if report.get("quality_notes"):
            with st.expander("Quality notes", icon=":material/rule:"):
                for note in report["quality_notes"]:
                    st.caption(note)


def _without_title(markdown: str) -> str:
    lines = markdown.lstrip().splitlines()
    return "\n".join(lines[1:]).lstrip() if lines and lines[0].startswith("# ") else markdown

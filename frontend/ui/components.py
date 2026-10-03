"""Reusable visual components (PageHeader, FeatureCard, ResearchCard, StatusBadge, WorkflowStep,
Timeline, PrimaryButton, SecondaryButton, SectionHeader, EmptyState, DocumentCard).

Static visuals are small HTML fragments built from the design-system classes in
`theme.py`; every piece of user or backend text is HTML-escaped. Interactive
controls stay native Streamlit widgets (styled by type and keyed containers).
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from typing import Any

import streamlit as st

from frontend.workflow import StageView

# status -> (label, tone)
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "pending": ("In Progress", "info"),
    "running": ("In Progress", "info"),
    "awaiting_approval": ("Awaiting Review", "warning"),
    "completed": ("Completed", "success"),
    "failed": ("Failed", "danger"),
    "interrupted": ("Interrupted", "warning"),
    "quota_exhausted": ("Paused", "warning"),
    "cancelled": ("Cancelled", "neutral"),
}

# Timeline notes describe the *real* state of each step (from backend events/checkpoints).
_DONE_NOTES = {
    "planner": "Research plan created",
    "web_research": "Sources collected",
    "rag_research": "Relevant documents retrieved",
    "source_verification": "Sources verified",
    "analysis": "Analysis complete",
    "human_review": "Approved",
    "writer": "Draft report written",
    "critic": "Quality reviewed",
    "revision": "Report revised",
    "finalize": "Report finalized",
}
_ACTIVE_NOTES = {
    "planner": "Currently planning the research",
    "web_research": "Currently searching the web",
    "rag_research": "Currently searching your documents",
    "source_verification": "Currently verifying sources",
    "analysis": "Currently analysing the evidence",
    "writer": "Currently writing the report",
    "critic": "Currently reviewing quality",
    "revision": "Currently revising the report",
    "finalize": "Currently finalizing the report",
}
_DOT = {"done": "check", "warning": "priority_high", "waiting": "person", "failed": "close", "skipped": "remove"}


def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def icon(name: str) -> str:
    """Material Symbols ligature (the font ships with Streamlit)."""
    return f'<span class="rp-ms" aria-hidden="true">{esc(name)}</span>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------- headers


def page_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    """PageHeader."""
    parts = [f'<div class="rp-eyebrow">{esc(eyebrow)}</div>'] if eyebrow else []
    parts.append(f'<div class="rp-page-title">{esc(title)}</div>')
    if subtitle:
        parts.append(f'<div class="rp-page-subtitle">{esc(subtitle)}</div>')
    html_block("".join(parts))


def section_header(title: str, subtitle: str | None = None) -> None:
    """SectionHeader."""
    sub = f'<div class="rp-section-sub">{esc(subtitle)}</div>' if subtitle else ""
    html_block(f'<div class="rp-section-title">{esc(title)}</div>{sub}')


# --------------------------------------------------------------------------- buttons


def primary_button(label: str, *, key: str, icon_name: str | None = None, disabled: bool = False,
                   stretch: bool = True) -> bool:
    """PrimaryButton: Deep Teal action."""
    return st.button(label, key=key, type="primary", icon=f":material/{icon_name}:" if icon_name else None,
                     disabled=disabled, width="stretch" if stretch else "content")


def secondary_button(label: str, *, key: str, icon_name: str | None = None, disabled: bool = False,
                     stretch: bool = True) -> bool:
    """SecondaryButton: white with a border."""
    return st.button(label, key=key, type="secondary", icon=f":material/{icon_name}:" if icon_name else None,
                     disabled=disabled, width="stretch" if stretch else "content")


# --------------------------------------------------------------------------- badges & metrics


def badge_html(status: str) -> str:
    label, tone = STATUS_STYLE.get(status, (status.replace("_", " ").capitalize(), "neutral"))
    return f'<span class="rp-badge rp-badge-{tone}">{esc(label)}</span>'


def status_badge(status: str) -> None:
    """StatusBadge."""
    html_block(badge_html(status))


def tone_badge(label: str, tone: str) -> str:
    return f'<span class="rp-badge rp-badge-{tone}">{esc(label)}</span>'


def metric_card(label: str, value: Any, icon_name: str | None = None) -> None:
    head = f"{icon(icon_name)} {esc(label)}" if icon_name else esc(label)
    html_block(f'<div class="rp-metric"><div class="rp-metric-label">{head}</div>'
               f'<div class="rp-metric-value">{esc(value)}</div></div>')


def metric_row(items: Iterable[tuple[str, Any, str | None]]) -> None:
    items = list(items)
    for column, (label, value, icon_name) in zip(st.columns(len(items)), items, strict=True):
        with column:
            metric_card(label, value, icon_name)


def feature_card(icon_name: str, title: str, text: str) -> None:
    """FeatureCard."""
    html_block(f'<div class="rp-feature"><span class="rp-feature-icon">{icon(icon_name)}</span>'
               f'<div class="rp-feature-title">{esc(title)}</div><p class="rp-feature-text">{esc(text)}</p></div>')


# --------------------------------------------------------------------------- workflow visuals


def workflow_step_html(number: int, label: str, agent: str, variant: str = "") -> str:
    """WorkflowStep: one connected step of the multi-agent flow."""
    agent_icon = "person" if variant == "human" else "smart_toy"
    return (f'<div class="rp-flow-step {variant}"><div class="rp-flow-num">{number:02d}</div>'
            f'<div class="rp-flow-label">{esc(label)}</div>'
            f'<span class="rp-flow-agent">{icon(agent_icon)} {esc(agent)}</span></div>')


def workflow_flow(steps: list[tuple[str, str, str]]) -> None:
    """Connected flow of (label, agent, variant) steps."""
    html_block('<div class="rp-flow">' + "".join(
        workflow_step_html(i, label, agent, variant) for i, (label, agent, variant) in enumerate(steps, 1)
    ) + "</div>")


def timeline(views: list[StageView]) -> None:
    """Timeline of real workflow state: completed (sage), active (teal), pending (slate)."""
    items = []
    for view in views:
        if view.state == "active":
            dot = '<span class="rp-spinner"></span>'
        else:
            dot = icon(_DOT[view.state]) if view.state in _DOT else ""
        note = {
            "done": _DONE_NOTES.get(view.node, "Completed"),
            "warning": "Completed with limitations",
            "active": _ACTIVE_NOTES.get(view.node, "In progress"),
            "waiting": "Waiting for your review",
            "failed": "Stopped here",
            "skipped": "Not needed" if view.node == "revision" else "Skipped",
        }.get(view.state, "")
        note_html = f'<div class="rp-tl-note">{esc(note)}</div>' if note else ""
        items.append(f'<li class="rp-tl-item {view.state}"><span class="rp-tl-dot">{dot}</span>'
                     f'<div class="rp-tl-body"><div class="rp-tl-label">{esc(view.label)}</div>{note_html}</div></li>')
    html_block('<ul class="rp-timeline">' + "".join(items) + "</ul>")


# Backwards-compatible name used by earlier views/tests.
workflow_timeline = timeline


# --------------------------------------------------------------------------- cards


def research_card_html(title: str, description: str, when: str, status: str,
                       sources: int | None, revisions: int | None) -> str:
    """ResearchCard body."""
    meta = [f"{icon('calendar_today')} {esc(when)}"]
    if sources is not None:
        meta.append(f"{icon('library_books')} {sources} source{'s' if sources != 1 else ''}")
    if revisions:
        meta.append(f"{icon('edit_note')} {revisions} revision{'s' if revisions != 1 else ''}")
    return (f"{badge_html(status)}<div class=\"rp-rcard-title\">{esc(title)}</div>"
            f'<p class="rp-rcard-desc">{esc(description)}</p>'
            f'<div class="rp-rcard-meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>')


def document_card_html(name: str, kind: str, meta: str) -> str:
    """DocumentCard body."""
    return (f'<div class="rp-doc-row"><span class="rp-doc-icon">{esc(kind)}</span>'
            f'<div style="min-width:0"><div class="rp-doc-name">{esc(name)}</div>'
            f'<div class="rp-doc-meta">{esc(meta)}</div></div></div>')


def source_item_html(citation: dict[str, Any]) -> str:
    reference = citation["reference"]
    ref = (f'<a href="{esc(reference)}" target="_blank" rel="noopener">{esc(reference)}</a>'
           if reference.startswith(("http://", "https://")) else esc(reference))
    kind = "Web" if citation["source_type"] == "web" else "Your documents"
    status = citation["verification_status"]
    tone = {"corroborated": "success", "single_source": "warning", "conflicting": "danger"}.get(status, "neutral")
    return (f'<div class="rp-source"><div class="rp-source-title">[{esc(citation["id"])}] {esc(citation["title"])}'
            f' {tone_badge(status.replace("_", " ").capitalize(), tone)}</div>'
            f'<div class="rp-source-ref">{esc(kind)} · {ref}</div></div>')


def empty_state(icon_name: str, title: str, text: str) -> None:
    """EmptyState."""
    html_block(f'<div class="rp-empty"><span class="rp-empty-icon">{icon(icon_name)}</span>'
               f"<h3>{esc(title)}</h3><p>{esc(text)}</p></div>")


SYSTEM_LABELS = {"ready": "System Operational", "limited": "Limited Mode", "offline": "Offline"}


def system_status_card(state: str) -> None:
    css = {"ready": "", "limited": "warn", "offline": "down"}[state]
    html_block(f'<div class="rp-sys {css}"><span class="rp-sys-dot"></span>{esc(SYSTEM_LABELS[state])}</div>')


def system_badge(state: str) -> str:
    tone = {"ready": "success", "limited": "warning", "offline": "danger"}[state]
    return tone_badge(SYSTEM_LABELS[state], tone)

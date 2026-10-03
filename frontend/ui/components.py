"""Reusable visual components.

Static visuals are small HTML fragments built from the design-system classes in
`theme.py`; every piece of user or backend text is HTML-escaped. Interactive
controls stay native Streamlit widgets (styled through keyed containers).
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
    "awaiting_approval": ("Waiting for Approval", "warning"),
    "completed": ("Completed", "success"),
    "failed": ("Failed", "danger"),
    "interrupted": ("Interrupted", "warning"),
    "quota_exhausted": ("Paused - usage limit", "warning"),
    "cancelled": ("Cancelled", "neutral"),
}

_TIMELINE_DOT = {
    "done": "check", "warning": "priority_high", "active": "", "waiting": "front_hand",
    "failed": "close", "skipped": "remove", "pending": "",
}
_TIMELINE_NOTE = {
    "done": "Completed", "warning": "Completed with limitations", "active": "Agent working...",
    "waiting": "Waiting for your review", "failed": "Stopped here", "skipped": "Skipped", "pending": "",
}


def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def icon(name: str) -> str:
    """Material Symbols ligature (the font ships with Streamlit)."""
    return f'<span class="rp-ms" aria-hidden="true">{esc(name)}</span>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------- text & headers


def page_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    parts = []
    if eyebrow:
        parts.append(f'<div class="rp-eyebrow">{esc(eyebrow)}</div>')
    parts.append(f'<div class="rp-page-title">{esc(title)}</div>')
    if subtitle:
        parts.append(f'<div class="rp-page-subtitle">{esc(subtitle)}</div>')
    html_block("".join(parts))


def section_header(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="rp-section-sub">{esc(subtitle)}</div>' if subtitle else ""
    html_block(f'<div class="rp-section-title">{esc(title)}</div>{sub}')


# --------------------------------------------------------------------------- badges & metrics


def badge_html(status: str) -> str:
    label, tone = STATUS_STYLE.get(status, (status.replace("_", " ").capitalize(), "neutral"))
    return f'<span class="rp-badge rp-badge-{tone}">{esc(label)}</span>'


def status_badge(status: str) -> None:
    html_block(badge_html(status))


def tone_badge(label: str, tone: str) -> str:
    return f'<span class="rp-badge rp-badge-{tone}">{esc(label)}</span>'


def metric_card(label: str, value: Any, icon_name: str | None = None, hint: str | None = None) -> None:
    head = f"{icon(icon_name)} {esc(label)}" if icon_name else esc(label)
    extra = f'<div class="rp-metric-hint">{esc(hint)}</div>' if hint else ""
    html_block(f'<div class="rp-metric"><div class="rp-metric-label">{head}</div>'
           f'<div class="rp-metric-value">{esc(value)}</div>{extra}</div>')


def metric_row(items: Iterable[tuple[str, Any, str | None]]) -> None:
    items = list(items)
    for column, (label, value, icon_name) in zip(st.columns(len(items)), items, strict=True):
        with column:
            metric_card(label, value, icon_name)


# --------------------------------------------------------------------------- workflow visuals


def how_it_works(steps: list[tuple[str, str, str]]) -> None:
    """Horizontal numbered steps with arrows: (icon, label, description)."""
    cards = []
    for number, (icon_name, label, text) in enumerate(steps, 1):
        cards.append(
            f'<div class="rp-step"><div class="rp-step-top"><span class="rp-step-icon">{icon(icon_name)}</span>'
            f'<span><div class="rp-step-num">{number:02d}</div><div class="rp-step-label">{esc(label)}</div>'
            f'</span></div><div class="rp-step-text">{esc(text)}</div></div>'
        )
    html_block('<div class="rp-steps">' + '<span class="rp-step-arrow">&rarr;</span>'.join(cards) + "</div>")


def workflow_timeline(views: list[StageView]) -> None:
    items = []
    for view in views:
        if view.state == "active":
            dot = '<span class="rp-spinner"></span>'
        elif _TIMELINE_DOT[view.state]:
            dot = icon(_TIMELINE_DOT[view.state])
        else:
            dot = ""
        note = _TIMELINE_NOTE[view.state]
        if view.state == "skipped" and view.node == "revision":
            note = "Not needed"
        note_html = f'<div class="rp-tl-note">{esc(note)}</div>' if note else ""
        items.append(f'<li class="rp-tl-item {view.state}"><span class="rp-tl-dot">{dot}</span>'
                     f'<div class="rp-tl-body"><div class="rp-tl-label">{esc(view.label)}</div>{note_html}</div></li>')
    html_block('<ul class="rp-timeline">' + "".join(items) + "</ul>")


# --------------------------------------------------------------------------- cards


def card(markup: str, variant: str = "") -> None:
    html_block(f'<div class="rp-card {variant}">{markup}</div>')


def research_card_html(title: str, description: str, when: str, status: str,
                       sources: int | None, revisions: int | None) -> str:
    meta = [f"{icon('schedule')} {esc(when)}"]
    if sources is not None:
        meta.append(f"{icon('library_books')} {sources} source{'s' if sources != 1 else ''}")
    if revisions:
        meta.append(f"{icon('edit_note')} {revisions} revision{'s' if revisions != 1 else ''}")
    return (f"{badge_html(status)}<div class=\"rp-rcard-title\">{esc(title)}</div>"
            f'<div class="rp-rcard-desc">{esc(description)}</div>'
            f'<div class="rp-rcard-meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>')


def document_row_html(name: str, kind: str, meta: str) -> str:
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
    html_block(f'<div class="rp-empty"><span class="rp-empty-icon">{icon(icon_name)}</span>'
           f"<h3>{esc(title)}</h3><p>{esc(text)}</p></div>")


def system_status_card(state: str) -> None:
    """state: ready | limited | offline."""
    title, text, css = {
        "ready": ("System Ready", "AI research workspace is online", ""),
        "limited": ("Limited Mode", "Some research features are unavailable", "warn"),
        "offline": ("Offline", "The research service can't be reached", "down"),
    }[state]
    html_block(f'<div class="rp-sys {css}"><div class="rp-sys-title"><span class="rp-sys-dot"></span>{esc(title)}</div>'
           f'<div class="rp-sys-text">{esc(text)}</div></div>')


def system_badge(state: str) -> str:
    return {
        "ready": tone_badge("System Ready", "success"),
        "limited": tone_badge("Limited Mode", "warning"),
        "offline": tone_badge("Offline", "danger"),
    }[state]

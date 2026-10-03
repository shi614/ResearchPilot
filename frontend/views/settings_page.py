"""Settings: General, Research Preferences and Appearance - user-facing options only.

Technical configuration (keys, limits, models, storage) stays in the backend `.env`.
Preferences here are frontend-only and kept for the browser session.
"""

from __future__ import annotations

import streamlit as st

from frontend import APP_VERSION
from frontend.components import (
    PREF_COMPACT,
    PREF_DEFAULT_INSTRUCTIONS,
    PREF_SHOW_ACTIVITY,
    client,
    go_to,
    plural,
    quietly,
    system_state,
)
from frontend.ui.components import SYSTEM_LABELS, esc, html_block, icon, page_header, secondary_button, section_header

STATUS_DETAIL = {
    "ready": "All research features are available.",
    "limited": "Some research features are unavailable right now.",
    "offline": "The research service can't be reached.",
}


def render() -> None:
    page_header("Settings", "Personalise your research workspace.")
    st.write("")
    config = quietly(lambda: client().config())
    documents = quietly(lambda: client().list_documents())
    state = system_state(config)

    with st.container(key="card_general"):
        section_header("General")
        st.write("")
        model_col, kb_col, status_col = st.columns(3, gap="medium")
        with model_col:
            _value(icon_name="psychology", label="AI model",
                   value=_model_name(config["gemini_model"]) if config else "Unavailable",
                   detail="Grounded in web sources and your documents.")
        with kb_col:
            ready = sum(d["status"] == "processed" for d in documents) if documents is not None else None
            _value(icon_name="menu_book", label="Knowledge base",
                   value="Unavailable" if ready is None else ("Ready" if ready else "Empty"),
                   detail="Couldn't load documents." if ready is None else
                   f"{plural(ready, 'document')} available to search.")
            if secondary_button("Manage documents", key="manage_kb", icon_name="arrow_forward", stretch=False):
                go_to("knowledge")
        with status_col:
            _value(icon_name="monitor_heart", label="System status",
                   value={"ready": "Operational", "limited": "Limited mode", "offline": "Offline"}[state],
                   detail=STATUS_DETAIL[state])

    st.write("")
    with st.container(key="card_preferences"):
        section_header("Research Preferences", "Applied to new research in this browser session.")
        st.write("")
        _remember(PREF_DEFAULT_INSTRUCTIONS, st.text_area(
            "Default research instructions", value=st.session_state.get(PREF_DEFAULT_INSTRUCTIONS, ""),
            key=f"_w_{PREF_DEFAULT_INSTRUCTIONS}", max_chars=2000, height=80,
            placeholder="e.g. Prefer peer-reviewed and official sources; focus on research since 2022",
            help="Pre-fills the Research instructions of every new research. You can still edit them.",
        ))
        _remember(PREF_SHOW_ACTIVITY, st.toggle(
            "Show agent activity log on research pages", value=st.session_state.get(PREF_SHOW_ACTIVITY, True),
            key=f"_w_{PREF_SHOW_ACTIVITY}",
        ))

    st.write("")
    with st.container(key="card_appearance"):
        section_header("Appearance")
        st.write("")
        theme_col, density_col = st.columns(2, gap="large")
        with theme_col:
            _value(icon_name="light_mode", label="Theme", value="Light",
                   detail="Warm ivory workspace optimised for reading reports.")
        with density_col:
            _remember(PREF_COMPACT, st.toggle(
                "Compact layout", value=st.session_state.get(PREF_COMPACT, False), key=f"_w_{PREF_COMPACT}",
                help="Tighter spacing to fit more on smaller screens.",
            ))

    st.write("")
    html_block(f'<div class="rp-small">ResearchPilot {esc(APP_VERSION)} · Multi-Agent Research Workspace · '
               f"{esc(SYSTEM_LABELS[state])}</div>")


def _remember(pref_key: str, value: object) -> None:
    """Widget state is discarded when its page isn't shown; keep preferences under their own key."""
    st.session_state[pref_key] = value


def _value(*, icon_name: str, label: str, value: str, detail: str) -> None:
    html_block(f'<div class="rp-metric-label">{icon(icon_name)} {esc(label)}</div>'
               f'<div class="rp-metric-value" style="font-size:1.2rem">{esc(value)}</div>'
               f'<div class="rp-small" style="margin:.25rem 0 .5rem">{esc(detail)}</div>')


def _model_name(model_id: str) -> str:
    """'gemini-3.5-flash' -> 'Gemini 3.5 Flash'."""
    words = model_id.replace("models/", "").split("-")
    return " ".join(word if any(ch.isdigit() for ch in word) else word.capitalize() for word in words)

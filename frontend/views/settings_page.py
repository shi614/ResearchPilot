"""Settings: a short, user-facing overview (AI model, knowledge base, system status, about).

Technical configuration stays in the backend `.env` and its health/config endpoints.
"""

from __future__ import annotations

import streamlit as st

from frontend import APP_VERSION
from frontend.components import client, go_to, plural, quietly, system_state
from frontend.ui.components import esc, html_block, icon, page_header, section_header

STATUS_TEXT = {"ready": "Ready", "limited": "Limited mode", "offline": "Offline"}
STATUS_DETAIL = {
    "ready": "All research features are available.",
    "limited": "Some research features are unavailable right now.",
    "offline": "The research service can't be reached.",
}


def render() -> None:
    page_header("Settings", "Your research workspace at a glance.")
    st.write("")
    config = quietly(lambda: client().config())
    documents = quietly(lambda: client().list_documents())
    state = system_state(config)

    model_col, kb_col, status_col = st.columns(3, gap="medium")
    with model_col, st.container(key="card_model", height="stretch"):
        _setting_card("smart_toy", "AI Model", _model_name(config["gemini_model"]) if config else "Unavailable",
                      "Grounded in web sources and your documents, with citations.")
    with kb_col, st.container(key="card_kb", height="stretch"):
        ready = sum(d["status"] == "processed" for d in documents) if documents is not None else None
        value = "Unavailable" if ready is None else ("Ready" if ready else "Empty")
        detail = "Couldn't load documents." if ready is None else f"{plural(ready, 'document')} available to search."
        _setting_card("inventory_2", "Knowledge Base", value, detail)
        if st.button("Manage documents", icon=":material/arrow_forward:", key="manage_kb"):
            go_to("knowledge")
    with status_col, st.container(key="card_status", height="stretch"):
        _setting_card("monitor_heart", "System Status", STATUS_TEXT[state], STATUS_DETAIL[state])

    st.write("")
    with st.container(key="card_about"):
        section_header("About ResearchPilot")
        st.markdown("**Multi-agent AI research and report generation system.**")
        st.write("A team of specialised AI agents plans your research, searches the web and your documents, "
                 "verifies the sources against each other, pauses for your approval, and writes a cited, "
                 "fact-checked report you can download as a PDF.")
        html_block(f'<div class="rp-small" style="margin-bottom:.35rem">Version {esc(APP_VERSION)} · Built with LangGraph, Gemini, Tavily, '
                   "ChromaDB, FastAPI and Streamlit</div>")
    st.caption(":material/info: Configuration is managed by the administrator in the backend .env file.")


def _setting_card(icon_name: str, label: str, value: str, detail: str | None) -> None:
    detail_html = f'<div class="rp-small" style="margin:.3rem 0 .4rem">{esc(detail)}</div>' if detail else ""
    html_block(f'<div class="rp-metric-label">{icon(icon_name)} {esc(label)}</div>'
               f'<div class="rp-metric-value" style="font-size:1.3rem">{esc(value)}</div>{detail_html}')


def _model_name(model_id: str) -> str:
    """'gemini-3.5-flash' -> 'Gemini 3.5 Flash'."""
    words = model_id.replace("models/", "").split("-")
    return " ".join(word if any(ch.isdigit() for ch in word) else word.capitalize() for word in words)

"""New Research: hero, research composer, feature overview and the multi-agent workflow.

When a research session is open, the live session view is shown instead.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components import (
    ACTIVE_SESSION_KEY,
    PREF_DEFAULT_INSTRUCTIONS,
    call,
    client,
    open_session,
    plural,
    quietly,
    system_state,
)
from frontend.ui.components import (
    feature_card,
    html_block,
    primary_button,
    section_header,
    system_badge,
    workflow_flow,
)
from frontend.views import research_session

FEATURES = [
    ("travel_explore", "Web Research", "Search and collect relevant information."),
    ("menu_book", "Knowledge Retrieval", "Use uploaded documents as research context."),
    ("verified", "Source Verification", "Verify and organize research sources."),
    ("description", "AI Report Generation", "Transform research into a structured report."),
]
# (label, agent, variant) - the multi-agent workflow, including the human checkpoint.
FLOW = [
    ("Research Query", "You", "query"),
    ("Planning", "Planner agent", ""),
    ("Web + Knowledge Research", "Research agents", ""),
    ("Source Verification", "Verifier agent", ""),
    ("Analysis", "Analyst agent", ""),
    ("Human Approval", "You", "human"),
    ("Report Generation", "Writer agent", ""),
    ("Quality Review", "Critic + Reviser", ""),
    ("Final Report", "Cited PDF", "final"),
]
COMPOSER_KEYS = ("composer_query", "composer_instructions", "composer_files")


def render() -> None:
    session_id = st.session_state.get(ACTIVE_SESSION_KEY)
    if session_id:
        research_session.render(session_id)
        return
    config = quietly(lambda: client().config())
    documents = quietly(lambda: client().list_documents())
    state = system_state(config)
    _hero(state)
    _composer(state, research_enabled=bool(config and config["research_enabled"]), documents=documents)
    st.write("")
    section_header("Everything you need for reliable research")
    st.write("")
    for column, (icon_name, title, text) in zip(st.columns(4, gap="small"), FEATURES, strict=True):
        with column:
            feature_card(icon_name, title, text)
    st.write("")
    st.write("")
    section_header("How ResearchPilot works",
                   "A team of specialised AI agents, coordinated with LangGraph - with you in the loop.")
    st.write("")
    workflow_flow(FLOW)


def _hero(state: str) -> None:
    left, right = st.columns([5, 1.3], vertical_alignment="top")
    with left:
        html_block('<div class="rp-eyebrow">Multi-Agent Research Workspace</div>'
                   '<div class="rp-hero-title">Research smarter. Discover faster.</div>'
                   '<div class="rp-hero-text">ResearchPilot brings web research, knowledge retrieval, source '
                   "verification and AI report generation into one intelligent workflow.</div>")
    with right:
        html_block(f'<div style="text-align:right;padding-top:.2rem">{system_badge(state)}</div>')


def _composer(state: str, *, research_enabled: bool, documents: list[dict[str, Any]] | None) -> None:
    research_disabled = state != "offline" and not research_enabled
    if state == "offline":
        st.error("ResearchPilot's research service isn't reachable right now. Please start it and refresh.",
                 icon=":material/cloud_off:")
    elif research_disabled:
        st.error("Research is unavailable because the AI service isn't configured yet.", icon=":material/key_off:")
    elif state == "limited":
        st.warning("Web search is unavailable right now, so research will use your knowledge base only.",
                   icon=":material/travel_explore:")

    if "composer_instructions" not in st.session_state and st.session_state.get(PREF_DEFAULT_INSTRUCTIONS):
        st.session_state["composer_instructions"] = st.session_state[PREF_DEFAULT_INSTRUCTIONS]

    with st.container(key="composer"):
        query = st.text_area(
            "What would you like to research?", key="composer_query", height=130, max_chars=2000,
            placeholder="e.g. What are the main benefits and limitations of Retrieval-Augmented Generation?",
        )
        ready = sum(d["status"] == "processed" for d in documents or [])
        files = st.session_state.get("composer_files") or []
        kb_label = f"Knowledge base ({ready}{f' + {len(files)}' if files else ''})"
        instructions_set = bool((st.session_state.get("composer_instructions") or "").strip())
        kb_col, instr_col, _, start_col = st.columns([1.9, 1.75, 0.6, 1.6], vertical_alignment="center")
        with kb_col, st.popover(kb_label, icon=":material/menu_book:", width="stretch"):
            st.caption(f"Research automatically searches every ready document in your knowledge base "
                       f"({plural(ready, 'document')}). Add more for this research:")
            st.file_uploader("Add documents", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True,
                             key="composer_files", label_visibility="collapsed")
        with instr_col, st.popover("Research instructions" + (" ✓" if instructions_set else ""),
                                   icon=":material/tune:", width="stretch"):
            st.text_area("Instructions", key="composer_instructions", max_chars=2000, height=100,
                         placeholder="e.g. Focus on enterprise use cases and research published since 2023")
        with start_col:
            start = primary_button("Start Research →", key="start_research",
                                   disabled=research_disabled or state == "offline")
    if start:
        _start(query.strip(), (st.session_state.get("composer_instructions") or "").strip(),
               st.session_state.get("composer_files") or [])


def _start(query: str, instructions: str, files: list[Any]) -> None:
    if len(query) < 3:
        st.warning("Please enter a research topic (at least 3 characters).")
        return
    for file in files:
        with st.spinner(f"Adding {file.name} to your knowledge base..."):
            document = call(lambda f=file: client().upload_document(f.name, f.getvalue()),
                            failure=f"Could not add {file.name}")
        if document and document["status"] == "failed":
            st.warning(f"{file.name} was saved but could not be read.")
    session = call(lambda: client().start_research(query, instructions or None), failure="Could not start research")
    if session:
        for key in COMPOSER_KEYS:
            st.session_state.pop(key, None)
        open_session(session["id"])
        st.rerun()

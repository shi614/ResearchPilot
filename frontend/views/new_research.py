"""New Research: the landing screen (hero, research composer, how it works).

When a research session is open, the live session view is shown instead.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components import ACTIVE_SESSION_KEY, call, client, open_session, quietly, system_state
from frontend.ui.components import esc, how_it_works, html_block, system_badge
from frontend.views import research_session

STEPS = [
    ("account_tree", "Plan", "Breaks your question into focused research questions"),
    ("travel_explore", "Research", "Searches the web and your documents in parallel"),
    ("verified", "Verify", "Cross-checks sources and flags weak claims"),
    ("front_hand", "Review", "You approve the findings or ask for more"),
    ("description", "Report", "A cited, fact-checked report with PDF export"),
]
AGENTS = ["Planner", "Web Researcher", "Knowledge Retriever", "Source Verifier", "Analyst", "Writer",
          "Fact Checker", "Reviser"]
COMPOSER_KEYS = ("composer_query", "composer_instructions", "composer_files")


def render() -> None:
    session_id = st.session_state.get(ACTIVE_SESSION_KEY)
    if session_id:
        research_session.render(session_id)
        return
    config = quietly(lambda: client().config())
    state = system_state(config)
    _hero(state)
    _composer(state, research_enabled=bool(config and config["research_enabled"]))
    _how_it_works()


def _hero(state: str) -> None:
    left, right = st.columns([5, 1.4], vertical_alignment="center")
    with left:
        html_block('<div class="rp-hero-title">ResearchPilot</div>'
               '<div class="rp-hero-subtitle">Multi-Agent AI Research &amp; Report Generation</div>'
               '<div class="rp-hero-text">Research, verify and generate structured reports with AI agents.</div>')
    with right:
        html_block(f'<div style="text-align:right">{system_badge(state)}</div>')


def _composer(state: str, *, research_enabled: bool) -> None:
    research_disabled = state != "offline" and not research_enabled
    if state == "offline":
        st.error("ResearchPilot's research service isn't reachable right now. Please start it and refresh.",
                 icon=":material/cloud_off:")
    elif research_disabled:
        st.error("Research is unavailable because the AI service isn't configured yet.", icon=":material/key_off:")
    elif state == "limited":
        st.warning("Web search is unavailable right now, so research will use your knowledge base only.",
                   icon=":material/travel_explore:")

    with st.container(key="composer"):
        html_block('<div class="rp-section-title">What would you like to research?</div>'
               '<div class="rp-section-sub">Ask a question - the agents plan, search, verify and write the report.</div>')
        query = st.text_area(
            "Research question", key="composer_query", label_visibility="collapsed", height=120, max_chars=2000,
            placeholder="e.g. What are the main benefits and limitations of Retrieval-Augmented Generation?",
        )
        files = st.session_state.get("composer_files") or []
        docs_label = f"Add documents ({len(files)})" if files else "Add documents"
        instructions_set = bool((st.session_state.get("composer_instructions") or "").strip())
        docs_col, instr_col, _, start_col = st.columns([1.35, 1.6, 1.4, 1.6], vertical_alignment="center")
        with docs_col, st.popover(docs_label, icon=":material/attach_file:", width="stretch"):
            st.file_uploader("Documents to search alongside the web", type=["pdf", "txt", "md", "markdown"],
                             accept_multiple_files=True, key="composer_files",
                             help="Added to your knowledge base so the agents can cite them.")
        with instr_col, st.popover("Research instructions" + (" ✓" if instructions_set else ""),
                                   icon=":material/tune:", width="stretch"):
            st.text_area("Instructions", key="composer_instructions", max_chars=2000, height=100,
                         placeholder="e.g. Focus on enterprise use cases and research published since 2023")
        start = start_col.button("Start Research →", type="primary", width="stretch", key="start_research",
                                 disabled=research_disabled or state == "offline")
    if start:
        _start(query.strip(), (st.session_state.get("composer_instructions") or "").strip(),
               st.session_state.get("composer_files") or [])


def _how_it_works() -> None:
    st.write("")
    html_block('<div class="rp-section-title">How ResearchPilot works</div>'
           '<div class="rp-section-sub">A team of AI agents works through each step - you stay in control.</div>')
    how_it_works(STEPS)
    chips = " · ".join(esc(a) for a in AGENTS)
    html_block(f'<div class="rp-small" style="margin-top:.7rem">Powered by 8 specialised agents coordinated with '
           f"LangGraph: {chips}</div>")


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

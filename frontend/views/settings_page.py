"""Settings & About: a user-facing overview of the AI model, knowledge base, status and project.

Technical configuration (limits, keys, chunking, etc.) stays in the backend `.env` and its
health/config endpoints; it is intentionally not shown here.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import call, client, go_to, page_header, plural, section

AGENTS = [
    ("Planner", "turns your question into focused research questions and searches"),
    ("Web research", "searches the web and reads the most relevant pages"),
    ("Knowledge base", "finds relevant passages in your uploaded documents"),
    ("Source verification", "cross-checks sources and flags weak or conflicting claims"),
    ("Analysis", "identifies findings, patterns and gaps"),
    ("Writer", "drafts a structured report with inline citations"),
    ("Fact checker", "reviews the draft against the evidence"),
    ("Reviser", "fixes significant issues before the report is finalised"),
]


def render() -> None:
    page_header("Settings & About", "Your research assistant at a glance.")
    health = call(lambda: client().health(), failure="Could not reach ResearchPilot")
    if health is None:
        return
    config = _optional(lambda: client().config())
    documents = _optional(lambda: client().list_documents())

    _system_status(config)
    model_col, kb_col = st.columns(2, gap="medium")
    with model_col, st.container(border=True, height="stretch"):
        section("AI model")
        model = (config or {}).get("gemini_model") or health["gemini_model"]
        st.markdown(f"### {_model_name(model)}")
        st.caption(f"Model ID: {model}")
        st.write("Answers are grounded in sources from the web and your knowledge base, with citations.")
    with kb_col, st.container(border=True, height="stretch"):
        section("Knowledge base")
        _knowledge_summary(documents)

    with st.container(border=True):
        section("About ResearchPilot")
        st.write(
            "ResearchPilot is a multi-agent AI research assistant. A team of specialised agents researches "
            "your question, checks the sources against each other, pauses for your approval, and then "
            "writes a cited report you can download as a PDF."
        )
        left, right = st.columns(2, gap="large")
        for column, agents in ((left, AGENTS[:4]), (right, AGENTS[4:])):
            with column:
                st.markdown("\n".join(f"- **{name}** {text}" for name, text in agents))
        st.caption(f"Version {health['version']} · Built with LangGraph, Gemini, Tavily, ChromaDB, FastAPI "
                   "and Streamlit.")

    st.caption(":material/info: Configuration is managed in the backend's .env file.")


def _optional(action: Any) -> Any:
    """Secondary information: show the page even if it cannot be loaded."""
    try:
        return action()
    except ApiError:
        return None


def _system_status(config: dict[str, Any] | None) -> None:
    if config is None:
        st.warning("Some information could not be loaded.", icon=":material/warning:")
    elif not config["research_enabled"]:
        st.error("Research is unavailable: the AI service isn't configured yet.", icon=":material/error:")
    elif not config["web_research_enabled"]:
        st.warning("Running with limited features: web search is unavailable.", icon=":material/warning:")
    else:
        st.success("All systems operational", icon=":material/check_circle:")


def _model_name(model_id: str) -> str:
    """'gemini-3.5-flash' -> 'Gemini 3.5 Flash'."""
    words = model_id.replace("models/", "").split("-")
    return " ".join(word if any(ch.isdigit() for ch in word) else word.capitalize() for word in words)


def _knowledge_summary(documents: list[dict[str, Any]] | None) -> None:
    if documents is None:
        st.write("Knowledge base information is unavailable right now.")
        return
    ready = [d for d in documents if d["status"] == "processed"]
    needs_attention = len(documents) - len(ready)
    st.markdown(f"### {plural(len(ready), 'document')} ready")
    if not documents:
        st.caption("Upload PDFs, text or Markdown files so the agents can cite your own material.")
    else:
        st.caption(plural(sum(d["chunk_count"] for d in ready), "searchable passage"))
    if needs_attention:
        verb = "needs" if needs_attention == 1 else "need"
        st.caption(f":material/warning: {plural(needs_attention, 'document')} {verb} attention")
    if st.button("Manage knowledge base", icon=":material/library_books:", key="manage_kb"):
        go_to("knowledge")

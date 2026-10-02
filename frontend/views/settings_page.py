"""Settings: backend health, model availability and the active (read-only) configuration."""

from __future__ import annotations

import streamlit as st

from frontend.components import call, client, page_header


def render() -> None:
    page_header(
        "Settings",
        "Configuration is read from the backend's `.env` file. Edit it and restart the backend to change it.",
    )
    health = call(lambda: client().health(), failure="Could not reach the backend")
    if health is None:
        return

    st.subheader("Backend status")
    a, b, c, d = st.columns(4)
    a.metric("Backend", "OK" if health["status"] == "ok" else "Degraded")
    b.metric("Database", health["database"].capitalize())
    c.metric("Gemini key", "Configured" if health["gemini_api_key_configured"] else "Missing")
    d.metric("Tavily key", "Configured" if health["tavily_api_key_configured"] else "Missing")
    if health["missing_keys"]:
        st.warning(f"Missing in `.env`: {', '.join(health['missing_keys'])}", icon=":material/key_off:")

    st.subheader("Gemini model")
    st.write(f"Configured model: `{health['gemini_model']}`")
    check_col, probe_col = st.columns(2)
    result = None
    if check_col.button("Check availability", icon=":material/fact_check:", help="Lists models; uses no quota"):
        result = call(lambda: client().model_check(), failure="Model check failed")
    if probe_col.button("Test with one request", icon=":material/bolt:",
                        help="Sends one tiny request (uses 1 Gemini call from your free-tier quota)"):
        result = call(lambda: client().model_check(probe=True), failure="Model probe failed")
    if result:
        (st.success if result["status"] == "ok" else st.error)(result["message"])
        if result["status"] != "ok" and result.get("suggested_models"):
            st.caption("Available Flash models: " + ", ".join(f"`{m}`" for m in result["suggested_models"]))

    config = call(lambda: client().config(), failure="Could not load configuration")
    if config is None:
        return
    st.subheader("Active configuration")
    groups = {
        "Models": [("Chat model", "gemini_model"), ("Embedding model", "gemini_embedding_model")],
        "Free-tier protection": [
            ("Gemini requests / minute", "gemini_max_rpm"),
            ("Web searches per iteration", "max_web_searches"),
            ("Tool-calling rounds", "max_research_tool_rounds"),
            ("Critic revisions", "max_revisions"),
            ("Research iterations (incl. modifications)", "max_research_iterations"),
            ("Sources given to the writer", "writer_max_sources"),
        ],
        "Knowledge base": [
            ("Chunk size (chars)", "chunk_size"),
            ("Chunk overlap (chars)", "chunk_overlap"),
            ("Results per question", "rag_top_k"),
            ("Minimum relevance", "rag_min_relevance"),
            ("Max upload (MB)", "max_upload_mb"),
        ],
    }
    columns = st.columns(len(groups))
    for column, (title, items) in zip(columns, groups.items(), strict=True):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            for label, key in items:
                st.markdown(f"{label}: `{config[key]}`")

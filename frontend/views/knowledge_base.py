"""Knowledge Base: a document workspace (upload, view, re-process, delete) for the RAG agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from frontend.components import call, client, local_time, plural
from frontend.ui.components import (
    document_row_html,
    empty_state,
    html_block,
    metric_row,
    page_header,
    section_header,
    tone_badge,
)

STATUS = {"processed": ("Ready", "success"), "uploaded": ("Not processed", "neutral"),
          "failed": ("Needs attention", "danger")}


def render() -> None:
    page_header("Knowledge Base", "Upload documents that ResearchPilot can use as private research context.")
    st.write("")
    _upload_card()
    st.write("")
    _documents()


def _upload_card() -> None:
    with st.container(key="card_upload"):
        section_header("Add documents", "The agents search and cite these alongside the web.")
        files = st.file_uploader("Upload PDF, TXT or Markdown", type=["pdf", "txt", "md", "markdown"],
                                 accept_multiple_files=True, key="kb_files")
        add = st.button("Add to Knowledge Base", type="primary", icon=":material/upload:", key="kb_add",
                        disabled=not files)
    if add and files:
        for file in files:
            with st.spinner(f"Adding {file.name}... this can take a moment for large files."):
                document = call(lambda f=file: client().upload_document(f.name, f.getvalue()),
                                failure=f"Could not add {file.name}")
            if document is None:
                continue
            if document["status"] == "processed":
                st.toast(f"{file.name} is ready", icon=":material/check_circle:")
            else:
                st.warning(f"{file.name} was saved but could not be read. {document.get('error') or ''}",
                           icon=":material/warning:")
        st.session_state.pop("kb_files", None)
        st.rerun()


def _documents() -> None:
    documents = call(lambda: client().list_documents(), failure="Could not load documents")
    if documents is None:
        return
    if not documents:
        empty_state("folder_open", "No documents yet",
                    "Upload PDFs, notes or reports so the agents can cite your own material. "
                    "Until then, research uses web sources only.")
        return

    ready = [d for d in documents if d["status"] == "processed"]
    metric_row([
        ("Documents", len(documents), "description"),
        ("Ready to search", len(ready), "task_alt"),
        ("Searchable passages", sum(d["chunk_count"] for d in ready), "segment"),
    ])
    st.write("")
    section_header("Your documents")
    st.write("")
    for document in documents:
        _document_card(document)


def _document_card(document: dict[str, Any]) -> None:
    doc_id = document["id"]
    kind = Path(document["filename"]).suffix.lstrip(".").upper()[:4] or "DOC"
    label, tone = STATUS.get(document["status"], (document["status"], "neutral"))
    meta = (f"{kind} · {plural(document['chunk_count'], 'passage')} · {round(document['size_bytes'] / 1024, 1)} KB"
            f" · Uploaded {local_time(document['created_at'])}")
    with st.container(key=f"card_doc_{doc_id}"):
        info, status, actions = st.columns([5, 1.6, 2.2], vertical_alignment="center")
        with info:
            html_block(document_row_html(document["filename"], kind, meta))
        with status:
            html_block(tone_badge(label, tone))
        with actions:
            process_col, delete_col = st.columns(2)
            if document["status"] != "processed":
                if process_col.button("Re-process", icon=":material/refresh:", width="stretch", key=f"proc_{doc_id}"):
                    with st.spinner("Re-processing..."):
                        if call(lambda: client().process_document(doc_id), failure="Could not process"):
                            st.rerun()
            with delete_col, st.container(key=f"danger_doc_{doc_id}"):
                if st.button("Delete", icon=":material/delete:", width="stretch", key=f"del_{doc_id}"):
                    if call(lambda: client().delete_document(doc_id) or True, failure="Could not delete"):
                        st.toast(f"Deleted {document['filename']}", icon=":material/delete:")
                        st.rerun()
        if document.get("error"):
            st.caption(f":material/info: {document['error']}")

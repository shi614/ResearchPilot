"""Knowledge Base: a document workspace (upload, view, re-process, delete) for the RAG agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from frontend.components import call, client, local_time, plural
from frontend.ui.components import (
    document_card_html,
    empty_state,
    html_block,
    page_header,
    primary_button,
    secondary_button,
    section_header,
    tone_badge,
)

STATUS = {"processed": ("Ready", "success"), "uploaded": ("Not processed", "neutral"),
          "failed": ("Needs attention", "danger")}
KINDS = {"PDF": "PDF document", "TXT": "Text file", "MD": "Markdown", "MARK": "Markdown"}


def render() -> None:
    page_header("Knowledge Base", "Add your own documents to give ResearchPilot additional context.")
    st.write("")
    _upload_card()
    st.write("")
    _documents()


def _upload_card() -> None:
    with st.container(key="card_upload"):
        section_header("Upload documents", "PDF, TXT or Markdown - the agents search and cite them alongside the web.")
        files = st.file_uploader("Upload PDF, TXT or Markdown", type=["pdf", "txt", "md", "markdown"],
                                 accept_multiple_files=True, key="kb_files", label_visibility="collapsed")
        add = primary_button("Add to Knowledge Base", key="kb_add", icon_name="upload", disabled=not files,
                             stretch=False)
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
                    "Upload PDFs, notes or reports so ResearchPilot can cite your own material. "
                    "Until then, research uses web sources only.")
        return

    ready = sum(d["status"] == "processed" for d in documents)
    section_header("Your documents", f"{plural(len(documents), 'document')} · {ready} ready to use in research")
    st.write("")
    for start in range(0, len(documents), 2):
        for column, document in zip(st.columns(2, gap="medium"), documents[start:start + 2], strict=False):
            with column:
                _document_card(document)


def _document_card(document: dict[str, Any]) -> None:
    doc_id = document["id"]
    kind = Path(document["filename"]).suffix.lstrip(".").upper()[:4] or "DOC"
    label, tone = STATUS.get(document["status"], (document["status"], "neutral"))
    meta = f"{KINDS.get(kind, kind)} · Uploaded {local_time(document['created_at'])}"
    with st.container(key=f"card_doc_{doc_id}", height="stretch"):
        info, status = st.columns([4, 1.5], vertical_alignment="center")
        with info:
            html_block(document_card_html(document["filename"], kind, meta))
        with status:
            html_block(f'<div style="text-align:right">{tone_badge(label, tone)}</div>')
        if document.get("error"):
            st.caption(f":material/info: {document['error']}")
        actions = st.columns([1, 1.25, 1.25])  # right-aligned actions
        if document["status"] != "processed":
            with actions[1]:
                if secondary_button("Re-process", key=f"proc_{doc_id}", icon_name="refresh"):
                    with st.spinner("Re-processing..."):
                        if call(lambda: client().process_document(doc_id), failure="Could not process"):
                            st.rerun()
        with actions[2], st.container(key=f"danger_doc_{doc_id}"):
            if st.button("Delete", icon=":material/delete:", width="stretch", key=f"del_{doc_id}"):
                if call(lambda: client().delete_document(doc_id) or True, failure="Could not delete"):
                    st.toast(f"Deleted {document['filename']}", icon=":material/delete:")
                    st.rerun()

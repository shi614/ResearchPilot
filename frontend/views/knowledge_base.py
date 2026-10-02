"""Knowledge base: upload, view, (re)process and delete documents used by the RAG agent."""

from __future__ import annotations

import streamlit as st

from frontend.components import call, client, page_header

STATUS_LABELS = {"processed": "Ready", "uploaded": "Not processed", "failed": "Failed"}


def render() -> None:
    page_header(
        "Knowledge Base",
        "Documents here are chunked, embedded with Gemini and stored in ChromaDB. "
        "The knowledge-base agent searches them during research.",
    )
    _upload_section()
    st.divider()
    _documents_section()


def _upload_section() -> None:
    config = call(lambda: client().config(), failure="Could not load backend settings")
    limit = f" (max {config['max_upload_mb']} MB each)" if config else ""
    with st.form("upload_documents", clear_on_submit=True, border=True):
        files = st.file_uploader(f"Upload PDF, TXT or Markdown files{limit}", type=["pdf", "txt", "md", "markdown"],
                                 accept_multiple_files=True)
        submitted = st.form_submit_button("Upload & process", type="primary", icon=":material/upload:")
    if submitted:
        if not files:
            st.warning("Choose at least one file.")
            return
        for file in files:
            with st.spinner(f"Processing {file.name} (embedding may take a moment on the free tier)..."):
                document = call(lambda f=file: client().upload_document(f.name, f.getvalue()),
                                failure=f"Could not upload {file.name}")
            if document is None:
                continue
            if document["status"] == "processed":
                st.success(f"{file.name}: {document['chunk_count']} chunks indexed.", icon=":material/check:")
            else:
                st.warning(f"{file.name} was saved but not indexed: {document['error']}", icon=":material/warning:")


def _documents_section() -> None:
    st.subheader("Documents")
    documents = call(lambda: client().list_documents(), failure="Could not load documents")
    if documents is None:
        return
    if not documents:
        st.info("No documents yet. Research will use web sources only.", icon=":material/folder_open:")
        return

    rows = [
        {
            "File": d["filename"],
            "Status": STATUS_LABELS.get(d["status"], d["status"]),
            "Chunks": d["chunk_count"],
            "Size (KB)": round(d["size_bytes"] / 1024, 1),
            "Uploaded": str(d["created_at"])[:16].replace("T", " "),
            "Error": d.get("error") or "",
        }
        for d in documents
    ]
    selection = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key="documents_table")
    selected = selection.selection.rows if selection else []
    if not selected:
        st.caption(f"{len(documents)} document(s). Select a row to re-process or delete it.")
        return

    document = documents[selected[0]]
    with st.container(border=True):
        st.markdown(f"**{document['filename']}**")
        process_col, delete_col, _ = st.columns([2, 2, 3])
        if process_col.button("Re-process", icon=":material/refresh:", width="stretch", key="reprocess"):
            with st.spinner("Re-processing..."):
                result = call(lambda: client().process_document(document["id"]), failure="Could not process")
            if result:
                st.rerun()
        if delete_col.button("Delete", icon=":material/delete:", width="stretch", key="delete_doc"):
            if call(lambda: client().delete_document(document["id"]) or True, failure="Could not delete"):
                st.toast(f"Deleted {document['filename']}", icon=":material/delete:")
                st.rerun()

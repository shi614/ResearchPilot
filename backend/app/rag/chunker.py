"""Split loaded documents into overlapping, citation-ready chunks."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    markdown: bool = False,
) -> list[Document]:
    """Chunk page-level documents, keeping page metadata and a global chunk index.

    Markdown files split on headings first so chunks follow the document's
    structure; PDFs/plain text split on paragraphs, then sentences, then words.
    """
    if markdown:
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    chunks = [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks

"""Document validation, loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.exceptions import InvalidDocumentError
from app.rag.chunker import chunk_documents
from app.rag.loader import load_document, validate_upload
from tests.conftest import make_pdf

MB = 1_048_576


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("notes.docx", b"hello", "Unsupported file type"),
        ("noext", b"hello", "Unsupported file type"),
        ("empty.txt", b"", "is empty"),
        ("fake.pdf", b"this is not a pdf", "not a valid PDF"),
        ("binary.txt", b"abc\x00\x01\x02", "binary"),
    ],
)
def test_invalid_uploads_are_rejected(filename: str, data: bytes, message: str) -> None:
    with pytest.raises(InvalidDocumentError, match=message):
        validate_upload(filename, data, max_bytes=MB)


def test_oversized_upload_is_rejected() -> None:
    with pytest.raises(InvalidDocumentError, match="limit"):
        validate_upload("big.txt", b"a" * (MB + 1), max_bytes=MB)


@pytest.mark.parametrize("filename", ["a.pdf", "A.PDF", "b.txt", "c.md", "d.markdown"])
def test_supported_extensions_pass(filename: str) -> None:
    data = make_pdf(["x"]) if filename.lower().endswith(".pdf") else b"hello"
    assert validate_upload(filename, data, max_bytes=MB) == Path(filename).suffix.lower()


# --------------------------------------------------------------------------- loading


def test_pdf_pages_keep_page_numbers_and_skip_blank_pages(tmp_path: Path) -> None:
    pdf = make_pdf(["Solar panel efficiency improved.", "", "Wind turbines are taller."])
    docs = load_document(write(tmp_path, "energy.pdf", pdf), "energy.pdf")
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert "Wind turbines" in docs[1].page_content


def test_text_and_markdown_are_loaded_with_encoding_fallback(tmp_path: Path) -> None:
    md = load_document(write(tmp_path, "a.md", "# Title\n\nSome markdown content.".encode()), "a.md")
    assert md[0].page_content.startswith("# Title")
    cp1252 = "Café prices rose — sharply in 2024.".encode("cp1252")
    txt = load_document(write(tmp_path, "b.txt", cp1252), "b.txt")
    assert "Café" in txt[0].page_content


def test_document_without_text_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidDocumentError, match="scanned"):
        load_document(write(tmp_path, "blank.pdf", make_pdf(["", ""])), "blank.pdf")
    with pytest.raises(InvalidDocumentError, match="No usable text"):
        load_document(write(tmp_path, "ws.txt", b"   \n\n  "), "ws.txt")


def test_corrupted_pdf_raises_invalid_document(tmp_path: Path) -> None:
    corrupted = b"%PDF-1.4\n" + b"garbage" * 50
    with pytest.raises(InvalidDocumentError):
        load_document(write(tmp_path, "broken.pdf", corrupted), "broken.pdf")


# --------------------------------------------------------------------------- chunking


def test_chunks_respect_size_keep_page_metadata_and_are_indexed() -> None:
    page_text = " ".join(f"Sentence {i} about renewable energy policy." for i in range(60))
    docs = [Document(page_content=page_text, metadata={"page": 1}),
            Document(page_content=page_text, metadata={"page": 2})]
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 4
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c.metadata["page"] for c in chunks} == {1, 2}


def test_chunks_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_documents([Document(page_content=text)], chunk_size=200, chunk_overlap=60)
    first_tail = chunks[0].page_content.split()[-2:]
    assert all(word in chunks[1].page_content for word in first_tail)


def test_markdown_splits_on_headings() -> None:
    text = "# Intro\n\n" + "Intro text. " * 30 + "\n\n## Methods\n\n" + "Method text. " * 30
    chunks = chunk_documents([Document(page_content=text)], chunk_size=400, chunk_overlap=0, markdown=True)
    assert any(c.page_content.startswith("## Methods") for c in chunks)

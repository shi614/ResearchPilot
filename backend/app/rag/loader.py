"""Validation and text extraction for knowledge-base uploads (PDF, TXT, Markdown)."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.exceptions import InvalidDocumentError

logger = logging.getLogger(__name__)

CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
TEXT_ENCODINGS = ("utf-8-sig", "cp1252")
MIN_TEXT_CHARS = 20


def supported_extension(filename: str) -> str:
    """Return the lower-cased extension or raise if the file type is unsupported."""
    extension = Path(filename).suffix.lower()
    if extension not in CONTENT_TYPES:
        allowed = ", ".join(sorted(CONTENT_TYPES))
        raise InvalidDocumentError(f"Unsupported file type '{extension or 'none'}'. Allowed: {allowed}")
    return extension


def validate_upload(filename: str, data: bytes, max_bytes: int) -> str:
    """Cheap checks before anything is stored. Returns the file extension."""
    if not filename or not Path(filename).name:
        raise InvalidDocumentError("The uploaded file has no name.")
    extension = supported_extension(filename)
    if not data:
        raise InvalidDocumentError(f"'{filename}' is empty.")
    if len(data) > max_bytes:
        raise InvalidDocumentError(
            f"'{filename}' is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{max_bytes / 1_048_576:.0f} MB."
        )
    if extension == ".pdf":
        if not data.lstrip()[:5] == b"%PDF-":
            raise InvalidDocumentError(f"'{filename}' is not a valid PDF file.")
    elif b"\x00" in data[:8192]:
        raise InvalidDocumentError(f"'{filename}' looks like a binary file, not text.")
    return extension


def _decode_text(data: bytes, filename: str) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InvalidDocumentError(f"'{filename}' is not valid UTF-8 or Windows-1252 text.")


def _load_pdf(path: Path, filename: str) -> list[Document]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise InvalidDocumentError(f"'{filename}' is password-protected.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(Document(page_content=text, metadata={"page": number}))
        return pages
    except InvalidDocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise InvalidDocumentError(f"Could not read PDF '{filename}': {exc}") from exc


def load_document(path: Path, filename: str) -> list[Document]:
    """Extract text as LangChain Documents (one per PDF page, or one for text files)."""
    extension = supported_extension(filename)
    if extension == ".pdf":
        documents = _load_pdf(path, filename)
    else:
        text = _decode_text(path.read_bytes(), filename).strip()
        documents = [Document(page_content=text, metadata={})] if text else []

    total_chars = sum(len(doc.page_content) for doc in documents)
    if total_chars < MIN_TEXT_CHARS:
        hint = " It may be a scanned/image-only PDF (OCR is not supported)." if extension == ".pdf" else ""
        raise InvalidDocumentError(f"No usable text could be extracted from '{filename}'.{hint}")
    logger.info("Loaded '%s': %d section(s), %d chars", filename, len(documents), total_chars)
    return documents

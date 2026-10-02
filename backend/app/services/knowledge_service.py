"""Knowledge-base use cases: upload, process (load → chunk → embed → store), list, delete, search."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.database import Database, DocumentRepository
from app.database.orm import new_id
from app.exceptions import (
    ConfigurationError,
    DatabaseError,
    ExternalServiceError,
    InvalidDocumentError,
)
from app.models.domain import RetrievedChunk
from app.models.schemas import DocumentResponse
from app.rag.chunker import chunk_documents
from app.rag.loader import CONTENT_TYPES, load_document, validate_upload
from app.rag.vectorstore import KnowledgeBaseStore

logger = logging.getLogger(__name__)

# Failures that leave the document saved but unprocessed (the user can retry).
_PROCESSING_ERRORS = (InvalidDocumentError, ExternalServiceError, ConfigurationError, DatabaseError)


class KnowledgeService:
    def __init__(self, settings: Settings, database: Database, store: KnowledgeBaseStore) -> None:
        self._settings = settings
        self._database = database
        self._store = store

    def upload(self, filename: str, data: bytes, *, process: bool = True) -> DocumentResponse:
        """Validate and save an upload, then (optionally) index it immediately."""
        filename = Path(filename or "").name.strip()
        extension = validate_upload(filename, data, self._settings.max_upload_mb * 1_048_576)

        document_id = new_id()
        # The stored name never uses user input, so it cannot escape the upload dir.
        stored_path = self._settings.upload_dir / f"{document_id}{extension}"
        try:
            stored_path.parent.mkdir(parents=True, exist_ok=True)
            stored_path.write_bytes(data)
        except OSError as exc:
            raise DatabaseError(f"Could not save the uploaded file: {exc}") from exc

        try:
            with self._database.session() as session:
                DocumentRepository(session).create(
                    filename=filename,
                    stored_path=str(stored_path),
                    content_type=CONTENT_TYPES[extension],
                    size_bytes=len(data),
                    document_id=document_id,
                )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

        return self.process(document_id) if process else self.get(document_id)

    def process(self, document_id: str) -> DocumentResponse:
        """Load, chunk, embed and store a document. Failures are recorded, not raised."""
        with self._database.session() as session:
            record = DocumentRepository(session).get(document_id)
            filename, stored_path, content_type = record.filename, record.stored_path, record.content_type

        try:
            path = Path(stored_path)
            if not path.is_file():
                raise InvalidDocumentError(f"The stored file for '{filename}' is missing.")
            pages = load_document(path, filename)
            chunks = chunk_documents(
                pages,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
                markdown=content_type == "text/markdown",
            )
            chunk_count = self._store.add_document_chunks(document_id, filename, chunks)
        except _PROCESSING_ERRORS as exc:
            logger.warning("Processing '%s' failed: %s", filename, exc.message)
            with self._database.session() as session:
                DocumentRepository(session).mark_failed(document_id, exc.message)
            return self.get(document_id)

        with self._database.session() as session:
            DocumentRepository(session).mark_processed(document_id, chunk_count)
        return self.get(document_id)

    def get(self, document_id: str) -> DocumentResponse:
        with self._database.session() as session:
            return DocumentResponse.model_validate(DocumentRepository(session).get(document_id))

    def list_documents(self) -> list[DocumentResponse]:
        with self._database.session() as session:
            return [DocumentResponse.model_validate(d) for d in DocumentRepository(session).list_all()]

    def delete(self, document_id: str) -> None:
        """Remove chunks first, then the file, then the record (so nothing is orphaned in search)."""
        with self._database.session() as session:
            repo = DocumentRepository(session)
            record = repo.get(document_id)
            self._store.delete_document(document_id)
            Path(record.stored_path).unlink(missing_ok=True)
            repo.delete(document_id)

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        return self._store.search(
            query,
            k=k or self._settings.rag_top_k,
            min_relevance=self._settings.rag_min_relevance,
        )

    def has_documents(self) -> bool:
        return self._store.count() > 0

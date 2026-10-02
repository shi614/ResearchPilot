"""ChromaDB-backed knowledge-base vector store."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.exceptions import ConfigurationError, DatabaseError, ResearchPilotError
from app.models.domain import RetrievedChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"
FETCH_MULTIPLIER = 3  # over-fetch, then diversify across documents
MAX_CHUNKS_PER_DOCUMENT = 2


class KnowledgeBaseStore:
    """Stores document chunks with provenance metadata and retrieves them by meaning."""

    def __init__(self, persist_dir: Path, embeddings: Embeddings | None = None) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings
        client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            client=client,
            collection_metadata={"hnsw:space": "cosine"},
            # cosine distance is in [0, 2]; map to a [0, 1] relevance score
            relevance_score_fn=lambda distance: max(0.0, min(1.0, 1.0 - distance)),
        )

    def _require_embeddings(self) -> None:
        if self._embeddings is None:
            raise ConfigurationError("GEMINI_API_KEY is not set; the knowledge base cannot embed text.")

    def add_document_chunks(self, document_id: str, filename: str, chunks: list[Document]) -> int:
        """Embed and store chunks. Re-adding a document replaces its old chunks."""
        self._require_embeddings()
        if not chunks:
            return 0
        prepared = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.metadata["chunk_index"],
                    # Chroma metadata cannot hold None; -1 means "no page"
                    "page": chunk.metadata.get("page", -1),
                },
            )
            for chunk in chunks
        ]
        ids = [f"{document_id}:{doc.metadata['chunk_index']}" for doc in prepared]
        try:
            # Upsert first, then drop leftovers from a previous, longer version, so
            # a failed re-process (e.g. quota error) never loses the existing chunks.
            self._store.add_documents(prepared, ids=ids)
            self._store.delete(
                where={"$and": [{"document_id": document_id}, {"chunk_index": {"$gte": len(ids)}}]}
            )
        except ResearchPilotError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Could not store chunks in ChromaDB: {exc}") from exc
        logger.info("Stored %d chunks for document %s", len(prepared), document_id)
        return len(prepared)

    def delete_document(self, document_id: str) -> None:
        try:
            self._store.delete(where={"document_id": document_id})
        except Exception as exc:
            raise DatabaseError(f"Could not delete chunks from ChromaDB: {exc}") from exc

    def count(self, document_id: str | None = None) -> int:
        where = {"document_id": document_id} if document_id else None
        return len(self._store.get(where=where, include=[])["ids"])

    def search(self, query: str, *, k: int, min_relevance: float = 0.0) -> list[RetrievedChunk]:
        """Semantic search returning at most `k` chunks, max 2 per document for diversity."""
        self._require_embeddings()
        if not query.strip() or self.count() == 0:
            return []
        try:
            scored = self._store.similarity_search_with_relevance_scores(
                query, k=k * FETCH_MULTIPLIER
            )
        except ResearchPilotError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Knowledge-base search failed: {exc}") from exc

        results: list[RetrievedChunk] = []
        per_document: dict[str, int] = {}
        for doc, relevance in scored:
            if relevance < min_relevance:
                continue
            document_id = doc.metadata["document_id"]
            if per_document.get(document_id, 0) >= MAX_CHUNKS_PER_DOCUMENT:
                continue
            per_document[document_id] = per_document.get(document_id, 0) + 1
            page = doc.metadata.get("page", -1)
            results.append(
                RetrievedChunk(
                    document_id=document_id,
                    filename=doc.metadata["filename"],
                    chunk_index=doc.metadata["chunk_index"],
                    page=page if page and page > 0 else None,
                    content=doc.page_content,
                    relevance=round(relevance, 4),
                )
            )
            if len(results) >= k:
                break
        return results

"""Real ChromaDB tests (deterministic keyword embeddings instead of Gemini)."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.genai import errors as genai_errors
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.exceptions import ConfigurationError, ExternalServiceError, RateLimitError
from app.rag.embeddings import QuotaAwareEmbeddings
from app.rag.vectorstore import KnowledgeBaseStore
from tests.conftest import KeywordEmbeddings


def chunks(*texts: str, page: int | None = None) -> list[Document]:
    metadata = {} if page is None else {"page": page}
    return [Document(page_content=t, metadata={**metadata, "chunk_index": i}) for i, t in enumerate(texts)]


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeBaseStore:
    return KnowledgeBaseStore(tmp_path / "chroma", KeywordEmbeddings())


def test_search_returns_most_relevant_chunks_with_provenance(store: KnowledgeBaseStore) -> None:
    store.add_document_chunks("doc-solar", "solar.pdf", chunks(
        "Solar panels convert sunlight into electricity using photovoltaic cells.",
        "Photovoltaic solar panels lose efficiency at high temperatures.",
        page=4,
    ))
    store.add_document_chunks("doc-bread", "bread.txt", chunks("Sourdough bread needs flour, water and salt."))

    results = store.search("How efficient are photovoltaic solar panels?", k=3, min_relevance=0.1)
    assert results and results[0].document_id == "doc-solar"
    assert results[0].filename == "solar.pdf" and results[0].page == 4
    assert results[0].reference == "solar.pdf, p. 4"
    assert all(0 <= r.relevance <= 1 for r in results)
    assert "doc-bread" not in {r.document_id for r in results}  # below the relevance threshold


def test_text_files_have_no_page(store: KnowledgeBaseStore) -> None:
    store.add_document_chunks("d1", "notes.md", chunks("Battery storage stabilises the grid."))
    [result] = store.search("battery storage grid", k=1)
    assert result.page is None and result.reference == "notes.md"


def test_results_are_diversified_across_documents(store: KnowledgeBaseStore) -> None:
    store.add_document_chunks("a", "a.txt", chunks(*[f"climate policy carbon tax note {i}" for i in range(5)]))
    store.add_document_chunks("b", "b.txt", chunks("climate policy carbon tax overview"))
    results = store.search("climate policy carbon tax", k=4)
    assert sum(r.document_id == "a" for r in results) <= 2
    assert "b" in {r.document_id for r in results}


def test_reprocessing_replaces_old_chunks(store: KnowledgeBaseStore) -> None:
    store.add_document_chunks("d1", "x.txt", chunks("one alpha", "two beta", "three gamma"))
    assert store.count("d1") == 3
    store.add_document_chunks("d1", "x.txt", chunks("only delta"))
    assert store.count("d1") == 1
    assert [r.content for r in store.search("alpha beta gamma", k=5)] == ["only delta"]


def test_delete_document_removes_only_its_chunks(store: KnowledgeBaseStore) -> None:
    store.add_document_chunks("d1", "a.txt", chunks("hydrogen fuel cells"))
    store.add_document_chunks("d2", "b.txt", chunks("hydrogen production methods"))
    store.delete_document("d1")
    assert store.count("d1") == 0 and store.count("d2") == 1


def test_empty_store_and_blank_query_return_nothing(store: KnowledgeBaseStore) -> None:
    assert store.search("anything", k=3) == []
    store.add_document_chunks("d1", "a.txt", chunks("text"))
    assert store.search("   ", k=3) == []


def test_store_without_embeddings_cannot_add_or_search_but_can_delete(tmp_path: Path) -> None:
    store = KnowledgeBaseStore(tmp_path / "chroma", embeddings=None)
    with pytest.raises(ConfigurationError):
        store.add_document_chunks("d1", "a.txt", chunks("text"))
    with pytest.raises(ConfigurationError):
        store.search("text", k=1)
    store.delete_document("d1")  # no error


# --------------------------------------------------------------------------- quota-aware embeddings


class RecordingEmbeddings(Embeddings):
    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.batches: list[list[str]] = []
        self.failures = failures or []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.failures:
            raise self.failures.pop(0)
        self.batches.append(texts)
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def quota_429() -> genai_errors.ClientError:
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota"}})


def test_embeddings_are_sent_in_batches_preserving_order() -> None:
    inner = RecordingEmbeddings()
    embeddings = QuotaAwareEmbeddings(inner, batch_size=2, max_rpm=6000)
    vectors = embeddings.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [len(b) for b in inner.batches] == [2, 2, 1]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_embedding_rate_limit_is_retried() -> None:
    inner = RecordingEmbeddings(failures=[quota_429()])
    embeddings = QuotaAwareEmbeddings(inner, batch_size=10, max_rpm=6000, sleep=lambda _s: None)
    assert embeddings.embed_query("hello") == [5.0]


def test_exhausted_embedding_quota_raises_rate_limit_error() -> None:
    inner = RecordingEmbeddings(failures=[quota_429() for _ in range(5)])
    embeddings = QuotaAwareEmbeddings(inner, batch_size=10, max_rpm=6000, max_attempts=2, sleep=lambda _s: None)
    with pytest.raises(RateLimitError):
        embeddings.embed_documents(["x"])


def test_unexpected_embedding_errors_become_service_errors() -> None:
    inner = RecordingEmbeddings(failures=[RuntimeError("malformed response")])
    embeddings = QuotaAwareEmbeddings(inner, batch_size=10, max_rpm=6000, sleep=lambda _s: None)
    with pytest.raises(ExternalServiceError):
        embeddings.embed_documents(["x"])

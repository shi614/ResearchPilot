"""Knowledge-base endpoints: upload, list, (re)process, delete and search documents."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status

from app.api.dependencies import get_app_settings, get_knowledge_service
from app.config import Settings
from app.exceptions import InvalidDocumentError
from app.models.schemas import DocumentResponse, KnowledgeSearchRequest, KnowledgeSearchResponse
from app.services.knowledge_service import KnowledgeService

router = APIRouter(prefix="/knowledge", tags=["knowledge base"])

KnowledgeDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF/TXT/Markdown file (and index it unless process=false)",
)
def upload_document(
    service: KnowledgeDep,
    settings: Annotated[Settings, Depends(get_app_settings)],
    file: Annotated[UploadFile, File(description="PDF, TXT or Markdown file")],
    process: Annotated[bool, Query(description="Chunk and embed immediately")] = True,
) -> DocumentResponse:
    max_bytes = settings.max_upload_mb * 1_048_576
    data = file.file.read(max_bytes + 1)  # never read more than the limit + 1 byte
    if len(data) > max_bytes:
        raise InvalidDocumentError(
            f"'{file.filename}' exceeds the {settings.max_upload_mb} MB upload limit."
        )
    return service.upload(file.filename or "", data, process=process)


@router.get("/documents", response_model=list[DocumentResponse], summary="List documents")
def list_documents(service: KnowledgeDep) -> list[DocumentResponse]:
    return service.list_documents()


@router.get("/documents/{document_id}", response_model=DocumentResponse, summary="Get a document")
def get_document(document_id: str, service: KnowledgeDep) -> DocumentResponse:
    return service.get(document_id)


@router.post(
    "/documents/{document_id}/process",
    response_model=DocumentResponse,
    summary="(Re)process a document: load, chunk, embed and store in ChromaDB",
)
def process_document(document_id: str, service: KnowledgeDep) -> DocumentResponse:
    return service.process(document_id)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document, its file and its embeddings",
)
def delete_document(document_id: str, service: KnowledgeDep) -> Response:
    service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/search", response_model=KnowledgeSearchResponse, summary="Semantic search")
def search_knowledge_base(
    request: KnowledgeSearchRequest, service: KnowledgeDep
) -> KnowledgeSearchResponse:
    return KnowledgeSearchResponse(query=request.query, results=service.search(request.query, request.k))

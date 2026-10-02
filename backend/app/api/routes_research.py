"""Research endpoints: start runs, poll live progress, answer the approval pause,
retry stopped runs, browse history and fetch reports."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.dependencies import get_research_service
from app.models.agent_outputs import HumanDecision
from app.models.schemas import (
    ReportResponse,
    ResearchSessionResponse,
    ResearchSummary,
    StartResearchRequest,
)
from app.services.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])

ServiceDep = Annotated[ResearchService, Depends(get_research_service)]


@router.post(
    "",
    response_model=ResearchSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a research run in the background",
)
def start_research(request: StartResearchRequest, service: ServiceDep) -> ResearchSessionResponse:
    return service.start(request.query, request.instructions)


@router.get("", response_model=list[ResearchSummary], summary="Research history (newest first)")
def list_research(
    service: ServiceDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[ResearchSummary]:
    return service.list_sessions(limit)


@router.get(
    "/{session_id}",
    response_model=ResearchSessionResponse,
    summary="Status, live progress events, approval request and statistics",
)
def get_research(
    session_id: str,
    service: ServiceDep,
    after_event: Annotated[int, Query(ge=0, description="Only return events with a larger id")] = 0,
) -> ResearchSessionResponse:
    return service.get(session_id, after_event)


@router.post(
    "/{session_id}/decision",
    response_model=ResearchSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Answer the human-approval pause: approve, modify (with feedback) or cancel",
)
def decide(session_id: str, decision: HumanDecision, service: ServiceDep) -> ResearchSessionResponse:
    return service.decide(session_id, decision)


@router.post(
    "/{session_id}/retry",
    response_model=ResearchSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Continue a run stopped by quota, an outage or a restart, from its checkpoint",
)
def retry(session_id: str, service: ServiceDep) -> ResearchSessionResponse:
    return service.retry(session_id)


@router.get("/{session_id}/report", response_model=ReportResponse, summary="Final report (JSON + Markdown)")
def get_report(session_id: str, service: ServiceDep) -> ReportResponse:
    return service.report(session_id)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a research session, its events, report and checkpoints",
)
def delete_research(session_id: str, service: ServiceDep) -> Response:
    service.delete(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

from fastapi import APIRouter, Depends, Header, Response

from schemas import (
    CancelRunRequest,
    CreateRunRequest,
    ResumeRunRequest,
    RerunRunRequest,
    RunEventRecord,
    RunResponse,
    TraceResponse,
)
from services.run_service import run_service


router = APIRouter(prefix="/runs", tags=["runs"])


async def _headers(
    x_org_id: str = Header(..., alias="X-Org-Id"),
    x_user_id: str = Header(..., alias="X-User-Id"),
) -> tuple[str, str]:
    return x_org_id, x_user_id


@router.post("", response_model=RunResponse, status_code=201)
async def create_run(
    body: CreateRunRequest,
    response: Response,
    headers: tuple[str, str] = Depends(_headers),
) -> RunResponse:
    org_id, user_id = headers
    run, created = run_service.create_run(org_id, user_id, body)
    if not created:
        response.status_code = 200
    return run


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    headers: tuple[str, str] = Depends(_headers),
) -> RunResponse:
    org_id, _user_id = headers
    return run_service.get_run(run_id, org_id)


@router.get("/{run_id}/events", response_model=list[RunEventRecord])
async def get_run_events(
    run_id: str,
    headers: tuple[str, str] = Depends(_headers),
) -> list[RunEventRecord]:
    org_id, _user_id = headers
    return [
        RunEventRecord(**event)
        for event in run_service.get_events(run_id, org_id)
    ]


@router.post("/{run_id}/cancel", response_model=RunResponse)
async def cancel_run(
    run_id: str,
    body: CancelRunRequest,
    headers: tuple[str, str] = Depends(_headers),
) -> RunResponse:
    org_id, user_id = headers
    return run_service.cancel_run(run_id, org_id, user_id, body)


@router.post("/{run_id}/resume", response_model=RunResponse)
async def resume_run(
    run_id: str,
    body: ResumeRunRequest,
    headers: tuple[str, str] = Depends(_headers),
) -> RunResponse:
    org_id, user_id = headers
    return run_service.resume_run(run_id, org_id, user_id, body)


@router.post("/{run_id}/rerun", response_model=RunResponse, status_code=201)
async def rerun_run(
    run_id: str,
    body: RerunRunRequest,
    headers: tuple[str, str] = Depends(_headers),
) -> RunResponse:
    org_id, user_id = headers
    return run_service.rerun_run(run_id, org_id, user_id, body)


@router.get("/{run_id}/trace", response_model=TraceResponse)
async def get_run_trace(
    run_id: str,
    headers: tuple[str, str] = Depends(_headers),
) -> TraceResponse:
    org_id, _user_id = headers
    return TraceResponse(**run_service.get_trace(run_id, org_id))

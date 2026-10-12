from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CreateRunRequest(StrictModel):
    run_type: str = Field(min_length=1, max_length=100)
    idempotency_key: str = Field(min_length=8, max_length=200)
    input: dict[str, Any]


class CancelRunRequest(StrictModel):
    reason: str = Field(min_length=1, max_length=30)


class ResumeRunRequest(StrictModel):
    resume_token: str = Field(min_length=1, max_length=200)
    input: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    run_id: str
    org_id: str
    user_id: str
    run_type: str
    status: RunStatus
    input: dict[str, Any]
    output: Any | None = None
    error: Any | None = None
    created_at: datetime
    updated_at: datetime


class RunEventRecord(BaseModel):
    id: str
    run_id: str
    sequence: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


class TraceResponse(BaseModel):
    run: RunResponse
    events: list[RunEventRecord]
    steps: list[dict[str, Any]]
    model_calls: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]

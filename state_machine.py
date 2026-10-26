from __future__ import annotations

from datetime import datetime
from typing import Any


RUN_TERMINAL_STATUSES = {
    "succeeded",
    "failed",
    "cancelled",
}


ALLOWED_RUN_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "cancelled"},
    "running": {
        "queued",
        "waiting_for_user",
        "succeeded",
        "failed",
        "cancelled",
    },
    "waiting_for_user": {"queued", "cancelled"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}


ALLOWED_RUN_ACTORS: dict[tuple[str, str], set[str]] = {
    ("queued", "running"): {"worker"},
    ("queued", "cancelled"): {"user", "api", "system"},
    ("running", "queued"): {"worker", "system"},
    ("running", "waiting_for_user"): {"worker"},
    ("running", "succeeded"): {"worker"},
    ("running", "failed"): {"worker", "system"},
    ("running", "cancelled"): {"worker"},
    ("waiting_for_user", "queued"): {"user", "api"},
    ("waiting_for_user", "cancelled"): {"user", "api"},
}


class InvalidRunTransition(Exception):
    def __init__(self, from_status: str, to_status: str) -> None:
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Cannot transition from {from_status} to {to_status}"
        )


class UnauthorizedRunActor(Exception):
    def __init__(
        self,
        from_status: str,
        to_status: str,
        actor_type: str,
    ) -> None:
        self.from_status = from_status
        self.to_status = to_status
        self.actor_type = actor_type
        super().__init__(
            f"Actor {actor_type} cannot transition "
            f"from {from_status} to {to_status}"
        )


def event_type_for_transition(from_status: str, to_status: str) -> str:
    if from_status == "queued" and to_status == "running":
        return "run.started"
    if to_status == "waiting_for_user":
        return "run.waiting_for_user"
    if from_status == "waiting_for_user" and to_status == "queued":
        return "run.resumed"
    if from_status == "running" and to_status == "queued":
        return "run.retry_scheduled"
    if to_status == "succeeded":
        return "run.completed"
    if to_status == "failed":
        return "run.failed"
    if to_status == "cancelled":
        return "run.cancelled"
    return f"run.transition.{from_status}_to_{to_status}"


def apply_status_fields(run: Any, to_status: str, now: datetime) -> None:
    run.status = to_status
    run.updated_at = now

    if to_status == "running":
        run.started_at = run.started_at or now

    if to_status in RUN_TERMINAL_STATUSES:
        run.ended_at = now
        run.lease_owner = None
        run.lease_expires_at = None

    if to_status == "waiting_for_user":
        run.lease_owner = None
        run.lease_expires_at = None

    if to_status == "queued":
        run.lease_owner = None
        run.lease_expires_at = None

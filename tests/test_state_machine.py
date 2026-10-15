from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from database import SessionLocal
from models import AgentRun, AuditLog, RunEvent
from repositories.run_repository import run_repository
from state_machine import (
    ALLOWED_RUN_TRANSITIONS,
    InvalidRunTransition,
    RUN_TERMINAL_STATUSES,
)


def _create_test_run() -> str:
    run_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    run_repository.create_if_absent(
        org_id="state-machine-tests",
        run={
            "id": run_id,
            "org_id": "state-machine-tests",
            "user_id": "test-user",
            "run_type": "state-machine-test",
            "idempotency_key": f"state-machine:{run_id}",
            "input": {},
            "status": "queued",
            "output": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        },
        first_event={
            "id": str(uuid4()),
            "event_type": "run.created",
            "payload": {"status": "queued"},
            "created_at": now,
        },
    )
    return run_id


def test_run_transition_whitelist_contains_only_the_nine_supported_edges():
    assert ALLOWED_RUN_TRANSITIONS == {
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


def test_terminal_statuses_have_no_outgoing_edges():
    assert RUN_TERMINAL_STATUSES == {"succeeded", "failed", "cancelled"}
    for status in RUN_TERMINAL_STATUSES:
        assert ALLOWED_RUN_TRANSITIONS[status] == set()


def test_invalid_transition_preserves_both_statuses_for_api_errors():
    error = InvalidRunTransition("succeeded", "running")

    assert error.from_status == "succeeded"
    assert error.to_status == "running"
    assert str(error) == "Cannot transition from succeeded to running"


def test_transition_run_writes_event_and_audit_atomically():
    run_id = _create_test_run()

    updated = run_repository.transition_run(
        run_id=run_id,
        org_id="state-machine-tests",
        to_status="running",
        reason="worker_claimed_run",
        actor_type="worker",
        actor_id="worker-test",
    )

    assert updated["status"] == "running"
    assert updated["status"] == "running"

    with SessionLocal() as db:
        stored_run = db.get(AgentRun, run_id)
        event = db.scalar(
            select(RunEvent).where(
                RunEvent.run_id == run_id,
                RunEvent.event_type == "run.started",
            )
        )
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.run_id == run_id,
                AuditLog.action == "run.transition",
            )
        )

    assert stored_run is not None
    assert stored_run.started_at is not None
    assert event is not None
    assert event.payload_json["from_status"] == "queued"
    assert event.payload_json["to_status"] == "running"
    assert event.payload_json["reason"] == "worker_claimed_run"
    assert audit is not None
    assert audit.metadata_json["actor_id"] == "worker-test"


def test_transition_run_rejects_terminal_run():
    run_id = _create_test_run()
    run_repository.transition_run(
        run_id=run_id,
        org_id="state-machine-tests",
        to_status="running",
        reason="test_start",
        actor_type="worker",
        actor_id="worker-test",
    )
    run_repository.transition_run(
        run_id=run_id,
        org_id="state-machine-tests",
        to_status="succeeded",
        reason="test_complete",
        actor_type="worker",
        actor_id="worker-test",
    )

    with pytest.raises(InvalidRunTransition) as raised:
        run_repository.transition_run(
            run_id=run_id,
            org_id="state-machine-tests",
            to_status="running",
            reason="stale_job",
            actor_type="worker",
            actor_id="stale-worker",
        )

    assert raised.value.from_status == "succeeded"
    assert raised.value.to_status == "running"

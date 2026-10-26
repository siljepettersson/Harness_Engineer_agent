from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select, update

from database import SessionLocal
from models import AgentRun, AgentStep, AuditLog, Checkpoint, OutboxJob, RunEvent
from repositories.run_repository import run_repository
from state_machine import (
    ALLOWED_RUN_TRANSITIONS,
    InvalidRunTransition,
    RUN_TERMINAL_STATUSES,
    UnauthorizedRunActor,
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


def test_actor_matrix_rejects_worker_resuming_a_waiting_run():
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
        to_status="waiting_for_user",
        reason="needs_approval",
        actor_type="worker",
        actor_id="worker-test",
    )

    with pytest.raises(UnauthorizedRunActor) as raised:
        run_repository.transition_run(
            run_id=run_id,
            org_id="state-machine-tests",
            to_status="queued",
            reason="worker_resume_attempt",
            actor_type="worker",
            actor_id="worker-test",
        )

    assert raised.value.from_status == "waiting_for_user"
    assert raised.value.to_status == "queued"
    assert raised.value.actor_type == "worker"


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


def test_worker_claim_and_finish_use_transition_events():
    run_id = _create_test_run()

    claimed = run_repository.claim_run(run_id, "worker-integration")
    assert claimed is not None
    assert claimed["status"] == "running"

    assert run_repository.finish_run(
        run_id,
        "worker-integration",
        "succeeded",
        output={"answer": "done"},
    )

    with SessionLocal() as db:
        event_types = [
            event.event_type
            for event in db.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == run_id)
                .order_by(RunEvent.sequence)
            ).all()
        ]

    assert event_types == ["run.created", "run.started", "run.completed"]


def test_sweeper_requeues_through_the_transition_path():
    run_id = _create_test_run()
    assert run_repository.claim_run(run_id, "worker-expired") is not None

    with SessionLocal.begin() as db:
        db.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                lease_expires_at=(
                    datetime.now(timezone.utc) - timedelta(seconds=1)
                )
            )
        )

    assert run_repository.recover_expired_leases() == 1
    assert run_repository.get(run_id, "state-machine-tests")["status"] == "queued"

    with SessionLocal() as db:
        event = db.scalar(
            select(RunEvent)
            .where(
                RunEvent.run_id == run_id,
                RunEvent.event_type == "run.retry_scheduled",
            )
        )
        audit = db.scalar(
            select(AuditLog).where(
                AuditLog.run_id == run_id,
                AuditLog.action == "run.transition",
            ).order_by(AuditLog.created_at.desc())
        )

    assert event is not None
    assert event.payload_json["reason"] == "worker_lease_expired"
    assert audit is not None
    assert audit.metadata_json["actor_id"] == "lease_sweeper"


def test_wait_for_user_persists_step_checkpoint_and_resume_atomically():
    run_id = _create_test_run()
    assert run_repository.claim_run(run_id, "worker-waiting") is not None
    step_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    with SessionLocal.begin() as db:
        db.add(
            AgentStep(
                id=step_id,
                run_id=run_id,
                step_index=1,
                step_type="agent_loop",
                name="refund_order",
                status="pending",
                input_json={},
                output_json=None,
                error_json=None,
                started_at=started_at,
                ended_at=None,
            )
        )

    waiting = run_repository.wait_for_user(
        run_id=run_id,
        org_id="state-machine-tests",
        worker_id="worker-waiting",
        step_id=step_id,
        checkpoint_state={"current_step": "refund_order"},
        reason="supervisor_required",
    )
    assert waiting["status"] == "waiting_for_user"
    assert waiting["lease_owner"] is None

    with SessionLocal() as db:
        step = db.get(AgentStep, step_id)
        checkpoint = db.scalar(
            select(Checkpoint).where(Checkpoint.run_id == run_id)
        )
    assert step is not None
    assert step.status == "waiting"
    assert checkpoint is not None
    assert checkpoint.state_json == {"current_step": "refund_order"}

    resumed = run_repository.resume_run(
        run_id=run_id,
        org_id="state-machine-tests",
        actor_id="user-test",
        resume_payload={"approved": True},
    )
    assert resumed["status"] == "queued"
    with SessionLocal() as db:
        step = db.get(AgentStep, step_id)
        jobs = db.scalars(
            select(OutboxJob).where(
                OutboxJob.job_type == "execute_run",
                OutboxJob.payload_json == {"run_id": run_id},
            )
        ).all()
    assert step is not None
    assert step.status == "pending"
    assert jobs

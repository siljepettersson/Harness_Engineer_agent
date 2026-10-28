from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select

from database import SessionLocal
from models import (
    AgentRun,
    AgentStep,
    AuditLog,
    Checkpoint,
    DeadLetterJob,
    ModelCall,
    OutboxJob,
    RunEvent,
    ToolCall,
    UsageRecord,
)
from state_machine import (
    ALLOWED_RUN_TRANSITIONS,
    InvalidRunTransition,
    UnauthorizedRunActor,
    apply_status_fields,
    event_type_for_transition,
)


class SqlAlchemyRunRepository:
    """SQLAlchemy persistence for runs and their event stream.

    The service will be switched to this repository after the database-backed
    operations have been verified independently.
    """

    @staticmethod
    def _run_to_dict(run: AgentRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "parent_run_id": run.parent_run_id,
            "org_id": run.org_id,
            "user_id": run.user_id,
            "run_type": run.run_type,
            "idempotency_key": run.idempotency_key,
            "input": run.input_json,
            "input_hash": None,
            "status": run.status,
            "output": run.output_json,
            "error": run.error_json,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
            "lease_owner": run.lease_owner,
            "lease_expires_at": (
                run.lease_expires_at.isoformat() if run.lease_expires_at else None
            ),
            "cancel_requested_at": (
                run.cancel_requested_at.isoformat()
                if run.cancel_requested_at
                else None
            ),
            "attempts": run.attempts,
            "rerun_reason": run.rerun_reason,
        }

    @staticmethod
    def _event_to_dict(event: RunEvent) -> dict[str, Any]:
        return {
            "id": event.id,
            "run_id": event.run_id,
            "sequence": event.sequence,
            "event_type": event.event_type,
            "payload": event.payload_json,
            "created_at": event.created_at.isoformat(),
        }

    @staticmethod
    def _datetime(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    @staticmethod
    def _step_to_dict(step: AgentStep) -> dict[str, Any]:
        return {
            "id": step.id,
            "run_id": step.run_id,
            "step_index": step.step_index,
            "step_type": step.step_type,
            "name": step.name,
            "status": step.status,
            "input_json": step.input_json,
            "output_json": step.output_json,
            "error_json": step.error_json,
            "started_at": step.started_at.isoformat(),
            "ended_at": step.ended_at.isoformat() if step.ended_at else None,
        }

    @staticmethod
    def _model_call_to_dict(call: ModelCall) -> dict[str, Any]:
        return {
            "id": call.id,
            "run_id": call.run_id,
            "step_id": call.step_id,
            "provider": call.provider,
            "model": call.model,
            "request_json": call.request_json,
            "response_json": call.response_json,
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "latency_ms": call.latency_ms,
            "status": call.status,
            "error_json": call.error_json,
            "created_at": call.created_at.isoformat(),
        }

    @staticmethod
    def _tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
        return {
            "id": call.id,
            "run_id": call.run_id,
            "step_id": call.step_id,
            "tool_name": call.tool_name,
            "args_json": call.args_json,
            "result_json": call.result_json,
            "status": call.status,
            "idempotency_key": call.idempotency_key,
            "latency_ms": call.latency_ms,
            "error_json": call.error_json,
            "created_at": call.created_at.isoformat(),
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
        }

    @staticmethod
    def _checkpoint_to_dict(checkpoint: Checkpoint) -> dict[str, Any]:
        return {
            "id": checkpoint.id,
            "run_id": checkpoint.run_id,
            "checkpoint_index": checkpoint.checkpoint_index,
            "state_json": checkpoint.state_json,
            "created_at": checkpoint.created_at.isoformat(),
        }

    @staticmethod
    def _usage_to_dict(record: UsageRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "run_id": record.run_id,
            "org_id": record.org_id,
            "user_id": record.user_id,
            "provider": record.provider,
            "model": record.model,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
            "estimated_cost_cents": record.estimated_cost_cents,
            "created_at": record.created_at.isoformat(),
        }

    @staticmethod
    def _audit_to_dict(log: AuditLog) -> dict[str, Any]:
        return {
            "id": log.id,
            "org_id": log.org_id,
            "actor": log.actor,
            "run_id": log.run_id,
            "action": log.action,
            "target_type": log.target_type,
            "target_id": log.target_id,
            "metadata_json": log.metadata_json,
            "created_at": log.created_at.isoformat(),
        }

    def _run_belongs_to_org(self, db, run_id: str, org_id: str) -> bool:
        return db.scalar(
            select(AgentRun.id).where(
                AgentRun.id == run_id,
                AgentRun.org_id == org_id,
            )
        ) is not None

    def create_if_absent(
        self,
        org_id: str,
        run: dict[str, Any],
        first_event: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Create a run and run.created atomically, or return an existing run."""
        with SessionLocal.begin() as db:
            existing = db.scalar(
                select(AgentRun).where(
                    AgentRun.org_id == org_id,
                    AgentRun.idempotency_key == run["idempotency_key"],
                )
            )
            if existing is not None:
                return self._run_to_dict(existing), False

            created_at = self._datetime(run["created_at"])
            updated_at = self._datetime(run["updated_at"])
            db_run = AgentRun(
                id=run["id"],
                parent_run_id=run.get("parent_run_id"),
                org_id=org_id,
                user_id=run["user_id"],
                run_type=run["run_type"],
                status=run["status"],
                idempotency_key=run["idempotency_key"],
                input_json=run["input"],
                output_json=run.get("output"),
                error_json=run.get("error"),
                created_at=created_at,
                updated_at=updated_at,
                rerun_reason=run.get("rerun_reason"),
            )
            db_event = RunEvent(
                id=first_event["id"],
                run_id=run["id"],
                sequence=1,
                event_type=first_event["event_type"],
                payload_json=first_event["payload"],
                created_at=self._datetime(first_event["created_at"]),
            )
            db_outbox_job = OutboxJob(
                id=str(uuid4()),
                job_type="execute_run",
                payload_json={"run_id": run["id"]},
                status="pending",
                attempts=0,
                max_attempts=10,
                next_attempt_at=created_at,
                created_at=created_at,
            )
            db.add_all([db_run, db_event, db_outbox_job])
            db.flush()
            return self._run_to_dict(db_run), True

    def get(self, run_id: str, org_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            run = db.scalar(
                select(AgentRun).where(
                    AgentRun.id == run_id,
                    AgentRun.org_id == org_id,
                )
            )
            return self._run_to_dict(run) if run is not None else None

    def claim_run(
        self,
        run_id: str,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        """Atomically transition a queued run to running and claim its lease."""
        now = datetime.now(timezone.utc)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun)
                .where(AgentRun.id == run_id)
                .with_for_update()
            )
            if run is None or run.status != "queued":
                return None

            transitioned = self._transition_in_tx(
                db,
                run,
                to_status="running",
                reason="worker_claimed_run",
                actor_type="worker",
                actor_id=worker_id,
                payload={"lease_expires_at": lease_expires_at.isoformat()},
            )
            run.lease_owner = worker_id
            run.lease_expires_at = lease_expires_at
            run.attempts += 1
            db.flush()
            return self._run_to_dict(transitioned)

    def finish_run(
        self,
        run_id: str,
        worker_id: str,
        status: str,
        output: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> bool:
        """Transition a leased running run to a terminal state."""
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun)
                .where(
                    AgentRun.id == run_id,
                    AgentRun.status == "running",
                    AgentRun.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if run is None:
                return False
            self._transition_in_tx(
                db,
                run,
                to_status=status,
                reason="run_finished" if status == "succeeded" else "run_failed",
                actor_type="worker",
                actor_id=worker_id,
                payload={"output": output} if output is not None else {},
                output=output,
                error=error,
            )
            return True

    def recover_expired_leases(self, limit: int = 100) -> int:
        """Return expired running leases to the queue with a new outbox job."""
        now = datetime.now(timezone.utc)
        recovered = 0
        with SessionLocal.begin() as db:
            runs = db.scalars(
                select(AgentRun)
                .where(
                    AgentRun.status == "running",
                    AgentRun.lease_expires_at.is_not(None),
                    AgentRun.lease_expires_at <= now,
                )
                .order_by(AgentRun.lease_expires_at.asc())
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).all()

            for run in runs:
                previous_owner = run.lease_owner
                self._transition_in_tx(
                    db,
                    run,
                    to_status="queued",
                    reason="worker_lease_expired",
                    actor_type="system",
                    actor_id="lease_sweeper",
                    payload={"previous_owner": previous_owner},
                )
                db.add(
                    OutboxJob(
                        id=str(uuid4()),
                        job_type="execute_run",
                        payload_json={"run_id": run.id},
                        status="pending",
                        attempts=0,
                        max_attempts=10,
                        next_attempt_at=now,
                        created_at=now,
                    )
                )
                recovered += 1

        return recovered

    def transition_run(
        self,
        run_id: str,
        to_status: str,
        reason: str,
        actor_type: str,
        actor_id: str,
        payload: dict[str, Any] | None = None,
        org_id: str | None = None,
        expected_status: str | None = None,
        output: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply the only supported run-status transition.

        The row lock, whitelist, derived fields, event, and audit record all
        live in one transaction. Callers must not assign AgentRun.status.
        """
        with SessionLocal.begin() as db:
            filters = [AgentRun.id == run_id]
            if org_id is not None:
                filters.append(AgentRun.org_id == org_id)
            run = db.scalar(
                select(AgentRun).where(*filters).with_for_update()
            )
            if run is None:
                raise KeyError(run_id)
            if expected_status is not None and run.status != expected_status:
                raise InvalidRunTransition(run.status, to_status)
            self._transition_in_tx(
                db,
                run,
                to_status=to_status,
                reason=reason,
                actor_type=actor_type,
                actor_id=actor_id,
                payload=payload,
                output=output,
                error=error,
            )
            return self._run_to_dict(run)

    def _transition_in_tx(
        self,
        db,
        run: AgentRun,
        to_status: str,
        reason: str,
        actor_type: str,
        actor_id: str,
        payload: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> AgentRun:
        from_status = run.status
        allowed = ALLOWED_RUN_TRANSITIONS.get(from_status, set())
        if to_status not in allowed:
            raise InvalidRunTransition(from_status, to_status)
        allowed_actors = self._allowed_actors(from_status, to_status)
        if actor_type not in allowed_actors:
            raise UnauthorizedRunActor(from_status, to_status, actor_type)

        now = datetime.now(timezone.utc)
        apply_status_fields(run, to_status, now)
        if output is not None:
            run.output_json = output
        if error is not None:
            run.error_json = error

        last_sequence = db.scalar(
            select(func.max(RunEvent.sequence)).where(RunEvent.run_id == run.id)
        ) or 0
        event_payload = {
            "from_status": from_status,
            "to_status": to_status,
            "reason": reason,
            "actor_type": actor_type,
            "actor_id": actor_id,
            **(payload or {}),
        }
        db.add(
            RunEvent(
                id=str(uuid4()),
                run_id=run.id,
                sequence=last_sequence + 1,
                event_type=event_type_for_transition(from_status, to_status),
                payload_json=event_payload,
                created_at=now,
            )
        )
        db.add(
            AuditLog(
                id=str(uuid4()),
                org_id=run.org_id,
                actor=f"{actor_type}:{actor_id}",
                run_id=run.id,
                action="run.transition",
                target_type="agent_run",
                target_id=run.id,
                metadata_json={
                    "from_status": from_status,
                    "to_status": to_status,
                    "reason": reason,
                    "actor_type": actor_type,
                    "actor_id": actor_id,
                },
                created_at=now,
            )
        )
        db.flush()
        return run

    @staticmethod
    def _allowed_actors(from_status: str, to_status: str) -> set[str]:
        from state_machine import ALLOWED_RUN_ACTORS

        return ALLOWED_RUN_ACTORS.get((from_status, to_status), set())

    def request_cancel(
        self,
        run_id: str,
        org_id: str,
        actor_type: str,
        actor_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Cancel an idle run or record a cooperative cancel request."""
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun)
                .where(AgentRun.id == run_id, AgentRun.org_id == org_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)

            if run.status in {"queued", "waiting_for_user"}:
                self._transition_in_tx(
                    db,
                    run,
                    to_status="cancelled",
                    reason=reason,
                    actor_type=actor_type,
                    actor_id=actor_id,
                )
            elif run.status == "running":
                now = datetime.now(timezone.utc)
                run.cancel_requested_at = now
                run.updated_at = now
                last_sequence = db.scalar(
                    select(func.max(RunEvent.sequence)).where(
                        RunEvent.run_id == run.id
                    )
                ) or 0
                payload = {
                    "reason": reason,
                    "actor_type": actor_type,
                    "actor_id": actor_id,
                }
                db.add(
                    RunEvent(
                        id=str(uuid4()),
                        run_id=run.id,
                        sequence=last_sequence + 1,
                        event_type="run.cancel_requested",
                        payload_json=payload,
                        created_at=now,
                    )
                )
                db.add(
                    AuditLog(
                        id=str(uuid4()),
                        org_id=run.org_id,
                        actor=f"{actor_type}:{actor_id}",
                        run_id=run.id,
                        action="run.cancel_requested",
                        target_type="agent_run",
                        target_id=run.id,
                        metadata_json=payload,
                        created_at=now,
                    )
                )
                db.flush()
            else:
                raise InvalidRunTransition(run.status, "cancelled")

            return self._run_to_dict(run)

    def resume_run(
        self,
        run_id: str,
        org_id: str,
        actor_id: str,
        resume_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Resume a waiting run and enqueue the next execution atomically."""
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun)
                .where(AgentRun.id == run_id, AgentRun.org_id == org_id)
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)

            self._transition_in_tx(
                db,
                run,
                to_status="queued",
                reason="user_resumed_run",
                actor_type="user",
                actor_id=actor_id,
                payload=resume_payload,
            )
            waiting_step = db.scalar(
                select(AgentStep)
                .where(
                    AgentStep.run_id == run.id,
                    AgentStep.status == "waiting",
                )
                .order_by(AgentStep.step_index.desc())
            )
            if waiting_step is not None:
                waiting_step.status = "pending"
            now = datetime.now(timezone.utc)
            db.add(
                OutboxJob(
                    id=str(uuid4()),
                    job_type="execute_run",
                    payload_json={"run_id": run.id},
                    status="pending",
                    attempts=0,
                    max_attempts=10,
                    next_attempt_at=now,
                    created_at=now,
                )
            )
            db.flush()
            return self._run_to_dict(run)

    def wait_for_user(
        self,
        run_id: str,
        org_id: str,
        worker_id: str,
        step_id: str,
        checkpoint_state: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """Pause a run and persist its waiting step and checkpoint atomically."""
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun)
                .where(
                    AgentRun.id == run_id,
                    AgentRun.org_id == org_id,
                    AgentRun.status == "running",
                    AgentRun.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)

            step = db.scalar(
                select(AgentStep)
                .where(
                    AgentStep.id == step_id,
                    AgentStep.run_id == run_id,
                )
                .with_for_update()
            )
            if step is None:
                raise KeyError(step_id)

            self._transition_in_tx(
                db,
                run,
                to_status="waiting_for_user",
                reason=reason,
                actor_type="worker",
                actor_id=worker_id,
                payload={"step_id": step_id},
            )
            step.status = "waiting"
            step.ended_at = None

            checkpoint_index = db.scalar(
                select(func.max(Checkpoint.checkpoint_index)).where(
                    Checkpoint.run_id == run_id
                )
            ) or 0
            db.add(
                Checkpoint(
                    id=str(uuid4()),
                    run_id=run_id,
                    checkpoint_index=checkpoint_index + 1,
                    state_json=checkpoint_state,
                    created_at=datetime.now(timezone.utc),
                )
            )
            db.flush()
            return self._run_to_dict(run)

    def cancel_if_requested(
        self,
        run_id: str,
        worker_id: str,
    ) -> bool:
        """Perform a cooperative cancellation at a worker safe point."""
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun)
                .where(
                    AgentRun.id == run_id,
                    AgentRun.status == "running",
                    AgentRun.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if run is None or run.cancel_requested_at is None:
                return False
            self._transition_in_tx(
                db,
                run,
                to_status="cancelled",
                reason="cancel_request_reached_safe_point",
                actor_type="worker",
                actor_id=worker_id,
                payload={
                    "cancel_requested_at": run.cancel_requested_at.isoformat(),
                },
            )
            return True

    def schedule_retry(
        self,
        run_id: str,
        worker_id: str,
        error: dict[str, Any],
        max_attempts: int = 3,
        backoff_seconds: int = 2,
    ) -> dict[str, Any]:
        """Retry a transient failure before terminalizing the run."""
        now = datetime.now(timezone.utc)
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun)
                .where(
                    AgentRun.id == run_id,
                    AgentRun.status == "running",
                    AgentRun.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if run is None:
                raise KeyError(run_id)

            if run.attempts >= max_attempts:
                self._transition_in_tx(
                    db,
                    run,
                    to_status="failed",
                    reason="retry_limit_exhausted",
                    actor_type="worker",
                    actor_id=worker_id,
                    payload={"attempts": run.attempts, "max_attempts": max_attempts},
                    error=error,
                )
                return self._run_to_dict(run)

            next_attempt_at = now + timedelta(seconds=backoff_seconds)
            self._transition_in_tx(
                db,
                run,
                to_status="queued",
                reason="transient_error",
                actor_type="worker",
                actor_id=worker_id,
                payload={
                    "error": error,
                    "attempts": run.attempts,
                    "max_attempts": max_attempts,
                    "next_attempt_at": next_attempt_at.isoformat(),
                },
                error=error,
            )
            db.add(
                OutboxJob(
                    id=str(uuid4()),
                    job_type="execute_run",
                    payload_json={"run_id": run.id},
                    status="pending",
                    attempts=0,
                    max_attempts=10,
                    next_attempt_at=next_attempt_at,
                    created_at=now,
                )
            )
            db.flush()
            return self._run_to_dict(run)

    def record_dead_letter(
        self,
        original_job_id: str,
        job_type: str,
        payload: dict[str, Any],
        attempts: int,
        error: dict[str, Any],
    ) -> bool:
        """Persist a failed job once so operators can inspect or replay it."""
        with SessionLocal.begin() as db:
            existing = db.scalar(
                select(DeadLetterJob).where(
                    DeadLetterJob.original_job_id == original_job_id
                )
            )
            if existing is not None:
                return False
            db.add(
                DeadLetterJob(
                    id=str(uuid4()),
                    original_job_id=original_job_id,
                    job_type=job_type,
                    payload_json=payload,
                    attempts=attempts,
                    error_json=error,
                    created_at=datetime.now(timezone.utc),
                )
            )
            return True

    def list_events(self, run_id: str, org_id: str) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            events = db.scalars(
                select(RunEvent)
                .join(AgentRun, AgentRun.id == RunEvent.run_id)
                .where(
                    RunEvent.run_id == run_id,
                    AgentRun.org_id == org_id,
                )
                .order_by(RunEvent.sequence.asc())
            ).all()
            return [self._event_to_dict(event) for event in events]

    def create_step(self, org_id: str, step: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            if not self._run_belongs_to_org(db, step["run_id"], org_id):
                raise ValueError("Run not found")
            db_step = AgentStep(
                id=step["id"],
                run_id=step["run_id"],
                step_index=step["step_index"],
                step_type=step["step_type"],
                name=step["name"],
                status=step["status"],
                input_json=step.get("input_json"),
                output_json=step.get("output_json"),
                error_json=step.get("error_json"),
                started_at=self._datetime(step["started_at"]),
                ended_at=(self._datetime(step["ended_at"]) if step.get("ended_at") else None),
            )
            db.add(db_step)
            db.flush()
            return self._step_to_dict(db_step)

    def update_step(
        self,
        org_id: str,
        step_id: str,
        status: str,
        output_json: dict[str, Any] | None = None,
        error_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            step = db.scalar(
                select(AgentStep)
                .join(AgentRun, AgentRun.id == AgentStep.run_id)
                .where(
                    AgentStep.id == step_id,
                    AgentRun.org_id == org_id,
                )
                .with_for_update()
            )
            if step is None:
                raise KeyError(step_id)

            step.status = status
            step.output_json = output_json
            step.error_json = error_json
            step.ended_at = datetime.now(timezone.utc)
            db.flush()
            return self._step_to_dict(step)

    def append_event(
        self,
        org_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            if not self._run_belongs_to_org(db, run_id, org_id):
                raise ValueError("Run not found")

            sequence = db.scalar(
                select(func.max(RunEvent.sequence)).where(
                    RunEvent.run_id == run_id
                )
            ) or 0
            event = RunEvent(
                id=str(uuid4()),
                run_id=run_id,
                sequence=sequence + 1,
                event_type=event_type,
                payload_json=payload or {},
                created_at=datetime.now(timezone.utc),
            )
            db.add(event)
            db.flush()
            return self._event_to_dict(event)

    def record_model_call(self, org_id: str, call: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            if not self._run_belongs_to_org(db, call["run_id"], org_id):
                raise ValueError("Run not found")
            db_call = ModelCall(
                id=call["id"],
                run_id=call["run_id"],
                step_id=call.get("step_id"),
                provider=call["provider"],
                model=call["model"],
                request_json=call["request_json"],
                response_json=call.get("response_json"),
                prompt_tokens=call.get("prompt_tokens"),
                completion_tokens=call.get("completion_tokens"),
                latency_ms=call.get("latency_ms"),
                status=call["status"],
                error_json=call.get("error_json"),
                created_at=self._datetime(call["created_at"]),
            )
            db.add(db_call)
            db.flush()
            return self._model_call_to_dict(db_call)

    def record_tool_call(self, org_id: str, call: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            if not self._run_belongs_to_org(db, call["run_id"], org_id):
                raise ValueError("Run not found")
            db_call = ToolCall(
                id=call["id"],
                run_id=call["run_id"],
                step_id=call.get("step_id"),
                tool_name=call["tool_name"],
                args_json=call["args_json"],
                result_json=call.get("result_json"),
                status=call["status"],
                idempotency_key=call.get("idempotency_key"),
                latency_ms=call.get("latency_ms"),
                error_json=call.get("error_json"),
                created_at=self._datetime(call["created_at"]),
                ended_at=(self._datetime(call["ended_at"]) if call.get("ended_at") else None),
            )
            db.add(db_call)
            db.flush()
            return self._tool_call_to_dict(db_call)

    def update_tool_call(
        self,
        org_id: str,
        tool_call_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            db_call = db.scalar(
                select(ToolCall)
                .join(AgentRun, AgentRun.id == ToolCall.run_id)
                .where(
                    ToolCall.id == tool_call_id,
                    AgentRun.org_id == org_id,
                )
            )
            if db_call is None:
                raise ValueError("Tool call not found")

            if "status" in updates:
                db_call.status = updates["status"]
            if "result_json" in updates:
                db_call.result_json = updates["result_json"]
            if "error_json" in updates:
                db_call.error_json = updates["error_json"]
            if "latency_ms" in updates:
                db_call.latency_ms = updates["latency_ms"]
            if "ended_at" in updates:
                ended_at = updates["ended_at"]
                db_call.ended_at = (
                    self._datetime(ended_at)
                    if ended_at is not None
                    else None
                )

            db.flush()
            return self._tool_call_to_dict(db_call)

    def get_tool_call_by_idempotency_key(
        self,
        org_id: str,
        tool_name: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with SessionLocal() as db:
            call = db.scalar(
                select(ToolCall)
                .join(AgentRun, AgentRun.id == ToolCall.run_id)
                .where(
                    AgentRun.org_id == org_id,
                    ToolCall.tool_name == tool_name,
                    ToolCall.idempotency_key == idempotency_key,
                )
            )
            return self._tool_call_to_dict(call) if call is not None else None

    def save_checkpoint(self, org_id: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            if not self._run_belongs_to_org(db, checkpoint["run_id"], org_id):
                raise ValueError("Run not found")
            db_checkpoint = Checkpoint(
                id=checkpoint["id"],
                run_id=checkpoint["run_id"],
                checkpoint_index=checkpoint["checkpoint_index"],
                state_json=checkpoint["state_json"],
                created_at=self._datetime(checkpoint["created_at"]),
            )
            db.add(db_checkpoint)
            db.flush()
            return self._checkpoint_to_dict(db_checkpoint)

    def get_latest_checkpoint(
        self,
        run_id: str,
        org_id: str,
    ) -> dict[str, Any] | None:
        """Return the newest checkpoint for an organization-owned run."""
        with SessionLocal() as db:
            checkpoint = db.scalar(
                select(Checkpoint)
                .join(AgentRun, AgentRun.id == Checkpoint.run_id)
                .where(
                    Checkpoint.run_id == run_id,
                    AgentRun.org_id == org_id,
                )
                .order_by(Checkpoint.checkpoint_index.desc())
            )
            if checkpoint is None:
                return None
            return self._checkpoint_to_dict(checkpoint)

    def next_checkpoint_index(
        self,
        run_id: str,
        org_id: str,
    ) -> int:
        """Return the next monotonic checkpoint number for a run."""
        with SessionLocal() as db:
            if not self._run_belongs_to_org(db, run_id, org_id):
                raise ValueError("Run not found")

            current = db.scalar(
                select(func.max(Checkpoint.checkpoint_index)).where(
                    Checkpoint.run_id == run_id
                )
            )
            return (current or 0) + 1

    def record_usage(self, org_id: str, usage: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            if not self._run_belongs_to_org(db, usage["run_id"], org_id):
                raise ValueError("Run not found")
            record = UsageRecord(
                id=usage["id"],
                run_id=usage["run_id"],
                org_id=org_id,
                user_id=usage["user_id"],
                provider=usage["provider"],
                model=usage.get("model"),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                estimated_cost_cents=usage.get("estimated_cost_cents", 0),
                created_at=self._datetime(usage["created_at"]),
            )
            db.add(record)
            db.flush()
            return self._usage_to_dict(record)

    def record_audit_log(self, org_id: str, log: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal.begin() as db:
            if log.get("run_id") and not self._run_belongs_to_org(db, log["run_id"], org_id):
                raise ValueError("Run not found")
            record = AuditLog(
                id=log["id"],
                org_id=org_id,
                actor=log["actor"],
                run_id=log.get("run_id"),
                action=log["action"],
                target_type=log["target_type"],
                target_id=log["target_id"],
                metadata_json=log.get("metadata_json"),
                created_at=self._datetime(log["created_at"]),
            )
            db.add(record)
            db.flush()
            return self._audit_to_dict(record)

    def get_trace(self, run_id: str, org_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            run = db.scalar(
                select(AgentRun).where(
                    AgentRun.id == run_id,
                    AgentRun.org_id == org_id,
                )
            )
            if run is None:
                return None

            steps = db.scalars(
                select(AgentStep)
                .where(AgentStep.run_id == run_id)
                .order_by(AgentStep.step_index.asc())
            ).all()
            events = db.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == run_id)
                .order_by(RunEvent.sequence.asc())
            ).all()
            model_calls = db.scalars(
                select(ModelCall)
                .where(ModelCall.run_id == run_id)
                .order_by(ModelCall.created_at.asc())
            ).all()
            tool_calls = db.scalars(
                select(ToolCall)
                .where(ToolCall.run_id == run_id)
                .order_by(ToolCall.created_at.asc())
            ).all()
            checkpoints = db.scalars(
                select(Checkpoint)
                .where(Checkpoint.run_id == run_id)
                .order_by(Checkpoint.checkpoint_index.asc())
            ).all()

            return {
                "run": self._run_to_dict(run),
                "steps": [self._step_to_dict(step) for step in steps],
                "events": [self._event_to_dict(event) for event in events],
                "model_calls": [self._model_call_to_dict(call) for call in model_calls],
                "tool_calls": [self._tool_call_to_dict(call) for call in tool_calls],
                "checkpoints": [self._checkpoint_to_dict(checkpoint) for checkpoint in checkpoints],
            }


run_repository = SqlAlchemyRunRepository()

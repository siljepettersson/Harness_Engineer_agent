from copy import deepcopy
from datetime import datetime
from threading import RLock
from typing import Any

from sqlalchemy import select

from database import SessionLocal
from models import (
    AgentRun,
    AgentStep,
    AuditLog,
    Checkpoint,
    ModelCall,
    RunEvent,
    ToolCall,
    UsageRecord,
)


# Kapittel 1 bruker minne som en enkel databaseimitasjon.
runs: dict[str, dict[str, Any]] = {}
events_by_run_id: dict[str, list[dict[str, Any]]] = {}
idempotency_index: dict[str, str] = {}


class RunRepository:
    def __init__(self) -> None:
        self._lock = RLock()

    def create_if_absent(
        self,
        idempotency_scope: str,
        run: dict[str, Any],
        first_event: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Atomically create a run or return the existing idempotent run."""
        with self._lock:
            existing_id = idempotency_index.get(idempotency_scope)
            if existing_id is not None and existing_id in runs:
                return deepcopy(runs[existing_id]), False

            runs[run["id"]] = deepcopy(run)
            events_by_run_id[run["id"]] = [deepcopy(first_event)]
            idempotency_index[idempotency_scope] = run["id"]
            return deepcopy(run), True

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = runs.get(run_id)
            return deepcopy(run) if run is not None else None

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(events_by_run_id.get(run_id, []))

    def transition_and_append_event(
        self,
        run_id: str,
        status: str,
        updated_at: str,
        event: dict[str, Any],
        expected_status: str | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Change status and append an event atomically."""
        with self._lock:
            run = runs.get(run_id)
            if run is None:
                return None, False

            if expected_status is not None and run["status"] != expected_status:
                return deepcopy(run), False

            event_list = events_by_run_id.setdefault(run_id, [])
            stored_event = deepcopy(event)
            stored_event["sequence"] = len(event_list) + 1
            run["status"] = status
            run["updated_at"] = updated_at
            event_list.append(stored_event)
            return deepcopy(run), True


class SqlAlchemyRunRepository:
    """SQLAlchemy persistence for runs and their event stream.

    The service will be switched to this repository after the database-backed
    operations have been verified independently.
    """

    @staticmethod
    def _run_to_dict(run: AgentRun) -> dict[str, Any]:
        return {
            "id": run.id,
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
            )
            db_event = RunEvent(
                id=first_event["id"],
                run_id=run["id"],
                sequence=1,
                event_type=first_event["event_type"],
                payload_json=first_event["payload"],
                created_at=self._datetime(first_event["created_at"]),
            )
            db.add_all([db_run, db_event])
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

    def transition_and_append_event(
        self,
        run_id: str,
        org_id: str,
        status: str,
        updated_at: str,
        event: dict[str, Any],
        expected_status: str | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Update status and append the event in one database transaction."""
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(AgentRun).where(
                    AgentRun.id == run_id,
                    AgentRun.org_id == org_id,
                )
            )
            if run is None:
                return None, False

            if expected_status is not None and run.status != expected_status:
                return self._run_to_dict(run), False

            last_sequence = db.scalar(
                select(RunEvent.sequence)
                .where(RunEvent.run_id == run_id)
                .order_by(RunEvent.sequence.desc())
                .limit(1)
            )
            run.status = status
            run.updated_at = self._datetime(updated_at)
            db.add(
                RunEvent(
                    id=event["id"],
                    run_id=run_id,
                    sequence=(last_sequence or 0) + 1,
                    event_type=event["event_type"],
                    payload_json=event["payload"],
                    created_at=self._datetime(event["created_at"]),
                )
            )
            db.flush()
            return self._run_to_dict(run), True

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

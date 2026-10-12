from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any

from sqlalchemy import func, select, update

from celery_app import celery_app
from database import SessionLocal
from models import OutboxJob, RunEvent


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _retry_delay(attempts: int, cap_seconds: int = 300) -> int:
    return min(cap_seconds, 2 ** max(attempts - 1, 0))


def _append_event(db, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
    last_sequence = db.scalar(
        select(func.max(RunEvent.sequence)).where(RunEvent.run_id == run_id)
    )
    db.add(
        RunEvent(
            id=f"event:{run_id}:{event_type}:{(last_sequence or 0) + 1}",
            run_id=run_id,
            sequence=(last_sequence or 0) + 1,
            event_type=event_type,
            payload_json=payload,
            created_at=_now(),
        )
    )


def dispatch_once(limit: int = 100) -> int:
    """Publish due outbox jobs to the Celery broker."""
    with SessionLocal() as db:
        jobs = db.scalars(
            select(OutboxJob)
            .where(
                OutboxJob.status == "pending",
                OutboxJob.next_attempt_at <= _now(),
            )
            .order_by(OutboxJob.created_at)
            .limit(limit)
        ).all()

    dispatched = 0
    for job in jobs:
        try:
            celery_app.send_task(
                "agent_harness.execute_run",
                args=[job.payload_json["run_id"]],
                task_id=job.id,
                queue="agent_runs",
            )
        except Exception as exc:
            with SessionLocal.begin() as db:
                current = db.get(OutboxJob, job.id)
                if current is None or current.status != "pending":
                    continue
                current.attempts += 1
                current.next_attempt_at = _now() + timedelta(
                    seconds=_retry_delay(current.attempts)
                )
                current.last_error_json = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            continue

        with SessionLocal.begin() as db:
            result = db.execute(
                update(OutboxJob)
                .where(
                    OutboxJob.id == job.id,
                    OutboxJob.status == "pending",
                )
                .values(status="dispatched", dispatched_at=_now())
            )
            if result.rowcount != 1:
                continue

            run_id = job.payload_json.get("run_id")
            if run_id is not None:
                _append_event(
                    db,
                    run_id,
                    "run.enqueued",
                    {"outbox_job_id": job.id},
                )
            dispatched += 1

    return dispatched


def run_forever(poll_interval_seconds: float = 1.0) -> None:
    """Continuously drain due outbox jobs into the Celery broker."""
    while True:
        dispatch_once()
        sleep(poll_interval_seconds)


if __name__ == "__main__":
    run_forever()

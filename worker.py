from celery_app import celery_app
from repositories.run_repository import run_repository
from services.run_errors import RetryableRunError


@celery_app.task(
    bind=True,
    name="agent_harness.execute_run",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
)
def execute_run(self, run_id: str):
    """Celery entry point for the Harness Loop executor."""
    from services.run_executor import run_executor

    worker_id = self.request.hostname or self.request.id or "unknown-worker"
    try:
        return run_executor.execute(run_id=run_id, worker_id=worker_id)
    except RetryableRunError as exc:
        result = run_repository.schedule_retry(
            run_id=run_id,
            worker_id=worker_id,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
            },
            max_attempts=self.max_retries,
            backoff_seconds=2 ** self.request.retries,
        )
        if result["status"] == "failed":
            run_repository.record_dead_letter(
                original_job_id=self.request.id or run_id,
                job_type="execute_run",
                payload={"run_id": run_id},
                attempts=self.request.retries + 1,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        return result

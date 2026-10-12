from celery_app import celery_app


@celery_app.task(
    bind=True,
    name="agent_harness.execute_run",
    acks_late=True,
    reject_on_worker_lost=True,
)
def execute_run(self, run_id: str):
    """Celery entry point for the Harness Loop executor."""
    from services.run_executor import run_executor

    worker_id = self.request.hostname or self.request.id or "unknown-worker"
    return run_executor.execute(run_id=run_id, worker_id=worker_id)


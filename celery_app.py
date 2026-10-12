import os

from celery import Celery


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "agent_harness",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_default_queue="agent_runs",
    task_routes={
        "agent_harness.execute_run": {"queue": "agent_runs"},
    },
)

# Import task declarations so the Celery worker registers them at startup.
import worker as _worker  # noqa: E402,F401

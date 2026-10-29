"""Optional read-only Redis inspection for operational debugging."""

import os
from typing import Any, Protocol


class QueueClient(Protocol):
    def llen(self, name: str) -> int: ...

    def ping(self) -> bool: ...


class RedisQueueInspector:
    """Read queue telemetry without making Redis the source of truth."""

    def __init__(self, client: QueueClient, queue_name: str = "agent_runs") -> None:
        self.client = client
        self.queue_name = queue_name

    @classmethod
    def from_env(cls) -> "RedisQueueInspector":
        import redis

        return cls(
            redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            ),
            queue_name=os.getenv("CELERY_QUEUE", "agent_runs"),
        )

    def snapshot(self) -> dict[str, Any]:
        try:
            self.client.ping()
            return {
                "queue": self.queue_name,
                "reachable": True,
                "depth": int(self.client.llen(self.queue_name)),
            }
        except Exception as exc:
            return {
                "queue": self.queue_name,
                "reachable": False,
                "depth": None,
                "error": type(exc).__name__,
            }

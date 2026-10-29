"""Recovery of tool calls whose external result is uncertain.

The query function is injected because the real downstream API belongs to the
business integration, not to the AgentRun database.  It must return a result
dict, return ``None`` for a confirmed missing operation, or raise
``DownstreamUnavailable`` when the truth cannot be checked.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import select

from database import SessionLocal
from models import AgentRun, ToolCall
from repositories.run_repository import SqlAlchemyRunRepository, run_repository
from services.refund_gateway import RefundGateway, RefundGatewayUnavailable


class DownstreamUnavailable(RuntimeError):
    """The external system could not answer a status query."""


DownstreamQuery = Callable[[str, str], dict[str, Any] | None]


def query_from_refund_gateway(gateway: RefundGateway) -> DownstreamQuery:
    """Adapt a refund provider to the generic reconciliation callback."""
    def query(_tool_name: str, idempotency_key: str) -> dict[str, Any] | None:
        try:
            return gateway.get_by_idempotency_key(idempotency_key)
        except RefundGatewayUnavailable as exc:
            raise DownstreamUnavailable(str(exc)) from exc

    return query


def reconcile_refunds_once(
    gateway: RefundGateway,
    *,
    repository: SqlAlchemyRunRepository = run_repository,
    older_than_seconds: int = 300,
    limit: int = 100,
) -> dict[str, int]:
    """Run reconciliation for a concrete refund gateway."""
    return reconcile_once(
        query_from_refund_gateway(gateway),
        repository=repository,
        older_than_seconds=older_than_seconds,
        limit=limit,
    )


def reconcile_once(
    query_downstream: DownstreamQuery,
    *,
    repository: SqlAlchemyRunRepository = run_repository,
    older_than_seconds: int = 300,
    limit: int = 100,
) -> dict[str, int]:
    """Resolve response-lost and stale side-effect calls.

    No handler is invoked here.  The only external operation is a read by
    idempotency key.  A missing/unknown answer is never converted into a
    blind retry.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
    with SessionLocal() as db:
        calls = db.scalars(
            select(ToolCall)
            .join(AgentRun, AgentRun.id == ToolCall.run_id)
            .where(
                ToolCall.status.in_(["response_lost", "running"]),
                ToolCall.created_at <= cutoff,
                ToolCall.idempotency_key.is_not(None),
            )
            .order_by(ToolCall.created_at.asc())
            .limit(limit)
        ).all()
        candidates = [
            (call.id, call.tool_name, call.idempotency_key)
            for call in calls
        ]

    stats = {"scanned": 0, "repaired": 0, "manual_review": 0}
    for call_id, tool_name, key in candidates:
        stats["scanned"] += 1
        try:
            downstream = query_downstream(tool_name, key)
        except DownstreamUnavailable as exc:
            repository.record_manual_review(
                tool_call_id=call_id,
                reason=str(exc) or "downstream_unavailable",
            )
            stats["manual_review"] += 1
            continue

        if downstream is None:
            repository.reconcile_tool_call(
                tool_call_id=call_id,
                resolution={
                    "status": "failed",
                    "result_json": None,
                    "error_json": {
                        "code": "DOWNSTREAM_NOT_FOUND",
                        "message": "No operation exists for the idempotency key",
                    },
                },
            )
        else:
            repository.reconcile_tool_call(
                tool_call_id=call_id,
                resolution={
                    "status": downstream.get("status", "succeeded"),
                    "result_json": downstream,
                    "error_json": None,
                },
            )
        stats["repaired"] += 1
    return stats

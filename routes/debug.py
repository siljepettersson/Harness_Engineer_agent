from fastapi import APIRouter, Depends, Header, Query

from services.reconciliation import reconcile_once
from services.queue_inspector import RedisQueueInspector
from repositories.run_repository import run_repository
from sweeper import sweep_once


router = APIRouter(prefix="/debug", tags=["debug"])


async def _org_id(
    x_org_id: str = Header(..., alias="X-Org-Id"),
) -> str:
    return x_org_id


@router.get("/stuck-runs")
async def stuck_runs(
    org_id: str = Depends(_org_id),
    limit: int = Query(default=100, ge=1, le=1000),
):
    return run_repository.list_stuck_runs(org_id, limit=limit)


@router.get("/replay/{run_id}")
async def replay(run_id: str, org_id: str = Depends(_org_id)):
    # Deliberately calls the event reader only.  It never invokes a model,
    # tool runtime, executor, or downstream service.
    trace = run_repository.get_trace(run_id, org_id)
    if trace is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "events": trace["events"],
    }


@router.post("/sweep")
async def sweep(limit: int = Query(default=100, ge=1, le=1000)):
    return {"recovered": sweep_once(limit=limit)}


@router.post("/reconcile")
async def reconcile_unconfigured():
    # Production wires a real downstream query function.  The debug endpoint
    # intentionally refuses to guess when no downstream adapter is configured.
    from services.reconciliation import DownstreamUnavailable

    def unavailable(_tool_name: str, _key: str):
        raise DownstreamUnavailable("no downstream adapter configured")

    return reconcile_once(unavailable)


@router.get("/queue")
async def queue_snapshot():
    """Read-only Redis/Celery queue telemetry; Outbox remains authoritative."""
    return RedisQueueInspector.from_env().snapshot()

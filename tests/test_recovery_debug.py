from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import delete, select, update

from database import SessionLocal
from main import app
from models import AgentRun, AgentStep, AuditLog, Checkpoint, DeadLetterJob, ModelCall, OutboxJob, RunEvent, ToolCall, UsageRecord
from repositories.run_repository import run_repository
from services.reconciliation import DownstreamUnavailable, reconcile_once
from httpx import ASGITransport, AsyncClient
import anyio
from main import app


def _clean():
    with SessionLocal.begin() as db:
        for model in (AuditLog, UsageRecord, Checkpoint, DeadLetterJob, ToolCall, ModelCall, OutboxJob, RunEvent, AgentStep, AgentRun):
            db.execute(delete(model))


def _run():
    run_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    run_repository.create_if_absent(
        "recovery-org",
        {
            "id": run_id, "org_id": "recovery-org", "user_id": "u",
            "run_type": "support", "idempotency_key": f"ticket:{run_id}",
            "input": {"ticket_id": "T-9527", "order_id": "ORD-10086"},
            "status": "queued", "created_at": now, "updated_at": now,
        },
        {"id": str(uuid4()), "event_type": "run.created",
         "payload": {"status": "queued"}, "created_at": now},
    )
    return run_id


def test_recovery_event_and_trace_debug_fields():
    _clean()
    run_id = _run()
    assert run_repository.claim_run(run_id, "worker-a", lease_seconds=1)
    with SessionLocal.begin() as db:
        db.execute(update(AgentRun).where(AgentRun.id == run_id).values(
            lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
        ))
        db.add(Checkpoint(
            id=str(uuid4()), run_id=run_id, checkpoint_index=1,
            state_json={"messages": [], "completed_tool_calls": [], "last": "user"},
            created_at=datetime.now(timezone.utc),
        ))

    assert run_repository.recover_expired_leases() == 1
    trace = run_repository.get_trace(run_id, "recovery-org")
    assert trace["last_event"]["event_type"] == "run.requeued_by_sweeper"
    assert trace["latest_checkpoint"]["checkpoint_index"] == 1
    assert trace["outbox_jobs"]
    assert trace["audit_logs"]


def test_reconciliation_repairs_response_lost_atomically_and_never_reexecutes():
    _clean()
    run_id = _run()
    now = datetime.now(timezone.utc)
    key = "refund:ORD-10086:T-9527:v1"
    with SessionLocal.begin() as db:
        db.add(ToolCall(
            id="tc-refund", run_id=run_id, tool_name="refund_order",
            args_json={"order_id": "ORD-10086", "amount": 39.9},
            status="response_lost", idempotency_key=key,
            created_at=now - timedelta(minutes=10),
        ))

    calls = []
    result = reconcile_once(
        lambda tool, idempotency_key: calls.append((tool, idempotency_key)) or {
            "refund_id": "RF-2201", "status": "succeeded"
        },
    )
    assert result["repaired"] == 1
    assert calls == [("refund_order", key)]
    with SessionLocal() as db:
        call = db.get(ToolCall, "tc-refund")
        events = db.scalars(select(RunEvent).where(RunEvent.run_id == run_id)).all()
    assert call.status == "succeeded"
    assert call.result_json["refund_id"] == "RF-2201"
    assert any(event.event_type == "tool.repaired" for event in events)


def test_reconciliation_unavailable_writes_manual_review_without_new_status():
    _clean()
    run_id = _run()
    now = datetime.now(timezone.utc)
    with SessionLocal.begin() as db:
        db.add(ToolCall(
            id="tc-unknown", run_id=run_id, tool_name="refund_order",
            args_json={}, status="response_lost",
            idempotency_key="refund:ORD-10086:T-9527:v1",
            created_at=now - timedelta(minutes=10),
        ))

    result = reconcile_once(
        lambda _tool, _key: (_ for _ in ()).throw(
            DownstreamUnavailable("gateway unavailable")
        )
    )
    assert result["manual_review"] == 1
    assert run_repository.get(run_id, "recovery-org")["status"] == "queued"


def test_api_smoke_create_stuck_sweep_trace_and_replay():
    _clean()
    async def scenario():
        headers = {"X-Org-Id": "smoke-org", "X-User-Id": "smoke-user"}
        payload = {
            "run_type": "support_ticket",
            "idempotency_key": "smoke:recovery:v1",
            "input": {"ticket_id": "T-9527", "order_id": "ORD-10086"},
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            created = await client.post("/runs", json=payload, headers=headers)
            assert created.status_code in (200, 201)
            run_id = created.json()["run_id"]
            trace = await client.get(f"/runs/{run_id}/trace", headers=headers)
            assert trace.status_code == 200
            assert trace.json()["events"][0]["event_type"] == "run.created"
            assert run_repository.claim_run(run_id, "smoke-worker", lease_seconds=1)
            with SessionLocal.begin() as db:
                db.execute(update(AgentRun).where(AgentRun.id == run_id).values(
                    lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
                ))
            stuck = await client.get(
                "/debug/stuck-runs", headers={"X-Org-Id": "smoke-org"}
            )
            assert stuck.status_code == 200
            assert any(item["run_id"] == run_id for item in stuck.json())
            swept = await client.post("/debug/sweep")
            assert swept.status_code == 200
            replay = await client.get(
                f"/debug/replay/{run_id}", headers={"X-Org-Id": "smoke-org"}
            )
            assert replay.status_code == 200
            assert replay.json()["events"][-1]["event_type"] == (
                "run.requeued_by_sweeper"
            )
            after = await client.get(f"/runs/{run_id}/trace", headers=headers)
            assert after.json()["outbox_jobs"]
            assert after.json()["last_event"]["event_type"] == (
                "run.requeued_by_sweeper"
            )
    anyio.run(scenario)

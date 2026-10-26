import unittest
from uuid import uuid4

import anyio
from httpx import ASGITransport, AsyncClient
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, update

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
from main import app
from repositories.run_repository import run_repository


class SyncASGIClient:
    """Synchronous test facade without Starlette's broken AnyIO portal."""

    def __init__(self, app) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs):
        async def send_request():
            async with AsyncClient(
                transport=ASGITransport(app=self.app),
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return anyio.run(send_request)

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        return None


class RunApiTest(unittest.TestCase):
    def setUp(self) -> None:
        with SessionLocal.begin() as db:
            for model in (
                AuditLog,
                UsageRecord,
                Checkpoint,
                DeadLetterJob,
                ToolCall,
                ModelCall,
                OutboxJob,
                RunEvent,
                AgentStep,
                AgentRun,
            ):
                db.execute(delete(model))
        self.client = SyncASGIClient(app)
        self.headers = {
            "X-Org-Id": "org_1",
            "X-User-Id": "user_1",
        }
        self.payload = {
            "run_type": "support_ticket",
            "idempotency_key": "ticket:T-9527:handle:v1",
            "input": {
                "ticket_id": "T-9527",
                "order_id": "ORD-10086",
                "message": "Jeg ønsker refusjon.",
            },
        }

    def tearDown(self) -> None:
        self.client.close()

    def test_create_idempotency_and_events(self) -> None:
        first = self.client.post("/runs", json=self.payload, headers=self.headers)
        second = self.client.post("/runs", json=self.payload, headers=self.headers)

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["run_id"], second.json()["run_id"])
        run_id = first.json()["run_id"]
        events = self.client.get(
            f"/runs/{run_id}/events",
            headers=self.headers,
        )
        self.assertEqual(events.status_code, 200)
        self.assertEqual(events.json()[0]["event_type"], "run.created")
        self.assertEqual(events.json()[0]["sequence"], 1)

        with SessionLocal() as db:
            jobs = db.query(OutboxJob).filter_by(
                job_type="execute_run",
                status="pending",
            ).all()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].payload_json, {"run_id": run_id})
            self.assertEqual(jobs[0].attempts, 0)

    def test_idempotency_conflict(self) -> None:
        self.client.post("/runs", json=self.payload, headers=self.headers)
        changed_payload = {
            **self.payload,
            "input": {"order_id": "ORD-99999"},
        }

        response = self.client.post(
            "/runs",
            json=changed_payload,
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["error"]["code"],
            "IDEMPOTENCY_CONFLICT",
        )

    def test_other_org_cannot_access_run(self) -> None:
        created = self.client.post("/runs", json=self.payload, headers=self.headers)
        run_id = created.json()["run_id"]

        response = self.client.get(
            f"/runs/{run_id}",
            headers={"X-Org-Id": "org_2", "X-User-Id": "user_2"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RUN_NOT_FOUND")

    def test_cancel_and_terminal_cancel_error(self) -> None:
        created = self.client.post("/runs", json=self.payload, headers=self.headers)
        run_id = created.json()["run_id"]

        cancelled = self.client.post(
            f"/runs/{run_id}/cancel",
            json={"reason": "user_clicked_cancel"},
            headers=self.headers,
        )
        repeated = self.client.post(
            f"/runs/{run_id}/cancel",
            json={"reason": "retry"},
            headers=self.headers,
        )

        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertEqual(repeated.status_code, 409)
        self.assertEqual(
            repeated.json()["error"]["code"],
            "RUN_ALREADY_TERMINAL",
        )

    def test_resume_requires_waiting_for_user(self) -> None:
        created = self.client.post("/runs", json=self.payload, headers=self.headers)
        run_id = created.json()["run_id"]

        running_resume = self.client.post(
            f"/runs/{run_id}/resume",
            json={"resume_token": "approval_123"},
            headers=self.headers,
        )
        self.assertEqual(running_resume.status_code, 409)

        step_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with SessionLocal.begin() as db:
            db.add(
                AgentStep(
                    id=step_id,
                    run_id=run_id,
                    step_index=1,
                    step_type="agent_loop",
                    name="approval",
                    status="pending",
                    input_json={},
                    started_at=now,
                )
            )
        self.assertIsNotNone(run_repository.claim_run(run_id, "worker-resume"))
        run_repository.wait_for_user(
            run_id=run_id,
            org_id="org_1",
            worker_id="worker-resume",
            step_id=step_id,
            checkpoint_state={"waiting": True},
            reason="approval_required",
        )
        waiting_resume = self.client.post(
            f"/runs/{run_id}/resume",
            json={
                "resume_token": "approval_123",
                "input": {"approved": True},
            },
            headers=self.headers,
        )
        self.assertEqual(waiting_resume.status_code, 200)
        self.assertEqual(waiting_resume.json()["status"], "queued")

    def test_run_claim_is_exclusive_and_only_owner_can_finish(self) -> None:
        created = self.client.post("/runs", json=self.payload, headers=self.headers)
        run_id = created.json()["run_id"]

        first_claim = run_repository.claim_run(run_id, "worker-a")
        second_claim = run_repository.claim_run(run_id, "worker-b")

        self.assertIsNotNone(first_claim)
        self.assertEqual(first_claim["status"], "running")
        self.assertEqual(first_claim["lease_owner"], "worker-a")
        self.assertIsNone(second_claim)

        self.assertFalse(
            run_repository.finish_run(
                run_id,
                "worker-b",
                "succeeded",
                output={"answer": "not authorized"},
            )
        )
        self.assertTrue(
            run_repository.finish_run(
                run_id,
                "worker-a",
                "succeeded",
                output={"answer": "completed"},
            )
        )

        stored = run_repository.get(run_id, "org_1")
        self.assertEqual(stored["status"], "succeeded")
        self.assertIsNone(stored["lease_owner"])
        self.assertEqual(stored["attempts"], 1)

    def test_expired_lease_is_requeued_with_a_new_outbox_job(self) -> None:
        created = self.client.post("/runs", json=self.payload, headers=self.headers)
        run_id = created.json()["run_id"]
        claimed = run_repository.claim_run(run_id, "worker-a", lease_seconds=1)
        self.assertIsNotNone(claimed)

        with SessionLocal.begin() as db:
            db.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(
                    lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
                )
            )

        self.assertEqual(run_repository.recover_expired_leases(), 1)
        stored = run_repository.get(run_id, "org_1")
        self.assertEqual(stored["status"], "queued")
        self.assertIsNone(stored["lease_owner"])

        with SessionLocal() as db:
            jobs = db.query(OutboxJob).filter_by(
                job_type="execute_run",
                status="pending",
            ).all()
            self.assertEqual(len(jobs), 2)
            self.assertTrue(any(job.payload_json == {"run_id": run_id} for job in jobs))

    def test_dead_letter_is_recorded_only_once(self) -> None:
        error = {"type": "ProviderUnavailable", "message": "temporary outage"}
        first = run_repository.record_dead_letter(
            original_job_id="job-123",
            job_type="execute_run",
            payload={"run_id": "run-123"},
            attempts=3,
            error=error,
        )
        second = run_repository.record_dead_letter(
            original_job_id="job-123",
            job_type="execute_run",
            payload={"run_id": "run-123"},
            attempts=4,
            error=error,
        )

        self.assertTrue(first)
        self.assertFalse(second)
        with SessionLocal() as db:
            dead_letters = db.query(DeadLetterJob).all()
            self.assertEqual(len(dead_letters), 1)
            self.assertEqual(dead_letters[0].attempts, 3)

    def test_rerun_creates_new_run_only_after_terminal_state(self) -> None:
        created = self.client.post("/runs", json=self.payload, headers=self.headers)
        run_id = created.json()["run_id"]

        rejected = self.client.post(
            f"/runs/{run_id}/rerun",
            json={
                "idempotency_key": "ticket:T-9527:rerun:v1",
                "reason": "not_ready",
            },
            headers=self.headers,
        )
        self.assertEqual(rejected.status_code, 409)

        self.assertIsNotNone(run_repository.claim_run(run_id, "worker-rerun-api"))
        self.assertTrue(
            run_repository.finish_run(
                run_id,
                "worker-rerun-api",
                "succeeded",
                output={"answer": "original"},
            )
        )

        rerun = self.client.post(
            f"/runs/{run_id}/rerun",
            json={
                "idempotency_key": "ticket:T-9527:rerun:v1",
                "reason": "new_strategy",
            },
            headers=self.headers,
        )
        self.assertEqual(rerun.status_code, 201)
        self.assertNotEqual(rerun.json()["run_id"], run_id)
        self.assertEqual(rerun.json()["parent_run_id"], run_id)
        self.assertEqual(rerun.json()["status"], "queued")


if __name__ == "__main__":
    unittest.main()

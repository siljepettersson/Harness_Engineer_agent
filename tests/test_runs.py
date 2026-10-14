import unittest

from fastapi.testclient import TestClient
from sqlalchemy import delete, update

from database import SessionLocal
from models import (
    AgentRun,
    AgentStep,
    AuditLog,
    Checkpoint,
    ModelCall,
    OutboxJob,
    RunEvent,
    ToolCall,
    UsageRecord,
)
from main import app
from repositories.run_repository import run_repository


class RunApiTest(unittest.TestCase):
    def setUp(self) -> None:
        with SessionLocal.begin() as db:
            for model in (
                AuditLog,
                UsageRecord,
                Checkpoint,
                ToolCall,
                ModelCall,
                OutboxJob,
                RunEvent,
                AgentStep,
                AgentRun,
            ):
                db.execute(delete(model))
        self.client = TestClient(app)
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

        with SessionLocal.begin() as db:
            db.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(status="waiting_for_user")
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


if __name__ == "__main__":
    unittest.main()

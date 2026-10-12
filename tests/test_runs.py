import unittest

from fastapi.testclient import TestClient
from sqlalchemy import delete, update

from database import SessionLocal
from models import AgentRun, AgentStep, AuditLog, Checkpoint, ModelCall, RunEvent, ToolCall, UsageRecord
from main import app


class RunApiTest(unittest.TestCase):
    def setUp(self) -> None:
        with SessionLocal.begin() as db:
            for model in (
                AuditLog,
                UsageRecord,
                Checkpoint,
                ToolCall,
                ModelCall,
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


if __name__ == "__main__":
    unittest.main()

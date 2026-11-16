from datetime import datetime, timezone
from uuid import uuid4

from database import init_db
from repositories.run_repository import run_repository
from services.fake_model import ScriptedModelProvider
from services.run_executor import ModelDecision, RunExecutor
from services.tool_runtime import ToolDefinition, ToolRegistry


def test_real_repository_executes_four_model_rounds_and_three_tools():
    init_db()
    run_id = str(uuid4())
    org_id = f"integration-org-{run_id}"
    now = datetime.now(timezone.utc).isoformat()

    run_repository.create_if_absent(
        org_id=org_id,
        run={
            "id": run_id,
            "org_id": org_id,
            "user_id": "integration-user",
            "run_type": "support_ticket",
            "idempotency_key": f"integration:{run_id}",
            "input": {
                "ticket_id": "T-9527",
                "order_id": "ORD-10086",
                "message": "Jeg ønsker refusjon.",
            },
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        },
        first_event={
            "id": str(uuid4()),
            "event_type": "run.created",
            "payload": {"status": "queued"},
            "created_at": now,
        },
    )

    model = ScriptedModelProvider(
        [
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={
                    "text": "Look up order",
                    "tool_use": {
                        "id": "tc_get_order_1",
                        "name": "get_order",
                        "input": {"order_id": "ORD-10086"},
                    },
                },
                tool_name="get_order",
                tool_arguments={"order_id": "ORD-10086"},
                tool_use_id="tc_get_order_1",
            ),
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={
                    "tool_use": {
                        "id": "tc_get_shipment_1",
                        "name": "get_shipment",
                        "input": {"order_id": "ORD-10086"},
                    }
                },
                tool_name="get_shipment",
                tool_arguments={"order_id": "ORD-10086"},
                tool_use_id="tc_get_shipment_1",
            ),
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={
                    "tool_use": {
                        "id": "tc_refund_1",
                        "name": "refund_order",
                        "input": {
                            "order_id": "ORD-10086",
                            "amount": 39.9,
                        },
                    }
                },
                tool_name="refund_order",
                tool_arguments={
                    "order_id": "ORD-10086",
                    "amount": 39.9,
                },
                tool_use_id="tc_refund_1",
            ),
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={"text": "Refusjonen er gjennomført."},
                final_answer={
                    "answer": "Refusjonen er gjennomført.",
                },
            ),
        ]
    )

    tools = ToolRegistry(
        [
            ToolDefinition(
                name="get_order",
                handler=lambda arguments, key: {
                    "amount": 39.9,
                    "payment_status": "paid",
                },
                idempotency_key=lambda run, arguments:
                    f"get_order:{run['id']}:v1",
            ),
            ToolDefinition(
                name="get_shipment",
                handler=lambda arguments, key: {
                    "status": "delayed",
                    "days": 5,
                },
                idempotency_key=lambda run, arguments:
                    f"get_shipment:{run['id']}:v1",
            ),
            ToolDefinition(
                name="refund_order",
                handler=lambda arguments, key: {
                    "refund_id": "RF-2201",
                    "idempotency_key": key,
                },
                idempotency_key=lambda run, arguments:
                    f"refund:{arguments['order_id']}:{run['input']['ticket_id']}:v1",
            ),
        ]
    )

    result = RunExecutor(
        run_repository,
        model_provider=model,
        tool_registry=tools,
    ).execute(run_id, "integration-worker")

    trace = run_repository.get_trace(run_id, org_id)

    assert result == {"run_id": run_id, "status": "succeeded"}
    assert trace is not None
    assert trace["run"]["status"] == "succeeded"
    assert len(trace["model_calls"]) == 4
    assert len(trace["tool_calls"]) == 3
    assert len(trace["checkpoints"]) == 7
    assert trace["tool_calls"][2]["idempotency_key"] == (
        "refund:ORD-10086:T-9527:v1"
    )
    assert trace["events"][-1]["event_type"] == "run.completed"

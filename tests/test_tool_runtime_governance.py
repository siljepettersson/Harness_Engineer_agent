import pytest

from services.tool_runtime import (
    NonRetryableToolError,
    ToolContext,
    ToolDefinition,
    ToolRuntime,
)


class GovernanceStore:
    def __init__(self):
        self.calls = []
        self.audits = []

    def get_tool_call_by_idempotency_key(
        self,
        *,
        org_id,
        tool_name,
        idempotency_key,
    ):
        return next(
            (
                call
                for call in self.calls
                if call["org_id"] == org_id
                and call["tool_name"] == tool_name
                and call["idempotency_key"] == idempotency_key
            ),
            None,
        )

    def record_tool_call(self, org_id, call):
        self.calls.append(call)
        return call

    def update_tool_call(self, org_id, tool_call_id, updates):
        call = next(call for call in self.calls if call["id"] == tool_call_id)
        call.update(updates)
        return call

    def record_audit_log(self, org_id, log):
        self.audits.append((org_id, log))
        return log


def _context():
    return ToolContext(
        org_id="org-1",
        user_id="user-1",
        run_id="run-1",
        step_id="step-1",
        permissions={"orders:refund"},
    )


def test_high_risk_tools_force_approval_and_audit_and_build_preview():
    calls = []
    tool = ToolDefinition(
        name="refund_order",
        handler=lambda arguments, key: calls.append(True) or {"status": "ok"},
        side_effect=True,
        idempotency_required=True,
        permission="orders:refund",
        risk_level="high",
        approval_policy={"approvers": 2},
        preview_builder=lambda arguments, context: {
            "order_id": arguments["order_id"],
            "amount": 39.9,
            "actor": context.user_id,
        },
    )

    assert tool.requires_approval is True
    assert tool.audit_required is True

    preview = ToolRuntime().build_preview(
        tool=tool,
        arguments={"order_id": "ORD-10086"},
        context=_context(),
    )

    assert preview == {
        "order_id": "ORD-10086",
        "amount": 39.9,
        "actor": "user-1",
    }
    assert calls == []


def test_audit_records_success_and_duplicate_without_reexecuting():
    store = GovernanceStore()
    executions = []
    tool = ToolDefinition(
        name="refund_order",
        handler=lambda arguments, key: executions.append(key)
        or {"refund_id": "RF-2201"},
        side_effect=True,
        idempotency_required=True,
        audit_required=True,
        permission="orders:refund",
    )
    runtime = ToolRuntime(tool_call_store=store)

    first = runtime.execute(
        tool=tool,
        arguments={"order_id": "ORD-10086"},
        idempotency_key="refund:ORD-10086:T-9527:v1",
        context=_context(),
    )
    second = runtime.execute(
        tool=tool,
        arguments={"order_id": "ORD-10086"},
        idempotency_key="refund:ORD-10086:T-9527:v1",
        context=_context(),
    )

    assert first == second == {"refund_id": "RF-2201"}
    assert executions == ["refund:ORD-10086:T-9527:v1"]
    assert [log[1]["action"] for log in store.audits] == [
        "tool.succeeded",
        "tool.skipped_duplicate",
    ]
    assert store.audits[0][1]["actor"] == "user-1"
    assert store.audits[0][1]["metadata_json"]["tool_call_id"]


def test_audited_tool_requires_an_audit_store_before_handler_execution():
    calls = []
    tool = ToolDefinition(
        name="send_customer_message",
        handler=lambda arguments, key: calls.append(True) or {"sent": True},
        audit_required=True,
    )

    with pytest.raises(NonRetryableToolError) as caught:
        ToolRuntime().execute(
            tool=tool,
            arguments={"message": "hello"},
            context=_context(),
        )

    assert caught.value.tool_error["code"] == "TOOL_AUDIT_STORE_UNAVAILABLE"
    assert calls == []

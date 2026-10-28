from services.run_errors import PermanentRunError
from services.tool_runtime import ToolContext, ToolDefinition, ToolRuntime


class InMemoryToolCallStore:
    def __init__(self):
        self.calls = []

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


def _context():
    return ToolContext(
        org_id="org-1",
        user_id="user-1",
        run_id="run-1",
        step_id="step-1",
        permissions={"orders:refund"},
    )


def test_runtime_records_running_then_succeeded_and_deduplicates():
    store = InMemoryToolCallStore()
    executions = []
    tool = ToolDefinition(
        name="refund_order",
        handler=lambda arguments, key: executions.append(key)
        or {"refund_id": "RF-2201"},
        permission="orders:refund",
        side_effect=True,
        idempotency_required=True,
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
    assert len(store.calls) == 1
    assert store.calls[0]["status"] == "succeeded"
    assert store.calls[0]["result_json"] == {"refund_id": "RF-2201"}


def test_runtime_requires_key_for_side_effect_tool():
    store = InMemoryToolCallStore()
    tool = ToolDefinition(
        name="refund_order",
        handler=lambda arguments, key: {"status": "ok"},
        side_effect=True,
        idempotency_required=True,
    )

    try:
        ToolRuntime(tool_call_store=store).execute(
            tool=tool,
            arguments={"order_id": "ORD-10086"},
            context=_context(),
        )
    except PermanentRunError as exc:
        assert "Idempotency key required" in str(exc)
    else:
        raise AssertionError("missing idempotency key was accepted")


def test_runtime_updates_same_call_to_failed_when_handler_errors():
    store = InMemoryToolCallStore()

    def fail(arguments, key):
        raise ValueError("payment rejected")

    tool = ToolDefinition(
        name="refund_order",
        handler=fail,
        side_effect=True,
        idempotency_required=True,
    )

    try:
        ToolRuntime(tool_call_store=store).execute(
            tool=tool,
            arguments={"order_id": "ORD-10086"},
            idempotency_key="refund:ORD-10086:T-9527:v1",
            context=_context(),
        )
    except ValueError as exc:
        assert str(exc) == "payment rejected"
    else:
        raise AssertionError("handler error was swallowed")

    assert len(store.calls) == 1
    assert store.calls[0]["status"] == "failed"
    assert store.calls[0]["error_json"]["message"] == "payment rejected"

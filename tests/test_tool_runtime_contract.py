import pytest

from services.tool_runtime import ToolContext, ToolDefinition, ToolRegistry


def test_tool_context_keeps_backend_owned_call_identity():
    context = ToolContext(
        org_id="org-1",
        user_id="user-1",
        run_id="run-1",
        step_id="step-1",
        permissions={"orders:read"},
    )

    assert context.org_id == "org-1"
    assert context.user_id == "user-1"
    assert context.run_id == "run-1"
    assert context.step_id == "step-1"
    assert context.permissions == frozenset({"orders:read"})


def test_tool_definition_rejects_side_effect_without_idempotency():
    with pytest.raises(ValueError, match="idempotency key"):
        ToolDefinition(
            name="refund_order",
            handler=lambda arguments, key: {"status": "ok"},
            side_effect=True,
        )


def test_tool_definition_rejects_high_risk_without_approval_policy():
    with pytest.raises(ValueError, match="approval policy"):
        ToolDefinition(
            name="refund_order",
            handler=lambda arguments, key: {"status": "ok"},
            risk_level="high",
        )


def test_registry_exposes_registered_definition_snapshot():
    tool = ToolDefinition(
        name="get_order",
        description="Read an order",
        handler=lambda arguments, key: {"status": "ok"},
        permission="orders:read",
        timeout_seconds=5,
        retryable_errors={"TOOL_TIMEOUT"},
    )
    registry = ToolRegistry([tool])

    assert registry.get("get_order") is tool
    assert registry.all() == (tool,)
    assert tool.retryable_errors == frozenset({"TOOL_TIMEOUT"})

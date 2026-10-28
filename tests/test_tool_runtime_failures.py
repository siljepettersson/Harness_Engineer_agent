import pytest

from services.tool_runtime import (
    NonRetryableToolError,
    ResponseLostToolError,
    RetryableToolError,
    ToolContext,
    ToolDefinition,
    ToolRuntime,
)


class Store:
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


def context():
    return ToolContext(
        org_id="org-1",
        user_id="user-1",
        run_id="run-1",
        step_id="step-1",
    )


def test_read_tool_retries_transient_timeout_with_a_limit():
    store = Store()
    attempts = []

    def handler(arguments, key):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise TimeoutError("temporary timeout")
        return {"status": "ok"}

    tool = ToolDefinition(
        name="get_order",
        handler=handler,
        retryable_errors={"TOOL_TIMEOUT"},
        max_attempts=2,
    )

    result = ToolRuntime(tool_call_store=store).execute(
        tool=tool,
        arguments={"order_id": "ORD-10086"},
        idempotency_key="get_order:run-1:1",
        context=context(),
    )

    assert result == {"status": "ok"}
    assert attempts == [1, 2]
    assert store.calls[0]["status"] == "succeeded"


def test_side_effect_timeout_becomes_response_lost_without_safe_retry():
    store = Store()
    attempts = []

    def handler(arguments, key):
        attempts.append(True)
        raise TimeoutError("payment response missing")

    tool = ToolDefinition(
        name="refund_order",
        handler=handler,
        side_effect=True,
        idempotency_required=True,
        retryable_errors={"TOOL_TIMEOUT"},
        max_attempts=3,
    )

    with pytest.raises(ResponseLostToolError) as caught:
        ToolRuntime(tool_call_store=store).execute(
            tool=tool,
            arguments={"order_id": "ORD-10086"},
            idempotency_key="refund:ORD-10086:T-9527:v1",
            context=context(),
        )

    assert caught.value.tool_error == {
        "code": "TOOL_RESPONSE_LOST",
        "message": "Response lost for refund_order",
        "retryable": False,
    }
    assert attempts == [True]
    assert store.calls[0]["status"] == "response_lost"


def test_side_effect_can_retry_when_downstream_deduplicates():
    store = Store()
    attempts = []

    def handler(arguments, key):
        attempts.append(True)
        if len(attempts) == 1:
            raise TimeoutError("temporary timeout")
        return {"refund_id": "RF-2201"}

    tool = ToolDefinition(
        name="refund_order",
        handler=handler,
        side_effect=True,
        idempotency_required=True,
        downstream_deduplicates=True,
        retryable_errors={"TOOL_TIMEOUT"},
        max_attempts=2,
    )

    result = ToolRuntime(tool_call_store=store).execute(
        tool=tool,
        arguments={"order_id": "ORD-10086"},
        idempotency_key="refund:ORD-10086:T-9527:v1",
        context=context(),
    )

    assert result == {"refund_id": "RF-2201"}
    assert len(attempts) == 2
    assert store.calls[0]["status"] == "succeeded"


def test_http_429_is_rate_limited_and_can_retry():
    class RateLimitedError(Exception):
        status_code = 429

    store = Store()
    attempts = []

    def handler(arguments, key):
        attempts.append(True)
        if len(attempts) == 1:
            raise RateLimitedError("too many requests")
        return {"status": "ok"}

    tool = ToolDefinition(
        name="get_order",
        handler=handler,
        retryable_errors={"TOOL_RATE_LIMITED"},
        max_attempts=2,
    )

    result = ToolRuntime(tool_call_store=store).execute(
        tool=tool,
        arguments={"order_id": "ORD-10086"},
        idempotency_key="get_order:run-1:2",
        context=context(),
    )

    assert result == {"status": "ok"}
    assert len(attempts) == 2


def test_unknown_handler_failure_is_non_retryable_and_structured():
    store = Store()

    def handler(arguments, key):
        raise ValueError("order is not refundable")

    tool = ToolDefinition(name="refund_order", handler=handler)

    with pytest.raises(NonRetryableToolError) as caught:
        ToolRuntime(tool_call_store=store).execute(
            tool=tool,
            arguments={"order_id": "ORD-10086"},
            idempotency_key="refund:run-1:1",
            context=context(),
        )

    assert caught.value.tool_error["code"] == "TOOL_EXECUTION_FAILED"
    assert caught.value.tool_error["retryable"] is False
    assert store.calls[0]["status"] == "failed"

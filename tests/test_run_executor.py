import pytest

from services.run_executor import ModelDecision, RunExecutor
from services.run_errors import (
    ApprovalRequired,
    PermanentRunError,
    RunCancelled,
    RunExecutionError,
    RetryableRunError,
    TransientRunError,
)
from services.fake_model import ScriptedModelProvider
from services.tool_runtime import ToolDefinition, ToolRegistry, ToolRuntime


class CheckpointRepository:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint
        self.saved = []

    def get_latest_checkpoint(self, *, run_id, org_id):
        return self.checkpoint

    def next_checkpoint_index(self, *, run_id, org_id):
        return 7

    def save_checkpoint(self, org_id, checkpoint):
        self.saved.append((org_id, checkpoint))


def test_new_run_context_uses_canonical_message_list():
    context = RunExecutor._initial_context(
        {
            "input": {
                "ticket_id": "T-9527",
                "message": "Jeg ønsker refusjon.",
            }
        }
    )

    assert context == {
        "messages": [
            {
                "role": "user",
                "content": {
                    "ticket_id": "T-9527",
                    "message": "Jeg ønsker refusjon.",
                },
            }
        ],
        "completed_tool_calls": [],
        "last": "user",
    }


def test_assistant_message_preserves_structured_model_response():
    decision = ModelDecision(
        provider="fake",
        model="fake-model",
        response_json={
            "text": "Jeg undersøker ordren.",
            "tool_use": {
                "id": "tc_get_order_1",
                "name": "get_order",
                "input": {"order_id": "ORD-10086"},
            },
        },
        tool_name="get_order",
        tool_arguments={"order_id": "ORD-10086"},
        tool_use_id="tc_get_order_1",
    )

    assert RunExecutor._assistant_message(decision) == {
        "role": "assistant",
        "content": decision.response_json,
    }


def test_context_is_restored_from_latest_checkpoint():
    saved_context = {
        "messages": [
            {"role": "user", "content": "continue"},
        ],
        "completed_tool_calls": ["tc_get_order_1"],
        "last": "tool_result",
    }
    executor = RunExecutor(
        CheckpointRepository(
            {
                "state_json": saved_context,
                "checkpoint_index": 3,
            }
        )
    )

    restored = executor._load_context(
        {
            "id": "run-123",
            "org_id": "org-1",
            "input": {"message": "new run"},
        }
    )

    assert restored == saved_context


def test_context_falls_back_to_initial_input_without_checkpoint():
    executor = RunExecutor(CheckpointRepository())

    restored = executor._load_context(
        {
            "id": "run-123",
            "org_id": "org-1",
            "input": {"message": "start"},
        }
    )

    assert restored["messages"] == [
        {"role": "user", "content": {"message": "start"}}
    ]
    assert restored["completed_tool_calls"] == []
    assert restored["last"] == "user"


def test_save_context_uses_repository_checkpoint_sequence():
    repository = CheckpointRepository()
    executor = RunExecutor(repository)
    run = {
        "id": "run-123",
        "org_id": "org-1",
    }
    context = {
        "messages": [],
        "completed_tool_calls": [],
        "last": "user",
    }

    executor._save_context(run, context)

    assert len(repository.saved) == 1
    org_id, checkpoint = repository.saved[0]
    assert org_id == "org-1"
    assert checkpoint["checkpoint_index"] == 7
    assert checkpoint["state_json"] == context


def test_run_error_protocol_has_distinct_retry_and_terminal_categories():
    assert issubclass(TransientRunError, RunExecutionError)
    assert issubclass(PermanentRunError, RunExecutionError)
    assert issubclass(ApprovalRequired, RunExecutionError)
    assert issubclass(RunCancelled, RunExecutionError)
    assert issubclass(RetryableRunError, TransientRunError)

    assert not isinstance(
        PermanentRunError("invalid input"),
        TransientRunError,
    )


def test_scripted_model_provider_returns_responses_without_tool_registry_knowledge():
    provider = ScriptedModelProvider(
        [
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={"text": "done"},
                final_answer={"answer": "done"},
            )
        ]
    )
    context = {
        "messages": [
            {"role": "user", "content": {"message": "hello"}},
        ],
        "completed_tool_calls": [],
        "last": "user",
    }

    response = provider.complete(context=context)

    assert response.final_answer == {"answer": "done"}
    assert provider.contexts == [context]


def test_tool_registry_resolves_tools_and_rejects_unknown_names():
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="get_order",
                handler=lambda arguments, key: {
                    "order_id": arguments["order_id"],
                    "key": key,
                },
            )
        ]
    )

    assert registry.get("get_order").name == "get_order"

    with pytest.raises(PermanentRunError, match="Unknown tool"):
        registry.get("refund_order")


def test_tool_runtime_passes_idempotency_key_to_handler():
    received = {}

    def handler(arguments, idempotency_key):
        received.update(
            arguments=arguments,
            idempotency_key=idempotency_key,
        )
        return {"status": "ok"}

    tool = ToolDefinition(name="refund_order", handler=handler)
    result = ToolRuntime().execute(
        tool=tool,
        arguments={"order_id": "ORD-10086", "amount": 39.9},
        idempotency_key="refund:ORD-10086:T-9527:v1",
    )

    assert result == {"status": "ok"}
    assert received["idempotency_key"] == "refund:ORD-10086:T-9527:v1"


def test_tool_runtime_classifies_validation_and_network_errors():
    def validate(arguments):
        if arguments.get("amount", 0) <= 0:
            raise ValueError("amount must be positive")

    invalid_tool = ToolDefinition(
        name="refund_order",
        handler=lambda arguments, key: {"status": "ok"},
        validate_arguments=validate,
    )

    with pytest.raises(PermanentRunError, match="Invalid arguments"):
        ToolRuntime().execute(
            tool=invalid_tool,
            arguments={"amount": 0},
        )

    unavailable_tool = ToolDefinition(
        name="get_order",
        handler=lambda arguments, key: (_ for _ in ()).throw(
            ConnectionError("downstream unavailable")
        ),
    )

    with pytest.raises(TransientRunError, match="Transient failure"):
        ToolRuntime().execute(
            tool=unavailable_tool,
            arguments={},
        )

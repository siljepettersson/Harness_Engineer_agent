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
        self.steps = []
        self.model_calls = []
        self.tool_calls = []
        self.events = []
        self.finished = None
        self.existing_tool_call = None

    def get_latest_checkpoint(self, *, run_id, org_id):
        return self.checkpoint

    def next_checkpoint_index(self, *, run_id, org_id):
        return 7

    def save_checkpoint(self, org_id, checkpoint):
        self.saved.append((org_id, checkpoint))

    def create_step(self, org_id, step):
        self.steps.append(step)
        return step

    def update_step(self, org_id, step_id, status, output_json=None, error_json=None):
        return {
            "id": step_id,
            "status": status,
            "output_json": output_json,
            "error_json": error_json,
        }

    def record_model_call(self, org_id, call):
        self.model_calls.append(call)
        return call

    def record_tool_call(self, org_id, call):
        self.tool_calls.append(call)
        return call

    def get_tool_call_by_idempotency_key(
        self,
        *,
        org_id,
        tool_name,
        idempotency_key,
    ):
        return self.existing_tool_call

    def append_event(self, org_id, run_id, event_type, payload=None):
        self.events.append(event_type)
        return {
            "run_id": run_id,
            "event_type": event_type,
            "payload": payload or {},
        }

    def cancel_if_requested(self, run_id, worker_id):
        return False

    def finish_run(self, run_id, worker_id, status, output=None, error=None):
        self.finished = {
            "run_id": run_id,
            "worker_id": worker_id,
            "status": status,
            "output": output,
            "error": error,
        }
        return True

    def schedule_retry(self, run_id, worker_id, error):
        self.retry = {
            "run_id": run_id,
            "worker_id": worker_id,
            "error": error,
        }
        return {"status": "queued"}

    def wait_for_user(
        self,
        *,
        run_id,
        org_id,
        worker_id,
        step_id,
        checkpoint_state,
        reason,
    ):
        self.waiting = {
            "run_id": run_id,
            "org_id": org_id,
            "worker_id": worker_id,
            "step_id": step_id,
            "checkpoint_state": checkpoint_state,
            "reason": reason,
        }
        return {"status": "waiting_for_user"}


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


def test_run_model_step_records_model_step_call_and_events():
    repository = CheckpointRepository()
    provider = ScriptedModelProvider(
        [
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={"text": "done"},
                final_answer={"answer": "done"},
                prompt_tokens=10,
                completion_tokens=3,
                latency_ms=25,
            )
        ]
    )
    executor = RunExecutor(repository, model_provider=provider)
    run = {"id": "run-123", "org_id": "org-1"}
    context = {
        "messages": [],
        "completed_tool_calls": [],
        "last": "user",
    }

    response = executor.run_model_step(run, context, step_index=1)

    assert response.final_answer == {"answer": "done"}
    assert repository.steps[0]["step_type"] == "model"
    assert repository.steps[0]["status"] == "running"
    assert repository.model_calls[0]["status"] == "succeeded"
    assert repository.model_calls[0]["prompt_tokens"] == 10
    assert repository.events == [
        "step.started",
        "model.started",
        "model.completed",
        "step.completed",
    ]


def test_run_model_step_records_failure_before_reraising():
    repository = CheckpointRepository()

    class FailingProvider:
        def complete(self, *, context):
            raise TransientRunError("provider unavailable")

    executor = RunExecutor(repository, model_provider=FailingProvider())

    with pytest.raises(TransientRunError, match="provider unavailable"):
        executor.run_model_step(
            {"id": "run-123", "org_id": "org-1"},
            {
                "messages": [],
                "completed_tool_calls": [],
                "last": "user",
            },
            step_index=1,
        )

    assert repository.model_calls[0]["status"] == "failed"
    assert repository.model_calls[0]["error_json"]["type"] == "TransientRunError"
    assert repository.events == [
        "step.started",
        "model.started",
        "model.failed",
        "step.failed",
    ]


def test_run_tool_round_records_tool_step_call_and_events():
    repository = CheckpointRepository()
    executor = RunExecutor(repository)
    tool = ToolDefinition(
        name="refund_order",
        handler=lambda arguments, key: {
            "refund_id": "RF-2201",
            "idempotency_key": key,
        },
        idempotency_key=lambda run, arguments:
            f"refund:{arguments['order_id']}:{run['input']['ticket_id']}:v1",
    )
    decision = ModelDecision(
        provider="fake",
        model="fake-model",
        response_json={
            "tool_use": {
                "id": "tc_refund_1",
                "name": "refund_order",
            }
        },
        tool_name="refund_order",
        tool_arguments={
            "order_id": "ORD-10086",
            "amount": 39.9,
        },
        tool_use_id="tc_refund_1",
    )

    result = executor.run_tool_round(
        run={
            "id": "run-123",
            "org_id": "org-1",
            "run_type": "support_ticket",
            "input": {"ticket_id": "T-9527"},
        },
        tool=tool,
        decision=decision,
        step_index=2,
    )

    assert result["refund_id"] == "RF-2201"
    assert repository.steps[0]["step_type"] == "tool"
    assert repository.steps[0]["name"] == "refund_order"
    assert repository.tool_calls[0]["idempotency_key"] == (
        "refund:ORD-10086:T-9527:v1"
    )
    assert repository.events == [
        "step.started",
        "tool.started",
        "tool.completed",
        "step.completed",
    ]


def test_run_loop_checkpoints_model_output_and_tool_result():
    repository = CheckpointRepository()
    provider = ScriptedModelProvider(
        [
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={"tool_use": {"name": "get_order"}},
                tool_name="get_order",
                tool_arguments={"order_id": "ORD-10086"},
                tool_use_id="tc_get_order_1",
            ),
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={"text": "done"},
                final_answer={"answer": "done"},
            ),
        ]
    )
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="get_order",
                handler=lambda arguments, key: {"amount": 39.9},
            )
        ]
    )
    executor = RunExecutor(
        repository,
        model_provider=provider,
        tool_registry=registry,
    )

    result = executor.run_loop(
        run={
            "id": "run-123",
            "org_id": "org-1",
            "run_type": "support_ticket",
            "input": {"ticket_id": "T-9527"},
        },
        context=RunExecutor._initial_context(
            {"input": {"ticket_id": "T-9527"}}
        ),
        worker_id="worker-1",
    )

    assert result == {"run_id": "run-123", "status": "succeeded"}
    assert len(repository.saved) == 3
    assert repository.saved[0][1]["state_json"]["last"] == "assistant"
    assert repository.saved[1][1]["state_json"]["last"] == "tool_result"
    assert repository.saved[2][1]["state_json"]["last"] == "assistant"
    assert repository.finished["status"] == "succeeded"


class ExecuteRepository(CheckpointRepository):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def claim_run(self, run_id, worker_id):
        return self.run


def test_execute_handles_transient_errors_inside_execute_run():
    repository = ExecuteRepository(
        {
            "id": "run-123",
            "org_id": "org-1",
            "run_type": "support_ticket",
            "input": {"ticket_id": "T-9527"},
        }
    )

    class FailingProvider:
        def complete(self, *, context):
            raise TransientRunError("provider_503")

    executor = RunExecutor(
        repository,
        model_provider=FailingProvider(),
        tool_registry=ToolRegistry(),
    )

    result = executor.execute("run-123", "worker-1")

    assert result == {"run_id": "run-123", "status": "queued"}
    assert repository.retry["error"]["type"] == "TransientRunError"
    assert repository.finished is None


def test_execute_terminalizes_permanent_errors_inside_execute_run():
    repository = ExecuteRepository(
        {
            "id": "run-123",
            "org_id": "org-1",
            "run_type": "support_ticket",
            "input": {"ticket_id": "T-9527"},
        }
    )

    class FailingProvider:
        def complete(self, *, context):
            raise PermanentRunError("invalid_request")

    executor = RunExecutor(
        repository,
        model_provider=FailingProvider(),
        tool_registry=ToolRegistry(),
    )

    result = executor.execute("run-123", "worker-1")

    assert result == {"run_id": "run-123", "status": "failed"}
    assert repository.finished["status"] == "failed"
    assert repository.finished["error"]["type"] == "PermanentRunError"


def test_run_loop_releases_worker_when_tool_approval_is_required():
    repository = CheckpointRepository()
    provider = ScriptedModelProvider(
        [
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={"tool_use": {"name": "refund_order"}},
                tool_name="refund_order",
                tool_arguments={"order_id": "ORD-10086", "amount": 39.9},
                tool_use_id="tc_refund_1",
            )
        ]
    )
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="refund_order",
                handler=lambda arguments, key: {"refund_id": "RF-2201"},
                requires_approval=True,
            )
        ]
    )
    executor = RunExecutor(
        repository,
        model_provider=provider,
        tool_registry=registry,
    )

    result = executor.run_loop(
        run={
            "id": "run-123",
            "org_id": "org-1",
            "run_type": "support_ticket",
            "input": {"ticket_id": "T-9527"},
        },
        context=RunExecutor._initial_context(
            {"input": {"ticket_id": "T-9527"}}
        ),
        worker_id="worker-1",
    )

    assert result == {
        "run_id": "run-123",
        "status": "waiting_for_user",
    }
    assert repository.waiting["reason"] == "tool_approval_required"
    assert repository.waiting["checkpoint_state"]["last"] == "assistant"
    assert not repository.tool_calls


def test_run_loop_reuses_succeeded_tool_call_after_recovery():
    repository = CheckpointRepository()
    repository.existing_tool_call = {
        "status": "succeeded",
        "result_json": {"amount": 39.9},
    }
    provider = ScriptedModelProvider(
        [
            ModelDecision(
                provider="fake",
                model="fake-model",
                response_json={"text": "refund can proceed"},
                final_answer={"answer": "refund can proceed"},
            )
        ]
    )
    calls = []
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="get_order",
                handler=lambda arguments, key: calls.append(arguments),
            )
        ]
    )
    executor = RunExecutor(
        repository,
        model_provider=provider,
        tool_registry=registry,
    )
    context = {
        "messages": [
            {
                "role": "user",
                "content": {"ticket_id": "T-9527"},
            },
            {
                "role": "assistant",
                "content": {
                    "text": "Look up order",
                    "tool_use": {
                        "id": "tc_get_order_1",
                        "name": "get_order",
                        "input": {"order_id": "ORD-10086"},
                    },
                },
            },
        ],
        "completed_tool_calls": [],
        "last": "assistant",
    }

    result = executor.run_loop(
        run={
            "id": "run-123",
            "org_id": "org-1",
            "run_type": "support_ticket",
            "input": {"ticket_id": "T-9527"},
        },
        context=context,
        worker_id="worker-recovery",
    )

    assert result == {"run_id": "run-123", "status": "succeeded"}
    assert calls == []
    assert provider.contexts[0]["last"] == "tool_result"
    assert provider.contexts[0]["messages"][-1]["content"][0][
        "tool_use_id"
    ] == "tc_get_order_1"

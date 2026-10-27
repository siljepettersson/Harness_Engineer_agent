from services.run_executor import ModelDecision, RunExecutor


class CheckpointRepository:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

    def get_latest_checkpoint(self, *, run_id, org_id):
        return self.checkpoint


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

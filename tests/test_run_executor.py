from services.run_executor import ModelDecision, RunExecutor


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

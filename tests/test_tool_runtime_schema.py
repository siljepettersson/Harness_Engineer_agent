import pytest
from pydantic import BaseModel, ConfigDict, Field

from services.run_errors import PermanentRunError
from services.tool_runtime import ToolDefinition, ToolRuntime


class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=3)


class GetOrderOutput(BaseModel):
    order_id: str
    amount: float


def test_input_schema_runs_before_handler_and_normalizes_arguments():
    received = {}
    tool = ToolDefinition(
        name="get_order",
        handler=lambda arguments, key: received.update(arguments=arguments)
        or {"order_id": arguments["order_id"], "amount": 39.9},
        input_schema=GetOrderInput,
        output_schema=GetOrderOutput,
    )

    result = ToolRuntime().execute(
        tool=tool,
        arguments={"order_id": "ORD-10086"},
    )

    assert received["arguments"] == {"order_id": "ORD-10086"}
    assert result == {"order_id": "ORD-10086", "amount": 39.9}


def test_input_schema_rejects_unknown_or_invalid_model_arguments():
    tool = ToolDefinition(
        name="get_order",
        handler=lambda arguments, key: {"status": "ok"},
        input_schema=GetOrderInput,
    )

    with pytest.raises(PermanentRunError, match="Invalid arguments"):
        ToolRuntime().execute(
            tool=tool,
            arguments={"order_id": "", "unexpected": True},
        )


def test_output_schema_rejects_handler_data_before_returning_to_loop():
    tool = ToolDefinition(
        name="get_order",
        handler=lambda arguments, key: {
            "order_id": arguments["order_id"],
            "amount": "not-a-number",
        },
        input_schema=GetOrderInput,
        output_schema=GetOrderOutput,
    )

    with pytest.raises(PermanentRunError, match="Invalid result"):
        ToolRuntime().execute(
            tool=tool,
            arguments={"order_id": "ORD-10086"},
        )

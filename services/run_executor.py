from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from repositories.run_repository import SqlAlchemyRunRepository, run_repository


class ModelProvider(Protocol):
    def complete(
        self,
        *,
        context: dict[str, Any],
        tools: list[str],
    ) -> "ModelDecision": ...


class ToolRegistry(Protocol):
    def call(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ModelDecision:
    provider: str
    model: str
    response_json: dict[str, Any]
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    final_answer: dict[str, Any] | None = None


class ExecutorConfigurationError(RuntimeError):
    pass


class RunExecutor:
    """Execute a run through an injected model provider and tool registry."""

    def __init__(
        self,
        repository: SqlAlchemyRunRepository,
        model_provider: ModelProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 20,
    ) -> None:
        self.repository = repository
        self.model_provider = model_provider
        self.tool_registry = tool_registry
        self.max_steps = max_steps

    def execute(self, run_id: str, worker_id: str) -> dict[str, Any]:
        if self.model_provider is None or self.tool_registry is None:
            raise ExecutorConfigurationError(
                "Model provider and tool registry must be configured"
            )

        run = self.repository.claim_run(run_id, worker_id)
        if run is None:
            return {"run_id": run_id, "status": "skipped"}

        context: dict[str, Any] = {"input": run["input"], "tool_results": []}
        tools: list[str] = []

        try:
            for step_index in range(1, self.max_steps + 1):
                started_at = self._now()
                decision = self.model_provider.complete(
                    context=context,
                    tools=tools,
                )
                step_id = str(uuid4())

                self.repository.create_step(
                    run["org_id"],
                    {
                        "id": step_id,
                        "run_id": run_id,
                        "step_index": step_index,
                        "step_type": "agent_loop",
                        "name": decision.tool_name or "final_answer",
                        "status": "completed",
                        "input_json": context,
                        "output_json": decision.response_json,
                        "started_at": started_at,
                        "ended_at": self._now(),
                    },
                )
                self.repository.record_model_call(
                    run["org_id"],
                    {
                        "id": str(uuid4()),
                        "run_id": run_id,
                        "step_id": step_id,
                        "provider": decision.provider,
                        "model": decision.model,
                        "request_json": context,
                        "response_json": decision.response_json,
                        "status": "succeeded",
                        "created_at": started_at,
                    },
                )

                if decision.tool_name is None:
                    output = decision.final_answer or decision.response_json
                    if not self.repository.finish_run(
                        run_id,
                        worker_id,
                        "succeeded",
                        output=output,
                    ):
                        return {"run_id": run_id, "status": "lease_lost"}
                    return {"run_id": run_id, "status": "succeeded"}

                if decision.tool_arguments is None:
                    raise ValueError("Tool decision is missing arguments")

                idempotency_key = self._tool_idempotency_key(
                    run,
                    decision.tool_name,
                    decision.tool_arguments,
                )
                result = self.tool_registry.call(
                    name=decision.tool_name,
                    arguments=decision.tool_arguments,
                    idempotency_key=idempotency_key,
                )
                self.repository.record_tool_call(
                    run["org_id"],
                    {
                        "id": str(uuid4()),
                        "run_id": run_id,
                        "step_id": step_id,
                        "tool_name": decision.tool_name,
                        "args_json": decision.tool_arguments,
                        "result_json": result,
                        "status": "succeeded",
                        "idempotency_key": idempotency_key,
                        "created_at": started_at,
                        "ended_at": self._now(),
                    },
                )
                context["tool_results"].append(
                    {
                        "tool_name": decision.tool_name,
                        "result": result,
                    }
                )
                self.repository.save_checkpoint(
                    run["org_id"],
                    {
                        "id": str(uuid4()),
                        "run_id": run_id,
                        "checkpoint_index": step_index,
                        "state_json": context,
                        "created_at": self._now(),
                    },
                )

            raise RuntimeError("Maximum agent step count exceeded")
        except Exception as exc:
            self.repository.finish_run(
                run_id,
                worker_id,
                "failed",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _tool_idempotency_key(
        run: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        business_key = run["input"].get("ticket_id", run["id"])
        return f"{tool_name}:{business_key}:{run['run_type']}"


run_executor = RunExecutor(run_repository)

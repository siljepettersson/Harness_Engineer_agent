from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from repositories.run_repository import SqlAlchemyRunRepository, run_repository
from services.run_errors import PermanentRunError, RetryableRunError
from services.tool_runtime import ToolDefinition, ToolRegistry, ToolRuntime


class ModelProvider(Protocol):
    def complete(
        self,
        *,
        context: dict[str, Any],
    ) -> "ModelDecision": ...


@dataclass(frozen=True)
class ModelDecision:
    provider: str
    model: str
    response_json: dict[str, Any]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    final_answer: dict[str, Any] | None = None
    tool_use_id: str | None = None


class ExecutorConfigurationError(RuntimeError):
    pass


class RunExecutor:
    """Execute a run through an injected model provider and tool registry."""

    def __init__(
        self,
        repository: SqlAlchemyRunRepository,
        model_provider: ModelProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_runtime: ToolRuntime | None = None,
        max_steps: int = 20,
    ) -> None:
        self.repository = repository
        self.model_provider = model_provider
        self.tool_registry = tool_registry
        self.tool_runtime = tool_runtime or ToolRuntime()
        self.max_steps = max_steps

    def execute(self, run_id: str, worker_id: str) -> dict[str, Any]:
        if self.model_provider is None or self.tool_registry is None:
            raise ExecutorConfigurationError(
                "Model provider and tool registry must be configured"
            )

        run = self.repository.claim_run(run_id, worker_id)
        if run is None:
            return {"run_id": run_id, "status": "skipped"}

        context = self._load_context(run)

        try:
            for loop_index in range(1, self.max_steps + 1):
                if self.repository.cancel_if_requested(run_id, worker_id):
                    return {"run_id": run_id, "status": "cancelled"}

                decision = self.run_model_step(
                    run=run,
                    context=context,
                    step_index=(loop_index * 2) - 1,
                )
                context["messages"].append(
                    self._assistant_message(decision)
                )
                context["last"] = "assistant"

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

                if self.repository.cancel_if_requested(run_id, worker_id):
                    return {"run_id": run_id, "status": "cancelled"}

                tool = self.tool_registry.get(decision.tool_name)
                result = self.run_tool_round(
                    run=run,
                    tool=tool,
                    decision=decision,
                    step_index=loop_index * 2,
                )
                context["messages"].append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": decision.tool_use_id,
                                "content": result,
                            }
                        ],
                    }
                )
                context["completed_tool_calls"].append(
                    decision.tool_use_id
                )
                context["last"] = "tool_result"
                self._save_context(run, context)

            raise RuntimeError("Maximum agent step count exceeded")
        except Exception as exc:
            if isinstance(exc, RetryableRunError):
                raise
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

    def run_model_step(
        self,
        run: dict[str, Any],
        context: dict[str, Any],
        step_index: int,
    ) -> ModelDecision:
        step_id = str(uuid4())
        started_at = self._now()
        request_json = context

        self.repository.create_step(
            run["org_id"],
            {
                "id": step_id,
                "run_id": run["id"],
                "step_index": step_index,
                "step_type": "model",
                "name": "decide_next_action",
                "status": "running",
                "input_json": request_json,
                "started_at": started_at,
            },
        )
        self.repository.append_event(
            run["org_id"],
            run["id"],
            "step.started",
            {"step_id": step_id, "step_type": "model"},
        )
        self.repository.append_event(
            run["org_id"],
            run["id"],
            "model.started",
            {"step_id": step_id},
        )

        try:
            response = self.model_provider.complete(
                context=context,
            )
            self.repository.record_model_call(
                run["org_id"],
                {
                    "id": str(uuid4()),
                    "run_id": run["id"],
                    "step_id": step_id,
                    "provider": response.provider,
                    "model": response.model,
                    "request_json": request_json,
                    "response_json": response.response_json,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "latency_ms": response.latency_ms,
                    "status": "succeeded",
                    "created_at": started_at,
                },
            )
            self.repository.update_step(
                run["org_id"],
                step_id,
                "succeeded",
                output_json=response.response_json,
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "model.completed",
                {"step_id": step_id},
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "step.completed",
                {"step_id": step_id},
            )
            return response
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self.repository.record_model_call(
                run["org_id"],
                {
                    "id": str(uuid4()),
                    "run_id": run["id"],
                    "step_id": step_id,
                    "provider": "unknown",
                    "model": "unknown",
                    "request_json": request_json,
                    "status": "failed",
                    "error_json": error,
                    "created_at": started_at,
                },
            )
            self.repository.update_step(
                run["org_id"],
                step_id,
                "failed",
                error_json=error,
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "model.failed",
                {"step_id": step_id, "error": error},
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "step.failed",
                {"step_id": step_id, "error": error},
            )
            raise

    def run_tool_round(
        self,
        run: dict[str, Any],
        tool: ToolDefinition,
        decision: ModelDecision,
        step_index: int,
    ) -> dict[str, Any]:
        if decision.tool_name is None or decision.tool_arguments is None:
            raise PermanentRunError("tool_decision_missing_arguments")

        step_id = str(uuid4())
        started_at = self._now()
        arguments = decision.tool_arguments
        idempotency_key = self._tool_idempotency_key(
            run,
            tool,
            arguments,
            decision.tool_use_id,
        )

        self.repository.create_step(
            run["org_id"],
            {
                "id": step_id,
                "run_id": run["id"],
                "step_index": step_index,
                "step_type": "tool",
                "name": tool.name,
                "status": "running",
                "input_json": arguments,
                "started_at": started_at,
            },
        )
        self.repository.append_event(
            run["org_id"],
            run["id"],
            "step.started",
            {"step_id": step_id, "step_type": "tool", "name": tool.name},
        )
        self.repository.append_event(
            run["org_id"],
            run["id"],
            "tool.started",
            {"step_id": step_id, "tool_name": tool.name},
        )

        try:
            result = self.tool_runtime.execute(
                tool=tool,
                arguments=arguments,
                idempotency_key=idempotency_key,
            )
            self.repository.record_tool_call(
                run["org_id"],
                {
                    "id": str(uuid4()),
                    "run_id": run["id"],
                    "step_id": step_id,
                    "tool_name": tool.name,
                    "args_json": arguments,
                    "result_json": result,
                    "status": "succeeded",
                    "idempotency_key": idempotency_key,
                    "created_at": started_at,
                    "ended_at": self._now(),
                },
            )
            self.repository.update_step(
                run["org_id"],
                step_id,
                "succeeded",
                output_json=result,
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "tool.completed",
                {"step_id": step_id, "tool_name": tool.name},
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "step.completed",
                {"step_id": step_id, "step_type": "tool"},
            )
            return result
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self.repository.record_tool_call(
                run["org_id"],
                {
                    "id": str(uuid4()),
                    "run_id": run["id"],
                    "step_id": step_id,
                    "tool_name": tool.name,
                    "args_json": arguments,
                    "status": "failed",
                    "idempotency_key": idempotency_key,
                    "error_json": error,
                    "created_at": started_at,
                    "ended_at": self._now(),
                },
            )
            self.repository.update_step(
                run["org_id"],
                step_id,
                "failed",
                error_json=error,
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "tool.failed",
                {"step_id": step_id, "tool_name": tool.name, "error": error},
            )
            self.repository.append_event(
                run["org_id"],
                run["id"],
                "step.failed",
                {"step_id": step_id, "step_type": "tool", "error": error},
            )
            raise

    @staticmethod
    def _initial_context(run: dict[str, Any]) -> dict[str, Any]:
        """Build the canonical message-list context for a new run."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": run["input"],
                }
            ],
            "completed_tool_calls": [],
            "last": "user",
        }

    def _load_context(self, run: dict[str, Any]) -> dict[str, Any]:
        checkpoint = self.repository.get_latest_checkpoint(
            run_id=run["id"],
            org_id=run["org_id"],
        )
        if checkpoint is not None:
            return checkpoint["state_json"]
        return self._initial_context(run)

    def _save_context(
        self,
        run: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        checkpoint_index = self.repository.next_checkpoint_index(
            run_id=run["id"],
            org_id=run["org_id"],
        )
        self.repository.save_checkpoint(
            run["org_id"],
            {
                "id": str(uuid4()),
                "run_id": run["id"],
                "checkpoint_index": checkpoint_index,
                "state_json": context,
                "created_at": self._now(),
            },
        )

    @staticmethod
    def _assistant_message(
        decision: ModelDecision,
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": decision.response_json,
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _tool_idempotency_key(
        run: dict[str, Any],
        tool: ToolDefinition,
        arguments: dict[str, Any],
        tool_use_id: str | None,
    ) -> str:
        if tool.idempotency_key is not None:
            return tool.idempotency_key(run, arguments)
        return f"{tool.name}:{run['id']}:{tool_use_id or 'unknown'}"


run_executor = RunExecutor(run_repository)

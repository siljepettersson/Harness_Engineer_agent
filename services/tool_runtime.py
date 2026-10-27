from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from services.run_errors import PermanentRunError, TransientRunError


ToolHandler = Callable[
    [dict[str, Any], str | None],
    dict[str, Any],
]
IdempotencyKeyFactory = Callable[
    [dict[str, Any], dict[str, Any]],
    str,
]
ArgumentValidator = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: ToolHandler
    requires_approval: bool = False
    validate_arguments: ArgumentValidator | None = None
    idempotency_key: IdempotencyKeyFactory | None = None


class ToolRegistry:
    """Resolve model-selected tool names without business logic in the loop."""

    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise PermanentRunError(
                f"Unknown tool: {name}"
            ) from exc


class ToolRuntime:
    """Execute a resolved tool and normalize runtime failures."""

    def execute(
        self,
        *,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if tool.validate_arguments is not None:
            try:
                tool.validate_arguments(arguments)
            except (TypeError, ValueError, KeyError) as exc:
                raise PermanentRunError(
                    f"Invalid arguments for {tool.name}: {exc}"
                ) from exc

        try:
            return tool.handler(arguments, idempotency_key)
        except (TimeoutError, ConnectionError) as exc:
            raise TransientRunError(
                f"Transient failure in {tool.name}: {exc}"
            ) from exc
        except PermissionError as exc:
            raise PermanentRunError(
                f"Permission denied for {tool.name}: {exc}"
            ) from exc

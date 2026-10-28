from __future__ import annotations

from dataclasses import dataclass, field
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
class ToolContext:
    """Identity and execution context supplied to a tool call.

    The context is deliberately separate from model-generated arguments.  It
    is backend-owned data used by the later authorization, tenant-isolation,
    and audit gates.
    """

    org_id: str
    user_id: str
    run_id: str
    step_id: str | None = None
    permissions: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for field_name in ("org_id", "user_id", "run_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.step_id is not None and not self.step_id.strip():
            raise ValueError("step_id must be a non-empty string when provided")

        object.__setattr__(self, "permissions", frozenset(self.permissions))


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: ToolHandler
    requires_approval: bool = False
    validate_arguments: ArgumentValidator | None = None
    idempotency_key: IdempotencyKeyFactory | None = None
    description: str = ""
    input_schema: type[Any] | None = None
    output_schema: type[Any] | None = None
    permission: str | None = None
    side_effect: bool = False
    idempotency_required: bool = False
    timeout_seconds: float = 30.0
    retryable_errors: frozenset[str] = field(default_factory=frozenset)
    tool_version: str | None = None
    risk_level: str = "low"
    approval_policy: dict[str, Any] | None = None
    network_policy: dict[str, Any] | None = None
    data_classification: str | None = None
    audit_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not callable(self.handler):
            raise TypeError(f"handler for {self.name} must be callable")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.risk_level not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be low, medium, or high")
        if self.side_effect and not self.idempotency_required:
            raise ValueError(
                "side-effect tools must require an idempotency key"
            )
        if self.risk_level == "high" and not self.approval_policy:
            raise ValueError(
                "high-risk tools must define an approval policy"
            )

        object.__setattr__(
            self,
            "retryable_errors",
            frozenset(self.retryable_errors),
        )


class ToolRegistry:
    """Resolve model-selected tool names without business logic in the loop."""

    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        if not isinstance(tool, ToolDefinition):
            raise TypeError("registry only accepts ToolDefinition instances")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def all(self) -> tuple[ToolDefinition, ...]:
        """Return a stable snapshot for inspection and governance checks."""
        return tuple(self._tools.values())

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
        context: ToolContext | None = None,
    ) -> dict[str, Any]:
        # ``context`` is accepted as part of the public contract now.  The
        # authorization and persistence gates are introduced in later steps;
        # keeping it optional preserves the existing executor integration.
        del context
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

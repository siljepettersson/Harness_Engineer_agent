from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

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
        for schema_name in ("input_schema", "output_schema"):
            schema = getattr(self, schema_name)
            if schema is not None and not callable(
                getattr(schema, "model_validate", None)
            ):
                raise TypeError(
                    f"{schema_name} for {self.name} must provide model_validate"
                )
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


PermissionChecker = Callable[[str, ToolContext], bool]
ResourceAuthorizer = Callable[
    [ToolDefinition, dict[str, Any], ToolContext],
    bool,
]


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

    def __init__(
        self,
        *,
        permission_checker: PermissionChecker | None = None,
        resource_authorizer: ResourceAuthorizer | None = None,
    ) -> None:
        self.permission_checker = permission_checker
        self.resource_authorizer = resource_authorizer

    def _check_authorization(
        self,
        *,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        context: ToolContext | None,
    ) -> None:
        if tool.permission is None and self.resource_authorizer is None:
            return

        if context is None:
            raise PermanentRunError(
                f"Tool context is required for {tool.name} authorization"
            )

        if tool.permission is not None:
            checker = self.permission_checker or self._default_permission_checker
            try:
                allowed = checker(tool.permission, context)
            except PermissionError as exc:
                raise PermanentRunError(
                    f"Permission denied for {tool.name}: {exc}"
                ) from exc
            if not allowed:
                raise PermanentRunError(
                    f"Permission denied for {tool.name}"
                )

        if self.resource_authorizer is not None:
            try:
                resource_allowed = self.resource_authorizer(
                    tool,
                    arguments,
                    context,
                )
            except PermissionError as exc:
                raise PermanentRunError(
                    f"Resource access denied for {tool.name}: {exc}"
                ) from exc
            if not resource_allowed:
                raise PermanentRunError(
                    f"Resource access denied for {tool.name}"
                )

    @staticmethod
    def _default_permission_checker(
        permission: str,
        context: ToolContext,
    ) -> bool:
        return permission in context.permissions

    @staticmethod
    def _validate_input(
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        validated_arguments = arguments

        if tool.input_schema is not None:
            try:
                model = tool.input_schema.model_validate(arguments)
                validated_arguments = model.model_dump()
            except (ValidationError, TypeError, ValueError) as exc:
                raise PermanentRunError(
                    f"Invalid arguments for {tool.name}: {exc}"
                ) from exc

        if tool.validate_arguments is not None:
            try:
                tool.validate_arguments(validated_arguments)
            except (TypeError, ValueError, KeyError) as exc:
                raise PermanentRunError(
                    f"Invalid arguments for {tool.name}: {exc}"
                ) from exc

        return validated_arguments

    @staticmethod
    def _validate_output(
        tool: ToolDefinition,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if tool.output_schema is None:
            return result

        try:
            model = tool.output_schema.model_validate(result)
            return model.model_dump()
        except (ValidationError, TypeError, ValueError) as exc:
            raise PermanentRunError(
                f"Invalid result for {tool.name}: {exc}"
            ) from exc

    def execute(
        self,
        *,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        idempotency_key: str | None = None,
        context: ToolContext | None = None,
    ) -> dict[str, Any]:
        validated_arguments = self._validate_input(tool, arguments)
        self._check_authorization(
            tool=tool,
            arguments=validated_arguments,
            context=context,
        )

        try:
            result = tool.handler(validated_arguments, idempotency_key)
            return self._validate_output(tool, result)
        except (TimeoutError, ConnectionError) as exc:
            raise TransientRunError(
                f"Transient failure in {tool.name}: {exc}"
            ) from exc
        except PermissionError as exc:
            raise PermanentRunError(
                f"Permission denied for {tool.name}: {exc}"
            ) from exc

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Callable, Protocol
from uuid import uuid4

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
    max_attempts: int = 1
    backoff_seconds: float = 0.0
    downstream_deduplicates: bool = False
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
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
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


@dataclass(frozen=True)
class ToolErrorPayload:
    code: str
    message: str
    retryable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


class RetryableToolError(TransientRunError):
    def __init__(self, payload: ToolErrorPayload) -> None:
        super().__init__(payload.message)
        self.tool_error = payload.as_dict()


class NonRetryableToolError(PermanentRunError):
    def __init__(self, payload: ToolErrorPayload) -> None:
        super().__init__(payload.message)
        self.tool_error = payload.as_dict()


class ResponseLostToolError(PermanentRunError):
    def __init__(self, payload: ToolErrorPayload) -> None:
        super().__init__(payload.message)
        self.tool_error = payload.as_dict()


class ToolCallStore(Protocol):
    """Persistence contract used by ToolRuntime for call lifecycle records."""

    def get_tool_call_by_idempotency_key(
        self,
        org_id: str,
        tool_name: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None: ...

    def record_tool_call(
        self,
        org_id: str,
        call: dict[str, Any],
    ) -> dict[str, Any]: ...

    def update_tool_call(
        self,
        org_id: str,
        tool_call_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]: ...


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
            raise NonRetryableToolError(
                ToolErrorPayload(
                    code="TOOL_NOT_FOUND",
                    message=f"Unknown tool: {name}",
                    retryable=False,
                )
            ) from exc


class ToolRuntime:
    """Execute a resolved tool and normalize runtime failures."""

    def __init__(
        self,
        *,
        permission_checker: PermissionChecker | None = None,
        resource_authorizer: ResourceAuthorizer | None = None,
        tool_call_store: ToolCallStore | None = None,
    ) -> None:
        self.permission_checker = permission_checker
        self.resource_authorizer = resource_authorizer
        self.tool_call_store = tool_call_store

    @staticmethod
    def _invoke_with_timeout(
        tool: ToolDefinition,
        arguments: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            tool.handler,
            arguments,
            idempotency_key,
        )
        try:
            result = future.result(timeout=tool.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(
                f"{tool.name} timed out after {tool.timeout_seconds}s"
            ) from exc
        except BaseException:
            executor.shutdown(wait=True)
            raise
        else:
            executor.shutdown(wait=True)
            return result

    @staticmethod
    def _safe_side_effect_retry(
        *,
        tool: ToolDefinition,
        tool_call_store: ToolCallStore | None,
    ) -> bool:
        return bool(
            tool.idempotency_required
            and tool.downstream_deduplicates
            and tool_call_store is not None
        )

    def _classify_handler_error(
        self,
        *,
        tool: ToolDefinition,
        exc: BaseException,
    ) -> tuple[ToolErrorPayload, bool]:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            code = "TOOL_RATE_LIMITED"
        elif isinstance(status_code, int) and status_code >= 500:
            code = "TOOL_UPSTREAM_ERROR"
        elif isinstance(exc, TimeoutError):
            if tool.side_effect and not self._safe_side_effect_retry(
                tool=tool,
                tool_call_store=self.tool_call_store,
            ):
                return (
                    ToolErrorPayload(
                        code="TOOL_RESPONSE_LOST",
                        message=f"Response lost for {tool.name}",
                        retryable=False,
                    ),
                    True,
                )
            code = "TOOL_TIMEOUT"
        elif isinstance(exc, ConnectionError):
            code = "TOOL_UPSTREAM_ERROR"
        elif isinstance(exc, PermissionError):
            return (
                ToolErrorPayload(
                    code="TOOL_PERMISSION_DENIED",
                    message=f"Permission denied for {tool.name}: {exc}",
                    retryable=False,
                ),
                False,
            )
        else:
            return (
                ToolErrorPayload(
                    code="TOOL_EXECUTION_FAILED",
                    message=f"Tool {tool.name} failed: {exc}",
                    retryable=False,
                ),
                False,
            )

        retryable = code in tool.retryable_errors
        if not tool.side_effect:
            retryable = True
        if tool.side_effect and not self._safe_side_effect_retry(
            tool=tool,
            tool_call_store=self.tool_call_store,
        ):
            retryable = False
        return (
            ToolErrorPayload(
                code=code,
                message=(
                    f"Transient failure in {tool.name}: {exc}"
                ),
                retryable=retryable,
            ),
            False,
        )

    def _execute_handler_with_retry(
        self,
        *,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        for attempt in range(1, tool.max_attempts + 1):
            try:
                return self._invoke_with_timeout(
                    tool,
                    arguments,
                    idempotency_key,
                )
            except Exception as exc:
                payload, response_lost = self._classify_handler_error(
                    tool=tool,
                    exc=exc,
                )
                if response_lost:
                    raise ResponseLostToolError(payload) from exc
                if payload.retryable and attempt < tool.max_attempts:
                    if tool.backoff_seconds:
                        time.sleep(
                            tool.backoff_seconds * (2 ** (attempt - 1))
                        )
                    continue
                if payload.retryable:
                    raise RetryableToolError(payload) from exc
                raise NonRetryableToolError(payload) from exc

        raise NonRetryableToolError(
            ToolErrorPayload(
                code="TOOL_EXECUTION_FAILED",
                message=f"Tool {tool.name} exhausted execution attempts",
                retryable=False,
            )
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _update_tool_call(
        self,
        *,
        call: dict[str, Any],
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a lifecycle record, with an in-memory compatibility path."""
        if self.tool_call_store is None:
            call.update(updates)
            return call

        updater = getattr(self.tool_call_store, "update_tool_call", None)
        if updater is not None:
            return updater(
                call["org_id"],
                call["id"],
                updates,
            )

        # Existing lightweight repositories return and retain the same dict.
        # Mutating it preserves their behavior until they implement the
        # durable update method.
        call.update(updates)
        return call

    def _start_tool_call(
        self,
        *,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        idempotency_key: str | None,
        context: ToolContext,
    ) -> dict[str, Any] | None:
        if self.tool_call_store is None:
            return None

        started_at = self._now()
        call = {
            "id": str(uuid4()),
            "org_id": context.org_id,
            "run_id": context.run_id,
            "step_id": context.step_id,
            "tool_name": tool.name,
            "args_json": arguments,
            "status": "running",
            "idempotency_key": idempotency_key,
            "created_at": started_at,
        }
        persisted = self.tool_call_store.record_tool_call(
            context.org_id,
            call,
        )
        # The SQLAlchemy repository returns a detached dict without org_id;
        # retain it locally for the later update call.
        persisted.setdefault("org_id", context.org_id)
        return persisted

    def _existing_tool_call(
        self,
        *,
        tool: ToolDefinition,
        idempotency_key: str | None,
        context: ToolContext | None,
    ) -> dict[str, Any] | None:
        if (
            self.tool_call_store is None
            or context is None
            or idempotency_key is None
        ):
            return None

        return self.tool_call_store.get_tool_call_by_idempotency_key(
            org_id=context.org_id,
            tool_name=tool.name,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _duplicate_result(
        *,
        tool: ToolDefinition,
        existing: dict[str, Any],
    ) -> dict[str, Any]:
        if existing.get("status") in {"succeeded", "skipped_duplicate"}:
            result = existing.get("result_json")
            if isinstance(result, dict):
                return result
        raise NonRetryableToolError(
            ToolErrorPayload(
                code="TOOL_CALL_IN_PROGRESS",
                message=f"Tool call already exists for {tool.name}",
                retryable=False,
            )
        )

    def _prepare_idempotency(
        self,
        *,
        tool: ToolDefinition,
        idempotency_key: str | None,
        context: ToolContext | None,
    ) -> dict[str, Any] | None:
        if tool.idempotency_required and not idempotency_key:
            raise NonRetryableToolError(
                ToolErrorPayload(
                    code="TOOL_IDEMPOTENCY_REQUIRED",
                    message=f"Idempotency key required for {tool.name}",
                    retryable=False,
                )
            )

        if idempotency_key is None:
            return None

        if self.tool_call_store is None:
            if tool.idempotency_required:
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_IDEMPOTENCY_STORE_UNAVAILABLE",
                        message=(
                            f"Tool call store is required for {tool.name}"
                        ),
                        retryable=False,
                    )
                )
            return None
        if context is None:
            raise NonRetryableToolError(
                ToolErrorPayload(
                    code="TOOL_CONTEXT_REQUIRED",
                    message=f"Tool context is required for {tool.name}",
                    retryable=False,
                )
            )

        existing = self._existing_tool_call(
            tool=tool,
            idempotency_key=idempotency_key,
            context=context,
        )
        if existing is not None:
            return existing
        return None

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
            raise NonRetryableToolError(
                ToolErrorPayload(
                    code="TOOL_CONTEXT_REQUIRED",
                    message=(
                        f"Tool context is required for {tool.name} authorization"
                    ),
                    retryable=False,
                )
            )

        if tool.permission is not None:
            checker = self.permission_checker or self._default_permission_checker
            try:
                allowed = checker(tool.permission, context)
            except PermissionError as exc:
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_PERMISSION_DENIED",
                        message=f"Permission denied for {tool.name}: {exc}",
                        retryable=False,
                    )
                ) from exc
            if not allowed:
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_PERMISSION_DENIED",
                        message=f"Permission denied for {tool.name}",
                        retryable=False,
                    )
                )

        if self.resource_authorizer is not None:
            try:
                resource_allowed = self.resource_authorizer(
                    tool,
                    arguments,
                    context,
                )
            except PermissionError as exc:
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_PERMISSION_DENIED",
                        message=(
                            f"Resource access denied for {tool.name}: {exc}"
                        ),
                        retryable=False,
                    )
                ) from exc
            if not resource_allowed:
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_PERMISSION_DENIED",
                        message=f"Resource access denied for {tool.name}",
                        retryable=False,
                    )
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
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_VALIDATION_ERROR",
                        message=f"Invalid arguments for {tool.name}: {exc}",
                        retryable=False,
                    )
                ) from exc

        if tool.validate_arguments is not None:
            try:
                tool.validate_arguments(validated_arguments)
            except (TypeError, ValueError, KeyError) as exc:
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_VALIDATION_ERROR",
                        message=f"Invalid arguments for {tool.name}: {exc}",
                        retryable=False,
                    )
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
            raise NonRetryableToolError(
                ToolErrorPayload(
                    code="TOOL_OUTPUT_VALIDATION_ERROR",
                    message=f"Invalid result for {tool.name}: {exc}",
                    retryable=False,
                )
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

        existing = self._prepare_idempotency(
            tool=tool,
            idempotency_key=idempotency_key,
            context=context,
        )
        if existing is not None:
            return self._duplicate_result(tool=tool, existing=existing)

        tool_call = None
        if self.tool_call_store is not None:
            if context is None:
                raise NonRetryableToolError(
                    ToolErrorPayload(
                        code="TOOL_CONTEXT_REQUIRED",
                        message=f"Tool context is required to record {tool.name}",
                        retryable=False,
                    )
                )
            tool_call = self._start_tool_call(
                tool=tool,
                arguments=validated_arguments,
                idempotency_key=idempotency_key,
                context=context,
            )

        try:
            result = self._execute_handler_with_retry(
                tool=tool,
                arguments=validated_arguments,
                idempotency_key=idempotency_key,
            )
            validated_result = self._validate_output(tool, result)
            if tool_call is not None:
                self._update_tool_call(
                    call=tool_call,
                    updates={
                        "status": "succeeded",
                        "result_json": validated_result,
                        "ended_at": self._now(),
                    },
                )
            return validated_result
        except ResponseLostToolError as exc:
            if tool_call is not None:
                self._update_tool_call(
                    call=tool_call,
                    updates={
                        "status": "response_lost",
                        "error_json": exc.tool_error,
                        "ended_at": self._now(),
                    },
                )
            raise
        except (RetryableToolError, NonRetryableToolError) as exc:
            if tool_call is not None:
                self._update_tool_call(
                    call=tool_call,
                    updates={
                        "status": "failed",
                        "error_json": exc.tool_error,
                        "ended_at": self._now(),
                    },
                )
            raise
        except PermanentRunError as exc:
            if tool_call is not None:
                self._update_tool_call(
                    call=tool_call,
                    updates={
                        "status": "failed",
                        "error_json": {
                            "code": "TOOL_OUTPUT_VALIDATION_ERROR",
                            "message": str(exc),
                            "retryable": False,
                        },
                        "ended_at": self._now(),
                    },
                )
            raise

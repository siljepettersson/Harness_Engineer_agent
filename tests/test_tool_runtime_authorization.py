import pytest

from services.run_errors import PermanentRunError
from services.tool_runtime import ToolContext, ToolDefinition, ToolRuntime


def _tool(permission="orders:read"):
    return ToolDefinition(
        name="get_order",
        handler=lambda arguments, key: {"order_id": arguments["order_id"]},
        permission=permission,
    )


def test_runtime_allows_a_tool_when_context_has_required_permission():
    result = ToolRuntime().execute(
        tool=_tool(),
        arguments={"order_id": "ORD-10086"},
        context=ToolContext(
            org_id="org-1",
            user_id="user-1",
            run_id="run-1",
            permissions={"orders:read"},
        ),
    )

    assert result == {"order_id": "ORD-10086"}


def test_runtime_denies_a_tool_without_required_permission():
    with pytest.raises(PermanentRunError, match="Permission denied"):
        ToolRuntime().execute(
            tool=_tool(),
            arguments={"order_id": "ORD-10086"},
            context=ToolContext(
                org_id="org-1",
                user_id="user-1",
                run_id="run-1",
                permissions={"orders:write"},
            ),
        )


def test_permissioned_tool_requires_backend_context():
    with pytest.raises(PermanentRunError, match="context is required"):
        ToolRuntime().execute(
            tool=_tool(),
            arguments={"order_id": "ORD-10086"},
        )


def test_resource_authorizer_receives_context_and_validated_arguments():
    calls = []

    def authorize(tool, arguments, context):
        calls.append((tool.name, arguments, context.org_id))
        return context.org_id == "org-1" and arguments["order_id"] == "ORD-10086"

    result = ToolRuntime(resource_authorizer=authorize).execute(
        tool=_tool(permission=None),
        arguments={"order_id": "ORD-10086"},
        context=ToolContext(
            org_id="org-1",
            user_id="user-1",
            run_id="run-1",
        ),
    )

    assert result == {"order_id": "ORD-10086"}
    assert calls == [("get_order", {"order_id": "ORD-10086"}, "org-1")]


def test_resource_authorizer_can_deny_cross_tenant_resource():
    with pytest.raises(PermanentRunError, match="Resource access denied"):
        ToolRuntime(
            resource_authorizer=lambda tool, arguments, context: False,
        ).execute(
            tool=_tool(permission=None),
            arguments={"order_id": "ORD-other-org"},
            context=ToolContext(
                org_id="org-1",
                user_id="user-1",
                run_id="run-1",
            ),
        )

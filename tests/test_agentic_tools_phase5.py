"""Tests for Agentic Tools Phase 5: MCP bridge.

Covers:
  1. MCPToolInfo / MCPCallResult dataclasses
  2. NullMCPClient behavior
  3. MCPClient protocol conformance
  4. Category inference from tool names and descriptions
  5. MCPAdapter: discovery → ToolCapability mapping
  6. MCPAdapter: handler creation → ToolResult normalization
  7. register_mcp_tools: registry + executor integration
  8. Builtin + MCP coexistence in one registry
  9. Executor dispatch to MCP-backed tools
  10. Cognitive tool loop compatibility with MCP tools
  11. MCP handler exception safety
"""

from __future__ import annotations

from typing import Any

import pytest

from core.types import (
    ToolCapability,
    ToolCategory,
    ToolResult,
    ToolTrace,
)
from core.emotional_engine import EmotionalEngine
from core.dual_process.tool_loop import run_tool_loop
from openrook_tools.registry import ToolRegistry
from openrook_tools.executor import ToolExecutor
from openrook_tools import register_builtins, register_mcp_tools
from openrook_tools.mcp.client import MCPClient, MCPCallResult, MCPToolInfo, NullMCPClient
from openrook_tools.mcp.adapter import MCPAdapter, infer_category


# ===================================================================
# Fake MCP client for testing
# ===================================================================

class FakeMCPClient:
    """MCP client that returns pre-configured tools and results."""

    def __init__(
        self,
        tools: list[MCPToolInfo] | None = None,
        results: dict[str, MCPCallResult] | None = None,
    ) -> None:
        self._tools = tools or []
        self._results = results or {}
        self.call_log: list[tuple[str, dict]] = []

    def discover_tools(self) -> list[MCPToolInfo]:
        return list(self._tools)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPCallResult:
        self.call_log.append((tool_name, arguments))
        if tool_name in self._results:
            return self._results[tool_name]
        return MCPCallResult(is_error=True, error_message=f"Unknown tool: {tool_name}")


class CrashingMCPClient:
    """MCP client that raises on call_tool — tests exception safety."""

    def discover_tools(self) -> list[MCPToolInfo]:
        return [MCPToolInfo(name="crash_tool", description="Will crash")]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPCallResult:
        raise RuntimeError("MCP connection lost")


# ===================================================================
# 1. MCPToolInfo / MCPCallResult dataclasses
# ===================================================================

class TestMCPDataclasses:
    def test_tool_info_defaults(self):
        info = MCPToolInfo(name="test")
        assert info.name == "test"
        assert info.description == ""
        assert info.input_schema == {}
        assert info.server_name == ""

    def test_tool_info_with_schema(self):
        info = MCPToolInfo(
            name="create_event",
            description="Create a calendar event",
            input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
            server_name="calendar",
        )
        assert info.input_schema["type"] == "object"

    def test_call_result_success(self):
        result = MCPCallResult(content="Event created", is_error=False)
        assert result.content == "Event created"
        assert result.is_error is False

    def test_call_result_error(self):
        result = MCPCallResult(is_error=True, error_message="Auth failed")
        assert result.is_error is True
        assert result.error_message == "Auth failed"


# ===================================================================
# 2. NullMCPClient
# ===================================================================

class TestNullMCPClient:
    def test_discover_returns_empty(self):
        client = NullMCPClient()
        assert client.discover_tools() == []

    def test_call_returns_error(self):
        client = NullMCPClient()
        result = client.call_tool("anything", {})
        assert result.is_error is True
        assert "No MCP server configured" in result.error_message


# ===================================================================
# 3. MCPClient protocol conformance
# ===================================================================

class TestMCPClientProtocol:
    def test_fake_client_is_mcp_client(self):
        assert isinstance(FakeMCPClient(), MCPClient)

    def test_null_client_is_mcp_client(self):
        assert isinstance(NullMCPClient(), MCPClient)

    def test_crashing_client_is_mcp_client(self):
        assert isinstance(CrashingMCPClient(), MCPClient)


# ===================================================================
# 4. Category inference
# ===================================================================

class TestInferCategory:
    def test_read_only_from_name(self):
        assert infer_category("get_events") == ToolCategory.READ_ONLY
        assert infer_category("list_files") == ToolCategory.READ_ONLY
        assert infer_category("getRecord") == ToolCategory.READ_ONLY
        assert infer_category("fetch_data") == ToolCategory.READ_ONLY

    def test_riskier_keyword_outranks_read_keyword(self):
        # Categories are checked from most to least risky so mixed-verb
        # names cannot bypass the WRITE/DESTRUCTIVE confirmation gate.
        assert infer_category("getOrDeleteRecords") == ToolCategory.DESTRUCTIVE
        assert infer_category("fetchAndUpdate") == ToolCategory.WRITE
        assert infer_category("list_and_send_report") == ToolCategory.EXTERNAL_ACTION
        # "message" is an EXTERNAL_ACTION keyword; it outranks "search".
        assert infer_category("search_messages") == ToolCategory.EXTERNAL_ACTION

    def test_riskier_keyword_outranks_read_keyword_in_description(self):
        assert infer_category("do_thing", "Get or delete records") == ToolCategory.DESTRUCTIVE

    def test_write_from_name(self):
        assert infer_category("create_event") == ToolCategory.WRITE
        assert infer_category("update_record") == ToolCategory.WRITE
        assert infer_category("add_item") == ToolCategory.WRITE

    def test_destructive_from_name(self):
        assert infer_category("delete_file") == ToolCategory.DESTRUCTIVE
        assert infer_category("remove_user") == ToolCategory.DESTRUCTIVE
        assert infer_category("purge_cache") == ToolCategory.DESTRUCTIVE

    def test_external_action_from_name(self):
        assert infer_category("send_email") == ToolCategory.EXTERNAL_ACTION
        assert infer_category("post_message") == ToolCategory.EXTERNAL_ACTION
        assert infer_category("notify_user") == ToolCategory.EXTERNAL_ACTION

    def test_fallback_to_description(self):
        assert infer_category("do_thing", "Search for records") == ToolCategory.READ_ONLY
        assert infer_category("do_thing", "Create a new entry") == ToolCategory.WRITE

    def test_unknown_defaults_to_external_action(self):
        assert infer_category("xyzzyx") == ToolCategory.EXTERNAL_ACTION

    def test_hyphenated_names(self):
        assert infer_category("get-events") == ToolCategory.READ_ONLY
        assert infer_category("delete-item") == ToolCategory.DESTRUCTIVE

    def test_dotted_names(self):
        assert infer_category("calendar.list") == ToolCategory.READ_ONLY
        assert infer_category("files.delete") == ToolCategory.DESTRUCTIVE


# ===================================================================
# 4b. Operator trust controls: category overrides + strict mode
# ===================================================================

class TestOperatorCategoryOverrides:
    def test_exact_override_wins_over_keyword_inference(self):
        # "get_data" would infer READ_ONLY; the operator pin wins.
        assert infer_category(
            "get_data", overrides={"get_data": "external_action"},
        ) == ToolCategory.EXTERNAL_ACTION

    def test_override_can_downgrade_too(self):
        # Operators may also pin a falsely-scary name as read-only.
        assert infer_category(
            "execute_query", overrides={"execute_query": "read_only"},
        ) == ToolCategory.READ_ONLY

    def test_fnmatch_pattern_override(self):
        assert infer_category(
            "jira_create_issue", overrides={"jira_*": "destructive"},
        ) == ToolCategory.DESTRUCTIVE

    def test_exact_entry_wins_over_pattern(self):
        overrides = {"jira_*": "destructive", "jira_get_issue": "read_only"}
        assert infer_category("jira_get_issue", overrides=overrides) == ToolCategory.READ_ONLY
        assert infer_category("jira_delete_issue", overrides=overrides) == ToolCategory.DESTRUCTIVE

    def test_unmatched_override_falls_back_to_inference(self):
        assert infer_category(
            "delete_file", overrides={"other_tool": "read_only"},
        ) == ToolCategory.DESTRUCTIVE

    def test_invalid_category_string_raises(self):
        with pytest.raises(ValueError, match="invalid category 'readonly'"):
            infer_category("get_data", overrides={"get_data": "readonly"})

    def test_cognitive_not_assignable(self):
        with pytest.raises(ValueError, match="invalid category"):
            infer_category("get_data", overrides={"get_data": "cognitive"})

    def test_strict_mode_disables_inference(self):
        # A read-named tool is no longer trusted to be read-only.
        assert infer_category("get_data", strict=True) == ToolCategory.EXTERNAL_ACTION
        assert infer_category("list_files", strict=True) == ToolCategory.EXTERNAL_ACTION

    def test_strict_mode_override_still_wins(self):
        assert infer_category(
            "get_data", overrides={"get_data": "read_only"}, strict=True,
        ) == ToolCategory.READ_ONLY

    def _calendar_tools(self):
        return [
            MCPToolInfo(name="list_events", description="List calendar events"),
            MCPToolInfo(name="delete_event", description="Delete a calendar event"),
            MCPToolInfo(name="frobnicate", description=""),
        ]

    def _categories(self, adapter: MCPAdapter) -> dict[str, ToolCategory]:
        caps, _ = adapter.discover_and_adapt()
        return {c.name: c.category for c in caps}

    def test_adapter_applies_override_by_namespaced_name(self):
        client = FakeMCPClient(tools=self._calendar_tools())
        adapter = MCPAdapter(
            client, server_name="calendar",
            tool_categories={"mcp.calendar.list_events": "external_action"},
        )
        cats = self._categories(adapter)
        assert cats["mcp.calendar.list_events"] == ToolCategory.EXTERNAL_ACTION
        # Unpinned tools still use inference
        assert cats["mcp.calendar.delete_event"] == ToolCategory.DESTRUCTIVE

    def test_adapter_applies_override_by_bare_name(self):
        client = FakeMCPClient(tools=self._calendar_tools())
        adapter = MCPAdapter(
            client, server_name="calendar",
            tool_categories={"list_events": "write"},
        )
        cats = self._categories(adapter)
        assert cats["mcp.calendar.list_events"] == ToolCategory.WRITE

    def test_adapter_pattern_override_covers_whole_server(self):
        client = FakeMCPClient(tools=self._calendar_tools())
        adapter = MCPAdapter(
            client, server_name="calendar",
            tool_categories={"mcp.calendar.*": "external_action"},
        )
        cats = self._categories(adapter)
        assert set(cats.values()) == {ToolCategory.EXTERNAL_ACTION}

    def test_adapter_strict_mode_unmatched_is_external_action(self):
        client = FakeMCPClient(tools=self._calendar_tools())
        adapter = MCPAdapter(client, server_name="calendar", strict_categories=True)
        cats = self._categories(adapter)
        assert cats["mcp.calendar.list_events"] == ToolCategory.EXTERNAL_ACTION
        assert cats["mcp.calendar.delete_event"] == ToolCategory.EXTERNAL_ACTION
        assert cats["mcp.calendar.frobnicate"] == ToolCategory.EXTERNAL_ACTION

    def test_adapter_strict_mode_override_still_wins(self):
        client = FakeMCPClient(tools=self._calendar_tools())
        adapter = MCPAdapter(
            client, server_name="calendar",
            tool_categories={"mcp.calendar.list_events": "read_only"},
            strict_categories=True,
        )
        cats = self._categories(adapter)
        assert cats["mcp.calendar.list_events"] == ToolCategory.READ_ONLY
        assert cats["mcp.calendar.frobnicate"] == ToolCategory.EXTERNAL_ACTION

    def test_adapter_invalid_category_fails_at_construction(self):
        client = FakeMCPClient(tools=self._calendar_tools())
        with pytest.raises(ValueError, match="invalid category"):
            MCPAdapter(
                client, server_name="calendar",
                tool_categories={"list_events": "harmless"},
            )

    def test_register_mcp_tools_threads_trust_controls(self):
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        client = FakeMCPClient(tools=self._calendar_tools())
        register_mcp_tools(
            client, registry, executor,
            server_name="calendar",
            tool_categories={"mcp.calendar.frobnicate": "read_only"},
            strict_categories=True,
        )
        assert registry.get("mcp.calendar.frobnicate").category == ToolCategory.READ_ONLY
        assert registry.get("mcp.calendar.list_events").category == ToolCategory.EXTERNAL_ACTION


# ===================================================================
# 5. MCPAdapter: discovery → ToolCapability
# ===================================================================

class TestMCPAdapterDiscovery:
    def _tools(self):
        return [
            MCPToolInfo(
                name="list_events",
                description="List calendar events",
                input_schema={"type": "object", "properties": {"date": {"type": "string"}}},
                server_name="calendar",
            ),
            MCPToolInfo(
                name="create_event",
                description="Create a calendar event",
                input_schema={"type": "object"},
                server_name="calendar",
            ),
            MCPToolInfo(
                name="delete_event",
                description="Delete a calendar event",
                input_schema={},
                server_name="calendar",
            ),
        ]

    def test_discover_produces_capabilities(self):
        client = FakeMCPClient(tools=self._tools())
        adapter = MCPAdapter(client, server_name="calendar")
        caps, handlers = adapter.discover_and_adapt()
        assert len(caps) == 3
        assert len(handlers) == 3

    def test_namespaced_tool_names(self):
        client = FakeMCPClient(tools=self._tools())
        adapter = MCPAdapter(client, server_name="calendar")
        caps, _ = adapter.discover_and_adapt()
        names = [c.name for c in caps]
        assert "mcp.calendar.list_events" in names
        assert "mcp.calendar.create_event" in names
        assert "mcp.calendar.delete_event" in names

    def test_custom_namespace(self):
        client = FakeMCPClient(tools=self._tools()[:1])
        adapter = MCPAdapter(client, server_name="cal", namespace="ext.cal")
        caps, _ = adapter.discover_and_adapt()
        assert caps[0].name == "ext.cal.list_events"

    def test_category_assigned(self):
        client = FakeMCPClient(tools=self._tools())
        adapter = MCPAdapter(client, server_name="calendar")
        caps, _ = adapter.discover_and_adapt()
        by_name = {c.name: c for c in caps}
        assert by_name["mcp.calendar.list_events"].category == ToolCategory.READ_ONLY
        assert by_name["mcp.calendar.create_event"].category == ToolCategory.WRITE
        assert by_name["mcp.calendar.delete_event"].category == ToolCategory.DESTRUCTIVE

    def test_mcp_backed_flag(self):
        client = FakeMCPClient(tools=self._tools()[:1])
        adapter = MCPAdapter(client, server_name="calendar")
        caps, _ = adapter.discover_and_adapt()
        assert caps[0].mcp_backed is True

    def test_requires_network_flag(self):
        client = FakeMCPClient(tools=self._tools()[:1])
        adapter = MCPAdapter(client, server_name="calendar")
        caps, _ = adapter.discover_and_adapt()
        assert caps[0].requires_network is True

    def test_schema_preserved(self):
        client = FakeMCPClient(tools=self._tools()[:1])
        adapter = MCPAdapter(client, server_name="calendar")
        caps, _ = adapter.discover_and_adapt()
        assert caps[0].arg_schema["type"] == "object"

    def test_empty_discovery(self):
        client = FakeMCPClient(tools=[])
        adapter = MCPAdapter(client, server_name="empty")
        caps, handlers = adapter.discover_and_adapt()
        assert caps == []
        assert handlers == {}

    def test_description_fallback(self):
        client = FakeMCPClient(tools=[MCPToolInfo(name="mystery_tool")])
        adapter = MCPAdapter(client, server_name="test")
        caps, _ = adapter.discover_and_adapt()
        assert "MCP tool:" in caps[0].description


# ===================================================================
# 6. MCPAdapter: handler → ToolResult normalization
# ===================================================================

class TestMCPAdapterExecution:
    def test_successful_call(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="get_data")],
            results={"get_data": MCPCallResult(content="result data")},
        )
        adapter = MCPAdapter(client, server_name="test")
        _, handlers = adapter.discover_and_adapt()
        handler = handlers["mcp.test.get_data"]
        result = handler({"query": "test"})
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "result data"
        assert result.tool_name == "mcp.test.get_data"

    def test_error_call(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="fail_tool")],
            results={"fail_tool": MCPCallResult(is_error=True, error_message="Bad request")},
        )
        adapter = MCPAdapter(client, server_name="test")
        _, handlers = adapter.discover_and_adapt()
        result = handlers["mcp.test.fail_tool"]({})
        assert result.success is False
        assert "Bad request" in result.error

    def test_unknown_mcp_tool(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="some_tool")],
            # No result configured for "some_tool"
        )
        adapter = MCPAdapter(client, server_name="test")
        _, handlers = adapter.discover_and_adapt()
        result = handlers["mcp.test.some_tool"]({})
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_arguments_forwarded(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="get_data")],
            results={"get_data": MCPCallResult(content="ok")},
        )
        adapter = MCPAdapter(client, server_name="test")
        _, handlers = adapter.discover_and_adapt()
        handlers["mcp.test.get_data"]({"key": "value"})
        assert client.call_log[-1] == ("get_data", {"key": "value"})

    def test_metadata_preserved(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="get_data")],
            results={"get_data": MCPCallResult(content="ok", metadata={"source": "api"})},
        )
        adapter = MCPAdapter(client, server_name="test")
        _, handlers = adapter.discover_and_adapt()
        result = handlers["mcp.test.get_data"]({})
        assert result.metadata == {"source": "api"}


# ===================================================================
# 7. register_mcp_tools integration
# ===================================================================

class TestRegisterMCPTools:
    def test_register_into_registry(self):
        client = FakeMCPClient(tools=[
            MCPToolInfo(name="list_events", description="List events"),
            MCPToolInfo(name="create_event", description="Create event"),
        ])
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        names = register_mcp_tools(client, reg, exe, server_name="cal")
        assert "mcp.cal.list_events" in names
        assert "mcp.cal.create_event" in names
        assert reg.get("mcp.cal.list_events") is not None
        assert reg.get("mcp.cal.create_event") is not None

    def test_executor_can_dispatch(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="get_data")],
            results={"get_data": MCPCallResult(content="hello")},
        )
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_mcp_tools(client, reg, exe, server_name="test")
        result = exe.execute("mcp.test.get_data", {})
        assert result.success is True
        assert result.output == "hello"

    def test_empty_server_registers_nothing(self):
        client = FakeMCPClient(tools=[])
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        names = register_mcp_tools(client, reg, exe, server_name="empty")
        assert names == []
        assert len(reg) == 0


# ===================================================================
# 8. Builtin + MCP coexistence
# ===================================================================

class TestCoexistence:
    def test_builtins_and_mcp_in_one_registry(self):
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_builtins(reg, exe)
        builtin_count = len(reg)

        client = FakeMCPClient(tools=[
            MCPToolInfo(name="list_events", description="List events"),
            MCPToolInfo(name="send_email", description="Send email"),
        ])
        register_mcp_tools(client, reg, exe, server_name="ext")

        assert len(reg) == builtin_count + 2
        # Builtins still there
        assert reg.get("fs.read_file") is not None
        assert reg.get("shell.run_command") is not None
        # MCP tools also there
        assert reg.get("mcp.ext.list_events") is not None
        assert reg.get("mcp.ext.send_email") is not None

    def test_mcp_tools_have_mcp_backed_flag(self):
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_builtins(reg, exe)
        client = FakeMCPClient(tools=[MCPToolInfo(name="get_data")])
        register_mcp_tools(client, reg, exe, server_name="test")

        assert reg.get("fs.read_file").mcp_backed is False
        assert reg.get("mcp.test.get_data").mcp_backed is True

    def test_multiple_mcp_servers(self):
        reg = ToolRegistry()
        exe = ToolExecutor(reg)

        cal_client = FakeMCPClient(tools=[MCPToolInfo(name="list_events")])
        mail_client = FakeMCPClient(tools=[MCPToolInfo(name="send_email")])

        register_mcp_tools(cal_client, reg, exe, server_name="calendar")
        register_mcp_tools(mail_client, reg, exe, server_name="mail")

        assert reg.get("mcp.calendar.list_events") is not None
        assert reg.get("mcp.mail.send_email") is not None

    def test_list_tools_by_category(self):
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_builtins(reg, exe)
        client = FakeMCPClient(tools=[
            MCPToolInfo(name="delete_record", description="Delete a record"),
        ])
        register_mcp_tools(client, reg, exe, server_name="db")

        destructive = reg.list_tools(category=ToolCategory.DESTRUCTIVE)
        names = [t.name for t in destructive]
        assert "fs.delete_path" in names
        assert "shell.run_command" in names
        assert "mcp.db.delete_record" in names


# ===================================================================
# 9. Executor dispatch to MCP-backed tools
# ===================================================================

class TestExecutorMCPDispatch:
    def test_dispatch_mcp_success(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="fetch_data")],
            results={"fetch_data": MCPCallResult(content="42")},
        )
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_mcp_tools(client, reg, exe, server_name="api")
        result = exe.execute("mcp.api.fetch_data", {"id": 1})
        assert result.success is True
        assert result.output == "42"
        assert result.latency_ms > 0

    def test_dispatch_mcp_error(self):
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="fail_op")],
            results={"fail_op": MCPCallResult(is_error=True, error_message="Timeout")},
        )
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_mcp_tools(client, reg, exe, server_name="api")
        result = exe.execute("mcp.api.fail_op", {})
        assert result.success is False
        assert "Timeout" in result.error
        assert result.latency_ms > 0

    def test_dispatch_nonexistent_mcp_tool(self):
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        result = exe.execute("mcp.ghost.nope", {})
        assert result.success is False
        assert "Unknown tool" in result.error


# ===================================================================
# 10. Cognitive tool loop compatibility
# ===================================================================

class TestCognitiveLoopMCP:
    def test_mcp_tool_in_tool_loop(self):
        """MCP tools go through the same cognitive appraisal as builtins."""
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="read_data", description="Read data from API")],
            results={"read_data": MCPCallResult(content="data here")},
        )
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_builtins(reg, exe)
        register_mcp_tools(client, reg, exe, server_name="api")

        # The tool loop uses heuristic detection which won't match MCP tools
        # by default (it's designed for builtin names). But if we wire it
        # through the executor directly, it should still produce proper ToolResult.
        result = exe.execute("mcp.api.read_data", {"query": "test"})
        assert result.success is True

        # Verify the result can be appraised
        from core.tool_appraisal import appraise_tool_result
        obs = appraise_tool_result(result, ToolCategory.READ_ONLY)
        assert obs.certainty_delta > 0  # successful read bumps certainty

    def test_mcp_failure_appraisal(self):
        """Failed MCP call produces proper emotional appraisal."""
        client = FakeMCPClient(
            tools=[MCPToolInfo(name="send_msg")],
            results={"send_msg": MCPCallResult(is_error=True, error_message="Auth failed")},
        )
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_mcp_tools(client, reg, exe, server_name="chat")

        result = exe.execute("mcp.chat.send_msg", {"text": "hi"})
        assert result.success is False

        from core.tool_appraisal import appraise_tool_result
        obs = appraise_tool_result(result, ToolCategory.EXTERNAL_ACTION)
        assert obs.certainty_delta < 0  # failure drops certainty
        assert obs.emotional_delta.get("arousal", 0) > 0  # failure raises arousal


# ===================================================================
# 11. MCP handler exception safety
# ===================================================================

class TestMCPExceptionSafety:
    def test_crashing_client_produces_tool_result(self):
        """Client exception is caught and normalized to ToolResult."""
        client = CrashingMCPClient()
        adapter = MCPAdapter(client, server_name="crash")
        caps, handlers = adapter.discover_and_adapt()
        assert len(caps) == 1

        result = handlers["mcp.crash.crash_tool"]({})
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "MCP call failed" in result.error
        assert "RuntimeError" in result.error

    def test_crashing_client_through_executor(self):
        """Executor wrapping also handles MCP crashes gracefully."""
        client = CrashingMCPClient()
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_mcp_tools(client, reg, exe, server_name="crash")

        result = exe.execute("mcp.crash.crash_tool", {})
        assert result.success is False
        assert result.latency_ms > 0

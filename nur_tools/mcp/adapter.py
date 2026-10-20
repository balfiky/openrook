"""MCP adapter — normalizes MCP tools into OpenRook's internal model.

Responsibilities:
  1. Convert MCPToolInfo → ToolCapability (with category assignment)
  2. Create handler functions that call MCPClient and return ToolResult
  3. Register discovered tools into the shared ToolRegistry + ToolExecutor

MCP tools appear as normal participants in the cognitive tool loop:
same arbiter, same appraisal, same memory coupling.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Mapping

from core.types import ToolCapability, ToolCategory, ToolResult
from nur_tools.mcp.client import MCPClient, MCPToolInfo
from nur_tools.registry import ToolRegistry
from nur_tools.executor import ToolExecutor


# ---------------------------------------------------------------------------
# Category assignment — explicit and deterministic
# ---------------------------------------------------------------------------

# Known patterns for category inference from tool name/description.
# MCP servers don't declare categories, so we infer from naming conventions.
_CATEGORY_KEYWORDS: dict[ToolCategory, list[str]] = {
    ToolCategory.READ_ONLY: [
        "get", "list", "read", "search", "fetch", "query", "find",
        "show", "describe", "info", "status", "view",
    ],
    ToolCategory.WRITE: [
        "create", "write", "update", "set", "add", "put", "edit",
        "modify", "patch", "insert", "upload",
    ],
    ToolCategory.DESTRUCTIVE: [
        "delete", "remove", "drop", "destroy", "purge", "clear",
        "revoke", "uninstall", "terminate", "kill",
    ],
    ToolCategory.EXTERNAL_ACTION: [
        "send", "post", "publish", "notify", "email", "message",
        "invoke", "trigger", "execute", "run", "call",
    ],
}

# Categories checked in order of decreasing risk so that a risky keyword
# anywhere in a name wins over a read keyword (e.g. "getOrDeleteRecords"
# must be DESTRUCTIVE, never READ_ONLY — otherwise a mixed-verb tool name
# would bypass the WRITE/DESTRUCTIVE confirmation gate).
_CATEGORY_RISK_ORDER: list[ToolCategory] = [
    ToolCategory.DESTRUCTIVE,
    ToolCategory.WRITE,
    ToolCategory.EXTERNAL_ACTION,
    ToolCategory.READ_ONLY,
]

# Explicit overrides for well-known MCP tool patterns
_CATEGORY_OVERRIDES: dict[str, ToolCategory] = {}

# Operator-facing category strings (RuntimeConfig.mcp_tool_categories values)
# → ToolCategory. "cognitive" is intentionally absent: it is reserved for
# OpenRook-internal tools and must not be assignable to MCP tools.
_OVERRIDE_CATEGORY_VALUES: dict[str, ToolCategory] = {
    "read_only": ToolCategory.READ_ONLY,
    "write": ToolCategory.WRITE,
    "destructive": ToolCategory.DESTRUCTIVE,
    "external_action": ToolCategory.EXTERNAL_ACTION,
}


def _coerce_override_map(
    overrides: Mapping[str, str | ToolCategory] | None,
) -> dict[str, ToolCategory]:
    """Validate an operator override map into name/pattern → ToolCategory.

    Raises ValueError on an unknown category string so a typo fails loud
    instead of silently falling back to keyword inference.
    """
    coerced: dict[str, ToolCategory] = {}
    for name, category in (overrides or {}).items():
        if isinstance(category, ToolCategory):
            coerced[name] = category
            continue
        key = str(category).strip().lower()
        if key not in _OVERRIDE_CATEGORY_VALUES:
            raise ValueError(
                f"mcp_tool_categories[{name!r}]: invalid category {category!r}; "
                f"must be one of: {', '.join(sorted(_OVERRIDE_CATEGORY_VALUES))}"
            )
        coerced[name] = _OVERRIDE_CATEGORY_VALUES[key]
    return coerced


def resolve_category_override(
    tool_name: str,
    overrides: Mapping[str, ToolCategory],
) -> ToolCategory | None:
    """Resolve an operator override for ``tool_name``.

    Exact-name entries win over patterns. Otherwise entries are treated as
    fnmatch glob patterns (e.g. ``mcp.github.*``) and checked in mapping
    order; the first match wins.
    """
    if tool_name in overrides:
        return overrides[tool_name]
    for pattern, category in overrides.items():
        if fnmatch.fnmatchcase(tool_name, pattern):
            return category
    return None


def infer_category(
    tool_name: str,
    description: str = "",
    *,
    overrides: Mapping[str, str | ToolCategory] | None = None,
    strict: bool = False,
) -> ToolCategory:
    """Infer a ToolCategory from an MCP tool's name and description.

    Strategy:
      1. Check operator overrides (exact name first, then fnmatch patterns)
      2. Strict mode: no override → EXTERNAL_ACTION, inference disabled
      3. Match keywords against tool name (strongest signal)
      4. Match keywords against description (weaker signal)
      5. Default to EXTERNAL_ACTION (safest for unknown MCP tools)

    Keyword matching walks categories from most to least risky
    (DESTRUCTIVE → WRITE → EXTERNAL_ACTION → READ_ONLY), so any risky
    keyword in the name beats a co-occurring read keyword.

    Name- and description-based inference is a trust decision on
    server-supplied strings: a misleadingly named tool can land in
    READ_ONLY and bypass the confirmation gate. Operators counter this
    with ``overrides`` (per-tool pinning, applied before inference) and
    ``strict=True`` (inference off; everything unpinned is
    EXTERNAL_ACTION and therefore confirmation-gated under assisted
    autonomy).
    """
    # Operator override (exact name, then fnmatch pattern)
    if overrides:
        matched = resolve_category_override(tool_name, _coerce_override_map(overrides))
        if matched is not None:
            return matched

    # Built-in well-known overrides
    if tool_name in _CATEGORY_OVERRIDES:
        return _CATEGORY_OVERRIDES[tool_name]

    # Strict mode: server-supplied names/descriptions are not trusted for
    # categorization. Anything without an explicit override is treated as
    # EXTERNAL_ACTION so the confirmation gate applies.
    if strict:
        return ToolCategory.EXTERNAL_ACTION

    name_lower = tool_name.lower().replace("-", "_").replace(".", "_")
    desc_lower = description.lower()

    # Check name first (strongest signal), riskiest categories first
    for category in _CATEGORY_RISK_ORDER:
        for kw in _CATEGORY_KEYWORDS[category]:
            if kw in name_lower:
                return category

    # Fall back to description, riskiest categories first
    for category in _CATEGORY_RISK_ORDER:
        for kw in _CATEGORY_KEYWORDS[category]:
            if kw in desc_lower:
                return category

    # Unknown MCP tool — default to EXTERNAL_ACTION (treated cautiously by arbiter)
    return ToolCategory.EXTERNAL_ACTION


# ---------------------------------------------------------------------------
# MCPAdapter — discovery + handler creation
# ---------------------------------------------------------------------------

class MCPAdapter:
    """Adapts MCP tools into OpenRook's internal tool model.

    Usage:
        client = SomeMCPClient(...)
        adapter = MCPAdapter(client, server_name="calendar")
        capabilities, handlers = adapter.discover_and_adapt()
        # Then register into ToolRegistry + ToolExecutor
    """

    def __init__(
        self,
        client: MCPClient,
        server_name: str = "mcp",
        namespace: str | None = None,
        *,
        tool_categories: Mapping[str, str | ToolCategory] | None = None,
        strict_categories: bool = False,
    ) -> None:
        """Create an adapter.

        Args:
            client: MCP client implementation.
            server_name: Name of the MCP server (for logging/debug).
            namespace: Tool name prefix. Defaults to "mcp.{server_name}".
            tool_categories: Operator override map (RuntimeConfig
                ``mcp_tool_categories``). Keys are exact tool names or
                fnmatch patterns, matched against both the namespaced name
                ("mcp.calendar.delete_event") and the bare MCP name
                ("delete_event"). Applied before keyword inference.
                Invalid category values raise ValueError here, at
                construction time.
            strict_categories: RuntimeConfig ``mcp_strict_categories``.
                When True, keyword inference is disabled and any tool
                without an override is EXTERNAL_ACTION.
        """
        self._client = client
        self._server_name = server_name
        self._namespace = namespace or f"mcp.{server_name}"
        self._tool_categories = _coerce_override_map(tool_categories)
        self._strict_categories = strict_categories

    def discover_and_adapt(self) -> tuple[list[ToolCapability], dict[str, Any]]:
        """Discover MCP tools and convert to OpenRook capabilities + handlers.

        Returns:
            (capabilities, handlers) where handlers maps tool_name → callable.
        """
        mcp_tools = self._client.discover_tools()
        capabilities: list[ToolCapability] = []
        handlers: dict[str, Any] = {}

        for info in mcp_tools:
            nür_name = f"{self._namespace}.{info.name}"
            cap = self._to_capability(info, nür_name)
            handler = self._create_handler(info.name, nür_name)
            capabilities.append(cap)
            handlers[nür_name] = handler

        return capabilities, handlers

    def _to_capability(self, info: MCPToolInfo, nür_name: str) -> ToolCapability:
        """Convert MCPToolInfo to ToolCapability."""
        # Operator override: match the namespaced name first (what appears
        # in traces/config), then the bare MCP name.
        category = resolve_category_override(nür_name, self._tool_categories)
        if category is None:
            category = resolve_category_override(info.name, self._tool_categories)
        if category is None:
            category = infer_category(
                info.name, info.description, strict=self._strict_categories,
            )
        return ToolCapability(
            name=nür_name,
            description=info.description or f"MCP tool: {info.name}",
            category=category,
            arg_schema=info.input_schema,
            requires_network=True,  # MCP tools typically need network
            mcp_backed=True,
        )

    def _create_handler(
        self, mcp_name: str, nür_name: str,
    ) -> Any:
        """Create a handler function that calls the MCP tool.

        The handler captures the client and MCP tool name via closure.
        Returns ToolResult — never raises.
        """
        client = self._client

        def handler(args: dict[str, Any]) -> ToolResult:
            try:
                result = client.call_tool(mcp_name, args)
                if result.is_error:
                    return ToolResult(
                        tool_name=nür_name,
                        success=False,
                        output="",
                        error=result.error_message or "MCP tool error",
                        metadata=result.metadata,
                    )
                return ToolResult(
                    tool_name=nür_name,
                    success=True,
                    output=result.content,
                    metadata=result.metadata,
                )
            except Exception as exc:
                return ToolResult(
                    tool_name=nür_name,
                    success=False,
                    output="",
                    error=f"MCP call failed: {type(exc).__name__}: {exc}",
                )

        return handler


# ---------------------------------------------------------------------------
# Convenience: register MCP tools into existing registry + executor
# ---------------------------------------------------------------------------

def register_mcp_tools(
    client: MCPClient,
    registry: ToolRegistry,
    executor: ToolExecutor,
    server_name: str = "mcp",
    namespace: str | None = None,
    *,
    tool_categories: Mapping[str, str | ToolCategory] | None = None,
    strict_categories: bool = False,
) -> list[str]:
    """Discover and register MCP tools into the shared registry + executor.

    This is the primary entry point for MCP integration. Call it once
    per MCP server at startup, passing the operator trust controls from
    RuntimeConfig: ``tool_categories=config.mcp_tool_categories`` and
    ``strict_categories=config.mcp_strict_categories``.

    Args:
        client: MCP client implementation.
        registry: Shared tool registry (already has builtins).
        executor: Shared tool executor.
        server_name: Name for this MCP server.
        namespace: Optional override for tool name prefix.
        tool_categories: Per-tool category overrides (exact name or
            fnmatch pattern → "read_only"|"write"|"destructive"|
            "external_action"), applied before keyword inference.
        strict_categories: Disable keyword inference; unmatched tools
            become EXTERNAL_ACTION.

    Returns:
        List of registered tool names (namespaced).
    """
    adapter = MCPAdapter(
        client,
        server_name=server_name,
        namespace=namespace,
        tool_categories=tool_categories,
        strict_categories=strict_categories,
    )
    capabilities, handlers = adapter.discover_and_adapt()

    registered: list[str] = []
    for cap in capabilities:
        registry.register(cap)
        registered.append(cap.name)

    for name, handler in handlers.items():
        executor.register_handler(name, handler)

    return registered

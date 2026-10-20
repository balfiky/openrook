"""Tool executor — thin orchestrator for tool execution.

Looks up a tool handler by name and normalizes ordinary failures into
structured ToolResult values. Research resource-control signals propagate.
"""

from __future__ import annotations

from core.clock import clock as time
from typing import Any, Callable

from core.types import ToolResult
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from nur_tools.registry import ToolRegistry

# Handler signature: takes arguments dict, returns ToolResult
ToolHandler = Callable[[dict[str, Any]], ToolResult]


class ToolExecutor:
    """Executes tools by name using registered handlers.

    The executor is intentionally thin: lookup, call, normalize.
    Cognitive decisions about *whether* to execute live elsewhere.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._handlers: dict[str, ToolHandler] = {}
        # Optional collaborators wired by the runtime after construction.
        self._tool_runner: Any | None = None
        self._self_action_context: Any | None = None
        self._owned_resources: list[Any] = []
        # Secret values scrubbed from normalized error text before it can
        # reach tool traces or the UI (wired from RuntimeConfig secrets).
        self._error_secrets: tuple[str, ...] = ()
        self._observer = None
        self._completion_verifier: Callable[..., dict[str, Any]] | None = None
        self._action_reconciler: Callable[..., dict[str, Any]] | None = None
        self._resource_guard: Callable[[str, dict], str | None] | None = None

    def set_resource_guard(self, guard: Callable[[str, dict], str | None]) -> None:
        self._resource_guard = guard

    def set_action_reconciler(self, reconciler: Callable[..., dict[str, Any]]) -> None:
        self._action_reconciler = reconciler

    def reconcile_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._action_reconciler is None:
            return {"status": "unknown", "method": "no_independent_reconciler"}
        return self._action_reconciler(action)

    def set_completion_verifier(self, verifier: Callable[..., dict[str, Any]]) -> None:
        self._completion_verifier = verifier

    def verify_completion(
        self, criteria: dict[str, Any], evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self._completion_verifier is None:
            return {"status": "unverified", "method": "no_independent_verifier"}
        return self._completion_verifier(criteria, evidence)

    def set_observer(self, observer: Callable[..., Any] | None) -> None:
        self._observer = observer

    def set_error_secrets(self, secrets: list[str] | tuple[str, ...]) -> None:
        """Register secret values to redact from tool error messages."""
        self._error_secrets = tuple(s for s in secrets if s)

    def _redact(self, text: str) -> str:
        for secret in self._error_secrets:
            if secret in text:
                text = text.replace(secret, "[redacted]")
        return text

    @property
    def registry(self) -> ToolRegistry:
        """Read-only access to the tool registry backing this executor."""
        return self._registry

    @property
    def tool_runner(self) -> Any | None:
        """Native LLM tool-call runner, if one was wired by the runtime."""
        return self._tool_runner

    def set_tool_runner(self, runner: Any | None) -> None:
        """Attach a native tool-call runner (e.g. NativeToolCallRunner)."""
        self._tool_runner = runner

    @property
    def self_action_context(self) -> Any | None:
        """Context object consumed by self.* tool handlers, if set."""
        return self._self_action_context

    def set_self_action_context(self, context: Any | None) -> None:
        """Attach the context object consumed by self.* tool handlers."""
        self._self_action_context = context

    @property
    def owned_resources(self) -> list[Any]:
        """Resources whose lifecycle this executor owns (closed on shutdown)."""
        return self._owned_resources

    def own_resource(self, resource: Any) -> None:
        """Register a resource to be closed when the executor's owner shuts down."""
        self._owned_resources.append(resource)

    def register_handler(self, tool_name: str, handler: ToolHandler) -> None:
        """Bind an executable handler to a registered tool name."""
        self._handlers[tool_name] = handler

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        action_id = self._observer("started", tool_name, arguments) if self._observer else None
        resource_stop = self._resource_guard(tool_name, arguments) if self._resource_guard else None
        result = ToolResult(tool_name=tool_name, success=False, output="", error=resource_stop, metadata={"resource_stop": True}) if resource_stop else self._execute(tool_name, arguments)
        if self._observer:
            self._observer("completed", tool_name, arguments, result, action_id)
        return result

    def _execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Returns a structured ToolResult except for research resource-control signals.
        """
        start = time.perf_counter()

        # Check tool exists in registry
        capability = self._registry.get(tool_name)
        if capability is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
                latency_ms=_elapsed_ms(start),
            )

        # Check handler exists
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"No handler registered for tool: {tool_name}",
                latency_ms=_elapsed_ms(start),
            )

        # Execute and normalize
        try:
            result = handler(arguments)
            result.latency_ms = _elapsed_ms(start)
            return result
        except (ResearchBudgetExceeded, ResearchStudyLimit):
            raise
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=self._redact(f"{type(exc).__name__}: {exc}"),
                latency_ms=_elapsed_ms(start),
            )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000

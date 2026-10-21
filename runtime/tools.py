"""Runtime tool wiring for production surfaces.

External agentic tools are opt-in via ``RuntimeConfig.tools_enabled``. When
disabled (the default), :func:`create_tool_executor` still returns a restricted
executor containing only OpenRook's internal skill-registry tools, so the assistant
can create/audit/enable durable skills through the observable tool trace.
Filesystem, browser, web, calendar, and shell capabilities remain excluded
until tools are enabled; ``shell.run_command`` also requires the separate
``shell_tool_enabled`` flag.
"""

from __future__ import annotations

import os
from typing import Optional

from runtime.config import RuntimeConfig
from openrook_tools import register_builtins, register_skill_builtins
from openrook_tools.executor import ToolExecutor
from openrook_tools.registry import ToolRegistry
from openrook_tools.builtin.web_provider import RequestsWebProvider
from openrook_tools.native_orchestrator import NativeToolCallRunner


def create_tool_executor(
    config: RuntimeConfig | None = None,
) -> Optional[ToolExecutor]:
    """Create a per-pipeline tool executor.

    Preserves the zero-arg signature for back-compat with tests that wire the
    factory directly; those tests receive a fully-enabled executor with
    filesystem tools sandboxed to the current working directory. Production
    callers pass a ``RuntimeConfig`` so external tools honor the
    ``tools_enabled`` / ``shell_tool_enabled`` / ``tools_workspace`` gates.
    Unsandboxed filesystem handlers are never wired implicitly; they require
    an explicit opt-in (importing ``openrook_tools.builtin.filesystem.HANDLERS``
    or calling ``filesystem.create_handlers(workspace=None)`` directly).
    """
    # Legacy zero-arg call path (tests/dev wiring). Every builtin registered,
    # filesystem sandboxed to cwd so this path can never grant whole-host
    # file access by accident.
    if config is None:
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        web_provider = RequestsWebProvider()
        register_builtins(
            registry,
            executor,
            web_provider=web_provider,
            fs_workspace=os.getcwd(),
        )
        executor.own_resource(web_provider)
        return executor

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    executor.set_error_secrets(
        [config.api_key, config.effective_llm_api_key, config.telegram_token]
    )

    if not config.tools_enabled:
        register_skill_builtins(registry, executor, skill_config=config)
        return executor

    web_provider = RequestsWebProvider()
    register_builtins(
        registry,
        executor,
        web_provider=web_provider,
        fs_workspace=config.resolved_tools_workspace,
        include_shell=config.shell_tool_enabled,
        shell_command_allowlist=config.shell_command_allowlist,
        skill_config=config,
    )
    executor.own_resource(web_provider)
    if (
        config.llm_backend != "codex"
        and config.llm_base_url.strip()
        and config.llm_model.strip()
    ):
        runner = NativeToolCallRunner(
            executor=executor,
            base_url=config.llm_base_url,
            model=config.llm_model,
            api_key=config.effective_llm_api_key,
            timeout=config.llm_request_timeout_seconds,
        )
        executor.set_tool_runner(runner)
        executor.own_resource(runner)
    return executor

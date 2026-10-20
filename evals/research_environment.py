"""A declared local environment using production tool handlers."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from openrook_tools.builtin import filesystem, self_actions
from openrook_tools.executor import ToolExecutor
from openrook_tools.native_orchestrator import NativeToolCallRunner
from openrook_tools.registry import ToolRegistry
from runtime.config import RuntimeConfig


PROFILE = "local_workspace_v1"
LOCAL_TOOLS = {
    "fs.read_file",
    "fs.list_dir",
    "fs.search_text",
    "fs.glob_paths",
    "fs.write_file",
    "self.snapshot_state",
    "self.log_event",
    "self.note",
}


def validate_environment(environment: dict[str, Any] | None) -> None:
    if environment is None:
        return
    if isinstance(environment, dict) and environment.get("profile") == "development_world_v1":
        from evals.development_environment import validate_development_environment

        validate_development_environment(environment)
        return
    if isinstance(environment, dict) and environment.get("profile") == "selectable_learning_v1":
        from evals.learning_environment import validate_learning_environment

        validate_learning_environment(environment)
        return
    if not isinstance(environment, dict) or environment.get("profile") != PROFILE:
        raise ValueError("Unknown research environment profile")
    if set(environment) - {"profile", "files"}:
        raise ValueError("Unknown research environment fields")
    files = environment.get("files", {})
    if not isinstance(files, dict) or not 1 <= len(files) <= 30:
        raise ValueError("Research environments require 1 to 30 seed files")
    for name, content in files.items():
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("Seed file paths must be relative POSIX paths")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or name == ".":
            raise ValueError("Seed file paths must stay inside the workspace")
        if not isinstance(content, str) or len(content) > 100_000:
            raise ValueError("Seed files must contain at most 100000 text characters")


def seed_environment(config: RuntimeConfig, environment: dict[str, Any]) -> dict[str, Any]:
    validate_environment(environment)
    if environment["profile"] == "development_world_v1":
        from evals.development_environment import seed_development_environment

        return seed_development_environment(config, environment)
    if environment["profile"] == "selectable_learning_v1":
        from evals.learning_environment import seed_learning_environment

        return seed_learning_environment(config, environment)
    workspace = Path(config.resolved_tools_workspace).resolve()
    if not workspace.is_relative_to(Path(config.data_dir).resolve()):
        raise ValueError("Research environments require a private subject workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, content in environment["files"].items():
        target = (workspace / name).resolve()
        if not target.is_relative_to(workspace):
            raise ValueError("Seed file resolves outside the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        if target.exists() and target.read_bytes() != encoded:
            raise ValueError("Refusing to replace an existing subject resource")
        target.write_bytes(encoded)
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    return {"profile": PROFILE, "workspace": str(workspace), "file_sha256": hashes}


def create_environment_executor(config: RuntimeConfig, environment: dict[str, Any]) -> ToolExecutor:
    validate_environment(environment)
    if environment["profile"] == "development_world_v1":
        from evals.development_environment import create_development_executor

        return create_development_executor(config)
    if environment["profile"] == "selectable_learning_v1":
        from evals.learning_environment import create_learning_executor

        return create_learning_executor(config)
    return create_local_executor(config)


def create_local_executor(config: RuntimeConfig) -> ToolExecutor:
    if not config.research_mode or not config.tools_enabled:
        raise ValueError("The local research environment requires enabled research tools")
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    executor.set_error_secrets(
        [config.api_key, config.effective_llm_api_key, config.telegram_token]
    )
    handlers = filesystem.create_handlers(workspace=config.resolved_tools_workspace)
    handlers.update(self_actions.create_handlers(get_context=lambda: executor.self_action_context))
    hint = (
        f" Workspace root: {config.resolved_tools_workspace}. Relative paths resolve there."
        " WORLD.md describes the local environment. Runtime databases and private journals"
        " are outside filesystem scope; self notes are supplied in agent-owned context."
    )
    for capability in filesystem.CAPABILITIES + self_actions.CAPABILITIES:
        if capability.name in LOCAL_TOOLS:
            registry.register(replace(capability, description=capability.description + hint))
            executor.register_handler(capability.name, handlers[capability.name])
    if config.llm_base_url and config.llm_model:
        runner = NativeToolCallRunner(
            executor=executor,
            base_url=config.llm_base_url,
            model=config.llm_model,
            api_key=config.effective_llm_api_key,
        )
        executor.set_tool_runner(runner)
        executor.own_resource(runner)
    return executor

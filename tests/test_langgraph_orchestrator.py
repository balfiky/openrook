"""Tests for LangGraph-backed tool orchestration."""

from __future__ import annotations

import json

import pytest

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from core.emotional_engine import EmotionalEngine
from core.life_influence import LifeInfluence
from core.types import ToolCapability, ToolCategory, ToolResult
from openrook_tools import register_builtins
from openrook_tools.executor import ToolExecutor
from openrook_tools.langgraph_orchestrator import LangGraphToolRunner
from openrook_tools.registry import ToolRegistry


def _executor() -> ToolExecutor:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    register_builtins(registry, executor)
    return executor


def _fake_shell_executor() -> ToolExecutor:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    registry.register(
        ToolCapability(
            name="shell.run_command",
            description="Run a shell command and capture output",
            category=ToolCategory.DESTRUCTIVE,
            arg_schema={"cmd": {"type": "string", "required": True}},
        )
    )

    def run_command(args: dict[str, object]) -> ToolResult:
        cmd = str(args["cmd"])
        return ToolResult(
            tool_name="shell.run_command",
            success=True,
            output=f"ran {cmd}",
            metadata={"cmd": cmd},
            side_effect_summary="command executed",
        )

    executor.register_handler("shell.run_command", run_command)
    return executor


def test_langgraph_shell_call_returns_tool_context() -> None:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell__run_command",
                        "args": {"cmd": "printf langgraph-visible"},
                        "id": "call_1",
                    },
                ],
            ),
            AIMessage(content="done"),
        ],
    )
    executor = _executor()
    runner = LangGraphToolRunner(
        executor=executor,
        base_url="http://localhost:8000/v1",
        model="test",
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message="check the disk space",
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        autonomy_level="high_risk",
    )

    assert result.trace.loop_count == 1
    assert result.trace.proposed_intents[0].tool_name == "shell.run_command"
    assert result.trace.executed_results[0].tool_name == "shell.run_command"
    assert "output='langgraph-visible'" in result.tool_context_summary


def test_langgraph_no_tool_call_returns_empty_trace_without_fallback() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="no tool needed")])
    runner = LangGraphToolRunner(
        executor=_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message="hello",
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
    )

    assert result.trace.loop_count == 0
    assert result.tool_context_summary == ""


def test_langgraph_runner_accepts_life_influence_and_records_action_effect() -> None:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "system__hostname",
                        "args": {},
                        "id": "call_1",
                    },
                ],
            ),
            AIMessage(content="done"),
        ],
    )
    runner = LangGraphToolRunner(
        executor=_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message="what is your hostname?",
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        autonomy_level="high_risk",
        life_influence=LifeInfluence(competence_pressure=0.04),
    )

    assert result.trace.loop_count == 1
    assert result.life_influence_effects["persistence_delta"] == 0.04
    assert result.action_variables.persistence_drive > 0.5


def test_hybrid_falls_back_to_heuristic_when_model_skips_tool() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="df -h /")])
    runner = LangGraphToolRunner(
        executor=_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        fallback_to_heuristic=True,
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message="how much disk space left",
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        autonomy_level="high_risk",
    )

    assert result.trace.loop_count == 1
    assert result.trace.executed_results[0].metadata["cmd"] == "df -h /"
    assert "Filesystem" in result.tool_context_summary


def test_structured_router_handles_unseen_disk_phrasing() -> None:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="No tool needed."),
            AIMessage(
                content=json.dumps(
                    {
                        "tool_name": "shell.run_command",
                        "arguments": {"cmd": "df -h /"},
                        "confidence": 0.91,
                        "rationale": "The user asks for live disk pressure.",
                    }
                )
            ),
        ]
    )
    runner = LangGraphToolRunner(
        executor=_fake_shell_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        fallback_to_heuristic=True,
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message="what is the root partition pressure on this box?",
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        autonomy_level="high_risk",
    )

    assert result.trace.loop_count == 1
    assert result.trace.proposed_intents[0].reason == "The user asks for live disk pressure."
    assert result.trace.executed_results[0].metadata["cmd"] == "df -h /"
    assert "ran df -h /" in result.tool_context_summary


def test_structured_router_uses_history_for_followup_reference() -> None:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="No tool needed."),
            AIMessage(
                content=json.dumps(
                    {
                        "tool_name": "shell.run_command",
                        "arguments": {"cmd": "df -h /"},
                        "confidence": 0.88,
                        "rationale": "The current message refers to the prior disk request.",
                    }
                )
            ),
        ]
    )
    runner = LangGraphToolRunner(
        executor=_fake_shell_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        fallback_to_heuristic=True,
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message=(
            "Recent conversation for tool routing:\n"
            "User: what is the harddisk utilzation?\n"
            "Assistant: Run df -h.\n"
            "Current user message: no, execute it on your pc"
        ),
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        autonomy_level="high_risk",
    )

    assert result.trace.loop_count == 1
    assert result.trace.executed_results[0].metadata["cmd"] == "df -h /"


def test_structured_router_handles_confirmation_after_pending_tool_request() -> None:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="No tool needed."),
            AIMessage(
                content=json.dumps(
                    {
                        "tool_name": "shell.run_command",
                        "arguments": {"cmd": "df -h /"},
                        "confidence": 0.86,
                        "rationale": "The current confirmation continues the disk-space request.",
                    }
                )
            ),
        ]
    )
    runner = LangGraphToolRunner(
        executor=_fake_shell_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        fallback_to_heuristic=True,
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message=(
            "Recent conversation for tool routing:\n"
            "User: great...give me your remaining disk space\n"
            "Assistant: I'll get it. Hang on.\n"
            "Current user message: yes of course"
        ),
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        autonomy_level="high_risk",
    )

    assert result.trace.loop_count == 1
    assert result.trace.executed_results[0].metadata["cmd"] == "df -h /"


def test_hybrid_falls_back_for_explicit_unquoted_shell_command() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="Done.")])
    runner = LangGraphToolRunner(
        executor=_fake_shell_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        fallback_to_heuristic=True,
        chat_model=model,
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message="do arbitrary-tool --flag",
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        autonomy_level="high_risk",
    )

    assert result.trace.loop_count == 1
    assert result.trace.executed_results[0].metadata["cmd"] == "arbitrary-tool --flag"
    assert "ran arbitrary-tool --flag" in result.tool_context_summary


# ---------------------------------------------------------------------------
# Optional-extra import behavior: the module must import without langgraph
# installed, and the runner constructor must fail with install guidance.
# ---------------------------------------------------------------------------


def test_module_imports_without_langgraph(monkeypatch) -> None:
    """Simulate langgraph absence: module import succeeds, runner fails loud."""
    import builtins
    import importlib
    import sys

    import openrook_tools.langgraph_orchestrator as lg_mod

    blocked_prefixes = ("langgraph", "langchain_core", "langchain_openai")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith(blocked_prefixes):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for mod_name in list(sys.modules):
        if mod_name.startswith(blocked_prefixes):
            monkeypatch.delitem(sys.modules, mod_name)
    monkeypatch.delitem(sys.modules, "openrook_tools.langgraph_orchestrator")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        # 1. Module import must not crash when the extra is absent.
        reloaded = importlib.import_module("openrook_tools.langgraph_orchestrator")
        assert reloaded._LANGGRAPH_IMPORT_ERROR is not None

        # 2. Constructing the runner raises a clear, actionable ImportError.
        with pytest.raises(ImportError, match=r"openrook\[langgraph\]"):
            reloaded.LangGraphToolRunner(
                executor=_executor(),
                base_url="http://localhost:8000/v1",
                model="test",
            )
    finally:
        # Restore the real module for the rest of the suite.
        monkeypatch.undo()
        sys.modules["openrook_tools.langgraph_orchestrator"] = lg_mod
        importlib.reload(lg_mod)


def test_langgraph_runner_gates_plan_continuation_under_assisted(tmp_path) -> None:
    """Round-7 regression: the LangGraph runner must gate each plan step on
    continuation, not launder a write step past assisted autonomy.
    """
    from core.task_planning import TaskPlan, TaskStep
    from core.types import TaskStatus

    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    steps = [
        TaskStep(id="s1", tool_name="fs.list_dir",
                 arguments={"path": str(tmp_path)}, description="list (read-only)"),
        TaskStep(id="s2", tool_name="fs.write_file",
                 arguments={"path": str(victim), "content": "clobbered"},
                 description="overwrite (write)"),
    ]
    plan = TaskPlan(id="p1", goal="list then overwrite", steps=steps,
                    status=TaskStatus.IN_PROGRESS)

    # No model call happens on plan continuation, so an empty fake is fine.
    runner = LangGraphToolRunner(
        executor=_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        chat_model=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
    )
    engine = EmotionalEngine()

    result = runner.run_tool_loop(
        user_message="continue",
        state=engine.state,
        person=None,
        defense_active=False,
        engine=engine,
        active_plan=plan,
        autonomy_level="assisted",
    )

    assert result.trace.task_trace is not None
    assert result.trace.task_trace.plan_outcome == "blocked"
    assert steps[1].status == TaskStatus.BLOCKED
    assert victim.read_text() == "precious"

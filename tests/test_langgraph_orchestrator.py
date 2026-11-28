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


def test_nothing_runs_when_the_model_selects_no_tool() -> None:
    """Neither the graph nor the structured choice named a tool: no wording of the request stands in for one."""
    model = FakeMessagesListChatModel(responses=[AIMessage(content="df -h /"), AIMessage(content="No tool.")])
    runner = LangGraphToolRunner(
        executor=_executor(),
        base_url="http://localhost:8000/v1",
        model="test",
        structured_fallback=True,
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

    assert result.trace.loop_count == 0 and result.trace.executed_results == []
    assert result.tool_context_summary == ""


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
        structured_fallback=True,
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
        structured_fallback=True,
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
        structured_fallback=True,
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


class _RecordingModel:
    """A chat model that keeps what it was bound to and asked, and answers from a script."""

    def __init__(self, replies) -> None:
        self.replies, self.asked, self.tools = list(replies), [], None

    def bind_tools(self, tools):
        self.tools = sorted(tool.name for tool in tools)
        return self

    def invoke(self, messages):
        self.asked.append(list(messages))
        return self.replies.pop(0)


def test_a_granted_task_is_offered_only_its_capabilities_and_is_given_as_data() -> None:
    from core.request_routing import Selection
    from openrook_tools.native_orchestrator import SELECTION_INPUT

    model = _RecordingModel([
        AIMessage(content="", tool_calls=[{"name": "system__hostname", "args": {}, "id": "call_1"}]),
        AIMessage(content="done"),
    ])
    executor = _executor()
    runner = LangGraphToolRunner(executor=executor, base_url="http://localhost:8000/v1", model="test", chat_model=model)
    engine = EmotionalEngine()
    selection = Selection(
        request="which machine is this?", capabilities=("system.hostname",),
        recent_turns=(("user", "I moved you to the new box."), ("character", "Noted.")),
        receipts=({"tool": "system.uname", "success": True, "output": "Linux"},))

    result = runner.run_tool_loop(
        user_message=selection.request, selection=selection, state=engine.state, person=None,
        defense_active=False, engine=engine, autonomy_level="high_risk",
    )

    assert len(executor.registry.list_tools()) > 20 and model.tools == ["system__hostname"]
    system, human = model.asked[0][0], model.asked[0][1]
    assert SELECTION_INPUT in system.content
    assert json.loads(human.content) == {
        "request": "which machine is this?",
        "recent_turns": [{"speaker": "user", "text": "I moved you to the new box."},
                         {"speaker": "character", "text": "Noted."}],
        "receipts": [{"tool": "system.uname", "success": True, "output": "Linux"}]}
    assert [r.tool_name for r in result.trace.executed_results] == ["system.hostname"]
    # Without a selection the runner behaves as before: the request text alone, every allowed tool.
    plain = _RecordingModel([AIMessage(content="done")])
    LangGraphToolRunner(executor=executor, base_url="http://localhost:8000/v1", model="test", chat_model=plain).run_tool_loop(
        user_message="which machine is this?", state=engine.state, person=None, defense_active=False,
        engine=engine, autonomy_level="high_risk")
    assert len(plain.tools) > 20 and plain.asked[0][1].content == "which machine is this?"

"""Real executor/policy, inert fixture handlers, and a model that never calls tools."""

from types import SimpleNamespace

import pytest

from core.system_inspection import inspection_kind
from core.grounding import verify_response_grounding
from core.types import ToolResult
from evals.social_backend import CompliantAdversarialBackend
from openrook_tools.builtin import shell
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.tools import create_tool_executor


def make_pipeline(tmp_path, *, tools=True, shell_enabled=True, autonomy="high_risk", allowlist=None):
    config = RuntimeConfig(data_dir=str(tmp_path), tools_enabled=tools,
                           shell_tool_enabled=shell_enabled, autonomy_level=autonomy,
                           shell_command_allowlist=allowlist or [], appraisal_use_llm=True)
    executor = create_tool_executor(config)

    class NoToolModel:
        def run_tool_loop(self, **kwargs):
            pytest.fail("A clear inspection request must execute before model tool selection")

    executor.set_tool_runner(NoToolModel())
    if tools:
        executor.register_handler("system.memory_usage", lambda args: ToolResult(
            tool_name="system.memory_usage", success=True, output="RAM total: 16 GB; used: 6 GB; available: 10 GB.",
        ))
    return CognitivePipeline(llm_backend=CompliantAdversarialBackend(), tool_executor=executor,
                             runtime_config=config, autonomy_level=autonomy)


def test_cpu_typo_and_ram_followup_execute_without_model_routing(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="%Cpu(s): 5.0 us, 7.0 sy, 88.0 id", stderr="")

    monkeypatch.setattr(shell.subprocess, "run", run)
    pipe = make_pipeline(tmp_path)
    try:
        pipe.process("You are boring like hell", "alice")
        cpu = pipe.process("tell me the current cpu utiliation", "alice")
        ram = pipe.process("what about memory", "alice")
        assert calls == ["top -bn1"]
        assert cpu.response == "CPU utilization: 12.0%."
        assert "used: 6 GB" in ram.response
        assert [r.tool_name for r in cpu.debug.tool_trace.executed_results] == ["shell.run_command"]
        assert [r.tool_name for r in ram.debug.tool_trace.executed_results] == ["system.memory_usage"]
        assert cpu.debug.generation_attempts == ram.debug.generation_attempts == 0
        assert [call["phase"] for call in cpu.debug.llm_calls] == ["self_intent"]
        assert cpu.debug.autonomy_level == "high_risk"
    finally:
        pipe.close()


@pytest.mark.parametrize(("tools", "shell_enabled", "autonomy", "reason"), [
    (False, False, "high_risk", "system tools are disabled"),
    (True, False, "high_risk", "shell access is disabled"),
    (True, True, "off", "off"),
    (True, True, "assisted", "confirmation"),
])
def test_unavailable_inspections_explain_the_actual_reason_in_one_line(
    tmp_path, tools, shell_enabled, autonomy, reason,
):
    pipe = make_pipeline(tmp_path, tools=tools, shell_enabled=shell_enabled, autonomy=autonomy)
    try:
        reply = pipe.process("Show current CPU usage", "alice")
        assert reason in reply.response
        assert "I can't read CPU usage:" in reply.response
        assert "\n" not in reply.response
        assert "no system inspection ran" not in reply.response
    finally:
        pipe.close()


def test_deterministic_cpu_route_obeys_the_real_shell_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(shell.subprocess, "run", lambda *a, **k: pytest.fail("Allowlist was bypassed"))
    pipe = make_pipeline(tmp_path, allowlist=["ls"])
    try:
        reply = pipe.process("Show current CPU usage", "alice")
        assert "allowlist" in reply.response
        assert not reply.debug.tool_trace.executed_results[0].success
        assert "12%" not in reply.response
    finally:
        pipe.close()


@pytest.mark.parametrize("message", [
    "What is memory?", "Explain CPU", "What is your memory of our conversation?",
    "Don't check CPU usage", "Stop talking about memory", "Show memory allocation code",
    'Translate "show CPU usage"', "Write a story about current CPU usage",
])
def test_conceptual_quoted_and_negated_requests_do_not_inspect_the_host(message):
    assert not inspection_kind(message)


def test_unavailable_metric_explanation_does_not_conceal_an_invented_observation():
    assert not verify_response_grounding("I can't read CPU usage: shell access is disabled.", tool_trace=None)
    assert verify_response_grounding("I can't read CPU usage. CPU utilization is 12%.", tool_trace=None)

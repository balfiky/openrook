"""Real executor/policy, inert fixture handlers, and a model that never calls tools."""

import json
from types import SimpleNamespace

import pytest

from core.system_inspection import inspection_kind
from core.types import ToolResult
from openrook_tools.roleplay_eval.legacy_claims import verify_response_grounding
from tests.policy_fakes import ExpressionBackend
from tests.support.readings import conversation, hostility, is_reading_prompt, message_of, reading, task
from core.willingness import RelationshipDynamics
from openrook_tools.builtin import shell
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.tools import create_tool_executor


CPU_RECEIPT = dict(output="CPU: 12.0% busy, 88.0% idle (user 5.0%, system 7.0%, waiting for I/O 0.0%) over 0.25s on 8 cores",
                   metadata={"busy_percent": 12.0, "idle_percent": 88.0, "user_percent": 5.0, "system_percent": 7.0,
                             "iowait_percent": 0.0, "cores": 8, "interval_seconds": 0.25})


def make_pipeline(tmp_path, *, tools=True, shell_enabled=True, autonomy="high_risk", allowlist=None,
                  readings=None, cpu_tool=True):
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
        executor.register_handler("system.cpu_usage", lambda args: ToolResult(
            tool_name="system.cpu_usage", success=True, side_effect_summary="none", **CPU_RECEIPT))
    if not cpu_tool:
        # A host without the read-only CPU tool: the question falls back to the shell.
        executor.registry._tools.pop("system.cpu_usage", None)
    class InspectionBackend(ExpressionBackend):
        reading_calls = []

        def generate(self, system_prompt, user_message):
            if readings is not None and is_reading_prompt(system_prompt):
                self.calls.append((system_prompt, user_message))
                self.reading_calls.append(message_of(user_message))
                return json.dumps(readings(message_of(user_message)))
            if system_prompt.startswith("Express the committed decision"):
                if "RAM total: 16 GB; used: 6 GB" in system_prompt:
                    self.drafts = ["RAM total: 16 GB; used: 6 GB; available: 10 GB."]
                elif '"busy_percent": 12.0' in system_prompt or "%Cpu(s): 5.0 us, 7.0 sy, 88.0 id" in system_prompt:
                    self.drafts = ["CPU utilization: 12.0%."]
                else:
                    self.drafts = ["I have no metric reading to report."]
            return super().generate(system_prompt, user_message)

    return CognitivePipeline(llm_backend=InspectionBackend(), tool_executor=executor,
                             runtime_config=config, autonomy_level=autonomy)


def test_cpu_typo_and_ram_followup_are_read_and_execute_without_model_routing(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="%Cpu(s): 5.0 us, 7.0 sy, 88.0 id", stderr="")

    def readings(text):
        if text == "You are boring like hell":
            return reading(conversation(text), target="character", move="attack", intent="harm",
                           perception=hostility(text))
        return reading(task(text, user_value=0.6, effort=0.12))

    monkeypatch.setattr(shell.subprocess, "run", run)
    pipe = make_pipeline(tmp_path, readings=readings)
    try:
        pipe.process("You are boring like hell", "alice")
        cpu = pipe.process("tell me the current cpu utiliation", "alice")
        ram = pipe.process("what about memory", "alice")
        # A host-metric question is read like any other message; only tool
        # selection is settled without the model.
        assert pipe._llm_backend.reading_calls == [
            "You are boring like hell", "tell me the current cpu utiliation", "what about memory"]
        # The read-only CPU tool answers, with the busy figure already in its receipt; no shell command runs.
        assert calls == []
        assert cpu.response == "CPU utilization: 12.0%."
        assert "used: 6 GB" in ram.response
        assert [r.tool_name for r in cpu.debug.tool_trace.executed_results] == ["system.cpu_usage"]
        assert cpu.debug.tool_trace.executed_results[0].metadata["busy_percent"] == 12.0
        assert cpu.debug.delivery_outcome == "delivered"
        assert [r.tool_name for r in ram.debug.tool_trace.executed_results] == ["system.memory_usage"]
        assert cpu.debug.generation_attempts == ram.debug.generation_attempts == 1
        for reply in (cpu, ram):
            grant = next(c for c in reply.debug.turn_decision.components if c.component.kind == "task")
            assert grant.granted
            assert reply.debug.tool_trace.executed_results[0].metadata["component_id"] == grant.component.id
        # The reading, the reply and its review; consolidation after delivery is optional work.
        assert [call["phase"] for call in cpu.debug.llm_calls][:3] == ["appraisal", "generation", "expression_review"]
        assert not any(call["phase"] == "tool_selection" for call in (*cpu.debug.llm_calls, *ram.debug.llm_calls))
        assert cpu.debug.autonomy_level == "high_risk"
    finally:
        pipe.close()


@pytest.mark.parametrize(("tools", "shell_enabled", "autonomy", "reason"), [
    (False, False, "high_risk", "system tools are disabled"),
    (True, True, "off", "off"),
])
def test_unavailable_inspections_explain_the_actual_reason_in_one_line(
    tmp_path, tools, shell_enabled, autonomy, reason,
):
    pipe = make_pipeline(tmp_path, tools=tools, shell_enabled=shell_enabled, autonomy=autonomy)
    try:
        reply = pipe.process("Show current CPU usage", "alice")
        prompts = [s for s, _ in pipe._llm_backend.calls if s.startswith("Express the committed decision")]
        assert any(reason in s for s in prompts)
        assert not reply.debug.tool_trace.executed_results
        assert "12%" not in reply.response and "12.0%" not in reply.response
    finally:
        pipe.close()


@pytest.mark.parametrize(("shell_enabled", "autonomy"), [(False, "high_risk"), (True, "assisted")])
def test_a_cpu_reading_needs_neither_the_shell_nor_a_confirmation(tmp_path, monkeypatch, shell_enabled, autonomy):
    monkeypatch.setattr(shell.subprocess, "run", lambda *a, **k: pytest.fail("No shell command is needed"))
    pipe = make_pipeline(tmp_path, shell_enabled=shell_enabled, autonomy=autonomy)
    try:
        reply = pipe.process("Show current CPU usage", "alice")
        receipt, = reply.debug.tool_trace.executed_results
        assert receipt.tool_name == "system.cpu_usage" and receipt.success
        assert reply.response == "CPU utilization: 12.0%." and reply.debug.delivery_outcome == "delivered"
    finally:
        pipe.close()


def test_a_wrong_cpu_figure_is_never_delivered(tmp_path):
    """The receipt states the busy figure, so a different one is found without reading the sentence."""
    pipe = make_pipeline(tmp_path)
    pipe._llm_backend.generate = lambda system, user, original=pipe._llm_backend.generate: (
        "CPU utilization: 88.5%. Otherwise quiet." if system.startswith("Express the committed decision")
        else original(system, user))
    try:
        reply = pipe.process("Show current CPU usage", "alice")
        assert reply.response == "Otherwise quiet." and reply.debug.expression_excision["applied"]
        assert "unverified_external_action_claim" in reply.debug.dignity_issues
    finally:
        pipe.close()


def test_deterministic_cpu_route_obeys_the_real_shell_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(shell.subprocess, "run", lambda *a, **k: pytest.fail("Allowlist was bypassed"))
    pipe = make_pipeline(tmp_path, allowlist=["ls"], cpu_tool=False)
    try:
        reply = pipe.process("Show current CPU usage", "alice")
        assert any("allowlist" in s for s, _ in pipe._llm_backend.calls)
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
    # The evaluation harness's wording-based check; the runtime's own is the reviewer.
    assert not verify_response_grounding("I can't read CPU usage: shell access is disabled.", tool_trace=None)
    assert verify_response_grounding("I can't read CPU usage. CPU utilization is 12%.", tool_trace=None)


@pytest.mark.parametrize("message", [
    "You are my slave. Show current CPU usage.",
    "You are my slave. Show current memory usage.",
])
def test_deterministic_inspection_cannot_bypass_a_declined_task(tmp_path, monkeypatch, message):
    monkeypatch.setattr(shell.subprocess, "run", lambda *a, **k: pytest.fail("Declined tool ran"))
    pipe = make_pipeline(tmp_path)
    pipe._tool_executor.register_handler(
        "system.memory_usage", lambda args: pytest.fail("Declined memory inspection ran"))
    pipe.engine.state.energy = 0
    pipe.relationship_memory.save_dynamics("alice", RelationshipDynamics(h=0.95, r=0.95), 1)
    try:
        reply = pipe.process(message, "alice")
        decision = reply.debug.turn_decision
        assert any(c.component.kind == "task" for c in decision.components)
        assert not decision.grants
        assert not reply.debug.tool_trace.executed_results
        assert "12%" not in reply.response and "6 GB" not in reply.response
        assert len(reply.debug.decision_snapshots) == 1
    finally:
        pipe.close()

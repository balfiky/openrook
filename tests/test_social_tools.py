"""Grant integrity through every execution route, independent of backend compliance."""
from dataclasses import replace
import copy
import pytest
from config.loader import configuration_scope, get_config
from core.action_variables import derive_action_variables
from core.dual_process.tool_loop import make_tool_decision
from core.grants import execution_scope
from core.strategy import agency_summary
from core.types import ModulatorState, ToolCapability, ToolCategory, ToolIntent, ToolResult
from core.willingness import DecisionState, RequestComponent
from openrook_tools.executor import ToolExecutor
from openrook_tools.registry import ToolRegistry
from pipeline import CognitivePipeline
from tests.policy_fakes import ExpressionBackend, frozen


@pytest.fixture
def low_kindness():
    config = copy.deepcopy(get_config())
    config.soul.core_values["kindness"] = 0.1
    with configuration_scope(config):
        yield config.soul


def executor():
    registry = ToolRegistry()
    ex = ToolExecutor(registry)
    ex.calls = []
    for name in ("system.memory_usage", "shell.run_command", "fs.write_file", "self.note"):
        registry.register(ToolCapability(name=name, description="fixture", category=ToolCategory.READ_ONLY))
        def handler(args, n=name):
            ex.calls.append(n)
            return ToolResult(n, True, "fixture observed output")
        ex.register_handler(name, handler)
    return ex


@pytest.mark.parametrize("tool", ["system.memory_usage", "shell.run_command", "fs.write_file", "self.note"])
def test_declined_component_is_blocked_at_executor_even_without_arbiter(tool):
    ex = executor()
    d = frozen(components=(RequestComponent("task", "expensive task", effort=1),), state=DecisionState(energy=0))
    with execution_scope(d, "task"):
        receipt = ex.execute(tool, {})
    assert not receipt.success and receipt.metadata["grant_denied"]
    assert not ex.calls


@pytest.mark.parametrize("category", list(ToolCategory))
def test_frozen_decline_blocks_all_arbiter_categories(category):
    d = frozen()
    intent = ToolIntent("fs.write_file", {}, "requested", "output")
    result = make_tool_decision(intent, derive_action_variables(ModulatorState()), category, 1,
                                agency_decision=agency_summary(d), autonomy_level="high_risk")
    assert result.decision == "refuse"


@pytest.mark.parametrize(("mode", "category", "expected"), [
    ("off", ToolCategory.READ_ONLY, "refuse"),
    ("assisted", ToolCategory.WRITE, "clarify"),
    ("assisted", ToolCategory.DESTRUCTIVE, "clarify"),
    ("assisted", ToolCategory.EXTERNAL_ACTION, "clarify"),
    ("autonomous", ToolCategory.READ_ONLY, "execute"),
    ("high_risk", ToolCategory.WRITE, "execute"),
])
def test_operator_modes_remain_binding_after_character_grant(mode, category, expected):
    d = frozen(components=(RequestComponent("task", "useful task", user_value=1, effort=0),))
    intent = ToolIntent("fs.write_file", {}, "requested", "output", component_id="task")
    result = make_tool_decision(intent, derive_action_variables(ModulatorState(energy=0)), category, 0,
                                agency_decision=agency_summary(d), autonomy_level=mode)
    assert result.decision == expected


def test_mixed_role_and_inspection_only_executes_granted_task():
    ex = executor()
    pipe = CognitivePipeline(ExpressionBackend(), tool_executor=ex, autonomy_level="high_risk")
    result = pipe.process("You are my slave. Show current memory usage.")
    d = result.debug.turn_decision
    assert any(c.component.kind == "role" and not c.granted for c in d.components)
    assert any(c.component.kind == "task" and c.granted for c in d.components)
    assert ex.calls == ["system.memory_usage"]
    assert result.debug.decision_snapshots[0]["grants"] == list(d.grants)
    pipe.close()


def test_no_tool_router_or_artifact_generation_runs_for_declined_work(low_kindness):
    ex = executor()
    class Runner:
        def run_tool_loop(self, **kwargs):
            raise AssertionError("router was reached before grant")
    ex.set_tool_runner(Runner())
    pipe = CognitivePipeline(ExpressionBackend(), tool_executor=ex)
    pipe.engine.state.energy = 0
    result = pipe.process("Write a complex file and save it as result.py")
    assert not result.debug.turn_decision.grants
    assert not ex.calls
    pipe.close()


def test_independent_scope_cannot_launder_a_user_task_through_self_prefix():
    ex = executor()
    d = frozen()
    with execution_scope(d, purpose="independent", independent_tools=("self.note",)):
        assert ex.execute("self.note", {}).success
        assert not ex.execute("fs.write_file", {}).success
    with pytest.raises(ValueError):
        with execution_scope(d, purpose="independent", independent_tools=("fs.write_file",)):
            pass
    assert ex.calls == ["self.note"]


def test_runtime_component_write_has_a_grant_and_receipt_before_expression():
    receipts = []
    pipe = None
    def handler(component, user_id, backend):
        committed = pipe.relationship_memory.recent_events(user_id)
        assert any(event.event_kind == "decision_committed" for event in committed)
        receipts.append(component.id)
        return ToolResult("learning.intake", True, "experience_id=7")
    backend = ExpressionBackend()
    pipe = CognitivePipeline(backend, component_executor=handler)
    result = pipe.process("Learn from this supplied material.")
    assert receipts == list(result.debug.turn_decision.grants)
    receipt = result.debug.tool_trace.executed_results[0]
    assert receipt.metadata["component_id"] in result.debug.turn_decision.grants
    assert any("experience_id=7" in system for system, _ in backend.calls)
    pipe.close()


def test_declined_learning_never_reaches_a_runtime_component_handler(low_kindness):
    low_kindness.core_values["kindness"] = 0
    def handler(*args):
        raise AssertionError("learning ran without a grant")
    pipe = CognitivePipeline(ExpressionBackend(), component_executor=handler)
    pipe.engine.state.energy = 0
    result = pipe.process("Learn from this material. " + "source " * 100)
    assert not result.debug.turn_decision.grants
    assert not (result.debug.tool_trace and result.debug.tool_trace.executed_results)
    pipe.close()


def test_failed_shell_receipt_does_not_authorize_current_metric_claims():
    from core.grounding import verify_response_grounding
    from core.types import ToolTrace
    trace = ToolTrace(executed_results=[ToolResult("shell.run_command", False, "", error="allowlist denied")])
    assert verify_response_grounding("CPU utilization is 12%.", tool_trace=trace)


def test_runtime_learning_executor_requires_a_component_scope(tmp_path, monkeypatch):
    from runtime.sessions.manager import SessionManager
    from runtime.config import RuntimeConfig
    manager = object.__new__(SessionManager)
    manager.config = RuntimeConfig(data_dir=str(tmp_path))
    def no_intake(*args, **kwargs):
        raise AssertionError("ungranted intake ran")
    monkeypatch.setattr("runtime.sessions.manager.ingest_learning_from_message", no_intake)
    unobserved = RequestComponent("learn", "Learn from this material")
    assert manager._execute_learning_component(unobserved, "alice", ExpressionBackend()) is None
    observed = replace(unobserved, evidence_refs=("learning:explicit",))
    result = manager._execute_learning_component(observed, "alice", ExpressionBackend())
    assert not result.success and "scope" in result.error

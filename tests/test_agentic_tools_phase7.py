"""Tests for Agentic Tools Phase 7: Multi-step task planning and task memory."""

from __future__ import annotations




from core.types import (
    ActionVariables,
    ModulatorState,
    PersonProfile,
    TaskPlan,
    TaskStatus,
    TaskStep,
    TaskTrace,
    ToolCategory,
    ToolObservation,
    ToolTrace as ToolTraceType,
)
from core.task_planning import (
    execute_plan,
    summarize_plan_status,
)
from core.tool_memory import (
    create_task_long_term_entry,
    create_task_unresolved_item,
    derive_task_self_observations,
)
from core.dual_process.tool_loop import continue_plan, run_tool_loop
from core.types import ToolIntent
from tests.support.readings import ScriptedReader, reading, task
from openrook_tools.registry import ToolRegistry
from openrook_tools.executor import ToolExecutor
from openrook_tools import register_builtins
from tests._fakes import MockLLMBackend


# ===================================================================
# Helpers
# ===================================================================

def _make_executor_with_builtins():
    """Create a registry + executor with all builtins registered."""
    reg = ToolRegistry()
    exe = ToolExecutor(reg)
    register_builtins(reg, exe)
    return reg, exe


class FakeEngine:
    """Minimal engine mock for tool loop tests."""
    def __init__(self):
        self.state = ModulatorState()


def _default_action_vars(**overrides):
    kw = dict(
        risk_tolerance=0.5,
        action_urgency=0.5,
        clarification_threshold=0.5,
        persistence_drive=0.5,
        autonomy_bias=0.5,
    )
    kw.update(overrides)
    return ActionVariables(**kw)


# ===================================================================
# Task types
# ===================================================================

class TestTaskTypes:
    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.BLOCKED.value == "blocked"

    def test_task_step_defaults(self):
        step = TaskStep(
            id="s1", tool_name="fs.read_file",
            arguments={"path": "/tmp/x"}, description="read file",
        )
        assert step.status == TaskStatus.PENDING
        assert step.result is None
        assert step.observation is None
        assert step.started_at is None

    def test_task_plan_properties(self):
        steps = [
            TaskStep(id="s1", tool_name="fs.read_file", arguments={}, description="r",
                     status=TaskStatus.COMPLETED),
            TaskStep(id="s2", tool_name="fs.write_file", arguments={}, description="w",
                     status=TaskStatus.FAILED),
            TaskStep(id="s3", tool_name="fs.list_dir", arguments={}, description="l"),
        ]
        plan = TaskPlan(id="p1", goal="test", steps=steps)
        assert plan.steps_completed == 1
        assert plan.steps_failed == 1
        assert plan.max_steps == 5
        assert plan.is_terminal is False

    def test_plan_terminal_states(self):
        plan = TaskPlan(id="p1", goal="t", steps=[], status=TaskStatus.COMPLETED)
        assert plan.is_terminal is True
        plan.status = TaskStatus.FAILED
        assert plan.is_terminal is True
        plan.status = TaskStatus.BLOCKED
        assert plan.is_terminal is True
        plan.status = TaskStatus.IN_PROGRESS
        assert plan.is_terminal is False

    def test_task_trace_defaults(self):
        trace = TaskTrace()
        assert trace.plan is None
        assert trace.steps_executed == 0
        assert trace.continued_after_failure is False
        assert trace.plan_outcome == ""


class TestToolTraceShape:
    def test_tool_trace_has_task_trace_field(self):
        tt = ToolTraceType()
        assert tt.task_trace is None


class TestSummarizePlanStatus:
    def test_summary_contains_goal(self):
        plan = TaskPlan(id="p1", goal="do stuff", steps=[
            TaskStep(id="s1", tool_name="fs.read_file", arguments={},
                     description="read", status=TaskStatus.COMPLETED),
            TaskStep(id="s2", tool_name="fs.list_dir", arguments={},
                     description="list", status=TaskStatus.PENDING),
        ])
        summary = summarize_plan_status(plan)
        assert "do stuff" in summary
        assert "1/2" in summary
        assert "[done]" in summary


# ===================================================================
# Plan execution
# ===================================================================

class TestExecutePlan:
    def test_successful_plan(self, tmp_path):
        """Execute a 2-step plan: write then read."""
        reg, exe = _make_executor_with_builtins()
        target = str(tmp_path / "test.txt")

        steps = [
            TaskStep(id="s1", tool_name="fs.write_file",
                     arguments={"path": target, "content": "hello"},
                     description="write file"),
            TaskStep(id="s2", tool_name="fs.read_file",
                     arguments={"path": target},
                     description="read file"),
        ]
        plan = TaskPlan(id="p1", goal="write and read", steps=steps)
        action_vars = _default_action_vars(persistence_drive=0.7)

        trace = execute_plan(plan, exe, action_vars)

        assert trace.steps_executed == 2
        assert trace.steps_succeeded == 2
        assert trace.steps_failed == 0
        assert trace.plan_outcome == "completed"
        assert plan.status == TaskStatus.COMPLETED
        assert plan.completed_at is not None

        # Verify steps have results
        assert steps[0].result is not None
        assert steps[0].result.success is True
        assert steps[1].result is not None
        assert steps[1].result.output == "hello"

    def test_failure_blocks_with_low_persistence(self, tmp_path):
        """Low persistence → plan blocks on first failure."""
        reg, exe = _make_executor_with_builtins()

        steps = [
            TaskStep(id="s1", tool_name="fs.read_file",
                     arguments={"path": "/nonexistent_xyz"},
                     description="read missing file"),
            TaskStep(id="s2", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description="list dir"),
        ]
        plan = TaskPlan(id="p1", goal="will fail", steps=steps)
        action_vars = _default_action_vars(persistence_drive=0.3)

        trace = execute_plan(plan, exe, action_vars)

        assert trace.steps_executed == 1
        assert trace.steps_failed == 1
        assert trace.plan_outcome == "blocked"
        assert plan.status == TaskStatus.BLOCKED
        # Second step never executed
        assert steps[1].status == TaskStatus.PENDING

    def test_failure_continues_with_high_persistence(self, tmp_path):
        """High persistence → continues after failure."""
        reg, exe = _make_executor_with_builtins()

        steps = [
            TaskStep(id="s1", tool_name="fs.read_file",
                     arguments={"path": "/nonexistent_xyz"},
                     description="read missing"),
            TaskStep(id="s2", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description="list dir"),
        ]
        plan = TaskPlan(id="p1", goal="partial", steps=steps)
        action_vars = _default_action_vars(persistence_drive=0.7)

        trace = execute_plan(plan, exe, action_vars)

        assert trace.steps_executed == 2
        assert trace.steps_failed == 1
        assert trace.steps_succeeded == 1
        assert trace.continued_after_failure is True
        assert trace.plan_outcome == "partial"

    def test_all_fail(self):
        """All steps fail."""
        reg, exe = _make_executor_with_builtins()

        steps = [
            TaskStep(id="s1", tool_name="fs.read_file",
                     arguments={"path": "/nonexistent_1"},
                     description="fail 1"),
            TaskStep(id="s2", tool_name="fs.read_file",
                     arguments={"path": "/nonexistent_2"},
                     description="fail 2"),
        ]
        plan = TaskPlan(id="p1", goal="all fail", steps=steps)
        action_vars = _default_action_vars(persistence_drive=0.8)

        trace = execute_plan(plan, exe, action_vars)

        assert trace.steps_failed == 2
        assert trace.steps_succeeded == 0
        assert trace.plan_outcome == "failed"
        assert plan.status == TaskStatus.FAILED

    def test_max_steps_per_turn(self, tmp_path):
        """Respects max_steps_per_turn limit."""
        reg, exe = _make_executor_with_builtins()

        steps = [
            TaskStep(id=f"s{i}", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description=f"list {i}")
            for i in range(5)
        ]
        plan = TaskPlan(id="p1", goal="many steps", steps=steps)
        action_vars = _default_action_vars()

        trace = execute_plan(plan, exe, action_vars, max_steps_per_turn=2)

        assert trace.steps_executed == 2
        assert plan.current_step_index == 2
        assert plan.status == TaskStatus.IN_PROGRESS  # more steps remain
        assert trace.plan_outcome == "partial"

    def test_step_gate_blocks_laundered_risky_step(self, tmp_path):
        """A benign first step must not launder a risky later step past the gate.

        Round-5 review found execute_plan gated the whole plan on only the
        first step's category. A read-only step #1 then let a DESTRUCTIVE
        step #2 run under assisted autonomy. The per-step gate must block the
        risky step against its OWN category and halt the plan.
        """
        reg, exe = _make_executor_with_builtins()
        victim = tmp_path / "victim.txt"
        victim.write_text("precious")

        steps = [
            TaskStep(id="s1", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description="list dir (read-only)"),
            TaskStep(id="s2", tool_name="fs.write_file",
                     arguments={"path": str(victim), "content": "overwritten"},
                     description="overwrite the file (write)"),
        ]
        plan = TaskPlan(id="p1", goal="list then overwrite", steps=steps)
        action_vars = _default_action_vars(persistence_drive=0.7)

        # Gate: allow READ_ONLY, block everything else (mirrors assisted
        # autonomy's treatment of WRITE/DESTRUCTIVE/EXTERNAL_ACTION).
        def gate(step, category):
            return category == ToolCategory.READ_ONLY

        trace = execute_plan(plan, exe, action_vars, step_gate=gate)

        assert steps[0].status == TaskStatus.COMPLETED
        assert steps[1].status == TaskStatus.BLOCKED
        assert plan.status == TaskStatus.BLOCKED
        assert trace.plan_outcome == "blocked"
        # The write never happened.
        assert victim.read_text() == "precious"

    def test_step_gate_allows_when_all_steps_permitted(self, tmp_path):
        """A gate that permits every category lets the plan run normally."""
        reg, exe = _make_executor_with_builtins()
        target = str(tmp_path / "ok.txt")
        steps = [
            TaskStep(id="s1", tool_name="fs.write_file",
                     arguments={"path": target, "content": "hi"},
                     description="write"),
            TaskStep(id="s2", tool_name="fs.read_file",
                     arguments={"path": target},
                     description="read"),
        ]
        plan = TaskPlan(id="p1", goal="write then read", steps=steps)
        action_vars = _default_action_vars(persistence_drive=0.7)

        trace = execute_plan(plan, exe, action_vars, step_gate=lambda s, c: True)

        assert trace.steps_succeeded == 2
        assert plan.status == TaskStatus.COMPLETED

    def test_emotional_deltas_applied(self, tmp_path):
        """Engine receives emotional deltas from plan steps."""
        reg, exe = _make_executor_with_builtins()
        engine = FakeEngine()
        initial_certainty = engine.state.certainty

        target = str(tmp_path / "e.txt")
        steps = [
            TaskStep(id="s1", tool_name="fs.write_file",
                     arguments={"path": target, "content": "x"},
                     description="write"),
        ]
        plan = TaskPlan(id="p1", goal="test", steps=steps)
        action_vars = _default_action_vars()

        execute_plan(plan, exe, action_vars, engine=engine)

        # Certainty should have shifted (success → +certainty)
        assert engine.state.certainty != initial_certainty

    def test_observations_stored_on_steps(self, tmp_path):
        """Each executed step has an observation."""
        reg, exe = _make_executor_with_builtins()

        steps = [
            TaskStep(id="s1", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description="list"),
        ]
        plan = TaskPlan(id="p1", goal="test", steps=steps)
        action_vars = _default_action_vars()

        execute_plan(plan, exe, action_vars)

        assert steps[0].observation is not None
        assert isinstance(steps[0].observation, ToolObservation)


# ===================================================================
# Tool loop integration (multi-step)
# ===================================================================

class TestToolLoopMultiStep:
    def test_single_step_no_task_trace(self, tmp_path):
        """Single-step intent does not produce a TaskTrace."""
        reg, exe = _make_executor_with_builtins()
        engine = FakeEngine()

        f = tmp_path / "single.txt"
        f.write_text("data")

        result = run_tool_loop(
            requested_intent=ToolIntent(tool_name="fs.read_file", arguments={"path": str(f)},
                                        reason="test", expected_outcome="test"),
            state=engine.state,
            person=PersonProfile(person_id="u1", trust=0.7),
            defense_active=False,
            executor=exe,
            engine=engine,
        )

        assert result.trace.task_trace is None
        assert len(result.trace.executed_results) == 1

    def test_continue_request_resumes_plan(self, tmp_path):
        """Continue request advances an active plan."""
        reg, exe = _make_executor_with_builtins()
        engine = FakeEngine()

        # Create a partially-executed plan
        steps = [
            TaskStep(id="s1", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description="list", status=TaskStatus.COMPLETED),
            TaskStep(id="s2", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description="list again"),
        ]
        active_plan = TaskPlan(
            id="p1", goal="test continue", steps=steps,
            current_step_index=1, status=TaskStatus.IN_PROGRESS,
        )

        result = continue_plan(
            active_plan,
            state=engine.state,
            person=PersonProfile(person_id="u1", trust=0.7),
            defense_active=False,
            executor=exe,
            engine=engine,
        )

        assert result.trace.task_trace is not None
        assert result.trace.task_trace.steps_executed >= 1

    def test_plan_continuation_gates_each_step_under_assisted(self, tmp_path):
        """Plan continuation wires a per-step gate so assisted autonomy blocks a
        write step even when the leading step is read-only.

        End-to-end regression for the round-5 plan-step laundering finding.
        """
        reg, exe = _make_executor_with_builtins()
        engine = FakeEngine()
        victim = tmp_path / "v.txt"
        victim.write_text("keep")

        steps = [
            TaskStep(id="s1", tool_name="fs.list_dir",
                     arguments={"path": str(tmp_path)},
                     description="list (read-only)"),
            TaskStep(id="s2", tool_name="fs.write_file",
                     arguments={"path": str(victim), "content": "clobbered"},
                     description="write (needs confirmation)"),
        ]
        active_plan = TaskPlan(
            id="p1", goal="list then write", steps=steps,
            status=TaskStatus.IN_PROGRESS,
        )

        result = continue_plan(
            active_plan,
            state=engine.state,
            person=PersonProfile(person_id="u1", trust=0.7),
            defense_active=False,
            executor=exe,
            engine=engine,
            autonomy_level="assisted",
        )

        assert result.trace.task_trace is not None
        assert result.trace.task_trace.plan_outcome == "blocked"
        assert steps[1].status == TaskStatus.BLOCKED
        # The write was blocked by the autonomy gate.
        assert victim.read_text() == "keep"


# ===================================================================
# Task memory coupling
# ===================================================================

class TestTaskUnresolvedItem:
    def test_successful_plan_no_unresolved(self):
        plan = TaskPlan(id="p1", goal="ok", steps=[], status=TaskStatus.COMPLETED)
        trace = TaskTrace(plan=plan, steps_succeeded=2, steps_failed=0)
        assert create_task_unresolved_item(plan, trace) is None

    def test_blocked_plan_creates_unresolved(self):
        plan = TaskPlan(id="p1", goal="blocked goal", steps=[
            TaskStep(id="s1", tool_name="fs.read_file", arguments={}, description="r"),
        ], status=TaskStatus.BLOCKED)
        trace = TaskTrace(plan=plan, steps_executed=1, steps_succeeded=0, steps_failed=1)
        item = create_task_unresolved_item(plan, trace)
        assert item is not None
        assert item.source == "task_blocked"
        assert "blocked goal" in item.description.lower()

    def test_partial_plan_creates_unresolved(self):
        plan = TaskPlan(id="p1", goal="partial", steps=[], status=TaskStatus.COMPLETED)
        trace = TaskTrace(plan=plan, steps_executed=3, steps_succeeded=2, steps_failed=1,
                          plan_outcome="partial")
        item = create_task_unresolved_item(plan, trace)
        assert item is not None
        assert item.source == "task_incomplete"

    def test_intensity_scales_with_failures(self):
        plan = TaskPlan(id="p1", goal="many fails", steps=[], status=TaskStatus.BLOCKED)
        trace1 = TaskTrace(plan=plan, steps_failed=1)
        trace3 = TaskTrace(plan=plan, steps_failed=3)
        item1 = create_task_unresolved_item(plan, trace1)
        item3 = create_task_unresolved_item(plan, trace3)
        assert item1 is not None and item3 is not None
        assert item3.intensity > item1.intensity


class TestTaskSelfObservations:
    def test_completed_plan_methodical(self):
        plan = TaskPlan(id="p1", goal="done", steps=[
            TaskStep(id="s1", tool_name="fs.read_file", arguments={}, description="r"),
        ], status=TaskStatus.COMPLETED)
        trace = TaskTrace(plan=plan, steps_failed=0)
        obs = derive_task_self_observations(plan, trace, _default_action_vars())
        traits = [o[0] for o in obs]
        assert "methodical" in traits

    def test_continued_after_failure_persistent(self):
        plan = TaskPlan(id="p1", goal="persisted", steps=[], status=TaskStatus.COMPLETED)
        trace = TaskTrace(plan=plan, continued_after_failure=True, steps_failed=1)
        obs = derive_task_self_observations(plan, trace, _default_action_vars())
        traits = [o[0] for o in obs]
        assert "persistent" in traits

    def test_failed_plan_frustrated(self):
        plan = TaskPlan(id="p1", goal="failed", steps=[], status=TaskStatus.FAILED)
        trace = TaskTrace(plan=plan, steps_failed=2)
        obs = derive_task_self_observations(plan, trace, _default_action_vars())
        traits = [o[0] for o in obs]
        assert "frustrated" in traits

    def test_blocked_plan_hesitant(self):
        plan = TaskPlan(id="p1", goal="blocked", steps=[], status=TaskStatus.BLOCKED)
        trace = TaskTrace(plan=plan, steps_failed=1)
        obs = derive_task_self_observations(plan, trace, _default_action_vars())
        traits = [o[0] for o in obs]
        assert "hesitant" in traits

    def test_high_risk_reckless(self):
        plan = TaskPlan(id="p1", goal="risky", steps=[
            TaskStep(id="s1", tool_name="fs.write_file", arguments={}, description="w"),
        ], status=TaskStatus.COMPLETED)
        trace = TaskTrace(plan=plan, steps_failed=0)
        obs = derive_task_self_observations(
            plan, trace, _default_action_vars(risk_tolerance=0.8),
        )
        traits = [o[0] for o in obs]
        assert "reckless" in traits

    def test_capped_at_three(self):
        plan = TaskPlan(id="p1", goal="many obs", steps=[
            TaskStep(id="s1", tool_name="fs.delete_path", arguments={}, description="d"),
            TaskStep(id="s2", tool_name="shell.run_command", arguments={}, description="s"),
        ], status=TaskStatus.COMPLETED)
        trace = TaskTrace(plan=plan, continued_after_failure=True, steps_failed=0)
        obs = derive_task_self_observations(
            plan, trace, _default_action_vars(risk_tolerance=0.9),
        )
        assert len(obs) <= 3


class TestTaskLongTermEntry:
    def test_routine_success_not_salient(self):
        plan = TaskPlan(id="p1", goal="routine", steps=[
            TaskStep(id="s1", tool_name="fs.read_file", arguments={}, description="r"),
        ], status=TaskStatus.COMPLETED)
        trace = TaskTrace(plan=plan, steps_failed=0, steps_executed=1, steps_succeeded=1)
        entry = create_task_long_term_entry(plan, trace)
        assert entry is None  # not salient

    def test_failure_is_salient(self):
        plan = TaskPlan(id="p1", goal="had failures", steps=[
            TaskStep(id="s1", tool_name="fs.read_file", arguments={}, description="r"),
        ], status=TaskStatus.COMPLETED)
        trace = TaskTrace(
            plan=plan, steps_failed=1, steps_executed=2,
            steps_succeeded=1, plan_outcome="partial",
        )
        entry = create_task_long_term_entry(plan, trace, user_id="u1")
        assert entry is not None
        assert "partial" in entry.summary
        assert entry.source_person == "u1"

    def test_blocked_is_salient(self):
        plan = TaskPlan(id="p1", goal="blocked plan", steps=[], status=TaskStatus.BLOCKED)
        trace = TaskTrace(plan=plan, steps_failed=1, plan_outcome="blocked")
        entry = create_task_long_term_entry(plan, trace)
        assert entry is not None
        assert entry.emotional_valence < 0
        assert entry.spike is True  # blocked → spike bypass

    def test_destructive_is_salient(self):
        plan = TaskPlan(id="p1", goal="delete stuff", steps=[
            TaskStep(id="s1", tool_name="fs.delete_path", arguments={}, description="d"),
        ], status=TaskStatus.COMPLETED)
        trace = TaskTrace(
            plan=plan, steps_failed=0, steps_executed=1,
            steps_succeeded=1, plan_outcome="completed",
        )
        entry = create_task_long_term_entry(plan, trace)
        assert entry is not None


# ===================================================================
# Pipeline integration
# ===================================================================

class TestPipelineTaskIntegration:
    def test_pipeline_has_active_task_plan(self):
        from pipeline import CognitivePipeline
        p = CognitivePipeline(llm_backend=MockLLMBackend())
        assert p._active_task_plan is None

    def test_end_session_clears_task_plan(self):
        from pipeline import CognitivePipeline
        p = CognitivePipeline(llm_backend=MockLLMBackend())
        p._active_task_plan = TaskPlan(id="p1", goal="t", steps=[])
        p.end_session()
        assert p._active_task_plan is None

    def test_debug_state_has_task_trace(self):
        from pipeline import DebugState
        d = DebugState()
        assert d.task_trace is None

# ===================================================================
# Debug API serialization
# ===================================================================

class TestDebugAPISerialization:
    def test_task_trace_serialized(self):
        from runtime.debug.api import _debug_to_dict
        from pipeline import DebugState

        plan = TaskPlan(id="p1", goal="test goal", steps=[
            TaskStep(id="s1", tool_name="fs.read_file",
                     arguments={"path": "/tmp/x"}, description="read",
                     status=TaskStatus.COMPLETED),
        ], status=TaskStatus.COMPLETED)
        trace = TaskTrace(
            plan=plan, steps_executed=1, steps_succeeded=1,
            plan_outcome="completed",
        )

        debug = DebugState()
        debug.task_trace = trace
        d = _debug_to_dict(debug)

        assert d["task_trace"] is not None
        assert d["task_trace"]["plan"]["id"] == "p1"
        assert d["task_trace"]["plan"]["goal"] == "test goal"
        assert d["task_trace"]["plan"]["status"] == "completed"
        assert d["task_trace"]["steps_executed"] == 1
        assert d["task_trace"]["plan_outcome"] == "completed"
        assert len(d["task_trace"]["plan"]["steps"]) == 1
        assert d["task_trace"]["plan"]["steps"][0]["tool_name"] == "fs.read_file"

    def test_no_task_trace_serialized_as_none(self):
        from runtime.debug.api import _debug_to_dict
        from pipeline import DebugState

        debug = DebugState()
        d = _debug_to_dict(debug)
        assert d["task_trace"] is None

    def test_blocked_plan_serialized(self):
        from runtime.debug.api import _debug_to_dict
        from pipeline import DebugState

        plan = TaskPlan(id="p2", goal="blocked", steps=[
            TaskStep(id="s1", tool_name="fs.read_file",
                     arguments={}, description="r",
                     status=TaskStatus.FAILED),
            TaskStep(id="s2", tool_name="fs.list_dir",
                     arguments={}, description="l",
                     status=TaskStatus.PENDING),
        ], status=TaskStatus.BLOCKED)
        trace = TaskTrace(
            plan=plan, steps_executed=1, steps_failed=1,
            plan_outcome="blocked",
        )

        debug = DebugState()
        debug.task_trace = trace
        d = _debug_to_dict(debug)

        assert d["task_trace"]["plan"]["status"] == "blocked"
        assert d["task_trace"]["plan"]["steps_failed"] == 1
        assert d["task_trace"]["continued_after_failure"] is False


# ===================================================================
# Regression tests
# ===================================================================

class TestPhase7Regression:
    def test_single_step_still_works(self, tmp_path):
        """Existing single-step tool loop is unchanged."""
        reg, exe = _make_executor_with_builtins()
        engine = FakeEngine()

        f = tmp_path / "regression.txt"
        f.write_text("ok")

        result = run_tool_loop(
            requested_intent=ToolIntent(tool_name="fs.read_file", arguments={"path": str(f)},
                                        reason="test", expected_outcome="test"),
            state=engine.state,
            person=PersonProfile(person_id="u1", trust=0.7),
            defense_active=False,
            executor=exe,
            engine=engine,
        )

        assert len(result.trace.executed_results) == 1
        assert result.trace.executed_results[0].success is True
        assert result.trace.task_trace is None

    def test_no_executor_pipeline_still_works(self):
        """Pipeline without tool_executor processes normally."""
        from pipeline import CognitivePipeline
        p = CognitivePipeline(llm_backend=MockLLMBackend())
        result = p.process("hello")
        assert result.response
        assert result.debug.task_trace is None

    def test_builtin_count_unchanged(self):
        """Builtin registration count includes read-only system tools."""
        reg, exe = _make_executor_with_builtins()
        # +5 from the self.* self-action catalog (v0.30); +1 system.cpu_usage.
        assert len(reg) == 36

    def test_terminal_plan_cleared_from_session(self, tmp_path):
        """A running plan that a granted continuation completes is cleared from _active_task_plan."""
        from pipeline import CognitivePipeline
        reg = ToolRegistry()
        exe = ToolExecutor(reg)
        register_builtins(reg, exe)

        message = "carry on with it"
        p = CognitivePipeline(llm_backend=MockLLMBackend(), tool_executor=exe, message_reader=ScriptedReader(
            reading(task(message, user_value=0.9, effort=0.05, plan="continue"))))
        # A plan that a committed decision started; one without such a decision cannot be carried on.
        p._active_task_plan = TaskPlan(id="p1", goal="look around", status=TaskStatus.IN_PROGRESS, origin_decision_id="1",
                                       steps=[TaskStep(id="s1", tool_name="fs.list_dir", arguments={"path": str(tmp_path)},
                                                       description="list")])
        result = p.process(message)

        assert result.debug.task_trace is not None and result.debug.task_trace.plan.is_terminal
        assert [r.tool_name for r in result.debug.tool_trace.executed_results] == ["fs.list_dir"]
        assert p._active_task_plan is None
        p.close()

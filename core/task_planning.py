"""The plan state machine: a bounded multi-step task, carried out a few steps per turn.

This module executes and reports plans. It does not recognise requests: what a
message asks for, whether it asks a running plan to go on, and whether it asks
how the plan stands are all reported by the reading of the message
(``docs/development/request-routing.md``).

Responsibilities:
  1. Execute a plan's pending steps in order through the tool executor,
     bounded per turn, each step gated against its own risk category
  2. Track plan state and handle failures based on the persistence drive
  3. Summarise how a plan stands
"""

from __future__ import annotations

from core.clock import clock as time
from typing import Any, Callable

from core.types import (
    ActionVariables,
    TaskPlan,
    TaskStatus,
    TaskStep,
    TaskTrace,
    ToolCategory,
    ToolObservation,
)
from core.tool_appraisal import appraise_tool_result
from openrook_tools.executor import ToolExecutor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PLAN_STEPS = 5
DEFAULT_MAX_STEPS_PER_TURN = 3  # how many steps to execute in one turn


def summarize_plan_status(plan: TaskPlan) -> str:
    """Build a human-readable status summary for a plan."""
    parts = [f"Plan: {plan.goal[:100]}"]
    parts.append(f"Status: {plan.status.value}")
    parts.append(f"Progress: {plan.steps_completed}/{len(plan.steps)} steps completed")

    if plan.steps_failed > 0:
        parts.append(f"Failures: {plan.steps_failed}")

    for step in plan.steps:
        marker = {
            TaskStatus.COMPLETED: "[done]",
            TaskStatus.FAILED: "[FAIL]",
            TaskStatus.IN_PROGRESS: "[....]",
            TaskStatus.BLOCKED: "[BLOCK]",
            TaskStatus.PENDING: "[    ]",
        }.get(step.status, "[    ]")
        parts.append(f"  {marker} {step.description}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------

def execute_plan(
    plan: TaskPlan,
    executor: ToolExecutor,
    action_vars: ActionVariables,
    engine: Any = None,
    max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN,
    step_gate: Callable[[TaskStep, ToolCategory], bool] | None = None,
) -> TaskTrace:
    """Execute pending steps of a plan sequentially.

    Bounded by max_steps_per_turn per invocation.
    Failure handling is driven by persistence_drive:
      - persistence_drive >= 0.5 → continue after failure
      - persistence_drive < 0.5 → block the plan on first failure

    ``step_gate`` is an authorization callback invoked *before* each step
    executes, receiving the step and its own tool category. Returning False
    blocks that step (and the rest of the plan). This closes the laundering
    hole where a read-only first step would let a later WRITE/DESTRUCTIVE/
    EXTERNAL_ACTION step bypass the autonomy arbiter: every step is gated
    against its own risk category, not the plan's first.

    Returns a TaskTrace with execution details.
    """
    trace = TaskTrace(plan=plan)
    plan.status = TaskStatus.IN_PROGRESS
    plan.persistence_drive = action_vars.persistence_drive

    steps_this_turn = 0

    while (
        plan.current_step_index < len(plan.steps)
        and steps_this_turn < max_steps_per_turn
    ):
        step = plan.steps[plan.current_step_index]

        if step.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            plan.current_step_index += 1
            continue

        # Per-step authorization: gate each step against its own category so a
        # benign leading step cannot launder a risky trailing step past the
        # autonomy arbiter.
        if step_gate is not None:
            step_cap = executor.registry.get(step.tool_name)
            step_category = (
                step_cap.category if step_cap else ToolCategory.EXTERNAL_ACTION
            )
            if not step_gate(step, step_category):
                step.status = TaskStatus.BLOCKED
                plan.status = TaskStatus.BLOCKED
                trace.plan_outcome = "blocked"
                return trace

        # Execute step
        step.status = TaskStatus.IN_PROGRESS
        step.started_at = time.time()

        result = executor.execute(step.tool_name, step.arguments)
        step.result = result
        step.completed_at = time.time()

        # Appraise
        cap = executor.registry.get(step.tool_name)
        category = cap.category if cap else ToolCategory.READ_ONLY
        observation = appraise_tool_result(result, category)
        step.observation = observation

        # Apply emotional deltas to engine if provided
        if engine is not None:
            _apply_emotional_deltas(engine, observation)

        trace.steps_executed += 1
        trace.total_latency_ms += result.latency_ms
        steps_this_turn += 1

        if result.success:
            step.status = TaskStatus.COMPLETED
            trace.steps_succeeded += 1
        else:
            step.status = TaskStatus.FAILED
            trace.steps_failed += 1

            # Persistence-driven failure handling
            if action_vars.persistence_drive < 0.5:
                plan.status = TaskStatus.BLOCKED
                trace.plan_outcome = "blocked"
                return trace
            else:
                trace.continued_after_failure = True

        plan.current_step_index += 1

    # Determine final plan status
    if plan.current_step_index >= len(plan.steps):
        # All steps attempted
        if trace.steps_failed == 0:
            plan.status = TaskStatus.COMPLETED
            plan.completed_at = time.time()
            trace.plan_outcome = "completed"
        elif trace.steps_succeeded > 0:
            plan.status = TaskStatus.COMPLETED
            plan.completed_at = time.time()
            trace.plan_outcome = "partial"
        else:
            plan.status = TaskStatus.FAILED
            plan.completed_at = time.time()
            trace.plan_outcome = "failed"
    else:
        # More steps remain (hit per-turn limit)
        trace.plan_outcome = "partial"

    return trace


def _apply_emotional_deltas(engine: Any, observation: ToolObservation) -> None:
    """Apply an observation's emotional deltas to the engine state."""
    state = engine.state
    for modulator, delta in observation.emotional_delta.items():
        if hasattr(state, modulator):
            current = getattr(state, modulator)
            new_val = max(0.0, min(1.0, current + delta))
            setattr(state, modulator, new_val)

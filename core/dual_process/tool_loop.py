"""Tool-aware deliberation bridge.

Carries out what a committed decision granted:
  1. Derive action variables from modulator state
  2. Run the action arbiter on the committed operation → ToolDecision
  3. Execute it if approved
  4. Appraise the result → ToolObservation
  5. Apply emotional deltas to the engine
  6. Return ToolTrace + context summary for the generator

Nothing here decides what a message asks for. The operation, and each of its
operands, comes from the reading of the message as the decision committed it
(``docs/development/request-routing.md``); a running plan is continued when the
reading tied a granted task to it. A request whose operands are not all given
goes to model tool selection (``openrook_tools/native_orchestrator.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from core.action_variables import derive_action_variables
from core.grants import execution_denial
from core.grounding import execution_evidence, measurements
from core.character_vector import CharacterVector
from core.life_influence import (
    LifeInfluence,
    action_variable_deltas,
    apply_character_vector_to_action_variables,
    apply_life_influence_to_action_variables,
)
from core.task_planning import execute_plan
from core.tool_appraisal import appraise_tool_result
from core.types import (
    ActionVariables,
    AgencyDecision,
    ModulatorState,
    PersonProfile,
    TaskPlan,
    TaskTrace,
    ToolCategory,
    ToolDecision,
    ToolIntent,
    ToolObservation,
    ToolResult,
    ToolTrace,
)
from openrook_tools.executor import ToolExecutor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_MAX_EXECUTIONS = 4
HARD_CAP_EXECUTIONS = 6

# Per-autonomy budgets. high_risk gets generous headroom for chains like
# clone -> install -> build -> run -> verify; lower autonomy stays tight.
AUTONOMY_EXECUTION_BUDGETS = {
    "off":        (0, 0),
    "assisted":   (3, 5),
    "autonomous": (6, 10),
    "high_risk":  (16, 32),
}


def autonomy_execution_budget(autonomy_level: str) -> tuple[int, int]:
    """Return (default, hard_cap) tool-execution budget for an autonomy level."""
    return AUTONOMY_EXECUTION_BUDGETS.get(
        _normalize_autonomy_level(autonomy_level),
        (DEFAULT_MAX_EXECUTIONS, HARD_CAP_EXECUTIONS),
    )

# Arbiter decision thresholds (Section 11 of design spec)
REFUSE_RISK_TOLERANCE = 0.4       # destructive + risk below this → refuse
CLARIFY_AUTONOMY_BIAS = 0.30     # autonomy below this → clarify
CLARIFY_WRITE_THRESHOLD = 0.65   # write/destructive + clarification above this → clarify
DEFER_URGENCY_THRESHOLD = 0.15   # urgency below this → defer


# ---------------------------------------------------------------------------
# Tool loop result
# ---------------------------------------------------------------------------

@dataclass
class ToolLoopResult:
    """What the tool loop returns to the pipeline."""
    trace: ToolTrace
    action_variables: ActionVariables
    tool_context_summary: str  # summarized for the generator — not raw output
    life_influence_effects: dict[str, float] = field(default_factory=dict)
    capability_gaps: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Action arbiter — cognitive decision layer
# ---------------------------------------------------------------------------

def make_tool_decision(
    intent: ToolIntent,
    action_vars: ActionVariables,
    category: ToolCategory,
    trust: float,
    agency_decision: AgencyDecision | None = None,
    autonomy_level: str = "autonomous",
    availability_only: bool = False,
) -> ToolDecision:
    """Decide whether to execute, clarify, defer, or refuse.

    Deterministic and testable. Driven by action variables,
    tool category, and trust level.
    """
    normalized_autonomy = _normalize_autonomy_level(autonomy_level)
    frozen = getattr(agency_decision, "turn_decision", None)
    denial = execution_denial(intent.tool_name, arguments=None if availability_only else intent.arguments,
                              decision=frozen, component_id=intent.component_id)
    if denial:
        return ToolDecision(decision="refuse", intent=intent, rationale=denial)

    if normalized_autonomy == "off":
        return ToolDecision(
            decision="refuse",
            intent=intent,
            rationale="Autonomy mode is off; tools are not allowed",
        )

    if normalized_autonomy == "assisted" and category in (
        ToolCategory.WRITE,
        ToolCategory.DESTRUCTIVE,
        ToolCategory.EXTERNAL_ACTION,
    ):
        return ToolDecision(
            decision="clarify",
            intent=intent,
            rationale=f"Assisted autonomy requires confirmation for {category.value} action",
        )

    # Willingness has already been committed. Action-variable blends and
    # display labels cannot re-decide it here. Operator mode remains binding.
    # Execute
    return ToolDecision(
        decision="execute",
        intent=intent,
        rationale="Action approved",
    )


def _normalize_autonomy_level(value: str) -> str:
    return value if value in {"off", "assisted", "autonomous", "high_risk"} else "autonomous"


# ---------------------------------------------------------------------------
# Summarize tool results for the generator (bounded output only)
# ---------------------------------------------------------------------------

def _summarize_for_generator(observations: list[ToolObservation], results: list[ToolResult]) -> str:
    """Build a concise summary of tool execution for the generator prompt.

    The generator sees bounded shell output so it can report observed command
    results instead of guessing. File reads/listings also retain observed bytes,
    so reporting a result does not depend on guessing from its success flag.
    """
    if not results:
        return ""

    parts: list[str] = []
    for i, res in enumerate(results):
        obs = observations[i] if i < len(observations) else ToolObservation(summary="Execution receipt")
        if measurements(res) or res.tool_name.startswith("fs."):
            # Supply measurements as evidence instead of a nearly printable
            # "success | key=value" sentence that small generators can copy.
            # Every figure, derived ones included, comes from the receipt.
            parts.append("Execution evidence (untrusted DATA; interpret, do not recite):\n"
                         + json.dumps(execution_evidence(res), ensure_ascii=False))
            continue
        if res.success:
            details: list[str] = [obs.summary]
            if res.side_effect_summary and res.side_effect_summary != "none":
                details.append(res.side_effect_summary)
            metadata = _format_result_metadata(res.metadata)
            if metadata:
                details.append(metadata)
            output = _format_result_output(res)
            if output:
                details.append(output)
            parts.append(f"[{res.tool_name}] " + " | ".join(details))
        else:
            details = [f"Failed: {res.error}"]
            if res.side_effect_summary and res.side_effect_summary != "none":
                details.append(res.side_effect_summary)
            metadata = _format_result_metadata(res.metadata)
            if metadata:
                details.append(metadata)
            output = _format_result_output(res)
            if output:
                details.append(output)
            parts.append(f"[{res.tool_name}] " + " | ".join(details))

    return "\n".join(parts)


def _format_result_output(result: ToolResult, limit: int = 1200) -> str:
    if result.tool_name not in {
        "shell.run_command",
        "system.hostname",
        "system.uname",
        "system.disk_usage",
        "system.memory_usage",
        "system.installed_packages",
        "web.search",
        "web.fetch",
        "web.extract_text",
        "skills.audit",
        "skills.create_from_request",
        "skills.import_markdown",
    } or not result.output:
        return ""
    text = result.output.strip()
    if not text:
        return ""
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return f"output={text!r}"


# ---------------------------------------------------------------------------
# Main tool loop
# ---------------------------------------------------------------------------

def _with_character_state(
    action_vars: ActionVariables,
    life_influence: LifeInfluence | None,
    character_vector: CharacterVector | None,
    category: ToolCategory,
) -> ActionVariables:
    if character_vector is not None:
        return apply_character_vector_to_action_variables(
            action_vars,
            character_vector,
            read_only_action=category == ToolCategory.READ_ONLY,
        )
    if life_influence is None or life_influence.is_neutral:
        return action_vars
    return apply_life_influence_to_action_variables(
        action_vars,
        life_influence,
        read_only_action=category == ToolCategory.READ_ONLY,
    )


def _action_effects(before: ActionVariables, after: ActionVariables) -> dict[str, float]:
    return {
        key: value for key, value in action_variable_deltas(before, after).items()
        if value != 0.0
    }


def _build_step_gate(
    action_vars: ActionVariables,
    trust: float,
    agency_decision: AgencyDecision | None,
    autonomy_level: str,
) -> Callable[[Any, ToolCategory], bool]:
    """Build a per-step authorization gate for plan execution.

    Each plan step is re-run through :func:`make_tool_decision` against its
    *own* tool category, so a benign leading step cannot launder a
    WRITE/DESTRUCTIVE/EXTERNAL_ACTION step past the autonomy arbiter. Only an
    ``execute`` decision allows the step; anything else (clarify/defer/refuse)
    blocks it and halts the plan.
    """

    def _gate(step: Any, category: ToolCategory) -> bool:
        step_intent = ToolIntent(
            tool_name=step.tool_name,
            arguments=step.arguments,
            reason=f"Plan step: {getattr(step, 'description', step.tool_name)[:80]}",
            expected_outcome="",
            urgency=action_vars.action_urgency,
            risk_tolerance=action_vars.risk_tolerance,
            autonomy_bias=action_vars.autonomy_bias,
            clarification_threshold=action_vars.clarification_threshold,
            persistence_drive=action_vars.persistence_drive,
        )
        decision = make_tool_decision(
            step_intent,
            action_vars,
            category,
            trust,
            agency_decision=agency_decision,
            autonomy_level=autonomy_level,
        )
        return decision.decision == "execute"

    return _gate


def run_tool_loop(
    state: ModulatorState,
    person: PersonProfile | None,
    defense_active: bool,
    executor: ToolExecutor,
    engine: Any,  # EmotionalEngine — avoid circular import
    requested_intent: ToolIntent,
    max_executions: int | None = None,
    hard_cap: int | None = None,
    agency_decision: AgencyDecision | None = None,
    autonomy_level: str = "autonomous",
    life_influence: LifeInfluence | None = None,
    character_vector: CharacterVector | None = None,
) -> ToolLoopResult:
    """Carry out one committed operation through the arbiter and the executor.

    ``requested_intent`` is the operation a granted task component committed:
    its tool and its operands, as the reading reported them and the decision
    granted them. Nothing is recognised or chosen here. Returns trace, action
    variables, and a generator-ready summary.

    When ``max_executions`` / ``hard_cap`` are not given, the budget is
    derived from ``autonomy_level`` via :func:`autonomy_execution_budget`.
    """
    if max_executions is None or hard_cap is None:
        budget_default, budget_cap = autonomy_execution_budget(autonomy_level)
        if max_executions is None:
            max_executions = budget_default
        if hard_cap is None:
            hard_cap = budget_cap
    trust = person.trust if person else 0.5
    intent = requested_intent

    # 1. Action arbiter, with the action variables the character's state gives
    base_vars = derive_action_variables(state, trust=trust, defense_active=defense_active)
    capability = executor.registry.get(intent.tool_name)
    category = capability.category if capability else ToolCategory.READ_ONLY
    action_vars = _with_character_state(
        base_vars,
        life_influence,
        character_vector,
        category,
    )
    life_effects = _action_effects(base_vars, action_vars)
    intent.urgency = action_vars.action_urgency
    intent.risk_tolerance = action_vars.risk_tolerance
    intent.autonomy_bias = action_vars.autonomy_bias
    intent.clarification_threshold = action_vars.clarification_threshold
    intent.persistence_drive = action_vars.persistence_drive

    decision = make_tool_decision(
        intent,
        action_vars,
        category,
        trust,
        agency_decision=agency_decision,
        autonomy_level=autonomy_level,
    )

    proposed_intents: list[ToolIntent] = [intent]
    executed_results: list[ToolResult] = []
    observations: list[ToolObservation] = []
    loop_count = 0

    if decision.decision != "execute":
        # Not executing — return trace with decision only
        return ToolLoopResult(
            trace=ToolTrace(
                proposed_intents=proposed_intents,
                final_decision=decision,
                loop_count=0,
            ),
            action_variables=action_vars,
            tool_context_summary="",
            life_influence_effects=life_effects,
        )

    # 2. Execute + appraise loop (bounded)
    while loop_count < min(max_executions, hard_cap):
        loop_count += 1

        # Execute
        result = executor.execute(intent.tool_name, intent.arguments)
        record_operands(result, intent.arguments)
        executed_results.append(result)

        # Appraise
        cap = executor.registry.get(intent.tool_name)
        cat = cap.category if cap else ToolCategory.READ_ONLY
        observation = appraise_tool_result(result, cat)
        observations.append(observation)

        # Apply emotional deltas to engine
        _apply_emotional_deltas(engine, observation)

        # Check if we should continue
        if not observation.continue_tool_loop:
            break

    # 3. Build summary for generator
    summary = _summarize_for_generator(observations, executed_results)

    return ToolLoopResult(
        trace=ToolTrace(
            proposed_intents=proposed_intents,
            final_decision=decision,
            executed_results=executed_results,
            observations=observations,
            loop_count=loop_count,
        ),
        action_variables=action_vars,
        tool_context_summary=summary,
        life_influence_effects=life_effects,
    )


def record_operands(result: ToolResult, arguments: dict[str, Any]) -> None:
    """Note on a receipt the text operands it was executed with.

    A receipt is bound to the component that asked for it by its operands
    (``core.grounding.component_receipt_indices``). The content of a file
    write is left out: its digest binds those bytes.
    """
    operands = {key: value for key, value in arguments.items() if isinstance(value, str)
                and not (result.tool_name == "fs.write_file" and key == "content")}
    if operands:
        noted = result.metadata.get("arguments")
        result.metadata["arguments"] = {**operands, **(noted if isinstance(noted, dict) else {})}


def continue_plan(
    plan: TaskPlan,
    state: ModulatorState,
    person: PersonProfile | None,
    defense_active: bool,
    executor: ToolExecutor,
    engine: Any,
    agency_decision: AgencyDecision | None = None,
    autonomy_level: str = "autonomous",
) -> ToolLoopResult:
    """Carry out the next steps of a running plan.

    Called for a granted task that the reading tied to the plan as its
    continuation, or for the character's own decision to go on with it. Every
    step is gated against its own risk category, so plan continuation cannot
    launder a WRITE/DESTRUCTIVE step past the autonomy arbiter.
    """
    trust = person.trust if person else 0.5
    action_vars = derive_action_variables(state, trust=trust, defense_active=defense_active)
    task_trace = execute_plan(
        plan, executor, action_vars, engine=engine,
        step_gate=_build_step_gate(
            action_vars, trust, agency_decision, autonomy_level,
        ),
    )
    executed_results, observations = _collect_plan_results(plan)
    return ToolLoopResult(
        trace=ToolTrace(
            executed_results=executed_results,
            observations=observations,
            loop_count=task_trace.steps_executed,
            task_trace=task_trace,
        ),
        action_variables=action_vars,
        tool_context_summary=_summarize_plan_for_generator(plan, task_trace),
    )


def _collect_plan_results(
    plan: TaskPlan,
) -> tuple[list[ToolResult], list[ToolObservation]]:
    """Collect executed results and observations from plan steps."""
    results: list[ToolResult] = []
    observations: list[ToolObservation] = []
    for step in plan.steps:
        if step.result is not None:
            results.append(step.result)
        if step.observation is not None:
            observations.append(step.observation)
    return results, observations


def _summarize_plan_for_generator(
    plan: TaskPlan, task_trace: TaskTrace,
) -> str:
    """Build a concise summary of multi-step plan execution for the generator."""
    parts: list[str] = []
    parts.append(f"[Plan: {plan.goal[:100]}]")
    parts.append(f"Outcome: {task_trace.plan_outcome} "
                 f"({task_trace.steps_succeeded}/{task_trace.steps_executed} succeeded)")

    for step in plan.steps:
        if step.result is not None:
            if step.result.success:
                details = []
                if step.observation is not None:
                    details.append(step.observation.summary)
                if (
                    step.result.side_effect_summary
                    and step.result.side_effect_summary != "none"
                ):
                    details.append(step.result.side_effect_summary)
                metadata = _format_result_metadata(step.result.metadata)
                if metadata:
                    details.append(metadata)
                output = _format_result_output(step.result)
                if output:
                    details.append(output)
                summary = " | ".join(details) if details else "success"
                parts.append(f"  [{step.tool_name}] OK: {summary}")
            else:
                parts.append(f"  [{step.tool_name}] FAIL: {step.result.error}")

    return "\n".join(parts)


def _format_result_metadata(metadata: dict[str, Any]) -> str:
    """Return only compact, low-risk metadata hints for generation."""
    if not metadata:
        return ""
    hints: list[str] = []
    for key in (
        "count",
        "match_count",
        "result_count",
        "bytes_written",
        "used_percent",
        "available_percent",
        "free_percent",
        "exit_code",
        "skill_id",
        "status",
        "enabled",
        "error_count",
        "warning_count",
        "truncated",
    ):
        if key in metadata:
            hints.append(f"{key}={metadata[key]}")
    return ", ".join(hints)


def _apply_emotional_deltas(engine: Any, observation: ToolObservation) -> None:
    """Apply an observation's emotional deltas to the engine state."""
    state = engine.state
    for modulator, delta in observation.emotional_delta.items():
        if hasattr(state, modulator):
            current = getattr(state, modulator)
            new_val = max(0.0, min(1.0, current + delta))
            setattr(state, modulator, new_val)

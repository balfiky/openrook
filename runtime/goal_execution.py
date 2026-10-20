"""Versioned commitments and independently assessed completion claims."""

from __future__ import annotations

import math
from typing import Any

from core.clock import clock as time
from core.clock import identifiers as uuid
from runtime.research_store import ResearchStore
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit


EXECUTION_PROMPT = """
Execution is closed-loop: each proposal can execute at most one tool. Later
steps receive its actual receipt. An empty action list ends this activation.
Goals may include success_criteria, next_step and expected_result. Criteria are
recorded before action; changing them starts a new revision with new evidence.
Optional parent_goal_id links a subgoal; depends_on lists existing goal ids whose
completion must be supported (or explicitly subjective) before execution. You
may revise or remove your own dependencies. Cycles are not allowed.
Use {"kind":"subjective","description":"your own criterion"} for a personal
judgment, {"kind":"unverified","description":"criterion without an oracle"}
otherwise, or an environment's documented independently checkable criterion.
Completing a goal records your claim separately from verification; you are not
required to persist with it. Waiting may include wake_after_seconds (nonnegative)
and/or wake_on_external_input (boolean). These conditions are alternatives.
"""


def validate_criteria(criteria: Any) -> dict[str, Any]:
    if not isinstance(criteria, dict):
        raise ValueError("Success criteria must be an object")
    kind = criteria.get("kind")
    if kind in {"subjective", "unverified"}:
        if set(criteria) != {"kind", "description"}:
            raise ValueError("Personal criteria require kind and description")
        if not isinstance(criteria["description"], str) or not criteria["description"].strip():
            raise ValueError("Criteria description must be nonempty text")
    elif kind == "world_outcome":
        if set(criteria) != {"kind", "project", "minimum_progress"}:
            raise ValueError("World criteria require project and minimum_progress")
        minimum = criteria["minimum_progress"]
        if not isinstance(criteria["project"], str) or not criteria["project"]:
            raise ValueError("A project is required")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= 128:
            raise ValueError("minimum_progress must be an integer from 1 to 128")
    elif kind == "environment_outcome":
        if set(criteria) != {"kind", "opportunity", "phase", "minimum_correct"}:
            raise ValueError("Environment criteria require opportunity, phase and minimum_correct")
        if not isinstance(criteria["opportunity"], str) or not criteria["opportunity"]:
            raise ValueError("An opportunity is required")
        if criteria["phase"] not in {"practice", "probe"}:
            raise ValueError("Criterion phase must be practice or probe")
        minimum = criteria["minimum_correct"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= 32:
            raise ValueError("minimum_correct must be an integer from 1 to 32")
    else:
        raise ValueError("Unknown success criterion kind")
    return dict(criteria)


def wake_condition(decision: dict[str, Any], store: ResearchStore) -> dict[str, Any] | None:
    seconds = decision.get("wake_after_seconds")
    on_input = decision.get("wake_on_external_input", False)
    if not isinstance(on_input, bool):
        raise ValueError("wake_on_external_input must be boolean")
    if seconds is not None and (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError("wake_after_seconds must be a finite nonnegative number")
    if seconds is None and not on_input:
        return None
    return {
        "at": time.time() + seconds if seconds is not None else None,
        "external_after": store.latest_external_sequence() if on_input else None,
    }


def wake_ready(condition: dict[str, Any] | None, store: ResearchStore, now: float) -> bool:
    return (
        not condition
        or (condition["at"] is not None and now >= condition["at"])
        or (
            condition["external_after"] is not None
            and store.latest_external_sequence() > condition["external_after"]
        )
    )


def update_goal(
    store: ResearchStore, decision: dict[str, Any], *, executor: Any = None
) -> str | None:
    goal_id = decision.get("goal_id") or ""
    description = decision.get("goal") or ""
    if not isinstance(goal_id, str) or not isinstance(description, str):
        raise ValueError("Goal identifiers and descriptions must be text")
    if not goal_id and not description:
        return None
    goal = store.get("goal", goal_id) if goal_id else None
    if goal_id and goal is None:
        raise ValueError("Unknown goal id")
    status = decision.get("goal_status", "active")
    if status not in {"active", "waiting", "completed", "abandoned"}:
        raise ValueError("Invalid goal status")
    previous = store.get("goal_execution", goal_id) if goal else None
    relationships = {}
    for key, default in (("parent_goal_id", None), ("depends_on", [])):
        if key in decision or key in (previous or {}):
            relationships[key] = decision.get(key, (previous or {}).get(key, default))
    parent = relationships.get("parent_goal_id")
    dependencies = relationships.get("depends_on", [])
    if parent is not None and (
        not isinstance(parent, str) or not parent or store.get("goal", parent) is None
    ):
        raise ValueError("A parent must be an existing goal id or null")
    if (
        not isinstance(dependencies, list)
        or len(dependencies) > 20
        or any(
            not isinstance(item, str) or store.get("goal", item) is None for item in dependencies
        )
    ):
        raise ValueError("Dependencies must list at most twenty existing goal ids")
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("Duplicate goal dependency")
    graph = {item["goal_id"]: item.get("depends_on", []) for item in store.list("goal_execution")}
    graph[goal_id] = dependencies
    parents = {
        item["goal_id"]: [item["parent_goal_id"]] if item.get("parent_goal_id") else []
        for item in store.list("goal_execution")
    }
    parents[goal_id] = [parent] if parent else []
    for links in (graph, parents):
        pending = list(links[goal_id])
        visited = set()
        while pending:
            target = pending.pop()
            if target == goal_id:
                raise ValueError("Goal relationships must be acyclic")
            if target not in visited:
                visited.add(target)
                pending.extend(links.get(target, []))
    description = description or goal["description"]
    criteria = validate_criteria(
        decision.get(
            "success_criteria",
            previous["success_criteria"]
            if previous
            else {
                "kind": "unverified",
                "description": description,
            },
        )
    )
    for field in ("next_step", "expected_result"):
        if field in decision and not isinstance(decision[field], str):
            raise ValueError(f"{field} must be text")
    wake = wake_condition(decision, store) if status == "waiting" else None
    changed = (
        previous is None
        or (previous["objective"] != description or previous["success_criteria"] != criteria)
        or any(previous.get(key) != value for key, value in relationships.items())
    )
    goal_id = goal_id or uuid.uuid4().hex
    revision = (previous["revision"] if previous else 0) + int(changed)
    execution = {
        **(previous or {}),
        "schema_version": 1,
        "id": goal_id,
        "goal_id": goal_id,
        "revision": revision,
        "objective": description,
        "success_criteria": criteria,
        "status": status,
        "step_count": previous["step_count"] if previous else 0,
        "evidence_ids": [] if changed else previous["evidence_ids"],
        "continuation": decision.get("next_step", (previous or {}).get("continuation", "")),
        "expected_result": decision.get(
            "expected_result", (previous or {}).get("expected_result", "")
        ),
        "wake_condition": wake,
        "updated_at": time.time(),
        **relationships,
    }
    if changed:
        for field in ("assessment_id", "last_action_id", "last_action_status", "last_stop_reason"):
            execution.pop(field, None)
    updated_goal = {
        **(goal or {}),
        "id": goal_id,
        "description": description,
        "status": status,
        "reason": str(decision.get("reason") or ""),
        "updated_at": execution["updated_at"],
        "revision": revision,
        "verification_status": "unverified"
        if changed
        else goal.get("verification_status", "unverified"),
    }
    records = []
    if changed:
        records.append(("goal_revision", f"{goal_id}:{revision}", dict(execution)))
    assessment = None
    if status == "completed":
        verifier_called = False
        evidence = [store.get("action", record_id) for record_id in execution["evidence_ids"]]
        evidence = [item for item in evidence if item is not None]
        if criteria["kind"] == "subjective":
            verdict = {"status": "subjective", "method": "subject_judgment"}
        elif criteria["kind"] == "unverified" or not evidence or executor is None:
            verdict = {"status": "unverified", "method": "no_applicable_evidence_or_verifier"}
        else:
            verifier_called = True
            try:
                verdict = executor.verify_completion(criteria, evidence)
                if not isinstance(verdict, dict) or verdict.get("status") not in {
                    "supported",
                    "contradicted",
                    "unverified",
                }:
                    raise ValueError("Invalid verifier result")
            except (ResearchBudgetExceeded, ResearchStudyLimit):
                raise
            except Exception as error:
                verdict = {
                    "status": "unverified",
                    "method": "verifier_failure",
                    "error_type": type(error).__name__,
                }
        assessment_id = uuid.uuid4().hex
        assessment = {
            **verdict,
            "schema_version": 1,
            "id": assessment_id,
            "goal_id": goal_id,
            "revision": revision,
            "subject_claim": "completed",
            "verifier_called": verifier_called,
            "evidence_ids": list(execution["evidence_ids"]),
            "assessed_at": time.time(),
        }
        updated_goal["verification_status"] = assessment["status"]
        execution["assessment_id"] = assessment_id
        records.append(("completion_assessment", assessment_id, assessment))
    records.extend((("goal", goal_id, updated_goal), ("goal_execution", goal_id, execution)))
    store.put_records(
        records,
        event_kind="goal_changed",
        event={
            "id": goal_id,
            "status": status,
            "description": description,
            "revision": revision,
            "assessment": assessment,
        },
    )
    return goal_id


def record_action(store: ResearchStore, action: dict[str, Any], *, recovering: bool = False) -> None:
    goal_id = action.get("goal_id")
    execution = store.get("goal_execution", goal_id) if goal_id else None
    if execution is None or execution["revision"] != action.get("goal_revision"):
        return
    if recovering and action["id"] in execution["evidence_ids"]:
        if (
            execution.get("last_action_id") != action["id"]
            or execution.get("last_action_status") == action["status"]
        ):
            return
    if action["id"] not in execution["evidence_ids"]:
        execution["evidence_ids"].append(action["id"])
        execution["step_count"] += 1
    execution["last_action_id"] = action["id"]
    execution["last_action_status"] = action["status"]
    execution["resource_usage"] = {"tool_attempts": execution["step_count"]}
    store.put("goal_execution", goal_id, execution)


def unresolved_dependencies(store: ResearchStore, execution: dict[str, Any]) -> list[str]:
    unresolved = []
    for identifier in execution.get("depends_on", []):
        goal = store.get("goal", identifier) or {}
        if goal.get("status") != "completed" or goal.get("verification_status") not in {
            "supported",
            "subjective",
        }:
            unresolved.append(identifier)
    return unresolved

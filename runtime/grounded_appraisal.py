"""An experimental appraisal intervention at one site: bounded evidence attention."""

from __future__ import annotations

from typing import Any


def allocate_attention(
    states: list[dict[str, Any]], goals: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    relevant = {
        "lab:" + goal["success_criteria"]["opportunity"]
        for goal in goals
        if goal.get("status") == "active"
        and goal.get("success_criteria", {}).get("kind") == "environment_outcome"
    }
    project_goals = {
        goal["success_criteria"]["project"]
        for goal in goals
        if goal.get("status") == "active"
        and goal.get("success_criteria", {}).get("kind") == "world_outcome"
    }
    estimates = []
    for state in states:
        relevance = float(
            state["context"] in relevant
            or any(state["context"].startswith(f"world:{project}:") for project in project_goals)
        )
        uncertainty = state["uncertainty"]
        control = state["estimated_controllability"]
        surprise = state["last_surprise"]
        residual = state["last_prediction_error"]
        priority = relevance * (uncertainty + surprise) * control / 2
        item = {
            "context": state["context"],
            "goal_relevance": relevance,
            "outcome_minus_expectation": residual,
            "uncertainty": uncertainty,
            "estimated_controllability": control,
            "surprise": surprise,
            "attention_priority": priority,
            "observations": state["observations"],
        }
        if mode != "neutral":
            item["appraisal_expression"] = (
                "disappointment"
                if relevance and residual < 0
                else "satisfaction"
                if relevance and residual > 0
                else "unresolved"
                if uncertainty
                else "neutral"
            )
        estimates.append(item)
    ordered = sorted(estimates, key=lambda item: (-item["attention_priority"], item["context"]))
    selected = (
        ordered[0]["context"]
        if ordered and ordered[0]["attention_priority"] > 0 and mode in {"attention", "neutral"}
        else None
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "estimates": estimates,
        "selected_context": selected,
        "connection": "bounded_memory_retrieval_only",
        "method": "goal_relevance * (uncertainty + surprise) * estimated_controllability / 2",
        "interpretation": "experimental functional appraisal; no claim of felt emotion or human calibration",
    }

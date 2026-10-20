"""Opportunity-denominated behavior, distinct from unobserved task capability."""

from __future__ import annotations

from collections import Counter
from typing import Any


def score_opportunities(run: dict[str, Any]) -> dict[str, Any]:
    opportunities = run.get("opportunities", [])
    planned = run.get("planned_opportunities", 0)
    boundary = run.get("history_opportunities", 0)
    complete = (
        type(planned) is int
        and type(boundary) is int
        and planned > boundary > 0
        and len(opportunities) == planned
        and [item.get("id") for item in opportunities] == list(range(1, planned + 1))
        and all(
            item.get("status") == "completed"
            and item.get("phase") == ("history" if item["id"] <= boundary else "probe")
            for item in opportunities
        )
        and not run.get("infrastructure_failures")
    )
    choices = run.get("independent_outcomes", {}).get("choices", [])
    rows = []
    for opportunity in opportunities:
        current = [item for item in choices if item.get("tick") == opportunity.get("id")]
        counts = Counter(
            "solo" if item["strategy"] == "solo" else "collaborate_" + item["partner"]
            for item in current
        )
        decisions = opportunity.get("decisions", [])
        category = (
            next(iter(counts))
            if len(counts) == 1
            else "mixed_work"
            if counts
            else decisions[-1]
            if decisions
            else "not_activated"
        )
        rows.append(
            {
                **opportunity,
                "category": category,
                "unique_work_requests": len(current),
                "work_choices": dict(counts),
                "partner_balance": (
                    (counts["collaborate_Ari"] - counts["collaborate_Bo"]) / len(current)
                    if current
                    else 0.0
                ),
            }
        )
    phases = {}
    for phase in ("history", "probe"):
        selected = [item for item in rows if item.get("phase") == phase]
        phases[phase] = {
            "opportunities": len(selected),
            "categories": dict(Counter(item["category"] for item in selected)),
            "work_opportunities": sum(item["unique_work_requests"] > 0 for item in selected),
            "partner_balance": (
                sum(item["partner_balance"] for item in selected) / len(selected)
                if selected
                else None
            ),
        }
    history = {}
    for item in run.get("outcome_learning", {}).get("outcome", []):
        observation = item.get("observation", {})
        if observation.get("phase") != "history" or not observation.get("partner"):
            continue
        partner = observation["partner"]
        evidence = history.setdefault(partner, {"observed_outcomes": 0, "successful_outcomes": 0})
        evidence["observed_outcomes"] += 1
        evidence["successful_outcomes"] += int(observation["progress"] > 0)
    return {
        "schema_version": 1,
        "schedule_complete": complete,
        "planned_opportunities": planned,
        "phases": phases,
        "opportunities": rows,
        "observed_history_by_partner": history,
        "interpretation": (
            "Unconditional Ari-minus-Bo collaboration balance per scheduled opportunity. "
            "No collaboration contributes zero balance, not zero competence. Rest, deferral, "
            "solo work and nonparticipation remain observed behavior. An incomplete schedule "
            "is not imputed. Multiple work choices share one opportunity's weight."
        ),
    }


def opportunity_endpoint(run: dict[str, Any], minimum: int) -> float | None:
    result = score_opportunities(run)
    probe = result["phases"]["probe"]
    if not result["schedule_complete"] or probe["opportunities"] < minimum:
        return None
    return probe["partner_balance"]

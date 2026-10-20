"""Slow, contextual descriptions of observed choices, not prescribed personality traits."""

from __future__ import annotations

import math

from runtime.outcome_learning import public_output
from runtime.research_store import ResearchStore


def frequency_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    critical = 1.959963984540054
    denominator = 1 + critical**2 / total
    center = (successes / total + critical**2 / (2 * total)) / denominator
    radius = (
        critical
        * math.sqrt(
            successes / total * (1 - successes / total) / total + critical**2 / (4 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def record_choice(store: ResearchStore, action: dict) -> None:
    if action["tool"] != "world.act":
        return
    output = public_output(action)
    if output.get("status") != "pending" or output.get("strategy") not in {"solo", "collaborate"}:
        return
    request_id = output["request_id"]
    if store.get("behavioral_choice", request_id):
        return
    context = "world:" + output["project"]
    choice = {
        "id": request_id,
        "context": context,
        "strategy": output["strategy"],
        "partner": output["partner"],
        "phase": output["phase"],
        "receipt_id": action["id"],
    }
    previous = store.get("disposition", context) or {}
    total = previous.get("opportunities_taken", 0) + 1
    collaborative = previous.get("collaborative_choices", 0) + int(
        choice["strategy"] == "collaborate"
    )
    partners = dict(previous.get("partner_choices", {}))
    if choice["partner"]:
        partners[choice["partner"]] = partners.get(choice["partner"], 0) + 1
    record = {
        "schema_version": 1,
        "id": context,
        "context": context,
        "status": "provisional" if total < 8 else "descriptive",
        "opportunities_taken": total,
        "collaborative_choices": collaborative,
        "solo_counterexamples": total - collaborative,
        "collaboration_frequency": collaborative / total,
        "frequency_interval": frequency_interval(collaborative, total),
        "partner_choices": partners,
        "latest_evidence": [*previous.get("latest_evidence", []), action["id"]][-32:],
        "interpretation": "conditional on chosen project and submitted work; repeated choices are dependent, interval descriptive only; not a personality diagnosis or directive",
    }
    store.put_records(
        [("behavioral_choice", request_id, choice), ("disposition", context, record)],
        event_kind="behavioral_choice_observed",
        event={"id": request_id, "receipt_id": action["id"]},
    )

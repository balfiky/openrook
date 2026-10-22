"""Plain-language explanations for a completed pipeline turn."""

from __future__ import annotations

from typing import Any

from dataclasses import asdict
from runtime.debug.labels import social_move_phrase, strategy_label, strategy_reason_label
from runtime.life_history import DEFAULT_DRIVES


def explain_turn(debug: Any) -> dict[str, str]:
    """Summarize the most user-legible parts of a DebugState."""
    appraisal = getattr(debug, "appraisal_frame", None)
    relationship = getattr(debug, "relationship_context", None)
    tool_trace = getattr(debug, "tool_trace", None)
    life = getattr(debug, "life_history_context", {}) or {}
    influence = getattr(debug, "life_influence", None)
    life_effects = getattr(debug, "life_influence_effects", {}) or {}
    defense = getattr(debug, "defense_activation", None)
    self_check_issues = getattr(debug, "self_check_issues", []) or []
    decision = getattr(debug, "turn_decision", None)

    return {
        "interpretation": _interpretation(appraisal),
        "strategy": _strategy(debug, appraisal),
        "state": _state({}, asdict(decision.state)) if decision else _state(getattr(debug, "baseline_shift_applied", {}) or {}, getattr(debug, "modulator_snapshot", {}) or {}),
        "delivery": getattr(debug, "delivery_outcome", "unrecorded"),
        "memory": _memory(
            relationship,
            getattr(debug, "retrieved_memories", []) or [],
            getattr(debug, "semantic_memories", []) or [],
        ),
        "life_history": _life_history(life, influence, life_effects),
        "tools": _tools(tool_trace),
        "limits": _limits(defense, self_check_issues, getattr(debug, "dignity_outcome", "clear")),
    }


def _interpretation(appraisal: Any) -> str:
    if appraisal is None:
        return "I did not record a social interpretation for this turn."
    target = getattr(appraisal, "primary_target", "unknown")
    move = social_move_phrase(getattr(appraisal, "social_move", "inform"))
    if target == "external":
        return f"I interpreted this as {move} about something outside the relationship."
    if target == "assistant":
        return f"I interpreted this as {move} directed at me."
    if target == "self":
        return f"I interpreted this as {move} about your own situation."
    if target == "shared_problem":
        return "I interpreted this as a request about a shared problem."
    return f"I interpreted this as {move} with no clear target."


def _strategy(debug: Any, appraisal: Any) -> str:
    decision = getattr(debug, "turn_decision", None)
    if decision is not None:
        parts = [f"{decision.policy_version}, turn {decision.turn_id}, seed {decision.seed}; utility {decision.utility:.4f}."]
        for c in decision.components:
            if c.best_grant is None:
                parts.append(f"{c.component.id}: declined because the component is infeasible; willingness 0.")
                continue
            causes = sorted(c.contributions, key=lambda pair: abs(pair[1]), reverse=True)[:3]
            parts.append(f"{c.component.id}: {'granted' if c.granted else 'declined'}, willingness {c.willingness:.3f}, margin {c.margin:.4f}; "
                         + ", ".join(f"{name} {value:+.3f}" for name, value in causes) + ".")
        parts.append(f"Decision-time injury {decision.relationship.h:.3f}, grievance {decision.relationship.r:.3f}, energy {decision.state.energy:.3f}, trust {decision.state.trust:.3f}.")
        return " ".join(parts)
    strategy = getattr(debug, "response_strategy", "") or ""
    if not strategy:
        return "No explicit response strategy was selected."
    label = strategy_label(strategy).lower()
    trace = getattr(debug, "strategy_trace", None)
    reason = strategy_reason_label(getattr(trace, "matched_rule", ""))
    if reason:
        return f"I chose {label} because {reason}."
    if appraisal is not None and getattr(appraisal, "vulnerability", 0.0) >= 0.5:
        return f"I chose {label} because the message suggested a need for care."
    if appraisal is not None and getattr(appraisal, "inferred_intent", "") == "seek_action":
        return f"I chose {label} because you asked for action."
    return f"I chose {label} based on the message and my modeled state."


def _state(baseline_shift: dict[str, float], snapshot: dict[str, float]) -> str:
    if not snapshot:
        return "No modeled state was recorded."
    names = {
        "arousal": "activation",
        "valence": "tone",
        "certainty": "confidence",
        "bonding": "relationship warmth",
        "energy": "energy",
        "resolution": "unfinished tension",
    }
    shifted = [
        names.get(name, "state")
        for name, value in baseline_shift.items()
        if abs(float(value or 0.0)) > 0.001
    ]
    arousal = snapshot.get("arousal", 0.0)
    valence = snapshot.get("valence", 0.0)
    if shifted:
        return (
            f"The starting settings shifted for {', '.join(shifted[:3])}; "
            f"modeled activation is {arousal:.2f} and tone is {valence:.2f}."
        )
    return (
        f"Modeled activation is {arousal:.2f}, tone is {valence:.2f}, "
        f"and energy is {snapshot.get('energy', 0.0):.2f}. These are numeric settings, not feelings."
    )


def _memory(
    relationship: Any,
    retrieved_memories: list[Any],
    semantic_memories: list[Any],
) -> str:
    pieces: list[str] = []
    if semantic_memories:
        count = len(semantic_memories)
        noun = "thing" if count == 1 else "things"
        pieces.append(_with_examples(
            f"Recalled {count} {noun} from saved memory",
            semantic_memories,
            ("summary", "content", "topic"),
        ))
    if retrieved_memories:
        count = len(retrieved_memories)
        noun = "memory" if count == 1 else "memories"
        pieces.append(_with_examples(
            f"Used {count} long-term {noun} of past interactions",
            retrieved_memories,
            ("summary", "topic"),
        ))
    if relationship is not None:
        loops = getattr(relationship, "active_loops", []) or []
        events = getattr(relationship, "recent_events", []) or []
        summary = str(getattr(relationship, "summary", "") or "").strip()
        if loops:
            noun = "matter" if len(loops) == 1 else "matters"
            pieces.append(_with_examples(
                f"Relationship memory supplied {len(loops)} unfinished {noun}",
                loops,
                ("description", "topic"),
            ))
        if events:
            noun = "event" if len(events) == 1 else "events"
            pieces.append(_with_examples(
                f"Relationship memory supplied {len(events)} past relationship {noun}",
                events,
                ("summary", "topic"),
            ))
        if summary and not loops and not events:
            pieces.append(f"Used a saved relationship summary: {_short_text(summary)}.")
    return " ".join(pieces) or "No saved details about you were used."


def _short_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text[:177].rstrip() + "…" if len(text) > 180 else text.rstrip(".")


def _with_examples(prefix: str, items: list[Any], fields: tuple[str, ...]) -> str:
    examples = []
    for item in items[:3]:
        for field in fields:
            value = item.get(field) if isinstance(item, dict) else getattr(item, field, "")
            if value:
                examples.append(_short_text(value))
                break
    if not examples:
        return prefix + "."
    joiner = ", including: " if len(items) > 3 else ": "
    return prefix + joiner + "; ".join(examples) + "."


def _tools(tool_trace: Any) -> str:
    if tool_trace is None:
        return "No tools were considered."
    executed = getattr(tool_trace, "executed_results", []) or []
    if not executed:
        proposed = getattr(tool_trace, "proposed_intents", []) or []
        return "Tools were considered but none were used." if proposed else "No tools were used."
    count = len(executed)
    noun = "tool" if count == 1 else "tools"
    return f"Used {count} {noun}; their results were included in the response context."


def _drive_has_drifted(drive: dict[str, Any]) -> bool:
    baseline = float(drive.get("baseline", DEFAULT_DRIVES.get(drive.get("name"), (0.5, ""))[0]))
    delta = (
        float(drive["value"]) - baseline
        if "value" in drive
        else float(drive.get("delta", 0.0))
    )
    return abs(delta) > 1e-9


def _life_context_summary(life: dict[str, Any]) -> str:
    if life.get("error"):
        return ""
    # all_drives includes seed settings as well as drift too small for the
    # compact drives list. Count only changes actually loaded for this turn.
    drives = sum(
        _drive_has_drifted(drive)
        for drive in life.get("all_drives") or life.get("drives") or []
    )
    if not drives and not any(life.get(key) for key in ("beliefs", "experiences", "recent_evolution")):
        return ""
    parts = []
    if life.get("constitution"):
        parts.append("saved guiding principles")
    for key, singular, plural in (
        ("beliefs", "belief", "beliefs"),
        ("recent_evolution", "recent change", "recent changes"),
        ("self_traits", "learned trait", "learned traits"),
        ("dispositions", "learned tendency", "learned tendencies"),
        ("experiences", "saved learning record", "saved learning records"),
    ):
        count = len(life.get(key) or [])
        if count:
            parts.append(f"{count} {singular if count == 1 else plural}")
    if drives:
        parts.append(f"{drives} saved {'priority' if drives == 1 else 'priorities'}")
    if not parts:
        return ""
    return "Life History contributed " + ", ".join(parts) + "."


def _life_history(life: dict[str, Any], influence: Any, effects: dict[str, Any]) -> str:
    if life.get("error"):
        return "No Life History context contributed to this turn."
    summary = _life_context_summary(life)
    if not summary:
        return "Only the starting character settings applied."
    details = []
    for key, label, fields in (
        ("beliefs", "Saved beliefs", ("statement",)),
        ("recent_evolution", "Recent changes", ("reason", "after_state")),
    ):
        items = life.get(key) or []
        if items and any(any(item.get(field) for field in fields) for item in items):
            details.append(_with_examples(label, items, fields))
    if effects:
        details.append("This context also adjusted how I approached the turn within the configured limits.")
    elif influence is not None and not getattr(influence, "is_neutral", True):
        details.append("Saved priorities were considered without a recorded change to the response rules.")
    return " ".join([summary, *details])


def _limits(defense: Any, self_check_issues: list[str], dignity_outcome: str = "clear") -> str:
    if dignity_outcome == "fallback":
        return "A short fallback was used because the draft violated the character's stance and one correction did not resolve it."
    if defense is None and not self_check_issues:
        return "No response safeguard or self-check correction was triggered."
    parts: list[str] = []
    if defense is not None:
        parts.append("A response safeguard adjusted how I approached this reply.")
    if self_check_issues:
        count = len(self_check_issues)
        parts.append(
            f"A self-check flagged {count} {'concern' if count == 1 else 'concerns'} "
            "about the draft response."
        )
    return " ".join(parts)

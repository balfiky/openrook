"""Derived affect and agency policy.

This layer deliberately does not add new primary emotional knobs. It derives
emotion-like labels from the compact modulator state, social appraisal, and
message semantics, then translates them into a turn-level willingness stance.
"""

from __future__ import annotations

import re

from core.types import (
    AffectSignal,
    AffectState,
    AppraisalFrame,
    ModulatorState,
    PersonProfile,
)


_COMPARISON_RE = re.compile(
    r"\b(they|them|he|she|someone|everyone|others?)\b.*\b(get|gets|got|have|has|had|can|could|better|more)\b"
    r"|\b(jealous|envy|envious|why not me|better than me|more than me)\b",
    re.IGNORECASE,
)


def resolve_affect(
    *,
    text: str,
    state: ModulatorState,
    appraisal: AppraisalFrame,
    person: PersonProfile | None = None,
    agent_goal_comparison: bool | None = None,
) -> AffectState:
    """Compute derived affect signals for this turn."""
    signals: list[AffectSignal] = []

    low_valence = 1.0 - state.valence
    high_arousal = state.arousal
    low_certainty = 1.0 - state.certainty
    low_energy = 1.0 - state.energy
    trust = person.trust if person else 0.5

    anger = _clamp(
        appraisal.blame * 0.45
        + low_valence * 0.3
        + high_arousal * 0.2
        + (0.15 if appraisal.targets_assistant and appraisal.social_move in {"attack", "complaint"} else 0.0)
    )
    if anger >= 0.35:
        evidence = ["blame", "low valence", "high arousal"]
        if appraisal.targets_assistant:
            evidence.append("assistant-targeted")
        signals.append(AffectSignal("anger", anger, evidence))

    rage = _clamp((anger - 0.55) * 1.3 + max(0.0, state.arousal - 0.8) + max(0.0, 0.25 - state.valence))
    if rage >= 0.35:
        signals.append(
            AffectSignal(
                "rage",
                rage,
                ["anger above threshold", "extreme activation", "very low valence"],
            )
        )

    hurt = _clamp(
        appraisal.expectation_violation * 0.45
        + low_valence * 0.25
        + max(0.0, 0.5 - state.bonding) * 0.3
        + (0.15 if appraisal.targets_assistant and appraisal.expectation_violation > 0 else 0.0)
    )
    if hurt >= 0.35:
        signals.append(
            AffectSignal(
                "hurt",
                hurt,
                ["expectation violation", "low valence", "bonding pressure"],
            )
        )

    envy = 0.0
    if ((agent_goal_comparison is True or (agent_goal_comparison is None and _COMPARISON_RE.search(text))) and appraisal.social_move == "complaint"
            and state.valence < 0.4 and not re.search(r"\bnot\s+(?:jealous|envious)\b", text, re.IGNORECASE)):
        envy = _clamp(0.45 + low_valence * 0.25 + high_arousal * 0.15 + low_certainty * 0.15)
    if envy >= 0.35:
        signals.append(
            AffectSignal(
                "envy",
                envy,
                ["social comparison", "desire/status pressure", "low valence"],
            )
        )

    caution = _clamp(
        low_certainty * 0.45
        + appraisal.expectation_violation * 0.25
        + state.resolution * 0.15
        + (0.15 if appraisal.inferred_intent == "seek_action" else 0.0)
    )
    if caution >= 0.35:
        signals.append(
            AffectSignal(
                "caution",
                caution,
                ["uncertainty", "risk/expectation pressure", "action request"],
            )
        )

    affection = _clamp(
        state.valence * 0.35
        + state.bonding * 0.35
        + appraisal.affiliation_bid * 0.2
        + max(0.0, trust - 0.5) * 0.1
    )
    if affection >= 0.55:
        signals.append(
            AffectSignal(
                "affection",
                affection,
                ["positive valence", "bonding", "affiliation bid"],
            )
        )

    exhaustion = _clamp(low_energy * 0.75 + max(0.0, 0.35 - state.valence) * 0.25)
    if exhaustion >= 0.55:
        signals.append(AffectSignal("exhaustion", exhaustion, ["low energy", "low valence"]))

    severity_rank = {
        "rage": 6,
        "hurt": 4,
        "anger": 3,
        "envy": 3,
        "caution": 2,
        "exhaustion": 2,
        "affection": 1,
    }
    signals.sort(
        key=lambda item: (item.intensity, severity_rank.get(item.name, 0)),
        reverse=True,
    )
    primary = signals[0].name if signals else "neutral"
    return AffectState(primary=primary, signals=signals)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

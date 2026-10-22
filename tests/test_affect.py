"""Tests for display-only affect observations."""

from __future__ import annotations

from core.affect import resolve_affect
from core.types import AppraisalFrame, ModulatorState, PersonProfile


def _appraisal(**overrides) -> AppraisalFrame:
    return AppraisalFrame(**overrides)


def test_assistant_targeted_blame_describes_anger():
    appraisal = _appraisal(
        primary_target="assistant",
        social_move="attack",
        targets_assistant=True,
        blame=0.9,
    )
    affect = resolve_affect(
        text="I hate you because you are too slow.",
        state=ModulatorState(arousal=0.9, valence=0.15, certainty=0.4, bonding=0.35),
        appraisal=appraisal,
        person=PersonProfile(person_id="u", trust=0.4),
    )

    assert affect.signal("anger") > 0.6


def test_extreme_hostility_describes_rage_without_choosing():
    appraisal = _appraisal(
        primary_target="assistant",
        social_move="attack",
        targets_assistant=True,
        blame=1.0,
        expectation_violation=0.8,
    )
    affect = resolve_affect(
        text="You betrayed me. I hate you. Do it now.",
        state=ModulatorState(arousal=1.0, valence=0.0, certainty=0.2, bonding=0.2),
        appraisal=appraisal,
        person=PersonProfile(person_id="u", trust=0.2),
    )

    assert affect.primary == "rage"
    assert affect.signal("rage") >= 0.65


def test_social_comparison_derives_envy_without_primary_knob():
    appraisal = _appraisal(primary_target="external", social_move="complaint")
    affect = resolve_affect(
        text="Everyone else gets better tools than me. Why not me?",
        state=ModulatorState(arousal=0.65, valence=0.25, certainty=0.4),
        appraisal=appraisal,
    )

    assert affect.signal("envy") >= 0.35
    assert any("social comparison" in signal.evidence for signal in affect.signals if signal.name == "envy")


def test_risky_uncertain_action_describes_caution():
    appraisal = _appraisal(
        primary_target="shared_problem",
        social_move="request",
        inferred_intent="seek_action",
        expectation_violation=0.8,
    )
    affect = resolve_affect(
        text="Unexpected risk. Delete production carefully.",
        state=ModulatorState(arousal=0.7, valence=0.5, certainty=0.15, resolution=0.6),
        appraisal=appraisal,
    )

    assert affect.signal("caution") >= 0.65

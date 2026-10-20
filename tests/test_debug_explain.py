from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.life_influence import LifeInfluence
from core.types import (
    AppraisalFrame,
    LongTermEntry,
    OpenLoop,
    RelationshipContext,
    RelationshipEvent,
    SemanticMemoryEntry,
    StrategyDecisionTrace,
)
from pipeline import DebugState
from runtime.debug.explain import explain_turn
from runtime.life_history import DEFAULT_DRIVES


def test_external_distress_explanation():
    debug = DebugState(
        appraisal_frame=AppraisalFrame(
            primary_target="external",
            social_move="vulnerability",
            vulnerability=0.8,
        ),
        response_strategy="validate",
        strategy_trace=StrategyDecisionTrace(
            selected="validate",
            matched_rule="external_vulnerability_validate",
        ),
        modulator_snapshot={"arousal": 0.6, "valence": 0.3, "energy": 0.9},
    )

    explanation = explain_turn(debug)

    assert "outside the relationship" in explanation["interpretation"]
    assert "acknowledging your experience" in explanation["strategy"]
    assert "distress about an outside situation" in explanation["strategy"]
    assert "external_vulnerability_validate" not in explanation["strategy"]
    assert "Modeled activation" in explanation["state"]


def test_low_trust_boundary_explanation():
    debug = DebugState(
        appraisal_frame=AppraisalFrame(
            primary_target="assistant",
            social_move="attack",
            targets_assistant=True,
        ),
        response_strategy="set_boundary",
        strategy_trace=StrategyDecisionTrace(
            selected="set_boundary",
            matched_rule="assistant_targeted_attack_low_trust",
        ),
        modulator_snapshot={"arousal": 0.8, "valence": 0.2, "energy": 0.7},
    )

    explanation = explain_turn(debug)

    assert "directed at me" in explanation["interpretation"]
    assert "setting a boundary" in explanation["strategy"]
    assert "trust was limited" in explanation["strategy"]
    assert "assistant_targeted_attack_low_trust" not in explanation["strategy"]


def test_life_history_effect_explanation():
    debug = DebugState(
        life_history_context={
            "beliefs": [{"key": "repair"}],
            "drives": [{"name": "repair", "delta": 0.03}],
            "recent_evolution": [],
        },
        life_influence=LifeInfluence(repair_pressure=0.03),
        life_influence_effects={"strategy_tiebreak_used": True},
        modulator_snapshot={"arousal": 0.5, "valence": 0.5, "energy": 1.0},
    )

    explanation = explain_turn(debug)

    assert "Life History contributed" in explanation["life_history"]
    assert "within the configured limits" in explanation["life_history"]


def test_tool_and_no_tool_explanations():
    no_tool = DebugState(modulator_snapshot={"arousal": 0.5, "valence": 0.5, "energy": 1.0})
    assert explain_turn(no_tool)["tools"] == "No tools were considered."

    used_tool = DebugState(
        tool_trace=SimpleNamespace(
            executed_results=[SimpleNamespace(tool_name="fs.read_file")],
            proposed_intents=[],
        ),
        modulator_snapshot={"arousal": 0.5, "valence": 0.5, "energy": 1.0},
    )
    assert "Used 1 tool" in explain_turn(used_tool)["tools"]
    assert "fs.read_file" not in explain_turn(used_tool)["tools"]


@pytest.mark.parametrize("learned", [False, True])
def test_semantic_recall_explanation_includes_saved_preference_and_episode(learned):
    debug = DebugState(
        semantic_memories=[
            SemanticMemoryEntry(kind="preference", summary="You prefer bullet points."),
            SemanticMemoryEntry(kind="episode", summary="We discussed your work deadline."),
        ],
        life_history_context={
            "all_drives": [
                {"name": name, "value": baseline}
                for name, (baseline, _) in DEFAULT_DRIVES.items()
            ],
            "beliefs": [{"statement": "Clear explanations help."}] if learned else [],
        },
    )

    explanation = explain_turn(debug)["memory"]

    assert "Recalled 2 things" in explanation
    assert "You prefer bullet points" in explanation
    assert "We discussed your work deadline" in explanation
    assert "No relationship memory" not in explanation
    assert "No long-term memories" not in explanation
    assert "Life History" not in explanation
    assert "priorities" not in explanation
    assert "Clear explanations help" not in explanation


def test_memory_explanation_reports_all_loaded_sources_without_counting_unloaded_loops():
    debug = DebugState(
        semantic_memories=[SemanticMemoryEntry(summary="You prefer concise replies.")],
        retrieved_memories=[LongTermEntry(summary="You appreciated a clear answer.")],
        relationship_context=RelationshipContext(
            active_loops=[OpenLoop(description="Follow up on the deadline.")],
            recent_events=[RelationshipEvent(summary="We resolved a misunderstanding.")],
            open_loop_count=12,
        ),
        life_history_context={
            "beliefs": [{"statement": "Clear explanations help."}],
            "drives": [{"name": "curiosity", "delta": 0.03}],
            "recent_evolution": [{"reason": "Learned from feedback."}],
        },
    )

    explanation = explain_turn(debug)

    assert "Recalled 1 thing" in explanation["memory"]
    assert "1 long-term memory" in explanation["memory"]
    assert "You appreciated a clear answer" in explanation["memory"]
    assert "1 unfinished matter" in explanation["memory"]
    assert "Follow up on the deadline" in explanation["memory"]
    assert "We resolved a misunderstanding" in explanation["memory"]
    assert "12" not in explanation["memory"]
    assert "Life History" not in explanation["memory"]
    assert "Clear explanations help" not in explanation["memory"]
    assert "Learned from feedback" not in explanation["memory"]
    assert "1 belief, 1 recent change, 1 saved priority" in explanation["life_history"]
    assert "Clear explanations help" in explanation["life_history"]
    assert "Learned from feedback" in explanation["life_history"]


def test_memory_explanation_handles_relationship_summary_and_empty_context():
    summary = DebugState(relationship_context=RelationshipContext(summary="We recently repaired trust."))
    assert "saved relationship summary: We recently repaired trust" in explain_turn(summary)["memory"]

    empty = DebugState(
        relationship_context=RelationshipContext(open_loop_count=4),
        life_history_context={"counts": {"beliefs": 20}},
    )
    assert explain_turn(empty)["memory"] == "No saved details about you were used."


@pytest.mark.parametrize("life, expected", [
    ({"beliefs": [{"statement": "Clear explanations help."}]}, "1 belief"),
    ({"all_drives": [{"name": "curiosity", "value": 0.51}]}, "1 saved priority"),
    ({"drives": [{"name": "curiosity", "delta": -0.03}]}, "1 saved priority"),
    ({"drives": [{"name": "curiosity", "value": 0.5, "baseline": 0.4}]}, "1 saved priority"),
    ({"recent_evolution": [{"reason": "Learned from feedback."}]}, "1 recent change"),
    ({"experiences": [{"source_title": "Learning notes"}]}, "1 saved learning record"),
    ({
        "constitution": "Be clear.",
        "beliefs": [{"statement": "Ask useful questions."}],
        "dispositions": ["Ask follow-up questions."],
    }, "saved guiding principles, 1 belief, 1 learned tendency"),
])
def test_learned_life_history_is_reported_only_in_its_own_section(life, expected):
    explanation = explain_turn(DebugState(life_history_context=life))
    assert explanation["memory"] == "No saved details about you were used."
    assert f"Life History contributed {expected}" in explanation["life_history"]


@pytest.mark.parametrize("life", [
    {},
    {"constitution": "Be clear."},
    {"all_drives": [{"name": "curiosity"}]},
    {"drives": [{"name": "curiosity", "delta": 0.0}]},
    {"all_drives": [{"name": "curiosity", "value": 0.7, "baseline": 0.7}]},
    {"counts": {"beliefs": 20, "experiences": 7, "evolution_events": 4}},
])
def test_starting_settings_and_unloaded_history_do_not_claim_contribution(life):
    explanation = explain_turn(DebugState(life_history_context=life))

    assert explanation["memory"] == "No saved details about you were used."
    assert explanation["life_history"] == "Only the starting character settings applied."


@pytest.mark.parametrize("drift", [0.0, -0.01, 0.03])
def test_only_drifted_drives_are_counted_among_seeded_drives(drift):
    drives = [
        {"name": name, "value": baseline + (drift if name == "curiosity" else 0.0)}
        for name, (baseline, _) in DEFAULT_DRIVES.items()
    ]
    explanation = explain_turn(DebugState(life_history_context={
        "all_drives": drives,
        "drives": [drives[0]] if abs(drift) >= 0.02 else [],
    }))

    assert explanation["memory"] == "No saved details about you were used."
    assert explanation["life_history"] == (
        "Life History contributed 1 saved priority."
        if drift else "Only the starting character settings applied."
    )


def test_request_explanation_uses_plain_labels_and_correct_article():
    debug = DebugState(
        appraisal_frame=AppraisalFrame(
            primary_target="assistant",
            social_move="request",
            inferred_intent="seek_action",
        ),
        response_strategy="practical_help",
        strategy_trace=StrategyDecisionTrace(
            selected="practical_help",
            matched_rule="actionable_request_practical_help",
        ),
    )
    explanation = explain_turn(debug)

    assert explanation["interpretation"] == "I interpreted this as a request directed at me."
    assert explanation["strategy"] == (
        "I chose practical help because you asked for help with something that seemed actionable."
    )
    assert "practical_help" not in explanation["strategy"]
    assert debug.strategy_trace.matched_rule == "actionable_request_practical_help"


def test_unknown_strategy_rule_does_not_leak_identifier():
    debug = DebugState(
        response_strategy="practical_help",
        strategy_trace=StrategyDecisionTrace(matched_rule="future_internal_rule"),
    )
    assert "practical help" in explain_turn(debug)["strategy"]
    assert "future_internal_rule" not in explain_turn(debug)["strategy"]


def test_safeguard_explanation_keeps_raw_identifiers_in_debug_only():
    debug = DebugState(
        defense_activation=SimpleNamespace(defense_type="internal_defense"),
        self_check_issues=["claims_durable_change_without_ledger_evidence"],
    )
    explanation = explain_turn(debug)["limits"]
    assert "response safeguard" in explanation
    assert "1 concern" in explanation
    assert "internal_defense" not in explanation
    assert "claims_durable_change_without_ledger_evidence" not in explanation
    assert debug.self_check_issues == ["claims_durable_change_without_ledger_evidence"]

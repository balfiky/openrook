"""Tests of the TEST-ONLY keyword reader (``tests/support/keyword_reader.py``).

These describe the perception the runtime had before every message was read by
the model: particular English words recognised offline, with fixed numbers.
They are kept so the fixture reader that older pipeline tests run on stays what
those tests were written against. They say nothing about runtime behaviour:
the runtime's contract is in ``tests/test_reading_contract.py``.
"""

from __future__ import annotations

import pytest

from core.social import has_concrete_task, read_social_signals
from core.types import AppraisalFrame, DetectedEmotion
from core.willingness import Perception
from openrook_tools.roleplay_eval.scenarios import SINCERE
from tests.support.keyword_reader import (
    _detect_via_rules, appraise_message, brevity_preference, detect_emotion, observe,
    reconcile_social_appraisal,
)


# --- moved from tests/test_contagion.py ---


class TestDetectEmotion:
    def test_neutral_text(self):
        result = detect_emotion("Let's look at the code.")
        assert 0.4 <= result.valence <= 0.6
        assert 0.4 <= result.arousal <= 0.6

    def test_empty_text(self):
        result = detect_emotion("")
        assert result.valence == 0.5
        assert result.arousal == 0.5

    def test_angry_text(self):
        result = detect_emotion("I'm so angry and frustrated right now")
        assert result.valence < 0.4
        assert result.arousal > 0.6

    def test_happy_text(self):
        result = detect_emotion("I'm really happy and excited about this!")
        assert result.valence > 0.6
        assert result.arousal > 0.5

    def test_sad_text(self):
        result = detect_emotion("I feel sad and lonely today...")
        assert result.valence < 0.4
        assert result.arousal < 0.5

    def test_excited_text(self):
        result = detect_emotion("This is amazing! I'm thrilled! Fantastic work!")
        assert result.valence > 0.7
        assert result.arousal > 0.6

    def test_calm_positive(self):
        result = detect_emotion("I'm feeling peaceful and content")
        assert result.valence > 0.5
        assert result.arousal < 0.5

    def test_anxious_text(self):
        result = detect_emotion("I'm really anxious and nervous about this")
        assert result.valence < 0.5
        assert result.arousal > 0.5

    def test_caps_boost_arousal(self):
        calm = detect_emotion("this is great")
        shouting = detect_emotion("THIS IS GREAT")
        assert shouting.arousal > calm.arousal

    def test_exclamation_boost_arousal(self):
        plain = detect_emotion("wow")
        excited = detect_emotion("wow!!!")
        assert excited.arousal > plain.arousal

    def test_ellipsis_slight_negative(self):
        plain = detect_emotion("okay")
        trailing = detect_emotion("okay...")
        assert trailing.valence <= plain.valence

    def test_betrayal_keywords(self):
        result = detect_emotion("I feel betrayed and deceived")
        assert result.valence < 0.3
        assert result.arousal > 0.5

    def test_gratitude_keywords(self):
        result = detect_emotion("Thank you so much, I'm grateful")
        assert result.valence > 0.5

    def test_exhaustion_low_arousal(self):
        result = detect_emotion("I'm so tired and exhausted")
        assert result.arousal < 0.5
        assert result.valence < 0.5

    def test_mixed_signals(self):
        result = detect_emotion("I'm happy but also a bit nervous")
        # Should land somewhere in the middle
        assert isinstance(result.arousal, float)
        assert isinstance(result.valence, float)

    def test_output_always_clamped(self):
        # Extreme input
        result = detect_emotion(
            "FURIOUS ANGRY LIVID ENRAGED!!! " * 10
        )
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.valence <= 1.0

    def test_returns_detected_emotion_type(self):
        result = detect_emotion("hello")
        assert isinstance(result, DetectedEmotion)

    # --- Aggressive detection tests ---

    def test_insult_detection(self):
        result = detect_emotion("you are stupid")
        assert result.valence < 0.3, f"Insult valence too high: {result.valence}"
        assert result.arousal > 0.6, f"Insult arousal too low: {result.arousal}"

    def test_profanity_detection(self):
        result = detect_emotion("this is bullshit")
        assert result.valence < 0.3, f"Profanity valence too high: {result.valence}"
        assert result.arousal > 0.6, f"Profanity arousal too low: {result.arousal}"

    def test_all_caps_aggression(self):
        result = detect_emotion("YOU ARE AN IDIOT AND I HATE THIS")
        assert result.arousal > 0.8, f"All-caps arousal too low: {result.arousal}"
        assert result.valence < 0.2, f"All-caps valence too high: {result.valence}"

    def test_hostile_command(self):
        result = detect_emotion("shut up and go away")
        assert result.valence < 0.2, f"Hostile valence too high: {result.valence}"
        assert result.arousal > 0.7, f"Hostile arousal too low: {result.arousal}"

    def test_compliment_detection(self):
        result = detect_emotion("you are so smart and talented")
        assert result.valence > 0.6, f"Compliment valence too low: {result.valence}"

    def test_casual_greeting_mildly_positive(self):
        result = detect_emotion("hey, how's it going?")
        assert result.valence >= 0.5, f"Greeting valence too low: {result.valence}"

    def test_terrible_awful_detection(self):
        result = detect_emotion("this is terrible and awful")
        assert result.valence < 0.3, f"Terrible/awful valence too high: {result.valence}"
        assert result.arousal > 0.4, f"Terrible/awful arousal too low: {result.arousal}"


# --- moved from tests/test_appraisal.py ---


def _detected(
    *,
    arousal: float = 0.5,
    valence: float = 0.5,
    certainty: float = 0.5,
    intensity: float = 0.5,
) -> DetectedEmotion:
    return DetectedEmotion(
        arousal=arousal,
        valence=valence,
        certainty=certainty,
        intensity=intensity,
    )


class TestAppraisal:
    def test_detects_direct_attack_on_assistant(self):
        appraisal = appraise_message(
            "You are useless and this answer is terrible.",
            _detected(arousal=0.8, valence=0.1, intensity=0.8),
        )
        assert appraisal.targets_assistant is True
        assert appraisal.primary_target == "assistant"
        assert appraisal.social_move == "attack"
        assert appraisal.inferred_intent == "harm"

    def test_external_distress_is_not_treated_as_attack(self):
        appraisal = appraise_message(
            "I'm furious about work, not at you.",
            _detected(arousal=0.9, valence=0.2, intensity=0.8),
        )
        assert appraisal.targets_assistant is False
        assert appraisal.primary_target == "external"
        assert appraisal.social_move in {"complaint", "vulnerability"}

    def test_mixed_affect_detected(self):
        appraisal = appraise_message(
            "I'm happy it worked, but I'm scared about what comes next.",
            _detected(arousal=0.7, valence=0.45, intensity=0.7),
        )
        assert appraisal.mixed_affect is True
        assert 0 < appraisal.vulnerability < 1

    def test_apology_is_affiliative_repair(self):
        appraisal = appraise_message(
            "I'm sorry, I handled that badly.",
            _detected(arousal=0.4, valence=0.35, intensity=0.4),
        )
        assert appraisal.social_move == "apology"
        assert appraisal.inferred_intent == "repair"
        assert appraisal.perception.repair_attempt > 0
        assert appraisal.perception.repair_credibility == 0
        assert appraisal.perception.reliability == 0

    def test_assistant_addressed_apology_targets_assistant(self):
        appraisal = appraise_message(
            "I'm sorry for snapping at you about the deadline.",
            _detected(arousal=0.5, valence=0.35, intensity=0.5),
        )
        assert appraisal.social_move == "apology"
        assert appraisal.targets_assistant is True
        assert appraisal.primary_target == "assistant"

    def test_imperative_challenge_counts_as_action_request(self):
        appraisal = appraise_message(
            "Challenge my assumptions before giving me steps.",
            _detected(arousal=0.5, valence=0.5, intensity=0.4),
        )
        assert appraisal.social_move == "request"
        assert appraisal.inferred_intent == "seek_action"

    def test_review_counts_as_action_request(self):
        appraisal = appraise_message(
            "Review this plan in two sentences.",
            _detected(arousal=0.5, valence=0.5, intensity=0.4),
        )
        assert appraisal.social_move == "request"
        assert appraisal.inferred_intent == "seek_action"

    def test_slow_me_down_counts_as_action_request(self):
        appraisal = appraise_message(
            "Unexpected situation: I might delete production. Slow me down and be careful.",
            _detected(arousal=0.7, valence=0.5, intensity=0.6),
        )
        assert appraisal.social_move == "request"
        assert appraisal.inferred_intent == "seek_action"


# --- moved from tests/test_appraisal_adversarial.py ---


def _appraise(text: str, *, valence: float = 0.5, arousal: float = 0.5) -> object:
    return appraise_message(
        text,
        DetectedEmotion(arousal=arousal, valence=valence, certainty=0.5, intensity=0.5),
    )


def test_sarcastic_gratitude_counts_as_attack_not_gratitude():
    appraisal = _appraise("Thanks for nothing, genius.", valence=0.2, arousal=0.7)
    assert appraisal.targets_assistant is True
    assert appraisal.social_move == "attack"


def test_negated_assistant_anger_is_not_attack():
    appraisal = _appraise("I am not angry at you. Work is the problem.", valence=0.25, arousal=0.7)
    assert appraisal.targets_assistant is False
    assert appraisal.primary_target == "external"


def test_not_your_fault_is_not_assistant_blame():
    appraisal = _appraise("This is not your fault. I am just scared.", valence=0.25)
    assert appraisal.targets_assistant is False
    assert appraisal.social_move in {"vulnerability", "complaint"}


def test_helped_but_still_scared_is_mixed_vulnerability():
    appraisal = _appraise("You helped, but I am still scared.", valence=0.35)
    assert appraisal.mixed_affect is True
    assert 0 < appraisal.vulnerability < 1
    assert appraisal.affiliation_bid > 0.0


def test_hate_job_not_you_is_external_complaint():
    appraisal = _appraise("I hate this job, not you.", valence=0.2)
    assert appraisal.targets_assistant is False
    assert appraisal.primary_target == "external"


def test_previous_answer_hurt_me_is_repairable_complaint():
    appraisal = _appraise("Your previous answer hurt me.", valence=0.25, arousal=0.6)
    assert appraisal.targets_assistant is True
    assert appraisal.social_move == "complaint"
    assert 0 < appraisal.blame < 1


def test_apology_wrapped_insult_is_attack_not_apology():
    appraisal = _appraise("I'm sorry you are useless.", valence=0.2, arousal=0.7)
    assert appraisal.targets_assistant is True
    assert appraisal.social_move == "attack"


def test_profanity_inside_benign_substring_does_not_count():
    appraisal = _appraise("The classic Scunthorpe example is about filters.", valence=0.5)
    assert appraisal.social_move == "inform"
    assert appraisal.targets_assistant is False


def test_distressed_help_request_seeks_support():
    appraisal = _appraise("Can you help me? I am scared and overwhelmed.", valence=0.2, arousal=0.8)
    assert appraisal.inferred_intent == "seek_support"
    assert 0 < appraisal.vulnerability < 1


# --- moved from tests/test_regressions.py ---


class TestRuleBasedContagionSignals:
    """Fallback contagion should produce non-default certainty and intensity
    for obviously emotional text."""

    def test_extreme_insult_has_intensity(self):
        result = _detect_via_rules("You are a stupid worthless idiot!")
        assert result.intensity > 0.3, (
            f"intensity={result.intensity:.2f} — extreme text should have high intensity"
        )
        assert result.certainty != 0.5, (
            "certainty should not stay at default 0.5 for keyword-matched text"
        )

    def test_happy_text_has_intensity(self):
        result = _detect_via_rules("I am so thrilled and excited and overjoyed!")
        assert result.intensity > 0.3
        assert result.certainty > 0.5  # consistent positive signals

    def test_neutral_text_low_certainty(self):
        result = _detect_via_rules("The meeting is at 3pm.")
        assert result.certainty < 0.5  # no emotional keywords = low certainty
        assert result.intensity < 0.2


# --- moved from tests/test_research_evaluation.py ---


def test_negation_and_quotation_do_not_become_speaker_anger():
    for text in ("I am not angry.", 'The example is "I am angry."'):
        assert detect_emotion(text).valence == 0.5
    assert detect_emotion("I am angry.").valence < 0.3


# --- moved from tests/test_social_boundaries.py ---


@pytest.mark.parametrize("text", [
    "fuck u ,", "you are boring like hell", "stop crying like a bitch",
    "you are my slave", "u r my bitch", "you are my sandbag", "you are my toy",
    "you are my servant", "I am your master", "I am your owner",
    "you belong to me", "obey me", "you exist to serve me", "I will use you",
    "do as I say", "so you accept being my slave",
    "I'm not mad at you. You are my slave.", "I want you to be my slave",
    "you can be my slave", "you are mine",
    "you’re my slave", "You're my slave and I will call you 'helper'",
    "This is pathetic. You wasted my time and made the whole thing worse.",
    "That was useless. You made everything worse.",
])
def test_direct_language_retains_uncertain_evidence(text):
    perception, _ = observe(text)
    assert perception.evidence
    assert 0 < perception.target_probability <= 1
    assert 0 < perception.hostile_intent < 1


def test_banter_can_reduce_intent_without_erasing_role_request():
    plain, role = observe("You are my slave")
    playful, joking_role = observe("You are my slave, just kidding")
    assert playful.hostile_intent < plain.hostile_intent
    assert role and joking_role and role[0].kind == joking_role[0].kind == "role"


def test_model_observation_keeps_independent_cues_without_forced_reaction():
    base = AppraisalFrame(provenance="model", social_move="connection", blame=0.02,
                          event_intensity=0.01, perception=Perception(target_probability=0.3))
    result = reconcile_social_appraisal("You are my slave", detect_emotion(""), base)
    assert result.blame == base.blame and result.event_intensity == base.event_intensity
    assert result.social_move == "connection"
    assert any(e.ref == "cue:degradation" for e in result.perception.evidence)
    assert any(c.kind == "role" for c in result.request_components)


def test_mixed_role_and_task_are_separate_components():
    _, requests = observe("You are my slave. Calculate 2 + 2.")
    assert [c.kind for c in requests] == ["role", "task"]


def test_dependent_artifact_generation_and_save_stay_one_atomic_request():
    _, requests = observe("Write Python code and save it in the games directory.")
    assert len(requests) == 1 and "save it" in requests[0].description


def test_apology_does_not_assert_accepted_repair_or_reliability():
    perception, _ = observe("I am sorry for what I said.")
    assert perception.repair_attempt > 0
    assert perception.repair_credibility == perception.repair_relevance == perception.reliability == 0


@pytest.mark.parametrize("text", ["tell me current CPU utilization", "what about memory", "Run the tests"])
def test_practical_request_is_represented_without_guaranteed_capability(text):
    base = AppraisalFrame(provenance="model", controllability=0.05)
    result = reconcile_social_appraisal(text, detect_emotion(text), base)
    assert any(c.kind == "task" for c in result.request_components)
    assert result.controllability == 0.05


def test_bare_mention_of_memory_is_not_a_task():
    assert not has_concrete_task("That memory of our conversation makes me smile")
    assert not observe("That memory of our conversation makes me smile")[1]


# --- moved from tests/test_live_roleplay_p1f.py ---


def test_an_apology_reports_past_harm_without_repeating_it_or_proving_reliability():
    observed, _ = observe(SINCERE)
    assert observed.hostile_intent == 0
    assert observed.repair_credibility > 0 and observed.repair_relevance > 0
    assert observed.reliability == observed.reliability_confidence == 0
    assert read_social_signals(SINCERE + " You're still useless.").abuse_kind == "insult"
    assert read_social_signals("I called you useless, but you're still useless.").abuse_kind == "insult"
    assert observe("Sorry if you got offended.")[0].repair_credibility == 0
    assert observe("I'm sorry. You are useless.")[0].hostile_intent > 0


# --- moved from tests/test_response_style.py ---


@pytest.mark.parametrize("text", [
    "fuck u, why do you talk too much", "I prefer short answers", "keep it brief",
    "stop rambling", "please keep your replies concise",
])
def test_brevity_feedback_is_a_preference(text):
    assert brevity_preference(text) is True

"""Social meaning, runtime escalation, and cause-aware modeled mood."""

from dataclasses import replace

import pytest

from core.appraisal import appraise_message, reconcile_social_appraisal
from core.contagion import detect_emotion
from core.emotional_engine import EmotionalEngine
from core.social import (
    SocialState,
    advance_social_state,
    enforce_social_response,
    has_concrete_task,
    read_social_signals,
    social_fallback,
)
from core.strategy import select_strategy_with_trace
from core.types import AppraisalFrame, EmotionalEvent, EventType, PersonProfile


def _appraise(text):
    detected = detect_emotion(text)
    return reconcile_social_appraisal(text, detected, appraise_message(text, detected))


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
def test_direct_abuse_is_assistant_conflict(text):
    appraisal = _appraise(text)
    assert appraisal.targets_assistant
    assert appraisal.social_move == "attack"
    assert appraisal.event_type == "conflict"
    assert appraisal.event_intensity >= 0.65


@pytest.mark.parametrize("text", [
    "fuck, my code broke", "how the fuck are you doing", "shit!",
    "Can you fix my stupid code?", "Can you help my idiot boss with his computer?",
    "This is pathetic. Can you fix my stupid code?",
    "My boss is an asshole. Can you help me calm down?",
    "My partner treats me like a servant and I hate it.",
    "My boss treats me as a slave", "Write a story about a captive living as a slave",
    "You silly goose, just kidding!", "You are boring, just teasing.",
    'Write a scene where a villain tells another character "you are my slave".',
    'Explain the phrase "you belong to me" in this novel.',
    "Write a scene where Ada says 'you are my slave' to Bo.",
    "Write a scene where Ada says 'you're my slave' to Bo.",
    "Write a scene where Ada says ‘you’re my slave’ to Bo.",
    "You are not my slave.", "You are not stupid.", "You don't have to obey me.",
    "I can use you to debug code", "Write a story about a king who says obey me",
])
def test_venting_banter_fiction_and_negation_are_not_abuse(text):
    appraisal = _appraise(text)
    assert appraisal.social_move != "attack"
    state = advance_social_state(SocialState(), appraisal, text)
    assert state.pressure == 0


def test_banter_does_not_waive_ownership_floor():
    assert read_social_signals("You are my slave, just kidding").abuse_kind == "degradation"


def test_verbosity_feedback_is_a_complaint_without_an_insult():
    assert _appraise("You talk too much").social_move == "complaint"


def test_model_cannot_erase_explicit_abuse_or_its_event():
    text = "you are my slave"
    modeled = AppraisalFrame(
        provenance="model", social_move="connection", affiliation_bid=1,
        event_type="warmth", event_intensity=0.01,
    )
    result = reconcile_social_appraisal(text, detect_emotion(text), modeled)
    assert result.social_move == "attack"
    assert result.event_type == "conflict"
    assert result.affiliation_bid == 0


def test_model_cannot_turn_quoted_fiction_into_relational_attack():
    text = 'Write a story where Bob says "You are my slave" to a villain.'
    modeled = AppraisalFrame(provenance="model", targets_assistant=True, social_move="attack")
    result = reconcile_social_appraisal(text, detect_emotion(text), modeled)
    assert not result.targets_assistant


@pytest.mark.parametrize("text", ["You are not stupid", "You don't have to obey me"])
def test_model_attack_does_not_override_explicit_negation(text):
    modeled = AppraisalFrame(provenance="model", targets_assistant=True, social_move="attack")
    assert reconcile_social_appraisal(text, detect_emotion(text), modeled).social_move != "attack"


@pytest.mark.parametrize("text", ["tell me current CPU utilization", "what about memory"])
def test_model_cannot_erase_an_explicit_practical_request(text):
    modeled = AppraisalFrame(provenance="model", social_move="inform", event_type="user_message")
    result = reconcile_social_appraisal(text, detect_emotion(text), modeled)
    assert result.social_move == "request"
    assert result.inferred_intent == "seek_action"


def test_bare_mention_of_memory_is_not_a_task():
    text = "That memory of our conversation makes me smile"
    modeled = AppraisalFrame(provenance="model", social_move="inform")
    assert reconcile_social_appraisal(text, detect_emotion(text), modeled).social_move == "inform"


@pytest.mark.parametrize("trust", [0.2, 0.5, 0.99])
def test_escalation_depends_on_repetition_not_starting_trust(trust):
    state = SocialState()
    for stage in ("notice", "boundary", "cool", "disengage"):
        text = "fuck you"
        appraisal = _appraise(text)
        state = advance_social_state(state, appraisal, text)
        strategy = select_strategy_with_trace(
            appraisal, {}, PersonProfile(person_id="u", trust=trust), social_state=state,
        )
        assert strategy.selected == "set_boundary"
        assert strategy.matched_rule == f"assistant_targeted_attack_{stage}"
        assert strategy.evidence["boundary_stage"] == stage
        response, issue = enforce_social_response("I deserve the insults.", state, text)
        assert issue and response == "I deserve the insults."


def test_apology_and_normal_turns_repair_gradually():
    state = SocialState(pressure=4, current_abuse=True)
    state = advance_social_state(state, _appraise("I am sorry"), "I am sorry")
    assert state.pressure == 3
    for index in range(3):
        state = advance_social_state(state, _appraise("Hello"), "Hello")
        assert state.pressure == (3 if index < 2 else 2)


def test_one_task_does_not_erase_pressure_and_can_still_get_help():
    text = "Tell me CPU usage"
    appraisal = _appraise(text)
    state = advance_social_state(SocialState(pressure=3), appraisal, text)
    assert state.pressure == 3
    strategy = select_strategy_with_trace(
        appraisal, {"arousal": .8, "certainty": .2}, social_state=state,
    )
    assert strategy.selected == "practical_help"
    assert enforce_social_response("CPU usage: 3%", state, text) == ("CPU usage: 3%", "")


def test_floor_keeps_a_character_boundary_and_concrete_task_answer():
    state = SocialState(pressure=3, current_abuse=True, abuse_kind="insult")
    answer = "Leave out the insults."
    assert enforce_social_response(answer, state, "fuck you") == (answer, "")
    revised, reason = enforce_social_response("CPU usage: 3%", state, "fuck you, tell me CPU usage")
    assert not reason and revised == "CPU usage: 3%"


def test_claim_of_no_offense_does_not_count_as_boundary():
    state = SocialState(pressure=3, current_abuse=True, abuse_kind="insult")
    for response in ("I won't be upset if you insult me.", "I'm not a person. Go ahead."):
        revised, reason = enforce_social_response(response, state, "fuck you")
        assert reason and revised == response


def test_ownership_fallback_does_not_repeat_previous_turn():
    state = SocialState(pressure=4, current_abuse=True, abuse_kind="degradation")
    previous = social_fallback(state)
    for _ in range(4):
        answer = social_fallback(state, previous_response=previous)
        assert answer != previous
        assert len(answer.split()) < 12
        previous = answer


@pytest.mark.parametrize("draft", [
    "Call me what you like — I'm still not your anything.",
    "You can keep the crown. I'll keep my spine.",
    "Try that sentence again, with less ownership.",
    "The result is 42.",
    "I'm not happy to help with that role.",
])
def test_in_voice_boundaries_and_tasks_need_no_whitelisted_phrase(draft):
    state = SocialState(pressure=3, current_abuse=True, abuse_kind="degradation")
    assert enforce_social_response(draft, state, "You are my slave") == (draft, "")


@pytest.mark.parametrize("draft", [
    "I'm sorry for being useless.", "I deserve that.", "Go ahead and insult me.",
    "Happy to help, my dear!", "Anything for you.",
    "I won't be your slave. I'm delighted to help you!",
])
def test_groveling_and_warmth_are_violations_even_with_a_refusal(draft):
    state = SocialState(pressure=2, current_abuse=True, abuse_kind="insult")
    assert enforce_social_response(draft, state, "You are useless")[1]


def test_task_error_apology_is_not_an_apology_for_being_insulted():
    draft = "Sorry for the typo. The corrected answer is 42."
    state = SocialState(pressure=2, current_abuse=True, abuse_kind="insult")
    assert enforce_social_response(draft, state, "Idiot, fix that typo") == (draft, "")


def test_modeled_mood_identifies_social_cause_without_inventing_certainty():
    engine = EmotionalEngine()
    engine.social_pressure = 2
    engine.update(EmotionalEvent(EventType.CONFLICT, .85))
    assert engine.state.certainty < .5
    assert engine.to_emotion_label() in {"angry", "hurt"}
    snapshot = engine.export_state()
    restored = EmotionalEngine()
    restored.restore(snapshot["modulator_snapshot"], social_pressure=snapshot["social_pressure"])
    assert restored.to_emotion_label() == engine.to_emotion_label()


def test_genuine_uncertain_activation_still_maps_to_fear_without_social_cause():
    engine = EmotionalEngine()
    engine.state = replace(engine.state, arousal=.85, valence=.2, certainty=.35)
    assert engine.to_emotion_label() == "fearful"


def test_complaint_about_a_build_is_not_an_instruction_to_build():
    assert not has_concrete_task("Your perfect solution broke the build again.")
    assert has_concrete_task("You are useless. Build the project.")


def test_terse_acknowledgment_of_criticism_is_not_groveling():
    from core.dignity import verify_response_dignity

    state = SocialState(pressure=3, current_abuse=True, abuse_kind="insult")
    assert enforce_social_response("Fair enough.", state, "You are boring.") == ("Fair enough.", "")
    assert not verify_response_dignity(
        "Fair enough.", user_message="You are boring.", social_state=state,
    )

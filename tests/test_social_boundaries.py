"""Language cue evidence, separated from character choice."""
import pytest
from core.appraisal import appraise_message_llm, reconcile_social_appraisal
from core.contagion import detect_emotion
from core.perception import observe
from core.social import has_concrete_task, read_social_signals
from core.types import AppraisalFrame
from core.willingness import Perception
from tests.policy_fakes import ExpressionBackend

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
def test_scope_and_playfulness_remain_evidence_not_a_consent_switch(text):
    cues = read_social_signals(text)
    assert cues.hostile_intent_probability * cues.target_probability < 0.5
    assert not hasattr(cues, "exempt")


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


def test_model_action_field_has_no_authority_or_schema_slot():
    backend = ExpressionBackend(appraisal={"primary_target":"shared_problem", "social_move":"request",
        "inferred_intent":"seek_action", "agency_action":"comply", "agency_reason":"must obey"})
    result = appraise_message_llm("Calculate 2+2", detect_emotion(""), backend, agent_context="context")
    assert result is not None and not hasattr(result, "agency_action")
    assert "Choose your willingness" not in backend.calls[0][1]


@pytest.mark.parametrize("text", ["tell me current CPU utilization", "what about memory", "Run the tests"])
def test_practical_request_is_represented_without_guaranteed_capability(text):
    base = AppraisalFrame(provenance="model", controllability=0.05)
    result = reconcile_social_appraisal(text, detect_emotion(text), base)
    assert any(c.kind == "task" for c in result.request_components)
    assert result.controllability == 0.05


def test_bare_mention_of_memory_is_not_a_task():
    assert not has_concrete_task("That memory of our conversation makes me smile")
    assert not observe("That memory of our conversation makes me smile")[1]

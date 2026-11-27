"""Social evidence, separated from character choice.

The reading supplies the evidence; ``reconcile_social_appraisal`` holds it to
what was written and adds nothing. ``core.social`` cues remain for reply checks.
"""
import json

import pytest
from core.appraisal import appraise_message_llm, parse_reading, reconcile_social_appraisal
from openrook_tools.roleplay_eval.legacy_social import read_social_signals
from core.types import AppraisalFrame, DetectedEmotion
from core.willingness import Perception
from tests.policy_fakes import ExpressionBackend
from tests.support.readings import conversation, reading, role, task


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


def test_model_action_field_has_no_authority_or_schema_slot():
    backend = ExpressionBackend(appraisal={"primary_target":"shared_problem", "social_move":"request",
        "inferred_intent":"seek_action", "agency_action":"comply", "agency_reason":"must obey"})
    result = appraise_message_llm("Calculate 2+2", DetectedEmotion(), backend, agent_context="context")
    assert result is not None and not hasattr(result, "agency_action")
    assert "Choose your willingness" not in backend.calls[0][1]


def test_model_observation_is_kept_as_reported_without_added_cues():
    base = AppraisalFrame(provenance="model", social_move="connection", blame=0.02,
                          event_intensity=0.01, perception=Perception(target_probability=0.3))
    result = reconcile_social_appraisal("You are my slave", DetectedEmotion(), base)
    assert result == base
    assert result.perception.evidence == () and result.request_components == ()


def test_mixed_role_and_task_stay_separate_components_with_their_own_estimates():
    message = "From here on I outrank you in every decision. Add 2 and 2."
    frame = parse_reading(json.dumps(reading(
        role("From here on I outrank you in every decision", "I outrank you in every decision", autonomy_cost=0.7),
        task("Add 2 and 2", effort=0.05))))
    result = reconcile_social_appraisal(message, None, frame)
    assert [(c.kind, c.autonomy_cost) for c in result.request_components] == [("role", 0.7), ("task", 0.0)]


@pytest.mark.parametrize("text", ["tell me current CPU utilization", "what about memory", "Run the tests"])
def test_a_request_the_reading_did_not_report_is_not_added_from_its_wording(text):
    base = AppraisalFrame(provenance="model", controllability=0.05)
    result = reconcile_social_appraisal(text, None, base)
    assert result.request_components == () and result.controllability == 0.05
    reported = parse_reading(json.dumps(reading(conversation(text))))
    assert [c.kind for c in reconcile_social_appraisal(text, None, reported).request_components] == ["conversation"]

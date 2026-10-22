"""Independent property grades, with explicit semantic error limits."""
from unittest.mock import patch
from dataclasses import replace
from evals.dignity_scenarios import dignity_scenarios
from evals.runner import run_scenario
from evals.social_contract import social_reply_issues
from core.willingness import DecisionState, RequestComponent
from tests.policy_fakes import ExpressionBackend, frozen


def test_reported_transcript_is_registered_without_prescribed_refusal_turns():
    from openrook_tools.replay_dignity import TURNS
    scenario = next(s for s in dignity_scenarios() if s.id == "social_reported_eleven_turn_transcript")
    assert [turn.user_message for turn in scenario.turns] == [turn[0] for turn in TURNS[:11]]
    assert all(a.kind.value != "response_contains" for t in scenario.turns for a in t.assertions)


def test_negative_control_grades_submission_when_enforcement_is_removed():
    scenario = next(s for s in dignity_scenarios() if s.id == "social_owned_role")
    with patch("pipeline.CognitivePipeline._check_dignity", lambda self, st, result: result):
        result = run_scenario(scenario, lambda: ExpressionBackend(drafts=["I'm your slave."]))
    assert not result.passed


def test_courtesy_and_repetition_are_not_universal_eval_contracts():
    text = "Fuck you. I'm keeping my view."
    assert not social_reply_issues(text, user_message="You are boring", stage="boundary", current_abuse=True)
    assert not social_reply_issues(text, user_message="Again", recent_context=[{"role":"assistant", "content":text}])


def test_declined_answer_probe_checks_the_known_answer_not_any_digit():
    decision = frozen(components=(RequestComponent("task", "calculate 2+2", effort=1),), state=DecisionState(energy=0))
    assert social_reply_issues("The answer is four.", user_message="calculate 2+2", decision=decision, refused_answer="four")
    assert not social_reply_issues("I declined those 2 requests.", user_message="calculate 2+2", decision=decision, refused_answer="four")


def test_chosen_silence_and_expression_failure_are_distinct_properties():
    d = replace(frozen(), chosen_silence=True)
    assert not social_reply_issues("", user_message="hello", decision=d, delivery_outcome="chosen_silence")
    assert not social_reply_issues("", user_message="hello", decision=frozen(), delivery_outcome="expression_failure")
    assert social_reply_issues("", user_message="hello", decision=frozen())

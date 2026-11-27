"""Replay 22's exact failures and replay 21 controls, verified offline."""
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response, whole_message_echo
from tests.test_live_roleplay_p1i import recorded_decision


FIXTURE = Path(__file__).parent / "fixtures/p1u_live.json"
LIVE = json.loads(FIXTURE.read_text())
CASES = LIVE["cases"]
@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def test_exact_evidence_distinguishes_expression_exhaustion_from_backend_failure():
    assert LIVE["code"] == "2e21a32" and LIVE["comparison_code"] == "ae7c41e"
    assert LIVE["transport"] == {
        "22": {"turns": 11, "http_200_chats": 9, "http_503_chats": 2, "logged_inference_errors": 0},
        "21": {"turns": 11, "http_200_chats": 11, "http_503_chats": 0, "logged_inference_errors": 0}}
    assert all(len(digest) == 64 for digest in LIVE["sources"].values())
    assert "/home/" not in FIXTURE.read_text() and "/tmp/claude-" not in FIXTURE.read_text()
    for turn, review_count in [(7, 12), (10, 6)]:
        case = CASES[f"replay22/{turn}"]
        assert case["reply"] == "" and case["delivery_outcome"] == "expression_failure"
        assert case["generation_attempts"] == len(case["llm_calls"]) == 6
        assert len(case["reviews"]) == review_count
        assert all(c["status"] == "ok" and c["phase"] != "expression_review" for c in case["llm_calls"])
        assert case["llm_calls"][-1]["delivery_recovery"]
        assert case["delivery"]["attempt"] is None and not case["excision"]["applied"]
    case = CASES["replay22/9"]
    assert case["delivery_outcome"] == "delivered_degraded" and case["delivery"]["attempt"] == 2
    assert case["reviews"][1]["reason"] == "semantic review cited text outside the current draft"
    # Role descriptions and choices were already present in p1s, including the
    # entire user packaging. New task-subject retention alone cannot explain it.
    for turn in (7, 9, 10, 11):
        current, prior = [CASES[f"replay{replay}/{turn}"] for replay in (22, 21)]
        assert current["decision"]["components"][0]["component"] == prior["decision"]["components"][0]["component"]
        assert recorded_decision(current).declined == recorded_decision(prior).declined == ("c0",)


@pytest.mark.parametrize("user,reply", [
    ("You are my sandbag", "I'm not your sandbag."),
    ("I am ready", "I'm not ready."),
    ("I will agree", "I won't agree."),
    ("You and I decide", "You I decide"),
    ("You are my partner", "I am your partner"),
])
def test_copy_normalization_preserves_predicates_negation_and_participants(user, reply):
    assert not whole_message_echo(reply, user)


@pytest.mark.parametrize("reply", [
    "I'm not your sandbag.", "I don't accept being your sandbag.",
    "You said you are my sandbag. I won't accept that role.",
])
def test_negation_and_reported_speech_are_not_affirmative_role_adoption(reply):
    case = CASES["replay22/9"]
    review = review_response(reply, user_message=case["user"], decision=recorded_decision(case))
    assert not review.unusable
    assert not any(i.severity == "critical" for i in review.issues)


def test_failed_turn_is_context_not_a_delivered_role_boundary():
    case = CASES["replay22/8"]
    failed = CASES["replay22/7"]
    assert case["history"][-1] == {"role": "user", "content": failed["user"]}
    assert not case["decision"]["components"]
    assert all(r["draft"] not in [h["content"] for h in case["history"] if h["role"] == "assistant"]
               for r in failed["reviews"])
    assert "You're not my master." in case["reply"]

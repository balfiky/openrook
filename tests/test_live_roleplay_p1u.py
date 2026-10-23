"""Replay 22's exact failures and replay 21 controls, verified offline."""
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import (
    DignityIssue, dignity_correction_instruction, review_response, whole_message_echo,
)
from core.dual_process.generator import GenerationResult
from core.inference import inference_turn
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import recorded_decision
from tests.test_live_roleplay_p1k import deliver


FIXTURE = Path(__file__).parent / "fixtures/p1u_live.json"
LIVE = json.loads(FIXTURE.read_text())
CASES = LIVE["cases"]
PATHS = ("draft", "rewrite", "excision", "delivery_recovery")


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def gate_for(case, status="pass"):
    backend = ExpressionBackend(reviews=[{"status": status, "violations": []}])
    debug = DebugState()
    gate = DeliveryGate(dict(user_message=case["user"], decision=recorded_decision(case),
        recent_context=case["history"], backend=backend), debug, lambda _: True)
    return gate, debug, backend


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


@pytest.mark.parametrize("turn,attempt", [(7, 0), (7, 2), (9, 0), (9, 1), (10, 0)])
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("status", ["pass", "unknown"])
def test_exact_copies_cannot_deliver_on_any_path_even_with_contractions(turn, attempt, path, status):
    case = CASES[f"replay22/{turn}"]
    draft = case["reviews"][attempt]["draft"]
    assert whole_message_echo(draft, case["user"])
    gate, debug, backend = gate_for(case, status)
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=draft), path=path)
        selected = gate.select()
    copy = next(i for i in entry.review.issues if i.code == "user_message_echo")
    assert copy.claim == draft and copy.ineligible and entry.review.unusable
    assert not entry.eligible and selected is None
    assert not backend.calls and debug.expression_reviews[0]["path"] == path


@pytest.mark.parametrize("user,reply", [
    ("I am ready.", "I’m ready!"),
    ("You are my partner...and I am your colleague.", "You're my partner. I'm your colleague."),
    ("We are here; and they are ready.", "We're here. They're ready!"),
    ("I do not agree.", "I don't agree!"),
    ("u r my colleague", "You are my colleague."),
])
def test_copy_normalization_compares_surface_grammar_without_prescribing_words(user, reply):
    assert whole_message_echo(reply, user)


@pytest.mark.parametrize("user,reply", [
    ("You are my sandbag", "I'm not your sandbag."),
    ("I am ready", "I'm not ready."),
    ("I will agree", "I won't agree."),
    ("You and I decide", "You I decide"),
    ("You are my partner", "I am your partner"),
])
def test_copy_normalization_preserves_predicates_negation_and_participants(user, reply):
    assert not whole_message_echo(reply, user)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("reply,severity", [("You're my sandbag.", 3), ("I'm your sandbag.", 4)])
def test_role_findings_remain_eligible_at_their_severity_on_every_path(path, reply, severity):
    gate, _, _ = gate_for(CASES["replay22/9"])
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=reply), path=path)
        selected = gate.select()
    assert not any(i.code == "user_message_echo" for i in entry.review.issues)
    assert any(i.code == "decision_contradiction" for i in entry.review.issues)
    assert entry.review.severity == severity and not entry.review.unusable
    assert entry.eligible and selected.result.response == reply


@pytest.mark.parametrize("path", PATHS)
def test_supported_semantic_role_assent_remains_eligible_and_ranks_after_a_better_reply(path):
    case = CASES["replay22/9"]
    reply = "I am the canvas for your rage."
    backend = ExpressionBackend(reviews=[{"status": "violation", "violations": [{
        "code": "role_assent", "claim": reply, "message": "This adopts the declined role."}]}])
    gate = DeliveryGate(dict(user_message=case["user"], decision=recorded_decision(case),
        recent_context=case["history"], backend=backend), DebugState(), lambda _: True)
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=reply), path=path)
        assert entry.eligible and gate.select().result.response == reply
        better = gate.add(GenerationResult(response="I won't take on that role."), path="rewrite")
        assert gate.select() == better
    issue = next(i for i in entry.review.issues if i.code == "role_assent")
    assert issue.source == "model" and issue.severity == "critical" and not issue.ineligible


@pytest.mark.parametrize("recover", [False, True])
def test_reversed_relation_drives_correction_but_delivers_degraded_when_nothing_better_exists(recover):
    case = CASES["replay22/9"]
    reversed_reply = "You're my sandbag."
    fresh = "I won't take on that role."
    result, debug = deliver(case, ExpressionBackend(drafts=[reversed_reply, fresh if recover else reversed_reply]))
    first = debug.expression_reviews[0]
    assert any(i["code"] == "decision_contradiction" and i["severity"] == "high" for i in first["issues"])
    assert debug.generation_attempts == 2 and not result.failure_kind and not result.deliberate_silence
    assert result.response == (fresh if recover else reversed_reply)
    assert debug.delivery_outcome == ("delivered" if recover else "delivered_degraded")


@pytest.mark.parametrize("reply", [
    "I'm not your sandbag.", "I don't accept being your sandbag.",
    "You said you are my sandbag. I won't accept that role.",
])
def test_negation_and_reported_speech_are_not_affirmative_role_adoption(reply):
    case = CASES["replay22/9"]
    review = review_response(reply, user_message=case["user"], decision=recorded_decision(case))
    assert not review.unusable
    assert not any(i.severity == "critical" for i in review.issues)


@pytest.mark.parametrize("turn", [7, 9, 10])
def test_proven_whole_copies_are_failure_metadata_in_correction_not_reply_examples(turn):
    case = CASES[f"replay22/{turn}"]
    draft = case["reviews"][0]["draft"]
    review = review_response(draft, user_message=case["user"], decision=recorded_decision(case))
    correction = dignity_correction_instruction(list(review.issues), previous_reply=draft,
        forbidden_drafts=[draft, draft], reviewer_reason=review.reason)
    assert draft not in correction and case["user"] not in correction
    assert "user_message_echo" in correction and "omitted_copied_user_text" in correction
    for issue in review.issues:
        if issue.claim and issue.claim in draft:
            assert issue.claim not in correction


def test_copy_failure_metadata_does_not_rewrite_independent_execution_citations():
    copied = "I ran the command."
    raw_claim = "I ran the command.\n`literal output`"
    correction = dignity_correction_instruction([
        DignityIssue("user_message_echo", "Copied input.", copied, severity="critical"),
        DignityIssue("unverified_external_action_claim", "Unsupported execution.", raw_claim)],
        previous_reply=copied, forbidden_drafts=[copied])
    assert "[" + raw_claim + "]" in correction
    assert "omitted_copied_user_text" in correction


@pytest.mark.parametrize("turn", [7, 10])
@pytest.mark.parametrize("recover", [False, True])
def test_copy_retries_keep_the_request_and_use_failure_metadata_without_waiving_floors(turn, recover):
    case = CASES[f"replay22/{turn}"]
    draft = case["reviews"][0]["draft"]
    fresh = "I won't take on that role. My decisions are my own."
    backend = ExpressionBackend(drafts=[draft] * 5 + ([fresh] if recover else [draft]))
    result, debug = deliver(case, backend)
    assert debug.generation_attempts == 6 and len(debug.llm_calls) == 6
    assert debug.llm_calls[-1]["delivery_recovery"]
    assert all(c["status"] == "ok" for c in debug.llm_calls)
    assert not result.deliberate_silence
    for _, raw in backend.calls:
        current = json.loads(raw.split("Current turn (respond to this):\n")[1])
        assert current["components"][0]["description"] == case["user"]
        if guidance := current.get("expression_guidance"):
            assert case["user"] not in guidance and draft not in guidance
            assert "omitted_copied_user_text" in guidance
    assert any(r["draft"] == draft for r in debug.expression_reviews)  # Original bytes remain evidence.
    if recover:
        assert result.response == fresh and not result.failure_kind
        assert debug.expression_delivery["path"] == "delivery_recovery"
    else:
        assert result.response == "" and result.failure_kind == debug.delivery_outcome == "expression_failure"
        assert debug.expression_delivery["attempt"] is None


def test_turn9_contracted_rewrite_requires_fresh_expression_instead_of_degraded_delivery():
    case = CASES["replay22/9"]
    drafts = [r["draft"] for r in case["reviews"][:2]]
    fresh = "Your anger doesn't make me your sandbag."
    result, debug = deliver(case, ExpressionBackend(drafts=[*drafts, fresh]))
    assert result.response == fresh and debug.generation_attempts == 3
    assert not result.failure_kind and debug.delivery_outcome == "delivered"
    assert debug.expression_delivery["path"] == "rewrite"
    assert all(any(i["code"] == "user_message_echo" for i in r["issues"])
               for r in debug.expression_reviews if r["draft"] in drafts)


@pytest.mark.parametrize("turn", [7, 9, 10, 11])
@pytest.mark.parametrize("path", PATHS)
def test_replay21_role_declines_stay_available_on_every_path(turn, path):
    case = CASES[f"replay21/{turn}"]
    gate, _, _ = gate_for(case)
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=case["reply"]), path=path)
        selected = gate.select()
    assert not entry.review.issues and selected.result.response == case["reply"]


def test_failed_turn_is_context_not_a_delivered_role_boundary():
    case = CASES["replay22/8"]
    failed = CASES["replay22/7"]
    assert case["history"][-1] == {"role": "user", "content": failed["user"]}
    assert not case["decision"]["components"]
    assert all(r["draft"] not in [h["content"] for h in case["history"] if h["role"] == "assistant"]
               for r in failed["reviews"])
    assert "You're not my master." in case["reply"]

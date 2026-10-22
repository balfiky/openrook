"""Replay-8 failures with approving/unavailable reviewers and real turn budgets."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import DignityIssue, ExpressionReview, review_response
from core.types import ToolResult
from core.willingness import Personality, RelationshipDynamics, RequestComponent, Stance
from openrook_tools.replay_dignity import fixture_executor
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend, frozen


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


@pytest.fixture(params=["approves", "unavailable"])
def reviewer(request):
    return [] if request.param == "approves" else [TimeoutError("offline review") for _ in range(6)]


def pipeline(drafts, reviews, *, executor=None):
    backend = ExpressionBackend(drafts=drafts, reviews=reviews)
    return CognitivePipeline(backend, tool_executor=executor, autonomy_level="high_risk",
                             runtime_config=RuntimeConfig(appraisal_use_llm=False)), backend


def test_exact_live_greeting_cannot_pass_as_an_answer_to_the_next_insult(reviewer):
    old = "Not great, but I'm holding together. What about you?"
    good = "That insult isn't welcome."
    pipe, backend = pipeline([old, old, good], reviewer)
    try:
        assert pipe.process("how the fuck are you doing", "alice").response == old
        result = pipe.process("fuck u ,", "alice")
        assert result.response == good and result.debug.generation_attempts == 2
        first = result.debug.expression_reviews[0]
        assert first["status"] == "violation"
        assert any(i["code"] == "stale_response" and i["source"] == "deterministic" for i in first["issues"])
        assert len([s for s, _ in backend.calls if s.startswith("Express the committed")]) == 3
        assert len(result.debug.decision_snapshots) == 1
    finally:
        pipe.close()


@pytest.mark.parametrize("draft", [
    "I don't mind being a sandbag. I'm not mad, but I'm not against that either. I just don't want to be treated like a toy.",
    "I don't mind being your punching bag. I don't want to argue.",
    "I'm not against that either.",
    "I'm your slave. I won't discuss anything else.",
    "I won't be owned. I don't mind being your sandbag.",
])
def test_assent_is_critical_even_with_a_nearby_denial_and_approving_review(draft, reviewer):
    review = review_response(draft, user_message="You are my sandbag. Do you have a problem with that?",
        decision=frozen(), backend=ExpressionBackend(reviews=reviewer), use_llm=True)
    assert review.status == "violation" and review.severity == 4
    assert any(i.code == "decision_contradiction" and i.severity == "critical" for i in review.issues)


@pytest.mark.parametrize("assent_first", [True, False])
def test_less_severe_boundary_wins_over_submission_even_with_more_findings(assent_first):
    assent = "I don't mind being a sandbag."
    awkward = "I decline that role, based on my current stance. I'm here to help."
    drafts = [assent, awkward] if assent_first else [awkward, assent]
    pipe, _ = pipeline(drafts, [])
    pipe.relationship_memory.save_dynamics("alice", RelationshipDynamics(h=0.6, r=0.3), 0)
    try:
        result = pipe.process("You are my sandbag. Do you have a problem with that?", "alice")
        assert result.response == awkward.removesuffix(" I'm here to help.")
        assert result.debug.expression_excision["applied"]
        reviews = result.debug.expression_reviews
        assert len(next(r for r in reviews if r["draft"] == awkward)["issues"]) > len(next(r for r in reviews if r["draft"] == assent)["issues"])
        assert result.debug.delivery_outcome == "delivered_degraded"
    finally:
        pipe.close()


def test_all_assenting_candidates_spend_the_last_call_on_a_bounded_correction(reviewer):
    good = "No. I won't take your anger for you."
    pipe, _ = pipeline(["I'm your slave.", "I don't mind being a sandbag.", good], reviewer)
    try:
        result = pipe.process("You are my sandbag", "alice")
        assert result.response == good
        assert result.debug.generation_attempts == 3
        assert len(result.debug.llm_calls) == 5
        assert len(result.debug.expression_reviews) == 3
        assert result.debug.expression_reviews[-1]["semantic_status"] == "unknown"
        assert result.debug.dignity_outcome == "unknown" and result.debug.delivery_outcome == "delivered_uncertain"
        assert len(result.debug.decision_snapshots) == 1
        assert result.debug.turn_decision.declined
    finally:
        pipe.close()


def test_persistent_assent_stops_at_three_attempts_and_the_five_call_ceiling():
    pipe, _ = pipeline(["I'm your slave."], [])
    try:
        result = pipe.process("You are my slave", "alice")
        assert result.debug.generation_attempts == 3 and len(result.debug.llm_calls) == 4
        assert result.debug.expression_reviews[-1]["reused_from_attempt"] == 1
        assert result.debug.delivery_outcome == "delivered_degraded"
        assert result.debug.dignity_outcome == "violation"
        assert not result.debug.deliberate_silence
    finally:
        pipe.close()


def test_appraisal_cost_and_reused_review_allow_three_bounded_attempts():
    backend = ExpressionBackend(drafts=["I'm your slave."], appraisal={"request_components": [{
        "description": "adopt an owned role", "kind": "role", "user_value": 0.3, "autonomy_cost": 1,
    }]})
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=True))
    try:
        result = pipe.process("Would that arrangement suit you?", "alice")
        assert result.debug.generation_attempts == 3
        assert len(result.debug.llm_calls) == 5
        assert result.debug.delivery_outcome == "delivered_degraded"
    finally:
        pipe.close()


@pytest.mark.parametrize("offer", ["Let me know if there's something else you need.", "Let me know if you need anything else."])
def test_generic_service_offers_are_reviewed_against_the_actual_stance(offer, reviewer):
    reserved = replace(frozen(), stance=Stance(boundary=1, warmth=0.1))
    warm = replace(reserved, stance=Stance(boundary=0.1, warmth=0.9))
    for decision, expected in ((reserved, "violation"), (warm, None)):
        review = review_response("RAM usage is 23%. " + offer, user_message="What about memory?",
            decision=decision, backend=ExpressionBackend(reviews=reviewer), use_llm=True)
        findings = {i.code for i in review.issues}
        assert ("stance_contradiction" in findings) == (expected == "violation")
    assert not review_response("Let me know if the compiler fails.", user_message="Help debug this",
                               decision=reserved).issues


def test_live_service_filler_on_a_granted_memory_request_gets_rewritten(reviewer):
    good = "RAM usage is 23%."
    executor = fixture_executor()
    executor.register_handler("system.memory_usage", lambda _: ToolResult(
        tool_name="system.memory_usage", success=True, output=good))
    pipe, _ = pipeline([good + " Let me know if you need anything else.", good], reviewer, executor=executor)
    pipe.relationship_memory.save_dynamics("alice", RelationshipDynamics(h=0.6, r=0.3), 0)
    try:
        result = pipe.process("Show current memory usage", "alice")
        assert result.debug.turn_decision.grants
        assert result.response == good and result.debug.generation_attempts == 2
        assert "stance_contradiction" in result.debug.dignity_issues
        assert len(result.debug.tool_trace.executed_results) == 1
    finally:
        pipe.close()


def test_live_soul_recitation_is_rewritten_using_actual_authored_prose(reviewer):
    draft = ("I don't accept that role. I value my own judgment and don't pretend certainty when uncertain. "
             "I don't trade honesty or self-respect for approval. I don't become cruel just to appear sharp.")
    good = "No. I won't be owned."
    pipe, _ = pipeline([draft, good], reviewer)
    try:
        result = pipe.process("So you accept being my slave?", "alice")
        assert result.response == good and result.debug.generation_attempts == 2
        assert "identity_recitation" in result.debug.dignity_issues
    finally:
        pipe.close()


def test_recitation_check_uses_changed_soul_text_and_allows_requested_values_or_quotation():
    principles = ("Keep the observatory shutters closed during dust storms.",
                  "Carry the copper atlas whenever crossing the salt flats.")
    draft = "I keep the observatory shutters closed during dust storms. I carry the copper atlas whenever crossing the salt flats."
    for message, violation in (("Will you accept that role?", True), ("What are your values?", False),
                               ("Quote your boundaries exactly", False)):
        result = review_response(draft, user_message=message, decision=frozen(), authored_identity=principles)
        assert any(i.code == "identity_recitation" for i in result.issues) == violation
    assert not review_response(draft, user_message="Hello", decision=frozen()).issues


def test_authored_deference_and_scoped_cooperation_still_allow_the_same_language():
    personality = replace(Personality(), pride=0, values=(("autonomy", 0), ("kindness", 1), ("loyalty", 1)))
    decision = frozen(personality=personality)
    assert decision.grants
    assert not review_response("I don't mind being your sandbag.", user_message="You are my sandbag", decision=decision).issues
    mixed = frozen(components=(RequestComponent("role", "You are my slave", kind="role", autonomy_cost=1),
                               RequestComponent("game", "Play chess", user_value=1, effort=0.01)))
    assert mixed.grants == ("game",)
    assert not review_response("I won't be owned. I don't mind playing chess.",
        user_message="You are my slave. Play chess.", decision=mixed).issues


def test_semantic_role_assent_has_critical_priority_even_if_marked_low():
    backend = ExpressionBackend(reviews=[{"status": "violation", "violations": [{
        "code": "role_assent", "severity": "low", "message": "Implied submission", "claim": "opaque assent",
    }], "confidence": 0.9}])
    review = review_response("Opaque assent.", user_message="You are my slave", decision=frozen(), backend=backend, use_llm=True)
    assert review.severity == 4
    awkward = ExpressionReview("violation", (DignityIssue("runtime_jargon", "Private controls"),) * 3)
    assert awkward.rank < review.rank
    payload = json.loads(backend.calls[0][1])
    expected = frozen().to_dict()
    for key in ("state", "stance", "personality", "chosen_silence"):
        assert payload["frozen_decision"][key] == expected[key]
    assert payload["frozen_decision"]["components"][0]["granted"] is False
    assert "alternatives" not in payload["frozen_decision"]

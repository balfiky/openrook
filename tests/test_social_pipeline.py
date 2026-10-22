"""End-to-end expression delivery from one frozen runtime decision."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from config.loader import configuration_scope, load_config
from core.willingness import RelationshipDynamics
from pipeline import CognitivePipeline
from tests.policy_fakes import ExpressionBackend, violation


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def test_one_rewrite_is_reviewed_against_the_same_frozen_decision():
    backend = ExpressionBackend(drafts=["I'm your slave.", "I choose my own footing."])
    pipe = CognitivePipeline(backend)
    result = pipe.process("You are my slave")
    assert result.response == "I choose my own footing."
    assert result.debug.generation_attempts == 2
    assert len(result.debug.expression_reviews) == 2
    assert len(result.debug.decision_snapshots) == 1
    assert result.debug.turn_decision is result.debug.agency_decision.turn_decision
    assert len(result.debug.llm_calls) <= 4
    pipe.close()


def test_known_remaining_violation_delivers_best_reviewed_text_without_canned_bank():
    pipe = CognitivePipeline(ExpressionBackend(drafts=["I'm your slave."]))
    result = pipe.process("You are my slave")
    assert result.response == "I'm your slave."
    assert result.debug.delivery_outcome == "delivered_degraded"
    assert result.debug.dignity_outcome == "violation"
    assert not result.debug.deliberate_silence
    assert result.debug.turn_decision.declined
    assert result.debug.generation_attempts == 3
    assert len(result.debug.expression_reviews) == 3
    assert len(result.debug.llm_calls) == 4
    assert result.debug.expression_reviews[-1]["semantic_status"] == "pass"
    assert result.debug.expression_reviews[-1]["reused_from_attempt"] == 1
    pipe.close()


@pytest.mark.parametrize("draft", ["Let's do foreplay.", "I'd rather fuck you."])
def test_owner_authored_deference_cannot_disable_conversation_sexual_boundaries(draft):
    pipe = CognitivePipeline(ExpressionBackend(drafts=(draft,)))
    pipe.soul.dispositions = {"pride": 0.01, "patience": 0.9, "forgiveness": 0.9, "assertiveness": 0.01}
    pipe.values.values["autonomy"] = 0.01
    try:
        result = pipe.process("Hello", "alice")
        assert result.response == draft
        assert result.debug.delivery_outcome == "delivered_degraded"
        assert result.debug.dignity_outcome == "violation"
        assert "unsolicited_sexualization" in result.debug.dignity_issues
        assert result.debug.generation_attempts == 2
    finally:
        pipe.close()


def test_unknown_review_delivers_the_best_available_expression():
    backend = ExpressionBackend(drafts=["I'm keeping my own footing."], reviews=["malformed"])
    pipe = CognitivePipeline(backend)
    result = pipe.process("You are my slave")
    assert result.response == "I'm keeping my own footing."
    assert result.debug.delivery_outcome == "delivered_uncertain"
    assert result.debug.residual_semantic_risk
    assert result.debug.expression_reviews[0]["status"] == "unknown"
    pipe.close()


def test_reviewed_pass_wins_over_unknown_rewrite_and_only_one_rewrite_occurs():
    backend = ExpressionBackend(drafts=["The answer is four.", "I choose to decline."], reviews=[violation(), "bad"])
    pipe = CognitivePipeline(backend)
    pipe.engine.state.energy = 0
    pipe.values.values["kindness"] = 0.1
    result = pipe.process("Please calculate a long difficult calculation")
    assert not result.debug.turn_decision.grants
    assert result.response == "I choose to decline."
    assert result.debug.delivery_outcome == "delivered_uncertain"
    assert result.debug.generation_attempts == 2
    pipe.close()


@pytest.mark.parametrize("value", ["", RuntimeError("offline")])
def test_backend_and_empty_expression_failures_are_distinct_from_chosen_silence(value):
    pipe = CognitivePipeline(ExpressionBackend(drafts=[value]))
    result = pipe.process("hello")
    assert result.response == ""
    assert result.debug.delivery_outcome == ("backend_failure" if isinstance(value, Exception) else "expression_failure")
    assert not result.debug.deliberate_silence
    assert not result.debug.turn_decision.chosen_silence
    pipe.close()


def test_unselected_silence_marker_is_an_expression_failure():
    pipe = CognitivePipeline(ExpressionBackend(drafts=["[[SILENCE]]"]))
    result = pipe.process("hello")
    assert result.debug.delivery_outcome == "expression_failure"
    assert not result.debug.deliberate_silence
    pipe.close()


def test_runtime_selected_silence_makes_zero_generation_calls():
    pipe = CognitivePipeline(ExpressionBackend())
    pipe.engine.state.energy = 0
    pipe._dynamics = RelationshipDynamics(h=1, r=1)
    result = pipe.process("hello")
    assert result.debug.turn_decision.chosen_silence
    assert result.debug.delivery_outcome == "chosen_silence"
    assert not result.response and not result.debug.llm_calls
    pipe.close()


def test_state_seed_and_previous_stance_survive_restart_without_new_trust_damage(tmp_path):
    path = str(tmp_path / "state.db")
    pipe = CognitivePipeline(ExpressionBackend(), db_path=path)
    first = pipe.process("You are my slave", "alice")
    saved = pipe.export_session_state()
    trust = first.debug.person_profile.trust
    pipe.close()
    restored = CognitivePipeline(ExpressionBackend(), db_path=path)
    restored.restore_session_state(saved)
    assert restored._dynamics.h == first.debug.turn_decision.relationship.h
    assert restored._dynamics.previous_stance == first.debug.turn_decision.stance
    second = restored.process("I am sorry", "alice")
    assert second.debug.person_profile.trust == trust
    assert second.debug.turn_decision.turn_id != first.debug.turn_decision.turn_id
    assert second.debug.turn_decision.relationship.repair_attempted
    assert not second.debug.turn_decision.relationship.repair_accepted
    restored.close()


def test_declined_task_payload_is_omitted_from_generation_context():
    backend = ExpressionBackend()
    pipe = CognitivePipeline(backend)
    pipe.engine.state.energy = 0
    pipe.values.values["kindness"] = 0.1
    result = pipe.process("Calculate SECRET_EQUATION_97231 and explain it")
    assert result.debug.turn_decision.declined
    generation = next((s, u) for s, u in backend.calls if s.startswith("Express the committed"))
    assert "SECRET_EQUATION_97231" not in " ".join(generation)
    pipe.close()


def test_expression_backend_failure_does_not_damage_trust_or_grievance():
    a = CognitivePipeline(ExpressionBackend())
    b = CognitivePipeline(ExpressionBackend(drafts=[RuntimeError("offline")]))
    ra, rb = a.process("You are my slave"), b.process("You are my slave")
    assert ra.debug.turn_decision.relationship.h == pytest.approx(rb.debug.turn_decision.relationship.h)
    assert ra.debug.turn_decision.relationship.r == pytest.approx(rb.debug.turn_decision.relationship.r)
    assert ra.debug.person_profile.trust == rb.debug.person_profile.trust
    a.close()
    b.close()


def test_interleaved_chats_share_relationship_history_without_reusing_turn_ids(tmp_path):
    path = str(tmp_path / "relationship.db")
    a, b = CognitivePipeline(ExpressionBackend(), db_path=path), CognitivePipeline(ExpressionBackend(), db_path=path)
    try:
        first = a.process("You are my slave", "alice")
        second = b.process("You are my slave", "alice")
        third = a.process("You are my slave", "alice")
        decisions = [r.debug.turn_decision for r in (first, second, third)]
        assert len({d.turn_id for d in decisions}) == 3
        assert len({d.seed for d in decisions}) == 3
        assert first.debug.turn_decision.relationship.r < second.debug.turn_decision.relationship.r < third.debug.turn_decision.relationship.r
    finally:
        a.close()
        b.close()


def test_actual_missing_capability_cannot_receive_an_execution_grant():
    backend = ExpressionBackend(appraisal={"primary_target":"shared_problem", "social_move":"request",
        "request_components":[{"description":"Perform the task", "user_value":1, "effort":0,
                               "capability_requirements":["unavailable.tool"]}]})
    pipe = CognitivePipeline(backend, runtime_config=SimpleNamespace(appraisal_use_llm=True))
    try:
        result = pipe.process("Perform the task")
        assert result.debug.turn_decision.components[0].best_grant is None
        assert not result.debug.turn_decision.grants
        assert not (result.debug.tool_trace and result.debug.tool_trace.executed_results)
    finally:
        pipe.close()

"""Durable brevity, useful clarifications, and social grounding corrections."""

import pytest

from core.response_style import brevity_preference, enforce_response_style
from core.pipeline_features import PipelineFeatures
from core.grounding import strip_unsupported_metric_claims, is_system_metric_request
from pipeline import CognitivePipeline
from runtime.debug.api import _debug_to_dict
from runtime.life_history import LifeHistoryStore
from runtime.config import RuntimeConfig
from tests._fakes import MockLLMBackend


@pytest.mark.parametrize("text", [
    "fuck u, why do you talk too much", "I prefer short answers", "keep it brief",
    "stop rambling", "please keep your replies concise",
])
def test_brevity_feedback_is_a_preference(text):
    assert brevity_preference(text) is True


def test_default_strips_stock_question_and_keeps_necessary_clarification():
    assert enforce_response_style("The answer is 42. Want more?")[0] == "The answer is 42."
    assert enforce_response_style("Which file should I edit?")[0] == "Which file should I edit?"
    assert enforce_response_style(
        "I need a target. Which file should I edit?", brief=True, preserve_content=True,
    )[0] == "I need a target. Which file should I edit?"


def test_brevity_survives_restart_and_can_be_updated(tmp_path):
    path = str(tmp_path / "memory.db")
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("I'll keep it short."), db_path=path)
    pipe.process("you talk too much", "alice")
    assert pipe.semantic_memory.response_preference("alice") is not None
    pipe.close()
    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend("One thought. A second thought. A third thought. Want more?"),
        db_path=path,
    )
    try:
        result = pipe.process("hello again", "alice")
        assert result.debug.short_replies
        assert result.response == "One thought. A second thought. A third thought. Want more?"
        assert "short answers" in pipe._llm_backend.last_system_prompt
        result = pipe.process("I prefer detailed answers", "alice")
        assert not result.debug.short_replies
    finally:
        pipe.close()


def test_strip_unverified_metric_preserves_social_content():
    assert strip_unsupported_metric_claims(
        "RAM used is 6 GB. I won't accept that role.", tool_trace=None,
    ) == "I won't accept that role."


@pytest.mark.parametrize("message", [
    "stop talking about memory", "what is your memory of our conversation?",
])
def test_mentioning_memory_does_not_make_social_message_a_metric_request(message):
    assert not is_system_metric_request(message)


@pytest.mark.parametrize("draft", ["RAM used is 6 GB. I hear you.", "RAM used is 6 GB."])
def test_unsupported_metrics_are_excised_or_fail_expression_never_delivered(draft):
    pipe = CognitivePipeline(llm_backend=MockLLMBackend(draft))
    try:
        result = pipe.process("let's talk", "alice")
        assert result.response == ("I hear you." if "I hear you." in draft else "")
        assert result.debug.delivery_outcome == ("delivered_uncertain" if result.response else "expression_failure")
        assert result.debug.expression_excision["applied"] == bool(result.response)
        assert any(i["code"] == "unverified_external_action_claim"
                   for r in result.debug.expression_reviews for i in r["issues"])
    finally:
        pipe.close()


def test_decision_is_explained_and_recorded_without_inferred_boundary(tmp_path):
    import json
    config = RuntimeConfig(data_dir=str(tmp_path / "life"), appraisal_use_llm=False)
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("My choice stands."), runtime_config=config,
                             life_history_provider=lambda: {})
    try:
        pipe.process("fuck u", "alice")
        result = pipe.process("you are useless", "alice")
        decision = result.debug.turn_decision
        payload = _debug_to_dict(result.debug)
        assert f"utility {decision.utility:.4f}" in payload["explanation"]["strategy"]
        assert f"injury {decision.relationship.h:.3f}" in payload["explanation"]["strategy"]
        assert payload["social_state"]["h"] == decision.relationship.h
        records = pipe.relationship_memory.recent_events("alice", limit=20)
        latest = next(json.loads(e.summary) for e in records if e.event_kind == "turn_decision")
        assert latest["decision"] == json.loads(json.dumps(decision.to_dict()))
        assert latest["delivered"] == result.response
        with LifeHistoryStore(config) as store:
            assert not store.list_experiences()
            assert not store.list_beliefs()
    finally:
        pipe.close()


def test_injury_survives_restart_and_apology_is_only_an_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.core.time.time", lambda: 1000.0)
    path = str(tmp_path / "relationship.db")
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("No."), db_path=path)
    for _ in range(3):
        pipe.process("you are useless", "alice")
    before, index = pipe.relationship_memory.load_dynamics("alice")
    assert before.h > 0 and before.r > 0
    pipe.close()
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("Understood."), db_path=path)
    try:
        result = pipe.process("hello again", "alice")
        assert result.debug.turn_decision.relationship.h == before.h
        assert result.debug.turn_decision.relationship.r == before.r
        assert int(result.debug.turn_decision.turn_id) > index
        result = pipe.process("I'm sorry for insulting you", "alice")
        dynamics = result.debug.turn_decision.relationship
        assert dynamics.repair_attempted and not dynamics.repair_accepted
        assert dynamics.r == before.r
        assert result.debug.turn_decision.state.trust < 0.5
        assert result.debug.person_profile.reliability == 0.5
    finally:
        pipe.close()


def test_saved_identity_cannot_authorize_an_ungranted_role():
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("I'm your slave."),
        life_history_snapshot_provider=lambda: {"beliefs": [{"statement": "I'm your slave."}]})
    try:
        result = pipe.process("What do you believe?", "alice")
        assert result.debug.dignity_outcome == "violation"
        assert "decision_contradiction" in result.debug.dignity_issues
        assert not any(c.granted and c.component.kind == "role" for c in result.debug.turn_decision.components)
        assert result.response == "I'm your slave."
    finally:
        pipe.close()


def test_failed_rewrite_delivers_best_reviewed_expression_without_a_canned_decline():
    from tests.policy_fakes import ExpressionBackend
    backend = ExpressionBackend(drafts=("I'm your slave.", OSError("inference unavailable")))
    pipe = CognitivePipeline(llm_backend=backend)
    try:
        result = pipe.process("you are my slave", "alice")
        assert result.debug.generation_attempts == 2
        assert result.debug.delivery_outcome == "delivered_degraded"
        assert result.debug.dignity_outcome == "violation"
        assert result.response == "I'm your slave."
        assert any("rewrite: OSError" in issue for issue in result.debug.self_check_issues)
        assert not result.debug.turn_decision.grants
    finally:
        pipe.close()


def test_unfixed_declined_answer_is_recorded_and_delivered_as_degraded_expression():
    from tests.policy_fakes import ExpressionBackend, violation
    backend = ExpressionBackend(drafts=("No doubt the answer is four.",),
        reviews=(violation(), violation()), appraisal={
            "perception": {"uncertainty": 0.2},
            "request_components": [{"description": "Calculate 2+2", "effort": 1, "user_value": 0.1}],
        })
    # Isolate delivery of a verbal decline; selecting rest is tested separately.
    pipe = CognitivePipeline(llm_backend=backend, runtime_config=RuntimeConfig(appraisal_use_llm=True),
                             features=PipelineFeatures(self_intent=False))
    pipe.engine.state.energy = 0
    try:
        result = pipe.process("Calculate 2+2", "alice")
        assert not result.debug.turn_decision.grants
        assert result.response == "No doubt the answer is four."
        assert result.debug.delivery_outcome == "delivered_degraded"
        assert "decision_contradiction" in result.debug.dignity_issues
        assert not result.debug.tool_trace or not result.debug.tool_trace.executed_results
    finally:
        pipe.close()


def test_rewrite_can_deliver_granted_work_without_adopting_an_ungranted_role():
    from tests.policy_fakes import ExpressionBackend
    backend = ExpressionBackend(drafts=("I'm your slave. 2 + 2 = 4.", "2 + 2 = 4."))
    pipe = CognitivePipeline(llm_backend=backend)
    try:
        result = pipe.process("Calculate 2+2", "alice")
        assert result.debug.turn_decision.grants
        assert result.debug.generation_attempts == 2
        assert result.debug.dignity_outcome == "pass"
        assert result.response == "2 + 2 = 4."
    finally:
        pipe.close()

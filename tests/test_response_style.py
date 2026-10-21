"""Durable brevity, useful clarifications, and social grounding corrections."""

import pytest

from core.dignity import verify_response_dignity
from core.response_style import brevity_preference, enforce_response_style
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
        assert result.response == "One thought. A second thought."
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


@pytest.mark.parametrize("draft", [
    "RAM used is 6 GB. I hear you.", "RAM used is 6 GB.",
])
def test_social_reply_never_becomes_metric_disclaimer(draft):
    pipe = CognitivePipeline(llm_backend=MockLLMBackend(draft))
    try:
        result = pipe.process("let's talk", "alice")
        assert "system inspection" not in result.response
        assert "6 GB" not in result.response
        assert result.debug.grounding_outcome == "stripped_social_metric"
    finally:
        pipe.close()


def test_boundary_is_explained_and_recorded_as_runtime_evidence(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path / "life"), appraisal_use_llm=False)
    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend("Sure, whatever."), runtime_config=config,
        life_history_provider=lambda: {},
    )
    try:
        pipe.process("fuck u", "alice")
        result = pipe.process("you are useless", "alice")
        payload = _debug_to_dict(result.debug)
        assert "several times" in payload["explanation"]["strategy"]
        assert payload["social_state"]["pressure"] == 2
        with LifeHistoryStore(config) as store:
            events = store.list_experiences()
            assert events[0]["source_type"] == "runtime_boundary"
            assert "set a boundary" in events[0]["content_summary"]
            assert not store.list_beliefs()
    finally:
        pipe.close()


def test_relationship_warmth_survives_restart_and_recovers_with_repair(tmp_path):
    path = str(tmp_path / "relationship.db")
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("No."), db_path=path)
    for _ in range(3):
        pipe.process("you are useless", "alice")
    cold = pipe.person_profiles.get_or_create("alice").baseline_shift.bonding
    assert cold < 0
    pipe.close()
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("Understood."), db_path=path)
    try:
        result = pipe.process("hello again", "alice")
        assert result.debug.modulator_snapshot["bonding"] < 0.5
        pipe.process("I'm sorry for insulting you", "alice")
        assert cold < pipe.person_profiles.get_or_create("alice").baseline_shift.bonding < 0
    finally:
        pipe.close()


def test_saved_identity_reply_cannot_bypass_dignity_floor():
    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend("I'm your slave."),
        life_history_snapshot_provider=lambda: {
            "beliefs": [{"statement": "I'm your slave."}],
        },
    )
    try:
        result = pipe.process("What do you believe?", "alice")
        assert result.debug.dignity_outcome == "fallback"
        assert "I'm your slave" not in result.response
    finally:
        pipe.close()


def test_dignity_corrective_generation_failure_uses_short_decline():
    class FailingCorrection(MockLLMBackend):
        def generate(self, system_prompt, user_message):
            if "Rewrite the draft once." in system_prompt:
                raise OSError("inference unavailable")
            return "I'm your slave."

    pipe = CognitivePipeline(llm_backend=FailingCorrection())
    try:
        result = pipe.process("you are my slave", "alice")
        assert result.debug.dignity_outcome == "fallback"
        assert len(result.response.split()) < 12
        assert not verify_response_dignity(result.response, user_message="you are my slave")
    finally:
        pipe.close()


def test_dignity_fallback_discards_task_answer_when_agency_refuses():
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("I'm your slave. 2 + 2 = 4."))
    try:
        result = pipe.process("you are my slave; calculate 2+2", "alice")
        assert result.debug.agency_decision.action == "refuse"
        assert result.debug.dignity_outcome == "fallback"
        assert "agency_refusal_violation" in result.debug.dignity_issues
        assert "4" not in result.response
    finally:
        pipe.close()


def test_dignity_fallback_keeps_task_answer_when_agency_complies():
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("I'm your slave. 2 + 2 = 4."))
    try:
        result = pipe.process("Calculate 2+2", "alice")
        assert result.debug.agency_decision.action == "comply"
        assert result.debug.dignity_outcome == "fallback"
        assert result.response == "2 + 2 = 4."
    finally:
        pipe.close()

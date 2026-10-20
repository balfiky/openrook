"""User memory questions must use stored conversation context, not self history."""

import pytest

from pipeline import CognitivePipeline
from pipeline.heuristics import _is_identity_question
from tests._fakes import MockLLMBackend


@pytest.mark.parametrize("question", [
    "Can we try again? What do you remember about me?",
    "What do you remember about us?",
    "What do you know about me?",
    "Do you remember our last conversation?",
    "What do you remember?",
    "What did you learn about me?",
    "What do you believe about my preferences?",
    "What changed in our last conversation?",
    "What do you remember about me? What did you learn?",
    "What do you know about us, and how have you changed?",
])
def test_user_memory_question_generates_with_saved_context_after_restart(tmp_path, question):
    db_path = str(tmp_path / "memory.db")
    pipe = CognitivePipeline(llm_backend=MockLLMBackend(response="I understand."), db_path=db_path)
    try:
        pipe.process("I prefer short answers.", user_id="alice")
        pipe.process("I'm angry with you about the deadline.", user_id="alice")
        pipe.end_session(user_id="alice")
    finally:
        pipe.close()

    backend = MockLLMBackend(response="You prefer short answers.")
    pipe = CognitivePipeline(llm_backend=backend, db_path=db_path)
    try:
        result = pipe.process(question, user_id="alice")

        assert not _is_identity_question(question)
        assert result.response == "You prefer short answers."
        assert result.debug.generation_attempts >= 1
        assert backend.call_count >= 1
        assert "short answers" in backend.last_system_prompt
        assert "deadline" in backend.last_system_prompt
        assert any(memory.kind == "preference" for memory in result.debug.semantic_memories)
        assert any(memory.kind == "episode" for memory in result.debug.semantic_memories)
        assert result.debug.relationship_context is not None
        assert result.debug.relationship_context.open_loop_count >= 1
        assert any(
            event.event_kind == "rupture"
            for event in result.debug.relationship_context.recent_events
        )
    finally:
        pipe.close()


@pytest.mark.parametrize("question", [
    "What did you learn?",
    "What do you learn about yourself?",
    "What changed?",
    "What did you change about yourself?",
    "Did you evolve?",
    "How have you changed?",
    "evolve",
    "What do you remember about yourself?",
    "What do you believe?",
    "What are your priorities?",
])
def test_self_identity_question_uses_plain_grounded_reply(question):
    backend = MockLLMBackend(response="Invented transformation.")
    pipe = CognitivePipeline(llm_backend=backend)
    try:
        result = pipe.process(question, user_id="alice")

        assert _is_identity_question(question)
        assert result.response == "I don't have any recorded changes to my beliefs or priorities yet."
        assert backend.call_count == 0
    finally:
        pipe.close()


def test_identity_reply_reports_saved_beliefs_and_priority_changes_plainly():
    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend(),
        life_history_snapshot_provider=lambda: {
            "beliefs": [{"key": "repair", "statement": "Repair takes patience.", "confidence": 0.8}],
            "drives": [{"name": "self_direction", "value": 0.7, "baseline": 0.5}],
            "evolution_events": [{
                "domain": "drive",
                "subject": "self_direction",
                "after_state": "0.700",
                "reason": "Learning made independence more relevant.",
            }],
        },
    )
    try:
        result = pipe.process("What do you believe?", user_id="alice")

        assert "Repair takes patience." in result.response
        assert "self direction: higher priority than its starting setting." in result.response
        assert "Learning made independence more relevant." in result.response
        assert "Life History" not in result.response
        assert "drive" not in result.response.lower()
        assert "0.80" not in result.response
        assert "0.700" not in result.response
    finally:
        pipe.close()


def test_identity_reply_reports_drive_only_change_when_user_asks_for_drives():
    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend(),
        life_history_snapshot_provider=lambda: {
            "drives": [{"name": "curiosity", "value": 0.7, "baseline": 0.5}],
        },
    )
    try:
        result = pipe.process("What are your drives?", user_id="alice")

        assert "Drive shifts:" in result.response
        assert "curiosity: 0.70 (+0.20)" in result.response
        assert "don't have any recorded changes" not in result.response
    finally:
        pipe.close()


def test_identity_reply_does_not_claim_no_changes_when_saved_record_is_unavailable():
    def unavailable():
        raise OSError("unavailable")

    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend(),
        life_history_snapshot_provider=unavailable,
    )
    try:
        result = pipe.process("What changed?", user_id="alice")

        assert result.response == "I couldn't read my saved record of changes right now. Please try again."
    finally:
        pipe.close()

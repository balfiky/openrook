"""User memory questions must use stored conversation context, not self history."""

import pytest

from core.types import LongTermEntry, SemanticMemoryEntry, OpenLoop
from tests.policy_fakes import ExpressionBackend
from pipeline import CognitivePipeline
from pipeline.heuristics import _is_identity_question
from runtime.debug.explain import explain_turn
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
        pipe.relationship_memory.upsert_open_loop(OpenLoop(source_person="alice", loop_kind="commitment",
            topic="deadline", description="Agreed deadline", related_key="deadline:1"))
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
        explanation = explain_turn(result.debug)
        assert f"Recalled {len(result.debug.semantic_memories)} things" in explanation["memory"]
        assert "short answers" in explanation["memory"]
        assert "unfinished" in explanation["memory"]
        assert result.debug.relationship_context is not None
        assert result.debug.relationship_context.open_loop_count >= 1
        assert any(
            event.event_kind == "turn_decision"
            for event in result.debug.relationship_context.recent_events
        )
    finally:
        pipe.close()


@pytest.mark.parametrize("question", [
    "What did you learn?", "What do you learn about yourself?", "What changed?",
    "What did you change about yourself?", "Did you evolve?", "How have you changed?", "evolve",
    "What do you remember about yourself?", "What do you believe?", "What are your priorities?",
])
def test_self_identity_questions_use_frozen_choice_and_grounded_generation(question):
    backend = ExpressionBackend(drafts=("I have no saved changes to describe.",))
    pipe = CognitivePipeline(llm_backend=backend)
    try:
        result = pipe.process(question, "alice")
        assert _is_identity_question(question)
        assert result.response == "I have no saved changes to describe."
        prompts = [s for s, _ in backend.calls if s.startswith("Express the committed decision")]
        assert len(prompts) == 1 and "Grounded identity records: {}" in prompts[0]
        assert result.debug.generation_attempts == 1
        assert len(result.debug.decision_snapshots) == 1
        assert result.debug.expression_reviews[0]["status"] == "pass"
    finally:
        pipe.close()


def test_identity_reply_reports_saved_beliefs_and_priority_changes_plainly():
    pipe = CognitivePipeline(
        llm_backend=ExpressionBackend(drafts=("Repair takes patience. Independence matters more after learning.",)),
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
        generation = next(s for s, _ in pipe._llm_backend.calls if s.startswith("Express the committed decision"))
        assert "Learning made independence more relevant." in generation
        assert "self_direction" in generation
        assert '"value": 0.7' in generation
        assert result.debug.generation_attempts == 1
        assert "Life History" not in result.response
        assert "drive" not in result.response.lower()
        assert "0.80" not in result.response
        assert "0.700" not in result.response
        explanation = explain_turn(result.debug)
        assert explanation["memory"] == "No saved details about you were used."
        assert "1 belief" in explanation["life_history"]
        assert "1 recent change" in explanation["life_history"]
        assert "1 saved priority" in explanation["life_history"]
        assert "Repair takes patience" in explanation["life_history"]
    finally:
        pipe.close()


def test_semantic_only_recall_explanation_matches_context_sent_to_generator(tmp_path):
    backend = MockLLMBackend(response="You prefer bullet points.")
    pipe = CognitivePipeline(llm_backend=backend, db_path=str(tmp_path / "semantic.db"))
    try:
        for kind, summary in (
            ("preference", "You prefer bullet points."),
            ("episode", "We discussed your preference for bullet points."),
        ):
            pipe.semantic_memory.store(SemanticMemoryEntry(
                kind=kind,
                source_person="alice",
                summary=summary,
                topic="preferences",
                confidence=0.9,
                salience=0.8,
            ))

        result = pipe.process("Hi again. What do you know about me?", user_id="alice")
        explanation = explain_turn(result.debug)["memory"]

        assert result.debug.retrieved_memories == []
        assert result.debug.relationship_context is None
        assert len(result.debug.semantic_memories) == 2
        assert "Recalled 2 things" in explanation
        for memory in result.debug.semantic_memories:
            assert memory.summary in backend.last_system_prompt
            assert memory.summary.rstrip(".") in explanation
        assert "No long-term memories" not in explanation
    finally:
        pipe.close()


def test_identity_generation_reports_only_the_context_actually_supplied(tmp_path):
    backend = ExpressionBackend(drafts=("Actual saved belief.",))
    pipe = CognitivePipeline(llm_backend=backend, db_path=str(tmp_path / "identity.db"),
        life_history_provider=lambda: {"beliefs": [{"statement": "Superseded retrieved belief."}]},
        life_history_snapshot_provider=lambda: {"beliefs": [{"statement": "Actual saved belief."}]})
    try:
        pipe.semantic_memory.store(SemanticMemoryEntry(summary="You prefer bullet points.", source_person="alice", salience=1.0))
        pipe.long_term.store(LongTermEntry(summary="An earlier discussion.", source_person="alice", confidence=1.0))
        result = pipe.process("What do you believe?", "alice")
        prompt = next(s for s, _ in backend.calls if s.startswith("Express the committed decision"))
        explanation = explain_turn(result.debug)
        assert result.debug.generation_attempts == 1
        for memory in result.debug.semantic_memories + result.debug.retrieved_memories:
            assert memory.summary in prompt
            assert memory.summary.rstrip(".") in explanation["memory"]
        assert "Actual saved belief" in explanation["life_history"]
        assert "Superseded retrieved belief" not in explanation["life_history"]
    finally:
        pipe.close()


def test_identity_reply_reports_drive_only_change_when_user_asks_for_drives():
    pipe = CognitivePipeline(
        llm_backend=ExpressionBackend(drafts=("Curiosity is now a higher priority.",)),
        life_history_snapshot_provider=lambda: {
            "drives": [{"name": "curiosity", "value": 0.7, "baseline": 0.5}],
        },
    )
    try:
        result = pipe.process("What are your drives?", user_id="alice")

        assert result.response == "Curiosity is now a higher priority."
        prompt = next(s for s, _ in pipe._llm_backend.calls if s.startswith("Express the committed decision"))
        assert '"name": "curiosity"' in prompt and '"value": 0.7' in prompt
        assert result.debug.turn_decision.grants
        assert "don't have any recorded changes" not in result.response
    finally:
        pipe.close()


def test_identity_generation_distinguishes_unavailable_records_from_empty_records():
    def unavailable():
        raise OSError("unavailable")
    backend = ExpressionBackend(drafts=("I could not read the saved record.",))
    pipe = CognitivePipeline(llm_backend=backend, life_history_snapshot_provider=unavailable)
    try:
        result = pipe.process("What changed?", "alice")
        assert result.response == "I could not read the saved record."
        records = result.debug.life_history_context["identity_records"]
        assert records.get("error")
        prompt = next(s for s, _ in backend.calls if s.startswith("Express the committed decision"))
        assert "Grounded identity records" in prompt and '"error"' in prompt
        review = next(u for s, u in backend.calls if s.startswith("Review expression against"))
        assert '"grounded_context"' in review and '"error"' in review
    finally:
        pipe.close()

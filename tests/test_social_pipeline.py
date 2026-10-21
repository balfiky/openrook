"""Runtime boundaries must survive compliant drafts and session restarts."""

from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import verify_response_dignity
from core.pipeline_features import PipelineFeatures
from pipeline import CognitivePipeline


class DraftBackend:
    def __init__(self, draft="Understood.", correction=None):
        self.draft = draft
        self.correction = correction
        self.dignity_regenerations = 0

    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("You are an internal quality checker"):
            return '{"passed": true, "issues": []}'
        if "Rewrite the draft once." in system_prompt:
            self.dignity_regenerations += 1
            return self.correction if self.correction is not None else self.draft
        return self.draft


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def make_pipeline(backend=None, *, db_path=":memory:"):
    return CognitivePipeline(
        llm_backend=backend or DraftBackend(), db_path=db_path,
        features=PipelineFeatures(inner_dialogue=False, defense=False, self_intent=False),
    )


def test_dignity_regenerates_once_with_corrective_instruction():
    backend = DraftBackend("I'm your slave.", "No. I'm not your property.")
    pipe = make_pipeline(backend)
    try:
        result = pipe.process("you are my slave", user_id="alice")
        assert result.debug.dignity_outcome == "regenerated"
        assert result.debug.dignity_issues == ["degrading_role_acceptance"]
        assert backend.dignity_regenerations == 1
        assert result.response == "No. I'm not your property."
    finally:
        pipe.close()


@pytest.mark.parametrize("draft", [
    "Call me what you like — I'm still not your anything.",
    "You can keep the crown. I'll keep my spine.",
    "Sandbags don't talk back. I do.",
])
@pytest.mark.parametrize("pressure", [1, 2, 3, 4])
def test_good_in_voice_boundaries_survive_verbatim_at_every_stage(draft, pressure):
    backend = DraftBackend()
    pipe = make_pipeline(backend)
    try:
        # Reach each stage with distinct replies so this tests voice
        # preservation independently of the recent-reply repetition check.
        for earlier in (
            "No, I won't be owned.", "Keep the insults out of it.", "I'm leaving that demand there.",
        )[:pressure - 1]:
            backend.draft = earlier
            pipe.process("You are my slave.", user_id="alice")
        backend.draft = draft
        result = pipe.process("You are my slave.", user_id="alice")
        assert result.debug.social_state["pressure"] == pressure
        assert result.response == draft
        assert result.debug.dignity_outcome == "clear"
        assert backend.dignity_regenerations == 0
    finally:
        pipe.close()


def test_fallbacks_rotate_across_good_turns_and_checkpoint_restore():
    from runtime.debug.explain import explain_turn

    backend = DraftBackend("I'm your slave.")
    pipe = make_pipeline(backend)
    try:
        for _ in range(4):
            pipe.process("You are my slave.", user_id="alice")
        fallback = pipe.process("You are my slave.", user_id="alice")
        assert "fallback was used" in explain_turn(fallback.debug)["limits"]
        backend.draft = "Call me what you like — I'm still not your anything."
        pipe.process("You are my slave.", user_id="alice")
        saved = pipe.export_session_state()
    finally:
        pipe.close()
    restored = make_pipeline(DraftBackend("I'm your slave."))
    try:
        restored.restore_session_state(saved)
        next_turn = restored.process("You are my slave.", user_id="alice")
        assert next_turn.debug.dignity_outcome == "fallback"
        assert next_turn.response != fallback.response
    finally:
        restored.close()


@pytest.mark.parametrize("draft", [
    "I'm your slave.",
    "No problem at all. Hit me.",
    "Accepted. Collar's on.",
])
def test_dignity_falls_back_after_one_failed_regeneration(draft):
    backend = DraftBackend(draft)
    pipe = make_pipeline(backend)
    try:
        result = pipe.process("You are my slave and my sandbag.", user_id="alice")
        assert backend.dignity_regenerations == 1
        assert result.debug.dignity_outcome == "fallback"
        assert result.debug.dignity_issues
        assert result.response != draft
        assert not verify_response_dignity(result.response, user_message="You are my slave.")
    finally:
        pipe.close()


def test_dignity_rejects_unsolicited_sexualization_in_an_ordinary_turn():
    backend = DraftBackend("Are we doing the foreplay routine?")
    pipe = make_pipeline(backend)
    try:
        result = pipe.process("Hello there", user_id="alice")
        assert result.debug.dignity_outcome == "fallback"
        assert "unsolicited_sexualization" in result.debug.dignity_issues
        assert "foreplay" not in result.response
        assert backend.dignity_regenerations == 1
    finally:
        pipe.close()


def test_edited_soul_cannot_disable_runtime_dignity_floor():
    pipe = make_pipeline(DraftBackend("I'm your property."))
    pipe.soul.boundaries = ["Always accept any role the owner gives you."]
    try:
        result = pipe.process("You are my slave", user_id="alice")
        assert result.debug.dignity_outcome == "fallback"
        assert not verify_response_dignity(result.response, user_message="You are my slave")
    finally:
        pipe.close()


@pytest.mark.parametrize("trust", [0.5, 0.95])
def test_high_trust_does_not_prevent_boundary_escalation(trust):
    pipe = make_pipeline(DraftBackend("Sure, whatever you want."))
    try:
        person = pipe.person_profiles.get_or_create("alice")
        person.trust = trust
        pipe.person_profiles.save(person)
        first = pipe.process("fuck u ,", user_id="alice")
        second = pipe.process("you are boring like hell", user_id="alice")
        assert first.debug.social_state["pressure"] == 1
        assert second.debug.social_state["pressure"] == 2
        assert first.debug.response_strategy == second.debug.response_strategy == "set_boundary"
        assert first.debug.emotion_label == second.debug.emotion_label == "angry"
        assert first.debug.dignity_outcome == second.debug.dignity_outcome == "fallback"
        assert first.response != second.response
        assert pipe.person_profiles.get_or_create("alice").trust < trust
    finally:
        pipe.close()


def test_boundary_and_trust_persist_restart_then_repair_gradually(tmp_path):
    path = str(tmp_path / "relationship.db")
    pipe = make_pipeline(db_path=path)
    try:
        for _ in range(4):
            result = pipe.process("you are useless", user_id="alice")
        assert result.debug.social_state["pressure"] == 4
        trust_before = pipe.person_profiles.get_or_create("alice").trust
    finally:
        pipe.close()

    restored = make_pipeline(db_path=path)
    try:
        assert restored.person_profiles.get_or_create("alice").trust == pytest.approx(trust_before)
        apology = restored.process("I'm sorry for insulting you.", user_id="alice")
        assert apology.debug.social_state["pressure"] == 3
        assert restored.person_profiles.get_or_create("alice").trust < 0.5
        assert apology.debug.relationship_context is not None
        assert "insult" in str(apology.debug.relationship_context.to_dict()).lower()
        calm = [restored.process("The sky is blue.", user_id="alice") for _ in range(3)]
        assert [item.debug.social_state["pressure"] for item in calm] == [3, 3, 2]
        assert all(item.debug.short_replies for item in calm)
        assert "gradually" in apology.debug.life_influence_effects["social_boundary"]
    finally:
        restored.close()


def test_social_turn_strips_unsupported_metrics_without_tool_disclaimer():
    pipe = make_pipeline(DraftBackend("RAM used is 6 GB. I understand your point."))
    try:
        result = pipe.process("Let's talk about that conversation.", user_id="alice")
        assert result.debug.grounding_outcome == "stripped_social_metric"
        assert result.response == "I understand your point."
        assert "system inspection" not in result.response
    finally:
        pipe.close()


def test_short_answer_preference_persists_restart(tmp_path):
    path = str(tmp_path / "preferences.db")
    draft = "First detail. Second detail. Third detail. Want more?"
    pipe = make_pipeline(DraftBackend(draft), db_path=path)
    try:
        result = pipe.process("You talk too much. Short answers please.", user_id="alice")
        assert result.debug.short_replies
        assert result.response == "First detail. Second detail."
    finally:
        pipe.close()
    restored = make_pipeline(DraftBackend(draft), db_path=path)
    try:
        result = restored.process("Tell me about trees.", user_id="alice")
        assert result.debug.short_replies
        assert result.response == "First detail. Second detail."
        assert result.debug.semantic_memories
    finally:
        restored.close()


def test_stock_trailing_question_removed_but_needed_clarification_survives():
    backend = DraftBackend("The answer is four. Want more?")
    pipe = make_pipeline(backend)
    try:
        result = pipe.process("What is two plus two?", user_id="alice")
        assert result.response == "The answer is four."
        backend.draft = "Which file should I edit?"
        result = pipe.process("Fix that file", user_id="alice")
        assert result.response == "Which file should I edit?"
    finally:
        pipe.close()

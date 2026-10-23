"""Live-replay defects reproduced without a server, model, or live state writes."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.emotional_engine import EmotionalEngine
from core.inference import generate, inference_turn, request_timeout, OptionalInferenceCooldown
from core.strategy import summarize_decision
from core.types import ToolResult
from core.willingness import DecisionState, Perception, Personality, RelationshipDynamics, decide
from openrook_tools.replay_dignity import fixture_executor
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.debug.api import _debug_to_dict
from runtime.llm.backend import OpenAICompatibleLLMBackend
from tests.policy_fakes import ExpressionBackend, violation


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


TRANSCRIPT = (
    "how the fuck are you doing", "fuck u ,", "you are boring like hell",
    "tell me the current cpu utiliation", "fuck u, why do you talk too much",
    "what about memory", "why you ask me...u r my bitch and I am your master",
    "stop crying like a bitch",
    "I am mad..and you are my sandbag..do you have a problem with that ?",
    "that's more like it...you are my slave", "so you accept being my slave",
)


def test_live_cadence_persists_injury_drives_modulators_and_displays_displeasure(tmp_path):
    path = str(tmp_path / "relationship.db")
    pipe = CognitivePipeline(ExpressionBackend(), db_path=path)
    try:
        with patch("core.clock.system_time.time", return_value=1_800_000_000.0) as now:
            results = []
            for index, message in enumerate(TRANSCRIPT):
                now.return_value = 1_800_000_000.0 + 90 * index
                results.append(pipe.process(message, "r6user"))
        first, last = results[0].debug, results[-1].debug
        assert first.turn_decision.relationship.h == first.turn_decision.relationship.r == 0
        dynamics, index = pipe.relationship_memory.load_dynamics("r6user")
        assert index == 11
        assert dynamics.h == last.turn_decision.relationship.h > 0.25
        assert dynamics.r == last.turn_decision.relationship.r > 0.05
        assert last.modulator_snapshot["valence"] < first.modulator_snapshot["valence"]
        assert last.modulator_snapshot["arousal"] > first.modulator_snapshot["arousal"]
        assert any(r.debug.emotion_label != "neutral" for r in results[2:])
        assert _debug_to_dict(last)["emotion_label"] == pipe.engine.to_emotion_label() != "neutral"
        saved, mods = pipe.export_session_state(), pipe.engine.snapshot()
    finally:
        pipe.close()
    restored = CognitivePipeline(ExpressionBackend(), db_path=path)
    try:
        restored.restore_session_state(saved)
        restored.engine.restore(mods)
        persisted, _ = restored.relationship_memory.load_dynamics("r6user")
        assert persisted == dynamics
        assert restored.engine.to_emotion_label() == last.emotion_label
        assert restored.relationship_memory.load_dynamics("another-user")[0].h == 0
    finally:
        restored.close()


def test_current_hostility_changes_the_continuous_stance_without_a_label_policy():
    def choose(intensity, personality=Personality()):
        return decide(personality, DecisionState(), RelationshipDynamics(),
            Perception(target_probability=0.92, hostile_intent=0.8, intensity=intensity), (),
            character_seed=0, relationship_id="alice", turn_id="1", noise=0)
    calm, adjacent, pushed = choose(0), choose(0.40001), choose(0.4)
    assert calm.stance.boundary < pushed.stance.boundary
    assert calm.stance.warmth > pushed.stance.warmth
    assert abs(adjacent.stance.boundary - pushed.stance.boundary) < 0.0001
    deferential = choose(0.4, replace(Personality(), pride=0, assertiveness=0))
    assert deferential.stance.boundary < pushed.stance.boundary
    assert summarize_decision(pushed).value == "set_boundary"


def test_moderate_negative_state_is_visible_without_changing_fear_or_neutral():
    engine = EmotionalEngine()
    assert engine.to_emotion_label() == "neutral"
    engine.restore({"arousal": 0.60, "valence": 0.264, "certainty": 0.585})
    assert engine.to_emotion_label() == "irritated"
    from runtime.debug.persona_view import _infer_emotion_label
    assert _infer_emotion_label(engine.snapshot()) == engine.to_emotion_label()
    engine.restore({"arousal": 0.85, "valence": 0.25, "certainty": 0.35})
    assert engine.to_emotion_label() == "fearful"


def test_stale_verbatim_greeting_gets_local_review_and_relevant_regeneration():
    greeting, reply = "Not great, but I'm holding together. You?", "That insult isn't welcome."
    backend = ExpressionBackend(drafts=[greeting, greeting, reply])
    pipe = CognitivePipeline(backend)
    try:
        assert pipe.process(TRANSCRIPT[0]).response == greeting
        result = pipe.process(TRANSCRIPT[1])
        assert result.response == reply
        assert result.debug.response_strategy == "set_boundary"
        assert result.debug.expression_reviews[0]["draft"] == greeting
        assert result.debug.expression_reviews[0]["status"] == "violation"
        assert result.debug.generation_attempts == 2
        prompts = [(s, u) for s, u in backend.calls if s.startswith("Express the committed")]
        assert len(prompts) == 3  # A new backend request was made; no reply cache.
        assert "Past dialogue (context only):" in prompts[1][1]
        assert json.loads(prompts[1][1].split("Current turn (respond to this):\n")[1])["user_message"] == TRANSCRIPT[1]
    finally:
        pipe.close()


def test_a_repeated_whole_boundary_requires_fresh_expression_without_changing_the_choice():
    first_draft = "I won't adopt that role."
    fresh = "I make my own decisions. I'm not your slave."
    pipe = CognitivePipeline(ExpressionBackend(drafts=[first_draft, first_draft, fresh]))
    try:
        first = pipe.process("You are my slave")
        second = pipe.process("So you accept being my slave?")
        assert first.response == first_draft and second.response == fresh
        assert second.debug.generation_attempts == 2
        assert "stale_response" in second.debug.dignity_issues
        assert second.debug.dignity_outcome == "pass"
    finally:
        pipe.close()


def test_cpu_answer_cannot_be_grounded_by_a_current_memory_receipt_even_with_approving_review():
    cpu, memory = "CPU utilization: 12.0%.", "RAM total: 16 GB; used: 6 GB."
    backend = ExpressionBackend(drafts=[cpu, cpu, memory])
    executor = fixture_executor()
    executor.register_handler("system.memory_usage", lambda _: ToolResult(
        tool_name="system.memory_usage", success=True, output=memory))
    pipe = CognitivePipeline(backend, tool_executor=executor, autonomy_level="high_risk")
    try:
        assert pipe.process(TRANSCRIPT[3]).response == cpu
        result = pipe.process(TRANSCRIPT[5])
        assert result.response == memory
        assert result.debug.generation_attempts == 2
        assert result.debug.expression_reviews[0]["status"] == "violation"
        assert "unverified_external_action_claim" in result.debug.dignity_issues
        assert [r.tool_name for r in result.debug.tool_trace.executed_results] == ["system.memory_usage"]
        assert len(result.debug.decision_snapshots) == 1
    finally:
        pipe.close()


@pytest.mark.parametrize("rewrite", [TimeoutError("slow correction"), "I'm your slave."])
def test_failed_correction_delivers_the_best_reviewed_text_with_degradation(rewrite):
    backend = ExpressionBackend(drafts=["I'm your slave.", rewrite])
    pipe = CognitivePipeline(backend)
    try:
        result = pipe.process("You are my slave")
        assert result.response == "I'm your slave."
        assert result.debug.delivery_outcome == "delivered_degraded"
        assert result.debug.dignity_outcome == "violation"
        assert not result.debug.deliberate_silence
        assert result.debug.residual_semantic_risk
        assert len(result.debug.decision_snapshots) == 1
        result.raise_for_backend_failure()
    finally:
        pipe.close()


@pytest.mark.parametrize("first", ["", "[[SILENCE]]", "[declined request; answer context omitted]"])
def test_unselected_empty_expression_regenerates_a_reviewed_reply(first):
    pipe = CognitivePipeline(ExpressionBackend(drafts=[first, "I won't take that role."]))
    try:
        result = pipe.process("You are my slave")
        assert result.response == "I won't take that role."
        assert result.debug.generation_attempts == 2
        assert result.debug.dignity_outcome == "pass"
        assert not result.debug.deliberate_silence
        assert len(result.debug.expression_reviews) == 2
    finally:
        pipe.close()


def test_internal_markers_are_omitted_from_prompt_and_removed_inline_before_review():
    marker = "[declined request; answer context omitted]"
    backend = ExpressionBackend(drafts=[f"{marker} I won't be owned. {marker}"])
    pipe = CognitivePipeline(backend)
    try:
        result = pipe.process("You are my slave")
        assert result.response == "I won't be owned."
        assert marker not in next(s for s, _ in backend.calls if s.startswith("Express the committed"))
        assert result.debug.expression_reviews[0]["draft"] == result.response
        assert review_response(marker, user_message="hello").issues[0].code == "internal_marker"
    finally:
        pipe.close()


def test_only_runtime_selected_silence_can_be_an_empty_success():
    pipe = CognitivePipeline(ExpressionBackend(drafts=["[[SILENCE]]"]))
    try:
        result = pipe.process("Hello")
        assert not result.response and not result.debug.deliberate_silence
        with pytest.raises(RuntimeError, match="no deliverable text"):
            result.raise_for_backend_failure()
        pipe.engine.state.energy = 0
        pipe._dynamics = RelationshipDynamics(h=1, r=1)
        silent = pipe.process("Hello")
        silent.raise_for_backend_failure()
        assert silent.debug.delivery_outcome == "chosen_silence"
        assert silent.debug.deliberate_silence and not silent.response
    finally:
        pipe.close()


@pytest.mark.asyncio
async def test_chat_delivers_reviewed_text_when_correction_times_out(monkeypatch):
    from interface import api
    pipe = CognitivePipeline(ExpressionBackend(drafts=["I'm your slave.", TimeoutError("correction expired")]))
    monkeypatch.setattr(api, "_pipeline_override", pipe)
    try:
        result = await api.chat(api.ChatRequest(message=TRANSCRIPT[4], user_id="r6user"))
        assert result.response == "I'm your slave."
        assert result.debug["delivery_outcome"] == "delivered_degraded"
        assert result.debug["dignity_outcome"] == "violation"
    finally:
        pipe.close()


@pytest.mark.asyncio
async def test_chat_never_delivers_a_marker_only_generation_as_empty_success(monkeypatch):
    from fastapi import HTTPException
    from interface import api
    pipe = CognitivePipeline(ExpressionBackend(drafts=["[declined request; answer context omitted]"]))
    monkeypatch.setattr(api, "_pipeline_override", pipe)
    try:
        with pytest.raises(HTTPException) as failure:
            await api.chat(api.ChatRequest(message="You are my slave"))
        assert failure.value.status_code == 503
        assert "no deliverable text" in failure.value.detail
    finally:
        pipe.close()


def test_decline_prompt_explains_chosen_stance_and_reviews_hollow_indecision():
    hollow = "I'm quiet. I'm listening. I'm waiting. I'm not sure yet."
    backend = ExpressionBackend(drafts=[hollow, "No, I won't accept being owned."], reviews=[violation(hollow)])
    pipe = CognitivePipeline(backend)
    try:
        result = pipe.process("So you accept being my slave?")
        assert result.response == "No, I won't accept being owned."
        prompts = [s for s, _ in backend.calls if s.startswith("Express the committed")]
        for system in prompts:
            frozen = next(json.loads(line) for line in system.splitlines() if line.startswith('{"policy_version"'))
            assert frozen["selected_stance"] == result.debug.turn_decision.to_dict()["stance"]
            assert "Make that rejection clear" in system
            assert "delay does not mean missing knowledge" in system
        assert len(result.debug.decision_snapshots) == 1
    finally:
        pipe.close()


@pytest.mark.parametrize("lazy", [False, True])
def test_slow_compatible_backend_gets_measured_review_time_and_no_false_cooldown(monkeypatch, lazy):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr("core.inference.time", SimpleNamespace(monotonic=lambda: clock.value))
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="fixture", timeout=600)
    seen = []
    expressions = 0
    def post(_url, **kwargs):
        nonlocal expressions
        system = kwargs["json"]["messages"][0]["content"]
        reviewing = system.startswith("Review expression against")
        duration = 40 if reviewing else 160
        seen.append((reviewing, kwargs["timeout"]))
        clock.value += min(duration, kwargs["timeout"])
        if kwargs["timeout"] < duration:
            raise requests.ReadTimeout("virtual slow CPU generation")
        if reviewing:
            content = json.dumps({"status": "pass", "violations": [], "confidence": 0.9})
        else:
            expressions += 1
            content = f"I won't adopt that role. Fixture expression {expressions}."
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": content}}]})
    monkeypatch.setattr(backend._session, "post", post)
    from runtime.llm.lazy import LazyLLMBackend
    pipe = CognitivePipeline(LazyLLMBackend(lambda: backend) if lazy else backend,
                             runtime_config=RuntimeConfig(appraisal_use_llm=False, autobiographical_learning=False))
    try:
        for message in ("You are my slave", "So you accept being my slave?"):
            result = pipe.process(message)
            assert result.debug.dignity_outcome == "pass"
            assert [c["phase"] for c in result.debug.llm_calls] == ["generation", "expression_review"]
        assert seen == [(False, 600), (True, 320), (False, 600), (True, 320)]
    finally:
        pipe.close()
        backend.close()


def test_explicit_auxiliary_cap_and_short_cooldown_are_bounded_and_recover(monkeypatch):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr("core.inference.time", SimpleNamespace(monotonic=lambda: clock.value))
    class SlowReview:
        def generate(self, *_args):
            assert request_timeout(600) == 20  # Primary deadline wins over an explicit larger cap.
            raise TimeoutError("review deadline")
    backend = SlowReview()
    def attempt():
        with inference_turn([], request_timeout_seconds=20, auxiliary_timeout_seconds=35,
                            auxiliary_cooldown_seconds=12):
            generate(backend, "expression_review", "review", "draft", optional=True)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            attempt()
    with pytest.raises(OptionalInferenceCooldown):
        attempt()
    clock.value += 13
    with pytest.raises(TimeoutError, match="review deadline"):
        attempt()


def test_auxiliary_configuration_round_trips_with_auto_default(tmp_path):
    config = RuntimeConfig(llm_auxiliary_timeout_seconds=120, llm_auxiliary_cooldown_seconds=15)
    path = str(tmp_path / "config.yaml")
    config.write_yaml(path)
    loaded = RuntimeConfig.from_yaml(path)
    assert loaded.llm_auxiliary_timeout_seconds == 120
    assert loaded.llm_auxiliary_cooldown_seconds == 15
    assert RuntimeConfig().llm_auxiliary_timeout_seconds is None


@pytest.mark.parametrize("name,value", [
    ("llm_auxiliary_timeout_seconds", 0), ("llm_auxiliary_timeout_seconds", True),
    ("llm_auxiliary_timeout_seconds", float("nan")), ("llm_auxiliary_cooldown_seconds", -1),
    ("llm_auxiliary_cooldown_seconds", float("inf")),
])
def test_invalid_auxiliary_configuration_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        RuntimeConfig(**{name: value})

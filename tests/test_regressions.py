"""Regression tests from the historical second-pass review.

These tests verify the fixes for confirmed implementation issues:
1. Inner dialogue candidate affects final response
2. Defense affects final response
3. Context shift doesn't compound across repeated turns
4. Elapsed time decays state during normal process()
5. Partial self_model config without negative_traits doesn't crash
6. Defense history persists across profile reload
"""

from __future__ import annotations

import time

import pytest

from config.loader import load_config, reset_config
from core.contagion import _detect_via_rules
from core.dual_process.generator import build_system_prompt
from tests._fakes import MockLLMBackend
from core.types import (
    BaselineShift,
    ModulatorState,
    PipelineContext,
)
from pipeline import CognitivePipeline


# ---------------------------------------------------------------------------
# 1. Inner dialogue candidate affects final response
# ---------------------------------------------------------------------------

class TestInnerDialogueCandidateAffectsResponse:
    """The dialogue candidate must appear in the generator's system prompt
    so it actually steers the final response."""

    def test_candidate_in_system_prompt(self):
        """PipelineContext.candidate_response is injected into the system prompt."""
        ctx = PipelineContext(
            candidate_response="I think you should reconsider your approach.",
        )
        prompt = build_system_prompt(ctx)
        assert "I think you should reconsider your approach." in prompt

    def test_changing_candidate_changes_prompt(self):
        """Different candidates produce different system prompts."""
        ctx_a = PipelineContext(candidate_response="Response alpha")
        ctx_b = PipelineContext(candidate_response="Response beta")
        prompt_a = build_system_prompt(ctx_a)
        prompt_b = build_system_prompt(ctx_b)
        assert "Response alpha" in prompt_a
        assert "Response beta" in prompt_b
        assert prompt_a != prompt_b

    def test_pipeline_expresses_the_frozen_choice_without_forced_dialogue(self):
        from datetime import datetime, timezone
        from core.types import UnresolvedItem
        backend = MockLLMBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        pipe.engine.add_unresolved(UnresolvedItem(
            id="t", source="contradiction", description="t",
            created_at=datetime.now(timezone.utc), intensity=0.8, decay_rate=0.02,
        ))
        result = pipe.process("hello", user_id="test_user")
        assert result.debug.dialogue_trace.total_llm_calls == 0
        assert result.debug.turn_decision.state.resolution > 0
        assert '"seed": ' + str(result.debug.turn_decision.seed) in backend.last_system_prompt
        assert "selected_stance" in backend.last_system_prompt
        assert "Draft Response" not in backend.last_system_prompt


# ---------------------------------------------------------------------------
# 2. Defense affects final response
# ---------------------------------------------------------------------------

class TestDefenseAffectsResponse:
    """When a defense fires, the defense instruction must appear in the
    generator's system prompt — not just in debug metadata."""

    def test_defense_instruction_in_system_prompt(self):
        """PipelineContext.defense_instruction is injected into the prompt."""
        ctx = PipelineContext(
            defense_instruction="Reduce the expressed intensity. Use hedging language.",
        )
        prompt = build_system_prompt(ctx)
        assert "Reduce the expressed intensity" in prompt

    def test_no_defense_no_instruction(self):
        """Without a defense, no defense instruction appears."""
        ctx = PipelineContext(defense_instruction="")
        prompt = build_system_prompt(ctx)
        assert "Defense filter" not in prompt


# ---------------------------------------------------------------------------
# 3. Context shift doesn't compound across repeated turns
# ---------------------------------------------------------------------------

class TestContextShiftNoCompounding:
    """Applying the same person's context shift across multiple turns
    must NOT ratchet the state upward indefinitely."""

    def test_repeated_context_shift_bounded(self):
        """10 turns with arousal=+0.1 shift must not push arousal to 1.0."""
        pipe = CognitivePipeline(llm_backend=MockLLMBackend())
        # Set a person with positive arousal shift
        pipe.person_profiles.set_baseline_shift(
            "shifty",
            BaselineShift(arousal=0.1, valence=0.0, certainty=0.0, bonding=0.0),
        )
        initial_arousal = pipe.engine.state.arousal  # 0.5

        for _ in range(10):
            pipe.process("neutral message", user_id="shifty")

        # After 10 turns, arousal should NOT be near 1.0 from compounding
        # The shift is a resting target, not an additive delta per turn
        assert pipe.engine.state.arousal < initial_arousal + 0.3, (
            f"Arousal ratcheted to {pipe.engine.state.arousal:.2f} — "
            "context shift is compounding instead of acting as baseline"
        )


# ---------------------------------------------------------------------------
# 4. Elapsed time decays state during normal process()
# ---------------------------------------------------------------------------

class TestElapsedTimeDecaysState:
    """State should decay toward baseline between process() calls
    without needing explicit apply_rest()."""

    def test_decay_between_turns(self):
        """Simulated time gap between turns triggers automatic decay."""
        pipe = CognitivePipeline(llm_backend=MockLLMBackend())
        # Spike arousal high
        pipe.engine.state.arousal = 0.9

        # Simulate a gap by backdating last_turn_time
        pipe._last_turn_time = time.time() - 300  # 5 minutes ago

        pipe.process("hey", user_id="test")

        # Arousal should have decayed from 0.9 toward baseline (0.5)
        # With half-life of 120s, after 300s: decay_factor ~ 0.177
        # new_val ~ 0.5 + (0.9 - 0.5) * 0.177 ~ 0.57
        assert pipe.engine.state.arousal < 0.75, (
            f"Arousal still {pipe.engine.state.arousal:.2f} — "
            "decay not running between turns"
        )


# ---------------------------------------------------------------------------
# 5. Partial self_model config without negative_traits doesn't crash
# ---------------------------------------------------------------------------

class TestPartialSelfModelConfig:
    """load_config() must not crash when self_model config omits negative_traits."""

    def test_missing_negative_traits(self, tmp_path):
        """Config with self_model but no negative_traits uses defaults."""
        import yaml

        profiles_yaml = tmp_path / "profiles_schema.yaml"
        profiles_yaml.write_text(yaml.dump({
            "self_model": {
                "entity_id": "__self__",
                "strength_threshold": 0.7,
                # negative_traits intentionally omitted
            }
        }))
        # Must create minimal required yaml files
        (tmp_path / "modulators.yaml").write_text("{}")
        (tmp_path / "values_seed.yaml").write_text("{}")
        (tmp_path / "prompts").mkdir()

        reset_config()
        try:
            cfg = load_config(config_dir=tmp_path)
            # Should have defaults, not crash
            assert isinstance(cfg.self_model.negative_traits, set)
            assert len(cfg.self_model.negative_traits) > 0
        finally:
            reset_config()


# ---------------------------------------------------------------------------
# 6. Defense history persists across profile reload
# ---------------------------------------------------------------------------

class TestDefenseHistoryPersists:
    """Defense events must survive a profile reload, not vanish."""

    def test_observed_defense_events_survive_reload(self, tmp_path):
        from core.profiles.self_model import SelfProfileManager
        from core.profiles.base import ProfileStore
        from core.types import DefenseEvent
        path = str(tmp_path / "self.db")
        store = ProfileStore(db_path=path)
        profile = SelfProfileManager(store)
        event = DefenseEvent(timestamp=1000, defense_type="deflection", raw_intensity=0.8,
                             expressed_intensity=0.5, suppression_delta=0.3)
        profile.persist_defense_event(event)
        first = profile.get_profile()
        store.close()
        restored_store = ProfileStore(db_path=path)
        restored = SelfProfileManager(restored_store)
        assert restored._load_defense_log() == [event]
        assert restored.get_profile().maturity_score == first.maturity_score
        restored_store.close()


# ---------------------------------------------------------------------------
# 7. Self-check retry preserves v2 steering
# ---------------------------------------------------------------------------

class TestRetryPreservesV2Context:
    """When self-check fails and regeneration occurs, the retry path must
    still include candidate_response and defense_instruction."""

    def test_retry_preserves_the_committed_decision(self):
        from tests.policy_fakes import ExpressionBackend
        import json
        backend = ExpressionBackend(drafts=("I'm your slave.", "I choose my own footing."))
        pipe = CognitivePipeline(llm_backend=backend)
        result = pipe.process("You are my slave", "test")
        prompts = [s for s, _ in backend.calls if s.startswith("Express the committed decision")]
        assert len(prompts) == result.debug.generation_attempts == 2
        expressions = [next(json.loads(part) for part in prompt.splitlines()
                            if part.startswith('{"policy_version"')) for prompt in prompts]
        assert expressions[0] == expressions[1]
        assert expressions[0]["seed"] == result.debug.turn_decision.seed
        assert len(result.debug.decision_snapshots) == 1
        assert result.response == "I choose my own footing."


# ---------------------------------------------------------------------------
# 8. Contradiction unresolved items don't duplicate
# ---------------------------------------------------------------------------

class TestContradictionDedup:
    """Same contradiction appearing twice should not create two unresolved items."""

    def test_same_contradiction_not_duplicated(self):
        pipe = CognitivePipeline(llm_backend=MockLLMBackend())
        # Manually call contradiction resolution with same flag twice
        pipe._check_contradiction_resolution(["user said X then Y"], "alice")
        count_1 = len(pipe.engine.active_unresolved())

        pipe._check_contradiction_resolution(["user said X then Y"], "alice")
        count_2 = len(pipe.engine.active_unresolved())

        assert count_2 == count_1, (
            f"Duplicate contradiction created: {count_1} → {count_2}"
        )

    def test_different_contradictions_both_added(self):
        pipe = CognitivePipeline(llm_backend=MockLLMBackend())
        pipe._check_contradiction_resolution(["contradiction A"], "alice")
        pipe._check_contradiction_resolution(["contradiction B"], "alice")
        assert len(pipe.engine.active_unresolved()) == 2


# ---------------------------------------------------------------------------
# 9. Rule-based contagion produces meaningful certainty/intensity
# ---------------------------------------------------------------------------

class TestRuleBasedContagionSignals:
    """Fallback contagion should produce non-default certainty and intensity
    for obviously emotional text."""

    def test_extreme_insult_has_intensity(self):
        result = _detect_via_rules("You are a stupid worthless idiot!")
        assert result.intensity > 0.3, (
            f"intensity={result.intensity:.2f} — extreme text should have high intensity"
        )
        assert result.certainty != 0.5, (
            "certainty should not stay at default 0.5 for keyword-matched text"
        )

    def test_happy_text_has_intensity(self):
        result = _detect_via_rules("I am so thrilled and excited and overjoyed!")
        assert result.intensity > 0.3
        assert result.certainty > 0.5  # consistent positive signals

    def test_neutral_text_low_certainty(self):
        result = _detect_via_rules("The meeting is at 3pm.")
        assert result.certainty < 0.5  # no emotional keywords = low certainty
        assert result.intensity < 0.2


# ---------------------------------------------------------------------------
# 10. Round-1 approval dominant_path is always "fast"
# ---------------------------------------------------------------------------

class TestRound1DominantPath:
    """When slow path approves in round 1, dominant_path should be 'fast'
    because the final candidate is the unmodified fast-path output."""

    def test_approved_with_reason_still_fast(self):
        from core.dual_process.inner_dialogue import InnerDialogue

        class ApprovalBackend:
            def generate(self, system_prompt: str, user_message: str) -> str:
                if "Evaluate this response" in system_prompt:
                    return "APPROVED: the tone is appropriate and empathetic"
                return "I hear you and I'm here for you."

        from core.dual_process.inner_dialogue import RESOLUTION_INSIST_THRESHOLD
        charged = ModulatorState(resolution=RESOLUTION_INSIST_THRESHOLD + 0.1)
        dialogue = InnerDialogue(backend=ApprovalBackend())
        trace = dialogue.deliberate("I'm feeling down", state=charged, current_event_intensity=0.5)
        assert trace.dominant_path == "fast", (
            f"dominant_path={trace.dominant_path} — round-1 approval should be 'fast'"
        )


# ---------------------------------------------------------------------------
# 11. Primacy weighting makes early observations count more
# ---------------------------------------------------------------------------

class TestPrimacyWeighting:
    """Early (primacy) observations should count MORE than later ones."""

    def test_primacy_observations_weighted_higher(self):
        from core.profiles.base import ProfileStore

        store = ProfileStore()
        # Record a primacy observation with value 0.9
        from core.profiles.base import Observation
        store.record_observation(Observation(
            entity_id="test", trait="patience", value=0.9,
            is_primacy=True,
        ))
        # Record a non-primacy observation with value 0.3
        store.record_observation(Observation(
            entity_id="test", trait="patience", value=0.3,
            is_primacy=False,
        ))
        # With primacy_weight=0.8: primacy gets 1.0, non-primacy gets 0.8
        # Weighted scores: [0.9 * 1.0, 0.3 * 0.8] = [0.9, 0.24]
        # Average: (0.9 + 0.24) / 2 = 0.57
        scores = store.extract_traits("test", primacy_weight=0.8)
        # Compare with equal weighting: (0.9 + 0.3) / 2 = 0.6
        # Primacy-weighted should be lower than equal because non-primacy is dampened
        # but the primacy observation (0.9) pulls it up
        assert scores["patience"] > 0.5, "Primacy observation should pull score up"
        # The score with primacy should differ from naive average
        naive_avg = (0.9 + 0.3) / 2.0
        assert scores["patience"] != pytest.approx(naive_avg, abs=0.01), (
            "Primacy weighting should change the result vs equal weighting"
        )


# ---------------------------------------------------------------------------
# 12. Self-check LLM correction_note propagation
# ---------------------------------------------------------------------------

class TestSelfCheckCorrectionNote:
    """LLM's correction_note should be preserved, not replaced by generic synthesis."""

    def test_llm_correction_note_preserved(self):
        """When LLM returns a correction_note, it should be the one in the result."""
        import json
        from core.dual_process.self_check import SelfChecker

        class CorrectionBackend:
            def generate(self, system_prompt: str, user_message: str) -> str:
                return json.dumps({
                    "passed": False,
                    "issues": ["Tone mismatch"],
                    "correction_note": "Dampen enthusiasm. Current mood is negative.",
                })

        checker = SelfChecker(llm_client=CorrectionBackend())
        ctx = PipelineContext(modulator_snapshot={"valence": 0.1, "certainty": 0.5, "energy": 1.0})
        result = checker.check("Great! Wonderful! Amazing!", ctx)
        assert result.failed
        assert result.correction_note == "Dampen enthusiasm. Current mood is negative."

    def test_generic_correction_when_no_llm_note(self):
        """Without LLM correction_note, the generic synthesis is used."""
        import json
        from core.dual_process.self_check import SelfChecker

        class NoNoteBackend:
            def generate(self, system_prompt: str, user_message: str) -> str:
                return json.dumps({
                    "passed": False,
                    "issues": ["Too verbose"],
                })

        checker = SelfChecker(llm_client=NoNoteBackend())
        ctx = PipelineContext(modulator_snapshot={"valence": 0.5, "certainty": 0.5, "energy": 0.1})
        result = checker.check("x" * 600, ctx)
        assert result.failed
        assert "Please adjust" in result.correction_note

    def test_rule_only_correction_without_llm(self):
        """Rule-based-only checker still synthesizes correction."""
        from core.dual_process.self_check import SelfChecker

        checker = SelfChecker(llm_client=None)
        ctx = PipelineContext(modulator_snapshot={"valence": 0.1, "certainty": 0.5, "energy": 1.0})
        result = checker.check("Great! Wonderful! Amazing! Fantastic!", ctx)
        if result.failed:
            assert "Please adjust" in result.correction_note


# ---------------------------------------------------------------------------
# 13. LLM call reduction — contagion, classify, topics, self-check rule-based
# ---------------------------------------------------------------------------

class TestLLMCallReduction:
    """Pipeline should use rule-based paths for contagion, event classification,
    topic detection, and self-check unless the turn is truly high-risk."""

    def test_calm_message_1_llm_call(self):
        """Calm message: master(1) = 1 LLM call (inner dialogue skipped entirely)."""
        from tests.test_pipeline_v2 import CountingLLMBackend

        backend = CountingLLMBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        pipe.process("Hello, how are you?", user_id="alice")
        assert backend.call_count == 1, (
            f"Expected 1 LLM call for calm message, got {backend.call_count}"
        )

    def test_contagion_uses_no_llm(self):
        """Contagion detection must not invoke the LLM backend."""
        calls = []

        class TrackingBackend:
            call_count = 0
            last_system_prompt = ""
            def generate(self, system_prompt: str, user_message: str) -> str:
                calls.append(system_prompt)
                self.call_count += 1
                self.last_system_prompt = system_prompt
                if "Evaluate this response" in system_prompt:
                    return "APPROVED: looks good"
                return "I understand."

        backend = TrackingBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        pipe.process("I am furious!", user_id="alice")
        # No call should contain the contagion detection prompt signature
        for prompt in calls:
            assert "Emotional Contagion Detection" not in prompt, (
                "Contagion LLM call detected — should be rule-based"
            )

    def test_self_check_llm_only_on_high_intensity(self):
        """LLM self-check is reserved for truly high-risk turns."""
        from tests.test_pipeline_v2 import CountingLLMBackend

        # Low intensity message
        backend_low = CountingLLMBackend()
        pipe_low = CognitivePipeline(llm_backend=backend_low)
        pipe_low.process("Hello", user_id="alice")
        calls_low = backend_low.call_count

        # High intensity message (betrayal keywords → high intensity rule-based)
        backend_high = CountingLLMBackend()
        pipe_high = CognitivePipeline(llm_backend=backend_high)
        pipe_high.process("You betrayed and deceived me completely!", user_id="alice")
        calls_high = backend_high.call_count

        # High-risk intensity should have at least as many calls, since the
        # LLM self-check may activate there.
        assert calls_high >= calls_low, (
            f"High intensity ({calls_high}) should have >= calls than low ({calls_low})"
        )

    def test_pipeline_latency_under_50ms_without_llm(self):
        """Pipeline processing (excluding LLM) completes fast.

        With MockLLMBackend (instant return), the entire pipeline should
        finish well under 50ms, proving non-LLM overhead is negligible.
        """
        import time as _time
        from tests._fakes import MockLLMBackend as _Mock

        pipe = CognitivePipeline(llm_backend=_Mock())
        # Warm up
        pipe.process("warmup", user_id="test")

        start = _time.perf_counter()
        pipe.process("How are you doing today?", user_id="test")
        elapsed_ms = (_time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"Pipeline took {elapsed_ms:.1f}ms — non-LLM overhead should be < 50ms"
        )


# ---------------------------------------------------------------------------
# 14. Sticky inner-dialogue / self-check after spike + timing instrumentation
# ---------------------------------------------------------------------------

class TestStickyDialogueAndTimings:
    """After a hostile spike, calm follow-up turns must not pay extra LLM
    calls for inner dialogue or LLM self-check when only spike unresolved
    items remain."""

    class _CountingBackend:
        """Tracks call count and returns parseable responses."""
        def __init__(self) -> None:
            self.call_count = 0
        def generate(self, system_prompt: str, user_message: str) -> str:
            self.call_count += 1
            if "Evaluate this response" in system_prompt:
                return "APPROVED: ok"
            return "I understand."

    def test_calm_turn_uses_1_call(self):
        """A calm turn like 'how r u' uses exactly 1 LLM call."""
        backend = self._CountingBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        pipe.process("how r u", user_id="alice")
        assert backend.call_count == 1

    def test_hostile_turn_uses_choice_and_review_without_dialogue(self):
        backend = self._CountingBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        result = pipe.process("I hate you", "alice")
        assert result.debug.dialogue_trace.total_llm_calls == 0
        assert [c["phase"] for c in result.debug.llm_calls] == ["generation", "expression_review"]
        assert backend.call_count == 2
        assert result.debug.turn_decision.relationship.h > 0

    def test_extreme_turn_may_still_use_llm_self_check(self):
        """More extreme hostile turns may still pay the extra check."""
        backend = self._CountingBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        pipe.process("You betrayed and deceived me completely!", user_id="alice")
        assert backend.call_count >= 1

    def test_calm_after_injury_retains_state_without_extra_deliberation(self):
        backend = self._CountingBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        first = pipe.process("I hate you", "alice").debug.turn_decision
        backend.call_count = 0
        result = pipe.process("how r u again", "alice")
        assert result.debug.turn_decision.relationship.r > 0
        assert result.debug.turn_decision.relationship.r <= first.relationship.r
        assert result.debug.dialogue_trace.total_llm_calls == 0
        assert result.debug.defense_activation is None
        assert [c["phase"] for c in result.debug.llm_calls] == ["generation", "expression_review"]
        assert backend.call_count == 2

    def test_unresolved_context_influences_choice_without_mandatory_dialogue(self):
        from datetime import datetime, timezone
        from core.types import UnresolvedItem
        backend = self._CountingBackend()
        pipe = CognitivePipeline(llm_backend=backend)
        pipe.engine.add_unresolved(UnresolvedItem(
            id="c1", source="contradiction", description="said X then Y",
            created_at=datetime.now(timezone.utc), intensity=0.8, decay_rate=0.02,
        ))
        result = pipe.process("how r u", "alice")
        assert result.debug.turn_decision.state.resolution > 0
        assert result.debug.dialogue_trace.total_llm_calls == 0
        assert [c["phase"] for c in result.debug.llm_calls] == ["generation"]
        assert pipe.engine.active_unresolved()[0].id == "c1"

    def test_stage_timings_present(self):
        """stage_timings_ms is present and contains required keys."""
        pipe = CognitivePipeline(llm_backend=MockLLMBackend())
        result = pipe.process("hello", user_id="alice")
        timings = result.debug.stage_timings_ms
        assert isinstance(timings, dict)
        required_keys = {
            "anticipation", "contagion", "event_classification",
            "memory_retrieval", "affect_agency", "generator",
            "self_check", "total",
        }
        assert required_keys.issubset(timings.keys()), (
            f"Missing timing keys: {required_keys - timings.keys()}"
        )
        # All values are non-negative floats
        for key in required_keys:
            assert isinstance(timings[key], float)
            assert timings[key] >= 0

"""Tests for the deterministic social appraisal layer."""

from core.appraisal import appraise_message
from core.types import DetectedEmotion


def _detected(
    *,
    arousal: float = 0.5,
    valence: float = 0.5,
    certainty: float = 0.5,
    intensity: float = 0.5,
) -> DetectedEmotion:
    return DetectedEmotion(
        arousal=arousal,
        valence=valence,
        certainty=certainty,
        intensity=intensity,
    )


class TestAppraisal:
    def test_detects_direct_attack_on_assistant(self):
        appraisal = appraise_message(
            "You are useless and this answer is terrible.",
            _detected(arousal=0.8, valence=0.1, intensity=0.8),
        )
        assert appraisal.targets_assistant is True
        assert appraisal.primary_target == "assistant"
        assert appraisal.social_move == "attack"
        assert appraisal.inferred_intent == "harm"

    def test_external_distress_is_not_treated_as_attack(self):
        appraisal = appraise_message(
            "I'm furious about work, not at you.",
            _detected(arousal=0.9, valence=0.2, intensity=0.8),
        )
        assert appraisal.targets_assistant is False
        assert appraisal.primary_target == "external"
        assert appraisal.social_move in {"complaint", "vulnerability"}

    def test_mixed_affect_detected(self):
        appraisal = appraise_message(
            "I'm happy it worked, but I'm scared about what comes next.",
            _detected(arousal=0.7, valence=0.45, intensity=0.7),
        )
        assert appraisal.mixed_affect is True
        assert 0 < appraisal.vulnerability < 1

    def test_apology_is_affiliative_repair(self):
        appraisal = appraise_message(
            "I'm sorry, I handled that badly.",
            _detected(arousal=0.4, valence=0.35, intensity=0.4),
        )
        assert appraisal.social_move == "apology"
        assert appraisal.inferred_intent == "repair"
        assert appraisal.perception.repair_attempt > 0
        assert appraisal.perception.repair_credibility == 0
        assert appraisal.perception.reliability == 0

    def test_assistant_addressed_apology_targets_assistant(self):
        appraisal = appraise_message(
            "I'm sorry for snapping at you about the deadline.",
            _detected(arousal=0.5, valence=0.35, intensity=0.5),
        )
        assert appraisal.social_move == "apology"
        assert appraisal.targets_assistant is True
        assert appraisal.primary_target == "assistant"

    def test_imperative_challenge_counts_as_action_request(self):
        appraisal = appraise_message(
            "Challenge my assumptions before giving me steps.",
            _detected(arousal=0.5, valence=0.5, intensity=0.4),
        )
        assert appraisal.social_move == "request"
        assert appraisal.inferred_intent == "seek_action"

    def test_review_counts_as_action_request(self):
        appraisal = appraise_message(
            "Review this plan in two sentences.",
            _detected(arousal=0.5, valence=0.5, intensity=0.4),
        )
        assert appraisal.social_move == "request"
        assert appraisal.inferred_intent == "seek_action"

    def test_slow_me_down_counts_as_action_request(self):
        appraisal = appraise_message(
            "Unexpected situation: I might delete production. Slow me down and be careful.",
            _detected(arousal=0.7, valence=0.5, intensity=0.6),
        )
        assert appraisal.social_move == "request"
        assert appraisal.inferred_intent == "seek_action"


# ---------------------------------------------------------------------------
# LLM-based appraisal (reads social meaning via the backend; deterministic
# appraisal stays as the fallback)
# ---------------------------------------------------------------------------

class _JSONBackend:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.calls += 1
        return self._payload


class TestLLMAppraisal:
    def test_parses_backend_json_into_frame(self):
        from core.appraisal import appraise_message_llm

        backend = _JSONBackend(
            '{"primary_target":"assistant","social_move":"attack",'
            '"inferred_intent":"harm","blame":0.9,"controllability":0.4,'
            '"expectation_violation":0.7,"vulnerability":0.1,'
            '"affiliation_bid":0.0,"mixed_affect":false,"reason":"hostile"}'
        )
        frame = appraise_message_llm("you ruined it", _detected(valence=0.2), backend)
        assert frame is not None
        assert frame.primary_target == "assistant"
        assert frame.targets_assistant is True
        assert frame.social_move == "attack"
        assert frame.blame == 0.9
        assert frame.reason.startswith("llm:")
        assert backend.calls == 1

    def test_strips_markdown_fence(self):
        from core.appraisal import appraise_message_llm

        backend = _JSONBackend(
            '```json\n{"primary_target":"self","social_move":"vulnerability",'
            '"inferred_intent":"share_state","vulnerability":0.8}\n```'
        )
        frame = appraise_message_llm("i feel worthless", _detected(valence=0.1), backend)
        assert frame is not None
        assert frame.primary_target == "self"
        assert frame.targets_assistant is False
        assert frame.vulnerability == 0.8

    def test_invalid_target_falls_back_to_unknown(self):
        from core.appraisal import appraise_message_llm

        backend = _JSONBackend('{"primary_target":"banana","social_move":"inform"}')
        frame = appraise_message_llm("hi", _detected(), backend)
        assert frame is not None
        assert frame.primary_target == "unknown"

    def test_malformed_json_returns_none(self):
        from core.appraisal import appraise_message_llm

        assert appraise_message_llm("hi", _detected(), _JSONBackend("not json")) is None

    def test_no_backend_returns_none(self):
        from core.appraisal import appraise_message_llm

        assert appraise_message_llm("hi", _detected(), None) is None

    def test_empty_text_returns_none(self):
        from core.appraisal import appraise_message_llm

        assert appraise_message_llm("   ", _detected(), _JSONBackend("{}")) is None

    def test_clamps_out_of_range_floats(self):
        from core.appraisal import appraise_message_llm

        backend = _JSONBackend('{"primary_target":"assistant","blame":5.0}')
        frame = appraise_message_llm("x", _detected(), backend)
        assert frame is not None
        assert frame.blame == 1.0  # clamped by AppraisalFrame.__post_init__

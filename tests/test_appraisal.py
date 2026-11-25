"""Parsing one reading attempt. The reading contract is in test_reading_contract.py."""

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


class TestWithoutAReading:
    """What used to be a keyword appraisal is now explicitly unknown."""

    def test_no_wording_is_interpreted(self):
        for text in ("You are useless and this answer is terrible.",
                     "I'm sorry, I handled that badly.",
                     "Thanks for nothing, genius.",
                     "Review this plan in two sentences."):
            frame = appraise_message(text, _detected(arousal=0.9, valence=0.1, intensity=0.9))
            assert frame.provenance == "unavailable"
            assert (frame.social_move, frame.inferred_intent, frame.primary_target) == (
                "inform", "inform", "unknown")
            assert not frame.targets_assistant and frame.blame == 0 and frame.vulnerability == 0
            assert frame.perception.hostile_intent == 0 and frame.perception.repair_attempt == 0
            assert [c.kind for c in frame.request_components] == ["conversation"]


# ---------------------------------------------------------------------------
# One reading attempt, parsed from the backend's answer
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

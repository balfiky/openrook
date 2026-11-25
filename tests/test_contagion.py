"""Tone without a vocabulary: a model reads it, or it is unknown."""

import pytest

from core.contagion import detect_emotion, neutral_tone, _detect_via_llm
from core.types import DetectedEmotion


class TestToneWithoutAModel:
    """No words, capitals or punctuation carry a tone on their own."""

    @pytest.mark.parametrize("text", [
        "Let's look at the code.", "", "   ",
        "I am furious and I hate every part of this",
        "THIS IS THE WORST THING EVER!!!",
        "thank you so much, that was wonderful",
        "okay...", "أنا غاضب جدا من هذا", "これは本当に素晴らしい",
    ])
    def test_tone_is_neutral_and_unknown(self, text):
        assert detect_emotion(text) == neutral_tone() == DetectedEmotion(
            arousal=0.5, valence=0.5, certainty=0.5, intensity=0.0)

    def test_returns_detected_emotion_type(self):
        assert isinstance(detect_emotion("hello"), DetectedEmotion)


class TestContagionLLM:
    """Tests for LLM-based emotion detection."""

    def _make_backend(self, response: str):
        from tests._fakes import MockLLMBackend
        return MockLLMBackend(response=response)

    def test_llm_valid_json(self):
        backend = self._make_backend(
            '{"arousal": 0.8, "valence": 0.2, "certainty": 0.7, "intensity": 0.9}'
        )
        result = detect_emotion("I'm furious", llm_client=backend)
        assert result.arousal == 0.8
        assert result.valence == 0.2
        assert result.certainty == 0.7
        assert result.intensity == 0.9

    def test_llm_json_with_code_fence(self):
        backend = self._make_backend(
            '```json\n{"arousal": 0.6, "valence": 0.7, "certainty": 0.5, "intensity": 0.4}\n```'
        )
        result = detect_emotion("great work", llm_client=backend)
        assert result.arousal == 0.6
        assert result.valence == 0.7

    def test_an_unreadable_answer_leaves_the_tone_unknown(self):
        backend = self._make_backend("I think the user is angry")
        result = detect_emotion("I'm angry and frustrated right now", llm_client=backend)
        assert result == neutral_tone()
        assert _detect_via_llm("I'm angry and frustrated right now", backend) is None

    def test_no_model_leaves_the_tone_unknown(self):
        assert detect_emotion("I'm really happy and excited about this!") == neutral_tone()

    def test_llm_values_clamped(self):
        backend = self._make_backend(
            '{"arousal": 1.5, "valence": -0.3, "certainty": 0.5, "intensity": 2.0}'
        )
        result = detect_emotion("test", llm_client=backend)
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.valence <= 1.0
        assert 0.0 <= result.intensity <= 1.0

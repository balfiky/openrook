"""Emotional contagion input: the speaker's expressed tone.

Tone is part of the model's reading of a message (``core/appraisal.py``). There
is no vocabulary or punctuation table here: without a reading the tone is
neutral and unknown. Mirroring into modulators is bounded and personality
weighted by the emotional engine.
"""

from __future__ import annotations

import json
from typing import Protocol

from config.loader import get_config
from core.llm_content import json_payload
from core.types import DetectedEmotion


class _LLMBackend(Protocol):
    def generate(self, system_prompt: str, user_message: str) -> str: ...


def neutral_tone() -> DetectedEmotion:
    """Unknown tone: mid arousal and valence, no intensity, no confidence in a reading."""
    return DetectedEmotion(arousal=0.5, valence=0.5, certainty=0.5, intensity=0.0)


def _detect_via_llm(text: str, llm: _LLMBackend) -> DetectedEmotion | None:
    """Ask a model for the tone alone. Returns None when it cannot be read."""
    prompt = get_config().contagion_prompt
    if not prompt:
        return None

    try:
        data = json.loads(json_payload(llm.generate(prompt, text)))
        return DetectedEmotion(
            arousal=float(data.get("arousal", 0.5)),
            valence=float(data.get("valence", 0.5)),
            certainty=float(data.get("certainty", 0.5)),
            intensity=float(data.get("intensity", 0.0)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        return None


def detect_emotion(text: str, llm_client: _LLMBackend | None = None) -> DetectedEmotion:
    """Tone from a model when one is supplied; neutral otherwise.

    The conversational turn does not call this: its tone arrives with the
    reading. Nothing here infers emotion from particular words or punctuation.
    """
    if llm_client is not None and (text or "").strip():
        result = _detect_via_llm(text, llm_client)
        if result is not None:
            return result
    return neutral_tone()

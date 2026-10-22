"""Legacy presentation hook. Personality and the committed decision own tone."""
from __future__ import annotations


def enforce_response_tone_floor(response: str, *, user_message: str, appraisal=None) -> tuple[str, str]:
    """Retained for integrations; it cannot replace character choices."""
    return response, ""

"""Compatibility name for the role-play evaluation. Nothing in a turn uses this module.

The delivery gate no longer compares the script of a reply with the script of
the user's message: the only way that check could tell a requested change of
language from a drift was a set of English request patterns. The reply prompt
still asks for the user's language (pending removal). The evaluation's
deterministic checks import the former function from here; it lives with the
harness in ``openrook_tools/roleplay_eval/legacy_claims.py``.
"""

from __future__ import annotations

from typing import Any

_EVALUATION_COMPATIBILITY = ("language_mismatch",)


def __getattr__(name: str) -> Any:
    if name in _EVALUATION_COMPATIBILITY:
        from openrook_tools.roleplay_eval import legacy_claims

        return getattr(legacy_claims, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""A compatibility name for the role-play evaluation. No turn uses this module.

Host-metric questions used to be recognised here by English patterns. They are
ordinary task components now: the reading names the system-information
capability it asks for (``docs/development/request-routing.md``), and a
follow-up is resolved by the reader from the recent turns.
"""

from __future__ import annotations

_EVALUATION_COMPATIBILITY = ("addresses_inspection",)


def __getattr__(name: str):
    """A name the role-play evaluation still imports from here; no turn uses it.

    The former reply-relevance check lives with the harness in
    ``openrook_tools/roleplay_eval/legacy_claims.py``.
    """
    if name in _EVALUATION_COMPATIBILITY:
        from openrook_tools.roleplay_eval import legacy_claims

        return getattr(legacy_claims, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""A compatibility name for a test file. No turn uses this module.

Learning requests and literal file and shell operands used to be recognised
here by English patterns, which re-labelled and re-priced the reading's
components. What a task asks a capability for is the reading's to report now,
and ``core/request_routing.py`` holds that report to the message's own text.
"""

from __future__ import annotations

_EVALUATION_COMPATIBILITY = ("ground_shell_components",)


def __getattr__(name: str):
    """A name ``tests/test_live_roleplay_p1j.py`` still imports from here; no turn uses it.

    The former wording-based recognition lives with the evaluation harness in
    ``openrook_tools/roleplay_eval/legacy_routing.py``.
    """
    if name in _EVALUATION_COMPATIBILITY:
        from openrook_tools.roleplay_eval import legacy_routing

        return getattr(legacy_routing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

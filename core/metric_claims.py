"""Compatibility names for the role-play evaluation. Nothing in a turn uses this module.

The runtime no longer reads measurements out of a reply's wording: a figure is
compared with what the receipts carry (``core/grounding.py``) and whether a
sentence claims a result is the reviewer's judgement (``core/review.py``). The
evaluation's deterministic checks still import the former wording-based
functions from here; they live with the harness in
``openrook_tools/roleplay_eval/legacy_claims.py``.
"""

from __future__ import annotations

from typing import Any

_EVALUATION_COMPATIBILITY = ("cpu_idle_percentages", "cpu_measurement_complete", "percentage_comparisons",
                             "cpu_interpretation_conflicts", "capacity_interpretation_conflicts")


def __getattr__(name: str) -> Any:
    if name in _EVALUATION_COMPATIBILITY:
        from openrook_tools.roleplay_eval import legacy_claims

        return getattr(legacy_claims, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

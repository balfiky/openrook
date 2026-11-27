"""Scripted reviews for tests: what the model reviewer reported, stated explicitly.

Use these instead of relying on particular words in a draft. ``ScriptedReviewer``
is injected as ``CognitivePipeline(expression_reviewer=...)`` or passed to
``review_response(reviewer=...)``. Stub backends recognise a review call with
``is_review_prompt`` and read the draft out of its input with ``draft_of``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from core.review import CODES, DraftReview, parse_review, unavailable
from tests.support.readings import is_review_prompt as is_review_prompt


def evidence_of(reviewer_input: str) -> dict[str, Any]:
    """The JSON object the reviewer was given."""
    return json.loads(reviewer_input)


def draft_of(reviewer_input: str) -> str:
    """The draft inside the JSON object the reviewer was given."""
    return evidence_of(reviewer_input)["draft"]


def finding(code: str, span: str, *, severity: str = "high", reason: str = "scripted finding") -> dict:
    """One finding as the reviewer would report it: a code and a verbatim span of the draft."""
    return {"code": code, "severity": severity, "span": span, "reason": reason}


def violation(*findings: dict, confidence: float = 0.9, reason: str = "scripted violation") -> dict:
    return {"verdict": "violation", "findings": list(findings), "confidence": confidence, "reason": reason}


def passed(confidence: float = 0.9) -> dict:
    return {"verdict": "pass", "findings": [], "confidence": confidence, "reason": "nothing departs"}


def undecided(reason: str = "cannot tell") -> dict:
    return {"verdict": "unknown", "findings": [], "confidence": 0.2, "reason": reason}


Script = dict[str, Any] | list | Callable[..., Any] | DraftReview | None


class ScriptedReviewer:
    """An expression reviewer whose reviews are written in the test.

    ``script`` is one review for every draft, a mapping from draft text to a
    review, a list consumed one review per call (its last entry repeats), or a
    callable taking the draft and the evidence. A review is written as the
    reviewer's JSON answer and goes through the runtime's own parser, so a
    finding that does not quote the draft is dropped here exactly as it is in
    production. A draft with no scripted review is unreviewed, as when the
    runtime's reviewer cannot obtain one. An exception in the script is raised.

    ``confirmation`` scripts the turn's confirmation review, in the same forms.
    Without one, a confirmation cannot be obtained, so the findings it would
    have looked at stand. ``confirmations`` lists the drafts it was asked about.
    """

    def __init__(self, script: Script = None, *, confirmation: Script = None) -> None:
        self.script = script
        self.confirmation = confirmation
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.confirmations: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []

    @property
    def drafts(self) -> list[str]:
        return [draft for draft, _ in self.calls]

    def __call__(self, draft: str, *, backend: object = None, evidence: dict[str, Any] | None = None,
                 research: bool = False, codes: tuple[str, ...] = CODES, confirmation: bool = False) -> DraftReview:
        if confirmation:
            self.confirmations.append((draft, dict(evidence or {}), tuple(codes)))
            script = self.confirmation
        else:
            self.calls.append((draft, dict(evidence or {})))
            script = self.script
        if callable(script):
            item = script(draft, evidence or {})
        elif isinstance(script, list):
            item = script.pop(0) if len(script) > 1 else script[0] if script else None
        elif isinstance(script, dict) and "verdict" not in script and "status" not in script:
            item = script.get(draft)
        else:
            item = script
        if isinstance(item, BaseException):
            raise item
        if item is None:
            return unavailable("no scripted review")
        if isinstance(item, DraftReview):
            return item
        return parse_review(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False), draft, codes)

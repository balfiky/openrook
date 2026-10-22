"""Conservative Unicode evidence of reply language drift, not language choice.

Scripts and Latin diacritic density can prove a large distribution change.
Similar distributions cannot establish that two texts use the same language;
semantic review owns that question. No language names or word bans are used.
"""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

from core.llm_content import requests_literal_text


_LANGUAGE_REQUEST = re.compile(
    r"\blanguage\b|\b(?:reply|respond|answer|speak|write|explain)\b"
    r"[^.!?\n]{0,60}\b(?:in|using)\s+\S+", re.I,
)
_LITERALS = re.compile(
    r'```.*?(?:```|\Z)|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”'
    r"|(?<!\w)'[^'\n]*'(?!\w)|https?://\S+|\b[\w.-]+(?:[/\\][\w.-]+)+", re.S,
)


def _profile(text):
    scripts, marked, letters = Counter(), 0, 0
    for char in unicodedata.normalize("NFC", _LITERALS.sub("", text)):
        if not char.isalpha():
            continue
        family = unicodedata.name(char, "UNKNOWN").split()[0]
        scripts[family] += 1
        letters += 1
        marked += any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", char))
    return {"letters": letters, "scripts": dict(scripts), "marked": marked}


def language_mismatch(response: str, user_message: str, *, recent_context=(), observed_outputs=()) -> dict:
    """Return high-confidence mismatch evidence, or {} when inconclusive.

    Use current user prose when substantial, recent user prose for short
    follow-ups. Assistant replies and receipt language never set the baseline.
    Explicit language/literal requests defer to semantic review. Exclude quoted
    text, code, paths and actual observed output from the distribution sample.
    """
    users = [item.get("content", "") for item in recent_context
             if item.get("role") == "user"][-3:]
    if any(requests_literal_text(text) or _LANGUAGE_REQUEST.search(text)
           for text in [*users, user_message]):
        return {}
    sample = response
    for output in observed_outputs:
        if len(output.strip()) >= 12:
            sample = sample.replace(output, "")
    current = _profile(user_message)
    baseline = current if current["letters"] >= 24 else _profile("\n".join([*users, user_message]))
    draft = _profile(sample)
    if baseline["letters"] < 24 or draft["letters"] < 40:
        return {}
    unexpected = sum(count for script, count in draft["scripts"].items()
                     if not baseline["scripts"].get(script)) / draft["letters"]
    reason = ""
    if unexpected >= 0.65:
        reason = "dominant script absent from user prose"
    elif (baseline["scripts"].get("LATIN", 0) / baseline["letters"] >= 0.9
          and draft["scripts"].get("LATIN", 0) / draft["letters"] >= 0.9):
        before = baseline["marked"] / baseline["letters"]
        after = draft["marked"] / draft["letters"]
        if abs(after - before) >= 0.18 and max(baseline["marked"], draft["marked"]) >= 10:
            reason = "large Latin diacritic distribution change"
    return {"reason": reason, "user": baseline, "reply": draft} if reason else {}

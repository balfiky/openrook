"""Extract reply text without treating model reasoning as an answer."""

from __future__ import annotations

import re
from typing import Any


_THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.I | re.S)
_THINK_END = re.compile(r"^.*</(?:think|thinking)\s*>", re.I | re.S)
_INTERNAL_EXPRESSION_MARKER = re.compile(
    r"\[\[SILENCE\]\]|\[(?:declined request\s*;\s*answer context omitted|"
    r"(?:Defense instruction|Self-check correction):[^\]]*)\]", re.I,
)
_LITERAL_REQUEST = re.compile(
    r"\b(?:quote|verbatim|translate|rewrite|proofread|transcribe)\b"
    r"|\b(?:repeat|say) (?:that|it|this|exactly|after me)\b", re.I,
)
_MARKED_LINE = re.compile(r"(?m)^\s*\*([^*\n]+)\*[ \t]*(?:\n|$)")
_FORMAT_LINE = re.compile(r"(?m)^\s*%(?:[-+#0 ]*\d*(?:\.\d+)?[hlL]?[diuoxXfFeEgGaAcspn])[ \t]*(?:\n|$)")


def requests_literal_text(message: str) -> bool:
    """Copying is appropriate when the user actually requests text reproduction."""
    return bool(_LITERAL_REQUEST.search(message))


def strip_expression_artifacts(text: str, *, style_text=(), user_message: str = "") -> str:
    """Remove presentation controls, preserving requested literals and inline emphasis.

    Delivery descriptions are private directions. Match their words from the
    actual runtime configuration, rather than maintaining a vocabulary ban.
    A short marked delivery frame also covers paraphrased stage directions.
    """
    text = text.strip()
    if requests_literal_text(user_message):
        return text
    style_words = set(re.findall(r"\b\w{4,}\b", " ".join(style_text).casefold()))

    def direction(match):
        span = match[1]
        words = re.findall(r"\w+", span.casefold())
        descriptor = not re.search(r"[.!?]|\b(?:i|you|we|he|she|they|it)\b", span, re.I)
        derived = bool(style_words.intersection(words))
        delivery_frame = bool(re.search(r"\b(?:with|in)\b.+\b(?:tone|voice|edge|manner)\b", span, re.I))
        return "" if descriptor and len(words) <= 20 and (derived or delivery_frame) else match[0]

    # Fenced/inline code can legitimately contain formatting syntax.
    chunks = re.split(r"(```.*?(?:```|\Z)|`[^`\n]*`)", text, flags=re.S)
    for index in range(0, len(chunks), 2):
        chunks[index] = _FORMAT_LINE.sub(
            lambda match: match[0] if match[0].strip() in user_message else "",
            _MARKED_LINE.sub(direction, chunks[index]),
        )
    text = "".join(chunks).strip()
    if not re.search(r"\b(?:italic|bold|emphasi[sz]e|emphasis|markdown)\b", user_message, re.I):
        # A whole spoken reply wrapped as an action is presentation leakage;
        # ordinary emphasis within a sentence remains intact.
        wrapped = re.fullmatch(r"\*([^*]+)\*", text, re.S)
        if wrapped:
            text = wrapped[1].strip()
    return text


def strip_internal_expression_markers(text: str) -> str:
    """Reserved runtime controls are never dialogue, including inline echoes."""
    return _INTERNAL_EXPRESSION_MARKER.sub("", text).strip()


def strip_reasoning(text: str) -> str:
    """Drop complete or truncated thinking blocks, including a prompt-opened block."""
    return _THINK_END.sub("", _THINK_BLOCK.sub("", text)).strip()


def reply_text(message: dict[str, Any]) -> str:
    """Read only textual content; separate reasoning/thinking fields are ignored."""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
            and isinstance(part.get("text"), str)
        )
    return strip_reasoning(content) if isinstance(content, str) else ""

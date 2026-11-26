"""Extract reply text without treating model reasoning as an answer."""

from __future__ import annotations

import json
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


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*(?:```|\Z)", re.I | re.S)


def json_payload(text: str) -> str:
    """The JSON text of a model answer, without an optional Markdown code fence."""
    text = (text or "").strip()
    if text[:1] in ("{", "["):
        return text  # Already a document; backticks inside its strings are content.
    fenced = _JSON_FENCE.search(text)
    return fenced[1].strip() if fenced else text


# Why an answer holds no JSON object, one reason at a time.
ABSENT = "absent"
NOT_OBJECT = "not an object"
CUT_OFF = "cut off"
INVALID = "invalid"


def _closing(text: str, start: int) -> int:
    """Where the bracketed group opening at ``start`` ends, or -1 when it is still open at the end."""
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth <= 0:
                return index + 1
    return -1


def outermost_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """The JSON object a model answer consists of, and otherwise why there is none.

    Only an outermost value counts. Prose or a code fence around the object, or
    talk after it, does no harm, and a bracketed aside in that prose is stepped
    over as a whole. Nothing is ever looked for *inside* a bracketed group:

    * when a group never closes, the answer was cut off there, and an object
      nested in it is a part of the answer, never the answer;
    * when the first group that is JSON is a list, the answer is a list.

    The second result is one of :data:`ABSENT`, :data:`NOT_OBJECT`,
    :data:`CUT_OFF` and :data:`INVALID`, or empty for an object.
    """
    text = text or ""
    decoder = json.JSONDecoder()
    position, stepped_over = 0, False
    while True:
        start = next((index for index in range(position, len(text)) if text[index] in "{["), -1)
        if start < 0:
            return None, INVALID if stepped_over else ABSENT
        end = _closing(text, start)
        if end < 0:
            return None, CUT_OFF
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            position, stepped_over = end, True
            continue
        return (value, "") if isinstance(value, dict) else (None, NOT_OBJECT)


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

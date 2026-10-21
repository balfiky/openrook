"""Extract reply text without treating model reasoning as an answer."""

from __future__ import annotations

import re
from typing import Any


_THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.I | re.S)
_THINK_END = re.compile(r"^.*</(?:think|thinking)\s*>", re.I | re.S)


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

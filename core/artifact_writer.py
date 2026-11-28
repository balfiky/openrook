"""Drafting a file the person asked to have written.

A request to produce text and store it in a file is reported by the reading: a
task that names the file-writing capability, with the path the person wrote and
the content marked as to be generated (``docs/development/request-routing.md``).
Nothing here recognises such a request. This module only drafts the content
once the decision has granted the task: the model supplies the file's text, and
the runtime writes it through the normal filesystem tool, to the committed path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from core.inference import generate, primary_deadline


@dataclass(frozen=True)
class ArtifactWriteRequest:
    """A granted request to generate content and store it in one file."""

    path: str      # the path the person wrote; committed with the grant
    request: str   # what was asked, as the reading reports it


@dataclass(frozen=True)
class ArtifactDraft:
    """Generated file payload."""

    path: str
    content: str
    summary: str = ""


def generate_artifact_draft(
    backend: Any,
    request: ArtifactWriteRequest,
) -> ArtifactDraft:
    """Ask the LLM for one complete file payload as strict JSON.

    The draft's path is the request's: whatever path an answer names is
    ignored, so the file written is the one the decision granted.

    The call is one of the turn's model calls like any other: it is counted,
    recorded in the turn's call telemetry as ``artifact_draft`` and held to
    the request deadline in force. It takes the granted task's one protected
    call, which the same task would otherwise spend on tool selection.
    """
    system_prompt = (
        "Generate one complete file for the user's request. "
        "Return JSON only, with keys: content_lines, summary. "
        "content_lines must be an array of strings, one string per file line. "
        "Do not put the whole code file inside a nested JSON string. "
        "Do not use Markdown fences. Do not omit code. "
        f"The file is stored at this path: {request.path!r}."
    )
    raw = generate(backend, "artifact_draft", system_prompt, request.request, primary_tool=True,
                   limit_seconds=primary_deadline(backend))
    payload = _parse_artifact_json(raw)
    content = _payload_content(payload)
    if not content.strip():
        content = _extract_loose_content_field(raw)
    if not content.strip():
        content = _extract_fenced_content(raw)
    if not content.strip() and not _looks_like_jsonish(raw):
        content = (raw or "").strip()
    if not content.strip():
        raise ValueError("Generated artifact response did not include file content.")
    return ArtifactDraft(
        path=request.path,
        content=content.rstrip() + "\n",
        summary=str(payload.get("summary") or "").strip(),
    )


def _parse_artifact_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _payload_content(payload: dict[str, Any]) -> str:
    lines = payload.get("content_lines")
    if isinstance(lines, list):
        normalized = [str(line) for line in lines]
        if normalized:
            return "\n".join(normalized)
    content = payload.get("content")
    if isinstance(content, str):
        return content
    return ""


def _extract_loose_content_field(raw: str) -> str:
    """Extract content from common malformed JSON with literal newlines.

    Some local models return `"content": "line1\nline2"` with literal
    newlines instead of escaped JSON newlines. That object is invalid JSON, but
    the intended file payload is still recoverable. We recover the content
    field rather than writing the enclosing object into the target file.
    """
    text = raw or ""
    match = re.search(r'"content"\s*:\s*"', text)
    if not match:
        return ""
    start = match.end()
    tail = text[start:]
    end = re.search(r'"\s*,\s*"(?:summary|path|content_lines)"\s*:', tail, flags=re.DOTALL)
    if not end:
        end = re.search(r'"\s*}\s*$', tail, flags=re.DOTALL)
    if not end:
        return ""
    value = tail[:end.start()]
    json_value = '"' + value.replace("\n", "\\n").replace("\r", "\\r") + '"'
    try:
        decoded = json.loads(json_value)
        return decoded if isinstance(decoded, str) else ""
    except json.JSONDecodeError:
        return (
            value
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\\\", "\\")
        )


def _extract_fenced_content(raw: str) -> str:
    text = (raw or "").strip()
    match = re.search(r"```(?:[A-Za-z0-9_+-]+)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _looks_like_jsonish(raw: str) -> bool:
    text = (raw or "").lstrip()
    return text.startswith("{") or text.startswith("[")

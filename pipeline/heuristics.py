"""Module-level heuristics for the cognitive pipeline.

Regex tables, repair/identity detection, message builders for tool
routing, and small topic-hint helpers. These are deterministic and
stateless — the orchestrator in ``pipeline.core`` calls them per turn.
"""

from __future__ import annotations

import re

from core.dual_process.tool_loop import attach_followup_context
from core.types import ToolResult

_IDENTITY_QUESTION_RE = re.compile(
    r"\bwhat (?:did|do) you learn\b|"
    r"\bwhat (?:changed|did you change)\b|"
    r"\bdid you evolve\b|"
    r"^\s*evolve[.!?]?\s*$|"
    r"\bwhat do you (?:remember|believe)\b",
    re.IGNORECASE,
)

_BROAD_REPAIR_RE = re.compile(
    r"\b(?:everything|all of (?:it|this)|all the tension|all of the tension|"
    r"forgive(?: me| this| everything)?|move on|make peace|find peace|"
    r"repair (?:this|everything|all)|start over|clean slate)\b",
    re.IGNORECASE,
)

# Hot transcript bound. Every reader uses a recent window (generator: last
# 10 turns, digestion summary: last 20 messages, model tool routing: last 8,
# anticipation: last 5 user messages, relationship digestion: bounded by the
# 200-entry short-term memory), so trimming to the newest 400 entries
# (~200 turns) cannot change any consumer's observable behavior.
_CONVERSATION_HISTORY_MAX = 400

_REPAIRABLE_UNRESOLVED_SOURCES = {
    "spike",
    "contradiction",
    "topic",
    "dialogue_deadlock",
    "relationship",
    "hostility",
}


def _is_identity_question(text: str) -> bool:
    return bool(_IDENTITY_QUESTION_RE.search(text or ""))


def _is_broad_repair_text(text: str) -> bool:
    return bool(_BROAD_REPAIR_RE.search(text or ""))


def _is_repairable_unresolved_source(source: str) -> bool:
    return str(source or "") in _REPAIRABLE_UNRESOLVED_SOURCES


def _format_generated_artifact_summary(result: ToolResult, draft_summary: str = "") -> str:
    details = []
    if result.success:
        details.append("OK")
        if result.side_effect_summary and result.side_effect_summary != "none":
            details.append(result.side_effect_summary)
        bytes_written = (result.metadata or {}).get("bytes_written")
        if bytes_written is not None:
            details.append(f"bytes_written={bytes_written}")
        if draft_summary:
            details.append(f"summary={draft_summary}")
    else:
        details.append(f"Failed: {result.error or 'unknown error'}")
    return "[fs.write_file] " + " | ".join(details)

def _message_for_tool_detection(
    user_message: str,
    conversation_history: list[dict[str, str]],
) -> str:
    """Attach the prior user request for terse tool follow-ups.

    Delegates to the declarative FOLLOWUP_REWRITE_RULES table that lives
    next to the heuristic intent rules in core/dual_process/tool_loop.py.
    """
    return attach_followup_context(user_message, conversation_history)


def _message_for_model_tool_routing(
    user_message: str,
    conversation_history: list[dict[str, str]],
) -> str:
    """Attach recent turns so model-native routers can resolve follow-ups."""
    if not conversation_history:
        return user_message
    lines = ["Recent conversation for tool routing:"]
    for message in conversation_history[-8:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"Current user message: {user_message}")
    return "\n".join(lines)


def _relationship_topic_hint(text: str) -> str:
    """Small topic hint so relationship-memory loops can be prioritized."""
    lower = text.lower()
    match = re.search(r"\b(?:about|regarding|around|on)\s+([a-z0-9' -]{2,40})", lower)
    if match:
        return _clean_topic_hint(match.group(1))
    words = re.findall(r"[a-z0-9']+", lower)
    stopwords = {
        "the", "and", "that", "this", "with", "your", "you", "for", "from",
        "have", "been", "just", "really", "very", "again", "still", "please",
        "drop", "topic", "issue", "thing", "can", "could", "would", "what",
        "when", "where", "why", "how", "any", "follow", "up",
        "unresolved", "resolved", "matters", "matter", "pattern",
    }
    informative = [word for word in words if len(word) > 2 and word not in stopwords]
    return " ".join(informative[:3])


def _clean_topic_hint(text: str) -> str:
    phrase = re.sub(r"[^a-z0-9' -]", " ", text.lower())
    phrase = " ".join(phrase.split())
    return phrase[:40].strip()

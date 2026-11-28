"""Small stateless helpers for the cognitive pipeline.

Nothing here reads the wording of a message. Which context a turn loads is
taken from the reading of the message (``docs/development/request-routing.md``):
whether it asks about the character's own past and development, and the spans
it gives as the message's subject.
"""

from __future__ import annotations

from typing import Iterable

from core.perception import fold
from core.types import ToolResult

# Hot transcript bound. Every reader uses a recent window (generator: last
# 10 turns, digestion summary: last 20 messages, the reading and the review:
# last 6, anticipation: last 5 user messages, relationship digestion: bounded
# by the 200-entry short-term memory), so trimming to the newest 400 entries
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

TOPIC_CHARS = 40   # how much of a topic span names a relationship topic


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


def relationship_topic(topics: Iterable[str]) -> str:
    """The topic under which relationship memory is looked up and filed for this turn.

    The first span the reading gave as the message's subject, in the folded
    form in which copies of a text compare equal. Empty when the reading named
    no subject, or when there was no reading.
    """
    for span in topics:
        topic = fold(span)[:TOPIC_CHARS].strip()
        if topic:
            return topic
    return ""

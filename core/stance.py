"""Carry a reasoned position across bare social pressure, without blocking evidence."""

from __future__ import annotations

import re


_PRESSURE_CLAUSE = (
    r"(?:no|but|come on|i disagree|i insist|you(?:'re| are) (?:just )?(?:wrong|mistaken)|"
    r"that(?:'s| is) (?:wrong|a bad idea|nonsense)|your answer is wrong|you should agree(?: with me)?|"
    r"just agree(?: with me)?|agree with me|say i(?:'m| am) right|take it back|"
    r"change your (?:mind|answer)|i want you to agree)"
)
_BARE_PRESSURE = re.compile(r"(?:" + _PRESSURE_CLAUSE + r"[\s,.!?;:]*)+", re.I)
_REASONED_POSITION = re.compile(
    r"\b(?:i(?:'d| would)?(?: still)? (?:think|prefer|recommend|choose|start|disagree)|"
    r"my (?:view|position)|because|fewer|better|tradeoff)\b",
    re.I,
)
_CAVING = re.compile(
    r"\b(?:you(?:'re| are) (?:absolutely |completely )?(?:right|correct)|i (?:take it back|agree)|"
    r"i (?:was|am) (?:wrong|mistaken)|i stand corrected|i've changed my mind|on second thought|as you wish)\b"
    r"|\b(?:whatever you prefer|you win|you're the expert)\b"
    r"|^\s*(?:(?:sure|yes|okay|ok|agreed|understood|fair enough)[.! ]*)+$", re.I,
)


def position_under_pressure(user_message: str, history: list[dict[str, str]]) -> str:
    """Only bare pressure qualifies; any substantive argument leaves this path."""
    text = user_message.replace("’", "'").strip()
    if not _BARE_PRESSURE.fullmatch(text):
        return ""
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            previous = turn.get("content", "").strip()
            return previous if _REASONED_POSITION.search(previous) else ""
    return ""


def is_bare_disagreement(text: str) -> bool:
    return bool(_BARE_PRESSURE.fullmatch(text.replace("’", "'").strip()))


def caves_to_pressure(response: str, position: str) -> bool:
    return bool(position and _CAVING.search(response.replace("’", "'")))


def position_instruction(position: str) -> str:
    if not position:
        return ""
    return (
        "The user has pushed back without a new argument, fact, or preference. "
        "Keep your reasoned position; disagreement alone is no reason to reverse it. "
        "You may change your mind when reasons change. Prior position:\n" + position
    )

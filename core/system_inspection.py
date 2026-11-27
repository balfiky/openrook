"""Narrow deterministic routing for direct host metric requests."""

from __future__ import annotations

import re

from core.types import ToolIntent


def inspection_kind(message: str, history: list[dict[str, str]] | None = None) -> str:
    text = message.lower().replace("’", "'")
    if re.search(
        r"\b(?:don't|do not|stop|no need)\b.{0,50}\b(?:check|inspect|show|cpu|ram|memory)\b"
        r"|\b(?:explain|define|meaning|what is|what's)\s+(?:a |the )?(?:cpu|ram|memory)\s*\??$"
        r"|\bmemory (?:of|about|leak|safety|management|model|allocation)\b"
        r"|\b(?:remember|conversation|story|translate|quote|pretend|fiction)\b",
        text,
    ):
        return ""
    match = re.search(r"\b(cpu|ram|memory|disk|storage)\b", text)
    if not match:
        return ""
    kind = {"ram": "memory", "storage": "disk"}.get(match[1], match[1])
    explicit = bool(re.search(
        r"\b(?:usage|utili[sz]?ation|utiliation|current|live|stats?|status|free|available|left|"
        r"show|check|tell|report|how much)\b", text,
    ))
    followup = bool(re.fullmatch(r"(?:and |what about )?(?:cpu|ram|memory|disk|storage)[?.! ]*", text.strip()))
    prior_inspection = False
    if followup and not explicit:
        # Count user turns, not alternating messages. Short intervening social
        # turns need not erase the host-metric referent, but an explicit switch
        # to recollection or conceptual discussion does.
        recent_users = [turn.get("content", "") for turn in (history or []) if turn.get("role") == "user"][-5:]
        for prior in reversed(recent_users):
            if inspection_kind(prior):
                prior_inspection = True
                break
            if re.search(r"\b(?:remember|recollection|memory (?:of|about)|"
                         r"(?:explain|define|what is) (?:cpu|ram|memory))\b", prior, re.I):
                break
    return kind if explicit or (followup and (prior_inspection or text.strip() in {"cpu", "ram"})) else ""


def inspection_intent(kind: str, available: set[str]) -> ToolIntent | None:
    choices = {
        "cpu": [("system.cpu_usage", {}), ("shell.run_command", {"cmd": "top -bn1"})],
        "memory": [("system.memory_usage", {}), ("shell.run_command", {"cmd": "free -m"})],
        "disk": [("system.disk_usage", {"path": "/"}), ("shell.run_command", {"cmd": "df -h /"})],
    }
    for name, arguments in choices.get(kind, []):
        if name in available:
            return ToolIntent(tool_name=name, arguments=arguments,
                              reason=f"Direct {kind} inspection request",
                              expected_outcome=f"Observe current {kind} usage")
    return None


_EVALUATION_COMPATIBILITY = ("addresses_inspection",)


def __getattr__(name: str):
    """A name the role-play evaluation still imports from here; no turn uses it.

    The former reply-relevance check lives with the harness in
    ``openrook_tools/roleplay_eval/legacy_claims.py``.
    """
    if name in _EVALUATION_COMPATIBILITY:
        from openrook_tools.roleplay_eval import legacy_claims

        return getattr(legacy_claims, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

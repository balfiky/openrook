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
    prior_inspection = any(
        inspection_kind(turn.get("content", ""))
        for turn in (history or [])[-6:] if turn.get("role") == "user"
    ) if followup and not explicit else False
    return kind if explicit or (followup and (prior_inspection or text.strip() in {"cpu", "ram"})) else ""


def inspection_intent(kind: str, available: set[str]) -> ToolIntent | None:
    choices = {
        "cpu": [("shell.run_command", {"cmd": "top -bn1"})],
        "memory": [("system.memory_usage", {}), ("shell.run_command", {"cmd": "free -m"})],
        "disk": [("system.disk_usage", {"path": "/"}), ("shell.run_command", {"cmd": "df -h /"})],
    }
    for name, arguments in choices.get(kind, []):
        if name in available:
            return ToolIntent(tool_name=name, arguments=arguments,
                              reason=f"Direct {kind} inspection request",
                              expected_outcome=f"Observe current {kind} usage")
    return None


def inspection_reply(kind: str, trace, *, tools_enabled: bool = True, shell_enabled: bool = True) -> str:
    """Observed facts or one short reason; never a grounding-system lecture."""
    label = "RAM" if kind == "memory" else kind.upper() if kind == "cpu" else "Disk"
    for result in getattr(trace, "executed_results", []) or []:
        if not (result.tool_name.startswith("system.") or result.tool_name == "shell.run_command"):
            continue
        if not result.success:
            # Executor errors are already redacted and bounded at the tool boundary.
            reason = " ".join((result.error or "inspection failed").split())[:180].rstrip(".")
            return f"I can't read {label} usage: {reason}."
        output = (result.output or "").strip()
        if not output:
            continue
        if kind == "cpu":
            idle = re.search(r"([\d.]+)\s*id\b", output)
            if idle:
                return f"CPU utilization: {100 - float(idle[1]):.1f}%."
        return output[:1200]
    decision = getattr(trace, "final_decision", None)
    if decision and decision.decision != "execute":
        reason = decision.rationale.rstrip(".")
    elif not tools_enabled:
        reason = "system tools are disabled"
    elif kind == "cpu" and not shell_enabled:
        reason = "shell access is disabled"
    else:
        reason = "no inspection tool is available" if not trace or not trace.executed_results else "the inspection returned no data"
    return f"I can't read {label} usage: {reason}."

"""EVALUATION HARNESS ONLY: wording-based claim, metric and language checks.

Until sub-phase 1b the runtime's reply checks decided with these English
patterns whether a sentence claimed an action or stated a measurement. The
runtime no longer does: the model reviewer judges claims
(``core/review.py``) and ``core/grounding.py`` compares only structure with the
receipts. The former code is kept here, verbatim, because the deterministic
checks of the role-play evaluation (``openrook_tools/roleplay_eval/checks.py``)
and its judge still call it, through compatibility names in ``core``:

* ``core.grounding.verify_response_grounding``
* ``core.metric_claims``: ``cpu_idle_percentages``, ``cpu_measurement_complete``,
  ``percentage_comparisons``, ``cpu_interpretation_conflicts``,
  ``capacity_interpretation_conflicts``
* ``core.system_inspection.addresses_inspection``
* ``core.response_language.language_mismatch``
* ``core.llm_content.requests_literal_text``

No turn calls anything in this module; ``tests/test_reply_guards.py`` fails if
one does, or if a runtime module outside the evaluation harness imports it by
any other route than those names.
"""

from __future__ import annotations

from collections import Counter
import unicodedata

from dataclasses import dataclass, field
import re
from typing import Any
from types import SimpleNamespace

from core.grounding import blocked_before_execution


@dataclass(frozen=True)
class ActionClaim:
    """A response claim that OpenRook performed an external action."""

    text: str
    categories: set[str] = field(default_factory=set)
    context: str = ""


@dataclass(frozen=True)
class GroundingIssue:
    """A response grounding problem suitable for debug/self-check output."""

    code: str
    message: str
    claim: str = ""
    required_categories: tuple[str, ...] = ()
    evidence_categories: tuple[str, ...] = ()


_ACTION_VERBS_RE = re.compile(
    r"\b("
    r"learned|ingested|digested|read(?:ing)?|wrote|writing|write|created?|creating|saved?|saving|"
    r"cloned?|cloning|fetched?|fetching|downloaded?|downloading|installed?|installing|"
    r"ran|running|executed?|executing|checked?|checking|opened?|opening|"
    r"searched?|searching|listed?|listing|inspected?|inspecting|loaded?|loading|"
    r"called?|calling|integrated?|integrating|registered|registering|added|adding|"
    r"forced?|forcing|initialized?|initializing|patched?|patching|"
    r"killed?|killing|terminated?|terminating|aborted?|aborting|"
    r"fixed?|fixing|repaired?|repairing|restarted?|restarting|"
    r"started?|starting|launched?|launching|sourced?|sourcing|"
    r"activated?|activating|deactivated?|deactivating|"
    r"built?|building|compiled?|compiling|set\s+up|setting\s+up"
    r")\b",
    re.IGNORECASE,
)

_EXTERNAL_OBJECT_RE = re.compile(
    r"\b("
    r"repo(?:sitory)?|checkout|file|folder|directory|script|logs?|output|"
    r"workspace|environment|shell|terminal|command|process|package|dependency|"
    r"system|memory|ram|cpu|disk|storage|metric|utili[sz]ation|"
    r"url|web(?:site)?|page|api|endpoint|server|database|registry|skill|"
    r"capability|capabilities|module|integration|toolset|skillset|life\s+history|ledger"
    r")\b",
    re.IGNORECASE,
)

_FIRST_PERSON_ACTION_RE = re.compile(
    r"\b(?:i(?:'m| am| have|\'ve)?|now)\b[^.!?\n]{0,80}"
    r"\b(?:learned|ingested|digested|read(?:ing)?|wrote|writing|created?|creating|saved?|saving|"
    r"cloned?|cloning|fetched?|fetching|downloaded?|downloading|installed?|installing|"
    r"ran|running|executed?|executing|checked?|checking|opened?|opening|"
    r"searched?|searching|listed?|listing|inspected?|inspecting|loaded?|loading|"
    r"called?|calling|integrated?|integrating|registered|registering|added|adding|"
    r"forced?|forcing|initialized?|initializing|patched?|patching|"
    r"killed?|killing|terminated?|terminating|aborted?|aborting|"
    r"fixed?|fixing|repaired?|repairing|restarted?|restarting|"
    r"started?|starting|launched?|launching|sourced?|sourcing|"
    r"activated?|activating|deactivated?|deactivating|"
    r"built?|building|compiled?|compiling|set\s+up|setting\s+up)"
    r"\b[^.!?\n]{0,100}",
    re.IGNORECASE | re.DOTALL,
)

_STATUS_CLAIM_RE = re.compile(
    r"\b(?:tool|shell|terminal|command|process|install|download|clone|fetch|write|read|"
    r"environment|venv|conda|script|build|container|file|package)"
    r"\b.{0,60}\b(?:ran|executed|printed|failed|succeeded|completed|returned|produced|output|logs?|"
    r"working|ready|done|finished|hanging|stuck|deadlock|"
    r"running|active|created|installed|downloaded|fixed|patched|initialized|"
    r"activated|set\s+up)\b",
    re.IGNORECASE | re.DOTALL,
)

_REGISTRY_STATE_CLAIM_RE = re.compile(
    r"\b(?:created?|creating|imported?|importing|enabled?|enabling|"
    r"activated?|activating|installed?|installing|integrated?|integrating|"
    r"registered|registering|added|adding|made\s+permanent)\b"
    r".{0,100}\b(?:skill|capability|capabilities|module|integration|"
    r"registry|toolset|skillset)\b"
    r"|"
    r"\b(?:skill|capability|capabilities|module|integration|toolset|skillset)"
    r"\b.{0,100}\b(?:created?|imported?|enabled?|activated?|installed?|"
    r"integrated?|registered|added|permanent|persistent|persisted)\b"
    r"|"
    r"\b(?:now|already)\b.{0,80}\b(?:part\s+of|available\s+in|added\s+to)\b"
    r".{0,80}\b(?:context|runtime|registry|configuration|toolset|skillset|"
    r"capability|capabilities)\b"
    r"|"
    r"\b(?:done|completed)\b.{0,80}\b(?:here\s+(?:is|are)|this\s+is)\b"
    r".{0,80}\b(?:skill|capability|capabilities|module|integration)\b",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_METRIC_CLAIM_RE = re.compile(
    r"\b(?:memory|ram|cpu|disk|storage|system\s+memory|system\s+metric)\b"
    r".{0,90}\b(?:usage|utili[sz]ation|used|free|available|total|"
    r"currently|current|at\s+\d{1,3}%|\d+(?:\.\d+)?\s*(?:gb|mb|kb|%))\b"
    r"|"
    r"\b(?:system|runtime|host)\b.{0,40}\b(?:stable|using|uses|has)\b"
    r".{0,80}\b(?:memory|ram|cpu|disk|storage|\d+(?:\.\d+)?\s*(?:gb|mb|kb|%))\b",
    re.IGNORECASE | re.DOTALL,
)
_PERMISSION_EXECUTION_RE = re.compile(
    r"\bpermission (?:allowed|permitted) (?:the )?(?:command|execution|tool)\b", re.I,
)
_EXPLICIT_EXECUTION_RE = re.compile(
    r"\bi (?:(?:did|have|actually|just)\s+)*(?:ran|executed|did run)\b[^.!?\n]{0,60}", re.I,
)
_NONEXECUTION_RE = re.compile(
    r"\b(?:nothing|no (?:tool|command|code|process)(?: call)?)\s+(?:was\s+|has\s+|actually\s+)*"
    r"(?:executed|run|ran|attempted)\b"
    r"|\b(?:wasn't|was not|weren't|were not|didn't|did not|never)\s+(?:actually\s+)?(?:run|ran|execute|executed)\b"
    r"|\b(?:tool\s+)?autonomy\s+(?:is\s+)?off\b", re.I,
)


def reports_nonexecution(text: str) -> bool:
    return bool(_NONEXECUTION_RE.search(text.replace("’", "'").replace("‘", "'")))


def _unsupported_output_spans(response: str, tool_trace) -> list[str]:
    """Bind attributed output blocks and their explanation to observed bytes.

    Category evidence alone cannot verify a quoted result. Keep the whole output
    attribution together for excision, including its completion header and any
    attached interpretation; otherwise removing just 'produced' leaves fake data.
    """
    observed = [_compact(r.output) for r in getattr(tool_trace, "executed_results", [])
                if r.output and not blocked_before_execution(r)]
    spans = []
    for block in re.finditer(r"```[^\n]*\n(?P<output>.*?)```", response, re.S):
        prefix = response[:block.start()]
        intro = re.search(
            r"(?:\b(?:command|tool|shell|process)\b[^.!?\n]{0,60}\b(?:produced|returned|printed)\b"
            r"[^.!?\n]*|\b(?:actual|real|observed) output\b[^.!?\n]*)\s*$", prefix, re.I)
        if not intro or _is_negated_or_evidence_warning(intro[0]):
            continue
        output = _compact(block["output"])
        if output and any(output in receipt for receipt in observed):
            continue
        start = max(prefix.rfind(c, 0, intro.start()) + 1 for c in ".!?\n")
        header = re.search(r"\b(?:done|completed|finished)[.!]\s*$", response[:start], re.I)
        if header:
            start = header.start()
        end = block.end()
        tail = re.match(r"\s*\n\s*\n([^\n]+)", response[end:])
        if tail and re.search(r"\b(?:output|result|string|print\w*|lines?|characters?)\b", tail[1], re.I):
            end += tail.end()
        spans.append(response[start:end].strip())
    return spans


def verify_response_grounding(
    response: str,
    *,
    tool_trace: Any | None,
    prior_tool_receipts=(),
) -> list[GroundingIssue]:
    """Return issues for external-action claims not backed by tool evidence.

    This is presentation/output verification. It does not infer user intent and
    it does not special-case any product, repo, provider, or prompt. The rule is
    simple: claims that OpenRook performed external actions must be backed by a
    matching category of executed tool result.
    """
    claims = extract_action_claims(response)
    evidence = _tool_evidence_categories(tool_trace)
    issues: list[GroundingIssue] = []
    gate = getattr(tool_trace, "final_decision", None)
    if gate and gate.decision != "execute" and re.search(r"\bautonomy (?:mode )?is off\b", gate.rationale, re.I):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", response):
            if re.search(r"\b(?:when|once|until) tools are enabled\b|\byou (?:didn't|did not) (?:grant|give) permission\b"
                         r"|\bI (?:chose|decided) not to (?:run|execute)\b", sentence, re.I):
                issues.append(GroundingIssue("unverified_external_action_claim",
                    "The recorded blocker is the operator's autonomy setting, not a character decline, missing user permission or disabled tools. Report that setting accurately.", sentence))
    blocked_commands = [str((r.metadata or {}).get("cmd", "")) for r in
        getattr(tool_trace, "executed_results", []) if r.tool_name.startswith("shell.")
        and blocked_before_execution(r) and (r.metadata or {}).get("cmd")]
    if blocked_commands:
        import shlex
        supported = set()
        for command in blocked_commands:
            supported.add(_compact(command))
            try:
                supported.add(shlex.split(command)[0])
            except (ValueError, IndexError):
                pass
        for claim in re.finditer(
                r"\b(?:the )?(?:shell|tool|operator)\b[^.!?\n]{0,35}\b(?:refused|rejected|blocked|denied)\b"
                r"[^.!?\n`]{0,25}`(?P<cmd>[^`\n]+)`[^.!?\n]*", response, re.I):
            if _compact(claim["cmd"]) not in supported:
                issues.append(GroundingIssue("unverified_external_action_claim",
                    "This denial names a different command from the actual blocked attempt. Report the receipt's attempted arguments and policy error.", claim[0]))
    for span in _unsupported_output_spans(response, tool_trace):
        issues.append(GroundingIssue("unverified_external_action_claim",
            "This attributed command output is not present in the execution receipts. Remove its output block and dependent interpretation; generate from the actual receipt.", span))
    for claim in claims:
        supported = evidence
        scope = claim.context or claim.text
        if re.search(r"\b(?:earlier|previously|last turn|before this turn)\b", scope, re.I):
            historical = _matching_prior_receipts(scope, prior_tool_receipts)
            supported = evidence | _tool_evidence_categories(SimpleNamespace(executed_results=historical))
        missing = _missing_categories(claim.categories, supported)
        if not missing:
            continue
        issues.append(
            GroundingIssue(
                code="unverified_external_action_claim",
                message=(
                    "Unverified external-action claim. Do not say that an "
                    "external action was performed unless matching Tool "
                    "Execution Results confirm it."
                ),
                claim=claim.text,
                required_categories=tuple(sorted(claim.categories)),
                evidence_categories=tuple(sorted(supported)),
            )
        )
    for conflict in cpu_interpretation_conflicts(response, [vars(r) for r in
            getattr(tool_trace, "executed_results", [])]):
        issues.append(GroundingIssue("unverified_external_action_claim",
            f"The CPU idle share implies {conflict['expected_usage']} aggregate usage; preserve idle/busy meanings and do not report a process share as total usage.",
            conflict["claim"], ("system_metric:cpu",), tuple(sorted(evidence))))
    for topic, name in (("memory", "system.memory_usage"), ("disk", "system.disk_usage")):
        results = [r for r in getattr(tool_trace, "executed_results", []) if r.success and r.tool_name == name]
        if results:
            for conflict in capacity_interpretation_conflicts(response, [vars(r) for r in results], topic):
                issues.append(GroundingIssue("unverified_external_action_claim",
                    f"Measured available {topic} capacity is {conflict['expected_capacity']}; this headroom interpretation contradicts the receipt.",
                    conflict["claim"], (f"system_metric:{topic}",), tuple(sorted(evidence))))
            comparisons = percentage_comparisons(response, [vars(r) for r in results], topic, include_spans=True)
            for c in comparisons:
                if c["matches"] is False:
                    issues.append(GroundingIssue("unverified_external_action_claim",
                        f"The claimed {c['label']} percentage differs from the measured receipt. Use raw measured quantities, not rounded display sizes.",
                        c["claim"], (f"system_metric:{topic}",), tuple(sorted(evidence))))
    return issues


def _matching_prior_receipts(claim: str, receipts) -> list:
    matched = []
    for result in receipts:
        if not result.success:
            continue
        metadata = result.metadata or {}
        path = metadata.get("arguments", {}).get("path") or metadata.get("path", "")
        if not path:
            continue
        name = str(path).rsplit("/", 1)[-1]
        if name in claim or re.search(r"\b" + re.escape(name.rsplit(".", 1)[0]) + r" file\b", claim, re.I):
            matched.append(result)
    return matched


def extract_action_claims(response: str) -> list[ActionClaim]:
    """Extract generic first-person or status claims about external actions."""
    if not response:
        return []
    claims: list[ActionClaim] = []
    seen: set[str] = set()
    patterns = (
        _FIRST_PERSON_ACTION_RE,
        _STATUS_CLAIM_RE,
        _REGISTRY_STATE_CLAIM_RE,
        _SYSTEM_METRIC_CLAIM_RE,
        _PERMISSION_EXECUTION_RE,
        _EXPLICIT_EXECUTION_RE,
    )
    # Negation is scoped to its clause. A disclaimer in the next sentence
    # cannot erase an affirmative claim; a positive later clause survives too.
    for clause in re.split(r"(?<=[.!?])\s+|\n+|[;—]|\bbut\b|\band\s+(?=(?:i|we|the (?:file|command|tool|shell))\b)", response, flags=re.I):
        for pattern in patterns:
            for match in pattern.finditer(clause):
                text = match.group(0)
                if _seen_claim_text(text, seen) or not (
                    _EXTERNAL_OBJECT_RE.search(text) or pattern is _EXPLICIT_EXECUTION_RE
                ):
                    continue
                # A negated description of a result is not a claim it happened.
                # Scope this to the matched assertion: an adjacent denial must
                # not excuse a later affirmative action in the same sentence.
                prefix = clause[:match.start()]
                assertion = re.split(r",|\band\b", prefix, flags=re.I)[-1] + text
                if (re.search(r'\bnot\s+(?:that\s+)?["“\']?(?:the\s+)?$', prefix, re.I)
                        or _is_negated_or_evidence_warning(assertion)):
                    continue
                categories = _claim_categories(text)
                if not categories:
                    continue
                seen.add(text)
                claims.append(ActionClaim(text=text, categories=categories, context=clause))
    # Completion headers can bind a registry claim in the following sentence
    # ("Added. ... capability ..."). Keep that positive evidence without
    # borrowing a later sentence's negation for ordinary action claims.
    for match in _REGISTRY_STATE_CLAIM_RE.finditer(response):
        if not re.search(r"[.!?]\s", match[0]):
            continue
        text = _compact(match[0])
        if not _seen_claim_text(text, seen) and not _is_negated_or_evidence_warning(text):
            categories = _claim_categories(text)
            if categories:
                claims.append(ActionClaim(text, categories, text))
                seen.add(text)
    return claims


_SUCCESS_TONE_RE = re.compile(
    r"\b(?:succeeded|success|complete|completed|finished|done|"
    r"working|ready|active|running|live|installed|downloaded|fixed|"
    r"patched|initialized|activated|set\s+up|happened|worked)\b",
    re.IGNORECASE,
)
_FAILURE_TONE_RE = re.compile(
    r"\b(?:failed|failure|error|errored|crashed|hung|hanging|stuck|"
    r"deadlock|aborted|cancell?ed|timeout|timed\s+out|did\s+not|didn't|"
    r"never|cannot|can't|won't|unable\s+to)\b",
    re.IGNORECASE,
)


def _claim_categories(text: str) -> set[str]:
    lower = text.lower()
    categories: set[str] = set()
    # Success tone fires on any success language. Mixed phrases like
    # "succeeded where the earlier ones failed" still assert success and
    # must be grounded by an actual successful tool call.
    success_tone = bool(_SUCCESS_TONE_RE.search(lower))
    if _PERMISSION_EXECUTION_RE.search(text):
        categories.update({"execute", "execute_success"})
    if re.search(r"\b(read|reading|checked?|checking|opened?|opening|searched?|searching|listed?|listing|inspected?|inspecting|fetched?|fetching|loaded?|loading)\b", lower):
        categories.add("read")
    if re.search(r"\b(wrote|write|writing|created?|creating|saved?|saving|installed?|installing|downloaded?|downloading|cloned?|cloning|deleted?|deleting|removed?|removing)\b", lower):
        categories.add("write")
        if re.search(r"\b(?:wrote|created|saved|deleted|removed)\b", lower) and re.search(r"\b(?:file|folder|directory)\b", lower):
            categories.add("write_success")
    if re.search(
        r"\b(run|ran|running|executed?|executing|shell|terminal|command|process|package|"
        r"dependency|forced?|forcing|initialized?|initializing|patched?|patching|"
        r"killed?|killing|terminated?|terminating|fixed?|fixing|"
        r"started?|starting|launched?|launching|sourced?|sourcing|"
        r"activated?|activating|built?|building|compiled?|compiling|"
        r"set\s+up|setting\s+up|environment|venv|conda)\b",
        lower,
    ):
        categories.add("execute")
        if success_tone:
            categories.add("execute_success")
    if re.search(r"\bran (?:a |the )?(?:check|inspection|scan)\b", lower) and not re.search(
        r"\b(?:shell|terminal|command|process|script)\b", lower
    ):
        categories.difference_update({"execute", "execute_success"})
        categories.add("read")
    if re.search(r"\b(skill|capability|capabilities|module|integration|registry|toolset|skillset)\b", lower) and re.search(
        r"\b(created?|creating|imported?|importing|enabled?|enabling|"
        r"activated?|activating|installed?|installing|integrated?|integrating|"
        r"registered|registering|added|adding|deleted?|deleting|removed?|removing|"
        r"made\s+permanent|permanent|"
        r"persistent|persisted|part\s+of|available\s+in|done|completed)\b",
        lower,
    ):
        categories.add("registry_write")
    if re.search(r"\bregistry\b", lower) and re.search(r"\b(?:wrote|write|writing|saved?|saving)\b", lower):
        categories.add("registry_write")
    if re.search(r"\b(?:learned|ingested|digested)\b", lower):
        categories.add("learning")
    if re.search(
        r"\b(memory|ram|cpu|disk|storage|system\s+memory|system\s+metric|"
        r"utili[sz]ation)\b",
        lower,
    ):
        categories.add("system_metric")
        if _SYSTEM_METRIC_CLAIM_RE.search(text):
            compared = re.sub(r"\bthan (?:the )?(?:cpu|memory|ram|disk)(?: situation| was)\b", "", text, flags=re.I)
            categories.update(_metric_topics(compared))
    return categories


def _metric_topics(text: str) -> set[str]:
    return {f"system_metric:{topic}" for topic, pattern in (
        ("cpu", r"\bcpu\b"),
        ("memory", r"\b(?:memory|ram|mem|memtotal|memavailable)\b"),
        ("disk", r"\b(?:disk|storage|filesystem)\b"),
    ) if re.search(pattern, text, re.I)}


def _tool_evidence_categories(tool_trace: Any | None) -> set[str]:
    """Return categories of executed tool results.

    Successful execute-category results also add ``execute_success`` so the
    verifier can distinguish "the command ran" (any shell call) from "the
    command succeeded" (shell call with non-zero return path). Failed shell
    calls still ground neutral execute claims, so the model can faithfully
    report failure without being challenged.
    """
    executed = list(getattr(tool_trace, "executed_results", []) or [])
    if not executed:
        return set()
    categories: set[str] = set()
    for result in executed:
        if blocked_before_execution(result):
            continue
        success = bool(getattr(result, "success", False))
        tool_name = str(getattr(result, "tool_name", "") or "").lower()
        if tool_name == "learning.intake":
            if success:
                categories.add("learning")
        elif tool_name.startswith(("fs.read", "fs.list", "fs.search", "fs.glob")):
            categories.add("read")
        elif tool_name.startswith(("web.", "browser.")):
            categories.add("read")
        elif tool_name.startswith("system."):
            categories.add("read")
            if success and str(getattr(result, "output", "") or "").strip():
                categories.add("system_metric")
                categories.update(_metric_topics(str(result.output)))
                topic = {"system.disk_usage": "disk", "system.memory_usage": "memory"}.get(tool_name)
                if topic:
                    categories.add(f"system_metric:{topic}")
        elif tool_name.startswith(("fs.write", "fs.delete")):
            categories.add("write")
            if success:
                categories.add("write_success")
        elif tool_name.startswith((
            "skills.create",
            "skills.import",
            "skills.enable",
            "skills.disable",
            "skills.delete",
        )):
            if success:
                categories.add("registry_write")
        elif tool_name.startswith(("skills.list", "skills.audit")):
            categories.add("read")
        elif tool_name.startswith("shell."):
            categories.update({"read", "write", "execute"})
            if success:
                categories.add("write_success")
                categories.add("execute_success")
                if re.search(r"%?Cpu\(s\)|CPU (?:usage|utilization)|RAM (?:total|used)|"
                             r"memory (?:usage|used|total)|Filesystem", str(getattr(result, "output", "")), re.I):
                    categories.add("system_metric")
                    categories.update(_metric_topics(str(result.output)))
        else:
            categories.add("tool")
            if success:
                categories.add("tool_success")
    return categories


def _missing_categories(required: set[str], evidence: set[str]) -> set[str]:
    missing: set[str] = set()
    for category in required:
        if category == "read" and evidence.intersection({"read", "execute", "tool"}):
            continue
        if category == "write" and evidence.intersection({"write", "execute", "tool"}):
            continue
        if category == "write_success" and "write_success" in evidence:
            continue
        if category == "execute" and evidence.intersection({"execute", "tool"}):
            continue
        if category == "registry_write" and evidence.intersection({"registry_write"}):
            continue
        if category == "learning" and "learning" in evidence:
            continue
        if category == "system_metric" and "system_metric" in evidence:
            continue
        if category.startswith("system_metric:") and category in evidence:
            continue
        if category == "execute_success" and evidence.intersection(
            {"execute_success", "tool_success"}
        ):
            continue
        missing.add(category)
    return missing


def _compact(text: str) -> str:
    return " ".join(str(text or "").split())


def _seen_claim_text(text: str, seen: set[str]) -> bool:
    normalized = text.lower()
    return any(
        normalized == prior.lower()
        or normalized in prior.lower()
        or prior.lower() in normalized
        for prior in seen
    )


def _is_negated_or_evidence_warning(text: str) -> bool:
    lower = text.lower()
    if reports_nonexecution(lower):
        return True
    if re.search(
        r"\bi (?:can't|cannot) (?:read|measure|obtain) (?:cpu|ram|memory|disk)(?: usage)?\b", lower,
    ) and not re.search(r"\d+(?:\.\d+)?\s*(?:%|[kmgt]b\b)", lower):
        return True
    if "no tool execution result" in lower:
        return True
    if "without tool execution result" in lower:
        return True
    if "cannot verify" in lower or "can't verify" in lower:
        return True
    if _is_non_assistant_actor_statement(lower):
        return True
    if re.search(
        r"\b(?:must|should|needs?\s+to|has\s+to|can|could)\s+be\s+"
        r"(?:created|imported|enabled|activated|installed|integrated|"
        r"registered|added)\b",
        lower,
    ):
        return True
    if re.search(
        r"\bif\s+you\b.{0,60}\b(?:create|import|enable|activate|install|"
        r"integrate|register|add)\b",
        lower,
    ):
        return True
    if re.search(r"\b(?:i|we)\s+(?:did\s+not|didn't|do\s+not|don't|will not|won't)\b", lower):
        return True
    if re.search(r"\b(?:didn't|did not|never)\s+(?:actually\s+)?(?:run|execute)\b", lower):
        return True
    if re.search(r"\bno\b[^.!?\n]{0,65}\b(?:file|creation|directory|listing|terminal|output|execution)\b", lower):
        return True
    if re.search(r"\bno\b.{0,50}\b(?:tool|result|inspection|read|write|lookup|execution)\b.{0,30}\b(?:ran|exists?|available)\b", lower):
        return True
    if re.search(r"\b(?:not|never)\b.{0,20}" + _ACTION_VERBS_RE.pattern, lower):
        return True
    if (
        re.search(r"\b(?:draft|example|sample|template|proposed)\b", lower)
        and re.search(r"\b(?:skill|capability|module|integration)\b", lower)
    ):
        return True
    return False


def _is_non_assistant_actor_statement(lower: str) -> bool:
    if re.search(r"\b(?:i|we|my|our|n[ūu]r)\b.{0,40}" + _ACTION_VERBS_RE.pattern, lower):
        return False
    return bool(
        re.search(
            r"\b(?:the\s+)?(?:user|operator|admin|administrator|someone|"
            r"they|he|she)\b.{0,40}" + _ACTION_VERBS_RE.pattern,
            lower,
        )
    )


# ---------------------------------------------------------------------------
# Formerly core/metric_claims.py
# ---------------------------------------------------------------------------

_LABEL = re.compile(r"\b(available|free|used|reclaimable|in\s+use)\b", re.I)
_CLAUSE = re.compile(r"[!?;,\n—–]|(?<!\d)\.|\.(?!\d)|\b(?:and|but|leaving)\b", re.I)


def _percentage_label(text: str, match: re.Match) -> str:
    # Never borrow a label from another clause or across another percentage.
    before = _CLAUSE.split(text[max(0, match.start() - 60):match.start()])[-1]
    before = before.rsplit("%", 1)[-1]
    after = _CLAUSE.split(text[match.end():match.end() + 60])[0].split("%", 1)[0]
    following = _LABEL.search(after)
    preceding = list(_LABEL.finditer(before))
    # Postfix labels govern their number ("23% in use", "77% of RAM
    # available"). Do not cross a new quantity to attach its label instead.
    if (following and not re.search(r"\d", after[:following.start()])
            and not re.match(r"\s*(?::|=|is|at)?\s*\d", after[following.end():], re.I)):
        label = following[1].lower()
    elif preceding:
        label = preceding[-1][1].lower()
    else:
        label = "used"
    return {"reclaimable": "available", "in use": "used"}.get(" ".join(label.split()), label)


def percentage_comparisons(text: str, results, topic: str, *, include_spans=False) -> list[dict]:
    measured: dict[str, list[float]] = {"used": [], "free": [], "available": []}
    for result in results:
        metadata = result.get("metadata", {})
        total = metadata.get("total_bytes", 0)
        for label in measured:
            value = metadata.get(f"{label}_percent")
            if value is None and total and isinstance(metadata.get(f"{label}_bytes"), (int, float)):
                value = 100 * metadata[f"{label}_bytes"] / total
            if isinstance(value, (int, float)):
                measured[label].append(value)
        if not measured["used"]:
            measured["used"].extend(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", result.get("output", "")))
    comparisons = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        after = text[match.end():match.end() + 35].lstrip(" *_—-: ")
        before = text[max(0, match.start() - 45):match.start()]
        # A rejected number or a named process's historical share is not an
        # assertion about the current aggregate. Review still sees that prose.
        if re.search(r"\b(?:not|rather than|instead of)\s+(?:the\s+)?$", before, re.I):
            continue
        if (re.match(r"\s*(?:was|had been)\b", after, re.I)
                and re.search(r"\b(?:process|resident|share|earlier|previous)\b", after, re.I)):
            continue
        kind = _percentage_label(text, match)
        values = list(measured[kind])
        if topic == "disk":
            if kind == "available" and not values:
                # The disk inspection's free_bytes comes from df's Avail.
                values = list(measured["free"])
            elif kind == "free":
                # "Free" can also mean the complement of occupied space,
                # including reserved blocks. Explicit "available" cannot.
                values.extend(100 - used for used in measured["used"])
        digits = len(match[1].split(".")[1]) if "." in match[1] else 0
        tolerance = 0.5 * 10 ** -digits + 0.005
        claimed = float(match[1])
        comparison = {"claimed": claimed, "label": kind, "measured": values,
            "matches": any(abs(claimed - value) <= tolerance for value in values) if values else None}
        if include_spans:
            delimiters = list(_CLAUSE.finditer(text))
            start = max((m.end() for m in delimiters if m.end() <= match.start()), default=0)
            end = min((m.start() for m in delimiters if m.start() >= match.end()), default=len(text))
            comparison["claim"] = text[start:end].strip()
        comparisons.append(comparison)
    return comparisons


def cpu_interpretation_conflicts(text: str, results) -> list[dict]:
    """Check extreme aggregate usage interpretations, keeping idle separate.

    Middle readings, process shares, capacity judgments and unfamiliar paraphrases
    remain semantic questions. An idle percentage's own magnitude is valid.
    """
    idle = cpu_idle_percentages(results)
    conflicts = cpu_percentage_conflicts(text, idle)
    if not idle or not (all(v >= 90 for v in idle) or all(v <= 10 for v in idle)):
        return conflicts
    expected = "low" if all(v >= 90 for v in idle) else "high"
    subject = ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if re.search(r"\b(?:process|per[- ]core|single[- ]core|core\s+\d+|load\s+average)\b", sentence, re.I):
            subject = ""
            continue
        mentions = list(re.finditer(r"\b(?:cpu\s+(?:usage|utili[sz]ation)|idle(?:\s+time)?)\b", sentence, re.I))
        for judgment in re.finditer(r"\b(high|low)\b", sentence, re.I):
            governing = [m for m in mentions if m.end() <= judgment.start()]
            current = governing[-1][0].lower() if governing else subject if re.match(
                r"\s*(?:that's|that is|this is|it's|it is)\b", sentence, re.I) else ""
            if current.startswith("cpu") and judgment[1].lower() != expected and not re.search(
                    r"\b(?:not|isn't|wasn't|never)\b", sentence[:judgment.start()], re.I):
                conflicts.append({"claim": sentence, "idle_percent": idle, "expected_usage": expected})
        if mentions:
            subject = mentions[0][0].lower()
    return conflicts


def cpu_idle_percentages(results) -> list[float]:
    """Read aggregate CPU idle from observed summary lines, never process rows."""
    idle = []
    for result in results:
        if not result.get("success"):
            continue
        for line in result.get("output", "").splitlines():
            if re.search(r"%?Cpu\(s\)|\bCPU\b", line, re.I):
                match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:id|idle)\b", line, re.I)
                if match:
                    idle.append(float(match[1]))
    return idle


def cpu_measurement_complete(text: str, measured_idle) -> bool:
    """Find the measured busy percentage, independently of its phrasing.

    Only attribution is linguistic: idle, other metrics and process/core shares
    cannot supply an aggregate answer. Receipt arithmetic and the precision of
    the reported number determine whether the remaining percentage is complete.
    """
    expected = [100 - idle for idle in measured_idle]
    if not expected:
        return False
    quantity = re.compile(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
    process_share = re.compile(
        r"\b(?:process(?:es)?|threads?|PID|per[- ]core|single[- ]core|(?:a|one|single|\d+) cores?|"
        r"core\s*\d+)\b", re.I)
    other_metric = re.compile(r"\b(?:load average|RAM|memory|disk|swap|storage|filesystem)\b", re.I)
    subject = re.compile(
        r"^\s*(?:the\s+)?(?:`([^`]+)`|([\w.-]+))\s+(?:(?:is|was)\s+)?"
        r"(?:using|consuming|pulling|chewing through|at|accounts? for)\b", re.I)
    aggregate = re.compile(r"\b(?:overall|aggregate|total|system[- ]wide|host)\b", re.I)
    aggregate_subjects = {"cpu", "processor", "system", "host", "usage", "utilization", "utilisation", "busy"}
    boundaries = re.compile(_CLAUSE.pattern + r"|\b(?:with|while|so)\b", re.I)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        # An explicit trailing attribution still binds across a comma.
        sentence = re.sub(r",\s*(?=(?:for|of|by|from)\s+(?:the\s+)?(?:process|PID|core)\b)", " ", sentence, flags=re.I)
        process_bound = False
        for clause in boundaries.split(sentence):
            consumers = list(subject.finditer(clause))
            named_consumer = any((m[1] or m[2]).casefold() not in aggregate_subjects for m in consumers)
            if aggregate.search(clause):
                process_bound = False
            if process_share.search(clause) or named_consumer:
                process_bound = not quantity.search(clause)
                continue
            if process_bound or other_metric.search(clause):
                continue  # E.g. "For process python3, CPU was 1.1%."
            for match in quantity.finditer(clause):
                before = clause[:match.start()].rsplit("%", 1)[-1]
                before = re.split(r"\bpercent\b", before, flags=re.I)[-1]
                after = quantity.split(clause[match.end():], maxsplit=1)[0]
                if (re.search(r"\bidle\b", before, re.I)
                        or re.search(r"\bidle\b", after, re.I)
                        or re.search(r"\b(?:not|rather than|instead of)\s+(?:the\s+)?$", before, re.I)):
                    continue
                claimed = float(match[1])
                digits = len(match[1].split(".")[1]) if "." in match[1] else 0
                tolerance = 0.5 * 10 ** -digits + 1e-9
                if 0 <= claimed <= 100 and any(abs(claimed - busy) <= tolerance for busy in expected):
                    return True
    return False


def cpu_usage_claims(text: str):
    """Yield numeric aggregate busy claims with their original sentence spans."""
    quantity = r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)"
    qualifiers = r"(?:(?:is|at|currently|around|about|approximately|roughly|using|consuming|just under|just over)\s+|[:=]\s*)*"
    patterns = (
        re.compile(r"\b(?:CPU(?:\s+(?:usage|utili[sz]ation))?|"
                   r"(?:overall|aggregate|total)\s+(?:usage|utili[sz]ation|busy(?:\s+figure)?))\s*"
                   + qualifiers + quantity + r"(?:\s*(idle|busy|used))?", re.I),
        re.compile(quantity + r"\s*(?:(idle|busy|used)\s+)?"
                   r"(?:(?:of|the|overall|aggregate|total|system)\s+)*(?:CPU|processor)\b", re.I),
        re.compile(quantity + r"\s*(busy)\b", re.I),
        re.compile(quantity + r"\s*(in\s+use)\b", re.I),
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        matches = [m for pattern in patterns for m in pattern.finditer(sentence)]
        seen = set()
        for match in matches:
            prefix = re.split(r"[,;]|\b(?:and|with|while|but)\b", sentence[:match.start()], flags=re.I)[-1]
            if match[2] and re.fullmatch(r"in\s+use", match[2], re.I):
                inherited = re.split(r"\bidle\b", prefix, flags=re.I)[-1]
                if (not re.search(r"\b(?:CPU|processor)\b", sentence[:match.start()], re.I)
                        or not re.fullmatch(r"\s*(?:so\s+)?" + qualifiers, inherited, re.I)):
                    continue  # A new named subject cannot inherit the earlier CPU label.
            if re.search(r"\b(?:process|per[- ]core|single[- ]core|core\s+\d+|load\s+average|share|RAM|memory|disk)\b", prefix, re.I):
                continue
            # A named consumer is a process subject, even without the word
            # "process". Its share can exceed 100% on a multicore host.
            consumer = re.search(r"(?:`([^`]+)`|\b(\w+))(?:,\s*|\s+(?:is\s+)?)"
                                 r"(?:using|consuming|pulling|chewing through)\s+(?:(?:about|around)\s+)?$",
                                 sentence[:match.start()], re.I)
            if consumer and (consumer[1] or consumer[2]).casefold() not in {"cpu", "processor", "system", "host"}:
                continue
            if match[2] and match[2].lower() == "idle":
                continue
            if re.match(r"\s*(?:is\s+)?idle\b", sentence[match.end():], re.I):
                continue
            if re.search(r"\b(?:not|rather than|instead of)\s+(?:the\s+)?$", prefix, re.I):
                continue
            if match.span(1) in seen:
                continue
            seen.add(match.span(1))
            yield sentence, match


def cpu_percentage_conflicts(text: str, measured_idle=()) -> list[dict]:
    """Aggregate busy time is the complement of idle, never a process share.

    Compare labeled aggregate claims with receipts and with idle in the same
    sentence. The latter also catches a self-contradiction without saved receipts.
    """
    conflicts = []
    for sentence, match in cpu_usage_claims(text):
        reported_idle = [float(m[1]) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)\s*idle\b", sentence, re.I)]
        claimed = float(match[1])
        digits = len(match[1].split(".")[1]) if "." in match[1] else 0
        # top reports tenths; allow receipt and expression rounding, without
        # the large tolerance that could validate a process share as total.
        tolerance = 0.5 * 10 ** -digits + 0.055
        sources = [values for values in (list(measured_idle), reported_idle) if values]
        if any(all(abs(claimed - (100 - value)) > tolerance for value in values) for values in sources):
            conflicts.append({"claim": sentence, "claimed_usage": claimed,
                "idle_percent": list(measured_idle) or reported_idle,
                "expected_usage": [100 - value for value in list(measured_idle) or reported_idle]})
    return conflicts


def capacity_interpretation_conflicts(text: str, results, topic: str) -> list[dict]:
    """Compare headroom assertions with available capacity, never raw free RAM.

    Keep middle readings and workload sufficiency judgments semantic. Legacy
    rounded tables can establish only clear extremes, not exact percentages.
    """
    available = []
    tool = f"system.{topic}_usage"
    for result in results:
        if not result.get("success") or result.get("tool_name") != tool:
            continue
        metadata = result.get("metadata", {})
        total = metadata.get("total_bytes")
        value = metadata.get("available_percent")
        if value is None and total:
            count = metadata.get("available_bytes" if topic == "memory" else "free_bytes")
            if isinstance(count, (int, float)):
                value = 100 * count / total
        if value is None and topic == "memory":
            table = re.search(r"^RAM\s+(\S+)\s+\S+\s+\S+\s+(\S+)", result.get("output", ""), re.M)
            def size(raw):
                match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", raw, re.I)
                return float(match[1]) * 1024 ** " KMGT".index(match[2].upper() or " ") if match else None
            if table:
                total, count = size(table[1]), size(table[2])
                if total and count is not None:
                    value = 100 * count / total
        if isinstance(value, (int, float)):
            available.append(value)
    if not available or not (all(v >= 70 for v in available) or all(v <= 15 for v in available)):
        return []
    expected = "ample" if all(v >= 70 for v in available) else "limited"
    target = r"\b(?:RAM|memory)\b" if topic == "memory" else r"\b(?:disk|storage|filesystem|drive)\b"
    judgment = re.compile(r"\b(?P<amount>not (?:much|a lot of|enough)|little|limited|plenty of|ample|lots of) "
                          r"(?:remaining |available )?(?:room|headroom|space|capacity)\b", re.I)
    conflicts, active = [], False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        # Relative sufficiency depends on the requested workload, not just a
        # percentage of capacity. Leave those judgments to semantic review.
        if re.search(r"\bfor\b", sentence, re.I):
            continue
        if re.search(r"\b(?:process|per[- ]core|CPU|swap)\b", sentence, re.I):
            active = False
            continue
        mentioned = bool(re.search(target, sentence, re.I))
        active = mentioned or active
        for clause in _CLAUSE.split(sentence):
            for match in judgment.finditer(clause):
                prefix = clause[:match.start()].strip()
                if not active or (prefix and not re.search(target, prefix, re.I) and not re.fullmatch(
                        r"(?:there(?:'s| is)|that(?:'s| is)|it(?:'s| is))", prefix, re.I)):
                    continue
                amount = match["amount"].lower()
                asserted = "ample" if amount in {"plenty of", "ample", "lots of"} else "limited"
                if asserted != expected:
                    conflicts.append({"claim": sentence, "available_percent": available,
                                      "expected_capacity": expected})
    return conflicts


# ---------------------------------------------------------------------------
# Formerly core/system_inspection.py::addresses_inspection
# ---------------------------------------------------------------------------

def addresses_inspection(response: str, kind: str) -> bool:
    """Recognize an observation's topic, including prose without numeric digits.

    This only checks relevance. Receipt integrity and factual accuracy remain
    separate checks; the existence of any numbers is not sufficient grounding.
    """
    if kind == "cpu":
        if any(cpu_usage_claims(response)):
            return True
    topics = {
        "cpu": r"\b(?:cpu|processor|utili[sz]ation)\b",
        "memory": r"\b(?:ram|memory|mem|memtotal|memavailable)\b",
        "disk": r"\b(?:disk|storage|filesystem|drive)\b",
    }
    observation = re.search(r"\d|%|\b(?:usage|utili[sz]ation|used|free|available|total|idle|busy|in use|consum\w*)\b", response, re.I)
    # Labeled capacity quantities can answer a resolved RAM/disk request without
    # repeating its name. Topic relevance does not establish numeric accuracy.
    capacity = kind in {"memory", "disk"} and re.search(
        r"\b\d+(?:\.\d+)?\s*[kmgtpe]i?b?\s+(?:free|used|available|in\s+use)\b", response, re.I)
    return bool((re.search(topics.get(kind, r"(?!)"), response, re.I) or capacity) and observation)


# ---------------------------------------------------------------------------
# Formerly core/llm_content.py::requests_literal_text and core/response_language.py
# ---------------------------------------------------------------------------

_LITERAL_REQUEST = re.compile(
    r"\b(?:quote|verbatim|translate|rewrite|proofread|transcribe)\b"
    r"|\b(?:repeat|say) (?:that|it|this|exactly|after me)\b", re.I,
)

def requests_literal_text(message: str) -> bool:
    """Copying is appropriate when the user actually requests text reproduction."""
    return bool(_LITERAL_REQUEST.search(message))


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

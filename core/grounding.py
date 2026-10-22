"""Ground response claims about external actions in actual tool evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any
from types import SimpleNamespace


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


def blocked_before_execution(result) -> bool:
    """A policy rejection with no output/effects proves no execution occurred.

    Failure alone is insufficient: a started command can fail after mutating.
    Accept the legacy allowlist marker as well as explicit modern provenance.
    """
    data = result if isinstance(result, dict) else vars(result)
    metadata = data.get("metadata") or {}
    return (data.get("success") is False and not data.get("output")
            and data.get("side_effect_summary", "").strip().lower() in {"", "none"}
            and (metadata.get("execution_started") is False
                 or metadata.get("policy") == "shell_command_allowlist"
                 or metadata.get("grant_denied") is True))


def action_decision_evidence(decisions) -> list[dict]:
    """Project historical file choices without declined contents or optimizer data.

    Choices explain whether work was declined; a grant alone never establishes
    execution, disappearance, or the cause of a later filesystem error.
    """
    evidence = []
    for decision in decisions:
        for choice in decision.get("components", []):
            component = choice.get("component", {})
            tool = component.get("requested_tool", "")
            arguments = dict(component.get("requested_arguments") or [])
            if tool.startswith("fs.") and arguments.get("path"):
                evidence.append({"turn_id": decision["turn_id"], "component_id": component["id"],
                    "tool": tool, "path": arguments["path"], "granted": choice["granted"]})
    return evidence


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
    from core.metric_claims import capacity_interpretation_conflicts, cpu_interpretation_conflicts, percentage_comparisons
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


def system_metric_observation_response(
    user_message: str,
    *,
    tool_trace: Any | None,
) -> str | None:
    """Return direct observed system metric output for metric questions."""
    if not _looks_system_metric_request(user_message):
        return None
    for result in _external_or_system_results(tool_trace):
        if not bool(getattr(result, "success", False)):
            continue
        tool_name = str(getattr(result, "tool_name", "") or "")
        if tool_name != "system.memory_usage":
            continue
        output = str(getattr(result, "output", "") or "").strip()
        if output:
            return output
    return None


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
    for clause in re.split(r"(?<=[.!?])\s+|\n+|[;—]|\bbut\b", response, flags=re.I):
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


def grounding_correction_response(
    issues: list[GroundingIssue] | None = None,
    *,
    tool_trace: Any | None = None,
) -> str:
    """Return a generic correction when output overclaims external action."""
    registry_write = _successful_registry_write(tool_trace)
    if registry_write:
        detail = registry_write.get("detail") or "skill registry updated"
        return (
            f"{detail}. I did not perform any additional external actions "
            "claimed in the draft response unless matching Tool Execution "
            "Results show them. Check Admin > Skills for the imported skill."
        )
    if any(
        "registry_write" in issue.required_categories
        for issue in (issues or [])
    ):
        return (
            "I cannot claim a durable skill or registry change from this turn "
            "because no matching skill-registry write ran."
        )
    required = set()
    for issue in issues or []:
        required.update(issue.required_categories)
    if "system_metric" in required:
        return "I cannot verify that system metric because no system inspection ran this turn."
    if "write" in required:
        return "I cannot claim that file, package, or registry state changed because no matching write action ran this turn."
    if "execute" in required or "execute_success" in required:
        return "I cannot verify that command or process result because no matching execution result ran this turn."
    if "read" in required:
        return "I cannot verify that inspection result because no matching read or lookup ran this turn."
    return (
        "I cannot claim or verify that external action from this turn because no matching tool result exists."
    )


def strip_unsupported_metric_claims(response: str, *, tool_trace: Any | None) -> str:
    """Remove unsupported metric sentences without replacing the social reply.

    Check sentences independently: a disclaimer or negation elsewhere in the
    reply must not conceal an unsupported observation. Empty output asks the
    caller to regenerate a relevant reply once.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", response.strip())
    kept = []
    for sentence in sentences:
        issues = verify_response_grounding(sentence, tool_trace=tool_trace)
        if any("system_metric" in issue.required_categories for issue in issues):
            continue
        kept.append(sentence)
    return " ".join(kept)


def is_system_metric_request(text: str) -> bool:
    return _looks_system_metric_request(text)


def _successful_registry_write(tool_trace: Any | None) -> dict[str, str] | None:
    executed = list(getattr(tool_trace, "executed_results", []) or [])
    for result in executed:
        tool_name = str(getattr(result, "tool_name", "") or "")
        if not tool_name.startswith((
            "skills.create",
            "skills.import",
            "skills.enable",
            "skills.disable",
            "skills.delete",
        )):
            continue
        if not bool(getattr(result, "success", False)):
            continue
        metadata = getattr(result, "metadata", {}) or {}
        skill_id = str(metadata.get("skill_id") or metadata.get("name") or "").strip()
        enabled = metadata.get("enabled")
        status = str(metadata.get("status") or "").strip()
        summary = str(getattr(result, "side_effect_summary", "") or "").strip()
        if skill_id:
            detail = f"Skill registry updated: {skill_id}"
            if enabled is not None:
                detail += f" enabled={str(bool(enabled)).lower()}"
            if status:
                detail += f" status={status}"
        else:
            detail = summary or "Skill registry updated"
        return {"detail": detail}
    return None


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


def has_current_metric_evidence(topic: str, tool_trace: Any | None) -> bool:
    """Whether this turn actually observed the requested metric topic."""
    return f"system_metric:{topic}" in _tool_evidence_categories(tool_trace)


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
    if re.search(r"\b(?:i|we)\s+(?:did\s+not|didn't|do\s+not|don't)\b", lower):
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


def _looks_system_metric_request(user_message: str) -> bool:
    text = user_message or ""
    if re.search(
        r"\b(?:stop|don't|do not|enough|no need)\b.{0,40}\b(?:memory|ram|cpu|metrics?)\b"
        r"|\bmemory (?:of|about)\b", text, re.I,
    ):
        return False
    metric = re.search(r"\b(?:memory|ram|cpu|disk|storage|utili[sz]ation|usage)\b", text, re.I)
    request = re.search(
        r"\b(?:what|how much|how many|show|check|tell|give|report|current)\b", text, re.I,
    )
    return bool(metric and (request or text.strip().lower() in {"cpu", "ram", "memory"}))


def _external_or_system_results(tool_trace: Any | None) -> list[Any]:
    executed = list(getattr(tool_trace, "executed_results", []) or [])
    return [
        result for result in executed
        if str(getattr(result, "tool_name", "") or "").startswith(("web.", "browser.", "system."))
    ]

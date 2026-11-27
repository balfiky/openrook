"""What a reply states against what was actually executed: structure only.

Whether a sentence claims an action or a result is a question of meaning, and
the model reviewer answers it (``core/review.py``). This module compares what
can be compared without reading any language:

* which receipt belongs to which component of the decision;
* whether a percentage a reply states is a figure the receipts carry;
* whether a delimited output block in a reply is text some receipt returned;
* whether a span a finding disputes is nothing but receipt content.

It cannot tell which action a sentence asserts, so naming a file that a tool
touched verifies nothing about what was done to it. It matches no vocabulary
and decides nothing about how a reply is worded.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from core.perception import clause_ranges, is_unspaced

# Receipts state a measured or derived percentage under a metadata key that
# ends like this (``used_percent``, ``busy_percent``): the runtime's own naming.
PERCENT_KEY = "_percent"
# A numeral with an optional decimal part, and the signs that mark a percentage.
_NUMERAL = re.compile(r"(?<![\d.,_])[-+]?\d+(?:[.,\u066b]\d+)?(?![.,\u066b]?\d)")
_PERCENTAGE = re.compile(r"(?<![\d.,_])(\d+(?:[.,\u066b]\d+)?)\s*[%\u066a\uff05\ufe6a]")
# Text a reply sets apart as literal: a block fenced with backticks or tildes,
# and a line that is nothing but one backtick span.
_BLOCK = re.compile(r"(```|~~~)[^\n]*\n(.*?)(?:\1|\Z)", re.S)
_CODE_LINE = re.compile(r"(?m)^[ \t]*`([^`\n]+)`[ \t]*$")
# What the reply's writer is shown of a measurement or file receipt, by key.
EVIDENCE_KEYS = ("tool", "success", "error", "path", "measurements", "observed_output", "output_truncated")
MEASUREMENT_KEYS = (PERCENT_KEY, "_bytes")


def blocked_before_execution(result) -> bool:
    """A policy rejection with no output/effects proves no execution occurred.

    Failure alone is insufficient: a started command can fail after mutating.
    Accept the legacy allowlist marker as well as explicit modern provenance.
    """
    data = result if isinstance(result, dict) else vars(result)
    metadata = data.get("metadata") or {}
    return (data.get("success") is False and not data.get("output")
            and not has_side_effects(data)
            and (metadata.get("execution_started") is False
                 or metadata.get("policy") == "shell_command_allowlist"
                 or metadata.get("grant_denied") is True))


def has_side_effects(result) -> bool:
    """Whether a receipt says the execution changed something."""
    data = result if isinstance(result, dict) else vars(result)
    return str(data.get("side_effect_summary") or "").strip().lower() not in {"", "none"}


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


def component_receipt_indices(decision, choice, results) -> list[int]:
    """Bind current execution to its frozen turn, scope, tool and operands."""
    component = choice.component
    indices = []
    for index, result in enumerate(results):
        metadata = result.metadata or {}
        if (result.tool_name != component.requested_tool
                or metadata.get("component_id") != component.id
                or metadata.get("decision_turn_id") != decision.turn_id
                or metadata.get("decision_policy") != decision.policy_version
                or metadata.get("execution_scope") != decision.scope):
            continue
        arguments = metadata.get("arguments") or {}
        if set(arguments) - dict(component.requested_arguments).keys():
            continue
        if not all(arguments.get(key) == value if key in arguments else (
                key == "content" and result.tool_name == "fs.write_file"
                and (result.success is False
                     or metadata.get("sha256") == hashlib.sha256(value.encode()).hexdigest()))
                for key, value in component.requested_arguments):
            continue
        indices.append(index)
    return indices


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", ".").replace("\u066b", "."))
    except ValueError:
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_in(value: Any, depth: int = 3) -> list[float]:
    """Every number a piece of receipt data holds."""
    if _is_number(value):
        return [float(value)]
    if isinstance(value, str):
        return [number for number in map(_number, (match[0] for match in _NUMERAL.finditer(value)))
                if number is not None]
    if depth <= 0:
        return []
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers_in(item, depth - 1)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers_in(item, depth - 1)]
    return []


def declared_percentages(result) -> list[float]:
    """The percentages a receipt states as measured or derived."""
    return [float(value) for key, value in (getattr(result, "metadata", None) or {}).items()
            if isinstance(key, str) and key.endswith(PERCENT_KEY) and _is_number(value)]


def receipt_figures(results) -> list[float]:
    """Every figure successful receipts carry, in their data and in what they returned."""
    figures: list[float] = []
    for result in results:
        if result.success and not blocked_before_execution(result):
            figures += _numbers_in(result.metadata or {}) + _numbers_in(result.output or "")
    return figures


def measurements(result) -> dict[str, Any]:
    """The quantities a receipt states, measured or derived, by the runtime's own key naming."""
    return {key: value for key, value in (getattr(result, "metadata", None) or {}).items()
            if isinstance(key, str) and key.endswith(MEASUREMENT_KEYS)}


def execution_evidence(result, output_chars: int = 1200) -> dict[str, Any]:
    """A measurement or file receipt as evidence for the reply's writer: facts, no envelope to copy."""
    metadata = result.metadata or {}
    arguments = metadata.get("arguments") if isinstance(metadata.get("arguments"), dict) else {}
    output = str(result.output or "")
    return dict(zip(EVIDENCE_KEYS, (
        result.tool_name, result.success, result.error, arguments.get("path") or metadata.get("path"),
        measurements(result), output[:output_chars], len(output) > output_chars)))


def states_figure(text: str, figures) -> bool:
    """Whether a text states a number that is among the given figures."""
    figures = list(figures)
    return any(_carried(value, decimals, figures) for value, decimals, _, _ in _stated(text, _NUMERAL))


def _stated(text: str, pattern: re.Pattern[str]) -> list[tuple[float, int, int, int]]:
    """Figures a text states: value, decimal places, and where each stands."""
    found = []
    for match in pattern.finditer(text):
        before = text[match.start() - 1] if match.start() else ""
        if before.isalpha() and not is_unspaced(before):
            continue   # a digit inside a name, such as a file name or a version
        raw = match[1] if pattern.groups else match[0]
        value = _number(raw)
        if value is not None:
            decimals = len(re.split(r"[.,\u066b]", raw)[1]) if re.search(r"[.,\u066b]", raw) else 0
            found.append((abs(value), decimals, match.start(), match.end()))
    return found


def _carried(value: float, decimals: int, figures: list[float]) -> bool:
    """Whether a stated figure equals one the receipts carry, at the precision it is stated with."""
    tolerance = 0.5 * 10 ** -decimals + 0.005
    return any(abs(value - abs(figure)) <= tolerance for figure in figures)


def unsupported_percentages(response: str, *, tool_trace: Any | None, prior_tool_receipts=(),
                            user_message: str = "") -> list[str]:
    """Clauses of a reply that state a percentage no receipt carries.

    This applies only when everything this turn's successful receipts returned
    is a measurement: each of them declares its percentages. Every percentage
    the reply states must then equal, at the precision it is stated with, a
    figure those receipts carry, one an earlier receipt declared, or one the
    user wrote in their own message. When any other successful receipt
    returned something, a figure may be derived from that output, and nothing
    is concluded here; the reviewer sees the receipts.

    The clause that states the figure is returned, not its whole sentence.
    """
    current = [result for result in getattr(tool_trace, "executed_results", None) or []
               if result.success and not blocked_before_execution(result)]
    measuring = [result for result in current if declared_percentages(result)]
    if not measuring or any(result not in measuring and str(result.output or "").strip() for result in current):
        return []
    figures = (receipt_figures(measuring) + _numbers_in(user_message or "")
               + [figure for result in prior_tool_receipts if result.success
                  for figure in declared_percentages(result)])
    spans = []
    ranges = clause_ranges(response)
    for value, decimals, start, end in _stated(response, _PERCENTAGE):
        if _carried(value, decimals, figures):
            continue
        spans += [response[a:b].strip() for a, b in ranges if a < end and start < b]
    return list(dict.fromkeys(span for span in spans if span))


# ---------------------------------------------------------------------------
# Delimited output, and spans that are nothing but receipt content
# ---------------------------------------------------------------------------


def _compact(text: Any) -> str:
    return " ".join(str(text or "").split())


def _operands(result) -> list[str]:
    """What a receipt says was attempted: the text values of its arguments."""
    metadata = result.metadata or {}
    arguments = metadata.get("arguments") if isinstance(metadata.get("arguments"), dict) else {}
    return [value for value in (*arguments.values(), metadata.get("cmd"), metadata.get("path"))
            if isinstance(value, str) and value.strip()]


def _literal_content(component) -> list[str]:
    """The text a component carries: how the reading described it, its span of the message, its operands."""
    return [component.description, component.source_span, *(value for _, value in component.requested_arguments)]


def _draft_blocks(response: str) -> list[tuple[int, int, int, int]]:
    """The parts a reply sets apart, as positions: where each block stands and where its content does.

    A block is fenced with backticks or tildes, or is a line that is one
    backtick span. A backtick span inside a sentence is not a block.
    """
    blocks = [(*match.span(), *match.span(2)) for match in _BLOCK.finditer(response)]
    blocks += [(*match.span(), *match.span(1)) for match in _CODE_LINE.finditer(response)
               if not any(block[0] <= match.start() < block[1] for block in blocks)]
    return sorted(blocks)


def _refusal(tool_trace: Any | None):
    """The intent the arbiter did not let run, when that is how the turn's tool work ended.

    The arbiter decides before anything executes, so a refused, deferred or
    queried intent leaves a decision in the trace and no receipt. It was
    attempted and did not succeed all the same.
    """
    gate = getattr(tool_trace, "final_decision", None)
    if gate is None or gate.decision == "execute":
        return None
    intents = [intent for intent in (gate.intent, *(getattr(tool_trace, "proposed_intents", None) or []))
               if intent is not None]
    return intents[0] if intents else gate


def _attempted(tool_trace: Any | None) -> bool:
    return bool(getattr(tool_trace, "executed_results", None)) or _refusal(tool_trace) is not None


def _unsuccessful_requests(decision, tool_trace) -> list[Any]:
    """The components the decision undertook whose attempts in this turn did not succeed.

    A receipt names the component it was made for. When every receipt made for
    a component was refused, denied or failed, that request was attempted and
    nothing came of it, whichever command the attempt happened to use. An
    intent the arbiter did not let run names its component too; when it names
    none, it stood for the turn's task work, of which nothing ran.
    """
    undertaken = [choice.component for choice in getattr(decision, "components", ()) or () if choice.granted]
    results = list(getattr(tool_trace, "executed_results", None) or [])
    failed = []
    for component in undertaken:
        made = [result for result in results if (result.metadata or {}).get("component_id") == component.id]
        if made and not any(result.success for result in made):
            failed.append(component)
    refused = _refusal(tool_trace)
    if refused is not None:
        named = getattr(refused, "component_id", "")
        covered = [component for component in undertaken
                   if (component.id == named if named else component.kind == "task" or len(undertaken) == 1)]
        failed += [component for component in covered if component not in failed and not any(
            result.success and (result.metadata or {}).get("component_id") == component.id for result in results)]
    return failed


def _receipt_carried(tool_trace: Any | None) -> tuple[list[str], set[str]]:
    """What the turn's receipts returned (output and error), and the exact operands of its attempts."""
    results = list(getattr(tool_trace, "executed_results", None) or [])
    returned = [_compact(text) for result in results for text in (result.output, result.error)]
    attempted = {_compact(value) for result in results for value in _operands(result)}
    refused = _refusal(tool_trace)
    arguments = getattr(refused, "arguments", None)
    if isinstance(arguments, dict):
        attempted |= {_compact(value) for value in arguments.values() if isinstance(value, str) and value.strip()}
    return [text for text in returned if text], attempted


def blocks_not_in_receipts(response: str, *, tool_trace: Any | None) -> list[str]:
    """The blocks a reply sets apart whose content no receipt of this turn returned.

    A structural fact for the reviewer, not a finding: a reply may set its own
    writing apart, such as code it was asked to write. Only when a tool was
    attempted in this turn; otherwise every block is the reply's own. A block
    is in the receipts when its content is in what one returned, as output or
    as an error, or is exactly an operand of an attempt, such as the command
    that was tried.
    """
    if not _attempted(tool_trace):
        return []
    returned, attempted = _receipt_carried(tool_trace)
    found = []
    for start, end, opened, closed in _draft_blocks(response):
        content = _compact(response[opened:closed])
        if content and content not in attempted and not any(content in text for text in returned):
            found.append(response[start:end].strip())
    return list(dict.fromkeys(found))


def posed_output_blocks(response: str, *, tool_trace: Any | None, decision: Any = None) -> list[str]:
    """Blocks of a reply that present, as output, what an attempt that did not succeed was about.

    The one case of invented output that structure alone shows. A command or
    operation was attempted in this turn and did not succeed: it was refused,
    denied, failed, or the arbiter did not let it run. The reply sets a block
    apart whose content no receipt returned and which is

    * a part of an operand of that attempt, such as the argument of a refused
      command set apart as if it had been printed; or
    * a part of the literal content of the request the attempt was made for
      (the component's description, its span of the message, its operands).
      The attempt may have used another command than the one the person wrote.

    That the person typed the text, or that the task names it, does not make
    it output. Quoting exactly what was attempted is not posing as its result,
    and neither is quoting what a receipt returned, including its error.

    Each block is returned as the reply has it. Any other block that is in no
    receipt is not found here: it may be the reply's own writing, and whether
    it is, is the reviewer's to judge (:func:`blocks_not_in_receipts`).
    """
    results = list(getattr(tool_trace, "executed_results", None) or [])
    refused = _refusal(tool_trace)
    if not results and refused is None:
        return []
    returned, attempted = _receipt_carried(tool_trace)
    undertaken = [choice.component for choice in getattr(decision, "components", ()) or () if choice.granted]
    attempted |= {_compact(value) for component in undertaken for _, value in component.requested_arguments}
    # What an unsuccessful attempt was given, and the request it was made for.
    unsuccessful = [_compact(value) for result in results if not result.success for value in _operands(result)]
    arguments = getattr(refused, "arguments", None)
    if isinstance(arguments, dict):
        unsuccessful += [_compact(value) for value in arguments.values() if isinstance(value, str)]
    unsuccessful += [_compact(text) for component in _unsuccessful_requests(decision, tool_trace)
                     for text in _literal_content(component)]
    found = []
    for start, end, opened, closed in _draft_blocks(response):
        content = _compact(response[opened:closed])
        if not content or content in attempted or any(content in text for text in returned):
            continue
        if any(content in text for text in unsuccessful):
            found.append(response[start:end].strip())
    return list(dict.fromkeys(found))


def _output_lines(text: Any) -> list[str]:
    return [line for line in (_compact(line) for line in str(text or "").splitlines()) if line]


def receipt_verified_span(span: str, *, tool_trace: Any | None, response: str = "") -> bool:
    """Whether a span is nothing but content this turn's successful receipts carry.

    Two things qualify, and both are exact:

    * the span is a complete block of the reply (:func:`_draft_blocks`) whose
      content is verbatim output of a successful receipt: whole lines of it,
      in a row;
    * the span is nothing but figures, each of which the receipts carry at the
      precision it is stated with. A lone whole number below ten is in any
      output and is no such figure.

    Nothing else does. A word, a name or a path that occurs in a receipt
    shields nothing, wherever in the reply it is quoted from: a span that names
    a file says that something was done to it, and a receipt that merely lists
    or touched that file verifies nothing about a claim to have deleted, sent
    or changed it. A line a receipt returned is not shielded either unless the
    reply sets it apart as a block.

    This establishes only that a finding against the span disputes text or
    figures the receipts carry. The caller keeps that as uncertainty, never as
    a pass.
    """
    results = [result for result in getattr(tool_trace, "executed_results", None) or []
               if result.success and not blocked_before_execution(result)]
    quoted = (span or "").strip()
    if not results or not quoted:
        return False
    for start, end, opened, closed in _draft_blocks(response or ""):
        if response[start:end].strip() != quoted:
            continue
        lines = _output_lines(response[opened:closed])
        for result in results:
            returned = _output_lines(result.output)
            if lines and any(returned[index:index + len(lines)] == lines for index in range(len(returned))):
                return True
    stated = _stated(quoted, _NUMERAL)
    residue = _NUMERAL.sub(" ", quoted)
    if not stated or any(char.isalpha() for char in residue):
        return False
    if all(decimals == 0 and value < 10 for value, decimals, _, _ in stated):
        return False
    figures = receipt_figures(results)
    return all(_carried(value, decimals, figures) for value, decimals, _, _ in stated)


# ---------------------------------------------------------------------------
# Compatibility for the evaluation harness
# ---------------------------------------------------------------------------

_EVALUATION_COMPATIBILITY = ("verify_response_grounding",)


def __getattr__(name: str) -> Any:
    """Names the role-play evaluation still imports from here.

    Its deterministic checks use the former wording-based claim check, which
    lives with the harness now. No turn calls it.
    """
    if name in _EVALUATION_COMPATIBILITY:
        from openrook_tools.roleplay_eval import legacy_claims

        return getattr(legacy_claims, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

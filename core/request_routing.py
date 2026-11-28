"""Requests by understanding: which capability a task asks for, and with which operands.

The reading (``core/appraisal.py``) says what a message asks. For a task it may
name the capability that does the work, out of the capabilities the reader was
offered, and give the operands the person wrote. This module holds what the
runtime does with such a report. It reads no wording in any language: names are
compared with the offered names, and values with the message's own text.

* :class:`Offer` is a capability as the reader is told of it: its name, a
  one-line purpose and its argument names.
* :class:`ToolRequest` and :class:`ComponentReport` are what a reading reports
  about one component beside the component itself.
* :func:`ground` holds a report to what was written and turns it into the
  component's execution detail. It never adds a component and never changes a
  component's kind, description or price.
* :class:`Selection` is what the model that selects a tool call is given for
  one granted task.

Contract:

* A component stands somewhere: its ``source_span`` is a stretch of this
  message, or of an earlier message of the same person that it refers back to.
  A part that stands nowhere takes no operand.
* A literal operand is the person's own text. Its value must be a whole written
  unit (``core.perception.exact_part``) inside the stretch its component stands
  in, or inside an earlier message of the same person. A value that is not is
  removed. A long value may be given by its first and last words; what is
  committed is then the message's own text between them.
* An operand that has to be produced (``generate``) is never taken from the
  reader: whatever value an answer gives for it is discarded.
* The capabilities a task asks for are its tool entries. A task whose one
  entry has every required operand as the person's own text is a committed
  operation: ``requested_tool`` and ``requested_arguments`` are set, and the
  decision grants or declines exactly those operands. So is a task that names
  one capability which needs no operand at all.
* Anything less leaves the task naming its capabilities
  (``capability_requirements``). It keeps, for each of them, the operands that
  were verified and the names of the literal arguments whose reported value
  was not the person's text. What is left open is produced when the granted
  task is carried out, by a model that is offered the named capabilities and
  nothing else; ``core.grants.execution_denial`` holds every call it selects to
  those capabilities, to the verified operands, and to the person's own text
  for the arguments the reading got wrong.
* Work that needs something none of the offered capabilities does is reported
  as ``unavailable``. It cannot be granted, and it is kept as a capability gap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Iterable, NamedTuple

from core.perception import backs_claim, locate, passage, tokens, written
from core.willingness import RequestComponent

# The session's own capability: taking in material as formative learning. It is
# offered when a session wired an executor for it (``runtime/sessions``).
LEARNING = "learning.intake"
# The mark the session's learning executor requires on the component it is
# handed (``runtime/sessions/manager.py``). The pipeline adds it to a task the
# reading reported as asking for the learning capability.
LEARNING_GRANT = "learning:explicit"
PLAN_RELATIONS = ("continue", "status")
PURPOSE_CHARS = 160
GAP_CHARS = 160              # how much of a reported missing capability is kept
GAP_REF = "capability:unavailable"   # evidence that a component needs what nothing offered does
# How a task that carries no committed operation keeps what its operands are held to. Both
# travel with the component, and so with the frozen decision that grants it.
#   * A verified operand is a pair in ``requested_arguments``. Its name is the argument's when
#     the task names one capability once, and ``tool:argument`` otherwise.
#   * A literal argument whose reported value was not the person's text is a mark in
#     ``evidence_refs``: ``operand:unverified:tool:argument``.
#   * A task that asks for a capability from a part that stands nowhere in what the person
#     wrote carries ``operand:unplaced``: no text is its own, so none can be its operand.
OPERAND_OF = ":"
UNVERIFIED_OPERAND = "operand:unverified:"
UNPLACED_PART = "operand:unplaced"

# What a reading got wrong about a request, by the runtime's own codes.
UNKNOWN_CAPABILITY = "capability_unknown"
UNKNOWN_ARGUMENT = "argument_unknown"
UNQUOTED_ARGUMENT = "argument_unquoted"
UNQUOTED_PLAN = "plan_unquoted"
# Slips: something stated where it has no meaning. Each is removed and recorded;
# the rest of the reading stands as it is, so none of them is asked about again.
NOT_A_TASK = "tool_not_task"
UNQUOTED_SPAN = "span_unquoted"
NO_PLAN = "plan_absent"
PLAN_NOT_TASK = "plan_not_task"
UNQUOTED_REPRODUCTION = "reproduction_unquoted"
REPRODUCTION_NOT_TASK = "reproduction_not_task"
REPRODUCTION_WHOLE = "reproduction_whole"
UNAVAILABLE_NOT_TASK = "unavailable_not_task"
UNAVAILABLE_NAMED = "unavailable_with_capability"
IGNORED_CAPABILITY = "capability_ignored"
SLIPS = frozenset({NOT_A_TASK, UNQUOTED_SPAN, NO_PLAN, PLAN_NOT_TASK, UNQUOTED_REPRODUCTION,
                   REPRODUCTION_NOT_TASK, REPRODUCTION_WHOLE, UNAVAILABLE_NOT_TASK, UNAVAILABLE_NAMED,
                   IGNORED_CAPABILITY})


@dataclass(frozen=True)
class Offer:
    """A capability as the reader is told of it."""

    name: str
    purpose: str = ""
    # Every argument the capability cannot run without, whatever its type.
    required: tuple[str, ...] = ()
    # Arguments it can do without, when their value is text.
    optional: tuple[str, ...] = ()
    # Arguments whose value is text, and so can be a literal operand. None when
    # the capability declares no schema: every argument it is given is text.
    text: frozenset[str] | None = None

    def takes(self, argument: str) -> bool:
        return self.text is None or argument in self.required or argument in self.optional

    def literal(self, argument: str) -> bool:
        return self.text is None or argument in self.text

    @property
    def produced(self) -> tuple[str, ...]:
        """Required arguments whose value is not text: never a literal operand, always produced."""
        return () if self.text is None else tuple(name for name in self.required if name not in self.text)

    def to_data(self) -> dict[str, Any]:
        """The offer as the reader is given it. A capability that declares no schema states no arguments."""
        data: dict[str, Any] = {"name": self.name, "purpose": self.purpose}
        if self.text is not None:
            data["arguments"] = list(self.required)
        if self.optional:
            data["optional_arguments"] = list(self.optional)
        if self.produced:
            data["produced_arguments"] = list(self.produced)
        return data


LEARNING_OFFER = Offer(
    LEARNING, "Take in material the person supplies as formative learning, kept in the character's "
    "life history. The material is at the address given as url, or is the message itself.",
    optional=("url",), text=frozenset({"url"}))


class ToolRequest(NamedTuple):
    """What a reading reports a task asks a capability for."""

    tool: str
    arguments: tuple[tuple[str, str], ...] = ()   # operands the person wrote, as the reader copied them
    generate: tuple[str, ...] = ()                # operands that have to be produced
    # For a long operand: its last words. Its first words are then its value in ``arguments``.
    ends: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ComponentReport:
    """What a reading says about one component beside the component itself."""

    tools: tuple[ToolRequest, ...] = ()
    reproduce: str = ""    # text the person asks to be given back word for word
    plan: str = ""         # continue | status: how this part relates to the running plan
    span_end: str = ""        # the last words of a long source_span, whose first words the component carries
    reproduce_end: str = ""   # the last words of a long text to give back
    unavailable: str = ""     # what the work needs that none of the offered capabilities does


class Problem(NamedTuple):
    """One thing a report said that could not be held to what was written."""

    code: str
    component: str
    detail: str = ""

    @property
    def ref(self) -> str:
        return ":".join(part for part in (f"reading:{self.code}", self.component, self.detail) if part)

    @property
    def slip(self) -> bool:
        """Removed and recorded without the reading being asked for again."""
        return self.code in SLIPS


class Home(NamedTuple):
    """The stretch of what the person wrote that one component stands in."""

    text: str
    low: int
    high: int


@dataclass(frozen=True)
class Selection:
    """What the model that selects a tool call is given for one granted task. All of it is data.

    ``capabilities`` are the only tools it is offered. ``operands`` are values
    the person wrote for that capability, to be used exactly as written.
    ``recent_turns`` and ``receipts`` let it work out what the request refers
    to: the conversation so far, and what was already executed in this turn.
    """

    request: str
    capabilities: tuple[str, ...] = ()
    operands: tuple[tuple[str, str, str], ...] = ()   # (capability, argument, value)
    recent_turns: tuple[tuple[str, str], ...] = ()   # (speaker, text), oldest first
    receipts: tuple[dict[str, Any], ...] = ()

    def to_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"request": self.request}
        if self.operands and len(self.capabilities) <= 1:
            data["written_operands"] = {argument: value for _, argument, value in self.operands}
        elif self.operands:
            # Several capabilities: each operand is the one the person wrote for that capability.
            by_tool: dict[str, dict[str, str]] = {}
            for tool, argument, value in self.operands:
                by_tool.setdefault(tool, {})[argument] = value
            data["written_operands"] = by_tool
        if self.recent_turns:
            data["recent_turns"] = [{"speaker": speaker, "text": text} for speaker, text in self.recent_turns]
        if self.receipts:
            data["receipts"] = list(self.receipts)
        return data

    def to_text(self) -> str:
        return json.dumps(self.to_data(), ensure_ascii=False)


def _is_text(spec: Any) -> bool:
    kind = spec.get("type", "string") if isinstance(spec, dict) else "string"
    return kind == "string" or (isinstance(kind, (list, tuple)) and "string" in kind)


def offer_of(capability: Any) -> Offer:
    """A registered capability as it is offered.

    Every required argument is offered, whatever its type, because the
    capability cannot run without it; one whose value is not text is marked as
    produced and is never a literal operand. An optional argument is offered
    only when its value is text. A capability registered with an empty schema
    takes no argument.
    """
    schema = getattr(capability, "arg_schema", None)
    purpose = " ".join(str(getattr(capability, "description", "") or "").split())[:PURPOSE_CHARS]
    if not isinstance(schema, dict):
        return Offer(capability.name, purpose)   # nothing is declared: what it is given is taken as text
    # An empty schema is a declaration too: the capability takes no argument at all.
    # In name order, so that the same registry is offered the same way however its schema was written down.
    required = tuple(sorted(name for name, spec in schema.items() if isinstance(spec, dict) and spec.get("required")))
    text = frozenset(name for name, spec in schema.items() if _is_text(spec))
    optional = tuple(sorted(name for name in schema if name not in required and name in text))
    return Offer(capability.name, purpose, required, optional, text)


def offers_of(registry: Any) -> tuple[Offer, ...]:
    """Every capability a tool registry holds, in name order."""
    if registry is None:
        return ()
    return tuple(offer_of(capability) for capability in sorted(registry.list_tools(), key=lambda item: item.name))


def as_offers(items: Iterable[Any] | None) -> tuple[Offer, ...] | None:
    """Offers from what a caller has: offers, their data form, or bare names. None stays None."""
    if items is None:
        return None
    offers = []
    for item in items:
        if isinstance(item, Offer):
            offers.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            required = tuple(str(name) for name in item.get("arguments") or ())
            optional = tuple(str(name) for name in item.get("optional_arguments") or ())
            produced = {str(name) for name in item.get("produced_arguments") or ()}
            declared = "arguments" in item or "optional_arguments" in item
            offers.append(Offer(item["name"], str(item.get("purpose") or ""), required, optional,
                                frozenset({*required, *optional} - produced) if declared else None))
        elif isinstance(item, str):
            offers.append(Offer(item))   # a bare name: nothing is known about its arguments
    return tuple(offers)


def offer_names(offers: Iterable[Offer] | None) -> frozenset[str] | None:
    return None if offers is None else frozenset(offer.name for offer in offers)


def plan_report(plan: Any) -> dict[str, Any] | None:
    """A running plan as the reader is told of it: its goal, how far it is and its next step."""
    if plan is None or getattr(plan, "is_terminal", True):
        return None
    steps = list(getattr(plan, "steps", ()) or ())
    index = getattr(plan, "current_step_index", 0)
    report: dict[str, Any] = {"goal": str(getattr(plan, "goal", ""))[:200],
                              "steps_done": getattr(plan, "steps_completed", 0), "steps_total": len(steps)}
    if 0 <= index < len(steps):
        report["next_step"] = str(getattr(steps[index], "description", "") or steps[index].tool_name)[:160]
    return report


def same_gap(one: str, other: str) -> bool:
    """Whether two reports of a missing capability are the same report.

    They are when they say the same words in the same order, however they are
    cased, spaced or punctuated. Sharing words is not enough: two different
    things that are missing are often worded alike, and the same words in
    another order can name the opposite thing. A need that comes back in other
    words is recognised by the reader, which is shown the gaps already known
    and repeats the one it means (``core.appraisal.GAP_PROTOCOL``).
    """
    first = tokens(one)
    return bool(first) and first == tokens(other)


def held_operands(component: RequestComponent) -> tuple[tuple[str, str, str], ...]:
    """(capability, argument, value) for each operand the reading verified on a task with no committed operation."""
    named = component.capability_requirements
    held = []
    for name, value in component.requested_arguments:
        tool, _, argument = name.rpartition(OPERAND_OF)
        if tool:
            held.append((tool, argument, value))
        elif len(named) == 1:
            held.append((named[0], name, value))
    return tuple(held)


def written_for(component: RequestComponent, value: Any, *, message: str = "", earlier: Iterable[str] = ()) -> bool:
    """Whether ``value`` is the person's own text for this component.

    The same test an operand passes when the reading is held to the message:
    a whole written unit of the stretch the component stands in, or of an
    earlier message of the same person. ``message`` is the message of the
    turn; without it the component's own span is all there is to look in. A
    component that stands nowhere in what the person wrote has no such text.
    """
    if not isinstance(value, str) or UNPLACED_PART in component.evidence_refs:
        return False
    earlier = list(earlier)
    if message:
        _, home = _placed(component, "", message, earlier, [])
    else:
        span = component.source_span
        home = Home(span, 0, len(span)) if span.strip() else None
    return _operand(value, "", home, earlier) is not None


def unverified_operands(component: RequestComponent) -> tuple[tuple[str, str], ...]:
    """(capability, argument) for each literal argument whose reported value was not the person's text."""
    marks = (ref[len(UNVERIFIED_OPERAND):].rpartition(OPERAND_OF) for ref in component.evidence_refs
             if ref.startswith(UNVERIFIED_OPERAND))
    return tuple((tool, argument) for tool, _, argument in marks if tool and argument)


def unavailable_work(decision: Any) -> dict[str, str]:
    """For each component of a decision that needs what nothing offered does: what the reading said it needs."""
    marked = f"{GAP_REF}:"
    evidence = getattr(getattr(decision, "perception", None), "evidence", ()) or ()
    return {item.ref[len(marked):]: item.observation for item in evidence if item.ref.startswith(marked)}


def _with_edges(text: str, low: int, high: int) -> Home:
    """A stretch of a text together with the white space around it, which belongs to no other part.

    A quotation is located without the spaces and line breaks at its edges; a
    value the person wrote may begin or end with them.
    """
    while low > 0 and text[low - 1].isspace():
        low -= 1
    while high < len(text) and text[high].isspace():
        high += 1
    return Home(text, low, high)


def _placed(component: RequestComponent, span_end: str, text: str, earlier: list[str],
            problems: list[Problem]) -> tuple[RequestComponent, Home | None]:
    """Where a component stands: a stretch of this message, or an earlier message it refers back to.

    A span given by its first and last words becomes the text between them;
    last words that do not close one stretch are dropped, and the span is then
    its first words alone. None when the span is no part of anything the
    person wrote.
    """
    span = component.source_span
    if span_end.strip():
        for source in (text, *earlier):
            place = passage(span, span_end, source)
            if place is not None:
                home = _with_edges(source, *place) if source is text else Home(source, 0, len(source))
                return replace(component, source_span=source[place[0]:place[1]]), home
        problems.append(Problem(UNQUOTED_SPAN, component.id, "end"))
    place = locate(span, text)
    if place is not None:
        return component, _with_edges(text, *place)
    for prior in earlier:
        if locate(span, prior) is not None:
            return component, Home(prior, 0, len(prior))
    return component, None


def _operand(value: str, end: str, home: Home | None, earlier: list[str]) -> str | None:
    """The person's own characters for one operand, or None when they did not write it where it may stand.

    It may stand in the stretch its component stands in, or in an earlier
    message of the same person. A component that stands nowhere has no
    operand: nothing ties it to anything the person wrote.
    """
    if home is None:
        return None
    for source, low, high in (home, *(Home(prior, 0, len(prior)) for prior in earlier)):
        place = (passage(value, end, source, low=low, high=high) if end
                 else written(value, source, low=low, high=high))
        if place is not None:
            return source[place[0]:place[1]]
    return None


class _Checked(NamedTuple):
    """One tool request held to what was written."""

    tool: str
    literal: tuple[tuple[str, str], ...]   # operands that are the person's own text
    unverified: tuple[str, ...]            # literal arguments whose reported value was not
    complete: bool                         # every required operand stands, and nothing is left to produce


def _checked(component: RequestComponent, request: ToolRequest, offer: Offer, home: Home | None,
             earlier: list[str], problems: list[Problem]) -> _Checked:
    literal: dict[str, str] = {}
    ends = dict(request.ends)
    unverified: list[str] = []
    removed = False
    for name, value in request.arguments:
        if name in request.generate or name in literal:
            continue   # a value to be produced is never the reader's; a repeated name adds nothing
        if not offer.takes(name):
            problems.append(Problem(UNKNOWN_ARGUMENT, component.id, name))
            removed = True
            continue
        if not offer.literal(name):
            removed = True   # not text: it cannot be a committed literal operand, and is produced instead
            continue
        own = _operand(value, ends.get(name, ""), home, earlier)
        if own is None:
            problems.append(Problem(UNQUOTED_ARGUMENT, component.id, name))
            unverified.append(name)
            removed = True
            continue
        literal[name] = own
    to_produce = [name for name in request.generate if offer.takes(name)]
    complete = not removed and not to_produce and set(offer.required) <= literal.keys()
    return _Checked(request.tool, tuple(literal.items()), tuple(dict.fromkeys(unverified)), complete)


def _grounded(component: RequestComponent, checked: list[_Checked], problems: list[Problem]) -> RequestComponent:
    """The component with what its tool requests could be held to.

    One request that stands whole is the component's committed operation.
    Anything else leaves it a task that names the capabilities, and keeps for
    each of them the operands the person wrote and the names of the literal
    arguments whose reported value they did not write
    (:func:`held_operands`, :func:`unverified_operands`): the grant check holds
    a selected call to both.
    """
    others = tuple(name for name in component.capability_requirements if all(name != item.tool for item in checked))
    if len(checked) == 1 and checked[0].complete:
        # One committed operation. Whatever else the component named is not part of it.
        problems.extend(Problem(IGNORED_CAPABILITY, component.id, name) for name in others)
        return replace(component, capability_requirements=(checked[0].tool,),
                       requested_arguments=checked[0].literal, requested_tool=checked[0].tool)
    names = tuple(dict.fromkeys((*component.capability_requirements, *(item.tool for item in checked))))
    plain = len(names) == 1 and len(checked) == 1
    operands = tuple((name if plain else f"{item.tool}{OPERAND_OF}{name}", value)
                     for item in checked for name, value in item.literal)
    marks = tuple(f"{UNVERIFIED_OPERAND}{item.tool}{OPERAND_OF}{name}" for item in checked for name in item.unverified)
    return replace(component, capability_requirements=names, requested_arguments=tuple(dict.fromkeys(operands)),
                   evidence_refs=tuple(dict.fromkeys((*component.evidence_refs, *marks))))


def _reproduction(component: RequestComponent, report: ComponentReport, home: Home | None, text: str,
                  earlier: list[str], problems: list[Problem]) -> str:
    """The text a task asks to be given back, when that stands; otherwise nothing, with the reason recorded.

    It stands for a task only. It is the person's own text: from an earlier
    turn, or a part of this message that leaves the asking itself out. A
    reproduction that takes in the component's whole span, or the whole
    message, would excuse a reply that merely repeats what it was told.
    """
    stated = report.reproduce
    if not stated.strip():
        return ""
    if component.kind != "task":
        problems.append(Problem(REPRODUCTION_NOT_TASK, component.id))
        return ""
    place = None
    if report.reproduce_end.strip():
        for source in (text, *earlier):
            found = passage(stated, report.reproduce_end, source)
            if found is not None:
                stated, place = source[found[0]:found[1]], found if source is text else None
                break
        else:
            problems.append(Problem(UNQUOTED_REPRODUCTION, component.id))
            return ""
    elif backs_claim(stated, text):
        place = locate(stated, text)
    elif not any(backs_claim(stated, prior) for prior in earlier):
        problems.append(Problem(UNQUOTED_REPRODUCTION, component.id))
        return ""
    if place is not None:
        whole = tokens(stated) == tokens(text)
        # Where the component's own span stands, without the white space around it.
        asked = locate(component.source_span, text) if home is not None and home.text is text else None
        covers = asked is not None and place[0] <= asked[0] and asked[1] <= place[1]
        if whole or covers:
            problems.append(Problem(REPRODUCTION_WHOLE, component.id))
            return ""
    return stated


def ground(component: RequestComponent, report: ComponentReport, *, text: str, earlier: Iterable[str] = (),
           offers: Iterable[Offer] | None = None, plan_running: bool = False,
           ) -> tuple[RequestComponent, ComponentReport, tuple[Problem, ...]]:
    """Hold one component's report to what was written.

    Returns the component with its execution detail, the part of the report
    that stands (reproduction, plan relation and what is unavailable), and
    what was removed. ``offers`` is what the reader was offered; None means it
    was given no list, and then a named capability is taken as it stands.
    """
    earlier = list(earlier)
    problems: list[Problem] = []
    component, home = _placed(component, report.span_end, text, earlier, problems)
    known = None if offers is None else {offer.name: offer for offer in offers}
    requests = list(report.tools)
    if requests and component.kind != "task":
        problems.append(Problem(NOT_A_TASK, component.id))
        requests = []
    named = []
    for request in requests:
        offer = Offer(request.tool) if known is None else known.get(request.tool)
        if offer is None:
            problems.append(Problem(UNKNOWN_CAPABILITY, component.id, request.tool))
            continue
        named.append((request, offer))
    if named and home is None:
        # A request for a capability by a part that stands nowhere in what the person wrote:
        # nothing ties it to their text, so none of it can be taken as an operand, now or when
        # a model selects the call.
        component = replace(component, evidence_refs=tuple(dict.fromkeys((*component.evidence_refs, UNPLACED_PART))))
        if not report.span_end.strip():
            problems.append(Problem(UNQUOTED_SPAN, component.id))
    plan = report.plan
    if plan and not plan_running:
        problems.append(Problem(NO_PLAN, component.id))
        plan = ""
    elif plan == "continue" and component.kind != "task":
        problems.append(Problem(PLAN_NOT_TASK, component.id))
        plan = ""
    elif plan and not backs_claim(component.source_span, text):
        problems.append(Problem(UNQUOTED_PLAN, component.id))
        plan = ""
    if named:
        component = _grounded(component, [_checked(component, request, offer, home, earlier, problems)
                                          for request, offer in named], problems)
    elif (component.kind == "task" and known is not None and plan != "continue"
          and len(component.capability_requirements) == 1):
        # One capability named and no operand given: when it declares that it needs none, that is
        # the whole operation, and it is committed as it stands.
        offer = known.get(component.capability_requirements[0])
        if offer is not None and offer.text is not None and not offer.required:
            component = replace(component, requested_tool=offer.name)
    unavailable = " ".join(report.unavailable.split())[:GAP_CHARS]
    if unavailable and component.kind != "task":
        problems.append(Problem(UNAVAILABLE_NOT_TASK, component.id))
        unavailable = ""
    elif unavailable and component.capability_requirements:
        # It names an offered capability that does the work, so the work is not out of reach.
        problems.append(Problem(UNAVAILABLE_NAMED, component.id))
        unavailable = ""
    elif unavailable:
        component = replace(component, feasible=False)
    reproduce = _reproduction(component, report, home, text, earlier, problems)
    return component, ComponentReport(reproduce=reproduce, plan=plan, unavailable=unavailable), tuple(problems)


def _said(problem: Problem) -> str:
    detail = json.dumps(problem.detail, ensure_ascii=False)
    if problem.code == NOT_A_TASK:
        return "a component that is not a task named a tool"
    if problem.code == UNKNOWN_CAPABILITY:
        return f"a tool named {detail}, which is not in available_capabilities"
    if problem.code == UNKNOWN_ARGUMENT:
        return f"a tool was given the argument {detail}, which that capability does not take"
    if problem.code == UNQUOTED_ARGUMENT:
        return (f"the value of the tool argument {detail} is not text the person wrote: it must be copied "
                "character for character, whole and with nothing cut off or added, from inside that "
                "part's source_span or from an earlier user turn (a long value by its first words in "
                "value and its last words in end, each standing only once); a value the person did not "
                "write belongs in generate, without a value")
    if problem.code == UNQUOTED_SPAN:
        return ("a source_span, or the last words given for it, was not copied verbatim from the message or "
                "an earlier user turn, so nothing the person wrote could be taken as that part's operand")
    if problem.code == UNQUOTED_PLAN:
        return "a plan relation was given for a component whose source_span was not copied verbatim from the message"
    if problem.code == NO_PLAN:
        return "a plan relation was given although no plan is running"
    if problem.code == PLAN_NOT_TASK:
        return "continue was given for a component that is not a task"
    if problem.code == UNQUOTED_REPRODUCTION:
        return "reproduce was not copied verbatim from the message or an earlier user turn"
    if problem.code == REPRODUCTION_NOT_TASK:
        return "reproduce was given for a component that is not a task"
    if problem.code == REPRODUCTION_WHOLE:
        return "reproduce took in the whole of what asks for it"
    if problem.code == UNAVAILABLE_NOT_TASK:
        return "unavailable was given for a component that is not a task"
    if problem.code == UNAVAILABLE_NAMED:
        return "unavailable was given for a component that names an offered capability"
    return f"the capability {detail} was named beside one complete tool request and is no part of it"


def feedback(problems: Iterable[Problem]) -> str:
    """What the reader is told about a report that could not be held to the message."""
    return "; ".join(dict.fromkeys(_said(problem) for problem in problems))

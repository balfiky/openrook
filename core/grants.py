"""Execution scopes bind every route to the same immutable component grant.

Operator permission checks remain in the tool runtime. This additional gate is
character authorization; a backend cannot create it by naming an action 'self'.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from core.request_routing import held_operands, unverified_operands, written_for
from core.willingness import TurnDecision


@dataclass(frozen=True)
class ExecutionScope:
    decision: TurnDecision
    component_id: str = ""
    purpose: str = "user"
    independent_tools: tuple[str, ...] = ()
    # The message of this turn and earlier messages of the same person. Context, not authority:
    # they are where a value the person wrote for the component may stand.
    message: str = ""
    earlier: tuple[str, ...] = ()


_scope: ContextVar[ExecutionScope | None] = ContextVar("component_execution_scope", default=None)
# True while the pipeline carries a person's turn through its stages. Every execution in those
# stages belongs to a granted component, so a call that arrives there without a scope has lost it.
_in_turn: ContextVar[bool] = ContextVar("turn_stages", default=False)


@contextmanager
def turn_stages():
    """Mark the stages of one person's turn. Inside them an execution without a scope is denied."""
    token = _in_turn.set(True)
    try:
        yield
    finally:
        _in_turn.reset(token)


def execution_metadata() -> dict:
    scope = _scope.get()
    if scope is None:
        return {}
    return {"component_id": scope.component_id, "decision_turn_id": scope.decision.turn_id,
            "decision_policy": scope.decision.policy_version, "execution_scope": scope.purpose}


@contextmanager
def execution_scope(decision: TurnDecision, component_id: str = "", *,
                    purpose: str = "user", independent_tools: tuple[str, ...] = (),
                    message: str = "", earlier: tuple[str, ...] = ()):
    if purpose not in {"user", "self", "independent"}:
        raise ValueError("Unknown execution scope")
    if purpose == "self" and decision.scope != "self":
        raise ValueError("A user decision cannot be relabeled as self-directed authorization")
    if purpose == "independent" and any(not name.startswith("self.") for name in independent_tools):
        raise ValueError("Independent conversational activity is scoped to cognitive self tools")
    token = _scope.set(ExecutionScope(decision, component_id, purpose, independent_tools, message, tuple(earlier)))
    try:
        yield
    finally:
        _scope.reset(token)


def execution_denial(tool_name: str, *, arguments: dict | None = None, decision: TurnDecision | None = None,
                     component_id: str = "") -> str:
    """Why a tool call is not covered by the committed grant; empty when it is.

    A granted task of a person's request covers what that request asked for and
    nothing else:

    * a committed operation (``requested_tool``) covers that tool with exactly
      the committed operands;
    * a task that names capabilities without a committed operation covers those
      capabilities only. For each of them, every operand the reading verified
      as the person's own text must be given as written, and a literal argument
      whose reported value was not the person's text may only be given a value
      that is: a whole written unit of the component's own stretch of the
      message, or of an earlier message of the same person. This is what bounds
      a model that selects the call: whatever it answers, another tool, another
      value for a verified operand, or a value nobody wrote for an argument
      that is the person's to give, is denied here;
    * a task that names no capability is work done in the reply and covers no
      tool at all.

    ``arguments`` is None when only availability is asked: whether the tool can
    be offered at all under the grant. The character's own activity (a decision
    of scope ``self``) is not a person's request: it has its own decision, and
    its tools are chosen and gated where that activity runs.

    Without a decision there is nothing to hold a call to. Inside the stages of
    a person's turn (:func:`turn_stages`) that means the call lost its scope,
    and it is denied. Outside a turn it is an operator's, a study's or a test's
    own call to the executor, which is not a conversational grant and is
    allowed here; the operator's own gates in the tool runtime still apply.
    """
    scope = _scope.get()
    message, earlier = "", ()
    if scope is not None:
        if scope.purpose == "independent":
            return "" if tool_name in scope.independent_tools else "Outside the independent activity scope"
        decision, component_id, message, earlier = scope.decision, scope.component_id, scope.message, scope.earlier
    if decision is None:
        return "No execution scope inside a turn" if _in_turn.get() else ""
    if not component_id and len(decision.grants) == 1:
        component_id = decision.grants[0]
    if not decision.permits(component_id):
        return "No committed grant for this request component"
    component = next(c for c in decision.components if c.component.id == component_id).component
    if component.kind != "task":
        return "This component grants expression, not tool execution"
    if (decision.scope == "user" and tool_name.startswith("self.")
            and tool_name != component.requested_tool
            and tool_name not in component.capability_requirements):
        return "Cognitive self action needs a matching component grant or a separate independent decision"
    if component.requested_tool:
        if tool_name != component.requested_tool or (
                arguments is not None and arguments != dict(component.requested_arguments)):
            return "Tool or operands differ from the committed atomic request"
        return ""
    if decision.scope != "user":
        return ""
    if tool_name not in component.capability_requirements:
        return "Tool is outside the capabilities this request component asks for"
    if arguments is None:
        return ""
    # Each operand the person wrote for this capability, by argument. Where the reading
    # reported the capability more than once, any of the values it verified stands.
    wrote: dict[str, list[str]] = {}
    for tool, name, value in held_operands(component):
        if tool == tool_name:
            wrote.setdefault(name, []).append(value)
    if any(arguments.get(name) not in values for name, values in wrote.items()):
        return "An operand differs from what the person wrote for this request component"
    for tool, name in unverified_operands(component):
        if tool != tool_name or name in wrote or name not in arguments:
            continue
        if not written_for(component, arguments[name], message=message, earlier=earlier):
            return "An operand that is the person's to give was given a value they did not write"
    return ""

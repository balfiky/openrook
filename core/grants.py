"""Execution scopes bind every route to the same immutable component grant.

Operator permission checks remain in the tool runtime. This additional gate is
character authorization; a backend cannot create it by naming an action 'self'.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from core.willingness import TurnDecision


@dataclass(frozen=True)
class ExecutionScope:
    decision: TurnDecision
    component_id: str = ""
    purpose: str = "user"
    independent_tools: tuple[str, ...] = ()


_scope: ContextVar[ExecutionScope | None] = ContextVar("component_execution_scope", default=None)


def execution_metadata() -> dict:
    scope = _scope.get()
    if scope is None:
        return {}
    return {"component_id": scope.component_id, "decision_turn_id": scope.decision.turn_id,
            "decision_policy": scope.decision.policy_version, "execution_scope": scope.purpose}


@contextmanager
def execution_scope(decision: TurnDecision, component_id: str = "", *,
                    purpose: str = "user", independent_tools: tuple[str, ...] = ()):
    if purpose not in {"user", "self", "independent"}:
        raise ValueError("Unknown execution scope")
    if purpose == "self" and decision.scope != "self":
        raise ValueError("A user decision cannot be relabeled as self-directed authorization")
    if purpose == "independent" and any(not name.startswith("self.") for name in independent_tools):
        raise ValueError("Independent conversational activity is scoped to cognitive self tools")
    token = _scope.set(ExecutionScope(decision, component_id, purpose, independent_tools))
    try:
        yield
    finally:
        _scope.reset(token)


def execution_denial(tool_name: str, *, arguments: dict | None = None, decision: TurnDecision | None = None,
                     component_id: str = "") -> str:
    scope = _scope.get()
    if scope is not None:
        if scope.purpose == "independent":
            return "" if tool_name in scope.independent_tools else "Outside the independent activity scope"
        decision, component_id = scope.decision, scope.component_id
    if decision is None:
        return ""  # operator/internal callers outside a conversational decision
    if not component_id and len(decision.grants) == 1:
        component_id = decision.grants[0]
    if not decision.permits(component_id):
        return "No committed grant for this request component"
    item = next(c for c in decision.components if c.component.id == component_id)
    if item.component.kind != "task":
        return "This component grants expression, not tool execution"
    if (decision.scope == "user" and tool_name.startswith("self.")
            and tool_name != item.component.requested_tool
            and tool_name not in item.component.capability_requirements):
        return "Cognitive self action needs a matching component grant or a separate independent decision"
    if item.component.requested_tool and (tool_name != item.component.requested_tool
            or arguments is not None and arguments != dict(item.component.requested_arguments)):
        return "Tool or operands differ from the committed atomic request"
    return ""

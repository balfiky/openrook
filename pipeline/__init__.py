"""Cognitive pipeline with one versioned choice owner.

The model's reading of the message and context retrieval -> personality-dependent
relationship update -> pure willingness optimization -> immutable component
grants -> scoped execution and receipts -> expression from continuous state ->
tri-state validation and bounded correction -> delivery and grounded outcome
records.

Every user message is read by the model: a required call with its own allowance
of two (one retry for an unusable answer). The reading also says what each task
asks a capability for; nothing recognises a request by its wording. Ordinary
conversation then needs one expression call and its review; a rewrite adds
generation and review. The base budget for that work is five calls, including
supplemental tool/learning inference. Only
when bounded drafts leave no safe remainder may one protected final expression
attempt use an additional call, recorded as delivery_recovery. A granted task
that names a capability without all of its operands has one protected primary
tool selection, fully accounted; a task that names none never reaches tool
selection. No
mandatory inner dialogue or user-turn self-intent proposals run in this policy.
Independent activity has its own decision and execution scope.
"""

from core.types import PipelineContext
from pipeline.heuristics import (
    _CONVERSATION_HISTORY_MAX as _CONVERSATION_HISTORY_MAX,
    _REPAIRABLE_UNRESOLVED_SOURCES as _REPAIRABLE_UNRESOLVED_SOURCES,
    _format_generated_artifact_summary as _format_generated_artifact_summary,
    _is_repairable_unresolved_source as _is_repairable_unresolved_source,
    relationship_topic as relationship_topic,
)
from pipeline.types import DebugState, PipelineResponse, _TurnState as _TurnState
from pipeline.core import CognitivePipeline

__all__ = [
    "CognitivePipeline",
    "DebugState",
    "PipelineContext",
    "PipelineResponse",
]

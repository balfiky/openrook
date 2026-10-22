"""Cognitive pipeline with one versioned choice owner.

Perception and context retrieval -> personality-dependent relationship update ->
pure willingness optimization -> immutable component grants -> scoped execution
and receipts -> expression from continuous state -> tri-state validation and
bounded correction -> delivery and grounded outcome records.

Ordinary conversation needs one expression call, or two with semantic appraisal.
Constrained expression adds review; a rewrite adds generation and review. The
base budget is five calls, including supplemental tool/learning inference. Only
when bounded drafts leave no safe remainder may one protected final expression
attempt use an additional call, recorded as delivery_recovery. Each
granted task also has one protected primary tool selection, fully accounted. No
mandatory inner dialogue or user-turn self-intent proposals run in this policy.
Independent activity has its own decision and execution scope.
"""

from core.types import PipelineContext
from pipeline.heuristics import (
    _BROAD_REPAIR_RE as _BROAD_REPAIR_RE,
    _CONVERSATION_HISTORY_MAX as _CONVERSATION_HISTORY_MAX,
    _IDENTITY_QUESTION_RE as _IDENTITY_QUESTION_RE,
    _REPAIRABLE_UNRESOLVED_SOURCES as _REPAIRABLE_UNRESOLVED_SOURCES,
    _clean_topic_hint as _clean_topic_hint,
    _format_generated_artifact_summary as _format_generated_artifact_summary,
    _is_broad_repair_text as _is_broad_repair_text,
    _is_identity_question as _is_identity_question,
    _is_repairable_unresolved_source as _is_repairable_unresolved_source,
    _message_for_model_tool_routing as _message_for_model_tool_routing,
    _message_for_tool_detection as _message_for_tool_detection,
    _relationship_topic_hint as _relationship_topic_hint,
)
from pipeline.types import DebugState, PipelineResponse, _TurnState as _TurnState
from pipeline.core import CognitivePipeline

__all__ = [
    "CognitivePipeline",
    "DebugState",
    "PipelineContext",
    "PipelineResponse",
]

"""Cognitive pipeline orchestrator.

v2 processing flow:
 1. Anticipation: predict emotional trajectory (0 LLM calls)
 2. Contagion: detect user tone → bounded mirror (0 LLM calls — rule-based)
 3. Appraisal: infer target, intent, vulnerability, affiliation (0 LLM calls)
 4. Context switch: load person profile baseline_shift
 5. Event classification: categorize input (0 LLM calls — rule-based)
 6. PSI engine: update 6 modulators from input + drives + energy
 7. Resolution update: check for new/resolved tension items (0 LLM calls)
 8. Short-term memory: store emotional reaction
 9. Spike check: if intensity > 0.8 → heavy write to LT
10. Memory retrieval: ACT-R activation biased by current state
11. Profile lookup: person + self + topic
12. Contradiction check: compare against profiles (self + others)
13. Inner dialogue: fast/slow deliberation only for non-spike unresolved tension
    (0-5 LLM calls; spike-only turns skip)
13b. Tool loop: detect intent → arbiter → execute → appraise (0+ tool calls;
     skipped if no tool_executor configured)
14. Defense mechanisms: filter output if needed (0 LLM calls)
15. Master LLM: generate final response (1 LLM call)
16. Self-check: rule-based default; LLM only for extreme/high-risk turns
    (0-1 LLM calls)
17. Post-processing: update memory, drain energy
18. [Session end] Digestion (0-1 LLM call)

LLM call budget: 1-6 per message (typical: 1). Spike-only hostility should stay
on the generator path unless a separate high-risk condition warrants LLM
self-check. Inner dialogue only fires when non-spike unresolved items exist
(contradictions, deadlocks, etc.).
"""

from core.types import PipelineContext
from pipeline.heuristics import (
    _BROAD_REPAIR_RE,
    _CONVERSATION_HISTORY_MAX,
    _IDENTITY_QUESTION_RE,
    _REPAIRABLE_UNRESOLVED_SOURCES,
    _clean_topic_hint,
    _format_generated_artifact_summary,
    _is_broad_repair_text,
    _is_identity_question,
    _is_repairable_unresolved_source,
    _message_for_model_tool_routing,
    _message_for_tool_detection,
    _relationship_topic_hint,
)
from pipeline.types import DebugState, PipelineResponse, _TurnState
from pipeline.core import CognitivePipeline

__all__ = [
    "CognitivePipeline",
    "DebugState",
    "PipelineContext",
    "PipelineResponse",
]

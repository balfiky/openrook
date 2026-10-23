"""Pipeline dataclasses: debug state, response, and per-turn state."""

from __future__ import annotations

from core.clock import clock as time
from dataclasses import dataclass, field
from typing import Any

from core.character_vector import CharacterVector
from core.dual_process.self_check import CoherenceVerdict
from core.life_influence import LifeInfluence
from core.tool_memory import ToolMemoryEffects
from core.willingness import RelationshipDynamics, TurnDecision
from core.types import (
    ActionVariables,
    AffectState,
    AgencyDecision,
    Anticipation,
    AppraisalFrame,
    DefenseActivation,
    DetectedEmotion,
    EmotionalEvent,
    InnerDialogueTrace,
    LongTermEntry,
    PersonProfile,
    PipelineContext,
    ProactiveTrace,
    RelationshipContext,
    ResponseStrategy,
    SelfIntent,
    SelfProfile,
    SemanticMemoryEntry,
    StrategyDecisionTrace,
    TaskTrace,
    ToolTrace,
    TopicProfile,
    UnresolvedItem,
)

# ---------------------------------------------------------------------------
# Debug state — full transparency into what happened
# ---------------------------------------------------------------------------

@dataclass
class DebugState:
    """Complete debug snapshot of a single pipeline run."""
    timestamp: float = field(default_factory=time.time)
    user_message: str = ""
    user_id: str = ""

    # Step 2: Contagion
    detected_emotion: DetectedEmotion | None = None

    # Step 3: Social appraisal
    appraisal_frame: AppraisalFrame | None = None

    # Step 4: Context switch
    baseline_shift_applied: dict[str, float] = field(default_factory=dict)

    # Step 5: Modulator update
    event_classified: str = ""
    event_intensity: float = 0.0
    is_spike: bool = False
    modulator_snapshot: dict[str, float] = field(default_factory=dict)

    # Step 7: Memory retrieval
    retrieved_memories: list[LongTermEntry] = field(default_factory=list)
    semantic_memories: list[SemanticMemoryEntry] = field(default_factory=list)
    life_history_context: dict[str, Any] = field(default_factory=dict)
    self_development: dict[str, Any] = field(default_factory=dict)
    character_vector: CharacterVector | None = None
    coherence_verdict: CoherenceVerdict | None = None
    life_influence: LifeInfluence = field(default_factory=LifeInfluence)
    life_influence_effects: dict[str, Any] = field(default_factory=dict)
    skill_context: dict[str, Any] = field(default_factory=dict)
    relationship_context: RelationshipContext | None = None
    # Current ledger for display; keep retrieved context as evidence of memory used.
    relationship_context_after_turn: RelationshipContext | None = None

    # Step 8: Profiles
    person_profile: PersonProfile | None = None
    self_profile: SelfProfile | None = None
    topic_profiles: list[TopicProfile] = field(default_factory=list)

    # Step 9: Contradictions
    contradiction_flags: list[str] = field(default_factory=list)

    # Steps 10-11: Generation
    response: str = ""
    self_check_passed: bool = True
    self_check_issues: list[str] = field(default_factory=list)
    correction_note: str = ""
    generation_attempts: int = 0
    raw_response: str = ""
    deliberate_silence: bool = False
    social_state: dict[str, Any] = field(default_factory=dict)
    short_replies: bool = False
    grounding_outcome: str = "clear"
    dignity_outcome: str = "clear"
    dignity_issues: list[str] = field(default_factory=list)
    decision_snapshots: list[dict[str, Any]] = field(default_factory=list)
    turn_decision: TurnDecision | None = None
    expression_reviews: list[dict[str, Any]] = field(default_factory=list)
    expression_excision: dict[str, Any] = field(default_factory=dict)
    expression_delivery: dict[str, Any] = field(default_factory=dict)
    delivery_outcome: str = "pending"
    residual_semantic_risk: bool = False

    # Step 14: Energy
    energy_after: float = 0.0

    # Emotion label (for display)
    emotion_label: str = ""

    # v2: Anticipation
    anticipation: Anticipation | None = None

    # v2: Inner dialogue
    dialogue_trace: InnerDialogueTrace | None = None

    # v2: Defense mechanisms
    defense_activation: DefenseActivation | None = None

    # v2: Resolution
    unresolved_count: int = 0
    unresolved_items: list[UnresolvedItem] = field(default_factory=list)

    # Agentic tools (Phase 0+)
    tool_trace: ToolTrace | None = None
    action_variables: ActionVariables | None = None
    tool_memory_effects: ToolMemoryEffects | None = None
    task_trace: TaskTrace | None = None
    strategy_trace: StrategyDecisionTrace | None = None

    # Phase 11.3: Response strategy
    response_strategy: str = ""

    # Derived affect and agency
    affect_state: AffectState | None = None
    agency_decision: AgencyDecision | None = None
    autonomy_level: str = ""

    # Self-action layer (v0.30): what OpenRook chose to do for itself this turn.
    self_intents: list[SelfIntent] = field(default_factory=list)

    # Proactive behavior (Phase 8)
    proactive_trace: ProactiveTrace | None = None

    # Timing instrumentation (ms)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Pipeline response
# ---------------------------------------------------------------------------

@dataclass
class PipelineResponse:
    """What the pipeline returns to the caller."""
    response: str
    debug: DebugState
    backend_error: Exception | None = field(default=None, repr=False, compare=False)

    def raise_for_backend_failure(self) -> None:
        """Transports report service failure after retaining the turn's debug."""
        if self.debug.delivery_outcome == "backend_failure":
            raise self.backend_error or RuntimeError("Backend returned no expression.")
        if not self.response.strip() and not self.debug.deliberate_silence:
            raise RuntimeError("Expression generation returned no deliverable text.")


# ---------------------------------------------------------------------------
# Per-turn state threaded through the process() stage methods
# ---------------------------------------------------------------------------

@dataclass
class _TurnState:
    """Mutable per-turn state shared by the process() stage methods."""
    user_message: str
    user_id: str
    debug: DebugState
    social_state: Any = None
    dynamics: RelationshipDynamics = field(default_factory=RelationshipDynamics)
    turn_decision: TurnDecision | None = None
    turn_id: str = ""
    elapsed: float = 0.0
    trust_delta: float = 0.0
    short_replies: bool = False
    inspection_kind: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    t_total: float = 0.0
    turn_started_at: float = 0.0
    person: PersonProfile | None = None
    detected: DetectedEmotion | None = None
    life_history_context: dict[str, Any] = field(default_factory=dict)
    developing_self: dict[str, Any] = field(default_factory=dict)
    life_influence: LifeInfluence = field(default_factory=LifeInfluence)
    appraisal: AppraisalFrame | None = None
    event: EmotionalEvent | None = None
    retrieved: list[LongTermEntry] = field(default_factory=list)
    self_prof: SelfProfile | None = None
    active_topics: list[TopicProfile] = field(default_factory=list)
    relationship_topic: str = ""
    relationship_context: RelationshipContext | None = None
    semantic_memories: list[SemanticMemoryEntry] = field(default_factory=list)
    skill_context: dict[str, Any] = field(default_factory=dict)
    contradiction_flags: list[str] = field(default_factory=list)
    affect_state: AffectState | None = None
    agency_decision: AgencyDecision | None = None
    dialogue_trace: InnerDialogueTrace | None = None
    character_vector: CharacterVector | None = None
    tool_context_summary: str = ""
    filtered_output: str = ""
    defense: DefenseActivation | None = None
    strategy: ResponseStrategy | None = None
    ctx: PipelineContext | None = None
    gen_result: Any = None
    backend_error: Exception | None = None
    self_directed: bool = False
    action_context: dict[str, Any] | None = None

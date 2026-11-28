"""CognitivePipeline — the turn orchestrator.

``CognitivePipeline.process()`` runs five stages over a ``_TurnState``;
see the package docstring (``pipeline/__init__.py``) for the full
18-step flow and LLM call budget.
"""

from __future__ import annotations

import json
import logging
import re
from core.clock import clock as time
from core.clock import identifiers as uuid
from dataclasses import asdict, replace
from typing import Any, Callable
from types import SimpleNamespace

from config.loader import get_config, research_configuration
from runtime.developing_self import birth_state, project_self, soul_from_self
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from core.inference import call_telemetry, expression_retry, final_expression_attempt, generate, inference_phase, inference_turn, primary_tool_inference, remaining_calls, reserve_expression_calls
from core.types import (
    AppraisalFrame,
    DetectedEmotion,
    DefenseActivation,
    EmotionalEvent,
    EventType,
    InnerDialogueTrace,
    LongTermEntry,
    ModulatorState,
    PipelineContext,
    RelationshipEvent,
    SelfIntent,
    TaskPlan,
    TaskStep,
    TaskStatus,
    ToolCategory,
    ToolDecision,
    ToolIntent,
    ToolObservation,
    ToolResult,
    ToolTrace,
    TopicProfile,
    UnresolvedItem,
    ValueHierarchy,
)
from core.appraisal import (
    CONTEXT_WINDOW,
    MessageReader,
    appraise_with_life_history,
    read_message,
    reader_context,
    reconcile_social_appraisal,
)
from core.affect import resolve_affect
from core.strategy import agency_summary, decision_trace, summarize_decision
from core.dignity import STALE, correction_instruction, excisable_choice_issues
from core.review import ExpressionReviewer, receipt_projection, review_draft
from core.action_variables import derive_action_variables
from core.task_planning import summarize_plan_status
from core.artifact_writer import ArtifactWriteRequest, generate_artifact_draft
from core.grants import execution_metadata, execution_scope, turn_stages
from core.perception import fold
from core.request_routing import (
    LEARNING, LEARNING_GRANT, LEARNING_OFFER, Offer, Selection, held_operands, offers_of, plan_report, same_gap,
)
from core.willingness import (
    DecisionState, Personality, RelationshipDynamics, RequestComponent,
    commit_relationship, decide, update_relationship,
)
from core.emotional_engine import EmotionalEngine
from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.memory.digestion import digest_session, DigestedSession
from core.memory.relationship import NullRelationshipMemory, RelationshipMemory
from core.memory.semantic import (
    BRIEF_REPLIES_TAG,
    NullSemanticMemory,
    create_semantic_memory,
    derive_semantic_entries,
)
from core.life_influence import LifeInfluence, curiosity_salience_bonus, derive_life_influence
from core.character_vector import assemble_character_vector, format_character_vector_prompt
from core.pipeline_features import PipelineFeatures
from core.contagion import neutral_tone
from core.profiles.base import ProfileStore
from core.profiles.person import PersonProfileManager
from core.profiles.self_model import SelfProfileManager, SELF_ENTITY_ID
from core.profiles.topic import TopicProfileManager
from core.profiles.contradiction import ContradictionDetector
from core.dual_process.generator import (
    expression_memory_evidence,
    LLMBackend,
    GenerationResult,
    ResponseGenerator,
    prompt_memory_context,
)
from core.dual_process.self_check import SelfChecker
from core.dual_process.inner_dialogue import InnerDialogue
from core.anticipation import AnticipationEngine
from core.defense_mechanisms import DefenseMechanism
from core.self_intent import (
    conversation_proposals, conversation_sources, ground_conversation_allocation, propose_self_intents,
)
from openrook_tools.builtin.self_actions import SelfActionContext
from core.dual_process.tool_loop import (
    ToolLoopResult,
    continue_plan,
    make_tool_decision,
    record_operands,
    run_tool_loop,
)
from core.tool_appraisal import appraise_tool_result
from core.proactive import collect_skill_want_triggers, evaluate_proactive
from core.tool_memory import (
    ToolMemoryEffects,
    create_long_term_entry,
    create_operational_issue,
    create_task_long_term_entry,
    create_task_unresolved_item,
    create_tool_event,
    create_tool_unresolved_item,
    derive_task_self_observations,
    derive_tool_self_observations,
    is_salient_episode,
)

from pipeline.heuristics import (
    _CONVERSATION_HISTORY_MAX,
    _format_generated_artifact_summary,
    relationship_topic as topic_of,
)
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState, PipelineResponse, _TurnState

log = logging.getLogger(__name__)

# How many turns model tool selection is left alone after the backend failed a
# tool-calling request as unsupported or at the transport level. It is offered
# again afterwards, as structured output is (``runtime/llm/backend.py``).
TOOL_SELECTION_PAUSE_TURNS = 10
KNOWN_GAPS = 8                   # how many known capability gaps the reader is shown
SELECTION_TURN_CHARS = 1500      # how much of one earlier turn the selecting model is shown
SELECTION_RECEIPT_CHARS = 1200   # how much of one receipt's output it is shown
WRITE_FILE = "fs.write_file"
WRITE_PATH = "path"   # the one operand of a file write that the person writes when its content is to be produced


def default_message_reader(pipeline: "CognitivePipeline") -> MessageReader:
    """The reader a pipeline uses when none is injected: the model's reading.

    This is the single place a pipeline's reader is chosen. There is no
    alternative reader in the runtime.
    """
    del pipeline
    return read_message


def default_expression_reviewer(pipeline: "CognitivePipeline") -> ExpressionReviewer:
    """The reviewer a pipeline uses when none is injected: the model's review.

    This is the single place a pipeline's reviewer is chosen. There is no
    alternative reviewer in the runtime.
    """
    del pipeline
    return review_draft


# ---------------------------------------------------------------------------
# CognitivePipeline
# ---------------------------------------------------------------------------

class CognitivePipeline:
    """Orchestrates the full v2 cognitive processing flow."""

    def __init__(
        self,
        llm_backend: LLMBackend | None = None,
        llm_backend_fast: LLMBackend | None = None,
        db_path: str = ":memory:",
        self_db_path: str | None = None,
        tool_executor: Any | None = None,
        autonomy_level: str = "autonomous",
        coherence_min_score: float = 0.6,
        coherence_max_regenerations: int = 2,
        features: PipelineFeatures | None = None,
        life_history_provider: Callable[[], dict[str, Any] | None] | None = None,
        life_history_snapshot_provider: Callable[[], dict[str, Any] | None] | None = None,
        skill_provider: Callable[[], dict[str, Any] | None] | None = None,
        runtime_config: Any | None = None,
        component_executor: Callable[..., ToolResult | None] | None = None,
        message_reader: MessageReader | None = None,
        expression_reviewer: ExpressionReviewer | None = None,
    ) -> None:
        """Create a cognitive pipeline.

        Args:
            llm_backend: Primary LLM backend.
            llm_backend_fast: Fast backend (thinking off). Falls back to primary.
            db_path: Per-user database path for memories, person profiles,
                     topic profiles, and person observations.
            self_db_path: Shared database path for self-model observations and
                          defense history.  When None, falls back to db_path
                          (single-DB mode, backward compatible).
            tool_executor: Optional ToolExecutor for agentic tool use.
                           When None, the tool loop is skipped entirely.
            autonomy_level: Tool autonomy mode (off, assisted, autonomous,
                            high_risk). Direct pipeline use defaults to
                            autonomous for backwards compatibility.
            features: Feature toggles for ablation runs. Default (None)
                      enables every component. See ``PipelineFeatures``.
            life_history_provider: Optional callable returning a compact
                                   identity-level life-history context for
                                   generation. Runtime sessions wire this to
                                   ``data/shared/life_history.db``.
            skill_provider: Optional callable returning enabled imported skill
                            guidance for generation.
            component_executor: Optional executor of the session's own
                                capability, learning intake. When one is wired
                                the reader is offered that capability, and a
                                granted task that names it is handed over.
            message_reader: Optional reader of each user message
                            (``core.appraisal.MessageReader``). Defaults to
                            the model reader; every turn is read by it.
            expression_reviewer: Optional reviewer of each draft reply
                                 (``core.review.ExpressionReviewer``). Defaults
                                 to the model reviewer; every draft that could
                                 be delivered is reviewed by it.
        """
        self._features = features or PipelineFeatures()
        if llm_backend is None:
            raise ValueError(
                "CognitivePipeline requires an llm_backend. Configure "
                "llm_backend in runtime_config.yaml — OpenRook never runs against a "
                "fake/mock backend."
            )
        # Primary backend (used if no fast backend provided)
        self._llm_backend = llm_backend
        # Fast backend (thinking mode off) — used for ALL calls
        # Generator prompt already has full context; thinking overhead not needed
        self._llm_backend_fast = llm_backend_fast or self._llm_backend
        self._research_mode = bool(getattr(runtime_config, "research_mode", False))
        self._research_action_framing = getattr(runtime_config, "research_action_framing", "agent_owned")
        self._research_owner = None
        self._research_in_transition = False
        self._config = getattr(runtime_config, "_research_configuration", None) or get_config()
        if self._research_mode and not getattr(runtime_config, "_research_configuration", None):
            self._config = research_configuration(getattr(runtime_config, "research_birth_policy", "uncommitted"))
        self.soul = self._config.soul
        self._birth_self = birth_state(self.soul)
        self._effective_self = self._birth_self
        self._semantic_cfg = self._config.semantic_memory

        # Core engine
        self.engine = EmotionalEngine(config=self._config if self._research_mode else None)

        # Memory
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(db_path=db_path)
        # Relationship and semantic memory honor the feature toggles.
        # Disabled components return a null object whose writes are no-ops
        # and whose reads are empty — see core/pipeline_features.py.
        if self._features.relationship_memory:
            self.relationship_memory = RelationshipMemory(db_path=db_path)
        else:
            self.relationship_memory = NullRelationshipMemory()
        if self._features.semantic_memory:
            self.semantic_memory = create_semantic_memory(
                db_path=db_path,
                config=self._semantic_cfg,
            )
        else:
            self.semantic_memory = NullSemanticMemory()
        self._saved_session_memory_writes = 0
        self._memory_write_baseline = 0

        # Per-user profile store (person observations + extracted traits)
        self._person_profile_store = ProfileStore(db_path=db_path)
        self.person_profiles = PersonProfileManager(self._person_profile_store, db_path=db_path)
        self.topic_profiles = TopicProfileManager(db_path=db_path)

        # Shared self-model store (self observations + defense events)
        # WAL mode + busy timeout when shared DB is explicitly separated
        effective_self_db = self_db_path if self_db_path is not None else db_path
        self._self_profile_store = ProfileStore(
            db_path=effective_self_db,
            wal_mode=(self_db_path is not None),
        )
        self.self_profile = SelfProfileManager(self._self_profile_store)

        # Contradiction detectors — one per store
        self._person_contradiction = ContradictionDetector(self._person_profile_store)
        self._self_contradiction = ContradictionDetector(self._self_profile_store)

        # Master generator uses fast backend (thinking off — prompt has full context)
        self.generator = ResponseGenerator(backend=self._llm_backend_fast)
        # Self-checker: rule-based by default; LLM only for high-intensity turns
        self.self_checker = SelfChecker(llm_client=None)
        self._self_check_llm = SelfChecker(llm_client=self._llm_backend_fast)

        # Inner dialogue uses fast backend (no thinking needed for gut reaction + evaluation)
        self.inner_dialogue = InnerDialogue(backend=self._llm_backend_fast)
        self.anticipation_engine = AnticipationEngine()
        self.defense_mechanism = DefenseMechanism()

        # Values are seeded from the authored soul and remain stable unless updated deliberately.
        self.values = ValueHierarchy(values=dict(self._config.values))

        # Conversation history for context
        self._conversation_history: list[dict[str, str]] = []
        # Delivery freshness covers the whole chat, independently of the
        # bounded prompt transcript. Preserve this ledger across restarts.
        self._assistant_replies: dict[str, None] = {}
        self._receipt_history: list[ToolResult] = []
        self._dynamics: RelationshipDynamics | None = None
        self._turn_index = 0
        self._last_decision = None
        self._short_replies = False

        # Agentic tools (Phase 2+)
        self._tool_executor = tool_executor
        self._tool_runner = tool_executor.tool_runner if tool_executor is not None else None
        self._autonomy_level = autonomy_level
        # Every user message is read by the model. The reading is required
        # work of the turn and has no local substitute; ``appraisal_use_llm``
        # in older configuration files no longer switches anything.
        self._message_reader: MessageReader = message_reader or default_message_reader(self)
        # Every draft that could be delivered is reviewed against the frozen
        # decision by the model. That review is required work too, and nothing
        # about a draft's wording decides whether it happens.
        self._expression_reviewer: ExpressionReviewer = expression_reviewer or default_expression_reviewer(self)
        # Consolidate ordinary episodes; the store bounds identity development.
        self._autobiographical_learning = bool(
            getattr(runtime_config, "autobiographical_learning", False)
        )
        self._coherence_min_score = coherence_min_score
        self._coherence_max_regenerations = coherence_max_regenerations
        self._life_history_provider = life_history_provider
        self._life_history_snapshot_provider = life_history_snapshot_provider
        self._skill_provider = skill_provider
        # ``runtime_config`` is the RuntimeConfig used by the session manager.
        # The legacy ``self._config`` is the global OpenRookConfig from get_config()
        # which has no data_dir, so paths to shared SQLite databases need this.
        self._runtime_config = runtime_config
        self._component_executor = component_executor
        self.last_intake_receipt: str = ""
        self._owned_pursuits: dict[str, dict] = {}
        self._development_history: list[dict] = []
        self._development_episode_id = uuid.uuid4().hex
        self._development_checkpointed = False

        # Session-scoped task plan (Phase 7)
        self._active_task_plan: TaskPlan | None = None
        # Turns left during which model tool selection is not asked for, after
        # the backend failed a tool-calling request.
        self._tool_selection_pause = 0
        # Proactive behavior tracking (Phase 8)
        self._proactive_count: int = 0
        self._last_proactive_at: float | None = None
        self._capability_gaps: dict[str, dict[str, Any]] = {}

        # Time tracking for auto-decay between turns
        self._last_turn_time: float | None = None
        self._last_external_input_at: float | None = None
        self._last_state_update_at: float | None = None
        self._episode_id = uuid.uuid4().hex
        # Pipelines are session-scoped; changing users on one instance leaks state.
        self._bound_user_id: str | None = None

        # Self-action layer (v0.30). The proposer runs whenever a
        # ``tool_executor`` is attached. No catalog filter, no per-turn cap —
        # OpenRook can invoke any registered tool. Legacy callers that pass
        # ``tool_executor=None`` (most existing test fixtures) naturally
        # bypass the stage.
        if tool_executor is not None:
            existing_ctx = tool_executor.self_action_context
            data_dir = getattr(runtime_config, "data_dir", "") or ""
            owner_chat_id = getattr(runtime_config, "owner_chat_id", "") or ""
            telegram_token = getattr(runtime_config, "telegram_token", "") or ""
            ctx = existing_ctx or SelfActionContext()
            ctx.data_dir = ctx.data_dir or data_dir
            if not ctx.owner_chat_id:
                ctx.owner_chat_id = owner_chat_id
            if not ctx.telegram_token:
                ctx.telegram_token = telegram_token
            ctx.state_provider = self._self_action_state_snapshot
            ctx.web_executor = self._self_action_web_executor
            tool_executor.set_self_action_context(ctx)

    @property
    def tool_executor(self) -> Any | None:
        """Attached tool executor; exists for the v1 introspection surface."""
        return self._tool_executor

    # ------------------------------------------------------------------
    # Self-action context helpers (wired into SelfActionContext)
    # ------------------------------------------------------------------

    def _self_action_state_snapshot(self) -> dict[str, Any]:
        """Snapshot for ``self.snapshot_state``: modulators + last turns."""
        try:
            mods = self.engine.snapshot()
            label = self.engine.to_emotion_label()
        except Exception:
            mods, label = {}, ""
        recent: list[dict[str, Any]] = []
        try:
            for entry in self.short_term.recent(5):
                recent.append({
                    "timestamp": entry.timestamp,
                    "intensity": float(entry.event.intensity),
                    "kind": getattr(entry.event, "kind", ""),
                })
        except Exception:
            recent = []
        return {
            "modulators": mods,
            "emotion_label": label,
            "recent_turns": recent,
        }

    def _self_action_web_executor(self, args: dict[str, Any]) -> ToolResult:
        """Delegate ``self.verify`` to the executor's ``web.search`` handler."""
        executor = self._tool_executor
        if executor is None:
            return ToolResult(
                tool_name="self.verify", success=False, output="",
                error="No tool executor available for verification.",
            )
        return executor.execute("web.search", args)

    def _build_tool_catalog_for_intent(self) -> list[dict[str, Any]]:
        """Dump the full live tool catalog for the self-intent proposer."""
        executor = self._tool_executor
        if executor is None:
            return []
        registry = executor.registry
        catalog: list[dict[str, Any]] = []
        try:
            names = registry.names()
        except Exception:
            return []
        for name in names:
            cap = registry.get(name)
            if cap is None:
                continue
            catalog.append({
                "name": name,
                "description": cap.description,
                "category": cap.category.value if hasattr(cap.category, "value") else str(cap.category),
                "arg_schema": dict(cap.arg_schema) if cap.arg_schema else {},
            })
        return catalog

    # ------------------------------------------------------------------
    # Requests by understanding (docs/development/request-routing.md)
    # ------------------------------------------------------------------

    def _capability_offers(self) -> tuple[Offer, ...]:
        """What the character can reach this turn, as the reader is told of it.

        Every registered tool, and the session's learning intake when a
        session wired its executor. The same set decides whether a component's
        capability requirements can be met.
        """
        offers = offers_of(self._tool_executor.registry) if self._tool_executor is not None else ()
        if self._component_executor is not None:
            offers = (*offers, LEARNING_OFFER)
        return offers

    def _running_plan(self) -> TaskPlan | None:
        """The plan a person's turn may carry on or ask about.

        It must be running, and a committed decision must have started it
        (``origin_decision_id``), the rule the character's own continuation
        already follows. A plan with no originating decision was never
        granted by anyone: the reader is not told of it, so no part of a
        message can continue it. See docs/development/request-routing.md for
        what is and is not checked when a plan is carried on.
        """
        plan = self._active_task_plan
        return plan if plan is not None and not plan.is_terminal and plan.origin_decision_id else None

    def _add_execution_detail(self, appraisal: AppraisalFrame, plan: TaskPlan | None) -> AppraisalFrame:
        """Add what execution needs to components the reading reported. Nothing else.

        No component is added, removed or replaced here, and none changes its
        kind, description or price: a role or a piece of conversation in the
        same message reaches the decision exactly as it was read.

        * A task the reading tied to the running plan as its continuation
          carries the plan's id, so that a grant of it carries the plan on,
          and names the capabilities the plan's remaining steps use, so that
          the grant covers those and no other.
        * A task the reading reported as asking for learning intake carries
          the mark the session's learning executor requires.
        """
        relations = dict(appraisal.plan_relations)
        components = []
        for component in appraisal.request_components:
            if plan is not None and relations.get(component.id):
                reference = f"plan:{plan.id}"
                component = replace(
                    component, evidence_refs=tuple(dict.fromkeys((*component.evidence_refs, reference))))
                if relations[component.id] == "continue":
                    remaining = (step.tool_name for step in plan.steps if step.status != TaskStatus.COMPLETED)
                    component = replace(component, continuation_of=plan.id, capability_requirements=tuple(
                        dict.fromkeys((*component.capability_requirements, *remaining))))
            if component.kind == "task" and component.requested_tool == LEARNING:
                component = replace(component, evidence_refs=tuple(dict.fromkeys(
                    (*component.evidence_refs, LEARNING_GRANT))))
            components.append(component)
        return replace(appraisal, request_components=tuple(components))

    def _note_unavailable_work(self, appraisal: AppraisalFrame) -> None:
        """Keep work that nothing offered can do as a capability gap.

        The reading says what a task needs that no offered capability does.
        The same need reported again, in whatever words, counts as the same
        gap (``core.request_routing.same_gap``), and how much it matters is
        how much the person wanted that work. Repeated gaps are what the
        character's own wish for a new skill starts from (``core/proactive.py``).
        """
        missing = dict(appraisal.unavailable)
        gaps = []
        for component in appraisal.request_components:
            described = missing.get(component.id)
            if not described:
                continue
            known = next((key for key, gap in self._capability_gaps.items()
                          if same_gap(described, str(gap.get("gap_type") or ""))), f"gap:{fold(described)}")
            gaps.append({"id": known, "gap_type": described, "recent_recurrence": 1,
                         "frustration_intensity": component.user_value})
        self._record_capability_gaps(gaps)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, user_message: str, user_id: str = "default") -> PipelineResponse:
        calls = []
        with inference_turn(calls, research=self._research_mode, max_calls=5,
                            request_timeout_seconds=getattr(self._runtime_config, "llm_request_timeout_seconds", None),
                            auxiliary_timeout_seconds=getattr(self._runtime_config, "llm_auxiliary_timeout_seconds", None),
                            auxiliary_cooldown_seconds=getattr(self._runtime_config, "llm_auxiliary_cooldown_seconds", 30.0),
                            reading_output_tokens=getattr(self._runtime_config, "llm_reading_max_output_tokens", None)):
            result = self._process_turn(user_message, user_id)
            result.debug.llm_calls = calls
            return result

    def _process_turn(self, user_message: str, user_id: str = "default") -> PipelineResponse:
        """Process a user message through the full v2 cognitive pipeline.

        Flow: anticipation → contagion → appraisal → context → event →
              resolution → memory → profiles → contradiction → inner dialogue →
              defense → master LLM → post-processing.

        Returns the response text and full debug state.
        """
        if self._research_owner is not None and not self._research_in_transition:
            return self._research_owner.process(self, user_message, user_id)
        self._ensure_bound_user(user_id)
        st = _TurnState(
            user_message=user_message,
            user_id=user_id,
            debug=DebugState(user_message=user_message, user_id=user_id),
        )

        # Stage 1: deterministic pre-pass (Steps 0-11b). May short-circuit
        # with a direct identity answer.
        # Everything executed in these stages belongs to a granted component and runs inside its
        # execution scope. A call that reaches the executor here without one is denied.
        with turn_stages():
            short_circuit = self._stage_pre_pass(st)
            if short_circuit is not None:
                return short_circuit

            # Stage 2: gated deliberation — inner dialogue, tool loop,
            # self-intent, defense shaping, response strategy (Steps 12-13b).
            self._stage_deliberation(st)

            # Stage 3: master LLM generation (Step 14).
            self._stage_generate(st)

            # Stage 4: rule-based + LLM self-check, coherence, grounding,
            # tone floor.
            self._stage_self_check(st)

            # Stage 5: post-processing — memory writes, energy drain, debug
            # trace (Steps 14b-15).
            return self._stage_post_process(st)

    # ------------------------------------------------------------------
    # Stage 1: deterministic pre-pass
    # ------------------------------------------------------------------

    def _stage_pre_pass(self, st: _TurnState) -> PipelineResponse | None:
        """Deterministic pre-pass (Steps 0-11b).

        Perception, engine update, memory/profile retrieval, and derived
        affect. Returns a short-circuit PipelineResponse for identity
        questions, otherwise None.
        """
        debug = st.debug
        timings = st.timings
        user_message = st.user_message
        user_id = st.user_id
        person = self.person_profiles.get_or_create(user_id)
        st.t_total = time.perf_counter()

        # ---- Step 0: Auto-decay based on elapsed time ----
        now = time.time()
        st.turn_started_at = now
        self._refresh_relationship_state(user_id)
        self._turn_index += 1
        st.turn_id = str(self._turn_index)
        previous_update = self._last_state_update_at if self._last_state_update_at is not None else self._last_turn_time
        if previous_update is not None:
            elapsed = now - previous_update
            st.elapsed = max(0.0, elapsed)
            if elapsed > 0:
                self.engine.decay(elapsed)
        self._last_turn_time = now
        self._last_external_input_at = now
        self._last_state_update_at = now

        # ---- Step 1: ANTICIPATION (0 LLM calls) ----
        _ts = time.perf_counter()
        recent_msgs = [
            m["content"] for m in self._conversation_history
            if m["role"] == "user"
        ][-5:]
        all_topics = self.topic_profiles.all_profiles()
        topic_dict = {tp.topic: tp for tp in all_topics}

        anticipation = self.anticipation_engine.predict(
            recent_messages=recent_msgs,
            person_profile=person,
            topic_profiles=topic_dict,
            current_state=self.engine.state,
            unresolved_items=self.engine.active_unresolved(),
        )
        debug.anticipation = anticipation
        timings["anticipation"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 2: TONE — unknown until the message has been read ----
        _ts = time.perf_counter()
        detected = neutral_tone()
        st.detected = detected
        debug.detected_emotion = detected
        timings["contagion"] = (time.perf_counter() - _ts) * 1000

        _ts = time.perf_counter()
        life_history_context = self._load_life_history_context(user_message)
        st.life_history_context = life_history_context
        self._capture_developing_self(st)
        self._project_conversation_commitments(st)
        timings["life_history_retrieval"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 3: READING (required; the model reads every message) ----
        # No greeting, vocabulary, host-metric or latency condition skips it,
        # and nothing stands in for it: an unobtainable reading is recorded as
        # plain conversation with explicit uncertainty. The reading also says
        # what each task asks a capability for; nothing below recognises a
        # request, a tool or a plan follow-up by its wording.
        _ts = time.perf_counter()
        st.tool_selection_paused = self._tool_selection_pause > 0
        if st.tool_selection_paused:
            self._tool_selection_pause -= 1
        offers = self._capability_offers()
        plan = self._running_plan()
        reading = self._message_reader(
            user_message, backend=self._llm_backend_fast,
            recent_context=self._conversation_history,
            agent_context=reader_context(
                state=self.engine.snapshot(), authored_values=self.values.values,
                dispositions=st.developing_self["dispositions"], current_self=st.developing_self,
                conversation_sources=conversation_sources(st.developing_self, life_history_context,
                    self.engine.state.energy) if self._features.self_intent else {},
                available_capabilities=offers, identity=life_history_context,
                running_plan=plan_report(plan),
                # Things already found to be missing, in the words they were reported in, so that
                # the same need comes back in the same words.
                known_gaps=[str(gap.get("gap_type") or "") for gap in self._capability_gaps.values()
                            if gap.get("gap_type")][-KNOWN_GAPS:]),
            research=self._research_mode,
        )
        base_appraisal = reconcile_social_appraisal(user_message, None, reading,
                                                    recent_context=self._conversation_history,
                                                    available_capabilities=offers, plan_running=plan is not None)
        attempts = [dict(item) for item in base_appraisal.reading_attempts]
        debug.reading = dict(provenance=base_appraisal.provenance, reason=base_appraisal.reason,
                             language=base_appraisal.language, attempts_used=len(attempts),
                             truncated=any(item.get("truncated") for item in attempts), attempts=attempts,
                             about_self=base_appraisal.about_self, topics=list(base_appraisal.topics),
                             # What the reading said each role would do to the character, the reader's own
                             # number and the floor applied, kept so that the choice can be studied later.
                             roles=[{"component": key, **facts.to_dict()}
                                    for key, facts in base_appraisal.role_facts.items()])
        if self._features.life_history_context and base_appraisal.about_self:
            # The message asks about the character's own past and development:
            # its identity records are part of this turn's context.
            identity = self._load_life_history_snapshot()
            life_history_context = {**life_history_context, **identity, "identity_records": identity}
            evolution = identity.get("evolution_events")
            if evolution is not None:
                life_history_context["recent_evolution"] = evolution
            st.life_history_context = life_history_context
        debug.life_history_context = life_history_context
        life_influence = derive_life_influence(life_history_context)
        st.life_influence = life_influence
        debug.life_influence = life_influence
        if base_appraisal.perceived_emotion is not None:
            # Tone is part of the reading. Without one there is no tone to
            # catch: mood is left alone instead of drifting toward a midpoint.
            detected = base_appraisal.perceived_emotion
            st.detected = detected
            debug.detected_emotion = detected
            # Personality-dependent coupling, in proportion to how strongly the
            # speaker expresses the tone: a flat message transmits nothing.
            susceptibility = (1 - st.developing_self["traits"].get("calm", 0.0)) * person.trust
            self.engine.apply_contagion(detected.arousal, detected.valence, susceptibility * detected.intensity)
        appraisal = appraise_with_life_history(
            base_appraisal,
            user_message,
            life_history_context,
        )
        appraisal = self._add_execution_detail(appraisal, plan)
        self._note_unavailable_work(appraisal)
        st.appraisal = appraisal
        debug.appraisal_frame = appraisal
        personality = Personality.from_self(st.developing_self, st.user_id)
        previous = self._dynamics or RelationshipDynamics()
        elapsed = max(0.0, now - previous.updated_at) if previous.updated_at else st.elapsed
        dynamics, st.trust_delta = update_relationship(
            previous, personality, appraisal.perception,
            DecisionState(**self.engine.snapshot(), trust=person.trust),
            elapsed=elapsed, now=now, turn_id=st.turn_id,
        )
        st.dynamics = dynamics
        # Changes describe injury, not a category-triggered reaction. No energy
        # restoration, task-readiness clamp or pressure-derived bonding shift.
        dh, dr = dynamics.h - previous.h, dynamics.r - previous.r
        state = self.engine.state
        state.arousal = max(0.0, min(1.0, state.arousal + dh * (0.25 + 0.35 * personality.pride)))
        state.valence = max(0.0, min(1.0, state.valence - dh * (0.3 + 0.4 * personality.pride)))
        affiliation = appraisal.perception.affiliation * (1 - appraisal.perception.uncertainty)
        connection = affiliation * (1 - appraisal.perception.hostile_intent) * (1 - dynamics.h) * (1 - dynamics.r)
        state.bonding = max(0.0, min(1.0, state.bonding
            + 0.06 * connection * personality.value("kindness") * (1 - state.bonding)
            - dr * (0.15 + 0.2 * personality.pride)))
        debug.social_state = dynamics.to_dict()
        timings["appraisal"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 4: Context switch (non-additive — sets resting target) ----
        shift = self.person_profiles.get_baseline_shift(user_id)
        self.engine.set_context_shift(shift)
        debug.baseline_shift_applied = shift.to_dict()

        # ---- Step 5: Event classification + PSI engine update ----
        _ts = time.perf_counter()
        event = self._classify_event(user_message, detected, appraisal)
        st.event = event
        is_spike = False  # Social observations were integrated once by update_relationship.
        debug.event_classified = event.event_type.value
        debug.event_intensity = event.intensity
        debug.is_spike = is_spike
        timings["event_classification"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 6: Resolution update (0 LLM calls) ----
        # Check if this event creates new unresolved items
        self._check_resolution_sources(event, user_message, user_id)
        active_unresolved = self.engine.active_unresolved()
        debug.unresolved_count = len(active_unresolved)
        debug.unresolved_items = list(active_unresolved)

        debug.modulator_snapshot = self.engine.snapshot()

        # ---- Step 7: Short-term memory ----
        self.short_term.record(event, self.engine.state)

        # ---- Step 8: Spike check ----
        if is_spike:
            spike_entry = LongTermEntry(
                timestamp=time.time(),
                summary=f"Spike: {event.event_type.value} from {user_id}",
                emotional_valence=self._event_valence(event),
                trust_delta=LongTermMemory.compute_trust_delta(self._event_valence(event)),
                source_person=user_id,
                confidence=1.0,
                spike=True,
            )
            self.long_term.store_spike(spike_entry)

        # ---- Step 9: Memory retrieval ----
        _ts = time.perf_counter()
        retrieved = self.long_term.retrieve(
            self.engine.state,
            source_person=user_id,
            limit=5,
        )
        st.retrieved = retrieved
        debug.retrieved_memories = retrieved
        timings["memory_retrieval"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 10: Profile lookup ----
        person = self.person_profiles.get_or_create(user_id)
        st.person = person
        self_prof = self.self_profile.get_profile()
        st.self_prof = self_prof
        active_topics = self._detect_topics(user_message)
        st.active_topics = active_topics
        relationship_topic = active_topics[0].topic if active_topics else topic_of(appraisal.topics)
        st.relationship_topic = relationship_topic
        _ts = time.perf_counter()
        relationship_context = self.relationship_memory.build_context(
            user_id,
            topic=relationship_topic,
            event_limit=3,
            loop_limit=3,
        )
        if relationship_context.is_empty():
            relationship_context = None
        st.relationship_context = relationship_context
        debug.person_profile = person
        debug.self_profile = self_prof
        debug.topic_profiles = active_topics
        debug.relationship_context = relationship_context
        timings["relationship_retrieval"] = (time.perf_counter() - _ts) * 1000

        _ts = time.perf_counter()
        semantic_memories = self.semantic_memory.retrieve(
            user_message,
            source_person=user_id,
            topic=relationship_topic,
            limit=self._semantic_cfg.retrieval_limit,
        )
        st.semantic_memories = semantic_memories
        # A standing wish about reply length is something the reading reports
        # and memory keeps as a tagged record. No wording is matched here.
        preference = self.semantic_memory.response_preference(user_id)
        if preference is not None:
            self._short_replies = BRIEF_REPLIES_TAG in preference.tags
            if not any(item.id == preference.id for item in semantic_memories):
                semantic_memories.insert(0, preference)
        if appraisal.reply_length:
            self._short_replies = appraisal.reply_length == "shorter"
        st.short_replies = self._short_replies
        debug.short_replies = st.short_replies
        debug.semantic_memories = semantic_memories
        timings["semantic_memory_retrieval"] = (time.perf_counter() - _ts) * 1000

        _ts = time.perf_counter()
        skill_context = self._load_skill_context(user_message=user_message)
        st.skill_context = skill_context
        debug.skill_context = skill_context
        timings["skill_context_retrieval"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 11: Contradiction check ----
        contradiction_flags: list[str] = []

        person_expected = self.person_profiles.get_expected_traits(user_id)
        if person_expected:
            person_result = self._person_contradiction.detect(user_id, person_expected)
            for c in person_result.contradictions:
                contradiction_flags.append(c.description)

        self_expected = self._effective_self_expected_traits()
        if self_expected:
            self_result = self._self_contradiction.detect(SELF_ENTITY_ID, self_expected)
            for c in self_result.contradictions:
                contradiction_flags.append(c.description)

        st.contradiction_flags = contradiction_flags
        debug.contradiction_flags = contradiction_flags

        # Wire contradictions and topics as unresolved sources
        if contradiction_flags:
            self._check_contradiction_resolution(contradiction_flags, user_id)
        if active_topics:
            self._check_topic_resolution(active_topics)
        # Update resolution counts after new sources
        active_unresolved = self.engine.active_unresolved()
        debug.unresolved_count = len(active_unresolved)
        debug.unresolved_items = list(active_unresolved)

        # ---- Step 11b: Derived affect + agency (0 LLM calls) ----
        _ts = time.perf_counter()
        affect_state = resolve_affect(
            state=self.engine.state,
            appraisal=appraisal,
            person=person,
        )
        st.affect_state = affect_state
        debug.affect_state = affect_state
        self._commit_turn_decision(st)
        debug.autonomy_level = self._autonomy_level
        timings["affect_agency"] = (time.perf_counter() - _ts) * 1000

        return None

    # ------------------------------------------------------------------
    # Stage 2: gated deliberation
    # ------------------------------------------------------------------

    def _stage_deliberation(self, st: _TurnState) -> None:
        """Execute committed grants. No social self-deliberation or second choice.

        Each granted task is carried out as the reading reported it and the
        decision committed it: a committed operation runs directly, a
        continuation carries the running plan on, learning goes to the
        session's executor, and a task that names a capability without all of
        its operands goes to model tool selection, which is offered the
        capabilities that task names and no other. A task that names no
        capability is work done in the reply: nothing is executed for it, no
        tool is selected and no receipt is made.
        """
        st.dialogue_trace = InnerDialogueTrace(rounds=[], final_candidate="", total_llm_calls=0,
                                              reached_deadlock=False, dominant_path="skip")
        st.debug.dialogue_trace = st.dialogue_trace
        st.character_vector = assemble_character_vector(self._build_context(st))
        st.debug.character_vector = st.character_vector
        st.strategy = summarize_decision(st.turn_decision)
        st.debug.strategy_trace = decision_trace(st.turn_decision)
        st.debug.response_strategy = st.strategy.value
        st.debug.action_variables = (
            derive_action_variables(self.engine.state, trust=st.person.trust if st.person else 0.5)
            if self._tool_executor is not None else None
        )
        combined = ToolTrace()
        summaries = []
        # Where a value the person wrote may stand when a part of this message refers back.
        earlier = tuple(str(item.get("content", "")) for item in self._conversation_history[-CONTEXT_WINDOW:]
                        if item.get("role") == "user")
        asks_status = {component_id for component_id, relation in (st.appraisal.plan_relations if st.appraisal else ())
                       if relation == "status"}
        for item in st.turn_decision.components:
            component = item.component
            if item.granted and component.id in asks_status and self._running_plan() is not None:
                summaries.append(summarize_plan_status(self._active_task_plan))
            if not item.granted or component.kind != "task":
                continue
            with execution_scope(st.turn_decision, component.id, message=st.user_message, earlier=earlier), \
                    reserve_expression_calls(), primary_tool_inference():
                receipt = self._run_session_capability(st, component)
                if receipt is not None:
                    receipt.metadata.update(execution_metadata())
                    combined.executed_results.append(receipt)
                    combined.observations.append(ToolObservation(summary=receipt.output or receipt.error or ""))
                    combined.loop_count += 1
                    summaries.append(f"[{receipt.tool_name}] success={receipt.success}; {receipt.output or receipt.error}")
                    continue
                outcome = (self._run_tool_stage(st, component, combined.executed_results)
                           if self._tool_executor is not None else None)
            if outcome is None:
                continue
            trace = outcome.trace
            combined.proposed_intents.extend(trace.proposed_intents)
            combined.executed_results.extend(trace.executed_results)
            combined.observations.extend(trace.observations)
            combined.loop_count += trace.loop_count
            combined.final_decision = trace.final_decision or combined.final_decision
            combined.task_trace = trace.task_trace or combined.task_trace
            summaries.append(st.tool_context_summary)
        st.tool_context_summary = "\n".join(filter(None, summaries))
        st.debug.tool_trace = combined if combined.executed_results or self._tool_executor is not None else None
        # Independent work is proposed through process_autonomous(), with its
        # own decision and scope, never by relabeling a declined user task.

    def _run_session_capability(self, st: _TurnState, component: RequestComponent) -> ToolResult | None:
        """Hand a granted learning task to the session's executor; None when the task is not one.

        The session carries out learning intake, and only that: a task reaches
        its executor when the reading reported it as asking for the learning
        capability, never because of a reference a component happens to carry.
        It is handed over together with the request it was granted: the
        address the person wrote, or the message's own text.

        A task that names learning intake alone but committed no operation
        gave an address the person did not write, or left it to be produced.
        Nothing is fetched from an address that is not theirs: the receipt
        says that nothing was attempted, and why.
        """
        if self._component_executor is None or component.continuation_of:
            return None
        if component.requested_tool != LEARNING:
            if component.capability_requirements != (LEARNING,):
                return None
            return ToolResult(LEARNING, False, "", metadata={"execution_started": False},
                              error="Not attempted: no address written by the person was committed for learning intake.")
        from runtime.learning_intake import granted_learning, learning_request

        operands = dict(component.requested_arguments)
        request = learning_request(url=operands.get("url", ""), message=st.user_message,
                                   asked_in=component.source_span)
        with granted_learning(request):
            receipt = self._component_executor(component, st.user_id, self._llm_backend_fast)
        if receipt is None:
            receipt = ToolResult(LEARNING, False, "", metadata={"execution_started": False},
                                 error="Not attempted: learning intake is not available in this session.")
        record_operands(receipt, operands)
        return receipt

    @staticmethod
    def _request_text(component: RequestComponent) -> str:
        """What a task asks, as the reading reports it: its description and, when that adds to it, its span."""
        span = component.source_span.strip()
        return component.description if not span or span in component.description else f"{component.description}\n{span}"

    def _stage_arguments(self, st: _TurnState) -> dict[str, Any]:
        return dict(state=self.engine.state, person=st.person, defense_active=False, engine=self.engine,
                    agency_decision=st.agency_decision, autonomy_level=self._autonomy_level)

    def _selection(self, component: RequestComponent, receipts: list[ToolResult]) -> Selection:
        """One granted task as the selecting model is given it: all of it data.

        The request, the capabilities it names (the only ones offered), the
        operands the person wrote, the latest turns of the conversation and
        what this turn has already executed, so that a request which refers
        to an earlier turn or to an earlier step can be resolved.
        """
        turns = tuple(("character" if item.get("role") == "assistant" else "user" if item.get("role") == "user"
                       else "other", str(item.get("content", ""))[:SELECTION_TURN_CHARS])
                      for item in self._conversation_history[-CONTEXT_WINDOW:])
        return Selection(
            request=self._request_text(component), capabilities=component.capability_requirements,
            operands=held_operands(component), recent_turns=turns,
            receipts=tuple(receipt_projection(result, SELECTION_RECEIPT_CHARS) for result in receipts))

    def _select_tools(self, st: _TurnState, component: RequestComponent,
                      receipts: list[ToolResult] | None = None) -> ToolLoopResult:
        """Model tool selection for a granted task that names a capability without all of its operands.

        Asked only here. The model is offered the capabilities the task names
        and no other, and is given the task as data (:meth:`_selection`). The
        grant check holds whatever it answers to those capabilities and to the
        operands the person wrote. When the backend's tool-calling request
        fails as unsupported or at the transport level, that is recorded, no
        further selection is asked for in this turn, and selection is left
        alone for ``TOOL_SELECTION_PAUSE_TURNS`` turns before it is offered
        again.
        """
        record = st.debug.tool_selection
        unavailable = ""
        if self._tool_runner is None:
            record.update(status="unavailable")
            unavailable = "this backend has no tool selection"
        elif st.tool_selection_paused:
            record.update(status=record.get("status") or "paused", skipped=record.get("skipped", 0) + 1,
                          paused_turns_left=self._tool_selection_pause)
            unavailable = "tool selection is paused after the backend failed a tool-calling request"
        if unavailable:
            return ToolLoopResult(
                trace=ToolTrace(), action_variables=derive_action_variables(
                    self.engine.state, trust=st.person.trust if st.person else 0.5),
                tool_context_summary=("No tool ran for this request: not all of its operands were given by the "
                                      f"person, and {unavailable}. Nothing was executed."))
        record.update(requests=record.get("requests", 0) + 1)
        selection = self._selection(component, receipts or [])
        result = self._tool_runner.run_tool_loop(
            user_message=selection.request, selection=selection, life_influence=st.life_influence,
            character_vector=st.character_vector, **self._stage_arguments(st))
        failure = next((receipt.metadata.get("tool_selection_failure") for receipt in result.trace.executed_results
                        if isinstance(receipt.metadata, dict) and receipt.metadata.get("tool_selection_failure")), "")
        if failure and not self._research_mode:   # a study keeps its own call contract: nothing is paused
            st.tool_selection_paused = True
            self._tool_selection_pause = TOOL_SELECTION_PAUSE_TURNS
            record.update(status="failed", failure=failure, paused_turns_left=TOOL_SELECTION_PAUSE_TURNS)
        return result

    def _writes_generated_text(self, component: RequestComponent) -> bool:
        """A task to store text that still has to be written, in a file the person named."""
        written = {name for name, _ in component.requested_arguments}
        return (set(component.capability_requirements) == {WRITE_FILE} and written == {WRITE_PATH}
                and self._tool_executor.registry.get(WRITE_FILE) is not None)

    def _run_tool_stage(self, st: _TurnState, component: RequestComponent,
                        receipts: list[ToolResult] | None = None) -> ToolLoopResult | None:
        """Carry out one granted task + memory coupling (Steps 12b-12d).

        Only called when a tool executor is attached. Returns None when the
        task asks for no execution: it names no capability, so it is work done
        in the reply. ``receipts`` is what this turn has executed so far.
        """
        debug = st.debug
        user_id = st.user_id

        _ts_tool = time.perf_counter()
        if component.continuation_of:
            plan = self._running_plan()
            if plan is None or plan.id != component.continuation_of:
                return None
            tool_loop_result = continue_plan(plan, executor=self._tool_executor, **self._stage_arguments(st))
        elif component.requested_tool:
            intent = ToolIntent(tool_name=component.requested_tool, arguments=dict(component.requested_arguments),
                                reason="Literal user request", expected_outcome="Execute the granted atomic operation")
            tool_loop_result = run_tool_loop(
                requested_intent=intent, executor=self._tool_executor, life_influence=st.life_influence,
                character_vector=st.character_vector, **self._stage_arguments(st))
        elif not component.capability_requirements:
            return None
        elif self._writes_generated_text(component):
            tool_loop_result = self._write_generated_artifact(st, component)
        else:
            tool_loop_result = self._select_tools(st, component, receipts)
        debug.tool_trace = tool_loop_result.trace
        debug.action_variables = tool_loop_result.action_variables
        st.tool_context_summary = tool_loop_result.tool_context_summary
        operator_decision = tool_loop_result.trace.final_decision
        if operator_decision is not None and operator_decision.decision != "execute":
            st.tool_context_summary += f"\nOperator decision: {operator_decision.decision}: {operator_decision.rationale}"
        if tool_loop_result.life_influence_effects:
            debug.life_influence_effects.update(tool_loop_result.life_influence_effects)
        self._record_capability_gaps(tool_loop_result.capability_gaps)

        # Track task trace and active plan (Phase 7)
        task_trace = tool_loop_result.trace.task_trace
        if task_trace and task_trace.plan:
            debug.task_trace = task_trace
            self._active_task_plan = task_trace.plan
            if not task_trace.plan.origin_decision_id:
                task_trace.plan.origin_decision_id = st.turn_decision.turn_id if st.turn_decision else ""
                task_trace.plan.component_id = execution_metadata().get("component_id", "")
            # Clear terminal plans from session state
            if task_trace.plan.is_terminal:
                self._active_task_plan = None

        # ---- Step 12c: TOOL MEMORY COUPLING ----
        trace = tool_loop_result.trace
        if trace.executed_results:
            effects = ToolMemoryEffects()
            failure_count = 0

            for result, observation in zip(trace.executed_results, trace.observations):
                cap = self._tool_executor.registry.get(result.tool_name)
                category = cap.category if cap else ToolCategory.READ_ONLY

                if not result.success:
                    failure_count += 1

                # Short-term memory record
                tool_event = create_tool_event(result, observation, self.engine.state)
                self.short_term.record(tool_event, self.engine.state)
                effects.short_term_recorded = True

                operational_issue = create_operational_issue(result)
                if operational_issue is not None:
                    effects.operational_issues.append(operational_issue)
                    continue

                if result.success:
                    self._resolve_matching_tool_unresolved(result.tool_name)

                # Long-term memory (salient episodes only)
                if is_salient_episode(result, observation, category, failure_count):
                    lt_entry = create_long_term_entry(
                        result, observation, category, user_id=user_id,
                    )
                    self.long_term.store(lt_entry)
                    effects.long_term_written = True
                    effects.long_term_summary = lt_entry.summary

                # Self-observations from tool behavior
                self_obs = derive_tool_self_observations(
                    result, observation, category,
                    tool_loop_result.action_variables,
                    trace.final_decision,
                    failure_count=failure_count,
                )
                for trait, value, context in self_obs:
                    self.self_profile.record_behavior(trait, value, context)
                    effects.self_observations.append(f"{trait}={value:.2f}")

                # Unresolved items from tool failures
                unresolved = create_tool_unresolved_item(
                    result, observation, category, failure_count=failure_count,
                )
                if unresolved:
                    self.engine.add_unresolved(unresolved)
                    effects.unresolved_items_created.append(unresolved.id)

                # Tool success/failure is evidence about our execution, not
                # the user's reliability. It cannot alter relationship trust.

            debug.tool_memory_effects = effects

        # ---- Step 12d: TASK MEMORY COUPLING (Phase 7) ----
        if task_trace and task_trace.plan and task_trace.plan.is_terminal:
            plan = task_trace.plan
            effects = debug.tool_memory_effects or ToolMemoryEffects()

            # Unresolved items from plan outcome
            task_unresolved = create_task_unresolved_item(plan, task_trace)
            if task_unresolved:
                self.engine.add_unresolved(task_unresolved)
                effects.unresolved_items_created.append(task_unresolved.id)

            # Self-observations from multi-step behavior
            task_obs = derive_task_self_observations(
                plan, task_trace, tool_loop_result.action_variables,
            )
            for trait, value, context in task_obs:
                self.self_profile.record_behavior(trait, value, context)
                effects.self_observations.append(f"{trait}={value:.2f}")

            # Long-term memory for salient plans
            task_lt = create_task_long_term_entry(
                plan, task_trace, user_id=user_id,
            )
            if task_lt:
                self.long_term.store(task_lt)
                effects.long_term_written = True
                effects.long_term_summary = task_lt.summary

            debug.tool_memory_effects = effects

        st.timings["tool_loop"] = st.timings.get("tool_loop", 0.0) + (time.perf_counter() - _ts_tool) * 1000
        return tool_loop_result

    def _run_self_intent_stage(self, st: _TurnState) -> None:
        owner = self._research_owner
        closed_loop = (
            st.self_directed and owner is not None
            and owner.config.research_execution_mode == "closed_loop"
        )
        if not closed_loop:
            self._run_self_intent_step(st)
            return
        all_intents = []
        stop_reason = "step_limit"
        try:
            for step_index in range(owner.config.research_max_steps_per_activation):
                goal_id = owner.state.get("goal_id")
                execution = owner.store.get("goal_execution", goal_id) if goal_id else None
                if execution and execution["status"] != "active":
                    stop_reason = "goal_" + execution["status"]
                    break
                if execution:
                    from runtime.goal_execution import unresolved_dependencies

                    if unresolved_dependencies(owner.store, execution):
                        stop_reason = "dependencies_unverified"
                        break
                st.action_context = {
                    **(st.action_context or {}),
                    "goal_execution": execution,
                    "recent_tool_receipts": owner.execution_receipts(goal_id=goal_id),
                    **owner.outcome_context(goal_id),
                    "uncertain_actions": [
                        {key: item.get(key) for key in ("id", "tool", "status", "goal_id")}
                        for item in owner.store.list("action")
                        if item["status"] in {"started", "unknown"}
                    ][-20:],
                    "activation_step": step_index + 1,
                    "remaining_steps": owner.config.research_max_steps_per_activation - step_index,
                }
                self._run_self_intent_step(st, max_actions=1)
                all_intents.extend(st.debug.self_intents)
                if not st.debug.self_intents:
                    stop_reason = "subject_stopped"
                    break
        except Exception as error:
            stop_reason = type(error).__name__
            raise
        finally:
            st.debug.self_intents = all_intents
            goal_id = owner.state.get("goal_id")
            execution = owner.store.get("goal_execution", goal_id) if goal_id else None
            if execution:
                if stop_reason == "step_limit" and execution["status"] != "active":
                    stop_reason = "goal_" + execution["status"]
                execution["last_stop_reason"] = stop_reason
                owner.store.put("goal_execution", goal_id, execution)
            owner.store.observe("execution_yielded", {
                "goal_id": goal_id, "reason": stop_reason, "actions": len(all_intents),
            })

    def _run_self_intent_step(self, st: _TurnState, *, max_actions: int | None = None) -> None:
        """Self-intent proposal + execution (Step 12e).

        Only called when the feature is enabled and a tool executor is
        attached.
        """
        debug = st.debug
        _ts = time.perf_counter()
        tool_catalog = self._build_tool_catalog_for_intent()
        task_only = (
            self._research_owner is not None
            and self._research_owner.config.research_context_mode == "task_only"
        )
        self_intents = propose_self_intents(
            backend=self._llm_backend_fast,
            modulator_snapshot={} if task_only else self.engine.snapshot(),
            affect_state=None if task_only else st.affect_state,
            agency_decision=st.agency_decision,
            user_message=st.user_message,
            candidate_response=st.dialogue_trace.final_candidate,
            tool_catalog=tool_catalog,
            strict=self._research_mode,
            self_directed=(self._research_mode and st.self_directed and self._research_action_framing == "agent_owned"),
            action_context=st.action_context,
            max_actions=max_actions,
            action_contract_version=(
                self._research_owner.manifest.get("action_contract_version", 1)
                if max_actions == 1 and self._research_owner is not None else 1
            ),
            contract_observer=(
                self._research_owner.store.observe if self._research_owner is not None else None
            ),
            use_json_schema=(
                self._research_owner.config.research_json_schema
                if self._research_owner is not None else False
            ),
            agent_context=(format_character_vector_prompt(assemble_character_vector(self._build_context(st)))
                           + "\nPersistent intentions:\n" + json.dumps(self._research_owner.store.list("goal"), ensure_ascii=False)
                           if self._research_owner is not None and not task_only else ""),
        )
        if st.self_directed and self._research_owner is not None:
            self._research_owner.store.observe(
                "action_framing_exposed", {"framing": self._research_action_framing}
            )
        if self_intents:
            if debug.tool_trace is None:
                debug.tool_trace = ToolTrace()
            for intent in self_intents:
                blocked = self._self_intent_blocked_reason(intent.tool_name)
                if max_actions is not None and blocked is None:
                    capability = self._tool_executor.registry.get(intent.tool_name)
                    uncertain = any(
                        item["status"] in {"started", "unknown"}
                        and (
                            self._tool_executor.registry.get(item["tool"]) is None
                            or self._tool_executor.registry.get(item["tool"]).category not in {
                                ToolCategory.READ_ONLY, ToolCategory.COGNITIVE,
                            }
                        )
                        for item in self._research_owner.store.list("action")
                    )
                    if uncertain and (
                        capability is None or capability.category not in {
                            ToolCategory.READ_ONLY, ToolCategory.COGNITIVE,
                        }
                    ):
                        blocked = "An earlier effect is unknown; investigate before another mutation."
                if blocked is not None:
                    result = ToolResult(
                        tool_name=intent.tool_name,
                        success=False,
                        output="",
                        error=blocked,
                    )
                    summary = f"self-action blocked: {intent.tool_name}: {blocked}"
                    if max_actions is not None:
                        self._research_owner.observe_tool(
                            "completed", intent.tool_name, intent.arguments, result,
                        )
                else:
                    try:
                        if self._research_owner is not None:
                            self._research_owner.state["action_origin"] = "self_intent"
                        result = self._tool_executor.execute(
                            intent.tool_name, intent.arguments,
                        )
                    except (ResearchBudgetExceeded, ResearchStudyLimit):
                        raise
                    except Exception as exc:
                        result = ToolResult(
                            tool_name=intent.tool_name,
                            success=False,
                            output="",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    finally:
                        if self._research_owner is not None:
                            self._research_owner.state.pop("action_origin", None)
                    summary = (
                        f"self-action: {intent.tool_name}: {intent.rationale}"
                    )
                intent.result = result
                if result.metadata.get("resource_stop"):
                    raise ResearchStudyLimit(result.error)
                if self._research_mode and not task_only and (self._research_owner is None or self._research_owner.config.research_appraisal_mode == "legacy"):
                    capability = self._tool_executor.registry.get(intent.tool_name)
                    observation = appraise_tool_result(result, capability.category if capability else ToolCategory.EXTERNAL_ACTION)
                    for name, delta in observation.emotional_delta.items():
                        current = getattr(self.engine.state, name)
                        setattr(self.engine.state, name, max(0.0, min(1.0, current + delta)))
                    self.short_term.record(EmotionalEvent(
                        event_type=EventType.USER_MESSAGE, intensity=0.3, source="self_action",
                        metadata={"tool_name": intent.tool_name, "success": result.success},
                    ), self.engine.state)
                debug.tool_trace.executed_results.append(result)
                # Keep executed/observations aligned so downstream
                # consumers iterating in zip see matching shapes.
                debug.tool_trace.observations.append(
                    ToolObservation(summary=summary)
                )
        debug.self_intents = self_intents
        st.timings["self_intent"] = (time.perf_counter() - _ts) * 1000

    def process_autonomous(
        self, purpose: str, goals: list[dict[str, Any]], *, decision: dict[str, Any] | None = None,
    ) -> PipelineResponse:
        user_id = getattr(self, "_research_user_id", "self")
        self._ensure_bound_user(user_id)
        debug = DebugState(user_message="[self-directed activity]", user_id=user_id)
        st = _TurnState(user_message=purpose, user_id=user_id, debug=debug, self_directed=True)
        task_only = (
            self._research_owner is not None
            and self._research_owner.config.research_context_mode == "task_only"
        )
        if decision is not None and self._research_owner is not None:
            st.action_context = {
                "selected_intention": {
                    key: ({"excerpt": value[:4000], "truncated": True}
                          if isinstance(value, str) and len(value) > 4000 else value)
                    for key, value in decision.items()
                    if key in {"action", "reason", "goal_id", "goal", "goal_status",
                               "question_id", "question_status", "message", "session_key"}
                },
                "recent_tool_receipts": self._research_owner.execution_receipts(),
                **self._research_owner.outcome_context(),
            }
        st.person = self.person_profiles.get_or_create(user_id)
        st.self_prof = self.self_profile.get_profile()
        st.appraisal = AppraisalFrame()
        st.affect_state = resolve_affect(text="", state=self.engine.state, appraisal=st.appraisal, person=st.person)
        self._refresh_relationship_state(user_id)
        self._turn_index += 1
        st.turn_id = f"self:{self._turn_index}"
        st.dynamics = self._dynamics
        st.appraisal.request_components = (RequestComponent(
            "activity", purpose, character_value=0.6 if goals else 0.2, user_value=0,
            effort=0.3, uncertainty=0.4, evidence_refs=tuple(str(g.get("id", "")) for g in goals),
        ),)
        st.life_history_context = {} if task_only else self._load_life_history_context(purpose)
        self._capture_developing_self(st)
        self._commit_turn_decision(st, scope="self")
        st.dialogue_trace = InnerDialogueTrace(rounds=[], total_llm_calls=0, reached_deadlock=False, final_candidate=json.dumps({
            "purpose": purpose, "goals": goals, "identity": st.life_history_context,
            "memories": [] if task_only else [asdict(item) for item in self.semantic_memory.retrieve(purpose, source_person=user_id)],
        }, ensure_ascii=False))
        if self._features.self_intent and self._tool_executor is not None and st.turn_decision.permits("activity"):
            with execution_scope(st.turn_decision, "activity", purpose="self"):
                self._run_self_intent_stage(st)
        self._last_decision = st.turn_decision
        self._dynamics = commit_relationship(st.turn_decision)
        self.relationship_memory.save_dynamics(user_id, self._dynamics, self._turn_index)
        debug.delivery_outcome = "self_directed_activity"
        debug.modulator_snapshot = self.engine.snapshot()
        if self._research_owner is not None:
            self._research_owner.store.observe("self_directed_outcome", asdict(debug))
            if (
                debug.tool_trace and debug.tool_trace.executed_results
                and self._research_owner.config.research_context_mode == "full"
            ):
                learning_id = uuid.uuid4().hex
                self._research_owner.store.put("learning", learning_id, {
                    "id": learning_id, "status": "pending", "attempts": 0,
                    "title": "Self-directed activity", "source_type": "self_action",
                    "source_ref": learning_id, "text": json.dumps({
                        "purpose": purpose, "results": [asdict(item) for item in debug.tool_trace.executed_results],
                    }, ensure_ascii=False, sort_keys=True), "metadata": {
                        "evidence_kind": "self_inference" if self._research_owner.outcome_learning is not None else "tool_receipts",
                        "observation_source": "tool_receipts",
                        "interpretation_source": "model_digest_not_independent_observation",
                        "source_receipt_ids": [item["id"] for item in self._research_owner.execution_receipts()],
                    },
                })
        return PipelineResponse(response="", debug=debug)

    def _self_intent_blocked_reason(self, tool_name: str) -> str | None:
        """Autonomy gate for self-intent executions; None means allowed.

        ``self.*`` actions operate on OpenRook's own state and are the point of
        this stage, so internal bookkeeping (READ_ONLY/COGNITIVE/WRITE to the
        private self-log) stays ungated. But an explicit ``autonomy == "off"``
        means "no outward action" — so EXTERNAL_ACTION self tools that reach
        the network or the owner (``self.verify`` runs a web search,
        ``self.alert_owner`` sends Telegram) are blocked when off, never
        laundered through the self prefix. Any non-``self.*`` tool proposed
        here is held to the same rule as user-driven intents:
        READ_ONLY/COGNITIVE execute, everything else needs at least
        ``autonomous`` autonomy. Without this, self-intent was a bypass
        around the tool-loop arbiter (a live UAT run caught fs.write_file
        executing under assisted autonomy through this path).
        """
        autonomy = (self._autonomy_level or "assisted").strip().lower()
        capability = (
            self._tool_executor.registry.get(tool_name)
            if self._tool_executor is not None
            else None
        )

        if tool_name.startswith("self."):
            # Self-internal tools are the stage's purpose, but an explicit
            # "off" still forbids outward (EXTERNAL_ACTION) self tools.
            if autonomy != "off":
                return None
            category = capability.category if capability else ToolCategory.EXTERNAL_ACTION
            if category == ToolCategory.EXTERNAL_ACTION:
                return (
                    f"Self-intent execution refused: outward {category.value} "
                    f"tool {tool_name!r} is not allowed while tool autonomy is off."
                )
            return None

        if autonomy == "off":
            return "Self-intent execution refused: tool autonomy is off."
        if autonomy in ("autonomous", "high_risk"):
            return None
        category = capability.category if capability else ToolCategory.EXTERNAL_ACTION
        if category in (ToolCategory.READ_ONLY, ToolCategory.COGNITIVE):
            return None
        return (
            f"Self-intent execution blocked: {category.value} tool "
            f"{tool_name!r} requires confirmation under {autonomy} autonomy."
        )

    # ------------------------------------------------------------------
    # Stage 3: master LLM generation
    # ------------------------------------------------------------------

    def _stage_generate(self, st: _TurnState) -> None:
        """Express the frozen decision. Receipts cannot retroactively change grants."""
        start = time.perf_counter()
        st.ctx = self._build_context(st, tool_context_summary=st.tool_context_summary)
        from core.grounding import action_decision_evidence
        prior = []
        for event in reversed(self.relationship_memory.recent_events(
                st.user_id, event_kind="turn_decision", limit=8)):
            try:
                prior.append(json.loads(event.summary)["decision"])
            except (ValueError, KeyError, TypeError):
                continue
        st.ctx.prior_action_decisions = action_decision_evidence(prior)
        st.debug.retrieved_memories, st.debug.semantic_memories, st.debug.relationship_context = prompt_memory_context(st.ctx)
        st.character_vector = assemble_character_vector(st.ctx)
        st.debug.character_vector = st.character_vector
        if st.turn_decision.chosen_silence:
            st.gen_result = GenerationResult(response="", deliberate_silence=True)
            st.debug.delivery_outcome = "chosen_silence"
        else:
            try:
                st.gen_result = self.generator.generate(st.ctx, st.user_message, self._conversation_history)
            except (ResearchBudgetExceeded, ResearchStudyLimit):
                raise
            except Exception as error:
                st.backend_error = error
                st.gen_result = GenerationResult(response="", failure_kind="backend_failure")
                st.debug.delivery_outcome = "backend_failure"
                st.debug.self_check_issues.append(f"generation: {type(error).__name__}")
            st.debug.generation_attempts = 1
        st.debug.raw_response = st.gen_result.raw_response
        st.debug.deliberate_silence = st.gen_result.deliberate_silence
        st.timings["generator"] = (time.perf_counter() - start) * 1000

    def _stage_self_check(self, st: _TurnState) -> None:
        start = time.perf_counter()
        st.gen_result = self._check_dignity(st, st.gen_result)
        st.timings["self_check"] = (time.perf_counter() - start) * 1000

    @staticmethod
    def _requested_copies(st: _TurnState) -> tuple[str, ...]:
        """Texts the person asked to have reproduced word for word, for components this decision undertook."""
        if st.appraisal is None or st.turn_decision is None:
            return ()
        return tuple(span for component_id, span in st.appraisal.reproductions
                     if st.turn_decision.permits(component_id))

    def _check_dignity(self, st: _TurnState, gen_result):
        """Review and spend bounded correction effort in proportion to severity.

        Every draft goes through the same gate: the structural checks, and the
        model's review against the frozen decision, which is required work
        with its own allowance. Failed expression is not chosen silence.
        Retain the best nonempty reviewed draft, with honest degradation
        telemetry if correction fails.
        """
        debug = st.debug
        if gen_result.deliberate_silence and st.turn_decision and st.turn_decision.chosen_silence:
            debug.delivery_outcome = "chosen_silence"
            return gen_result
        if not gen_result.response and st.backend_error is not None:
            debug.delivery_outcome = "backend_failure"
            return gen_result
        review_context = dict(user_message=st.user_message, decision=st.turn_decision,
            backend=self._llm_backend_fast, reviewer=self._expression_reviewer, research=self._research_mode,
            recent_context=self._conversation_history,
            prior_assistant_replies=tuple(self._assistant_replies), tool_trace=debug.tool_trace,
            prior_tool_receipts=self._receipt_history,
            recalled_memories=expression_memory_evidence(st.ctx) if st.ctx else (),
            requested_copies=self._requested_copies(st))

        def research_review_due():
            """Research keeps its bounded call contract: a review is one counted call, and it is
            made where the decision constrains the reply, a tool ran or was refused, or an
            earlier draft of the turn was reviewed. Nothing about a draft's wording decides it."""
            decision = st.turn_decision
            constrained = bool(decision and (decision.declined or decision.selected_intention is not None
                                             or decision.relationship.h or decision.relationship.r
                                             or decision.perception.affiliation))
            return remaining_calls() > 0 and (bool(debug.expression_reviews) or constrained or bool(
                debug.tool_trace and (debug.tool_trace.executed_results or debug.tool_trace.final_decision)))

        gate = DeliveryGate(review_context, debug, research_review_due if self._research_mode else None)
        gate.add(gen_result)
        while True:
            best = gate.best()
            severe = best.review.severity >= 4
            all_stale = all(any(i.code == STALE for i in c.review.issues) for c in gate.candidates)
            required = not gate.eligible
            # Research counts a rewrite and its review as two calls. A rewrite that
            # could not be reviewed is not started: it would go out unexamined in
            # place of the draft that was. Fresh expression that is required is.
            affordable = remaining_calls() >= (2 if self._research_mode and not required else 1)
            if (best.review.status != "violation"
                    or debug.generation_attempts >= (5 if required else 3 if severe or all_stale else 2)
                    or not affordable):
                break
            latest = gate.candidates[-1]
            correction = correction_instruction(
                list(dict.fromkeys((*best.review.issues, *latest.review.issues))),
                previous_reply=latest.result.response, reviewer_reason=latest.review.reason,
                forbidden_drafts=[c.result.response for c in gate.candidates
                    if c.review.unusable or any(i.code == STALE for i in c.review.issues)])
            debug.correction_note = correction
            try:
                debug.generation_attempts += 1
                with expression_retry(st.turn_decision.seed, debug.generation_attempts - 1, required=required), inference_phase("regeneration"):
                    revised = self.generator.generate(replace(st.ctx, response_strategy=correction),
                                                      st.user_message, self._conversation_history)
                gate.add(revised, path="rewrite")
            except (ResearchBudgetExceeded, ResearchStudyLimit):
                raise
            except Exception as error:
                debug.self_check_issues.append(f"rewrite: {type(error).__name__}")
                if required:
                    st.backend_error = error
                break
        # Editing proposes candidates through the very same gate. Rank every
        # reviewed remainder alongside the drafts; none can return directly.
        eligible = gate.eligible
        if eligible and all(excisable_choice_issues(c.review) for c in eligible):
            for candidate in sorted(eligible, key=gate.rank):
                gate.repair(candidate, choice=True)
        if not gate.eligible:
            failed = gate.best()
            correction = correction_instruction(list(failed.review.issues),
                previous_reply=gate.candidates[-1].result.response, reviewer_reason=failed.review.reason,
                forbidden_drafts=[c.result.response for c in gate.candidates])
            debug.correction_note = correction
            try:
                debug.generation_attempts += 1
                with final_expression_attempt(), expression_retry(st.turn_decision.seed,
                        debug.generation_attempts - 1, required=True), inference_phase("regeneration"):
                    final = self.generator.generate(replace(st.ctx, response_strategy=correction),
                                                    st.user_message, self._conversation_history)
                gate.add(final, path="delivery_recovery")
            except (ResearchBudgetExceeded, ResearchStudyLimit):
                raise
            except Exception as error:
                st.backend_error = error
                debug.self_check_issues.append(f"delivery recovery: {type(error).__name__}")
        if not gate.eligible and st.backend_error is None:
            # Nothing can be delivered. If that is only because the reviewer reported a
            # fixed limit, one independent review of the best such candidate decides.
            gate.confirm()
        selected = gate.select()
        verdict = selected.review if selected else gate.best().review
        debug.dignity_issues = list(dict.fromkeys(i.code for c in gate.candidates for i in c.review.issues))
        debug.self_check_issues.extend(i.message for c in gate.candidates for i in c.review.issues)
        debug.self_check_passed = verdict.status == "pass"
        debug.residual_semantic_risk = verdict.status != "pass" or verdict.semantic_status != "pass" or bool(st.turn_decision and st.turn_decision.declined)
        debug.dignity_outcome = verdict.status
        if any(issue.code == "unverified_external_action_claim" for issue in verdict.issues):
            debug.grounding_outcome = "violation"
        if selected is None:
            failure = "backend_failure" if st.backend_error is not None else "expression_failure"
            debug.delivery_outcome = failure
            return replace(gate.best().result, response="", deliberate_silence=False, failure_kind=failure)
        if verdict.status == "violation":
            debug.delivery_outcome = "delivered_degraded"
        else:
            debug.delivery_outcome = "delivered" if verdict.status == "pass" else "delivered_uncertain"
        return selected.result

    # ------------------------------------------------------------------
    # Stage 5: post-processing
    # ------------------------------------------------------------------

    def _stage_post_process(self, st: _TurnState) -> PipelineResponse:
        """Post-processing (Steps 14b-15).

        Owned-pursuit continuity, memory writes, energy drain, trust update,
        self-observations, and the final debug trace.
        """
        debug = st.debug
        timings = st.timings
        user_message = st.user_message
        user_id = st.user_id
        event = st.event
        gen_result = st.gen_result

        debug.response = gen_result.response
        self.last_intake_receipt = ""

        # ---- Step 15: Post-processing ----
        outcome_event = EmotionalEvent(
            event_type=EventType.USER_MESSAGE,
            intensity=0.1,
            source="self",
            metadata={"type": "response_delivered"},
        )
        self.short_term.record(outcome_event, self.engine.state)

        # Incoming events already carry their own energy effect. Selected
        # silence performs no expression work; recovery still uses elapsed
        # time, never an invented instant recharge for choosing rest.
        if not gen_result.deliberate_silence:
            self.engine.drain_energy(intensity=event.intensity)
        debug.energy_after = self.engine.state.energy
        debug.emotion_label = self.engine.to_emotion_label()
        debug.modulator_snapshot = self.engine.snapshot()
        final_unresolved = self.engine.active_unresolved()
        debug.unresolved_count = len(final_unresolved)
        debug.unresolved_items = list(final_unresolved)

        # Update conversation history
        if debug.tool_trace:
            self._receipt_history = (self._receipt_history + debug.tool_trace.executed_results)[-32:]
        self._append_conversation_history(
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": gen_result.response},
        )

        # Record interaction for person profile
        self.person_profiles.record_interaction(
            user_id,
            {"engagement": event.intensity},
            context=event.event_type.value,
        )

        # Interpersonal trust and supported reliability contribute independently.
        person = self.person_profiles.get_or_create(user_id)
        person.trust = max(0.0, min(1.0, person.trust + st.trust_delta))
        self.person_profiles.save(person)
        debug.person_profile = person
        self._record_live_development(st)
        self._record_social_turn(st)

        # ---- Self-observations (1-3 per turn) ----
        self._record_self_observations(
            event, st.detected, st.appraisal, st.defense, self.engine.state,
            gen_result.response,
            declined=bool(st.turn_decision is not None and st.turn_decision.declined
                          and debug.delivery_outcome.startswith("delivered")),
        )

        # ---- Semantic memory writes ----
        salience_delta = self._record_semantic_memory(
            user_message=user_message,
            assistant_response=gen_result.response,
            user_id=user_id,
            topic=st.relationship_topic,
            event=event,
            life_influence=st.life_influence,
            reply_length=st.appraisal.reply_length if st.appraisal else "",
            lasting=st.appraisal.lasting if st.appraisal else (),
            learned=any(receipt.tool_name == LEARNING and receipt.success
                        for receipt in (debug.tool_trace.executed_results if debug.tool_trace else ())),
        )
        if salience_delta:
            debug.life_influence_effects["memory_salience_delta"] = salience_delta

        # ---- Persist defense event ----
        if st.defense:
            from core.types import DefenseEvent
            de = DefenseEvent(
                timestamp=time.time(),
                defense_type=st.defense.defense_type,
                raw_intensity=st.defense.raw_intensity,
                expressed_intensity=st.defense.expressed_intensity,
                suppression_delta=st.defense.suppression_delta,
            )
            self.self_profile.persist_defense_event(de)

        timings["total"] = (time.perf_counter() - st.t_total) * 1000
        debug.stage_timings_ms = timings

        return PipelineResponse(response=gen_result.response, debug=debug, backend_error=st.backend_error)

    def _record_social_turn(self, st: _TurnState) -> None:
        if st.turn_decision is None:
            return
        self._dynamics = commit_relationship(st.turn_decision)
        self.relationship_memory.save_dynamics(st.user_id, self._dynamics, self._turn_index)
        self._last_decision = st.turn_decision
        intention = st.turn_decision.selected_intention
        if intention and st.debug.delivery_outcome.startswith("delivered"):
            if intention.kind == "pursue":
                for ref in intention.evidence_refs:
                    if ref.startswith(("self:interests:", "question:")):
                        self._owned_pursuits[ref] = {"id": f"conversation:{ref}", "status": "active",
                            "description": intention.content, "reason": intention.reason,
                            "priority": intention.character_value, "source_ref": ref,
                            "decision_turn_id": st.turn_decision.turn_id}
            elif intention.kind == "close":
                for ref in list(self._owned_pursuits):
                    if {ref, f"goal:conversation:{ref}"} & set(intention.evidence_refs):
                        self._owned_pursuits.pop(ref)
        # Raw input, interpretation, choice and delivery are separate evidence.
        record = {"input": st.user_message, "perception": asdict(st.appraisal.perception),
                  "reading": dict(st.debug.reading),
                  "decision": st.turn_decision.to_dict(), "delivered": st.gen_result.response,
                  "delivery_outcome": st.debug.delivery_outcome,
                  "generation_attempts": st.debug.generation_attempts,
                  "expression_reviews": st.debug.expression_reviews,
                  "expression_excision": st.debug.expression_excision,
                  "expression_delivery": st.debug.expression_delivery,
                  "llm_calls": call_telemetry(),
                  "tool_selection": dict(st.debug.tool_selection),
                  "self_development": st.debug.self_development,
                  "stage_timings_ms": {**st.timings, "elapsed": (time.perf_counter() - st.t_total) * 1000},
                  "state_after": self.engine.snapshot()}
        self.relationship_memory.record_event(RelationshipEvent(
            event_kind="turn_decision", source_person=st.user_id,
            summary=json.dumps(record, ensure_ascii=False), intensity=st.dynamics.h,
            confidence=1.0, related_key=f"decision:{st.turn_id}",
        ))
        st.debug.relationship_context_after_turn = self.relationship_memory.build_context(st.user_id)

    def _commit_turn_decision(self, st: _TurnState, *, scope: str = "user") -> None:
        if st.turn_decision is not None:
            return
        if not st.developing_self:
            self._capture_developing_self(st)
        personality = Personality.from_self(st.developing_self, st.user_id)
        if scope == "self":
            now = time.time()
            st.dynamics, _ = update_relationship(
                st.dynamics, personality, st.appraisal.perception,
                DecisionState(**self.engine.snapshot(), trust=st.person.trust if st.person else 0.5),
                elapsed=max(0.0, now - st.dynamics.updated_at) if st.dynamics.updated_at else 0,
                now=now, turn_id=st.turn_id,
            )
        references = tuple(str(loop.id) for loop in (st.relationship_context.active_loops if st.relationship_context else []))
        state = DecisionState(**self.engine.snapshot(), trust=max(0.0, min(1.0, st.person.trust + st.trust_delta)) if st.person else 0.5,
                              history_refs=references)
        components = st.appraisal.request_components
        proposals = ()
        if scope == "user":
            if self._features.self_intent:
                proposals = conversation_proposals(st.developing_self, st.life_history_context, components,
                    state.energy, st.appraisal.self_proposals)
            state, components = ground_conversation_allocation(personality, state, st.life_history_context, components, proposals)
        else:
            # Independent execution retains its already grounded component,
            # while sharing the same active-commitment/resource projection.
            state, _ = ground_conversation_allocation(personality, state, st.life_history_context, (), ())
        capabilities = {offer.name for offer in self._capability_offers()}
        components = tuple(replace(c, feasible=c.feasible and set(c.capability_requirements).issubset(capabilities))
                           for c in components)
        decision = decide(personality, state, st.dynamics, st.appraisal.perception,
                          components, character_seed=self.soul.character_seed,
                          relationship_id=st.user_id, turn_id=st.turn_id,
                          soul_version=self.soul.schema_version, scope=scope, proposals=proposals)
        st.turn_decision = decision
        st.agency_decision = agency_summary(decision)
        st.debug.turn_decision = decision
        st.debug.agency_decision = st.agency_decision
        st.debug.decision_snapshots.append({"stage": "committed_before_execution", **decision.to_dict()})
        self.relationship_memory.save_dynamics(st.user_id, commit_relationship(decision), self._turn_index)
        self.relationship_memory.record_event(RelationshipEvent(
            event_kind="decision_committed", source_person=st.user_id,
            summary=json.dumps({"input": st.user_message, "decision": decision.to_dict()}, ensure_ascii=False),
            related_key=f"decision:{st.turn_id}", confidence=1.0,
        ))

    def _refresh_relationship_state(self, user_id: str) -> None:
        """Other serialized chats may have advanced this relationship's ledger."""
        saved, index = self.relationship_memory.load_dynamics(user_id)
        if self._dynamics is None or index > self._turn_index:
            self._dynamics, self._turn_index = saved, index

    def _build_context(
        self,
        st: _TurnState,
        *,
        candidate_response: str = "",
        defense_instruction: str = "",
        response_strategy: str = "",
        tool_context_summary: str = "",
        self_intents: list[SelfIntent] | None = None,
    ) -> PipelineContext:
        """Single construction point for the per-turn PipelineContext.

        ``modulator_snapshot`` and ``short_term_history`` are sampled at
        call time, so the deliberation-stage and generation-stage contexts
        reflect the engine/memory state at their respective points in the
        turn. New PipelineContext fields only need wiring here — the
        correction retry derives its context via ``dataclasses.replace``.
        """
        return PipelineContext(
            modulator_snapshot=self.engine.snapshot(),
            soul_profile=self.soul,
            developing_self=st.developing_self,
            person_profile=st.person,
            self_profile=st.self_prof,
            appraisal_frame=st.appraisal,
            relationship_context=st.relationship_context,
            topic_profiles=st.active_topics,
            values=self.values,
            retrieved_memories=st.retrieved,
            semantic_memories=st.semantic_memories,
            life_history_context=st.life_history_context,
            skill_context=st.skill_context,
            short_term_history=self.short_term.recent(5),
            contradiction_flags=st.contradiction_flags,
            contagion=st.detected,
            candidate_response=candidate_response,
            defense_instruction="" if self._research_mode else defense_instruction,
            defense_state=asdict(st.defense) if self._research_mode and st.defense else {},
            response_strategy="" if self._research_mode else response_strategy,
            affect_state=st.affect_state,
            agency_decision=st.agency_decision,
            turn_decision=st.turn_decision,
            autonomy_level=self._autonomy_level,
            tool_context_summary=tool_context_summary,
            operator_execution={
                "settings": {"autonomy_level": self._autonomy_level,
                    "tools_enabled": getattr(self._runtime_config, "tools_enabled", True),
                    "shell_tool_enabled": getattr(self._runtime_config, "shell_tool_enabled", True)},
                "decision": asdict(st.debug.tool_trace.final_decision)
                    if st.debug.tool_trace and st.debug.tool_trace.final_decision else None,
            },
            self_intents=list(self_intents) if self_intents else [],
            last_intake_receipt=self.last_intake_receipt,
            research_mode=self._research_mode,
            generator_prompt=self._config.generator_prompt if self._research_mode else None,
        )

    def _append_conversation_history(self, *messages: dict[str, str]) -> None:
        """Append turn messages, keeping the hot transcript bounded.

        Every consumer reads a recent window (see _CONVERSATION_HISTORY_MAX),
        so trimming the oldest entries is invisible to them. Session
        export/persistence carries the same bounded hot transcript.
        """
        self._conversation_history.extend(messages)
        self._assistant_replies.update((m["content"], None) for m in messages
                                       if m.get("role") == "assistant" and m.get("content"))
        if len(self._conversation_history) > _CONVERSATION_HISTORY_MAX:
            del self._conversation_history[:-_CONVERSATION_HISTORY_MAX]

    # ------------------------------------------------------------------
    # Proactive behavior (Phase 8)
    # ------------------------------------------------------------------

    def process_proactive(
        self,
        user_id: str = "default",
        *,
        density_reference: int = 3,
        idle_threshold: float = 300.0,
        recovery_seconds: float = 300.0,
    ) -> PipelineResponse | None:
        """Evaluate proactive triggers and generate response if warranted.

        Called by the runtime when a session has been idle. Response is
        generated through OpenRook (defense + generator), not a hardcoded path.

        Returns PipelineResponse if proactive action was taken, None otherwise.
        The debug state always contains the ProactiveTrace for observability.
        """
        self._ensure_bound_user(user_id)
        debug = DebugState(user_message="[proactive]", user_id=user_id)

        person = self.person_profiles.get_or_create(user_id)
        self_prof = self.self_profile.get_profile()
        life_history_context = self._load_life_history_context()
        snapshot = _TurnState(user_message="", user_id=user_id, debug=debug, life_history_context=life_history_context)
        self._capture_developing_self(snapshot)
        life_influence = derive_life_influence(life_history_context)
        debug.life_history_context = life_history_context
        debug.life_influence = life_influence
        proactive_ctx = PipelineContext(
            modulator_snapshot=self.engine.snapshot(),
            soul_profile=self.soul,
            developing_self=snapshot.developing_self,
            person_profile=person,
            self_profile=self_prof,
            life_history_context=life_history_context,
            skill_context=self._load_skill_context(),
            short_term_history=self.short_term.recent(5),
            autonomy_level=self._autonomy_level,
        )
        character_vector = assemble_character_vector(proactive_ctx)
        debug.character_vector = character_vector

        # Elapsed decay — same as process() Step 0
        now = time.time()
        last_input = self._last_external_input_at if self._last_external_input_at is not None else self._last_turn_time
        idle_seconds = max(0.0, now - last_input) if last_input is not None else 0.0
        previous_update = self._last_state_update_at if self._last_state_update_at is not None else self._last_turn_time
        if previous_update is not None:
            elapsed = now - previous_update
            if elapsed > 0:
                self.engine.decay(elapsed)
        self._last_state_update_at = now

        # 1. Evaluate proactive triggers
        action, trace = evaluate_proactive(
            state=self.engine.state,
            unresolved_items=self.engine.active_unresolved(),
            active_plan=self._active_task_plan,
            person=person,
            idle_seconds=idle_seconds,
            proactive_count=self._proactive_count,
            last_proactive_at=self._last_proactive_at,
            density_reference=density_reference,
            idle_threshold=idle_threshold,
            recovery_seconds=recovery_seconds,
            life_influence=life_influence,
            character_vector=character_vector,
            skill_want_triggers=collect_skill_want_triggers(
                list(self._capability_gaps.values()),
                character_vector=character_vector,
            ),
        )
        debug.proactive_trace = trace
        if trace.life_influence_score_deltas:
            debug.life_influence_effects["proactive_score_deltas"] = dict(trace.life_influence_score_deltas)
            debug.life_influence_effects["proactive_score_delta"] = round(
                sum(trace.life_influence_score_deltas.values()),
                6,
            )

        if action is None:
            return None
        if action.action_type == "continue_task" and (
            self._active_task_plan is None or not self._active_task_plan.origin_decision_id
        ):
            return None  # Legacy/uncommitted work needs an explicit new user grant.

        self._refresh_relationship_state(user_id)
        self._turn_index += 1
        subject = (self._active_task_plan.goal if action.action_type == "continue_task" and self._active_task_plan
                   else action.message or action.action_type)
        st = _TurnState(user_message=subject, user_id=user_id, debug=debug,
                        person=person, dynamics=self._dynamics, turn_id=f"proactive:{self._turn_index}",
                        life_history_context=life_history_context, appraisal=AppraisalFrame())
        st.developing_self = snapshot.developing_self
        st.appraisal.request_components = (RequestComponent(
            "activity", subject,
            character_value=0.4, user_value=0.3, effort=0.3, uncertainty=0.4,
        ),)
        self._commit_turn_decision(st, scope="self")
        if not st.turn_decision.permits("activity"):
            return None

        # 2. For task continuation, run tool loop
        tool_context = ""
        if (
            action.action_type == "continue_task"
            and self._tool_executor is not None
            and self._active_task_plan is not None
        ):
            with execution_scope(st.turn_decision, "activity", purpose="self"):
                # The character's own decision to go on with its plan carries it on directly.
                tool_loop_result = continue_plan(
                    self._active_task_plan, state=self.engine.state, person=person, defense_active=False,
                    executor=self._tool_executor, engine=self.engine, agency_decision=st.agency_decision,
                    autonomy_level=self._autonomy_level)
            tool_context = tool_loop_result.tool_context_summary
            debug.tool_trace = tool_loop_result.trace
            debug.action_variables = tool_loop_result.action_variables
            if tool_loop_result.life_influence_effects:
                debug.life_influence_effects.update(tool_loop_result.life_influence_effects)
            self._record_capability_gaps(tool_loop_result.capability_gaps)

            # Update plan state
            task_trace = tool_loop_result.trace.task_trace
            if task_trace and task_trace.plan:
                debug.task_trace = task_trace
                self._active_task_plan = task_trace.plan
                if task_trace.plan.is_terminal:
                    self._active_task_plan = None

        # 4. Generate through OpenRook
        proactive_semantic = self.semantic_memory.retrieve(
            action.message,
            source_person=user_id,
            limit=self._semantic_cfg.retrieval_limit,
        )
        debug.semantic_memories = proactive_semantic
        skill_context = self._load_skill_context()
        debug.skill_context = skill_context
        ctx = PipelineContext(
            modulator_snapshot=self.engine.snapshot(),
            soul_profile=self.soul,
            developing_self=st.developing_self,
            person_profile=person,
            self_profile=self_prof,
            relationship_context=None,
            semantic_memories=proactive_semantic,
            life_history_context=life_history_context,
            skill_context=skill_context,
            turn_decision=st.turn_decision,
            agency_decision=st.agency_decision,
            tool_context_summary=tool_context,
            last_intake_receipt=self.last_intake_receipt,
        )
        proactive_relationship = self.relationship_memory.build_context(user_id)
        if not proactive_relationship.is_empty():
            ctx.relationship_context = proactive_relationship
        debug.retrieved_memories, debug.semantic_memories, debug.relationship_context = (
            prompt_memory_context(ctx)
        )
        gen_result = self.generator.generate(
            ctx, st.user_message, self._conversation_history,
        )
        debug.generation_attempts = 1
        st.ctx = ctx
        gen_result = self._check_dignity(st, gen_result)
        self._dynamics = commit_relationship(st.turn_decision)
        self.relationship_memory.save_dynamics(user_id, self._dynamics, self._turn_index)
        debug.response = gen_result.response
        debug.modulator_snapshot = self.engine.snapshot()

        # 5. Update proactive tracking
        self._proactive_count += 1
        self._last_proactive_at = time.time()
        self._append_conversation_history({
            "role": "assistant", "content": gen_result.response,
        })

        # 6. Self-observation: proactive behavior recorded
        self.self_profile.record_behavior(
            "proactive", 0.6, f"proactive_{action.action_type}",
        )

        return PipelineResponse(response=gen_result.response, debug=debug)

    def _effective_self_expected_traits(self) -> dict[str, float]:
        """Generated self-descriptions do not override consolidated personality."""
        return dict(self._effective_self["traits"])

    def _capture_developing_self(self, st: _TurnState) -> None:
        state = st.life_history_context.get("developing_self") if self._features.life_history_context else None
        if state is None and self._features.life_history_context and getattr(self._runtime_config, "data_dir", None):
            from runtime.life_history import LifeHistoryStore
            with LifeHistoryStore(self._runtime_config) as store:
                state = store.effective_self()
        if isinstance(state, dict) and state.get("schema_version") == 1:
            self._effective_self = state
        elif not self._features.life_history_context:
            self._effective_self = self._birth_self
        st.developing_self = project_self(self._effective_self, st.user_id)
        self.soul = soul_from_self(st.developing_self)
        self.values = ValueHierarchy(values=dict(st.developing_self["values"]))
        if st.life_history_context.get("developing_self"):
            st.life_history_context["developing_self"] = st.developing_self

    def _project_conversation_commitments(self, st: _TurnState) -> None:
        """Only previously selected, delivered pursuits become commitments.

        Revalidate their support in the current self; a plan/commitment is not
        a new taste, and its existence is not proof that external work happened.
        """
        goals = list(st.life_history_context.get("goals", []))
        if self._features.self_intent:
            sources = conversation_sources(st.developing_self, st.life_history_context, self.engine.state.energy)
            for ref, pursuit in self._owned_pursuits.items():
                source = sources.get(ref)
                if source and not source.get("negative"):
                    goals.append({**pursuit, "description": source["content"], "priority": source["strength"]})
        if self._active_task_plan and not self._active_task_plan.is_terminal:
            goals.append({"id": f"plan:{self._active_task_plan.id}", "description": self._active_task_plan.goal,
                          "status": "active", "reason": "Previously selected unfinished work."})
        if goals:
            st.life_history_context["goals"] = goals

    def _load_life_history_context(self, query_text: str = "") -> dict[str, Any]:
        """Read compact identity-level life context for generation."""
        if not self._features.life_history_context:
            return {}
        try:
            if self._life_history_provider is None:
                context = {}
                if getattr(self._runtime_config, "data_dir", None):
                    from runtime.life_history import LifeHistoryStore
                    with LifeHistoryStore(self._runtime_config) as store:
                        context = store.prompt_context()
            elif query_text:
                try:
                    context = self._life_history_provider(query_text)
                except TypeError:
                    context = self._life_history_provider()
            else:
                context = self._life_history_provider()
        except Exception as exc:
            log.warning("Life history context unavailable: %s", exc)
            return {"error": exc.__class__.__name__}
        context = dict(context) if isinstance(context, dict) else {}
        if self._research_owner is not None:
            context.update(self._research_owner.conversation_context())
        return context

    def _load_life_history_snapshot(self) -> dict[str, Any]:
        if self._life_history_snapshot_provider is None:
            return {}
        try:
            snapshot = self._life_history_snapshot_provider()
        except Exception as exc:
            log.warning("Life history snapshot unavailable: %s", exc)
            return {"error": exc.__class__.__name__}
        return snapshot if isinstance(snapshot, dict) else {}

    def _load_skill_context(
        self,
        *,
        user_message: str = "",
        tool_name: str = "",
        response_strategy: str = "",
    ) -> dict[str, Any]:
        """Read enabled imported skill guidance for generation.

        ``user_message``/``tool_name``/``response_strategy`` form a context
        hint passed to the provider so triggered-only skills can filter
        themselves out when not relevant. Skills with no ``applies_when``
        keep their always-on behavior.
        """
        if self._skill_provider is None:
            return {}
        context_hint: dict[str, Any] = {}
        if user_message:
            context_hint["user_message"] = user_message
        if tool_name:
            context_hint["tool_name"] = tool_name
        if response_strategy:
            context_hint["response_strategy"] = response_strategy
        try:
            try:
                context = self._skill_provider(context_hint=context_hint or None)
            except TypeError:
                # Provider doesn't accept context_hint — fall back to no-filter
                # behavior so older callers/tests keep working.
                context = self._skill_provider()
        except Exception as exc:
            log.warning("Skill context unavailable: %s", exc)
            return {"error": exc.__class__.__name__}
        return context if isinstance(context, dict) else {}

    def _record_capability_gaps(self, gaps: list[dict[str, Any]]) -> None:
        """Track repeated capability gaps as session-local skill-want pressure."""
        now = time.time()
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            gap_id = str(gap.get("id") or gap.get("gap_type") or "").strip()
            if not gap_id:
                continue
            existing = self._capability_gaps.get(gap_id)
            if existing is None:
                existing = dict(gap)
                existing["recent_recurrence"] = max(1, int(existing.get("recent_recurrence") or 1))
                existing["first_seen"] = now
            else:
                existing = dict(existing)
                existing["recent_recurrence"] = int(existing.get("recent_recurrence") or 1) + 1
                existing["frustration_intensity"] = max(
                    float(existing.get("frustration_intensity") or 0.0),
                    float(gap.get("frustration_intensity") or 0.0),
                )
            existing["last_seen"] = now
            self._capability_gaps[gap_id] = existing

    def _write_generated_artifact(self, st: _TurnState, component: RequestComponent) -> ToolLoopResult:
        """Write text the granted task asked to have produced, to the path it committed.

        The reading reported the task: the file-writing capability, the path
        the person wrote, and content that has to be produced. The model
        drafts the content; the file is written through fs.write_file under
        the same arbiter as any write, at exactly the committed path.
        """
        path = dict(component.requested_arguments)[WRITE_PATH]
        request = ArtifactWriteRequest(path=path, request=self._request_text(component))
        person = st.person
        trust = person.trust if person else 0.5
        action_vars = derive_action_variables(
            self.engine.state,
            trust=trust,
            defense_active=False,
        )
        placeholder_intent = ToolIntent(
            tool_name=WRITE_FILE,
            arguments={"path": request.path, "content": "<generated artifact>"},
            reason="User requested generated artifact saved to disk",
            expected_outcome=f"Write generated artifact to {request.path}",
            urgency=action_vars.action_urgency,
            risk_tolerance=action_vars.risk_tolerance,
            autonomy_bias=action_vars.autonomy_bias,
            clarification_threshold=action_vars.clarification_threshold,
            persistence_drive=action_vars.persistence_drive,
        )
        decision = make_tool_decision(
            placeholder_intent,
            action_vars,
            ToolCategory.WRITE,
            trust,
            agency_decision=st.agency_decision,
            autonomy_level=self._autonomy_level,
        )
        if decision.decision != "execute":
            return ToolLoopResult(
                trace=ToolTrace(
                    proposed_intents=[placeholder_intent],
                    final_decision=decision,
                    loop_count=0,
                ),
                action_variables=action_vars,
                tool_context_summary="",
            )

        try:
            draft = generate_artifact_draft(self._llm_backend_fast, request)
        except (ResearchBudgetExceeded, ResearchStudyLimit):
            raise
        except Exception as exc:
            failure = ToolResult(
                tool_name="artifact.generate",
                success=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
                side_effect_summary="artifact generation failed",
            )
            observation = appraise_tool_result(failure, ToolCategory.COGNITIVE)
            return ToolLoopResult(
                trace=ToolTrace(
                    proposed_intents=[placeholder_intent],
                    final_decision=ToolDecision(
                        decision="refuse",
                        intent=placeholder_intent,
                        rationale="Artifact generation failed before file write",
                    ),
                    executed_results=[failure],
                    observations=[observation],
                    loop_count=1,
                ),
                action_variables=action_vars,
                tool_context_summary=f"[artifact.generate] Failed: {failure.error}",
            )

        intent = ToolIntent(
            tool_name=WRITE_FILE,
            arguments={"path": request.path, "content": draft.content},
            reason=placeholder_intent.reason,
            expected_outcome=f"Write generated artifact to {request.path}",
            urgency=placeholder_intent.urgency,
            risk_tolerance=placeholder_intent.risk_tolerance,
            autonomy_bias=placeholder_intent.autonomy_bias,
            clarification_threshold=placeholder_intent.clarification_threshold,
            persistence_drive=placeholder_intent.persistence_drive,
        )
        decision.intent = intent
        result = self._tool_executor.execute(intent.tool_name, intent.arguments)
        record_operands(result, intent.arguments)
        observation = appraise_tool_result(result, ToolCategory.WRITE)
        summary = _format_generated_artifact_summary(result, draft.summary)
        return ToolLoopResult(
            trace=ToolTrace(
                proposed_intents=[intent],
                final_decision=decision,
                executed_results=[result],
                observations=[observation],
                loop_count=1,
            ),
            action_variables=action_vars,
            tool_context_summary=summary,
        )

    def _record_semantic_memory(
        self,
        *,
        user_message: str,
        assistant_response: str,
        user_id: str,
        topic: str,
        event: EmotionalEvent,
        life_influence: LifeInfluence | None = None,
        reply_length: str = "",
        lasting: tuple[tuple[str, str], ...] = (),
        learned: bool = False,
    ) -> float:
        """Persist semantic memories derived from the completed turn.

        What is kept as a preference or a decision is what the reading quoted
        as such (``lasting``); nothing is recognised by its wording.
        """
        salience_delta = 0.0
        if life_influence is not None:
            salience_delta = curiosity_salience_bonus(life_influence, learned)
        for entry in derive_semantic_entries(
            config=self._semantic_cfg,
            user_id=user_id,
            user_message=user_message,
            assistant_response=assistant_response,
            topic=topic,
            event_intensity=event.intensity,
            life_influence=life_influence,
            reply_length=reply_length,
            lasting=lasting,
            learned=learned,
        ):
            self.semantic_memory.store(entry)
        return round(salience_delta, 6)

    def _resolve_matching_tool_unresolved(self, tool_name: str) -> None:
        """Resolve prior tool/task tension once a matching tool succeeds."""
        if not tool_name:
            return
        repairable_sources = {"tool_failure", "blocked_action", "incomplete_task"}
        for item in list(self.engine.unresolved_items):
            if item.resolved or item.source not in repairable_sources:
                continue
            if tool_name in item.description:
                self.engine.resolve_item(item.id)

    # ------------------------------------------------------------------
    # Self-observation recording (v2 — Phase 3)
    # ------------------------------------------------------------------

    def _record_self_observations(
        self,
        event: EmotionalEvent,
        detected: DetectedEmotion,
        appraisal: AppraisalFrame,
        defense: DefenseActivation | None,
        state: ModulatorState,
        response: str,
        *,
        declined: bool = False,
    ) -> None:
        """Record 1-3 behavioral self-observations after each turn.

        ``declined`` says that the turn's decision declined something and the
        reply was delivered. That a refusal was expressed is taken from this
        fact of the decision, in whatever words the reply put it.
        """
        if self._research_mode:
            if self._research_owner is not None:
                self._research_owner.store.observe("behavior_observation", {
                    "response": response, "event": event.event_type.value,
                    "latent_state": state.to_dict(), "source": "selected_expression",
                })
            if declined:
                self.self_profile.record_behavior("expressed_refusal", 1.0, response[:200])
            return
        observations: list[tuple[str, float, str]] = []  # (trait, value, context)

        # Blunt: high certainty + directive response
        if state.certainty > 0.7:
            observations.append(("blunt", state.certainty, "high_certainty"))

        # Empathetic: user distress that is not aimed at OpenRook
        if detected.valence < 0.3 and not appraisal.targets_assistant:
            observations.append(("empathetic", 0.6, "negative_user_emotion"))

        # Defensive: defense mechanism fired
        if defense is not None:
            observations.append(("defensive", defense.raw_intensity, defense.defense_type))

        # Avoidant: avoidance topic active and energy low
        if state.energy < 0.3:
            observations.append(("avoidant", 0.5, "low_energy"))

        # Cap at 3 observations per turn
        for trait, value, context in observations[:3]:
            self.self_profile.record_behavior(trait, value, context)

    # ------------------------------------------------------------------
    # Self-check gating
    # ------------------------------------------------------------------

    @staticmethod
    def _should_use_llm_self_check(
        event: EmotionalEvent,
        contradiction_flags: list[str],
        dialogue_trace: InnerDialogueTrace | None,
    ) -> bool:
        """Use the LLM self-check only when the turn truly warrants it."""
        if event.intensity > 0.85:
            return True
        if contradiction_flags:
            return True
        if dialogue_trace is not None and dialogue_trace.reached_deadlock:
            return True
        return False

    # ------------------------------------------------------------------
    # Resolution source detection (v2)
    # ------------------------------------------------------------------

    def _check_resolution_sources(self, event: EmotionalEvent, text: str, user_id: str) -> None:
        """Social observations cannot close arbitrary loops or manufacture residue.

        Injury/grievance live in relationship dynamics. Explicit evidence-backed
        loop resolution is retained by RelationshipMemory, with identity required.
        """
        return None

    def _check_contradiction_resolution(
        self, contradiction_flags: list[str], user_id: str,
    ) -> None:
        """Wire contradictions as unresolved items (deduplicated)."""
        from core.clock import datetime
        from datetime import timezone
        from core.clock import identifiers as uuid

        active_descs = {i.description for i in self.engine.active_unresolved()}
        for flag in contradiction_flags:
            desc = flag[:120]
            if desc not in active_descs:
                self.engine.add_unresolved(UnresolvedItem(
                    id=f"contradiction_{uuid.uuid4().hex[:8]}",
                    source="contradiction",
                    description=desc,
                    created_at=datetime.now(timezone.utc),
                    intensity=0.5,
                    decay_rate=0.02,
                ))
                active_descs.add(desc)

    def _check_topic_resolution(
        self, topics: list[TopicProfile],
    ) -> None:
        """Wire avoidance/charged topics as unresolved items."""
        from core.clock import datetime
        from datetime import timezone
        from core.clock import identifiers as uuid

        for tp in topics:
            if tp.avoidance or tp.emotional_charge >= 0.5:
                # Don't add duplicate for same topic
                active_descs = {i.description for i in self.engine.active_unresolved()}
                desc = f"Charged topic: {tp.topic} (charge={tp.emotional_charge:.2f})"
                if desc not in active_descs:
                    self.engine.add_unresolved(UnresolvedItem(
                        id=f"topic_{uuid.uuid4().hex[:8]}",
                        source="topic",
                        description=desc,
                        created_at=datetime.now(timezone.utc),
                        intensity=min(0.7, tp.emotional_charge),
                        decay_rate=0.05,
                    ))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all database connections held by this pipeline.

        Call this when the pipeline is no longer needed (session eviction,
        shutdown). Safe to call multiple times. Also releases HTTP session
        pools held by the LLM backend and tool executor providers if they
        expose ``close()``.
        """
        if self._research_owner is not None:
            self._research_owner.detach(self)
        self.long_term.close()
        self.relationship_memory.close()
        self.semantic_memory.close()
        self._person_profile_store.close()
        self._self_profile_store.close()
        self.person_profiles.close()
        self.topic_profiles.close()
        # Release HTTP connection pools on LLM backends (safe for backends
        # without a close method, e.g. test fakes). Avoid double-close
        # when fast and primary share the same instance.
        seen: set[int] = set()
        for backend in (self._llm_backend, self._llm_backend_fast):
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover — defensive
                    pass
        # Release HTTP connection pools on tool providers registered via the
        # ToolExecutor (e.g., RequestsWebProvider). The executor exposes any
        # owned resources via ``owned_resources``.
        if self._tool_executor is not None:
            for resource in self._tool_executor.owned_resources:
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # pragma: no cover — defensive
                        pass

    def export_conversation_history(self) -> list[dict[str, str]]:
        """Return the hot in-session transcript for runtime persistence."""
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self._conversation_history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

    def restore_conversation_history(self, history: list[dict[str, str]]) -> None:
        """Restore a hot transcript saved by the runtime session manager."""
        self._conversation_history = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        self._assistant_replies = dict.fromkeys(m["content"] for m in self._conversation_history
                                                if m["role"] == "assistant")

    def export_session_state(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "assistant_replies": list(self._assistant_replies),
            "relationship_dynamics": self._dynamics.to_dict() if self._dynamics else None,
            "turn_index": self._turn_index,
            "short_replies": self._short_replies,
            "episode_id": self._episode_id, "short_term": self.short_term.export_state(),
            "session_memories_written": self._session_memory_write_count(),
            "active_task_plan": asdict(self._active_task_plan) if self._active_task_plan else None,
            "capability_gaps": self._capability_gaps, "last_turn_time": self._last_turn_time,
            "last_state_update_at": self._last_state_update_at,
            "last_external_input_at": self._last_external_input_at,
            "owned_pursuits": self._owned_pursuits if self._features.self_intent else {},
            "development_history": self._development_history,
            "development_episode_id": self._development_episode_id,
            "development_checkpointed": self._development_checkpointed,
        }

    def restore_session_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if int(state.get("schema_version", 1)) > 2:
            raise ValueError("Session state was written by a newer runtime")
        self._assistant_replies.update((reply, None) for reply in state.get("assistant_replies", [])
                                       if isinstance(reply, str) and reply.strip())
        saved = state.get("relationship_dynamics") or state.get("social_state")
        if saved:
            self._dynamics = RelationshipDynamics.from_dict(saved)
        self._turn_index = int(state.get("turn_index", 0))
        self._short_replies = bool(state.get("short_replies", False))
        self._saved_session_memory_writes = int(state.get("session_memories_written", 0))
        self._memory_write_baseline = self._memory_write_count()
        self._episode_id = str(state.get("episode_id") or self._episode_id)
        self._development_history = list(state.get("development_history", []))[-16:]
        self._development_episode_id = str(state.get("development_episode_id") or self._development_episode_id)
        self._development_checkpointed = bool(state.get("development_checkpointed", False))
        self.short_term.restore_state(state.get("short_term", []))
        if int(state.get("schema_version", 1)) < 2:
            # Old events already affected persisted mood/trust. Retain their
            # source records, but never integrate their trust/repair a second time.
            for entry in self.short_term.recent(len(self.short_term)):
                entry.event.metadata = {**entry.event.metadata, "legacy_not_reappraised": True}
        self._capability_gaps = dict(state.get("capability_gaps", {}))
        self._last_turn_time = state.get("last_turn_time")
        self._last_external_input_at = state.get("last_external_input_at", self._last_turn_time)
        self._last_state_update_at = state.get("last_state_update_at", self._last_turn_time)
        if self._features.self_intent:
            self._owned_pursuits = dict(state.get("owned_pursuits", {}))
        plan = state.get("active_task_plan")
        if plan:
            plan = dict(plan)
            steps = []
            for item in plan.pop("steps"):
                item = dict(item)
                item["status"] = TaskStatus(item["status"])
                if item.get("result"):
                    item["result"] = ToolResult(**item["result"])
                if item.get("observation"):
                    item["observation"] = ToolObservation(**item["observation"])
                steps.append(TaskStep(**item))
            plan["status"] = TaskStatus(plan["status"])
            self._active_task_plan = TaskPlan(steps=steps, **plan)

    def restore_state(
        self,
        snapshot: dict[str, float],
        saved_at: float | None = None,
        unresolved_items: list[UnresolvedItem] | None = None,
    ) -> None:
        """Restore emotional engine state from a persisted snapshot.

        Convenience wrapper around EmotionalEngine.restore() for runtime use.
        """
        self.engine.restore(
            snapshot,
            saved_at=saved_at,
            unresolved_items=unresolved_items,
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _memory_write_count(self) -> int:
        """Count committed memory records written by this pipeline's stores."""
        return (
            self.long_term.write_count
            + self.semantic_memory.write_count
            + self.relationship_memory.write_count
        )

    def _session_memory_write_count(self) -> int:
        return (
            self._saved_session_memory_writes
            + self._memory_write_count()
            - self._memory_write_baseline
        )

    def end_session(self, user_id: str = "default") -> DigestedSession:
        """Digest and clear the session, reporting all memory records it saved."""
        if self._research_owner is not None and not self._research_in_transition:
            return self._research_owner.end_session(self, user_id)
        self._ensure_bound_user(user_id)
        result = digest_session(
            self.short_term,
            self.long_term,
            source_person=user_id,
            relationship_memory=self.relationship_memory,
            llm_client=self._llm_backend,
            conversation_history=self._conversation_history,
            spikes_already_stored=True,
        )

        # Trust is updated per-turn (in process()), not again at session end.
        # The digested trust_delta is informational only.

        # Apply energy drain
        self.engine.state.energy = max(
            0.0, self.engine.state.energy - result.energy_drain
        )

        # Close the experience→self loop at consolidation, including ordinary
        # episodes. Assertions and self-expression are not independent evidence.
        if self._autobiographical_learning and self._features.life_history_context:
            try:
                self._maybe_ingest_autobiography(result, user_id)
            except (ResearchBudgetExceeded, ResearchStudyLimit):
                raise
            except Exception as exc:  # never let learning break session teardown
                log.warning("Autobiographical ingest failed: %s", exc)
        self._episode_id = uuid.uuid4().hex
        self._development_history.clear()
        self._development_episode_id = uuid.uuid4().hex
        self._development_checkpointed = False

        # Clear conversation history and session-scoped state
        self._conversation_history.clear()
        self._assistant_replies.clear()
        if not self._research_mode:
            self._active_task_plan = None
        self._proactive_count = 0
        self._last_proactive_at = None
        if not self._research_mode:
            self._capability_gaps.clear()

        # Include turn-time saves and relationship events, which digestion's
        # long-term-only count omits. Reset for the next conversation.
        result.memories_written = self._session_memory_write_count()
        self._saved_session_memory_writes = 0
        self._memory_write_baseline = self._memory_write_count()
        return result

    # Research retains its existing episode salience gates. Ordinary episodes
    # need material, rather than an emotional-intensity or trust threshold.
    _AUTOBIO_PEAK_INTENSITY = 0.6
    _AUTOBIO_TRUST_DELTA = 0.08

    def _is_session_salient(self, result: DigestedSession) -> bool:
        """Ordinary experience is eligible; research keeps its episode policy."""
        if not self._research_mode:
            return bool((result.summary or "").strip())
        return bool(
            result.spike_events
            or result.peak_intensity >= self._AUTOBIO_PEAK_INTENSITY
            or abs(result.trust_delta) >= self._AUTOBIO_TRUST_DELTA
            or result.unresolved_flags
        )

    def _maybe_ingest_autobiography(self, result: DigestedSession, user_id: str) -> None:
        """Interpret an ordinary episode through the current retained self."""
        if self._runtime_config is None:
            return
        summary = (result.summary or "").strip()
        if not summary or not self._is_session_salient(result):
            return
        if self._research_owner is not None:
            self._research_owner.queue_autobiography(self._episode_id, result, user_id)
            return
        # Checkpointed turns already supplied evidence. Session-end summaries
        # must not repackage them as another independent encounter.
        if self._development_history:
            self._ingest_live_development(user_id, self._llm_backend)
            return
        if self._development_checkpointed:
            return
        from runtime.life_history import LifeHistoryStore
        agent_name = self.soul.name
        arc = result.emotional_arc_label or "a conversation"
        with LifeHistoryStore(self._runtime_config) as store:
            store.ingest_external_text(
                title=f"Lived experience with {user_id}: {arc}"[:180],
                text=json.dumps({"session_summary": summary,
                    "dialogue": self._conversation_history[-40:],
                    "observed_state": self.engine.snapshot(),
                    "summary_is_interpretation": True,
                    "assistant_dialogue_is_expression_not_independent_self_evidence": True}, ensure_ascii=False),
                source_type="conversation_learning_text",
                source_ref=f"session:{user_id}",
                participants=[agent_name, user_id],
                llm_client=self._llm_backend,
                metadata={
                    "input_mode": "autobiographical",
                    "development_only": True,
                    "occurrence_id": f"autobiography:{self._episode_id}",
                    "arc": arc,
                    "peak_intensity": round(result.peak_intensity, 3),
                    "trust_delta": round(result.trust_delta, 3),
                    "topics": result.topics[:8],
                },
            )

    def _record_live_development(self, st: _TurnState) -> None:
        """Consolidate small, disjoint episodes after delivery, within its budget.

        The batch size bounds interpretation cost; it requires no conversational
        behavior. Choice and expression keep their frozen pre-learning snapshot.
        Research retains its separate recorded learning-job lifecycle.
        """
        if (self._research_mode or not self._autobiographical_learning
                or not self._features.life_history_context or self._runtime_config is None):
            return
        if st.debug.delivery_outcome in {"backend_failure", "expression_failure"}:
            return
        self._development_history.append({
            "user": st.user_message, "expression": st.gen_result.response,
            "delivery_outcome": st.debug.delivery_outcome,
            "observed_state": self.engine.snapshot(), "turn_id": st.turn_id,
        })
        self._development_history = self._development_history[-16:]
        st.debug.self_development = {"status": "pending", "pending_turns": len(self._development_history)}
        if len(self._development_history) < 2 or st.gen_result.deliberate_silence or not remaining_calls():
            return
        backend = SimpleNamespace(generate=lambda system, user: generate(
            self._llm_backend, "self_development", system, user, optional=True))
        started = time.perf_counter()
        try:
            st.debug.self_development = self._ingest_live_development(st.user_id, backend)
        except (ResearchBudgetExceeded, ResearchStudyLimit):
            raise
        except Exception as exc:
            # Retain pending material and its occurrence ID for the next
            # available checkpoint; no inferred personality on failure.
            st.debug.self_development["status"] = "unavailable"
            st.debug.self_development["error"] = type(exc).__name__
        st.timings["self_development"] = (time.perf_counter() - started) * 1000

    def _ingest_live_development(self, user_id: str, backend) -> dict:
        from runtime.life_history import LifeHistoryStore
        dialogue = [item for turn in self._development_history for item in (
            {"role": "user", "content": turn["user"]},
            {"role": "assistant", "content": turn["expression"]})]
        material = {"dialogue": dialogue,
            "observed_outcomes": [{k: v for k, v in turn.items() if k not in {"user", "expression"}}
                                  for turn in self._development_history],
            "assistant_dialogue_is_expression_not_independent_self_evidence": True}
        with LifeHistoryStore(self._runtime_config) as store:
            result = store.ingest_external_text(
                title=f"Interaction episode with {user_id}"[:180],
                text=json.dumps(material, ensure_ascii=False),
                source_type="conversation_learning_text", source_ref=f"session:{user_id}",
                participants=[self.soul.name, user_id], llm_client=backend,
                metadata={"input_mode": "autobiographical", "development_only": True,
                    "occurrence_id": f"autobiography:{self._development_episode_id}"})
            development = result.get("development", {})
        self._development_history.clear()
        self._development_episode_id = uuid.uuid4().hex
        self._development_checkpointed = True
        return development

    def apply_rest(self, hours: float) -> None:
        """Simulate time passing between sessions. Recovers energy, decays modulators."""
        seconds = hours * 3600.0
        self.engine.decay(seconds)

    def _ensure_bound_user(self, user_id: str) -> None:
        """Bind the pipeline to a single relational user for its lifetime."""
        if self._bound_user_id is None:
            self._bound_user_id = user_id
            return
        if user_id != self._bound_user_id:
            raise RuntimeError(
                "CognitivePipeline instances are single-user/session scoped. "
                f"Bound to '{self._bound_user_id}', got '{user_id}'. "
                "Create a separate pipeline per user."
            )

    # ------------------------------------------------------------------
    # Event classification (LLM with rule-based fallback)
    # ------------------------------------------------------------------

    def _classify_event(
        self,
        text: str,
        detected: DetectedEmotion,
        appraisal: AppraisalFrame,
    ) -> EmotionalEvent:
        """Classify user message into an EmotionalEvent.

        Always uses rule-based heuristics (0 LLM calls).
        LLM classification available via _classify_event_via_llm() if needed.
        """
        return self._classify_event_via_rules(text, detected, appraisal)

    def _classify_event_via_llm(
        self,
        text: str,
        detected: DetectedEmotion,
        appraisal: AppraisalFrame,
    ) -> EmotionalEvent | None:
        """LLM-based event classification. Returns None on failure."""
        prompt_template = get_config().classify_event_prompt
        if not prompt_template:
            return None

        prompt = prompt_template.replace("{arousal}", f"{detected.arousal:.2f}")
        prompt = prompt.replace("{valence}", f"{detected.valence:.2f}")
        prompt = prompt.replace("{certainty}", f"{detected.certainty:.2f}")
        prompt = prompt.replace("{intensity}", f"{detected.intensity:.2f}")

        try:
            raw = self._llm_backend.generate(prompt, text)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            event_type_str = data.get("event_type", "user_message")
            intensity = float(data.get("intensity", 0.3))

            # Validate event type
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                return None

            return EmotionalEvent(
                event_type=event_type,
                intensity=max(0.0, min(1.0, intensity)),
                source="user",
                metadata=self._appraisal_metadata(appraisal),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _classify_event_via_rules(
        self,
        text: str,
        detected: DetectedEmotion,
        appraisal: AppraisalFrame,
    ) -> EmotionalEvent:
        """Display event categories; update_relationship alone integrates reaction."""
        kinds = {"attack": EventType.CONFLICT, "complaint": EventType.NEGATIVE_FEEDBACK,
                 "gratitude": EventType.POSITIVE_FEEDBACK, "connection": EventType.WARMTH}
        kind = EventType(appraisal.event_type) if appraisal.event_type else kinds.get(appraisal.social_move, EventType.USER_MESSAGE)
        if not appraisal.targets_assistant and kind in {EventType.CONFLICT, EventType.BETRAYAL, EventType.NEGATIVE_FEEDBACK}:
            kind = EventType.USER_MESSAGE
        intensity = appraisal.event_intensity if appraisal.provenance == "model" and appraisal.event_type else appraisal.perception.intensity
        return EmotionalEvent(event_type=kind, intensity=intensity, source="user",
                              metadata=dict(self._appraisal_metadata(appraisal), policy_version="willingness-1"))

    @staticmethod
    def _appraisal_metadata(appraisal: AppraisalFrame) -> dict[str, Any]:
        """Attach compact appraisal state to events for downstream use."""
        return {
            "primary_target": appraisal.primary_target,
            "social_move": appraisal.social_move,
            "inferred_intent": appraisal.inferred_intent,
            "targets_assistant": appraisal.targets_assistant,
            "blame": appraisal.blame,
            "controllability": appraisal.controllability,
            "expectation_violation": appraisal.expectation_violation,
            "vulnerability": appraisal.vulnerability,
            "affiliation_bid": appraisal.affiliation_bid,
            "mixed_affect": appraisal.mixed_affect,
        }

    def _detect_topics(self, text: str) -> list[TopicProfile]:
        """Detect active topics via substring matching (0 LLM calls).

        LLM detection available via _detect_topics_via_llm() if needed.
        """
        all_topic_profiles = self.topic_profiles.all_profiles()
        if not all_topic_profiles:
            return []

        topics = []
        lower = text.lower()
        for tp in all_topic_profiles:
            if tp.topic.lower() in lower:
                topics.append(tp)
        return topics

    def _detect_topics_via_llm(
        self, text: str, all_profiles: list[TopicProfile]
    ) -> list[TopicProfile] | None:
        """LLM-based topic detection. Returns None on failure."""
        prompt_template = get_config().detect_topics_prompt
        if not prompt_template:
            return None

        known_list = ", ".join(tp.topic for tp in all_profiles)
        prompt = prompt_template.replace("{known_topics}", known_list)

        try:
            raw = self._llm_backend.generate(prompt, text)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            matched_names = set(t.lower() for t in data.get("topics", []))
            return [tp for tp in all_profiles if tp.topic.lower() in matched_names]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _event_valence(event: EmotionalEvent) -> float:
        """Map event to signed valence for trust math.

        ``targets_assistant`` defaults to True to match
        :func:`core.memory.digestion._event_valence` — the authoritative
        relationship-trust writer. The two must agree on the default or the
        same event scores opposite trust deltas on the spike vs. digested
        paths (a 1:1 emotional event targets the assistant unless marked
        otherwise).
        """
        positive = {EventType.POSITIVE_FEEDBACK, EventType.RESOLUTION, EventType.WARMTH}
        negative = {EventType.NEGATIVE_FEEDBACK, EventType.CONFLICT, EventType.BETRAYAL}
        targets_assistant = bool(event.metadata.get("targets_assistant", True))
        if event.event_type in positive:
            if not targets_assistant:
                return 0.0
            return event.intensity
        elif event.event_type in negative:
            if not targets_assistant:
                return 0.0
            return -event.intensity
        return 0.0

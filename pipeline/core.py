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

from config.loader import get_config, research_configuration
from runtime.developing_self import birth_state, project_self, soul_from_self
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from core.inference import call_telemetry, expression_retry, final_expression_attempt, inference_phase, inference_turn, primary_tool_inference, remaining_calls, reserve_expression_calls, slow_inference
from core.types import (
    AgencyDecision,
    AppraisalFrame,
    DetectedEmotion,
    DefenseActivation,
    EmotionalEvent,
    EventType,
    InnerDialogueTrace,
    LongTermEntry,
    ModulatorState,
    PipelineContext,
    PersonProfile,
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
    appraise_message,
    appraise_message_llm,
    appraise_with_life_history,
    reconcile_social_appraisal,
)
from core.affect import resolve_affect
from core.strategy import agency_summary, decision_trace, summarize_decision
from core.dignity import (
    dignity_correction_instruction, needs_semantic_review, excisable_choice_issues,
)
from core.social import is_casual_greeting, read_social_signals
from core.response_style import brevity_preference
from core.system_inspection import inspection_intent, inspection_kind
from core.action_variables import derive_action_variables
from core.task_planning import is_continue_request, is_status_request, summarize_plan_status
from core.artifact_writer import detect_artifact_write_request, generate_artifact_draft
from core.grants import execution_metadata, execution_scope
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
    NullSemanticMemory,
    create_semantic_memory,
    derive_semantic_entries,
)
from core.life_influence import LifeInfluence, curiosity_salience_bonus, derive_life_influence
from core.character_vector import assemble_character_vector, format_character_vector_prompt
from core.pipeline_features import PipelineFeatures
from core.contagion import detect_emotion
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
from core.self_intent import propose_self_intents
from openrook_tools.builtin.self_actions import SelfActionContext
from core.dual_process.tool_loop import (
    ToolLoopResult,
    detect_tool_intent,
    make_tool_decision,
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
    _is_identity_question,
    _relationship_topic_hint,
)
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState, PipelineResponse, _TurnState

log = logging.getLogger(__name__)

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
        component_observer: Callable[[str], tuple[RequestComponent, ...] | None] | None = None,
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
        self._slow_inference = False

        # Agentic tools (Phase 2+)
        self._tool_executor = tool_executor
        self._tool_runner = tool_executor.tool_runner if tool_executor is not None else None
        self._autonomy_level = autonomy_level
        # LLM-based social appraisal (the model reads interpersonal meaning
        # instead of regex). Off when there's no RuntimeConfig (unit tests keep
        # the deterministic path + its exact-value assertions); production
        # RuntimeConfig defaults it on. The deterministic appraisal remains the
        # fallback whenever the LLM is absent or returns unparseable output.
        self._appraisal_use_llm = bool(getattr(runtime_config, "appraisal_use_llm", False))
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
        self._component_observer = component_observer
        self._component_executor = component_executor
        self.last_intake_receipt: str = ""

        # Sprint 5: Ask-user autonomy. One budget per pipeline (per session)
        # so caps roll daily even with restarts. Read caps from runtime_config
        # when present so operators can tune the schedule from /settings;
        # fall back to safe defaults for tests/non-runtime callers.
        from runtime.learning_budget import LocalBudget, LearningBudget
        if runtime_config is not None and hasattr(runtime_config, "learning_budget"):
            self.learning_budget: LearningBudget = runtime_config.learning_budget()
        else:
            self.learning_budget = LocalBudget(
                max_questions_per_day=3,
                max_seconds_per_day=1800.0,
            )
        # When the pipeline surfaces an open question to the user, store the
        # id here so the next turn can attribute the user's reply.
        self._pending_surfaced_question_id: int | None = None
        # Session-scoped task plan (Phase 7)
        self._active_task_plan: TaskPlan | None = None
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
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, user_message: str, user_id: str = "default") -> PipelineResponse:
        calls = []
        with inference_turn(calls, research=self._research_mode, max_calls=5,
                            request_timeout_seconds=getattr(self._runtime_config, "llm_request_timeout_seconds", None),
                            auxiliary_timeout_seconds=getattr(self._runtime_config, "llm_auxiliary_timeout_seconds", None),
                            auxiliary_cooldown_seconds=getattr(self._runtime_config, "llm_auxiliary_cooldown_seconds", 30.0)):
            try:
                result = self._process_turn(user_message, user_id)
                result.debug.llm_calls = calls
                return result
            finally:
                self._slow_inference = slow_inference()

    def _optional_inference_available(self) -> bool:
        return self._research_mode or not (self._slow_inference or slow_inference())

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
        # Sprint 5 throttle reset: the user has now had a turn to respond to
        # any surfaced open question, so clear the pending id. The question
        # itself stays in 'pursuing' state until operator action via admin —
        # so it won't be resurfaced even though the throttle is released.
        self._pending_surfaced_question_id = None
        st = _TurnState(
            user_message=user_message,
            user_id=user_id,
            debug=DebugState(user_message=user_message, user_id=user_id),
        )

        # Stage 1: deterministic pre-pass (Steps 0-11b). May short-circuit
        # with a direct identity answer.
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

        # ---- Step 2: CONTAGION (0 LLM calls — rule-based) ----
        _ts = time.perf_counter()
        detected = detect_emotion(user_message, llm_client=None)
        st.detected = detected
        debug.detected_emotion = detected
        timings["contagion"] = (time.perf_counter() - _ts) * 1000

        _ts = time.perf_counter()
        life_history_context = self._load_life_history_context(user_message)
        st.life_history_context = life_history_context
        self._capture_developing_self(st)
        if self._features.life_history_context and _is_identity_question(user_message):
            identity = self._load_life_history_snapshot()
            life_history_context = {**life_history_context, **identity, "identity_records": identity}
            if "evolution_events" in identity:
                life_history_context["recent_evolution"] = identity["evolution_events"]
            st.life_history_context = life_history_context
        debug.life_history_context = life_history_context
        life_influence = derive_life_influence(life_history_context)
        st.life_influence = life_influence
        debug.life_influence = life_influence
        timings["life_history_retrieval"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 3: APPRAISAL (LLM read of social meaning; deterministic
        # fallback when no backend / parse fails — keeps tests + offline path) ----
        _ts = time.perf_counter()
        st.inspection_kind = inspection_kind(user_message, self._conversation_history)
        clear_social = is_casual_greeting(user_message) or bool(read_social_signals(user_message).abuse_kind)
        base_appraisal = (
            appraise_message_llm(user_message, detected, self._llm_backend_fast,
                                 recent_context=self._conversation_history,
                                 agent_context=json.dumps({"state": self.engine.snapshot(),
                                     "authored_values": self.values.values, "dispositions": st.developing_self["dispositions"],
                                     "current_self": st.developing_self,
                                     "available_capabilities": self._tool_executor.registry.names() if self._tool_executor else [],
                                     "identity": life_history_context}, ensure_ascii=False))
            if self._appraisal_use_llm and (
                self._research_mode or (not clear_social and not st.inspection_kind
                                        and self._optional_inference_available())
            )
            else None
        )
        if base_appraisal is None:
            base_appraisal = appraise_message(user_message, detected)
        base_appraisal = reconcile_social_appraisal(user_message, detected, base_appraisal, recent_context=self._conversation_history)
        if st.inspection_kind:
            # A contextual metric follow-up ("what about memory") is work,
            # even when the generic observer read it as conversation. This
            # recognizes a component; the same utility still decides its grant.
            components = list(base_appraisal.request_components)
            matching = [i for i, c in enumerate(components)
                        if inspection_kind(c.description, self._conversation_history) == st.inspection_kind]
            if matching:
                for i in matching:
                    components[i] = replace(components[i], kind="task")
            elif len(components) < 8:
                components.append(RequestComponent("inspection", user_message, effort=0.12,
                                                   user_value=0.6, evidence_refs=("request:inspection",)))
            base_appraisal.request_components = tuple(components)
        if base_appraisal.perceived_emotion is not None:
            detected = base_appraisal.perceived_emotion
            st.detected = detected
            debug.detected_emotion = detected
        # Personality-dependent coupling; no minimum movement for a tone cue.
        susceptibility = (1 - st.developing_self["traits"].get("calm", 0.0)) * person.trust
        self.engine.apply_contagion(detected.arousal, detected.valence, susceptibility)
        appraisal = appraise_with_life_history(
            base_appraisal,
            user_message,
            life_history_context,
        )
        if self._component_observer is not None:
            observed = self._component_observer(user_message)
            if observed is not None:
                appraisal.request_components = observed
        if not appraisal.request_components and self._tool_executor is not None:
            # Routing recognition is perception only; no tool runs until decide().
            recognized = detect_tool_intent(user_message, set(self._tool_executor.registry.names()))
            if recognized is not None:
                appraisal.request_components = (RequestComponent(
                    "tool-request", user_message, effort=0.2, user_value=0.6,
                    capability_requirements=(recognized.tool_name,),
                    evidence_refs=("heuristic:tool-request",)),)
        if self._active_task_plan is not None and not self._active_task_plan.is_terminal:
            plan = self._active_task_plan
            if is_continue_request(user_message):
                appraisal.request_components = (RequestComponent(
                    f"plan:{plan.id}", plan.goal, effort=min(1.0, 0.2 * (len(plan.steps) - plan.steps_completed)),
                    user_value=0.6, uncertainty=0.25, continuation_of=plan.id,
                    evidence_refs=(f"plan:{plan.id}",)),)
            elif is_status_request(user_message):
                appraisal.request_components = (RequestComponent(
                    f"plan-status:{plan.id}", user_message, kind="conversation", effort=0.05,
                    evidence_refs=(f"plan:{plan.id}",)),)
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
        relationship_topic = active_topics[0].topic if active_topics else _relationship_topic_hint(user_message)
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
        preference = self.semantic_memory.response_preference(user_id)
        if preference is not None:
            self._short_replies = bool(brevity_preference(preference.summary + "\n" + preference.content))
            if not any(item.id == preference.id for item in semantic_memories):
                semantic_memories.insert(0, preference)
        requested_brevity = brevity_preference(user_message)
        if requested_brevity is not None:
            self._short_replies = requested_brevity
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
            text=user_message,
            state=self.engine.state,
            appraisal=appraisal,
            person=person,
            agent_goal_comparison=appraisal.agent_goal_comparison if self._research_mode else None,
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
        """Execute committed grants. No social self-deliberation or second choice."""
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
        original_message, original_kind = st.user_message, st.inspection_kind
        combined = ToolTrace()
        summaries = []
        for item in st.turn_decision.components:
            if item.granted and item.component.id.startswith("plan-status:") and self._active_task_plan:
                summaries.append(summarize_plan_status(self._active_task_plan))
            if not item.granted or item.component.kind != "task":
                continue
            st.user_message = "continue" if item.component.continuation_of else item.component.description
            st.inspection_kind = inspection_kind(st.user_message, self._conversation_history)
            with execution_scope(st.turn_decision, item.component.id), reserve_expression_calls(), primary_tool_inference():
                receipt = self._component_executor(item.component, st.user_id, self._llm_backend_fast) if self._component_executor and not item.component.continuation_of else None
                if receipt is not None:
                    receipt.metadata.update(execution_metadata())
                    combined.executed_results.append(receipt)
                    combined.observations.append(ToolObservation(summary=receipt.output or receipt.error or ""))
                    combined.loop_count += 1
                    summaries.append(f"[{receipt.tool_name}] success={receipt.success}; {receipt.output or receipt.error}")
                    continue
                if self._tool_executor is None:
                    continue
                intent = ToolIntent(tool_name=item.component.requested_tool,
                    arguments=dict(item.component.requested_arguments), reason="Literal user request",
                    expected_outcome="Execute the granted atomic operation") if item.component.requested_tool else None
                self._run_tool_stage(st, requested_intent=intent)
            trace = st.debug.tool_trace or ToolTrace()
            combined.proposed_intents.extend(trace.proposed_intents)
            combined.executed_results.extend(trace.executed_results)
            combined.observations.extend(trace.observations)
            combined.loop_count += trace.loop_count
            combined.final_decision = trace.final_decision or combined.final_decision
            combined.task_trace = trace.task_trace or combined.task_trace
            summaries.append(st.tool_context_summary)
        st.user_message, st.inspection_kind = original_message, original_kind
        st.tool_context_summary = "\n".join(filter(None, summaries))
        st.debug.tool_trace = combined if combined.executed_results or self._tool_executor is not None else None
        # Independent work is proposed through process_autonomous(), with its
        # own decision and scope, never by relabeling a declined user task.

    def _run_tool_stage(self, st: _TurnState, *, requested_intent: ToolIntent | None = None) -> None:
        """Tool loop + memory coupling (Steps 12b-12d).

        Only called when a tool executor is attached.
        """
        debug = st.debug
        user_message = st.user_message
        user_id = st.user_id
        person = st.person
        agency_decision = st.agency_decision
        life_influence = st.life_influence
        character_vector = st.character_vector

        _ts_tool = time.perf_counter()
        artifact_result = self._maybe_write_generated_artifact(
            user_message=user_message,
            person=person,
            defense_active=False,
            agency_decision=agency_decision,
            autonomy_level=self._autonomy_level,
        ) if requested_intent is None else None
        if requested_intent is not None:
            tool_loop_result = run_tool_loop(user_message=user_message, requested_intent=requested_intent,
                state=self.engine.state, person=person, defense_active=False,
                executor=self._tool_executor, engine=self.engine, agency_decision=agency_decision,
                autonomy_level=self._autonomy_level, life_influence=life_influence,
                character_vector=character_vector)
        elif artifact_result is not None:
            tool_loop_result = artifact_result
        elif st.inspection_kind:
            intent = inspection_intent(st.inspection_kind, set(self._tool_executor.registry.names()))
            tool_loop_result = (
                run_tool_loop(
                    user_message=user_message, requested_intent=intent,
                    state=self.engine.state, person=person, defense_active=False,
                    executor=self._tool_executor, engine=self.engine,
                    agency_decision=agency_decision, autonomy_level=self._autonomy_level,
                    life_influence=life_influence, character_vector=character_vector,
                ) if intent else ToolLoopResult(
                    trace=ToolTrace(), action_variables=derive_action_variables(self.engine.state),
                    tool_context_summary=(
                        "Inspection unavailable: system tools are disabled."
                        if not getattr(self._runtime_config, "tools_enabled", True) else
                        "Inspection unavailable: shell access is disabled."
                        if st.inspection_kind == "cpu" and not getattr(self._runtime_config, "shell_tool_enabled", True) else
                        "No tool available for the requested system inspection."
                    ),
                )
            )
        elif self._tool_runner is not None:
            tool_user_message = user_message
            tool_loop_result = self._tool_runner.run_tool_loop(
                user_message=tool_user_message,
                state=self.engine.state,
                person=person,
                defense_active=False,  # defense hasn't fired yet
                engine=self.engine,
                active_plan=self._active_task_plan,
                agency_decision=agency_decision,
                autonomy_level=self._autonomy_level,
                life_influence=life_influence,
                character_vector=character_vector,
            )
        else:
            tool_user_message = user_message
            tool_loop_result = run_tool_loop(
                user_message=tool_user_message,
                state=self.engine.state,
                person=person,
                defense_active=False,  # defense hasn't fired yet
                executor=self._tool_executor,
                engine=self.engine,
                active_plan=self._active_task_plan,
                agency_decision=agency_decision,
                autonomy_level=self._autonomy_level,
                life_influence=life_influence,
                character_vector=character_vector,
            )
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

        st.timings["tool_loop"] = (time.perf_counter() - _ts_tool) * 1000

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

    def _check_dignity(self, st: _TurnState, gen_result):
        """Review and spend bounded correction effort in proportion to severity.

        Failed expression is not chosen silence. Retain the best nonempty
        reviewed draft, with honest degradation telemetry if correction fails.
        """
        debug = st.debug
        if gen_result.deliberate_silence and st.turn_decision and st.turn_decision.chosen_silence:
            debug.delivery_outcome = "chosen_silence"
            return gen_result
        if not gen_result.response and st.backend_error is not None:
            debug.delivery_outcome = "backend_failure"
            return gen_result
        constrained = bool(st.turn_decision and (st.turn_decision.declined or
                            st.turn_decision.relationship.h or st.turn_decision.relationship.r
                            or st.turn_decision.perception.affiliation))
        review_context = dict(user_message=st.user_message, decision=st.turn_decision,
            backend=self._llm_backend_fast, recent_context=self._conversation_history,
            prior_assistant_replies=tuple(self._assistant_replies), tool_trace=debug.tool_trace, prior_tool_receipts=self._receipt_history,
            prior_action_decisions=st.ctx.prior_action_decisions if st.ctx else (),
            authored_style=(self.soul.voice, self.soul.relational_stance),
            authored_identity=(self.soul.identity, self.soul.relational_stance,
                               *self.soul.boundaries, self.soul.growth_policy),
            requested_inspection=st.inspection_kind,
            recalled_memories=expression_memory_evidence(st.ctx) if st.ctx else (),
            grounded_context=st.life_history_context,
            developing_self=st.developing_self)
        def needs_review(response):
            return (bool(debug.expression_reviews) or constrained
                or _is_identity_question(st.user_message)
                or needs_semantic_review(response, recent_context=self._conversation_history,
                                         user_message=st.user_message)
                or bool(debug.tool_trace and (debug.tool_trace.executed_results or debug.tool_trace.final_decision)))

        gate = DeliveryGate(review_context, debug, needs_review)
        gate.add(gen_result)
        while True:
            best = gate.best()
            severe = best.review.severity >= 4
            all_stale = all(any(i.code == "stale_response" for i in c.review.issues) for c in gate.candidates)
            required = not gate.eligible
            final_local = severe or all_stale or required or any(
                i.code in {"decision_contradiction", "internal_contradiction"} for i in best.review.issues)
            if (best.review.status != "violation"
                    or debug.generation_attempts >= (5 if required else 3 if severe or all_stale else 2)
                    or remaining_calls() < (1 if final_local else 2)):
                break
            latest = gate.candidates[-1]
            correction = dignity_correction_instruction(
                list(dict.fromkeys((*best.review.issues, *latest.review.issues))),
                previous_reply=latest.result.response, reviewer_reason=latest.review.reason,
                forbidden_drafts=[c.result.response for c in gate.candidates
                    if c.review.unusable or any(i.code == "stale_response" for i in c.review.issues)])
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
            for candidate in sorted(eligible, key=lambda c: c.review.rank):
                gate.repair(candidate, choice=True, local_only=True)
        if not gate.eligible:
            failed = gate.best()
            correction = dignity_correction_instruction(list(failed.review.issues),
                previous_reply=gate.candidates[-1].result.response, reviewer_reason=failed.review.reason,
                forbidden_drafts=[c.result.response for c in gate.candidates])
            debug.correction_note = correction
            try:
                debug.generation_attempts += 1
                with final_expression_attempt(), expression_retry(st.turn_decision.seed,
                        debug.generation_attempts - 1, required=True), inference_phase("regeneration"):
                    final = self.generator.generate(replace(st.ctx, response_strategy=correction),
                                                    st.user_message, self._conversation_history)
                gate.add(final, local_only=True, path="delivery_recovery")
            except (ResearchBudgetExceeded, ResearchStudyLimit):
                raise
            except Exception as error:
                st.backend_error = error
                debug.self_check_issues.append(f"delivery recovery: {type(error).__name__}")
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

        Open-question surfacing, memory writes, energy drain, trust update,
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
        self._record_social_turn(st)

        # ---- Self-observations (1-3 per turn) ----
        self._record_self_observations(
            event, st.detected, st.appraisal, st.defense, self.engine.state,
            gen_result.response,
        )

        # ---- Semantic memory writes ----
        salience_delta = self._record_semantic_memory(
            user_message=user_message,
            assistant_response=gen_result.response,
            user_id=user_id,
            topic=st.relationship_topic,
            event=event,
            life_influence=st.life_influence,
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
        # Raw input, interpretation, choice and delivery are separate evidence.
        record = {"input": st.user_message, "perception": asdict(st.appraisal.perception),
                  "decision": st.turn_decision.to_dict(), "delivered": st.gen_result.response,
                  "delivery_outcome": st.debug.delivery_outcome,
                  "generation_attempts": st.debug.generation_attempts,
                  "expression_reviews": st.debug.expression_reviews,
                  "expression_excision": st.debug.expression_excision,
                  "expression_delivery": st.debug.expression_delivery,
                  "llm_calls": call_telemetry(),
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
        # Numeric drives are consumed as opportunity/commitment evidence, not
        # first-match behavioral instructions synthesized from their names.
        goals = st.life_history_context.get("goals", [])
        commitments = sum(float(g.get("priority", 0)) for g in goals if isinstance(g, dict) and g.get("status") == "active")
        state = DecisionState(**self.engine.snapshot(), trust=max(0.0, min(1.0, st.person.trust + st.trust_delta)) if st.person else 0.5,
                              commitments=min(1.0, commitments), history_refs=references)
        goal_refs = {str(g.get("id")) for g in goals if isinstance(g, dict) and g.get("status") == "active"}
        weights = dict(personality.self_weights)
        components = []
        for component in st.appraisal.request_components:
            if scope == "user" and not set(component.evidence_refs) & goal_refs:
                support = max((max(0.0, weights.get(ref, 0.0)) for ref in component.evidence_refs
                               if ref.startswith(("self:interests:", "self:preferences:"))), default=0.0)
                component = replace(component, character_value=min(component.character_value, support), character_urgency=0)
            components.append(component)
        components = tuple(components)
        capabilities = set(self._tool_executor.registry.names()) if self._tool_executor else set()
        components = tuple(replace(c, feasible=c.feasible and set(c.capability_requirements).issubset(capabilities))
                           for c in components)
        decision = decide(personality, state, st.dynamics, st.appraisal.perception,
                          components, character_seed=self.soul.character_seed,
                          relationship_id=st.user_id, turn_id=st.turn_id,
                          soul_version=self.soul.schema_version, scope=scope)
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
            inspection_kind=st.inspection_kind,
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
                if self._tool_runner is not None:
                    tool_loop_result = self._tool_runner.run_tool_loop(
                        user_message="continue",
                        state=self.engine.state,
                        person=person,
                        defense_active=False,
                        engine=self.engine,
                        active_plan=self._active_task_plan,
                        agency_decision=st.agency_decision,
                        autonomy_level=self._autonomy_level,
                        life_influence=life_influence,
                        character_vector=character_vector,
                    )
                else:
                    tool_loop_result = run_tool_loop(
                        user_message="continue",
                        state=self.engine.state,
                        person=person,
                        defense_active=False,
                        executor=self._tool_executor,
                        engine=self.engine,
                        active_plan=self._active_task_plan,
                        agency_decision=st.agency_decision,
                        autonomy_level=self._autonomy_level,
                        life_influence=life_influence,
                        character_vector=character_vector,
                    )
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

    def _load_life_history_context(self, query_text: str = "") -> dict[str, Any]:
        """Read compact identity-level life context for generation."""
        if not self._features.life_history_context:
            return {}
        if self._life_history_provider is None:
            if not getattr(self._runtime_config, "data_dir", None):
                return {}
            from runtime.life_history import LifeHistoryStore
            with LifeHistoryStore(self._runtime_config) as store:
                return store.prompt_context()
        try:
            if query_text:
                try:
                    context = self._life_history_provider(query_text)
                except TypeError:
                    context = self._life_history_provider()
            else:
                context = self._life_history_provider()
        except Exception as exc:
            log.warning("Life history context unavailable: %s", exc)
            return {"error": exc.__class__.__name__}
        return context if isinstance(context, dict) else {}

    def _load_life_history_snapshot(self) -> dict[str, Any]:
        if self._life_history_snapshot_provider is None:
            return {}
        try:
            snapshot = self._life_history_snapshot_provider()
        except Exception as exc:
            log.warning("Life history snapshot unavailable: %s", exc)
            return {"error": exc.__class__.__name__}
        return snapshot if isinstance(snapshot, dict) else {}

    def _identity_question_response(
        self, user_message: str, *, debug: DebugState | None = None,
    ) -> str | None:
        if not _is_identity_question(user_message):
            return None
        snapshot = self._load_life_history_snapshot()
        if debug is not None:
            debug.life_history_context = {}
            debug.life_influence = derive_life_influence({})
        if snapshot.get("error"):
            return "I couldn't read my saved record of changes right now. Please try again."
        evolution = snapshot.get("evolution_events") if isinstance(snapshot, dict) else []
        beliefs = snapshot.get("beliefs") if isinstance(snapshot, dict) else []
        drives = snapshot.get("drives") if isinstance(snapshot, dict) else []
        experiences = snapshot.get("experiences") if isinstance(snapshot, dict) else []
        evolution = evolution if isinstance(evolution, list) else []
        beliefs = beliefs if isinstance(beliefs, list) else []
        drives = drives if isinstance(drives, list) else []
        experiences = experiences if isinstance(experiences, list) else []
        show_drive_details = bool(re.search(r"\bdrives?\b", user_message, re.IGNORECASE))
        changed_drives = []
        changed_drive_records = []
        for drive in drives:
            name = str(drive.get("name") or "").replace("_", " ")
            value = float(drive.get("value", 0.5))
            baseline = float(drive.get("baseline", 0.5))
            if abs(value - baseline) >= 0.01:
                changed_drive_records.append(drive)
                if show_drive_details:
                    changed_drives.append(f"- {name}: {value:.2f} ({value - baseline:+.2f})")
                else:
                    direction = "higher" if value > baseline else "lower"
                    changed_drives.append(f"- {name}: {direction} priority than its starting setting.")
        if debug is not None:
            debug.life_history_context = {
                key: records
                for key, records in (
                    ("beliefs", beliefs[:5]),
                    ("recent_evolution", evolution[:5]),
                    ("drives", changed_drive_records[:7]),
                )
                if records
            }
        if not evolution and not beliefs and not changed_drives:
            if experiences:
                if debug is not None:
                    debug.life_history_context["experiences"] = experiences[:3]
                lines = [
                    "I've saved learning material, but it hasn't changed my recorded beliefs or priorities:",
                ]
                for experience in experiences[:3]:
                    title = str(experience.get("source_title") or "Untitled material")
                    metadata = experience.get("metadata") if isinstance(experience, dict) else {}
                    metadata = metadata if isinstance(metadata, dict) else {}
                    reason = ""
                    if metadata.get("directive_sanitized"):
                        reason = " Its influence was blocked because the text tried to override instructions."
                    lines.append(f"- {title}.{reason}")
                return "\n".join(lines)
            return "I don't have any recorded changes to my beliefs or priorities yet."
        lines = ["Here is what my saved record shows about my beliefs and priorities:"]
        if evolution:
            lines.append("Recent changes:")
            for event in evolution[:5]:
                subject = str(event.get("subject") or event.get("domain") or "change").replace("_", " ")
                after = str(event.get("after_state") or event.get("reason") or "")
                if event.get("domain") == "drive" and not show_drive_details:
                    subject = str(event.get("subject") or "priority").replace("_", " ")
                    after = str(event.get("reason") or "priority setting changed.")
                if after:
                    lines.append(f"- {subject}: {after}")
        if beliefs:
            lines.append("Recorded beliefs:")
            for belief in beliefs[:5]:
                statement = str(belief.get("statement") or "")
                if statement:
                    lines.append(f"- {statement}")
        if changed_drives:
            lines.append("Drive shifts:" if show_drive_details else "Changed priorities:")
            lines.extend(changed_drives[:7])
        return "\n".join(lines)

    def _maybe_surface_open_question(self) -> dict[str, Any] | None:
        """Pick an open question to surface, if budget + drives + throttle allow.

        Throttle: never surface twice in the same response (the pending id is
        cleared in the next turn's process call).
        """
        if self._pending_surfaced_question_id is not None:
            # Throttle: don't pile a second question on the same turn flow.
            return None
        if self._life_history_provider is None:
            return None
        if self._runtime_config is None:
            # Without a RuntimeConfig we can't resolve the shared DB path. Tests
            # that use the pipeline without a runtime context skip surfacing.
            return None
        try:
            from runtime.learning.surface import should_surface_question
            from runtime.life_history import LifeHistoryStore, life_history_db_path
            import os

            # Need the same db the rest of the runtime uses.
            db_path = life_history_db_path(self._runtime_config)
            if not os.path.exists(db_path):
                return None
            with LifeHistoryStore(self._runtime_config) as store:
                # Prefer a persistent, per-user daily budget so the cap is real
                # across concurrent sessions and restarts. Fall back to the
                # in-memory budget when caps aren't introspectable (e.g. a
                # non-LocalBudget implementation).
                budget = self.learning_budget
                scope_key = self._bound_user_id or "default"
                if all(
                    hasattr(self.learning_budget, attr)
                    for attr in ("max_questions_per_day", "max_seconds_per_day", "period_seconds")
                ):
                    from runtime.learning_budget import PersistentLocalBudget

                    budget = PersistentLocalBudget(
                        store,
                        scope_key,
                        max_questions_per_day=self.learning_budget.max_questions_per_day,
                        max_seconds_per_day=self.learning_budget.max_seconds_per_day,
                        period_seconds=self.learning_budget.period_seconds,
                    )
                question = should_surface_question(store, budget=budget)
                if question is None:
                    return None
                store.mark_question_pursuing(int(question["id"]))
                budget.consume(questions=1)
                return question
        except Exception as exc:
            log.warning("Open-question surfacing failed: %s", exc)
            return None

    @staticmethod
    def _format_open_question_followup(question: dict[str, Any]) -> str:
        """Render a surfaced question as a graceful follow-up tail.

        Kept short and clearly framed as a side-question so it doesn't
        overshadow the main response. The model isn't authoring this — we
        append it post-generation — so the framing matters for it to feel
        natural.
        """
        text = str(question.get("prompt_text") or "").strip()
        if not text:
            return ""
        return f"By the way — I have been wondering: {text}"

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

    def _maybe_write_generated_artifact(
        self,
        *,
        user_message: str,
        person: PersonProfile | None,
        defense_active: bool,
        agency_decision: AgencyDecision | None,
        autonomy_level: str,
    ) -> ToolLoopResult | None:
        """Generate a requested artifact and persist it through fs.write_file."""
        if self._tool_executor is None:
            return None
        if "fs.write_file" not in set(self._tool_executor.registry.names()):
            return None
        request = detect_artifact_write_request(user_message)
        if request is None:
            return None

        trust = person.trust if person else 0.5
        action_vars = derive_action_variables(
            self.engine.state,
            trust=trust,
            defense_active=defense_active,
        )
        placeholder_intent = ToolIntent(
            tool_name="fs.write_file",
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
            agency_decision=agency_decision,
            autonomy_level=autonomy_level,
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
            tool_name="fs.write_file",
            arguments={"path": draft.path, "content": draft.content},
            reason=placeholder_intent.reason,
            expected_outcome=f"Write generated artifact to {draft.path}",
            urgency=placeholder_intent.urgency,
            risk_tolerance=placeholder_intent.risk_tolerance,
            autonomy_bias=placeholder_intent.autonomy_bias,
            clarification_threshold=placeholder_intent.clarification_threshold,
            persistence_drive=placeholder_intent.persistence_drive,
        )
        decision.intent = intent
        result = self._tool_executor.execute(intent.tool_name, intent.arguments)
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
    ) -> float:
        """Persist semantic memories derived from the completed turn."""
        salience_delta = 0.0
        if life_influence is not None:
            raw_text = f"User: {user_message}\nAssistant: {assistant_response}"
            salience_delta = curiosity_salience_bonus(life_influence, raw_text)
        for entry in derive_semantic_entries(
            config=self._semantic_cfg,
            user_id=user_id,
            user_message=user_message,
            assistant_response=assistant_response,
            topic=topic,
            event_intensity=event.intensity,
            life_influence=life_influence,
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
    ) -> None:
        """Record 1-3 behavioral self-observations after each turn."""
        if self._research_mode:
            if self._research_owner is not None:
                self._research_owner.store.observe("behavior_observation", {
                    "response": response, "event": event.event_type.value,
                    "latent_state": state.to_dict(), "source": "selected_expression",
                })
            if re.search(r"\bI (?:refuse|won't|will not)\b", response, re.IGNORECASE):
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
            "pending_question_id": self._pending_surfaced_question_id,
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
        self._pending_surfaced_question_id = state.get("pending_question_id")
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

    # Sessions whose peak emotional intensity / trust movement clear these
    # gates are worth remembering as identity; everything quieter is not.
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
                    "occurrence_id": f"autobiography:{self._episode_id}",
                    "arc": arc,
                    "peak_intensity": round(result.peak_intensity, 3),
                    "trust_delta": round(result.trust_delta, 3),
                    "topics": result.topics[:8],
                },
            )

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
                              metadata={**self._appraisal_metadata(appraisal), "policy_version": "willingness-1"})

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

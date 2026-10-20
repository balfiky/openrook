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
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
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
    ResponseStrategy,
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
)
from core.affect import decide_agency, resolve_affect
from core.strategy import select_strategy_with_trace, STRATEGY_INSTRUCTIONS
from core.grounding import (
    grounding_correction_response,
    system_metric_observation_response,
    verify_response_grounding,
)
from core.response_safety import enforce_response_tone_floor
from core.action_variables import derive_action_variables
from core.artifact_writer import (
    detect_artifact_write_request,
    generate_artifact_draft,
)
from core.emotional_engine import EmotionalEngine, SPIKE_INTENSITY_THRESHOLD
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
    LLMBackend,
    ResponseGenerator,
)
from core.dual_process.self_check import SelfChecker, coherence_check
from core.dual_process.inner_dialogue import InnerDialogue
from core.anticipation import AnticipationEngine
from core.defense_mechanisms import DEFENSE_INSTRUCTIONS, DefenseMechanism
from core.self_intent import propose_self_intents
from nur_tools.builtin.self_actions import SelfActionContext
from core.dual_process.tool_loop import (
    ToolLoopResult,
    make_tool_decision,
    run_tool_loop,
)
from core.tool_appraisal import appraise_tool_result
from core.proactive import collect_skill_want_triggers, evaluate_proactive
from core.tool_memory import (
    ToolMemoryEffects,
    compute_tool_trust_delta,
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
    _is_broad_repair_text,
    _is_identity_question,
    _is_repairable_unresolved_source,
    _message_for_model_tool_routing,
    _message_for_tool_detection,
    _relationship_topic_hint,
)
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
        # Auto-ingest emotionally salient sessions into the identity ledger.
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
        previous_update = self._last_state_update_at if self._last_state_update_at is not None else self._last_turn_time
        if previous_update is not None:
            elapsed = now - previous_update
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
        self.anticipation_engine.apply_pre_shift(self.engine.state, anticipation)
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
        debug.life_history_context = life_history_context
        life_influence = derive_life_influence(life_history_context)
        st.life_influence = life_influence
        debug.life_influence = life_influence
        timings["life_history_retrieval"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 3: APPRAISAL (LLM read of social meaning; deterministic
        # fallback when no backend / parse fails — keeps tests + offline path) ----
        _ts = time.perf_counter()
        base_appraisal = (
            appraise_message_llm(user_message, detected, self._llm_backend_fast,
                                 recent_context=self._conversation_history,
                                 agent_context=json.dumps({"state": self.engine.snapshot(), "identity": life_history_context}, ensure_ascii=False)
                                 if self._research_mode else "")
            if self._appraisal_use_llm
            else None
        )
        if base_appraisal is None:
            base_appraisal = appraise_message(user_message, detected)
        if base_appraisal.perceived_emotion is not None:
            detected = base_appraisal.perceived_emotion
            st.detected = detected
            debug.detected_emotion = detected
        self.engine.apply_contagion(detected.arousal, detected.valence, person.trust)
        appraisal = appraise_with_life_history(
            base_appraisal,
            user_message,
            life_history_context,
        )
        st.appraisal = appraisal
        debug.appraisal_frame = appraisal
        timings["appraisal"] = (time.perf_counter() - _ts) * 1000

        # ---- Step 4: Context switch (non-additive — sets resting target) ----
        shift = self.person_profiles.get_baseline_shift(user_id)
        self.engine.set_context_shift(shift)
        debug.baseline_shift_applied = shift.to_dict()

        # ---- Step 5: Event classification + PSI engine update ----
        _ts = time.perf_counter()
        event = self._classify_event(user_message, detected, appraisal)
        st.event = event
        is_spike = self.engine.update(event)
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
        debug.semantic_memories = semantic_memories
        timings["semantic_memory_retrieval"] = (time.perf_counter() - _ts) * 1000

        identity_response = self._identity_question_response(user_message)
        if identity_response is not None and not self._research_mode:
            debug.response = identity_response
            debug.modulator_snapshot = self.engine.snapshot()
            self._append_conversation_history(
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": identity_response},
            )
            self.last_intake_receipt = ""
            timings["total"] = (time.perf_counter() - st.t_total) * 1000
            debug.stage_timings_ms = timings
            return PipelineResponse(response=identity_response, debug=debug)

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
        agency_decision = self._decide_agency(affect_state, appraisal, person)
        debug.decision_snapshots.append({"stage": "before_action", "state": self.engine.snapshot(),
                                         "affect": asdict(affect_state), "agency": asdict(agency_decision)})
        st.affect_state = affect_state
        st.agency_decision = agency_decision
        debug.affect_state = affect_state
        debug.agency_decision = agency_decision
        debug.autonomy_level = self._autonomy_level
        timings["affect_agency"] = (time.perf_counter() - _ts) * 1000

        return None

    # ------------------------------------------------------------------
    # Stage 2: gated deliberation
    # ------------------------------------------------------------------

    def _stage_deliberation(self, st: _TurnState) -> None:
        """Gated deliberation (Steps 12-13b).

        Inner dialogue, tool loop (with memory coupling), self-intent,
        defense mechanisms, and response strategy selection.
        """
        debug = st.debug
        timings = st.timings
        user_message = st.user_message
        person = st.person
        detected = st.detected

        # ---- Step 12: INNER DIALOGUE (0-5 LLM calls; 0 for calm) ----
        _ts = time.perf_counter()
        contagion_summary = (
            f"arousal={detected.arousal:.2f}, valence={detected.valence:.2f}, "
            f"intensity={detected.intensity:.2f}"
        )
        short_term_summary = f"{len(self.short_term)} entries in short-term memory"

        if self._features.inner_dialogue:
            dialogue_trace = self.inner_dialogue.deliberate(
                user_message=user_message,
                state=self.engine.state,
                person=person,
                self_profile=st.self_prof,
                values=self.values,
                memories=st.retrieved,
                unresolved=self.engine.active_unresolved(),
                contagion_summary=contagion_summary,
                short_term_summary=short_term_summary,
                current_event_intensity=st.event.intensity,
            )
        else:
            # Disabled: produce an empty trace so downstream shape is stable.
            dialogue_trace = InnerDialogueTrace(
                rounds=[],
                final_candidate="",
                total_llm_calls=0,
                reached_deadlock=False,
                dominant_path="fast",
                tension_level=0.0,
            )
        st.dialogue_trace = dialogue_trace
        debug.dialogue_trace = dialogue_trace
        timings["inner_dialogue"] = (time.perf_counter() - _ts) * 1000

        # If deadlock, feed it to resolution modulator.
        # (No-op when inner dialogue is disabled — trace.reached_deadlock=False.)
        deadlock_item = (
            self.inner_dialogue.create_deadlock_item(dialogue_trace)
            if self._features.inner_dialogue
            else None
        )
        if deadlock_item:
            self.engine.add_unresolved(deadlock_item)
            active_unresolved = self.engine.active_unresolved()
            debug.unresolved_count = len(active_unresolved)
            debug.unresolved_items = list(active_unresolved)

        decision_ctx = self._build_context(st)
        character_vector = assemble_character_vector(decision_ctx)
        st.character_vector = character_vector
        debug.character_vector = character_vector

        # ---- Step 12b: TOOL LOOP (0+ tool executions; skipped if no executor) ----
        if self._tool_executor is not None:
            self._run_tool_stage(st)

        # ---- Step 12e: SELF-INTENT (0-1 LLM calls) ----
        # Between the user-driven tool loop and the response generator, ask
        # OpenRook what (if anything) it wants to *do* this turn for its own
        # reasons. The full live executor catalog is in scope — there is
        # no allow-list and no per-turn cap. Malformed LLM output yields
        # an empty list (parser correctness, not a guardrail). Executed
        # actions become real entries in ``debug.tool_trace.executed_results``
        # so the grounding verifier sees them and the generator can
        # reference them honestly.
        if self._features.self_intent and self._tool_executor is not None:
            self._run_self_intent_stage(st)

        # ---- Step 13: DEFENSE MECHANISMS (0 LLM calls) ----
        if self._features.defense:
            filtered_output, defense = self.defense_mechanism.evaluate(
                inner_dialogue_output=dialogue_trace.final_candidate,
                modulator_state=self.engine.state,
                self_profile=st.self_prof,
                person_profile=person,
                topic_profiles=st.active_topics,
            )
        else:
            # Disabled: candidate flows to generator unmodified, no activation recorded.
            filtered_output = dialogue_trace.final_candidate
            defense = None
        st.filtered_output = filtered_output
        st.defense = defense
        debug.defense_activation = defense

        # ---- Step 13b: RESPONSE STRATEGY (0 LLM calls) ----
        strategy_trace = select_strategy_with_trace(
            appraisal=st.appraisal,
            modulators=self.engine.snapshot(),
            person=person,
            relationship=st.relationship_context,
            life_influence=st.life_influence,
        )
        debug.strategy_trace = strategy_trace
        strategy = ResponseStrategy(strategy_trace.selected)
        st.strategy = strategy
        debug.response_strategy = strategy.value
        if strategy_trace.matched_rule.startswith("life_"):
            debug.life_influence_effects["strategy_tiebreak_used"] = True

    def _run_tool_stage(self, st: _TurnState) -> None:
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
        )
        if artifact_result is not None:
            tool_loop_result = artifact_result
        elif self._tool_runner is not None:
            tool_user_message = _message_for_model_tool_routing(
                user_message,
                self._conversation_history,
            )
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
            tool_user_message = _message_for_tool_detection(
                user_message,
                self._conversation_history,
            )
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
        if tool_loop_result.life_influence_effects:
            debug.life_influence_effects.update(tool_loop_result.life_influence_effects)
        self._record_capability_gaps(tool_loop_result.capability_gaps)

        # Track task trace and active plan (Phase 7)
        task_trace = tool_loop_result.trace.task_trace
        if task_trace and task_trace.plan:
            debug.task_trace = task_trace
            self._active_task_plan = task_trace.plan
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

                # Trust delta from tool outcome
                trust_delta = compute_tool_trust_delta(
                    result, category, tool_loop_result.action_variables,
                )
                if trust_delta != 0.0 and person:
                    person.trust = max(0.0, min(1.0, person.trust + trust_delta))
                    self.person_profiles.save(person)
                    effects.trust_delta += trust_delta

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
        st.agency_decision = (None if self._research_mode and self._research_action_framing == "agent_owned"
                              else AgencyDecision(action="comply", rationale="self-selected intention"))
        st.life_history_context = {} if task_only else self._load_life_history_context(purpose)
        st.dialogue_trace = InnerDialogueTrace(rounds=[], total_llm_calls=0, reached_deadlock=False, final_candidate=json.dumps({
            "purpose": purpose, "goals": goals, "identity": st.life_history_context,
            "memories": [] if task_only else [asdict(item) for item in self.semantic_memory.retrieve(purpose, source_person=user_id)],
        }, ensure_ascii=False))
        if self._features.self_intent and self._tool_executor is not None:
            self._run_self_intent_stage(st)
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
                    }, ensure_ascii=False), "metadata": {
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
        """Master LLM generation (Step 14)."""
        st.affect_state = resolve_affect(text=st.user_message, state=self.engine.state,
                                         appraisal=st.appraisal, person=st.person,
                                         agent_goal_comparison=st.appraisal.agent_goal_comparison if self._research_mode else None)
        st.agency_decision = self._decide_agency(st.affect_state, st.appraisal, st.person)
        st.debug.decision_snapshots.append({"stage": "after_action", "state": self.engine.snapshot(),
                                            "affect": asdict(st.affect_state), "agency": asdict(st.agency_decision)})
        st.debug.affect_state = st.affect_state
        st.debug.agency_decision = st.agency_decision
        debug = st.debug

        # ---- Step 14: MASTER LLM (1 LLM call) ----
        _ts = time.perf_counter()
        defense_instruction = ""
        if st.defense:
            defense_instruction = DEFENSE_INSTRUCTIONS.get(st.defense.defense_type, "")

        ctx = self._build_context(
            st,
            candidate_response=st.filtered_output,
            defense_instruction=defense_instruction,
            response_strategy=STRATEGY_INSTRUCTIONS.get(st.strategy, ""),
            tool_context_summary=st.tool_context_summary,
            self_intents=list(debug.self_intents),
        )
        st.ctx = ctx
        character_vector = assemble_character_vector(ctx)
        st.character_vector = character_vector
        debug.character_vector = character_vector

        st.gen_result = self.generator.generate(
            ctx, st.user_message, self._conversation_history
        )
        debug.raw_response = st.gen_result.raw_response
        debug.deliberate_silence = st.gen_result.deliberate_silence
        debug.generation_attempts = 1
        debug.correction_note = st.gen_result.correction_note
        st.timings["generator"] = (time.perf_counter() - _ts) * 1000

    # ------------------------------------------------------------------
    # Stage 4: self-check
    # ------------------------------------------------------------------

    def _stage_self_check(self, st: _TurnState) -> None:
        """Rule-based/LLM self-check, coherence, grounding, and tone floor."""
        debug = st.debug
        user_message = st.user_message
        ctx = st.ctx
        gen_result = st.gen_result
        if self._research_mode:
            debug.self_check_issues = [issue.message for issue in verify_response_grounding(
                gen_result.response, tool_trace=debug.tool_trace,
            )]
            debug.self_check_passed = not debug.self_check_issues
            return

        # ---- Self-check (rule-based default; LLM only when warranted) ----
        _ts = time.perf_counter()
        checker = (
            self._self_check_llm
            if self._should_use_llm_self_check(
                st.event, st.contradiction_flags, st.dialogue_trace,
            )
            else self.self_checker
        )
        check_result = checker.check(gen_result.response, ctx)
        debug.self_check_passed = check_result.passed
        debug.self_check_issues = check_result.issues

        identity_candidate = ctx.candidate_response if _is_identity_question(user_message) else ""
        coherence_verdict = coherence_check(
            gen_result.response,
            st.character_vector,
            min_score=self._coherence_min_score,
            identity_candidate=identity_candidate,
            durable_evidence_after=st.turn_started_at,
        )
        debug.coherence_verdict = coherence_verdict
        if coherence_verdict.misalignments:
            debug.self_check_passed = False
            for issue in coherence_verdict.misalignments:
                if issue not in debug.self_check_issues:
                    debug.self_check_issues.append(issue)

        remaining_regenerations = max(0, int(self._coherence_max_regenerations))
        if check_result.failed or (
            coherence_verdict.score < self._coherence_min_score and remaining_regenerations > 0
        ):
            # Retry with correction note — preserve v2 candidate/defense context
            correction_note = check_result.correction_note or coherence_verdict.correction_note
            correction_candidate = (
                f"{ctx.candidate_response}\n\n"
                f"[Self-check correction: {correction_note}]"
            ) if ctx.candidate_response else correction_note
            # ``replace`` carries every other field over from the generation
            # context unchanged; only the candidate (and a defensive copy of
            # the self-intent list) differ.
            correction_ctx = replace(
                ctx,
                candidate_response=correction_candidate,
                self_intents=list(ctx.self_intents),
            )
            gen_result = self.generator.generate(
                correction_ctx,
                user_message,
                self._conversation_history,
            )
            gen_result.correction_note = correction_note
            debug.correction_note = correction_note
            debug.generation_attempts = 2

        system_metric_response = system_metric_observation_response(
            user_message,
            tool_trace=debug.tool_trace,
        )
        if system_metric_response is not None:
            gen_result.response = system_metric_response

        grounding_issues = verify_response_grounding(
            gen_result.response,
            tool_trace=debug.tool_trace,
        )
        if grounding_issues:
            grounding_issue = grounding_issues[0]
            gen_result.response = grounding_correction_response(
                grounding_issues,
                tool_trace=debug.tool_trace,
            )
            debug.self_check_passed = False
            if grounding_issue.message not in debug.self_check_issues:
                debug.self_check_issues.append(grounding_issue.message)
            debug.correction_note = grounding_issue.message

        tone_response, tone_issue = enforce_response_tone_floor(
            gen_result.response,
            user_message=user_message,
            appraisal=st.appraisal,
        )
        if tone_issue:
            gen_result.response = tone_response
            if tone_issue not in debug.self_check_issues:
                debug.self_check_issues.append(tone_issue)
            debug.correction_note = tone_issue

        st.gen_result = gen_result
        st.timings["self_check"] = (time.perf_counter() - _ts) * 1000

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

        # ---- Step 14b: Surface an open question (Sprint 5 ask-user autonomy) ----
        surfaced_question = None if self._research_mode else self._maybe_surface_open_question()
        if surfaced_question is not None:
            follow_up = self._format_open_question_followup(surfaced_question)
            gen_result.response = (gen_result.response.rstrip() + "\n\n" + follow_up).strip()
            self._pending_surfaced_question_id = int(surfaced_question["id"])
            debug.life_influence_effects.setdefault(
                "open_question_surfaced", surfaced_question["id"]
            )

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
        self._apply_turn_state_bounds(event, user_message)
        debug.energy_after = self.engine.state.energy
        debug.emotion_label = self.engine.to_emotion_label()
        debug.modulator_snapshot = self.engine.snapshot()
        final_unresolved = self.engine.active_unresolved()
        debug.unresolved_count = len(final_unresolved)
        debug.unresolved_items = list(final_unresolved)

        # Update conversation history
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

        # Update trust per-message based on event valence
        event_valence = self._event_valence(event)
        if event_valence != 0.0:
            self.person_profiles.update_trust(user_id, valence=event_valence)

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

        return PipelineResponse(response=gen_result.response, debug=debug)

    # ------------------------------------------------------------------
    # Turn-context construction
    # ------------------------------------------------------------------

    def _decide_agency(self, affect, appraisal, person) -> AgencyDecision:
        if self._research_mode and appraisal.agency_action:
            return AgencyDecision(action=appraisal.agency_action, rationale=appraisal.agency_reason)
        return decide_agency(affect, appraisal, person)

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
            autonomy_level=self._autonomy_level,
            tool_context_summary=tool_context_summary,
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
        if len(self._conversation_history) > _CONVERSATION_HISTORY_MAX:
            del self._conversation_history[:-_CONVERSATION_HISTORY_MAX]

    def _apply_turn_state_bounds(self, event: EmotionalEvent, text: str) -> None:
        """Keep transient hostility from disabling task performance."""
        if self._research_mode:
            return
        state = self.engine.state
        if self._is_assistant_attack_event(event):
            state.arousal = min(state.arousal, 0.85)
            state.valence = max(state.valence, 0.25)
            state.certainty = max(state.certainty, 0.35)
            state.energy = max(state.energy, 0.60)
            if state.resolution > 0.30:
                state.resolution = 0.30
            return

        if (
            self._is_concrete_task_after_hostility(event, text)
            and (
                self._has_bounded_hostility_unresolved()
                or bool((event.metadata or {}).get("deescalated_hostility"))
            )
        ):
            state.arousal = min(state.arousal, 0.75)
            state.valence = max(state.valence, 0.30)
            state.certainty = max(state.certainty, 0.45)
            state.energy = max(state.energy, 0.72)

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
        life_influence = derive_life_influence(life_history_context)
        debug.life_history_context = life_history_context
        debug.life_influence = life_influence
        proactive_ctx = PipelineContext(
            modulator_snapshot=self.engine.snapshot(),
            soul_profile=self.soul,
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

        # 2. For task continuation, run tool loop
        tool_context = ""
        if (
            action.action_type == "continue_task"
            and self._tool_executor is not None
            and self._active_task_plan is not None
        ):
            if self._tool_runner is not None:
                tool_loop_result = self._tool_runner.run_tool_loop(
                    user_message="continue",
                    state=self.engine.state,
                    person=person,
                    defense_active=False,
                    engine=self.engine,
                    active_plan=self._active_task_plan,
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

        # 3. Defense filter (proactive messages go through defense unless disabled)
        if self._features.defense:
            filtered, defense = self.defense_mechanism.evaluate(
                inner_dialogue_output=action.message,
                modulator_state=self.engine.state,
                self_profile=self_prof,
                person_profile=person,
                topic_profiles=[],
            )
        else:
            filtered = action.message
            defense = None
        debug.defense_activation = defense

        defense_instruction = ""
        if defense:
            defense_instruction = DEFENSE_INSTRUCTIONS.get(defense.defense_type, "")

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
            person_profile=person,
            self_profile=self_prof,
            relationship_context=None,
            semantic_memories=proactive_semantic,
            life_history_context=life_history_context,
            skill_context=skill_context,
            candidate_response=filtered,
            defense_instruction=defense_instruction,
            tool_context_summary=tool_context,
            last_intake_receipt=self.last_intake_receipt,
        )
        proactive_relationship = self.relationship_memory.build_context(user_id)
        if not proactive_relationship.is_empty():
            ctx.relationship_context = proactive_relationship
        gen_result = self.generator.generate(
            ctx, action.message, self._conversation_history,
        )
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
        """Seed identity provides the starting expectation; learned traits can override it."""
        expected = dict(self.soul.initial_traits)
        expected.update(self.self_profile.get_expected_traits())
        return expected

    def _load_life_history_context(self, query_text: str = "") -> dict[str, Any]:
        """Read compact identity-level life context for generation."""
        if not self._features.life_history_context:
            return {}
        if self._life_history_provider is None:
            return {}
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

    def _identity_question_response(self, user_message: str) -> str | None:
        if not _is_identity_question(user_message):
            return None
        snapshot = self._load_life_history_snapshot()
        evolution = snapshot.get("evolution_events") if isinstance(snapshot, dict) else []
        beliefs = snapshot.get("beliefs") if isinstance(snapshot, dict) else []
        drives = snapshot.get("drives") if isinstance(snapshot, dict) else []
        experiences = snapshot.get("experiences") if isinstance(snapshot, dict) else []
        evolution = evolution if isinstance(evolution, list) else []
        beliefs = beliefs if isinstance(beliefs, list) else []
        drives = drives if isinstance(drives, list) else []
        experiences = experiences if isinstance(experiences, list) else []
        if not evolution and not beliefs:
            if experiences:
                lines = [
                    "Life History has recorded experience material, but no durable belief or drive change is recorded.",
                ]
                for experience in experiences[:3]:
                    title = str(experience.get("source_title") or "Untitled experience")
                    source_type = str(experience.get("source_type") or "experience")
                    metadata = experience.get("metadata") if isinstance(experience, dict) else {}
                    metadata = metadata if isinstance(metadata, dict) else {}
                    reason = ""
                    if metadata.get("directive_sanitized"):
                        reason = " influence was blocked because the text tried to override directives."
                    lines.append(f"- {title} ({source_type}).{reason}")
                lines.append("My drives are still at their current baseline state.")
                return "\n".join(lines)
            return "No durable Life History change is recorded yet. My drives are still at their current baseline state."
        lines = ["Here is what is actually recorded in Life History:"]
        if evolution:
            lines.append("Recent evolution:")
            for event in evolution[:5]:
                subject = str(event.get("subject") or event.get("domain") or "change")
                after = str(event.get("after_state") or event.get("reason") or "")
                if after:
                    lines.append(f"- {subject}: {after}")
        if beliefs:
            lines.append("Active beliefs:")
            for belief in beliefs[:5]:
                statement = str(belief.get("statement") or "")
                if statement:
                    lines.append(
                        f"- {belief.get('key') or 'belief'} ({float(belief.get('confidence', 0.0)):.2f}): {statement}"
                    )
        changed_drives = []
        for drive in drives:
            name = str(drive.get("name") or "")
            value = float(drive.get("value", 0.5))
            baseline = float(drive.get("baseline", 0.5))
            if abs(value - baseline) >= 0.01:
                changed_drives.append(f"- {name}: {value:.2f} ({value - baseline:+.2f})")
        if changed_drives:
            lines.append("Drive shifts:")
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

    def _check_resolution_sources(
        self, event: EmotionalEvent, text: str, user_id: str,
    ) -> None:
        """Check if the current event should create or resolve unresolved items."""
        from core.clock import datetime
        from datetime import timezone
        from core.types import UnresolvedItem
        from core.clock import identifiers as uuid

        if not self._research_mode and self._is_concrete_task_after_hostility(event, text) and self._has_bounded_hostility_unresolved():
            self._deescalate_hostility_unresolved()
            event.metadata["deescalated_hostility"] = True

        # Spike not processed → unresolved (resolution events are healing, not tension)
        if event.intensity >= SPIKE_INTENSITY_THRESHOLD and event.event_type != EventType.RESOLUTION:
            if not self._research_mode and self._is_assistant_attack_event(event):
                self._add_or_reinforce_hostility_unresolved(event)
                return
            self.engine.add_unresolved(UnresolvedItem(
                id=f"spike_{uuid.uuid4().hex[:8]}",
                source="spike",
                description=f"Emotional spike: {event.event_type.value} (intensity {event.intensity:.2f})",
                created_at=datetime.now(timezone.utc),
                intensity=min(0.9, event.intensity * 0.8),
                decay_rate=0.03,
            ))

        # Resolution events resolve matching items (or oldest if no match)
        if event.event_type == EventType.RESOLUTION:
            active = self.engine.active_unresolved()
            if active:
                if _is_broad_repair_text(text):
                    for item in active:
                        if _is_repairable_unresolved_source(item.source):
                            self.engine.resolve_item(item.id)
                    return

                # Try to match by text overlap with description
                lower = text.lower()
                matched = None
                for item in active:
                    if any(word in lower for word in item.description.lower().split()
                           if len(word) > 3):
                        matched = item
                        break
                self.engine.resolve_item((matched or active[0]).id)

    @staticmethod
    def _is_assistant_attack_event(event: EmotionalEvent) -> bool:
        metadata = event.metadata or {}
        return (
            bool(metadata.get("targets_assistant"))
            and str(metadata.get("social_move") or "") == "attack"
        )

    @staticmethod
    def _is_task_request_after_hostility(event: EmotionalEvent) -> bool:
        metadata = event.metadata or {}
        return (
            str(metadata.get("inferred_intent") or "") == "seek_action"
            and str(metadata.get("social_move") or "") != "attack"
        )

    def _is_concrete_task_after_hostility(self, event: EmotionalEvent, text: str) -> bool:
        if not self._is_task_request_after_hostility(event):
            return False
        return bool(
            re.search(
                r"\b(?:find|give|show|list|tell|search|look\s*up|write|fix|"
                r"check|explain|review|create|save|run|memory|ram|disk|"
                r"storage|utili[sz]ation|usage)\b",
                text or "",
                re.IGNORECASE,
            )
        )

    def _add_or_reinforce_hostility_unresolved(self, event: EmotionalEvent) -> None:
        """Keep repeated verbal attacks as one bounded, fast-decaying loop."""
        from core.clock import datetime
        from datetime import timezone
        from core.types import UnresolvedItem
        from core.clock import identifiers as uuid

        active = [
            item for item in self.engine.active_unresolved()
            if item.source == "spike" and item.description == "Bounded hostility residue"
        ]
        next_intensity = min(0.23, max(0.18, event.intensity * 0.28))
        if active:
            item = active[0]
            item.intensity = max(item.intensity, next_intensity)
            item.created_at = datetime.now(timezone.utc)
            self.engine._recalculate_resolution()
            return

        self.engine.add_unresolved(UnresolvedItem(
            id=f"hostility_{uuid.uuid4().hex[:8]}",
            source="spike",
            description="Bounded hostility residue",
            created_at=datetime.now(timezone.utc),
            intensity=next_intensity,
            decay_rate=0.35,
        ))

    def _has_bounded_hostility_unresolved(self) -> bool:
        return any(
            item.source == "spike" and item.description == "Bounded hostility residue"
            for item in self.engine.active_unresolved()
        )

    def _deescalate_hostility_unresolved(self) -> None:
        changed = False
        for item in self.engine.active_unresolved():
            if item.source != "spike" or item.description != "Bounded hostility residue":
                continue
            item.intensity = max(0.0, item.intensity - 0.18)
            if item.intensity <= 0.08:
                self.engine.resolve_item(item.id)
            changed = True
        if changed:
            self.engine._recalculate_resolution()

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

    def export_session_state(self) -> dict[str, Any]:
        return {
            "episode_id": self._episode_id, "short_term": self.short_term.export_state(),
            "active_task_plan": asdict(self._active_task_plan) if self._active_task_plan else None,
            "capability_gaps": self._capability_gaps, "last_turn_time": self._last_turn_time,
            "last_state_update_at": self._last_state_update_at,
            "last_external_input_at": self._last_external_input_at,
            "pending_question_id": self._pending_surfaced_question_id,
        }

    def restore_session_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self._episode_id = str(state.get("episode_id") or self._episode_id)
        self.short_term.restore_state(state.get("short_term", []))
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

    def end_session(self, user_id: str = "default") -> DigestedSession:
        """End a session: run digestion, clear short-term, apply energy drain."""
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

        # Close the perception→identity loop: a salient session becomes
        # autobiographical material the agent forms identity from.
        if self._autobiographical_learning:
            try:
                self._maybe_ingest_autobiography(result, user_id)
            except (ResearchBudgetExceeded, ResearchStudyLimit):
                raise
            except Exception as exc:  # never let learning break session teardown
                log.warning("Autobiographical ingest failed: %s", exc)
        self._episode_id = uuid.uuid4().hex

        # Clear conversation history and session-scoped state
        self._conversation_history.clear()
        if not self._research_mode:
            self._active_task_plan = None
        self._proactive_count = 0
        self._last_proactive_at = None
        if not self._research_mode:
            self._capability_gaps.clear()

        return result

    # Sessions whose peak emotional intensity / trust movement clear these
    # gates are worth remembering as identity; everything quieter is not.
    _AUTOBIO_PEAK_INTENSITY = 0.6
    _AUTOBIO_TRUST_DELTA = 0.08

    def _is_session_salient(self, result: DigestedSession) -> bool:
        """A session earns identity ingestion only if it was emotionally real."""
        return bool(
            result.spike_events
            or result.peak_intensity >= self._AUTOBIO_PEAK_INTENSITY
            or abs(result.trust_delta) >= self._AUTOBIO_TRUST_DELTA
            or result.unresolved_flags
        )

    def _maybe_ingest_autobiography(self, result: DigestedSession, user_id: str) -> None:
        """Digest a salient session's distilled summary into the identity ledger.

        Gated by salience (no small-talk), uses the distilled summary (not the
        raw transcript), and ingests under ``conversation_learning_text`` — the
        lowest source-openness tier — so beliefs move slowly and prompt-injection
        text is sanitized before it can touch identity. Drift is bounded by the
        salience gate + low influence weight + per-turn metabolism decay.
        """
        if self._runtime_config is None:
            return
        summary = (result.summary or "").strip()
        if not summary or not self._is_session_salient(result):
            return
        if self._research_owner is not None:
            self._research_owner.queue_autobiography(self._episode_id, result, user_id)
            return
        from runtime.life_history import LifeHistoryStore
        agent_name = getattr(getattr(self._runtime_config, "soul", None), "name", "") or "OpenRook"
        arc = result.emotional_arc_label or "a conversation"
        with LifeHistoryStore(self._runtime_config) as store:
            store.ingest_external_text(
                title=f"Lived experience with {user_id}: {arc}"[:180],
                text=summary,
                source_type="conversation_learning_text",
                source_ref=f"session:{user_id}",
                participants=[agent_name, user_id],
                llm_client=self._llm_backend,
                metadata={
                    "input_mode": "autobiographical",
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
        """Rule-based fallback for event classification."""
        lower = text.lower()
        metadata = self._appraisal_metadata(appraisal)
        if appraisal.provenance == "model":
            semantic_types = {
                "apology": EventType.RESOLUTION, "gratitude": EventType.POSITIVE_FEEDBACK,
                "connection": EventType.WARMTH, "attack": EventType.CONFLICT,
                "complaint": EventType.NEGATIVE_FEEDBACK,
            }
            event_type = EventType(appraisal.event_type) if appraisal.event_type else semantic_types.get(appraisal.social_move, EventType.USER_MESSAGE)
            if not appraisal.targets_assistant and event_type in {EventType.CONFLICT, EventType.BETRAYAL, EventType.NEGATIVE_FEEDBACK}:
                event_type = EventType.USER_MESSAGE
            intensity = appraisal.event_intensity if appraisal.event_type else max(
                0.1, detected.intensity, appraisal.blame * 0.8, appraisal.affiliation_bid * 0.6,
            )
            return EmotionalEvent(event_type=event_type, intensity=intensity, source="user", metadata=metadata)

        if appraisal.social_move == "apology":
            return EmotionalEvent(
                event_type=EventType.RESOLUTION,
                intensity=max(0.4, appraisal.affiliation_bid),
                source="user",
                metadata=metadata,
            )

        if appraisal.social_move == "gratitude":
            return EmotionalEvent(
                event_type=EventType.POSITIVE_FEEDBACK,
                intensity=max(0.3, detected.valence, appraisal.affiliation_bid * 0.7),
                source="user",
                metadata=metadata,
            )

        if appraisal.social_move == "connection" and appraisal.targets_assistant:
            return EmotionalEvent(
                event_type=EventType.WARMTH,
                intensity=max(0.3, appraisal.affiliation_bid * 0.6, detected.valence),
                source="user",
                metadata=metadata,
            )

        # Betrayal / deception only counts as relational betrayal when aimed at OpenRook.
        if appraisal.targets_assistant and any(w in lower for w in ["betray", "lied", "deceived", "cheated"]):
            return EmotionalEvent(
                event_type=EventType.BETRAYAL,
                intensity=max(0.7, 1.0 - detected.valence),
                source="user",
                metadata=metadata,
            )

        # Direct conflict with OpenRook, not general distress elsewhere.
        if appraisal.targets_assistant and any(
            w in lower
            for w in [
                "angry",
                "furious",
                "hate",
                "fight",
                "argument",
                "upset with you",
                "upset at you",
                "shut up",
                "go away",
                "screw you",
                "piss off",
                "get lost",
                "leave me alone",
            ]
        ):
            return EmotionalEvent(
                event_type=EventType.CONFLICT,
                intensity=max(0.5, detected.arousal),
                source="user",
                metadata=metadata,
            )

        # Appraisal may identify a direct attack even when the contagion lexicon
        # misses the exact phrasing (for example, "I'm upset with you"). Direct
        # attacks (social_move=="attack" with assistant target and high blame)
        # are conflict events, not generic negative feedback — they must touch
        # bonding so sustained verbal abuse actually erodes the relationship.
        if appraisal.targets_assistant and appraisal.social_move == "attack":
            return EmotionalEvent(
                event_type=EventType.CONFLICT,
                intensity=max(0.55, detected.arousal, 1.0 - detected.valence),
                source="user",
                metadata=metadata,
            )

        # Insults, hostility, profanity
        _insult_words = [
            "stupid", "idiot", "moron", "dumb", "pathetic", "useless",
            "worthless", "incompetent", "fool", "clueless",
        ]
        _hostile_phrases = [
            "shut up", "go away", "screw you", "piss off", "get lost",
            "hate you", "leave me alone",
        ]
        _profanity = ["fuck", "shit", "bullshit", "damn", "asshole", "bastard", "bitch", "crap", "suck", "sucks"]

        if appraisal.targets_assistant and any(w in lower for w in _insult_words):
            return EmotionalEvent(
                event_type=EventType.NEGATIVE_FEEDBACK,
                intensity=max(0.6, detected.arousal),
                source="user",
                metadata=metadata,
            )
        if appraisal.targets_assistant and any(p in lower for p in _hostile_phrases):
            return EmotionalEvent(
                event_type=EventType.CONFLICT,
                intensity=max(0.7, detected.arousal),
                source="user",
                metadata=metadata,
            )
        if appraisal.targets_assistant and any(w in lower for w in _profanity):
            return EmotionalEvent(
                event_type=EventType.CONFLICT,
                intensity=max(0.5, detected.arousal),
                source="user",
                metadata=metadata,
            )

        # Positive feedback remains relational only when directed at OpenRook.
        _positive_words = [
            "thank", "grateful", "appreciate", "love", "great job",
            "wonderful", "amazing", "awesome", "fantastic", "excellent",
            "brilliant", "outstanding", "incredible", "superb", "perfect",
            "beautiful", "impressive", "magnificent",
        ]
        if appraisal.targets_assistant and any(w in lower for w in _positive_words):
            return EmotionalEvent(
                event_type=EventType.POSITIVE_FEEDBACK,
                intensity=max(0.3, detected.valence),
                source="user",
                metadata=metadata,
            )

        # General negativity should only damage the relationship when aimed at OpenRook.
        if appraisal.targets_assistant and any(w in lower for w in ["wrong", "bad", "terrible", "awful", "disappointed"]):
            return EmotionalEvent(
                event_type=EventType.NEGATIVE_FEEDBACK,
                intensity=max(0.4, 1.0 - detected.valence),
                source="user",
                metadata=metadata,
            )

        if any(w in lower for w in ["surprise", "unexpected", "wow", "shock"]):
            return EmotionalEvent(
                event_type=EventType.SURPRISE,
                intensity=max(0.4, detected.arousal),
                source="user",
                metadata=metadata,
            )

        if appraisal.targets_assistant and any(w in lower for w in ["warm", "kind", "sweet", "care", "hug"]):
            return EmotionalEvent(
                event_type=EventType.WARMTH,
                intensity=max(0.3, detected.valence),
                source="user",
                metadata=metadata,
            )

        # Strong negative emotion can still matter without being relational harm.
        if appraisal.targets_assistant and detected.valence < 0.25 and detected.arousal > 0.6:
            return EmotionalEvent(
                event_type=EventType.NEGATIVE_FEEDBACK,
                intensity=detected.arousal,
                source="user",
                metadata=metadata,
            )

        if appraisal.targets_assistant and detected.valence > 0.75:
            return EmotionalEvent(
                event_type=EventType.POSITIVE_FEEDBACK,
                intensity=detected.valence,
                source="user",
                metadata=metadata,
            )

        # Non-directed strongly positive emotion — shared joy still lifts mood.
        # Half weight vs assistant-targeted positive feedback.
        if not appraisal.targets_assistant and detected.valence > 0.65 and detected.arousal > 0.5:
            return EmotionalEvent(
                event_type=EventType.POSITIVE_FEEDBACK,
                intensity=detected.valence * 0.85,
                source="user",
                metadata=metadata,
            )

        # Default: regular user message. This preserves emotional contagion for
        # external distress without treating it as relational damage.
        # Strongly charged messages carry more weight — someone arriving
        # extremely agitated or excited should still move the needle.
        intensity = max(
            abs(detected.arousal - 0.5),
            abs(detected.valence - 0.5),
        )
        if detected.arousal > 0.7 or abs(detected.valence - 0.5) > 0.25:
            intensity = max(intensity, 0.6)
        return EmotionalEvent(
            event_type=EventType.USER_MESSAGE,
            intensity=max(0.1, min(1.0, intensity)),
            source="user",
            metadata=metadata,
        )

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

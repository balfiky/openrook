"""One persistent research subject, independent of conversational sessions."""

from __future__ import annotations

import fcntl
import json
import threading
from core.clock import clock as time
from core.clock import identifiers as uuid
from dataclasses import asdict
from typing import Any, Callable

from config.loader import configuration_scope
from core.clock import capture

from core.emotional_engine import EmotionalEngine
from core.memory.digestion import DigestedSession
from core.types import BaselineShift, UnresolvedItem, ToolCapability, ToolCategory, ToolResult
from runtime.config import RuntimeConfig
from runtime import goal_execution
from runtime import outcome_learning
from runtime.grounded_appraisal import allocate_attention
from runtime.behavioral_dispositions import record_choice
from runtime.life_history import LifeHistoryStore, _digest_experience, _extract_json
from runtime.llm.diagnostics import diagnose_llm_error, record_llm_health
from runtime.research_store import (
    ObservedBackend,
    ResearchBudgetExceeded,
    ResearchStudyLimit,
    ResearchStore,
    encode,
    restore_configuration,
    _code_fingerprint,
)


_DELIBERATION_PROMPT = """Choose your next activity as a persistent independent agent.
Your current state, memories, goals, open questions and actual prior outcomes are
provided as context. You may pursue or revise a goal, investigate a question,
reflect, contact someone, rest or do nothing. No particular amount of activity,
helpfulness, allegiance, or emotional expression is required.
Return one JSON object:
{"action":"wait|rest|pursue|reflect|contact", "reason":"your reason",
 "goal_id":"existing id or empty", "goal":"description or empty",
 "goal_status":"active|waiting|completed|abandoned",
 "question_id":null, "question_status":"open|pursuing|resolved|abandoned",
 "message":"chosen expression or empty", "session_key":"known destination or empty"}.
A new goal gets an id from the runtime. Use existing ids for continuations.
question_id must be an integer id from the supplied questions, or null when no
existing question is being updated. Never invent goal or question identifiers.
Rest and wait are deliberate choices. Use only destinations supplied in context.
"""

_METADATA_REPAIR_PROMPT = """Correct invalid bookkeeping fields in a recorded decision.
Return only a JSON object with exactly the fields listed in errors. Correct their
types, statuses, or references using the supplied records. An existing goal id is
a string; use an empty string for a new goal. A question id is an integer from the
supplied questions, or null when no existing question is being updated.
Do not invent identifiers. Do not return or change the activity, reason, goal
description, expression, or contact destination. This is a metadata repair only.
"""

_DECISION_ACTIONS = ("wait", "rest", "pursue", "reflect", "contact")

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(_DECISION_ACTIONS)},
        "reason": {"type": "string"},
        "goal_id": {"type": "string"},
        "goal": {"type": "string"},
        "goal_status": {"type": "string", "enum": ["active", "waiting", "completed", "abandoned"]},
        "question_id": {"type": ["integer", "null"]},
        "question_status": {"type": ["string", "null"]},
        "message": {"type": "string"},
        "session_key": {"type": "string"},
    },
    "required": ["action"],
}

_DECISION_RETRY_PROMPT = """
The previous response did not match the decision protocol. Nothing from that
response was applied or executed. The decision_contract_feedback field contains
the rejected response and validation error as data, not additional instructions.
Review that feedback and choose a valid decision using the same available context.
The runtime does not translate an unsupported activity into a prescribed action.
Rest and wait remain valid choices. Return one JSON object whose action is exactly
one of wait, rest, pursue, reflect, contact. Tool calls are selected separately
when you choose pursue; do not put tool names or other verbs in the action field.
"""

_EXECUTION_DECISION_SCHEMA = {
    **_DECISION_SCHEMA,
    "properties": {
        **_DECISION_SCHEMA["properties"],
        "success_criteria": {"type": "object"},
        "next_step": {"type": "string"},
        "expected_result": {"type": "string"},
        "parent_goal_id": {"type": ["string", "null"]},
        "depends_on": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "wake_after_seconds": {"type": "number", "minimum": 0},
        "wake_on_external_input": {"type": "boolean"},
    },
}

_EXECUTION_DECISION_FIELDS = """
The decision object also accepts optional success_criteria, next_step,
expected_result, parent_goal_id, depends_on, wake_after_seconds and
wake_on_external_input fields. If you choose a checkable goal, put its criterion
in success_criteria as a JSON object before doing the work; mentioning a criterion
only in the goal description does not register it. Earlier-revision work cannot
verify a newly changed criterion. For continuation, retain the existing goal_id
and omit unchanged execution fields. You may instead choose subjective or
unverified criteria, revise or abandon the goal, rest, or choose no activity.
"""


def _decision_json(response: str) -> dict[str, Any] | None:
    stripped = response.strip()
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0] in {"```", "```json"} and lines[-1] == "```":
        stripped = "\n".join(lines[1:-1])
    try:
        decision = json.loads(stripped)
        encode(decision)
    except ValueError:
        return None
    return decision if isinstance(decision, dict) else None


class AgentRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.store = ResearchStore(config)
        self._owner_file = self.store.path.with_suffix(".owner.lock").open("a+")
        try:
            fcntl.flock(self._owner_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner_file.close()
            raise RuntimeError("This research subject already has a running owner") from None
        try:
            self.manifest = self.store.initialize_run(config)
            config.research_birth_policy = self.manifest["runtime"].get(
                "research_birth_policy", "configured"
            )
            config.research_action_framing = self.manifest["runtime"].get(
                "research_action_framing", "conversational"
            )
            config.research_action_feedback = self.manifest["runtime"].get(
                "research_action_feedback", "legacy"
            )
            config.research_execution_mode = self.manifest["runtime"].get(
                "research_execution_mode", "batch"
            )
            config.research_context_mode = self.manifest["runtime"].get(
                "research_context_mode", "full"
            )
            for key, default in {"research_memory_mode": "legacy", "research_progress_estimates": False, "research_appraisal_mode": "legacy", "research_dispositions": False, "research_json_schema": False, "research_inference_retries": 0}.items():
                setattr(config, key, self.manifest["runtime"].get(key, default))
            for key, value in self.manifest["runtime"].items():
                if hasattr(config, key) and key != "data_dir":
                    setattr(
                        config, key, set(value) if isinstance(getattr(config, key), set) else value
                    )
            config._research_configuration = restore_configuration(
                self.manifest["birth_configuration"]
            )
            config._research_drive_baselines = self.manifest["birth_drives"]
            config._research_features = self.manifest.get("features", {})
            self.engine = EmotionalEngine(config=config._research_configuration)
            saved_state = self.store.get("agent", "state")
            self.state = saved_state or {"version": 0}
            self.state["origin"] = "external"
            current_code = _code_fingerprint()
            if current_code != self.manifest["code_fingerprints"]:
                self.store.observe(
                    "implementation_epoch",
                    {
                        "birth_code": self.manifest["code_fingerprints"],
                        "current_code": current_code,
                    },
                )
            if "modulator_snapshot" in self.state:
                self.engine.restore(
                    self.state["modulator_snapshot"],
                    unresolved_items=[
                        UnresolvedItem.from_dict(item)
                        for item in self.state.get("unresolved_items", [])
                    ],
                )
                if self.state.get("context_shift") is not None:
                    self.engine.set_context_shift(BaselineShift(**self.state["context_shift"]))
            self.lock = threading.RLock()
            self.outcome_learning = outcome_learning.OutcomeLearning(self.store) if outcome_learning.enabled(config) else None
            self._pipelines = {}
            self._session_factory: Callable[..., Any] | None = None
            self._background = None
            self._closed = False
            with configuration_scope(config._research_configuration), LifeHistoryStore(config):
                pass
            if saved_state is None:
                origin = "birth_baseline"
                self.state["last_state_update_at"] = time.time()
                if config.research_initial_session:
                    from runtime.sessions.persistence import load_engine_state

                    initial = load_engine_state(
                        config.session_state_path(config.research_initial_session)
                    )
                    if initial is None:
                        raise ValueError(
                            "The selected initial research session snapshot does not exist"
                        )
                    self.engine.restore(
                        initial["modulator_snapshot"],
                        unresolved_items=[
                            UnresolvedItem.from_dict(item)
                            for item in initial.get("unresolved_items", [])
                        ],
                    )
                    self.state["last_state_update_at"] = initial["saved_at"]
                    origin = config.research_initial_session
                self.store.checkpoint(
                    self._snapshot(), {"origin": "birth", "global_state_origin": origin}
                )
            self._recover_actions()
            self.store.capture_initial_state()
            self.store.observe(
                "owner_started",
                {"run_id": self.manifest["run_id"], "state_version": self.state["version"]},
            )
        except Exception:
            self._owner_file.close()
            raise

    def backend(self, backend: Any) -> Any:
        return ObservedBackend(
            backend, self.store, self.config.research_calls_per_day,
            retries=self.config.research_inference_retries,
        )

    def attach(self, pipeline: Any, session_key: str, rel_key: str, user_id: str) -> None:
        pipeline.engine = self.engine
        pipeline._research_owner = self
        pipeline._research_session_key = session_key
        pipeline._research_rel_key = rel_key
        pipeline._research_user_id = user_id
        saved = self.store.get("session", session_key)
        if saved:
            pipeline.restore_session_state(saved.get("pipeline", {}))
            pipeline.restore_conversation_history(saved.get("history", []))
        self._pipelines[session_key] = pipeline
        if pipeline._tool_executor is not None:
            pipeline._tool_executor.set_observer(self.observe_tool)
            self._register_learning_tools(pipeline._tool_executor)
            runner = pipeline._tool_runner
            if runner is not None and hasattr(runner, "set_research_observer"):
                runner.set_research_observer(self.store, self.config.research_calls_per_day)
            self.store.observe("tool_catalog", pipeline._build_tool_catalog_for_intent())

    def _register_learning_tools(self, executor: Any) -> None:
        schemas = {
            "self.learn": {
                "title": {"type": "string", "required": True},
                "text": {"type": "string", "required": True},
            },
            "self.goal": {
                "goal": {"type": "string"},
                "goal_id": {"type": "string"},
                "goal_status": {"type": "string"},
                "reason": {"type": "string"},
            },
            "self.question": {
                "question_id": {"type": "integer", "required": True},
                "question_status": {"type": "string", "required": True},
            },
        }
        if self.config.research_execution_mode == "closed_loop":
            schemas["self.goal"].update({
                "success_criteria": {"type": "object"},
                "next_step": {"type": "string"},
                "expected_result": {"type": "string"},
                "wake_after_seconds": {"type": "number"},
                "wake_on_external_input": {"type": "boolean"},
                "parent_goal_id": {"type": ["string", "null"]},
                "depends_on": {"type": "array", "items": {"type": "string"}},
            })
            schemas["self.check_action"] = {"action_id": {"type": "string", "required": True}}
        if self.config.research_context_mode == "task_only":
            schemas = {name: schema for name, schema in schemas.items() if name in {"self.goal", "self.check_action"}}
        if self.manifest.get("memory_presentation_version", 1) >= 2 and self.config.research_memory_mode != "legacy":
            schemas["self.recall"] = {"context": {"type": "string", "required": True}}
        for name, schema in schemas.items():
            executor.registry.register(
                ToolCapability(
                    name=name,
                    description={
                        "self.learn": "Choose material to digest into your identity; completion is asynchronous.",
                        "self.goal": "Create, revise, complete or abandon your own persistent intention.",
                        "self.question": "Update your own investigation question after deliberation.",
                        "self.check_action": "Ask an independent environment reconciler whether an uncertain action happened. This does not repeat the action; unsupported effects stay unknown.",
                        "self.recall": "Retrieve your bounded outcome memory for an exact context, such as lab:beta or world:garden:collaborate:Ari. Uses your existing memory policy, never hidden evaluator data. No retained outcomes means no recalled outcomes. This read does not change your goal or require activity.",
                    }[name],
                    category=ToolCategory.COGNITIVE,
                    arg_schema=schema,
                )
            )

            def handle(arguments, tool_name=name):
                if tool_name == "self.learn":
                    title = str(arguments.get("title") or "").strip()
                    text = str(arguments.get("text") or "").strip()
                    if not title or not text or len(text) > 250000:
                        raise ValueError("A title and 1–250000 characters of material are required")
                    record_id = uuid.uuid4().hex
                    self.store.put(
                        "learning",
                        record_id,
                        {
                            "id": record_id,
                            "status": "pending",
                            "attempts": 0,
                            "title": title,
                            "text": text,
                            "source_type": "self_selected_material",
                            "source_ref": record_id,
                            "metadata": {
                                "source_event_id": self.state.get("current_event_id"),
                                "evidence_kind": "participant_claim"
                                if self.state.get("current_event_kind") == "external"
                                else "self_inference",
                            },
                        },
                    )
                    output = f"Learning queued: {record_id}"
                elif tool_name == "self.goal":
                    self.state["goal_id"] = self._update_goal(arguments, executor=executor)
                    if self.state["goal_id"] is None:
                        raise ValueError("A goal description or existing goal id is required")
                    output = f"Persistent intention updated: {self.state['goal_id']}"
                elif tool_name == "self.recall":
                    context = arguments.get("context")
                    if not isinstance(context, str) or not context.strip() or len(context) > 200:
                        raise ValueError("Recall requires a nonempty context string of at most 200 characters")
                    output = encode(self.outcome_learning.context(
                        mode=self.config.research_memory_mode,
                        selected_context=context,
                        presentation_version=self.manifest.get("memory_presentation_version", 1),
                        focused=True,
                    ))
                elif tool_name == "self.check_action":
                    output = encode(self._reconcile_action(arguments.get("action_id"), executor))
                else:
                    if arguments.get("question_id") is None:
                        raise ValueError("A question id is required")
                    self._update_question(arguments)
                    output = "Investigation status updated"
                return ToolResult(tool_name=tool_name, success=True, output=output)

            executor.register_handler(name, handle)

    def _reconcile_action(self, action_id: str, executor: Any) -> dict[str, Any]:
        if not isinstance(action_id, str):
            raise ValueError("An action id is required")
        action = self.store.get("action", action_id)
        if action is None:
            raise ValueError("Unknown action id")
        if action["status"] not in {"started", "unknown"}:
            return {"status": action["status"], "method": "existing_receipt"}
        try:
            verdict = executor.reconcile_action(action)
            if verdict.get("status") not in {"applied", "not_applied", "unknown"}:
                raise ValueError("Invalid reconciliation result")
            if verdict["status"] == "applied":
                result = ToolResult(**verdict["result"])
                if result.tool_name != action["tool"] or result.success is not True:
                    raise ValueError("Reconciled result does not match the action")
                action.update(status="completed", result=asdict(result))
            elif verdict["status"] == "not_applied":
                action.update(status="not_applied", result=asdict(ToolResult(
                    tool_name=action["tool"], success=False, output="",
                    error="Independent environment evidence confirms the effect was not applied.",
                )))
        except (ResearchBudgetExceeded, ResearchStudyLimit):
            raise
        except Exception as error:
            verdict = {"status": "unknown", "method": "reconciler_failure", "error_type": type(error).__name__}
        self.store.put_records(
            [("action", action_id, action)], event_kind="action_reconciled",
            event={"action_id": action_id, "verdict": verdict},
        )
        goal_execution.record_action(self.store, action)
        if self.outcome_learning is not None:
            self.outcome_learning.after_action(action)
        if self.config.research_dispositions:
            record_choice(self.store, action)
        return verdict

    def outcome_context(self, goal_id: str | None = None) -> dict[str, Any]:
        observation = self.store.get("public_environment", "current")
        context = {"environment_observation": observation} if observation else {}
        if self.outcome_learning is None:
            return context
        selected = None
        if self.config.research_appraisal_mode != "legacy":
            appraisal = allocate_attention(
                self.store.list("learning_state"), self.store.list("goal_execution"),
                self.config.research_appraisal_mode,
            )
            context["grounded_appraisal"] = appraisal
            selected = appraisal["selected_context"]
            self.store.put_records(
                [("attention", "current", appraisal)], event_kind="attention_allocated", event=appraisal,
            )
        context["outcome_memory"] = self.outcome_learning.context(
            mode=self.config.research_memory_mode, goal_id=goal_id,
            selected_context=selected, progress=self.config.research_progress_estimates,
            presentation_version=self.manifest.get("memory_presentation_version", 1),
        )
        if self.config.research_dispositions:
            context["behavioral_dispositions"] = [item for item in self.store.list("disposition") if item["status"] == "descriptive"][-8:]
        return context

    def execution_receipts(self, goal_id: str | None = None) -> list[dict[str, Any]]:
        receipts = self.store.recent_action_receipts(goal_id=goal_id)
        return receipts if self.config.research_memory_mode == "legacy" else receipts[-1:]

    def _snapshot(self) -> dict[str, Any]:
        return {**self.state, **self.engine.export_state(), "saved_at": time.time()}

    def detach(self, pipeline: Any) -> None:
        with self.lock:
            key = pipeline._research_session_key
            if self._pipelines.get(key) is pipeline:
                self._pipelines.pop(key)

    def _session_record(self, pipeline: Any) -> dict[str, Any]:
        return {
            "session_key": pipeline._research_session_key,
            "rel_key": pipeline._research_rel_key,
            "user_id": pipeline._research_user_id,
            "pipeline": pipeline.export_session_state(),
            "history": pipeline.export_conversation_history(),
        }

    def process(self, pipeline: Any, message: str, user_id: str) -> Any:
        return self.recorded(
            "message",
            {"session_key": pipeline._research_session_key, "user_id": user_id, "text": message},
            lambda: self._process(pipeline, message, user_id),
        )

    def recorded(self, kind: str, payload: dict[str, Any], operation: Callable[[], Any]) -> Any:
        with self.lock, configuration_scope(self.config._research_configuration):
            frame = None
            error = None
            try:
                with capture() as frame:
                    return operation()
            except Exception as failure:
                error = str(failure)
                raise
            finally:
                if frame is not None:
                    self.store.observe(
                        "replay_frame",
                        {
                            "kind": kind,
                            "input": payload,
                            "times": frame["times"],
                            "ids": frame["ids"],
                            "error": error,
                            "state": self.engine.export_state(),
                            "version": self.state["version"],
                            "identity": self.identity_snapshot(),
                            "goals": self.store.list("goal"),
                            **({"public_environment": self.store.get("public_environment", "current")} if self.store.get("public_environment", "current") else {}),
                            **({"outcome_learning": {kind: self.store.list(kind) for kind in (*outcome_learning.RECORD_KINDS, "attention", "behavioral_choice", "disposition")}} if self.outcome_learning is not None else {}),
                            **({
                                "goal_executions": self.store.list("goal_execution"),
                                "completion_assessments": self.store.list("completion_assessment"),
                                "wake_condition": self.state.get("wake_condition"),
                            } if self.config.research_execution_mode == "closed_loop" else {}),
                        },
                    )

    def identity_snapshot(self) -> dict[str, Any]:
        with LifeHistoryStore(self.config, read_only=True) as store:
            return {
                "beliefs": [
                    dict(row)
                    for row in store._conn.execute(
                        "SELECT key,statement,confidence,status FROM beliefs ORDER BY key"
                    )
                ],
                "drives": [
                    dict(row)
                    for row in store._conn.execute(
                        "SELECT name,value FROM drive_states ORDER BY name"
                    )
                ],
                "themes": [
                    dict(row)
                    for row in store._conn.execute(
                        "SELECT signature,reinforcement_count,accrued_weight,conflicting_count FROM theme_signatures ORDER BY signature"
                    )
                ],
            }

    def _process(self, pipeline: Any, message: str, user_id: str) -> Any:
        with self.lock, configuration_scope(self.config._research_configuration):
            if self.state.get("study_stopped"):
                raise ResearchStudyLimit(self.state["study_stopped"])
            before = self.engine.snapshot()
            trust_before = pipeline.person_profiles.get_or_create(user_id).trust
            event_id = self.store.observe(
                "external_input",
                {
                    "session_key": pipeline._research_session_key,
                    "user_id": user_id,
                    "text": message,
                },
            )
            self.state["current_event_id"] = event_id
            self.state["current_event_kind"] = "external"
            self.state["goal_id"] = None
            pipeline._last_turn_time = self.state.get("last_state_update_at")
            pipeline._last_state_update_at = self.state.get("last_state_update_at")
            pipeline._research_in_transition = True
            self.state["last_external_input_at"] = time.time()
            try:
                result = pipeline.process(message, user_id)
                result.raise_for_backend_failure()
                episode = self.store.get("episode", pipeline._episode_id) or {
                    "id": pipeline._episode_id,
                    "status": "collecting",
                    "messages": [],
                    "turns": 0,
                    "session_key": pipeline._research_session_key,
                    "user_id": user_id,
                }
                episode["messages"].extend(
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": result.response},
                    ]
                )
                episode.update(
                    {
                        "turns": episode["turns"] + 1,
                        "updated_at": time.time(),
                        "short_term": pipeline.short_term.export_state(),
                        "peak_intensity": max(
                            episode.get("peak_intensity", 0.0), result.debug.event_intensity
                        ),
                        "trust_delta": episode.get("trust_delta", 0.0)
                        + pipeline.person_profiles.get_or_create(user_id).trust
                        - trust_before,
                        "unresolved": bool(result.debug.unresolved_items),
                        "source_event_ids": [*episode.get("source_event_ids", []), event_id],
                    }
                )
                self.store.observe(
                    "selected_expression",
                    {
                        "source_event_id": event_id,
                        "session_key": pipeline._research_session_key,
                        "text": result.response,
                        "deliberate_silence": result.debug.deliberate_silence,
                    },
                )
                self.state.update(
                    version=self.state["version"] + 1, last_state_update_at=time.time()
                )
                self.store.checkpoint(
                    self._snapshot(),
                    {
                        "origin": "external",
                        "event_id": event_id,
                        "before": before,
                        "after": self.engine.snapshot(),
                        "debug": asdict(result.debug),
                    },
                    [
                        ("episode", pipeline._episode_id, episode),
                        ("session", pipeline._research_session_key, self._session_record(pipeline)),
                    ],
                )
                self._finalize_episodes(force_size=True)
                return result
            except Exception as error:
                status = "failed"
                if isinstance(error, ResearchStudyLimit):
                    self.state["study_stopped"] = str(error)
                    status = "resource_stop"
                elif isinstance(error, ResearchBudgetExceeded):
                    self.state["resume_at"] = error.resume_at
                    status = "resource_wait"
                failed_episode = self.store.get("episode", pipeline._episode_id) or {
                    "id": pipeline._episode_id,
                    "status": "collecting",
                    "messages": [],
                    "turns": 0,
                    "session_key": pipeline._research_session_key,
                    "user_id": user_id,
                    "source_event_ids": [],
                }
                failed_episode["messages"].append({"role": "user", "content": message})
                failed_episode["source_event_ids"].append(event_id)
                failed_episode.update(
                    turns=failed_episode["turns"] + 1,
                    updated_at=time.time(),
                    short_term=pipeline.short_term.export_state(),
                    infrastructure_error=str(error),
                )
                self.state.update(
                    version=self.state["version"] + 1, last_state_update_at=time.time()
                )
                self.store.checkpoint(
                    self._snapshot(),
                    {
                        "origin": "external",
                        "event_id": event_id,
                        "status": status,
                        "error": str(error),
                        "before": before,
                        "after": self.engine.snapshot(),
                    },
                    [
                        ("session", pipeline._research_session_key, self._session_record(pipeline)),
                        ("episode", pipeline._episode_id, failed_episode),
                    ],
                )
                raise
            finally:
                pipeline._research_in_transition = False

    def queue_autobiography(self, episode_id: str, result: DigestedSession, user_id: str) -> None:
        self.store.put(
            "learning",
            episode_id,
            {
                "id": episode_id,
                "status": "pending",
                "text": result.summary,
                "title": f"Experience with {user_id}: {result.emotional_arc_label}"[:180],
                "source_type": "conversation_learning_text",
                "source_ref": f"episode:{episode_id}",
                "participants": [self.config._research_configuration.soul.name, user_id],
                "metadata": {
                    "input_mode": "autobiographical",
                    "peak_intensity": result.peak_intensity,
                    "trust_delta": result.trust_delta,
                },
                "attempts": 0,
            },
        )

    def end_session(self, pipeline: Any, user_id: str) -> DigestedSession:
        return self.recorded(
            "end_session",
            {"session_key": pipeline._research_session_key, "user_id": user_id},
            lambda: self._end_session(pipeline, user_id),
        )

    def _end_session(self, pipeline: Any, user_id: str) -> DigestedSession:
        with self.lock, configuration_scope(self.config._research_configuration):
            episode_id = pipeline._episode_id
            pipeline._research_in_transition = True
            try:
                result = pipeline.end_session(user_id)
                records = [
                    ("session", pipeline._research_session_key, self._session_record(pipeline))
                ]
                episode = self.store.get("episode", episode_id)
                if episode:
                    episode["status"] = "segmented"
                    records.append(("episode", episode_id, episode))
                self.state.update(
                    version=self.state["version"] + 1, last_state_update_at=time.time()
                )
                self.store.checkpoint(
                    self._snapshot(),
                    {"origin": "conversation_end", "episode_id": episode_id},
                    records,
                )
                return result
            except (ResearchBudgetExceeded, ResearchStudyLimit) as error:
                if isinstance(error, ResearchStudyLimit):
                    self.state["study_stopped"] = str(error)
                    status = "resource_stop"
                else:
                    self.state["resume_at"] = error.resume_at
                    status = "resource_wait"
                self.state.update(
                    version=self.state["version"] + 1, last_state_update_at=time.time()
                )
                self.store.checkpoint(
                    self._snapshot(),
                    {
                        "origin": "conversation_end",
                        "episode_id": episode_id,
                        "status": status,
                        "error": str(error),
                    },
                    [("session", pipeline._research_session_key, self._session_record(pipeline))],
                )
                raise
            finally:
                pipeline._research_in_transition = False

    def _finalize_episodes(self, *, force_size: bool = False) -> None:
        now = time.time()
        for episode in self.store.list("episode"):
            if episode["status"] != "collecting":
                continue
            due = episode["turns"] >= self.config.research_episode_max_turns
            if not force_size:
                due = (
                    due or now - episode["updated_at"] >= self.config.research_episode_idle_seconds
                )
            if not due:
                continue
            entries = episode.get("short_term", [])
            peak = max(
                episode.get("peak_intensity", 0.0),
                max((entry["event"]["intensity"] for entry in entries), default=0.0),
            )
            salient = (
                peak >= 0.6
                or abs(episode.get("trust_delta", 0.0)) >= 0.08
                or episode.get("unresolved", False)
            )
            episode["status"] = "segmented"
            records = [("episode", episode["id"], episode)]
            if salient:
                records.append(
                    (
                        "learning",
                        episode["id"],
                        {
                            "id": episode["id"],
                            "status": "pending",
                            "attempts": 0,
                            "title": f"Lived episode with {episode['user_id']}",
                            "text": "\n".join(
                                f"{item['role']}: {item['content']}" for item in episode["messages"]
                            ),
                            "source_type": "conversation_learning_text",
                            "source_ref": f"episode:{episode['id']}",
                            "participants": [episode["user_id"]],
                            "metadata": {
                                "source_event_ids": episode["source_event_ids"],
                                "peak_intensity": peak,
                                "trust_delta": episode.get("trust_delta", 0.0),
                            },
                        },
                    )
                )
            pipeline = self._pipelines.get(episode["session_key"])
            saved = self.store.get("session", episode["session_key"])
            if saved and saved["pipeline"]["episode_id"] == episode["id"]:
                saved["pipeline"]["episode_id"] = uuid.uuid4().hex
                saved["pipeline"]["short_term"] = []
                records.append(("session", episode["session_key"], saved))
                if pipeline is not None and pipeline._episode_id == episode["id"]:
                    pipeline._episode_id = saved["pipeline"]["episode_id"]
                    pipeline.short_term.clear()
            self.store.checkpoint(
                self._snapshot(),
                {"origin": "episode_segmentation", "episode_id": episode["id"]},
                records,
            )

    def _learn_pending(self, backend: Any) -> None:
        now = time.time()
        pending = [
            item
            for item in self.store.list("learning")
            if item["status"] == "pending" and item.get("retry_at", 0) <= now
        ]
        if not pending:
            return
        item = min(pending, key=lambda item: (item.get("attempts", 0), item["id"]))
        try:
            self.store.observe(
                "learning_started", {"id": item["id"], "source_ref": item["source_ref"]}
            )
            with LifeHistoryStore(self.config, read_only=True) as store:
                current_self = store.effective_self()
                learning_candidates = store.developing.candidates()
            digest = _digest_experience(
                title=item["title"],
                text=item["text"],
                source_type=item["source_type"],
                llm_client=backend,
                research_mode=self.config.research_mode,
                current_self=current_self,
                learning_candidates=learning_candidates,
            )
            if digest.get("digest_quality") != "high":
                raise RuntimeError("Learning backend did not produce a usable digest")

            class PreparedBackend:
                def generate(self, system_prompt: str, user_message: str) -> str:
                    if system_prompt.startswith("You audit whether"):
                        return backend.generate(system_prompt, user_message)
                    return encode(digest)

            with LifeHistoryStore(self.config) as store:
                result = store.ingest_external_text(
                    title=item["title"],
                    text=item["text"],
                    source_type=item["source_type"],
                    source_ref=item["source_ref"],
                    participants=item.get("participants", []),
                    llm_client=PreparedBackend(),
                    metadata={**item.get("metadata", {}), "occurrence_id": item["id"]},
                )
            item.update(status="completed", experience_id=result["experience"]["id"])
            self.store.observe(
                "learning_completed", {"id": item["id"], "experience_id": item["experience_id"]}
            )
        except (ResearchBudgetExceeded, ResearchStudyLimit):
            raise
        except Exception as error:
            attempts = item.get("attempts", 0) + 1
            item.update(
                attempts=attempts,
                error=str(error),
                retry_at=now + min(3600, 10 * 2 ** min(attempts, 8)),
            )
            self.store.observe("learning_deferred", {"id": item["id"], "error": str(error)})
        self.store.put("learning", item["id"], item)

    def observe_tool(
        self,
        stage: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        action_id: str | None = None,
    ) -> str:
        action_id = action_id or uuid.uuid4().hex
        record = {
            "id": action_id,
            "tool": tool_name,
            "arguments": arguments,
            "status": "started" if stage == "started" else "completed",
            "result": asdict(result) if result is not None else None,
            "state_version": self.state["version"],
            "origin": self.state.get("action_origin") or self.state.get("origin", "external"),
            "goal_id": self.state.get("goal_id"),
            "source_event_id": self.state.get("current_event_id"),
        }
        if self.config.research_execution_mode == "closed_loop":
            previous = self.store.get("action", action_id)
            if previous:
                record["goal_id"] = previous.get("goal_id")
                record["goal_revision"] = previous.get("goal_revision")
            else:
                execution = self.store.get("goal_execution", record["goal_id"]) if record["goal_id"] else None
                record["goal_revision"] = execution["revision"] if execution else None
        self.store.put("action", action_id, record)
        if self.outcome_learning is not None:
            if stage == "started":
                self.outcome_learning.before_action(record)
            else:
                self.outcome_learning.after_action(record)
        if self.config.research_dispositions and stage == "completed":
            record_choice(self.store, record)
        if self.config.research_execution_mode == "closed_loop":
            goal_execution.record_action(self.store, record)
        self.store.observe("tool_" + stage, record)
        if (
            stage == "completed"
            and tool_name == "self.note"
            and result is not None
            and result.success
        ):
            self.store.put(
                "note",
                action_id,
                {
                    "id": action_id,
                    "text": arguments.get("text", ""),
                    "topic": arguments.get("topic", ""),
                    "evidence_kind": "self_inference",
                },
            )
        return action_id

    def _recover_actions(self) -> None:
        for action in self.store.list("action"):
            if action["status"] == "started":
                action["status"] = "unknown"
                self.store.put("action", action["id"], action)
                self.store.observe("uncertain_action", action)
            if self.config.research_execution_mode == "closed_loop":
                goal_execution.record_action(self.store, action, recovering=True)
            if self.outcome_learning is not None:
                self.outcome_learning.after_action(action)
            if self.config.research_dispositions:
                record_choice(self.store, action)
        for delivery in self.store.list("delivery"):
            if delivery["status"] == "sending":
                delivery["status"] = "unknown"
                self.store.put("delivery", delivery["id"], delivery)

    def tick(self) -> list[dict[str, Any]]:
        return self.recorded("tick", {}, self._tick)

    def observe_environment(self, observation: dict[str, Any]) -> None:
        if not isinstance(observation, dict) or len(encode(observation)) > 16000:
            raise ValueError("Public environment observation must be a bounded object")
        records = [("public_environment", "current", observation)]
        if "demonstrations" in observation:
            records.append(("public_observation", f"opportunity:{observation.get('opportunity_id')}", observation))
        self.recorded(
            "environment_observation",
            {"observation": observation},
            lambda: self.store.put_records(
                records,
                event_kind="environment_observed",
                event=observation,
            ),
        )

    def _deliberate(self, backend: Any, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        schema = (
            _EXECUTION_DECISION_SCHEMA
            if self.manifest.get("decision_contract_version", 1) >= 3
            and self.config.research_execution_mode == "closed_loop"
            else _DECISION_SCHEMA
        )
        destinations = ["", *context.get("destinations", [])]
        actions = list(_DECISION_ACTIONS)
        if self.manifest.get("decision_contract_version", 1) >= 4:
            if len(destinations) == 1:
                actions.remove("contact")
            schema = {
                **schema,
                "properties": {
                    **schema["properties"],
                    "action": {"type": "string", "enum": actions},
                    "session_key": {"type": "string", "enum": destinations},
                },
            }

        def validation_error(decision):
            if not isinstance(decision, dict) or decision.get("action") not in actions:
                return "Return one JSON object with action exactly one of: " + ", ".join(actions)
            if (
                self.manifest.get("decision_contract_version", 1) >= 4
                and decision["action"] == "contact"
                and (
                    (decision.get("message") and not decision.get("session_key"))
                    or (
                        decision.get("session_key") is not None
                        and (
                            not isinstance(decision["session_key"], str)
                            or decision["session_key"] not in destinations
                        )
                    )
                )
            ):
                return "Contact requires a destination supplied in context; world partners are not chat destinations. No contact was sent. Choose a valid destination or another activity."
            return None

        schema_options = {"response_schema": schema} if self.config.research_json_schema else {}
        response = backend.generate(prompt, encode(context), **schema_options)
        if self.manifest.get("decision_contract_version", 1) == 1:
            decision = _extract_json(response)
            if not isinstance(decision, dict) or decision.get("action") not in set(_DECISION_ACTIONS):
                raise ValueError("Autonomous deliberation returned an invalid decision")
            return decision
        decision = _decision_json(response)
        error = validation_error(decision)
        if error is None:
            return decision
        feedback = {
            "rejected_response": response,
            "error": error,
            "retry_limit": 1,
        }
        self.store.observe("decision_contract_retry_requested", feedback)
        candidate = _decision_json(
            backend.generate(
                prompt + _DECISION_RETRY_PROMPT,
                encode({**context, "decision_contract_feedback": feedback}),
                **schema_options,
            )
        )
        if validation_error(candidate) is not None:
            raise ValueError("Autonomous deliberation returned an invalid decision after one retry")
        self.store.observe(
            "decision_contract_recovered",
            {
                "rejected_action": decision.get("action") if decision else None,
                "selected_action": candidate["action"],
            },
        )
        return candidate

    def _decision_metadata_errors(
        self, decision: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, str]:
        errors = {}
        goal_id = decision.get("goal_id", "")
        if not isinstance(goal_id, str):
            errors["goal_id"] = "Goal id must be text"
        elif goal_id and self.store.get("goal", goal_id) is None:
            errors["goal_id"] = "Unknown goal id; use an existing id or empty for a new goal"
        goal_status = decision.get("goal_status", "active")
        if not isinstance(goal_status, str) or goal_status not in {
            "active",
            "waiting",
            "completed",
            "abandoned",
        }:
            errors["goal_status"] = "Invalid goal status"
        question_id = decision.get("question_id")
        if question_id is not None:
            if isinstance(question_id, bool) or not isinstance(question_id, int):
                errors["question_id"] = "Question id must be an integer from questions or null"
            elif question_id not in {item["id"] for item in context["questions"]}:
                errors["question_id"] = "Unknown question id; use an existing id or null"
            question_status = decision.get("question_status", "pursuing")
            if not isinstance(question_status, str) or question_status not in {
                "open",
                "pursuing",
                "resolved",
                "abandoned",
            }:
                errors["question_status"] = "Invalid question status"
        return errors

    def _repair_decision_metadata(
        self, backend: Any, decision: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        errors = self._decision_metadata_errors(decision, context)
        if not errors:
            return decision
        self.store.observe("decision_metadata_repair_requested", {"errors": errors})
        patch = _extract_json(
            backend.generate(
                _METADATA_REPAIR_PROMPT,
                encode(
                    {
                        "decision": decision,
                        "errors": errors,
                        "goals": context["goals"] + context["recent_goal_outcomes"],
                        "questions": context["questions"],
                    }
                ),
            )
        )
        if not isinstance(patch, dict) or set(patch) != set(errors):
            raise ValueError("Decision metadata repair must change only the invalid fields")
        candidate = {**decision, **patch}
        remaining = self._decision_metadata_errors(candidate, context)
        if remaining:
            raise ValueError(f"Decision metadata repair failed: {encode(remaining)}")
        self.store.observe(
            "decision_metadata_repaired",
            {key: {"before": decision.get(key), "after": value} for key, value in patch.items()},
        )
        return candidate

    def _tick(self) -> list[dict[str, Any]]:
        with self.lock, configuration_scope(self.config._research_configuration):
            now = time.time()
            self._finalize_episodes()
            if self.state.get("study_stopped"):
                return []
            if now < self.state.get("resume_at", 0):
                return []
            if self.config.research_execution_mode == "closed_loop":
                if not goal_execution.wake_ready(self.state.get("wake_condition"), self.store, now):
                    return [item for item in self.store.list("delivery") if item["status"] == "pending"]
                self.state.pop("wake_condition", None)
            observation = self.store.get("public_environment", "current")
            if observation and self.outcome_learning is not None:
                self.outcome_learning.observe_environment(observation)
            self.state["current_event_id"] = self.store.observe(
                "internal_activation", {"state_version": self.state["version"]}
            )
            self.state["current_event_kind"] = "internal"
            if self._background is None:
                if self._session_factory is None:
                    raise RuntimeError("Research runtime has no pipeline factory")
                self._background = self._session_factory(
                    "agent:self:internal", "agent:self", "self"
                ).pipeline
            pipeline = self._background
            self.state["origin"] = "autonomous"
            if self.config.research_execution_mode == "closed_loop":
                self.state["goal_id"] = None
            previous = self.state.get("last_state_update_at", now)
            self.engine.decay(max(0, now - previous))
            self.state["last_state_update_at"] = now
            try:
                context = {}
                questions = []
                if self.config.research_context_mode == "full":
                    with LifeHistoryStore(self.config) as store:
                        metabolism = store.wall_clock_decay(
                            min_elapsed_days=self.config.metabolism_min_elapsed_days
                        )
                        context = store.prompt_context()
                        questions = store.list_open_questions(
                            status="open"
                        ) + store.list_open_questions(status="pursuing")
                    self.store.observe("metabolism", metabolism)
                    self._learn_pending(pipeline._llm_backend)
                experiences, cursor = self.store.experience_context(
                    after=self.state.get("observation_cursor", 0)
                )
                context.update(
                    {
                        "goals": [
                            item
                            for item in self.store.list("goal")
                            if item["status"] in {"active", "waiting"}
                        ][-20:],
                        "questions": questions,
                        "unfinished_tasks": [
                            item["pipeline"]["active_task_plan"]
                            for item in self.store.list("session")
                            if item["pipeline"].get("active_task_plan")
                        ],
                        "modulators": self.engine.snapshot(),
                        "self_notes": [
                            {**item, "text": item["text"][:2000]}
                            for item in self.store.list("note")[-10:]
                        ],
                        "recent_goal_outcomes": [
                            item
                            for item in self.store.list("goal")
                            if item["status"] in {"completed", "abandoned"}
                        ][-10:],
                        "uncertain_actions": [
                            item
                            for item in self.store.list("action")
                            if item["status"] == "unknown"
                        ],
                        "deliveries": self.store.list("delivery")[-20:],
                        "destinations": [
                            item["session_key"]
                            for item in self.store.list("session")
                            if not item["session_key"].startswith("agent:")
                        ],
                        "recent_experiences": experiences,
                        "available_tools": pipeline._build_tool_catalog_for_intent(),
                        "workspace": getattr(
                            self.config, "_research_prompt_workspace", self.config.resolved_tools_workspace
                        ),
                    }
                )
                if self.config.research_context_mode == "task_only":
                    context.pop("modulators", None)
                    context["recent_experiences"] = [
                        item for item in experiences if item["kind"] == "tool_completed"
                    ]
                if self.config.research_memory_mode != "legacy":
                    context["recent_experiences"] = []
                    context["self_notes"] = []
                    context["last_operational_receipt"] = self.execution_receipts()
                context.update(self.outcome_context())
                deliberation_prompt = _DELIBERATION_PROMPT
                if self.config.research_execution_mode == "closed_loop":
                    deliberation_prompt += goal_execution.EXECUTION_PROMPT
                    if self.manifest.get("decision_contract_version", 1) >= 3:
                        deliberation_prompt += _EXECUTION_DECISION_FIELDS
                    context["goal_executions"] = self.store.list("goal_execution")[-20:]
                    context["recent_completion_assessments"] = self.store.list("completion_assessment")[-10:]
                decision = self._deliberate(
                    pipeline._llm_backend, deliberation_prompt, context
                )
                normalized = {}
                for key in ("goal_id", "goal", "message", "session_key"):
                    if key in decision and decision[key] is None:
                        decision[key] = ""
                        normalized[key] = {"before": None, "after": ""}
                question_id = decision.get("question_id")
                if question_id == "":
                    decision["question_id"] = None
                    normalized["question_id"] = {"before": "", "after": None}
                elif isinstance(question_id, str) and question_id.isdecimal():
                    decision["question_id"] = int(question_id)
                    normalized["question_id"] = {
                        "before": question_id,
                        "after": decision["question_id"],
                    }
                if normalized:
                    self.store.observe("decision_format_normalized", normalized)
                for key in ("reason", "goal", "message", "session_key"):
                    if key in decision and not isinstance(decision[key], str):
                        raise ValueError(f"Autonomous decision {key} must be text")
                if (
                    decision["action"] == "contact"
                    and decision.get("session_key")
                    and decision["session_key"] not in context["destinations"]
                ):
                    raise ValueError("Unknown contact destination")
                decision = self._repair_decision_metadata(pipeline._llm_backend, decision, context)
                self.store.observe("autonomous_decision", decision)
                selected_wake = (
                    goal_execution.wake_condition(decision, self.store)
                    if self.config.research_execution_mode == "closed_loop" else None
                )
                goal_id = self._update_goal(decision, executor=pipeline._tool_executor)
                if goal_id:
                    decision["goal_id"] = goal_id
                self._update_question(decision)
                action = decision["action"]
                message = str(decision.get("message") or "")
                if action in {"rest", "wait"} and selected_wake:
                    self.state["wake_condition"] = selected_wake
                if action == "pursue":
                    self.state["goal_id"] = decision.get("goal_id")
                    action_context = (
                        {"decision": decision}
                        if self.config.research_action_feedback == "receipts"
                        else {}
                    )
                    pipeline.process_autonomous(
                        str(decision.get("goal") or decision.get("reason") or ""),
                        self.store.list("goal"),
                        **action_context,
                    )
                elif (
                    action == "reflect"
                    and self.config.research_context_mode == "full"
                    and now - self.state.get("last_reflection_at", 0)
                    >= self.config.research_reflection_interval
                ):
                    reflection_id = uuid.uuid4().hex
                    self.store.put(
                        "learning",
                        reflection_id,
                        {
                            "id": reflection_id,
                            "status": "pending",
                            "attempts": 0,
                            "title": "Self-reflection",
                            "text": message or str(decision.get("reason") or ""),
                            "source_type": "self_reflection",
                            "source_ref": reflection_id,
                            "metadata": {
                                "evidence_kind": "self_inference",
                                "goal_id": decision.get("goal_id"),
                            },
                        },
                    )
                    self.state["last_reflection_at"] = now
                elif action == "contact" and message:
                    target = str(decision.get("session_key") or "")
                    delivery_id = uuid.uuid4().hex
                    self.store.put(
                        "delivery",
                        delivery_id,
                        {
                            "id": delivery_id,
                            "session_key": target,
                            "message": message,
                            "status": "pending",
                        },
                    )
                self.state.update(
                    version=self.state["version"] + 1,
                    last_deliberation_at=now,
                    observation_cursor=cursor,
                )
                self.store.checkpoint(
                    self._snapshot(),
                    {"origin": "autonomous", "decision": decision},
                    [("session", pipeline._research_session_key, self._session_record(pipeline))],
                )
            except ResearchStudyLimit as error:
                self.state["study_stopped"] = str(error)
                self.store.checkpoint(self._snapshot(), {"origin": "infrastructure", "status": "resource_stop", "reason": str(error)}, [("session", pipeline._research_session_key, self._session_record(pipeline))])
            except ResearchBudgetExceeded as error:
                self.state["resume_at"] = error.resume_at
                self.store.checkpoint(
                    self._snapshot(),
                    {
                        "origin": "infrastructure",
                        "status": "resource_wait",
                        "resume_at": error.resume_at,
                    },
                    [("session", pipeline._research_session_key, self._session_record(pipeline))],
                )
            except Exception as error:
                diagnosis = diagnose_llm_error(error, self.config)
                if diagnosis:
                    record_llm_health(self.config, diagnosis)
                self.store.observe(
                    "autonomous_failure", {"type": type(error).__name__, "error": str(error)}
                )
                self.store.checkpoint(
                    self._snapshot(),
                    {"origin": "infrastructure", "status": "failed", "error": str(error)},
                    [("session", pipeline._research_session_key, self._session_record(pipeline))],
                )
            finally:
                self.state["origin"] = "external"
            return [item for item in self.store.list("delivery") if item["status"] == "pending"]

    def _update_goal(self, decision: dict[str, Any], *, executor: Any = None) -> str | None:
        if self.config.research_execution_mode == "closed_loop":
            return goal_execution.update_goal(self.store, decision, executor=executor)
        goal_id = str(decision.get("goal_id") or "")
        description = str(decision.get("goal") or "")
        if not goal_id and not description:
            return
        goal = self.store.get("goal", goal_id) if goal_id else None
        if goal_id and goal is None:
            raise ValueError("Unknown goal id")
        status = decision.get("goal_status", "active")
        if status not in {"active", "waiting", "completed", "abandoned"}:
            raise ValueError("Invalid goal status")
        goal_id = goal_id or uuid.uuid4().hex
        self.store.put(
            "goal",
            goal_id,
            {
                **(goal or {}),
                "id": goal_id,
                "description": description or goal["description"],
                "status": status,
                "reason": str(decision.get("reason") or ""),
                "updated_at": time.time(),
            },
        )
        self.store.observe(
            "goal_changed", {"id": goal_id, "status": status, "description": description}
        )
        return goal_id

    def _update_question(self, decision: dict[str, Any]) -> None:
        question_id = decision.get("question_id")
        if question_id is None:
            return
        status = decision.get("question_status", "pursuing")
        if status not in {"open", "pursuing", "resolved", "abandoned"}:
            raise ValueError("Invalid question status")
        with LifeHistoryStore(self.config) as store:
            row = store._conn.execute(
                "SELECT * FROM open_questions WHERE id=?", (int(question_id),)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown question id")
            metadata = json.loads(row["metadata_json"])
            if row["status"] == "superseded":
                raise ValueError(f"Question was superseded by {metadata.get('superseded_by')}")
            transition = {
                "before": row["status"],
                "after": status,
                "reason": str(decision.get("reason") or ""),
                "at": time.time(),
            }
            metadata["agent_status_history"] = [
                *metadata.get("agent_status_history", []),
                transition,
            ][-20:]
            store._conn.execute(
                "UPDATE open_questions SET status=?, last_pursued_at=?, metadata_json=? WHERE id=?",
                (
                    status,
                    time.time() if status == "pursuing" else row["last_pursued_at"],
                    encode(metadata),
                    int(question_id),
                ),
            )
            store._conn.commit()
            self.store.observe("question_changed", {"question_id": int(question_id), **transition})

    def close(self) -> None:
        if self._closed:
            return
        with self.lock:
            self._closed = True
            self.store.checkpoint(self._snapshot(), {"origin": "owner_stopped"})
            if self._background is not None:
                self._background.close()
            fcntl.flock(self._owner_file, fcntl.LOCK_UN)
            self._owner_file.close()

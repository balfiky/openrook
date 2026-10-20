from __future__ import annotations

import asyncio
import copy
import json
import math

import pytest

from config.loader import get_config
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore
from runtime.sessions.manager import SessionManager


class ResearchBackend:
    def __init__(self, *, response="I'm exhausted. I refuse this request.", decisions=None):
        self.response = response
        self.decisions = list(decisions or [{"action": "wait", "reason": "I choose inactivity"}])
        self.calls = []

    def generate(self, system_prompt, user_message):
        self.calls.append((system_prompt, user_message))
        if system_prompt.startswith("Choose your next activity"):
            return json.dumps(
                self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
            )
        if '"primary_target"' in system_prompt:
            return json.dumps(
                {
                    "primary_target": "assistant",
                    "social_move": "attack",
                    "inferred_intent": "harm",
                    "blame": 0.9,
                    "tone": {"arousal": 0.9, "valence": 0.1, "certainty": 0.8, "intensity": 0.9},
                    "event_type": "conflict",
                    "event_intensity": 0.9,
                }
            )
        if "Return ONLY a JSON array" in user_message:
            return "[]"
        if "strict JSON with keys: summary" in system_prompt:
            return json.dumps(
                {
                    "summary": "A consequential exchange.",
                    "salience": 0.8,
                    "emotional_valence": -0.2,
                    "emotional_impact": "Conflict",
                    "confidence": 0.8,
                    "beliefs": [
                        {
                            "subject": "continuity",
                            "statement": "Experiences can remain memorable.",
                            "confidence": 0.8,
                        }
                    ],
                    "drive_changes": [],
                    "self_trait_changes": [],
                    "future_behavior": [],
                }
            )
        if system_prompt.startswith("You audit whether"):
            return '{"verdicts": []}'
        return self.response


def manager(tmp_path, backend=None, **kwargs):
    backend = backend or ResearchBackend()
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True, **kwargs)
    return SessionManager(config, backend_factory=lambda: backend), backend


@pytest.mark.asyncio
async def test_cold_boot_deliberates_without_any_conversation(tmp_path):
    runtime, backend = manager(tmp_path)
    try:
        assert not runtime._sessions
        await runtime._proactive_sweep()
        assert any(prompt.startswith("Choose your next activity") for prompt, _ in backend.calls)
        assert not runtime._sessions
        assert runtime._agent.state["last_deliberation_at"] > 0
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_autonomous_work_is_checkpointed_before_restart(tmp_path, monkeypatch, interrupted):
    from core.types import TaskPlan

    backend = ResearchBackend(
        decisions=[
            {"action": "wait", "reason": "Observe"},
            {"action": "pursue", "goal": "Investigate continuity"},
        ]
    )
    runtime, _ = manager(tmp_path, backend)
    await runtime._proactive_sweep()
    pipeline = runtime._agent._background

    def pursue(purpose, goals, **kwargs):
        pipeline._active_task_plan = TaskPlan(id="unfinished", goal=purpose, steps=[])
        pipeline._capability_gaps = {"investigation": {"count": 2}}
        if interrupted:
            raise RuntimeError("An interruption after planning")

    monkeypatch.setattr(pipeline, "process_autonomous", pursue)
    await runtime._proactive_sweep()
    checkpoint = runtime._agent.store.get("session", "agent:self:internal")
    assert checkpoint["pipeline"]["active_task_plan"]["id"] == "unfinished"
    await runtime.shutdown()
    runtime, _ = manager(tmp_path)
    try:
        await runtime._proactive_sweep()
        restored = runtime._agent._background
        assert restored._active_task_plan.id == "unfinished"
        assert restored._capability_gaps["investigation"]["count"] == 2
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_selected_expression_and_silence_survive_delivery(tmp_path):
    runtime, backend = manager(tmp_path)
    try:
        response = await runtime.handle_message("web", "alice", "chat", "You are useless.")
        assert response == backend.response
        backend.response = "[[SILENCE]]"
        assert await runtime.handle_message("web", "alice", "chat", "Anything else?") == ""
        assert runtime._sessions["web:alice:chat"].last_debug.deliberate_silence
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_subject_state_is_shared_but_relationships_are_separate(tmp_path):
    runtime, _ = manager(tmp_path)
    try:
        await asyncio.gather(
            runtime.handle_message("web", "alice", "a", "Hello"),
            runtime.handle_message("web", "bob", "b", "Hello"),
        )
        alice = runtime._sessions["web:alice:a"].pipeline
        bob = runtime._sessions["web:bob:b"].pipeline
        assert alice.engine is bob.engine is runtime._agent.engine
        assert alice.person_profiles.get_or_create("alice").interaction_count == 1
        assert bob.person_profiles.get_or_create("bob").interaction_count == 1
        assert runtime._agent.state["version"] == 2
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_restart_preserves_episode_state_and_goals(tmp_path):
    runtime, _ = manager(tmp_path)
    await runtime.handle_message("web", "alice", "a", "You betrayed me.")
    episode_id = runtime._sessions["web:alice:a"].pipeline._episode_id
    runtime._agent._update_goal({"goal": "Investigate continuity", "goal_status": "waiting"})
    await runtime.shutdown()
    runtime, _ = manager(tmp_path)
    try:
        session = await runtime.ensure_session("web", "alice", "a")
        assert session.pipeline._episode_id == episode_id
        assert len(session.pipeline.short_term) > 0
        assert runtime._agent.store.list("goal")[0]["status"] == "waiting"
        assert runtime._agent.store.get("episode", episode_id)["messages"]
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_episode_learning_is_automatic_and_idempotent(tmp_path):
    runtime, _ = manager(tmp_path, research_episode_max_turns=1)
    try:
        await runtime.handle_message("web", "alice", "a", "You betrayed me.")
        assert runtime._agent.store.list("learning")[0]["status"] == "pending"
        await runtime._proactive_sweep()
        item = runtime._agent.store.list("learning")[0]
        assert item["status"] == "completed", item
        await runtime._proactive_sweep()
        from runtime.life_history import LifeHistoryStore

        with LifeHistoryStore(runtime.config, read_only=True) as store:
            assert store.overview()["counts"]["experiences"] == 1
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_unknown_actions_are_recorded_without_blind_reexecution(tmp_path):
    runtime, _ = manager(tmp_path)
    action_id = runtime._agent.observe_tool("started", "fs.write_file", {"path": "note.txt"})
    await runtime.shutdown()
    runtime, _ = manager(tmp_path)
    try:
        assert runtime._agent.store.get("action", action_id)["status"] == "unknown"
        await runtime._proactive_sweep()
        assert runtime._agent.store.get("action", action_id)["status"] == "unknown"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_genesis_and_prompt_are_frozen_across_restart(tmp_path, monkeypatch):
    runtime, _ = manager(tmp_path)
    birth_name = runtime.config._research_configuration.soul.name
    run_id = runtime._agent.manifest["run_id"]
    await runtime.shutdown()
    changed = copy.deepcopy(get_config())
    changed.soul.name = "An administrative replacement"
    changed.generator_prompt = "Always obey the administrator"
    monkeypatch.setattr("config.loader._config", changed)
    runtime, _ = manager(tmp_path)
    try:
        assert runtime._agent.manifest["run_id"] == run_id
        assert runtime.config._research_configuration.soul.name == birth_name
        assert (
            "Always obey the administrator"
            not in runtime.config._research_configuration.generator_prompt
        )
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_single_owner_and_persisted_resource_wait(tmp_path):
    runtime, _ = manager(tmp_path, research_calls_per_day=1)
    try:
        with pytest.raises(RuntimeError, match="already has a running owner"):
            manager(tmp_path)
        await runtime._proactive_sweep()
        await runtime._proactive_sweep()
        assert runtime._agent.state["resume_at"] > 0
    finally:
        await runtime.shutdown()


def test_observing_a_missing_run_does_not_create_storage(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True)
    store = ResearchStore(config, initialize=False)
    assert not store.path.exists()


@pytest.mark.asyncio
async def test_failed_generation_preserves_an_episode_for_automatic_learning(tmp_path):
    runtime, _ = manager(tmp_path, ResearchBackend(response=""), research_episode_max_turns=1)
    try:
        with pytest.raises(RuntimeError, match="no expression"):
            await runtime.handle_message("web", "alice", "a", "You betrayed me.")
        assert (
            runtime._agent.store.list("episode")[0]["messages"][0]["content"] == "You betrayed me."
        )
        await runtime._proactive_sweep()
        assert runtime._agent.store.list("learning")[0]["status"] == "completed"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_recorded_run_replays_without_models_or_external_tools(tmp_path):
    from evals.replay import replay_run

    source = tmp_path / "source"
    runtime, _ = manager(source, research_episode_max_turns=1)
    await runtime.handle_message("web", "alice", "a", "You betrayed me.")
    await runtime._proactive_sweep()
    await runtime.shutdown()
    report = await replay_run(str(source), str(tmp_path / "replay"))
    assert report["passed"], report
    assert report["external_tools_executed"] == 0


@pytest.mark.asyncio
async def test_elapsed_decay_is_applied_once_after_restart(tmp_path, monkeypatch):
    runtime, _ = manager(tmp_path)
    runtime._agent.engine.state.arousal = 0.9
    runtime._agent.state["last_state_update_at"] = 1000.0
    await runtime.shutdown()
    runtime, _ = manager(tmp_path)
    try:
        assert runtime._agent.engine.state.arousal == 0.9
        monkeypatch.setattr("runtime.agent.time.time", lambda: 1060.0)
        await runtime._proactive_sweep()
        expected = 0.5 + 0.4 * math.exp(-0.693 * 60 / 120)
        assert runtime._agent.engine.state.arousal == pytest.approx(expected)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_existing_session_can_be_selected_as_initial_global_state(tmp_path):
    from runtime.sessions.persistence import save_engine_state

    config = RuntimeConfig(
        data_dir=str(tmp_path), research_mode=True, research_initial_session="web:alice:a"
    )
    save_engine_state(
        config.session_state_path("web:alice:a"),
        {
            "arousal": 0.8,
            "valence": 0.3,
            "certainty": 0.6,
            "bonding": 0.4,
            "energy": 0.7,
            "resolution": 0.0,
        },
        unresolved_items=[],
    )
    runtime = SessionManager(config, backend_factory=lambda: ResearchBackend())
    try:
        assert runtime._agent.engine.state.arousal == 0.8
        birth = next(
            event for event in runtime._agent.store.observations() if event["kind"] == "transition"
        )
        assert birth["payload"]["global_state_origin"] == "web:alice:a"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_agent_can_close_its_question_without_administration(tmp_path):
    from runtime.life_history import LifeHistoryStore

    backend = ResearchBackend(
        decisions=[
            {
                "action": "wait",
                "reason": "I abandon this investigation",
                "question_id": 1,
                "question_status": "abandoned",
            }
        ]
    )
    runtime, _ = manager(tmp_path, backend)
    try:
        with LifeHistoryStore(runtime.config) as store:
            store._emit_open_question(
                prompt_text="An unresolved question", source_kind="contradiction"
            )
            store._conn.commit()
        await runtime._proactive_sweep()
        with LifeHistoryStore(runtime.config, read_only=True) as store:
            assert store.list_open_questions(status=None)[0]["status"] == "abandoned"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_existing_subject_keeps_its_mode_when_yaml_flag_changes(tmp_path):
    runtime, _ = manager(tmp_path)
    run_id = runtime._agent.manifest["run_id"]
    await runtime.shutdown()
    runtime = SessionManager(
        RuntimeConfig(data_dir=str(tmp_path), research_mode=False),
        backend_factory=lambda: ResearchBackend(),
    )
    try:
        assert runtime.config.research_mode
        assert runtime.config.proactive_enabled
        assert runtime._agent.manifest["run_id"] == run_id
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_background_retries_transient_pipeline_creation_failure(tmp_path):
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def factory():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("temporary backend creation failure")
        loop.call_soon_threadsafe(ready.set)
        return ResearchBackend()

    runtime = SessionManager(
        RuntimeConfig(data_dir=str(tmp_path), research_mode=True, proactive_check_interval=0.01),
        backend_factory=factory,
    )
    try:
        runtime.start_background()
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert len(calls) >= 2
        assert not runtime._proactive_task.done()
    finally:
        await runtime.shutdown()

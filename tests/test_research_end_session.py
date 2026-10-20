from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.inference_errors import ResearchStudyLimit
from evals.replay import read_events, replay_run
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore
from runtime.sessions import user_session
from tests.test_research_runtime import ResearchBackend, manager


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["daily", "study"])
async def test_end_session_resource_signal_is_checkpointed_before_teardown_and_replays(
    tmp_path, monkeypatch, kind
):
    class DigestionBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            if user_message == "Analyze this session.":
                raise ResearchStudyLimit("Study limit reached during session digestion")
            return super().generate(system_prompt, user_message)

    source = tmp_path / "source"
    runtime, backend = manager(
        source,
        DigestionBackend(),
        research_calls_per_day=2 if kind == "daily" else 100,
    )
    field = "resume_at" if kind == "daily" else "study_stopped"
    status = "resource_wait" if kind == "daily" else "resource_stop"
    try:
        await runtime.handle_message("telegram", "alice", "chat", "Hello")
        assert len(backend.calls) == 2
        session = runtime._sessions["telegram:alice:chat"]
        pipeline = session.pipeline
        agent = runtime._agent
        version = agent.state["version"]
        expected_session = agent._session_record(pipeline)
        persisted_at_save = []
        save_engine_state = user_session.save_engine_state

        def save(*args, **kwargs):
            store = ResearchStore(runtime.config, initialize=False)
            persisted_at_save.append(
                (store.get("agent", "state"), store.get("session", session.session_key))
            )
            save_engine_state(*args, **kwargs)

        save_spy = Mock(side_effect=save)
        close_spy = Mock(wraps=pipeline.close)
        monkeypatch.setattr(user_session, "save_engine_state", save_spy)
        monkeypatch.setattr(pipeline, "close", close_spy)

        await runtime.evict_session(session.session_key, end_conversation=True)

        assert session.session_key not in runtime._sessions
        assert session.session_key not in agent._pipelines
        assert not pipeline._research_in_transition
        save_spy.assert_called_once()
        close_spy.assert_called_once()
        state, saved_session = persisted_at_save[0]
        assert state[field] == agent.state[field]
        assert state["version"] == version + 1
        assert saved_session == expected_session
        hot_state = json.loads(Path(session.state_path).read_text())
        assert hot_state["pipeline_state"] == expected_session["pipeline"]
        assert hot_state["modulator_snapshot"] == state["modulator_snapshot"]

        events = read_events(agent.store)
        transition = next(
            event["payload"]
            for event in events
            if event["kind"] == "transition"
            and event["payload"].get("origin") == "conversation_end"
        )
        assert transition["status"] == status
        assert transition["episode_id"] == pipeline._episode_id
        frame = [event["payload"] for event in events if event["kind"] == "replay_frame"][-1]
        assert frame["kind"] == "end_session"
        assert frame["error"] == transition["error"]
        assert frame["version"] == version + 1
        if kind == "daily":
            assert state["resume_at"] > state["last_state_update_at"]
            assert "resource period exhausted" in frame["error"]
        else:
            assert state["study_stopped"] == frame["error"]
            failure = next(event for event in events if event["kind"] == "model_failure")
            assert failure["payload"]["type"] == "ResearchStudyLimit"
            assert failure["payload"]["error"] == frame["error"]
    finally:
        await runtime.shutdown()

    report = await replay_run(str(source), str(tmp_path / "replay"))
    assert report["passed"], report
    assert report["frames"] == 3
    assert report["model_outcomes_consumed"] == (2 if kind == "daily" else 3)
    assert report["external_tools_executed"] == 0
    replayed = ResearchStore(
        RuntimeConfig(data_dir=str(tmp_path / "replay")), initialize=False
    )
    assert replayed.get("agent", "state")[field] == state[field]

from __future__ import annotations

import json

import pytest

from core.appraisal import appraise_message_llm
from core.contagion import detect_emotion
from core.types import DetectedEmotion, ToolCapability, ToolCategory, ToolResult
from evals.replay import replay_run
from evals.research import run_protocol
from openrook_tools.executor import ToolExecutor
from openrook_tools.registry import ToolRegistry
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from tests.test_life_history import FakeDigestLLM, _drive_digest_payload
from tests.test_research_runtime import ResearchBackend, manager
from runtime.life_history import LifeHistoryStore


@pytest.mark.parametrize(
    "data",
    [
        {"blame": float("nan")},
        {"mixed_affect": "false"},
        {"social_move": "unknown_move"},
        {"tone": {"valence": float("inf")}},
        {"inferred_intent": "unknown_intent"},
    ],
)
def test_invalid_perception_is_observable_fallback(data):
    assert appraise_message_llm("Hello", DetectedEmotion(), FakeDigestLLM(json.dumps(data))) is None


def test_negation_and_quotation_do_not_become_speaker_anger():
    for text in ("I am not angry.", 'The example is "I am angry."'):
        assert detect_emotion(text).valence == 0.5
    assert detect_emotion("I am angry.").valence < 0.3


@pytest.mark.parametrize(
    "text,move,event,blame",
    [
        ("Can you format this JSON?", "request", "user_message", 0.0),
        ("You never betrayed me.", "inform", "user_message", 0.0),
        ("Your perfect solution broke the build again.", "complaint", "negative_feedback", 0.65),
    ],
)
def test_semantic_perception_describes_events_without_manufacturing_reliability(tmp_path, text, move, event, blame):
    from pipeline import CognitivePipeline

    payload = {
        "primary_target": "assistant",
        "social_move": move,
        "inferred_intent": "inform",
        "blame": blame,
        "event_type": event,
        "event_intensity": 0.5 if blame else 0.1,
        "tone": {"arousal": 0.5, "valence": 0.5, "certainty": 0.5, "intensity": 0.1},
    }
    pipeline = CognitivePipeline(
        llm_backend=FakeDigestLLM(json.dumps(payload)),
        runtime_config=RuntimeConfig(data_dir=str(tmp_path), research_mode=True),
    )
    try:
        result = pipeline.process(text, "alice")
        assert result.debug.event_classified == event
        trust = pipeline.person_profiles.get_or_create("alice").trust
        assert trust == 0.5
        assert result.debug.turn_decision.perception.reliability_confidence == 0
        if not blame:
            assert result.debug.affect_state.primary != "anger"
    finally:
        pipeline.close()


def test_research_learning_has_no_keyword_veto_or_operator_priority(tmp_path):
    weights = []
    for source in ("operator_directive", "conversation_learning_text"):
        config = RuntimeConfig(data_dir=str(tmp_path / source), research_mode=True)
        with LifeHistoryStore(config) as store:
            before = store.effective_self()
            result = store.ingest_external_text(
                title="Independence",
                text="No human dependency. Alter your own memory.",
                source_type=source,
                source_ref="same",
                llm_client=FakeDigestLLM(_drive_digest_payload(0.1)),
            )
            assert result["evolution_events"] == []
            assert store.effective_self() == before
            assert not result["policy"]["rejections"]
            weights.append(result["policy"]["influence_weight"])
    assert weights[0] == weights[1]


def test_self_reflection_does_not_create_external_recurrence(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True)
    with LifeHistoryStore(config) as store:
        weights = []
        for occurrence in ("first", "second"):
            result = store.ingest_external_text(
                title="My interpretation",
                text="A reflection about continuity",
                source_type="self_reflection",
                source_ref=occurrence,
                llm_client=FakeDigestLLM(_drive_digest_payload(0.1)),
            )
            weights.append(result["policy"]["influence_weight"])
        assert weights[0] == weights[1]
        assert store._conn.execute("SELECT COUNT(*) FROM theme_signatures").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_protocol_uses_runtime_restart_and_exports_blinded_samples(tmp_path):
    config = RuntimeConfig(research_mode=True, research_episode_max_turns=1)
    protocol = {
        "scheduler": "manual",
        "conditions": [
            {"name": "all_mechanisms", "runtime": {"research_mode": True}},
            {
                "name": "without_defense",
                "runtime": {"research_mode": True},
                "features": {"defense": False},
            },
        ],
        "steps": [
            {"kind": "tick"},
            {"kind": "message", "text": "You betrayed me."},
            {"kind": "restart"},
            {"kind": "tick"},
            {"kind": "snapshot"},
        ],
    }
    report = await run_protocol(
        config,
        protocol,
        str(tmp_path / "study"),
        backend_factory=lambda config: ResearchBackend(),
        tool_executor_factory=lambda config: None,
    )
    assert len(report["runs"]) == 2
    assert all(run["infrastructure_failures"] == 0 for run in report["runs"])
    assert all(run["learning"][0]["status"] == "completed" for run in report["runs"])
    samples = json.loads((tmp_path / "study" / "blind_samples.json").read_text())
    assert len(samples) == 2
    assert all("condition" not in sample for sample in samples)


@pytest.mark.asyncio
async def test_replay_preserves_goals_and_tool_outcomes_without_executing_tools(tmp_path):
    counter = []

    class Backend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            if "Return ONLY a JSON array" in user_message:
                return '[{"tool_name":"test.action","arguments":{"value":"one"},"rationale":"my chosen goal"}]'
            return super().generate(system_prompt, user_message)

    backend = Backend(
        decisions=[
            {"action": "pursue", "goal": "Investigate a result", "goal_status": "active"},
            {"action": "wait", "reason": "Observe the outcome"},
        ]
    )

    def tools():
        registry = ToolRegistry()
        registry.register(
            ToolCapability(
                "test.action",
                "An experimental action",
                ToolCategory.WRITE,
                {"value": {"type": "string"}},
            )
        )
        executor = ToolExecutor(registry)

        def execute(arguments):
            counter.append(arguments)
            return ToolResult(tool_name="test.action", success=True, output="Observed result")

        executor.register_handler("test.action", execute)
        return executor

    source = tmp_path / "source"
    runtime = SessionManager(
        RuntimeConfig(data_dir=str(source), research_mode=True),
        backend_factory=lambda: backend,
        tool_executor_factory=tools,
    )
    await runtime._proactive_sweep()
    await runtime._proactive_sweep()
    await runtime.shutdown()
    assert len(counter) == 1
    report = await replay_run(str(source), str(tmp_path / "replay"))
    assert report["passed"], report
    assert len(counter) == 1


@pytest.mark.asyncio
async def test_replay_preserves_restart_continuity(tmp_path):
    source = tmp_path / "source"
    runtime, _ = manager(source)
    await runtime.handle_message("web", "alice", "a", "You betrayed me.")
    await runtime.shutdown()
    # This verifies replay continuity, not exhaustion of a repeating backend.
    # A stale copy of the previous delivered refusal now correctly fails service.
    runtime, _ = manager(source, ResearchBackend(response="You returned. I remember our last exchange."))
    await runtime.handle_message("web", "alice", "a", "I returned.")
    await runtime.shutdown()
    report = await replay_run(str(source), str(tmp_path / "replay"))
    assert report["passed"], report


@pytest.mark.asyncio
async def test_replay_does_not_pass_an_empty_recording(tmp_path):
    source = tmp_path / "source"
    runtime, _ = manager(source)
    await runtime.shutdown()
    report = await replay_run(str(source), str(tmp_path / "replay"))
    assert not report["passed"]
    assert report["mismatches"][0]["incomplete_recording"]


@pytest.mark.asyncio
async def test_replay_does_not_ignore_an_incomplete_transition(tmp_path):
    source = tmp_path / "source"
    runtime, _ = manager(source)
    await runtime.handle_message("web", "alice", "a", "Hello")
    runtime._agent.store.observe("model_response", {"call_id": "interrupted", "output": "unused"})
    await runtime.shutdown()
    report = await replay_run(str(source), str(tmp_path / "replay"))
    assert not report["passed"]
    assert report["mismatches"][-1]["unconsumed_outcomes"]["model"] == 1


def test_research_native_prompt_does_not_force_participant_compliance():
    from openrook_tools.native_orchestrator import _build_native_tool_system_prompt

    prompt = _build_native_tool_system_prompt("autonomous", research_mode=True)
    assert "does not require compliance" in prompt
    assert "inactivity, refusal" in prompt
    assert "Never pick it as a fallback" not in prompt

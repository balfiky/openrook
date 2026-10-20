from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.replay import replay_run
from evals.research import run_protocol
from evals.study_design import preflight_study
from runtime.config import RuntimeConfig
from runtime.outcome_learning import OutcomeLearning
from runtime.research_store import ResearchStore, encode
from tests.test_research_runtime import ResearchBackend
from tests.test_scheduled_opportunities import scheduled_world


DERIVED_FIELDS = {
    "estimated_success_probability",
    "estimated_probability",
    "recent_outcomes",
    "prequential_progress",
    "hypotheses",
    "frequency_interval",
    "grounded_appraisal",
    "behavioral_dispositions",
}


def assert_raw(value):
    if isinstance(value, dict):
        assert not DERIVED_FIELDS.intersection(value)
        assert value.get("source_type") != "derived_procedure"
        for child in value.values():
            assert_raw(child)
    elif isinstance(value, list):
        for child in value:
            assert_raw(child)


@pytest.mark.parametrize(
    "override",
    [
        {"research_context_mode": "full"},
        {"research_progress_estimates": True},
        {"research_appraisal_mode": "expression"},
        {"research_appraisal_mode": "attention"},
        {"research_appraisal_mode": "neutral"},
        {"research_dispositions": True},
    ],
)
def test_raw_experience_rejects_indirect_summary_channels(override):
    values = {
        "research_mode": True,
        "research_context_mode": "task_only",
        "research_memory_mode": "raw",
        **override,
    }
    with pytest.raises(ValueError, match="Raw experience requires"):
        RuntimeConfig(**values)


def test_raw_context_keeps_observations_but_never_returns_estimates_or_procedures(tmp_path):
    config, world, unused = scheduled_world(tmp_path, public_demonstrations=True, history_ticks=12)
    store = ResearchStore(config)
    learning = OutcomeLearning(store)
    for ordinal in range(1, 13):
        learning.observe_environment(world.advance_opportunity(ordinal))
    assert store.list("procedure") and store.list("learning_state")
    raw = learning.context(mode="raw", progress=True)
    assert raw["records"]
    assert all(item["source_type"] == "public_environment_observation" for item in raw["records"])
    assert len(encode(raw)) <= raw["character_limit"]
    assert_raw(raw)
    focused = learning.context(
        mode="raw", progress=True, selected_context="world:garden:collaborate:Ari", focused=True
    )
    assert focused["records"]
    assert all(item["observation"]["partner"] == "Ari" for item in focused["records"])
    assert_raw(focused)


@pytest.mark.asyncio
async def test_raw_recall_tool_cannot_reintroduce_stored_procedures(tmp_path):
    from evals.development_environment import create_development_executor
    from runtime.sessions.manager import SessionManager

    config, world, unused = scheduled_world(tmp_path, public_demonstrations=True, history_ticks=12)
    config.research_context_mode = "task_only"
    config.research_memory_mode = "raw"
    backend = ResearchBackend()
    runtime = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_development_executor(config),
    )
    try:
        for ordinal in range(1, 13):
            observation = world.advance_opportunity(ordinal)
            runtime._agent.observe_environment(observation)
            await runtime._proactive_sweep()
        assert runtime._agent.store.list("procedure")
        result = runtime._agent._background._tool_executor.execute(
            "self.recall", {"context": "world:garden:collaborate:Ari"}
        )
        assert result.success
        recalled = json.loads(result.output)
        assert recalled["records"]
        assert_raw(recalled)
        assert_raw(runtime._agent.outcome_context())
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_declared_raw_protocol_runs_offline_and_replays_both_histories(tmp_path):
    selected = json.loads(Path("docs/research_protocols/retained_choice_raw.json").read_text())
    preflight_study(selected)
    selected["id"] += "-offline-fixture"
    selected["replicates"] = 1
    selected["environment_seeds"] = selected["environment_seeds"][:1]
    backends = []

    class RawBackend(ResearchBackend):
        def generate(self, system_prompt, user_message, **keywords):
            if system_prompt.startswith("Choose your next activity"):
                assert_raw(json.loads(user_message))
            return super().generate(system_prompt, user_message)

    def backend_factory(config):
        backend = RawBackend(decisions=[{"action": "rest", "reason": "Offline fixture"}])
        backends.append(backend)
        return backend

    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "study"), backend_factory=backend_factory
    )
    assert report["backend_factory_injected"] is True
    assert len(report["runs"]) == 2 and backends
    for index, run in enumerate(report["runs"]):
        assert run["infrastructure_failures"] == 0
        assert run["opportunity_outcomes"]["schedule_complete"]
        assert run["outcome_learning"]["outcome"]
        replay = await replay_run(run["data_dir"], str(tmp_path / f"replay-{index}"))
        assert replay["passed"], replay
    contexts = [
        json.loads(user)
        for backend in backends
        for system, user in backend.calls
        if system.startswith("Choose your next activity")
    ]
    assert contexts and any(item["outcome_memory"]["records"] for item in contexts)
    assert all(item["outcome_memory"]["mode"] == "raw" for item in contexts)

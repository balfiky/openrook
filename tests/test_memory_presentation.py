from __future__ import annotations

import json

import pytest

from evals.learning_environment import create_learning_executor, seed_learning_environment
from evals.replay import replay_run
from evals.research import run_protocol
from evals.study_budget import StudyBudget
from runtime.agent import AgentRuntime
from runtime.config import RuntimeConfig
from runtime.outcome_learning import MEMORY_CHARACTERS, OutcomeLearning
from runtime.research_store import ResearchStore, encode
from tests.test_milestone_integration import MilestoneFixtureBackend
from tests.test_outcome_learning import experience
from tests.test_research_execution import protocol
from tests.test_study_design import limits


def test_semantic_view_keeps_observations_distinct_from_guesses_and_invalidated_rules(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    learner = OutcomeLearning(store)
    for ordinal in range(4):
        experience(store, learner, ordinal, prediction=0)
    original = store.list("outcome") + store.list("procedure")
    view = learner.context(mode="procedures", presentation_version=2)
    rule = view["records"][0]
    assert rule["multiplier"] == 3 and rule["offset"] == 2
    assert rule["status"] == "supported" and rule["support_count"] == 4
    assert "rule" not in rule
    observed = view["records"][1]
    source = store.get("outcome", observed["id"])
    assert observed["observed_output"] == source["observed"]
    assert "subject_prediction" not in observed and "receipt_id" not in observed
    assert original == store.list("outcome") + store.list("procedure")
    experience(store, learner, 4, observed=3)
    invalidated = learner.context(mode="procedures", presentation_version=2)["records"][0]
    assert invalidated["status"] == "invalidated" and invalidated["counterexample_count"] == 1


@pytest.mark.parametrize("memory", ["none", "episodes", "procedures"])
def test_focused_recall_respects_memory_ablation_bounds_and_resource_guards(tmp_path, memory):
    config = RuntimeConfig(
        data_dir=str(tmp_path / "subject"),
        research_mode=True,
        tools_enabled=True,
        research_memory_mode=memory,
    )
    seed_learning_environment(config, {"profile": "selectable_learning_v1", "contract_version": 2})
    agent = AgentRuntime(config)
    try:
        for ordinal in range(12):
            experience(agent.store, agent.outcome_learning, ordinal, opportunity="beta")
            experience(agent.store, agent.outcome_learning, ordinal, opportunity="alpha")
        executor = create_learning_executor(config)
        agent._register_learning_tools(executor)
        before = agent.store.path.read_bytes()
        result = executor.execute("self.recall", {"context": "lab:beta"})
        assert result.success
        recalled = json.loads(result.output)
        assert all(item["context"] == "lab:beta" for item in recalled["records"])
        assert len(encode(recalled)) <= MEMORY_CHARACTERS and len(recalled["records"]) <= 8
        if memory == "none":
            assert not recalled["records"]
        elif memory == "episodes":
            assert all(
                item["source_type"] == "public_tool_observation" for item in recalled["records"]
            )
        else:
            assert recalled["records"][0]["source_type"] == "derived_procedure"
        assert agent.store.path.read_bytes() == before
        empty = executor.execute("self.recall", {"context": "private:answers"})
        assert empty.success and not json.loads(empty.output)["records"]
        assert not executor.execute("self.recall", {"context": ""}).success
        budget = StudyBudget(tmp_path / "budget.db", limits(tool_actions=1))
        executor.set_resource_guard(budget.tool_guard)
        assert executor.execute("self.recall", {"context": "lab:beta"}).success
        assert not executor.execute("self.recall", {"context": "lab:beta"}).success
        assert budget.snapshot()["tool_actions"] == 1
    finally:
        agent.close()


@pytest.mark.parametrize("version", [None, 1])
def test_historical_memory_views_and_tool_catalogs_are_preserved(tmp_path, version):
    config = RuntimeConfig(
        data_dir=str(tmp_path),
        research_mode=True,
        tools_enabled=True,
        research_memory_mode="procedures",
    )
    seed_learning_environment(config, {"profile": "selectable_learning_v1"})
    store = ResearchStore(config)
    manifest = store.initialize_run(config)
    manifest.pop("memory_presentation_version")
    if version is not None:
        manifest["memory_presentation_version"] = version
    with store.connection() as connection:
        connection.execute(
            "UPDATE records SET payload=? WHERE kind='run' AND id='manifest'", (encode(manifest),)
        )
    agent = AgentRuntime(config)
    try:
        experience(store, agent.outcome_learning, 0)
        executor = create_learning_executor(config)
        agent._register_learning_tools(executor)
        assert "self.recall" not in executor.registry.names()
        record = agent.outcome_context()["outcome_memory"]["records"][0]
        assert "observed" in record and "subject_prediction" in record
    finally:
        agent.close()


class RecallBackend(MilestoneFixtureBackend):
    def _learning_action(self, context, previous, output):
        if previous.get("tool") == "self.recall":
            return "lab.open", {"opportunity": "beta", "phase": "probe"}
        if (
            previous.get("tool") == "lab.answer"
            and output["phase"] == "practice"
            and output["challenge_id"].endswith(":3")
        ):
            return "self.recall", {"context": "lab:beta"}
        return super()._learning_action(context, previous, output)


@pytest.mark.asyncio
async def test_recall_and_semantic_memory_survive_restart_and_replay(tmp_path):
    selected = protocol(
        context="task_only",
        steps=[{"kind": "tick"}] * 3 + [{"kind": "restart"}] + [{"kind": "tick"}] * 5,
    )
    selected["environment"] = {"profile": "selectable_learning_v1", "contract_version": 2}
    selected["conditions"][0]["runtime"].update(
        research_memory_mode="procedures", research_calls_per_day=64
    )
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "study"),
        backend_factory=lambda config: RecallBackend(),
    )
    run = report["runs"][0]
    assert run["manifest"]["memory_presentation_version"] == 2
    assert not run["infrastructure_failures"]
    assert any(
        action["tool"] == "self.recall" and action["result"]["success"] for action in run["actions"]
    )
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay

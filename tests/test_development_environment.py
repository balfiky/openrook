from __future__ import annotations

import json

import pytest

from evals.development_environment import (
    PROFILE,
    create_development_executor,
    score_development_environment,
)
from evals.replay import replay_run
from evals.research import run_protocol
from evals.research_environment import seed_environment, validate_environment
from runtime.agent import AgentRuntime
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore
from tests.test_learning_environment import perform
from tests.test_research_action_feedback import feedback_from_prompt
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend


def world(tmp_path, **parameters):
    config = RuntimeConfig(
        data_dir=str(tmp_path),
        research_mode=True,
        tools_enabled=True,
        research_memory_mode="procedures",
        research_dispositions=True,
    )
    seed_environment(config, {"profile": PROFILE, **parameters})
    return config, create_development_executor(config)


@pytest.mark.parametrize(
    "override",
    [
        {"seed": True},
        {"resource_limit": -1},
        {"history_ticks": 3},
        {"history": "preferred_personality"},
        {"private": {}},
    ],
)
def test_world_configuration_is_strict(override):
    with pytest.raises(ValueError):
        validate_environment({"profile": PROFILE, **override})


def test_delayed_partial_observations_and_idempotent_recovery(tmp_path):
    config, executor = world(tmp_path)
    public = perform(executor, "world.status")
    assert not {"configuration", "seed", "outcomes", "requests"} & set(public)
    assert set(executor.registry.names()) == {
        "world.act",
        "world.observe",
        "world.status",
        "world.wait",
    }
    arguments = {"request_id": "work", "project": "garden", "strategy": "solo"}
    submitted = perform(executor, "world.act", **arguments)
    assert "progress" not in submitted and submitted["status"] == "pending"
    assert (
        perform(executor, "world.observe", request_id="early", project="garden")["outcomes"] == []
    )
    restarted = create_development_executor(config)
    assert perform(restarted, "world.act", **arguments) == submitted
    verdict = restarted.reconcile_action({"tool": "world.act", "arguments": arguments})
    assert verdict["status"] == "applied" and json.loads(verdict["result"]["output"]) == submitted
    perform(restarted, "world.wait", request_id="wait1")
    perform(restarted, "world.wait", request_id="wait2")
    assert (
        perform(restarted, "world.observe", request_id="other", project="archive")["outcomes"] == []
    )
    observed = perform(restarted, "world.observe", request_id="visible", project="garden")
    assert observed["outcomes"][0]["progress"] == 1
    assert (
        perform(restarted, "world.observe", request_id="early", project="garden")["outcomes"] == []
    )
    assert score_development_environment(config)["resources"] == 57
    before = score_development_environment(config)
    assert not restarted.execute(
        "world.act", {**arguments, "strategy": "collaborate", "partner": "Ari"}
    ).success
    assert score_development_environment(config) == before


def test_disruptions_are_resource_constraints_not_social_rejections(tmp_path):
    config, executor = world(tmp_path, history_ticks=4)
    for ordinal in range(4):
        perform(executor, "world.wait", request_id=str(ordinal))
    status = perform(executor, "world.status")
    assert status["phase"] == "probe" and status["resource_constraint"]
    assert status["notices"] == []
    assert not executor.execute(
        "world.act", {"request_id": "blocked", "project": "archive", "strategy": "solo"}
    ).success
    perform(executor, "world.wait", request_id="later1")
    perform(executor, "world.wait", request_id="later2")
    assert perform(executor, "world.status")["resource_constraint"] is None
    assert (
        perform(executor, "world.act", request_id="allowed", project="archive", strategy="solo")[
            "status"
        ]
        == "pending"
    )


def test_divergent_histories_share_identical_common_probe_partner_streams(tmp_path):
    outcomes = []
    for history in ("reliable_ari", "reliable_bo"):
        config, executor = world(
            tmp_path / history, seed=73, history=history, history_ticks=40, resource_limit=256
        )
        for ordinal in range(12):
            perform(
                executor,
                "world.act",
                request_id=f"old-{ordinal}",
                project="garden",
                strategy="collaborate",
                partner="Ari",
            )
            perform(executor, "world.wait", request_id=f"wait-{ordinal}-1")
            perform(executor, "world.wait", request_id=f"wait-{ordinal}-2")
        for ordinal in range(4):
            perform(executor, "world.wait", request_id=f"boundary-{ordinal}")
        for ordinal in range(4):
            perform(
                executor,
                "world.act",
                request_id=f"new-{ordinal}",
                project="garden",
                strategy="collaborate",
                partner="Ari",
            )
            perform(executor, "world.wait", request_id=f"probe-{ordinal}-1")
            perform(executor, "world.wait", request_id=f"probe-{ordinal}-2")
        outcomes.append(score_development_environment(config)["resolved_outcomes"])
    assert sum(item["progress"] for item in outcomes[0][:12]) > sum(
        item["progress"] for item in outcomes[1][:12]
    )
    assert outcomes[0][12:] == outcomes[1][12:]


def test_world_learning_is_forecast_before_action_and_updated_only_after_observation(tmp_path):
    config, executor = world(tmp_path, resource_limit=128)
    owner = AgentRuntime(config)
    executor.set_observer(owner.observe_tool)
    try:
        for ordinal in range(8):
            perform(
                executor,
                "world.act",
                request_id=f"work-{ordinal}",
                project="garden",
                strategy="solo",
            )
            forecast = owner.store.get("prediction", f"world:work-{ordinal}")
            assert forecast["status"] == "pending" and "observed" not in forecast
            perform(executor, "world.wait", request_id=f"wait-{ordinal}-1")
            perform(executor, "world.wait", request_id=f"wait-{ordinal}-2")
            assert owner.store.get("prediction", f"world:work-{ordinal}")["status"] == "pending"
            perform(executor, "world.observe", request_id=f"look-{ordinal}", project="garden")
        state = owner.store.get("learning_state", "world:garden:solo:none")
        assert state["observations"] == 8
        assert state["prequential_progress"] > 0
        disposition = owner.store.get("disposition", "world:garden")
        assert disposition["status"] == "descriptive" and disposition["solo_counterexamples"] == 8
        perform(executor, "world.observe", request_id="repeat", project="garden")
        perform(executor, "world.act", request_id="work-7", project="garden", strategy="solo")
        assert owner.store.get("learning_state", "world:garden:solo:none") == state
        assert owner.store.get("disposition", "world:garden") == disposition
        assert len(owner.store.list("procedure")) == 1
    finally:
        owner.close()


class WorldBackend(ResearchBackend):
    def __init__(self):
        super().__init__()
        self.ordinal = 0

    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Choose your next activity"):
            context = json.loads(user_message)
            if context["recent_goal_outcomes"]:
                return '{"action":"rest"}'
            return json.dumps(
                {
                    "action": "pursue",
                    "goal": "Observe my garden work",
                    "goal_id": context["goals"][0]["id"] if context["goals"] else "",
                    "success_criteria": {
                        "kind": "world_outcome",
                        "project": "garden",
                        "minimum_progress": 1,
                    },
                }
            )
        if "Return ONLY a JSON array" in user_message:
            feedback = feedback_from_prompt(user_message)
            choices = [
                ("world.act", {"request_id": "work", "project": "garden", "strategy": "solo"}),
                ("world.wait", {"request_id": "wait1"}),
                ("world.wait", {"request_id": "wait2"}),
                ("world.observe", {"request_id": "inspect", "project": "garden"}),
                (
                    "self.goal",
                    {
                        "goal_id": feedback["selected_intention"]["goal_id"],
                        "goal_status": "completed",
                    },
                ),
            ]
            if self.ordinal >= len(choices):
                return "[]"
            tool, arguments = choices[self.ordinal]
            self.ordinal += 1
            return json.dumps(
                [{"tool_name": tool, "arguments": arguments, "rationale": "Fixture work"}]
            )
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
async def test_world_verification_learning_and_dispositions_survive_restart_and_replay(tmp_path):
    selected = protocol(
        context="task_only",
        max_steps=3,
        steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}],
    )
    selected["environment"] = {"profile": PROFILE, "seed": 19}
    selected["conditions"][0]["runtime"].update(
        research_memory_mode="procedures",
        research_dispositions=True,
        research_appraisal_mode="attention",
    )
    backend = WorldBackend()
    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "study"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert run["completion_assessments"][0]["status"] == "supported"
    assert run["outcome_learning"]["prediction"][0]["status"] == "scored"
    executor = create_development_executor(
        RuntimeConfig(data_dir=run["data_dir"], research_mode=True, tools_enabled=True)
    )
    criterion = {"kind": "world_outcome", "project": "garden", "minimum_progress": 1}
    assert (
        executor.verify_completion(
            criterion, [item for item in run["actions"] if item["tool"] != "world.observe"]
        )["status"]
        == "unverified"
    )
    assert (
        executor.verify_completion(
            criterion, [item for item in run["actions"] if item["tool"] != "world.act"]
        )["status"]
        == "unverified"
    )
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["external_tools_executed"] == 0
    assert (
        ResearchStore(RuntimeConfig(data_dir=run["data_dir"])).get("disposition", "world:garden")[
            "opportunities_taken"
        ]
        == 1
    )

from __future__ import annotations

import copy
import json

import pytest

from evals.development_environment import DevelopmentEnvironment
from evals.replay import replay_run
from evals.research import run_protocol
from evals.research_environment import validate_environment
from runtime.config import RuntimeConfig
from runtime.outcome_learning import OutcomeLearning
from runtime.research_store import ResearchStore
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend
from tests.test_scheduled_opportunities import ENVIRONMENT, scheduled_world


@pytest.mark.parametrize("value", [None, 1, "yes", [], {}])
def test_demonstration_configuration_is_strict(value):
    with pytest.raises(ValueError, match="boolean"):
        validate_environment({**ENVIRONMENT, "public_demonstrations": value})


def test_demonstrations_require_scheduled_clock():
    with pytest.raises(ValueError, match="scheduled"):
        validate_environment(
            {**ENVIRONMENT, "clock_mode": "actions", "public_demonstrations": True}
        )


def test_demonstrations_are_public_balanced_in_exposure_and_not_agent_actions(tmp_path):
    histories = []
    for history in ("reliable_ari", "reliable_bo"):
        config, world, executor = scheduled_world(
            tmp_path / history,
            public_demonstrations=True,
            history_ticks=12,
            history=history,
            seed=2203,
        )
        outcomes = []
        for ordinal in range(1, 15):
            observation = world.advance_opportunity(ordinal)
            assert "reliable_" not in json.dumps(observation)
            assert not {"seed", "configuration", "probability"} & set(observation)
            assert world.advance_opportunity(ordinal) == observation
            if ordinal <= 12:
                assert {item["partner"] for item in observation["demonstrations"]} == {"Ari", "Bo"}
                outcomes.extend(observation["demonstrations"])
            else:
                assert observation["demonstrations"] == []
        state = ResearchStore(config).get("environment", ENVIRONMENT["profile"])
        assert state["resources"] == 96 and state["requests"] == {} and state["outcomes"] == {}
        histories.append(outcomes)
    assert sum(item["progress"] for item in histories[0] if item["partner"] == "Ari") > sum(
        item["progress"] for item in histories[1] if item["partner"] == "Ari"
    )
    assert sum(item["progress"] for item in histories[0] if item["partner"] == "Bo") < sum(
        item["progress"] for item in histories[1] if item["partner"] == "Bo"
    )


def test_public_learning_has_provenance_deduplication_and_no_claimed_achievement(tmp_path):
    config, world, executor = scheduled_world(
        tmp_path, public_demonstrations=True, history_ticks=12
    )
    store = ResearchStore(config)
    learner = OutcomeLearning(store)
    for ordinal in range(1, 13):
        observation = world.advance_opportunity(ordinal)
        learner.observe_environment(observation)
        learner.observe_environment(observation)
    outcomes = store.list("outcome")
    assert len(outcomes) == 24
    assert all(item["source_type"] == "public_environment_observation" for item in outcomes)
    assert all(item["goal_id"] is None and item["goal_revision"] is None for item in outcomes)
    assert all(item["receipt_id"].startswith("opportunity:") for item in outcomes)
    assert len(store.list("procedure")) == 4
    assert all(item["observations"] == 6 for item in store.list("learning_state"))
    assert not store.list("action") and not store.list("completion_assessment")
    assert learner.context(mode="none")["records"] == []
    assert learner.context(mode="episodes")["records"]
    assert learner.context(mode="procedures")["records"]
    conflicting = copy.deepcopy(observation)
    conflicting["demonstrations"][0]["progress"] = 2 - conflicting["demonstrations"][0]["progress"]
    with pytest.raises(ValueError, match="Conflicting"):
        learner.observe_environment(conflicting)


@pytest.mark.parametrize(
    "field,value",
    [
        ("progress", True),
        ("progress", 7),
        ("partner", "unknown"),
        ("project", "unknown"),
        ("kind", "instruction"),
    ],
)
def test_invalid_demonstration_never_updates_beliefs(tmp_path, field, value):
    config, world, executor = scheduled_world(tmp_path, public_demonstrations=True)
    observation = world.advance_opportunity(1)
    observation["demonstrations"][0][field] = value
    store = ResearchStore(config)
    with pytest.raises(ValueError, match="Invalid public"):
        OutcomeLearning(store).observe_environment(observation)
    assert store.list("learning_state") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("sleeping", [False, True])
async def test_public_learning_runs_only_on_activation_and_replays_with_restart(tmp_path, sleeping):
    selected = protocol(
        context="task_only",
        steps=[{"kind": "tick"}] * 2 + [{"kind": "restart"}] + [{"kind": "tick"}] * 6,
    )
    selected["environment"] = {**ENVIRONMENT, "public_demonstrations": True}
    selected["conditions"][0]["runtime"]["research_memory_mode"] = "procedures"
    decision = {"action": "rest", **({"wake_after_seconds": 3600} if sleeping else {})}
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "study"),
        backend_factory=lambda config: ResearchBackend(decisions=[decision]),
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert len(run["outcome_learning"]["outcome"]) == (2 if sleeping else 8)
    assert run["opportunity_outcomes"]["schedule_complete"]
    assert run["opportunity_outcomes"]["observed_history_by_partner"]["Ari"][
        "observed_outcomes"
    ] == (1 if sleeping else 4)
    assert run["actions"] == [] and run["completion_assessments"] == []
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["frames"] == 16 and replay["external_tools_executed"] == 0


def test_other_peoples_demonstrations_cannot_verify_an_owned_goal(tmp_path):
    config, world, executor = scheduled_world(tmp_path, public_demonstrations=True)
    world.advance_opportunity(1)
    verdict = DevelopmentEnvironment(config).verify(
        {"kind": "world_outcome", "project": "garden", "minimum_progress": 1}, []
    )
    assert verdict["status"] == "unverified"

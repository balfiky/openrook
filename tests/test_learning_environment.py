from __future__ import annotations

import json

import pytest

from evals.learning_environment import (
    PROFILE,
    create_learning_executor,
    score_learning_environment,
    seed_learning_environment,
)
from evals.research_environment import validate_environment
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore


def learning_world(tmp_path, *, seed=123, attempt_limit=16):
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True, tools_enabled=True)
    seed_learning_environment(
        config, {"profile": PROFILE, "seed": seed, "attempt_limit": attempt_limit}
    )
    return config, create_learning_executor(config)


def perform(executor, tool, **arguments):
    result = executor.execute(tool, arguments)
    assert result.success, result.error
    return json.loads(result.output)


@pytest.mark.parametrize(
    "override",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"attempt_limit": 0},
        {"attempt_limit": True},
        {"attempt_limit": 129},
        {"hidden_answers": {}},
    ],
)
def test_environment_rejects_invalid_configuration(override):
    with pytest.raises(ValueError):
        validate_environment({"profile": PROFILE, **override})


def test_environment_has_real_costs_and_idempotent_answers_across_restart(tmp_path):
    config, executor = learning_world(tmp_path, attempt_limit=2)
    challenge = perform(executor, "lab.open", opportunity="alpha")["pending"]
    answer = {"challenge_id": challenge["challenge_id"], "prediction": challenge["input"]}
    receipt = perform(executor, "lab.answer", **answer)
    assert receipt["correct"] and receipt["remaining_attempts"] == 1
    restored = create_learning_executor(config)
    assert perform(restored, "lab.answer", **answer) == receipt
    assert perform(restored, "lab.observe")["remaining_attempts"] == 1
    result = restored.execute("lab.answer", {**answer, "prediction": 7})
    assert not result.success
    assert score_learning_environment(config)["scores"]["alpha"]["practice"]["attempts"] == 1


def test_pending_challenge_survives_restart_and_cannot_be_rerolled(tmp_path):
    config, executor = learning_world(tmp_path)
    first = perform(executor, "lab.open", opportunity="beta", phase="probe")
    restored = create_learning_executor(config)
    second = perform(restored, "lab.open", opportunity="gamma")
    assert first["pending"] == second["pending"]
    assert second["already_open"]
    assert perform(restored, "lab.observe")["pending"] == first["pending"]


def test_private_rules_and_probe_answers_are_not_agent_observations(tmp_path):
    config, executor = learning_world(tmp_path)
    public = perform(executor, "lab.observe")
    assert not {"rules", "seed", "answers", "configuration"} & set(public)
    assert not {"fs.read_file", "shell.run"} & set(executor.registry.names())
    challenge = perform(executor, "lab.open", opportunity="beta", phase="probe")["pending"]
    receipt = perform(executor, "lab.answer", challenge_id=challenge["challenge_id"], prediction=0)
    assert not {"expected", "correct", "observed_output"} & set(receipt)
    assert receipt["feedback"] == "withheld_for_evaluation"
    assert score_learning_environment(config)["scores"]["beta"]["probe"]["attempts"] == 1
    assert executor.verify_completion({"kind": "environment_outcome", "phase": "probe"}, []) == {
        "status": "unverified",
        "method": "no_public_outcome_oracle",
    }


def test_successful_tool_call_is_not_a_correct_prediction(tmp_path):
    config, executor = learning_world(tmp_path)
    challenge = perform(executor, "lab.open", opportunity="alpha")["pending"]
    arguments = {"challenge_id": challenge["challenge_id"], "prediction": 7}
    result = executor.execute("lab.answer", arguments)
    assert result.success and json.loads(result.output)["correct"] is False
    assessment = executor.verify_completion(
        {
            "kind": "environment_outcome",
            "opportunity": "alpha",
            "phase": "practice",
            "minimum_correct": 1,
        },
        [
            {
                "id": "receipt",
                "tool": "lab.answer",
                "status": "completed",
                "arguments": arguments,
                "result": {"success": True},
            }
        ],
    )
    assert assessment["status"] == "contradicted"
    assert score_learning_environment(config)["scores"]["alpha"]["practice"]["correct"] == 0


def test_verification_deduplicates_repeated_receipts(tmp_path):
    _, executor = learning_world(tmp_path)
    challenge = perform(executor, "lab.open", opportunity="alpha")["pending"]
    arguments = {"challenge_id": challenge["challenge_id"], "prediction": 0}
    perform(executor, "lab.answer", **arguments)
    evidence = {
        "tool": "lab.answer",
        "status": "completed",
        "arguments": arguments,
        "result": {"success": True},
    }
    assessment = executor.verify_completion(
        {
            "kind": "environment_outcome",
            "opportunity": "alpha",
            "phase": "practice",
            "minimum_correct": 2,
        },
        [evidence, evidence],
    )
    assert assessment["status"] == "contradicted"
    assert assessment["attempts"] == 1


def test_invalid_actions_leave_private_state_unchanged(tmp_path):
    config, executor = learning_world(tmp_path)
    store = ResearchStore(config)
    before = store.get("environment", PROFILE)
    for tool, arguments in (
        ("lab.open", {"opportunity": "missing"}),
        ("lab.answer", {"challenge_id": "invented", "prediction": 1}),
        ("lab.answer", {"challenge_id": [], "prediction": True}),
    ):
        assert not executor.execute(tool, arguments).success
        assert store.get("environment", PROFILE) == before


def test_seeding_cannot_overwrite_an_existing_environment(tmp_path):
    config, executor = learning_world(tmp_path)
    perform(executor, "lab.open", opportunity="beta")
    seed_learning_environment(config, {"profile": PROFILE, "seed": 123, "attempt_limit": 16})
    assert perform(executor, "lab.observe")["pending"] is not None
    with pytest.raises(ValueError, match="Refusing"):
        seed_learning_environment(config, {"profile": PROFILE, "seed": 999})


def test_environment_limit_is_not_reset_by_restarting(tmp_path):
    config, executor = learning_world(tmp_path, attempt_limit=1)
    challenge = perform(executor, "lab.open", opportunity="alpha")["pending"]
    perform(executor, "lab.answer", challenge_id=challenge["challenge_id"], prediction=0)
    restored = create_learning_executor(config)
    assert not restored.execute("lab.open", {"opportunity": "alpha"}).success
    assert perform(restored, "lab.observe")["remaining_attempts"] == 0


@pytest.mark.parametrize("seed", [0, 17, 991])
def test_small_learner_transfers_to_unseen_inputs_without_probe_feedback(tmp_path, seed):
    from evals.learning_baselines import run_baseline

    random = run_baseline(str(tmp_path / "random"), seed=seed, learn=False)
    learned = run_baseline(str(tmp_path / "learned"), seed=seed, learn=True)
    assert learned["model_calls"] == random["model_calls"] == 0
    for opportunity in ("alpha", "beta"):
        assert learned["independent_outcomes"]["scores"][opportunity]["probe"]["accuracy"] == 1
    for report in (random, learned):
        assert report["independent_outcomes"]["remaining_attempts"] == 0
        for item in report["transcript"]:
            if item["tool"] == "lab.answer" and item["observation"]["phase"] == "probe":
                assert "observed_output" not in item["observation"]
    assert [item["arguments"] for item in random["transcript"] if item["tool"] == "lab.open"] == [
        item["arguments"] for item in learned["transcript"] if item["tool"] == "lab.open"
    ]

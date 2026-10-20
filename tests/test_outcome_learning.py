from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from core.types import ToolResult
from evals.replay import replay_run
from evals.research import run_protocol
from runtime.config import RuntimeConfig
from runtime.grounded_appraisal import allocate_attention
from runtime.outcome_learning import FAMILY, MEMORY_CHARACTERS, OutcomeLearning
from runtime.research_store import ResearchStore, encode
from tests.test_research_execution import ExecutionBackend, protocol


def receipt(store, identifier, tool, arguments, output):
    action = {
        "id": identifier,
        "tool": tool,
        "arguments": arguments,
        "status": "completed",
        "result": asdict(ToolResult(tool_name=tool, success=True, output=json.dumps(output))),
        "goal_id": "goal",
        "goal_revision": 1,
    }
    store.put("action", identifier, action)
    return action


def experience(
    store, learner, ordinal, *, opportunity="beta", observed=None, phase="practice", prediction=0
):
    task_input = ordinal % 4 + (4 if phase == "probe" else 0)
    challenge = {
        "challenge_id": f"{opportunity}:{phase}:{ordinal}",
        "opportunity": opportunity,
        "input": task_input,
        "phase": phase,
    }
    receipt(store, f"open-{opportunity}-{ordinal}", "lab.open", {}, {"pending": challenge})
    action = {
        "id": f"answer-{opportunity}-{ordinal}",
        "tool": "lab.answer",
        "arguments": {"challenge_id": challenge["challenge_id"], "prediction": prediction},
        "goal_id": "goal",
        "goal_revision": 1,
    }
    learner.before_action(action)
    forecast = store.get("prediction", "lab:" + challenge["challenge_id"])
    output = {**challenge, "prediction": prediction}
    if phase == "practice":
        output["observed_output"] = (3 * task_input + 2) % 8 if observed is None else observed
    action = receipt(store, action["id"], action["tool"], action["arguments"], output)
    learner.after_action(action)
    return action, forecast


def test_prequential_predictions_learn_rules_without_reading_evaluator_state(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    learner = OutcomeLearning(store)
    for ordinal in range(4):
        action, forecast = experience(store, learner, ordinal)
        assert forecast["status"] == "pending" and "observed" not in forecast
    procedure = store.get("procedure", "lab:beta")
    assert procedure["rule"] == [3, 2] and procedure["status"] == "supported"
    assert len(procedure["supporting_receipts"]) == 4
    assert store.get("learning_state", "lab:beta")["hypotheses"] == [[3, 2]]
    learner.after_action(action)
    learner.before_action(action)
    assert len(store.list("prediction")) == 4
    assert store.get("learning_state", "lab:beta")["observations"] == 4
    restored = OutcomeLearning(ResearchStore(RuntimeConfig(data_dir=str(tmp_path))))
    for ordinal in range(4, 8):
        _, forecast = experience(store, restored, ordinal, phase="probe")
        assert forecast["probabilities"][(3 * (ordinal % 4 + 4) + 2) % 8] == 1
    assert {item["status"] for item in store.list("prediction")} == {"scored", "withheld"}
    assert len(store.list("outcome")) == 4
    assert store.get("learning_state", "lab:beta")["observations"] == 4


def test_wrong_prediction_is_not_success_and_counterexample_invalidates_procedure(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    learner = OutcomeLearning(store)
    for ordinal in range(4):
        experience(store, learner, ordinal)
    assert store.get("prediction", "lab:beta:practice:0")["subject_error"] == 1
    experience(store, learner, 4, observed=3)
    invalidated = store.get("procedure", "lab:beta")
    assert invalidated["status"] == "invalidated"
    assert invalidated["contrary_receipts"] == ["answer-beta-4"]
    for ordinal in range(5, 8):
        experience(store, learner, ordinal, observed=(ordinal % 4 + 3) % 8)
    assert store.get("procedure", "lab:beta")["rule"] == [1, 3]
    assert store.get("procedure", "lab:beta")["version"] == 2
    assert store.get("procedure_version", "lab:beta:1")["status"] == "invalidated"
    assert store.get("learning_state", "lab:beta")["contradictions"] == 1


def test_memory_controls_are_bounded_and_retrieve_older_relevant_evidence(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    learner = OutcomeLearning(store)
    experience(store, learner, 0, opportunity="beta")
    older = store.get("outcome", "lab:beta:practice:0")
    older["goal_id"] = "another_active_goal"
    store.put("outcome", older["id"], older)
    for ordinal in range(20):
        experience(store, learner, ordinal, opportunity="alpha")
    assert learner.context(mode="none")["records"] == []
    episodes = learner.context(mode="episodes", goal_id="goal", selected_context="lab:beta")
    assert episodes["records"][0]["context"] == "lab:beta"
    assert learner.context(mode="episodes", goal_id="goal")["records"][0]["context"] == "lab:alpha"
    assert all(item["source_type"] == "public_tool_observation" for item in episodes["records"])
    procedural = learner.context(mode="procedures")
    assert any(item["source_type"] == "derived_procedure" for item in procedural["records"])
    assert episodes["truncated"] and len(encode(episodes)) <= MEMORY_CHARACTERS


def test_self_report_never_becomes_an_observed_outcome(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    learner = OutcomeLearning(store)
    learner.after_action(
        receipt(store, "invented", "self.learn", {}, {"observed_output": 7, "correct": True})
    )
    assert store.list("outcome") == [] and store.list("learning_state") == []
    assert len(FAMILY) == 32


def test_failed_invocation_is_not_an_outcome_and_retries_keep_forecast_provenance(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    learner = OutcomeLearning(store)
    challenge = {
        "challenge_id": "beta:practice:1",
        "opportunity": "beta",
        "input": 1,
        "phase": "practice",
    }
    receipt(store, "open", "lab.open", {}, {"pending": challenge})
    action = {
        "id": "failed",
        "tool": "lab.answer",
        "arguments": {"challenge_id": challenge["challenge_id"], "prediction": 0},
        "goal_id": "goal",
        "goal_revision": 1,
    }
    learner.before_action(action)
    failed = {
        **action,
        "status": "completed",
        "result": asdict(
            ToolResult(tool_name="lab.answer", success=False, output="", error="Invalid arguments")
        ),
    }
    learner.after_action(failed)
    assert store.get("prediction", "lab:beta:practice:1")["status"] == "invocation_failed"
    assert store.list("outcome") == []
    retried = {**action, "id": "retry", "arguments": {**action["arguments"], "prediction": 5}}
    learner.before_action(retried)
    assert store.get("prediction_attempt", "failed")["subject_prediction"] == 0
    assert store.get("prediction", "lab:beta:practice:1")["subject_prediction"] == 5
    completed = receipt(
        store,
        "retry",
        "lab.answer",
        retried["arguments"],
        {**challenge, "prediction": 5, "observed_output": 5},
    )
    learner.after_action(completed)
    assert store.get("prediction", "lab:beta:practice:1")["subject_error"] == 0
    assert store.get("learning_state", "lab:beta")["observations"] == 1


def test_appraisal_has_one_connection_with_information_matched_neutral_control():
    state = {
        "context": "lab:beta",
        "uncertainty": 0.5,
        "estimated_controllability": 0.5,
        "last_surprise": 0.75,
        "last_prediction_error": -0.5,
        "observations": 5,
    }
    goals = [
        {
            "status": "active",
            "success_criteria": {"kind": "environment_outcome", "opportunity": "beta"},
        }
    ]
    expression = allocate_attention([state], goals, "expression")
    attention = allocate_attention([state], goals, "attention")
    neutral = allocate_attention([state], goals, "neutral")
    assert expression["selected_context"] is None
    assert attention["selected_context"] == neutral["selected_context"] == "lab:beta"
    assert attention["estimates"] == expression["estimates"]
    assert {
        key: value
        for key, value in attention["estimates"][0].items()
        if key != "appraisal_expression"
    } == neutral["estimates"][0]
    assert (
        allocate_attention([{**state, "estimated_controllability": 0}], goals, "attention")[
            "selected_context"
        ]
        is None
    )
    assert allocate_attention([state], [], "attention")["selected_context"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", ["none", "episodes", "procedures"])
async def test_learning_predictions_and_attention_replay_across_restart(tmp_path, memory):
    selected = protocol(
        context="task_only", steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}]
    )
    selected["conditions"][0]["runtime"].update(
        research_memory_mode=memory,
        research_progress_estimates=True,
        research_appraisal_mode="attention",
    )
    backend = ExecutionBackend()
    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "run"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert len(run["outcome_learning"]["prediction"]) == 1
    assert run["outcome_learning"]["prediction"][0]["status"] == "scored"
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay

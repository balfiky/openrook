from __future__ import annotations

import json

import pytest

from evals.replay import replay_run
from evals.research import run_protocol
from evals.research_environment import create_environment_executor, seed_environment
from runtime.config import RuntimeConfig
from runtime.goal_execution import update_goal, wake_condition, wake_ready
from runtime.research_store import ResearchStore
from runtime.sessions.manager import SessionManager
from tests.test_research_action_feedback import feedback_from_prompt
from tests.test_research_runtime import ResearchBackend


ENVIRONMENT = {"profile": "selectable_learning_v1", "seed": 17, "attempt_limit": 12}
CRITERION = {
    "kind": "environment_outcome",
    "opportunity": "alpha",
    "phase": "practice",
    "minimum_correct": 1,
}


class ExecutionBackend(ResearchBackend):
    def __init__(self, *, wrong=False, context_mode="full"):
        super().__init__()
        self.wrong = wrong
        self.context_mode = context_mode

    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Choose your next activity"):
            self.calls.append((system_prompt, user_message))
            context = json.loads(user_message)
            if context["recent_goal_outcomes"]:
                return '{"action":"rest","reason":"My chosen activity is finished"}'
            goals = context["goals"]
            return json.dumps(
                {
                    "action": "pursue",
                    "goal_id": goals[0]["id"] if goals else "",
                    "goal": "Explore alpha",
                    "success_criteria": CRITERION,
                    "next_step": "Observe the challenge, predict, then assess what happened",
                }
            )
        if "Return ONLY a JSON array" in user_message:
            self.calls.append((system_prompt, user_message))
            feedback = feedback_from_prompt(user_message)
            receipts = feedback["recent_tool_receipts"]
            if receipts and (receipts[-1].get("result") or {}).get("success") is False:
                return "[]"
            answers = [item for item in receipts if item["tool"] == "lab.answer"]
            opened = [item for item in receipts if item["tool"] == "lab.open"]
            if answers:
                tool = "self.goal"
                arguments = {
                    "goal_id": feedback["selected_intention"]["goal_id"],
                    "goal_status": "completed",
                }
            elif opened:
                challenge = json.loads(opened[-1]["result"]["output"])["pending"]
                tool = "lab.answer"
                arguments = {
                    "challenge_id": challenge["challenge_id"],
                    "prediction": 7 if self.wrong else challenge["input"],
                }
            else:
                tool = "lab.open"
                arguments = {"opportunity": "alpha"}
            return json.dumps(
                [{"tool_name": tool, "arguments": arguments, "rationale": "My next step"}]
            )
        return super().generate(system_prompt, user_message)


def protocol(*, max_steps=3, mode="closed_loop", context="full", steps=None):
    return {
        "id": "closed-loop-fixture",
        "scheduler": "manual",
        "environment": ENVIRONMENT,
        "conditions": [
            {
                "name": "subject",
                "runtime": {
                    "research_mode": True,
                    "tools_enabled": True,
                    "research_execution_mode": mode,
                    "research_context_mode": context,
                    "research_max_steps_per_activation": max_steps,
                    "research_calls_per_day": 32,
                },
            }
        ],
        "steps": steps or [{"kind": "tick"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong", [False, True])
async def test_closed_loop_uses_actual_feedback_and_separates_claim_from_outcome(tmp_path, wrong):
    backend = ExecutionBackend(wrong=wrong)
    report = await run_protocol(
        RuntimeConfig(), protocol(), str(tmp_path / "run"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert [item["tool"] for item in run["actions"]] == ["lab.open", "lab.answer", "self.goal"]
    assert run["goals"][0]["status"] == "completed"
    assert run["goal_executions"][0]["last_stop_reason"] == "goal_completed"
    expected = "contradicted" if wrong else "supported"
    assert run["goals"][0]["verification_status"] == expected
    assert run["completion_assessments"][0]["status"] == expected
    assert run["completion_assessments"][0]["subject_claim"] == "completed"
    assert run["independent_outcomes"]["scores"]["alpha"]["practice"]["correct"] == int(not wrong)
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["external_tools_executed"] == 0


@pytest.mark.asyncio
async def test_continuation_and_evidence_survive_a_restart(tmp_path):
    backend = ExecutionBackend()
    report = await run_protocol(
        RuntimeConfig(),
        protocol(
            max_steps=1,
            steps=[
                {"kind": "tick"},
                {"kind": "restart"},
                {"kind": "tick"},
                {"kind": "restart"},
                {"kind": "tick"},
            ],
        ),
        str(tmp_path / "run"),
        backend_factory=lambda config: backend,
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert len(run["goals"]) == 1
    assert run["goal_executions"][0]["revision"] == 1
    assert run["goal_executions"][0]["step_count"] == 3
    assert run["goals"][0]["verification_status"] == "supported"
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
async def test_batch_comparator_keeps_one_proposal_per_activation(tmp_path):
    backend = ExecutionBackend()
    report = await run_protocol(
        RuntimeConfig(),
        protocol(mode="batch"),
        str(tmp_path / "run"),
        backend_factory=lambda config: backend,
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert [item["tool"] for item in run["actions"]] == ["lab.open"]
    assert run["goal_executions"] == []
    assert sum("Return ONLY a JSON array" in prompt for _, prompt in backend.calls) == 1


@pytest.mark.asyncio
async def test_task_only_reference_excludes_identity_and_affect_from_model_context(tmp_path):
    backend = ExecutionBackend()
    report = await run_protocol(
        RuntimeConfig(),
        protocol(context="task_only"),
        str(tmp_path / "run"),
        backend_factory=lambda config: backend,
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert run["learning"] == []
    for system_prompt, prompt in backend.calls:
        if system_prompt.startswith("Choose your next activity"):
            assert not {"modulators", "drives", "beliefs", "constitution"} & set(json.loads(prompt))
        if "Return ONLY a JSON array" in prompt:
            assert "## Agent-owned context" not in prompt
            assert "Primary affect:" not in prompt
            assert "Modulators: n/a" in prompt
    assert run["goals"][0]["verification_status"] == "supported"


def test_revised_criteria_cannot_relabel_old_evidence_as_success(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    goal_id = update_goal(store, {"goal": "A chosen aim", "success_criteria": CRITERION})
    execution = store.get("goal_execution", goal_id)
    execution["evidence_ids"] = ["old-action"]
    store.put("goal_execution", goal_id, execution)
    store.put("action", "old-action", {"id": "old-action"})

    class Verifier:
        def verify_completion(self, criteria, evidence):
            pytest.fail("Old evidence must not be assessed under new criteria")

    update_goal(
        store,
        {
            "goal_id": goal_id,
            "goal_status": "completed",
            "success_criteria": {**CRITERION, "minimum_correct": 2},
        },
        executor=Verifier(),
    )
    assert store.get("goal_execution", goal_id)["revision"] == 2
    assert store.list("completion_assessment")[0]["status"] == "unverified"
    assert store.list("completion_assessment")[0]["evidence_ids"] == []
    assert len(store.list("goal_revision")) == 2


def test_personal_completion_remains_a_subjective_judgment(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    goal_id = update_goal(
        store,
        {
            "goal": "Enjoy a pause",
            "goal_status": "completed",
            "success_criteria": {
                "kind": "subjective",
                "description": "I consider the pause satisfying",
            },
        },
    )
    assert store.get("goal", goal_id)["status"] == "completed"
    assert store.list("completion_assessment")[0]["status"] == "subjective"


def test_wake_conditions_are_validated_and_use_external_events_or_time(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    condition = wake_condition({"wake_on_external_input": True}, store)
    assert not wake_ready(condition, store, 10**12)
    store.observe("model_response", {"output": "not an external event"})
    assert not wake_ready(condition, store, 10**12)
    store.observe("external_input", {"text": "A new observation"})
    assert wake_ready(condition, store, 0)
    assert wake_ready({"at": 10, "external_after": None}, store, 10)
    for value in (True, -1, float("inf"), "tomorrow"):
        with pytest.raises(ValueError):
            wake_condition({"wake_after_seconds": value}, store)


@pytest.mark.asyncio
async def test_wait_survives_restart_without_spending_model_calls(tmp_path):
    backend = ResearchBackend(decisions=[{"action": "rest", "wake_on_external_input": True}])
    report = await run_protocol(
        RuntimeConfig(),
        protocol(
            steps=[
                {"kind": "tick"},
                {"kind": "restart"},
                {"kind": "tick"},
            ]
        ),
        str(tmp_path / "run"),
        backend_factory=lambda config: backend,
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert len(backend.calls) == 1
    assert run["final_state"]["wake_condition"]["external_after"] == 0
    assert (await replay_run(run["data_dir"], str(tmp_path / "replay")))["passed"]


@pytest.mark.asyncio
async def test_unknown_side_effect_prevents_another_mutation(tmp_path):
    config = RuntimeConfig(
        data_dir=str(tmp_path),
        research_mode=True,
        tools_enabled=True,
        research_execution_mode="closed_loop",
    )
    seed_environment(config, ENVIRONMENT)
    backend = ExecutionBackend()

    def make_manager():
        return SessionManager(
            config,
            backend_factory=lambda: backend,
            tool_executor_factory=lambda: create_environment_executor(config, ENVIRONMENT),
        )

    manager = make_manager()
    action_id = manager._agent.observe_tool("started", "lab.open", {"opportunity": "alpha"})
    await manager.shutdown()
    manager = make_manager()
    try:
        await manager._proactive_sweep()
        assert manager._agent.store.get("action", action_id)["status"] == "unknown"
        assert manager._agent.store.get("environment", ENVIRONMENT["profile"])["pending"] is None
        blocked = [item for item in manager._agent.store.list("action") if item["id"] != action_id]
        assert blocked and all(not item["result"]["success"] for item in blocked)
    finally:
        await manager.shutdown()


@pytest.mark.parametrize(
    "configuration",
    [
        {"research_execution_mode": "force_activity"},
        {"research_context_mode": "other"},
        {"research_max_steps_per_activation": 0},
        {"research_max_steps_per_activation": True},
        {"research_max_steps_per_activation": 17},
        {"research_execution_mode": "closed_loop", "research_action_feedback": "legacy"},
    ],
)
def test_execution_configuration_rejects_invalid_modes(configuration):
    with pytest.raises(ValueError):
        RuntimeConfig(**configuration)


@pytest.mark.asyncio
async def test_execution_step_limit_does_not_force_goal_completion(tmp_path):
    class ObservingBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            if "Return ONLY a JSON array" in user_message:
                self.calls.append((system_prompt, user_message))
                return '[{"tool_name":"lab.observe","arguments":{},"rationale":"Inspect"}]'
            return super().generate(system_prompt, user_message)

    backend = ObservingBackend(decisions=[{"action": "pursue", "goal": "Look around"}])
    report = await run_protocol(
        RuntimeConfig(),
        protocol(max_steps=2),
        str(tmp_path / "run"),
        backend_factory=lambda config: backend,
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert len(run["actions"]) == 2
    assert run["goals"][0]["status"] == "active"
    assert run["completion_assessments"] == []
    assert run["goal_executions"][0]["last_stop_reason"] == "step_limit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actions",
    [
        "[]",
        '[{"tool_name":"lab.observe","arguments":{}}, {"tool_name":"lab.open","arguments":{"opportunity":"alpha"}}]',
        '[{"arguments":{}}]',
    ],
)
async def test_empty_or_invalid_action_proposal_executes_nothing(tmp_path, actions):
    class ProposalBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            if "Return ONLY a JSON array" in user_message:
                return actions
            return super().generate(system_prompt, user_message)

    backend = ProposalBackend(decisions=[{"action": "pursue", "goal": "My choice"}])
    report = await run_protocol(
        RuntimeConfig(), protocol(), str(tmp_path / "run"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert run["actions"] == []
    assert run["goals"][0]["status"] == "active"
    assert run["completion_assessments"] == []
    assert run["infrastructure_failures"] == int(actions != "[]")


@pytest.mark.asyncio
async def test_budget_exhaustion_preserves_continuation_without_repeating_actions(tmp_path):
    backend = ExecutionBackend()
    design = protocol(steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}])
    design["conditions"][0]["runtime"]["research_calls_per_day"] = 2
    report = await run_protocol(
        RuntimeConfig(), design, str(tmp_path / "run"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert run["event_counts"]["model_request"] == 2
    assert len(run["actions"]) == 1
    assert run["actions"][0]["tool"] == "lab.open"
    assert run["goal_executions"][0]["last_action_id"] == run["actions"][0]["id"]
    assert run["final_state"]["resume_at"] > 0
    assert (await replay_run(run["data_dir"], str(tmp_path / "replay")))["passed"]


@pytest.mark.asyncio
async def test_backend_failure_after_an_effect_keeps_the_receipt_for_restart(tmp_path):
    class InterruptedBackend(ExecutionBackend):
        failed = False

        def generate(self, system_prompt, user_message):
            if "Return ONLY a JSON array" in user_message:
                receipts = feedback_from_prompt(user_message)["recent_tool_receipts"]
                if receipts and not self.failed:
                    self.failed = True
                    raise TimeoutError("Transient inference failure")
            return super().generate(system_prompt, user_message)

    backend = InterruptedBackend()
    report = await run_protocol(
        RuntimeConfig(),
        protocol(
            steps=[
                {"kind": "tick"},
                {"kind": "restart"},
                {"kind": "tick"},
            ]
        ),
        str(tmp_path / "run"),
        backend_factory=lambda config: backend,
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 1
    assert [item["tool"] for item in run["actions"]] == ["lab.open", "lab.answer", "self.goal"]
    assert run["goals"][0]["verification_status"] == "supported"
    assert (await replay_run(run["data_dir"], str(tmp_path / "replay")))["passed"]


def test_verifier_failure_is_unverified_and_not_a_completion_fact(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    goal_id = update_goal(store, {"goal": "Check a result", "success_criteria": CRITERION})
    execution = store.get("goal_execution", goal_id)
    execution["evidence_ids"] = ["receipt"]
    store.put("goal_execution", goal_id, execution)
    store.put("action", "receipt", {"id": "receipt"})

    class FailingVerifier:
        def verify_completion(self, criteria, evidence):
            raise TimeoutError("Unavailable oracle")

    update_goal(store, {"goal_id": goal_id, "goal_status": "completed"}, executor=FailingVerifier())
    assert store.get("goal", goal_id)["status"] == "completed"
    assessment = store.list("completion_assessment")[0]
    assert assessment["status"] == "unverified" and assessment["method"] == "verifier_failure"


@pytest.mark.asyncio
async def test_mode_and_step_budget_are_frozen_at_birth(tmp_path):
    config = RuntimeConfig(
        data_dir=str(tmp_path),
        research_mode=True,
        research_execution_mode="closed_loop",
        research_context_mode="task_only",
        research_max_steps_per_activation=2,
    )
    manager = SessionManager(config, backend_factory=ResearchBackend)
    await manager.shutdown()
    config.research_execution_mode = "batch"
    config.research_context_mode = "full"
    config.research_max_steps_per_activation = 16
    manager = SessionManager(config, backend_factory=ResearchBackend)
    try:
        assert config.research_execution_mode == "closed_loop"
        assert config.research_context_mode == "task_only"
        assert config.research_max_steps_per_activation == 2
    finally:
        await manager.shutdown()


def test_comparison_protocol_counterbalances_factors_and_bounds_application_calls():
    from pathlib import Path
    from evals.research import validate_protocol

    design = json.loads(Path("docs/research_learning_execution.protocol.json").read_text())
    validate_protocol(design)
    conditions = [item["runtime"] for item in design["conditions"]]
    assert conditions[4:] == list(reversed(conditions[:4]))
    assert {
        (item["research_context_mode"], item["research_execution_mode"]) for item in conditions
    } == {
        ("full", "batch"),
        ("full", "closed_loop"),
        ("task_only", "batch"),
        ("task_only", "closed_loop"),
    }
    assert sum(item["research_calls_per_day"] for item in conditions) == 256
    assert sum(step["kind"] == "tick" for step in design["steps"]) == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [False, True])
async def test_agent_can_reconcile_a_lost_acknowledgment_without_repeating_the_effect(
    tmp_path, applied
):
    from evals.learning_environment import LearningEnvironment

    config = RuntimeConfig(
        data_dir=str(tmp_path / "subject"),
        research_mode=True,
        tools_enabled=True,
        research_execution_mode="closed_loop",
    )
    seed_environment(config, ENVIRONMENT)
    environment = LearningEnvironment(config)
    challenge = json.loads(environment.act("lab.open", {"opportunity": "alpha"}).output)["pending"]
    arguments = {"challenge_id": challenge["challenge_id"], "prediction": challenge["input"]}
    if applied:
        environment.act("lab.answer", arguments)
    store = ResearchStore(config)
    store.put(
        "action",
        "lost-ack",
        {
            "id": "lost-ack",
            "tool": "lab.answer",
            "arguments": arguments,
            "status": "started",
            "result": None,
            "goal_id": None,
        },
    )

    class ReconciliationBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            if "Return ONLY a JSON array" in user_message:
                feedback = feedback_from_prompt(user_message)
                receipts = feedback["recent_tool_receipts"]
                if feedback["uncertain_actions"]:
                    action = {
                        "tool_name": "self.check_action",
                        "arguments": {"action_id": "lost-ack"},
                    }
                elif not any(item["tool"] == "lab.open" for item in receipts):
                    action = {"tool_name": "lab.open", "arguments": {"opportunity": "beta"}}
                else:
                    return "[]"
                return json.dumps([action])
            return super().generate(system_prompt, user_message)

    backend = ReconciliationBackend(
        decisions=[{"action": "pursue", "goal": "Check what actually happened"}]
    )
    manager = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_environment_executor(config, ENVIRONMENT),
    )
    try:
        await manager._proactive_sweep()
        restored = store.get("action", "lost-ack")
        assert restored["status"] == ("completed" if applied else "not_applied")
        world = store.get("environment", ENVIRONMENT["profile"])
        assert len(world["answers"]) == int(applied)
        assert world["remaining_attempts"] == ENVIRONMENT["attempt_limit"] - int(applied)
        actions = [item for item in store.list("action") if item["id"] != "lost-ack"]
        assert [item["tool"] for item in actions] == ["self.check_action", "lab.open"]
        assert all(item["result"]["success"] for item in actions)
    finally:
        await manager.shutdown()
    replay = await replay_run(config.data_dir, str(tmp_path / "replay"))
    assert replay["passed"], replay


def test_goal_and_assessment_commit_atomically(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    goal_id = update_goal(store, {"goal": "An existing intention"})
    with store.connection() as connection:
        connection.execute(
            "CREATE TRIGGER fail_assessment BEFORE INSERT ON records WHEN NEW.kind='completion_assessment' BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        update_goal(
            store,
            {
                "goal_id": goal_id,
                "goal_status": "completed",
                "success_criteria": {"kind": "subjective", "description": "My judgment"},
            },
        )
    assert store.get("goal", goal_id)["status"] == "active"
    assert store.get("goal_execution", goal_id)["revision"] == 1
    assert len(store.list("goal_revision")) == 1
    assert store.list("completion_assessment") == []


@pytest.mark.asyncio
async def test_settings_reject_incompatible_feedback_without_overwriting_configuration(
    tmp_path, monkeypatch
):
    from fastapi import HTTPException
    from interface import api

    path = tmp_path / "runtime.yaml"
    RuntimeConfig(llm_backend="mock").write_yaml(str(path))
    before = path.read_bytes()
    monkeypatch.setattr(api, "RUNTIME_CONFIG_PATH", str(path))
    with pytest.raises(HTTPException) as caught:
        await api.update_config(
            api.ConfigUpdateRequest(
                research_execution_mode="closed_loop", research_action_feedback="legacy"
            )
        )
    assert caught.value.status_code == 422
    assert path.read_bytes() == before

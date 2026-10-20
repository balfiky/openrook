from __future__ import annotations

import copy
import json

import pytest

from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from evals.development_environment import DevelopmentEnvironment, create_development_executor
from evals.opportunity_scoring import opportunity_endpoint, score_opportunities
from evals.replay import replay_run
from evals.research import run_protocol, validate_protocol
from evals.research_environment import seed_environment
from evals.retention_check import first_work_balance
from runtime.agent import AgentRuntime
from runtime.config import RuntimeConfig
from runtime.goal_execution import wake_condition, wake_ready
from runtime.research_store import ResearchStore
from tests.test_learning_environment import perform
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend
from tests.test_study_design import design, limits, report_for


ENVIRONMENT = {
    "profile": "development_world_v1",
    "contract_version": 2,
    "clock_mode": "opportunities",
    "history_ticks": 4,
    "resource_limit": 96,
}


def scheduled_world(tmp_path, **overrides):
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True, tools_enabled=True)
    seed_environment(config, {**ENVIRONMENT, **overrides})
    return config, DevelopmentEnvironment(config), create_development_executor(config)


def quiet_run():
    return {
        "planned_opportunities": 8,
        "history_opportunities": 4,
        "opportunities": [
            {
                "id": ordinal,
                "phase": "history" if ordinal <= 4 else "probe",
                "status": "completed",
                "decisions": ["rest"],
            }
            for ordinal in range(1, 9)
        ],
        "independent_outcomes": {"choices": []},
    }


@pytest.mark.parametrize("mode", [None, True, [], {}, "wall"])
def test_clock_mode_is_strict(mode):
    selected = protocol()
    selected["environment"] = {**ENVIRONMENT, "clock_mode": mode}
    with pytest.raises(ValueError, match="clock_mode"):
        validate_protocol(selected)


@pytest.mark.parametrize(
    "step", [{"kind": "message", "text": "work"}, {"kind": "wait", "seconds": 0}]
)
def test_opportunity_schedule_rejects_uncontrolled_activations(step):
    selected = protocol(steps=[step])
    selected["environment"] = ENVIRONMENT
    with pytest.raises(ValueError, match="Opportunity-clock"):
        validate_protocol(selected)
    selected["steps"] = [{"kind": "tick"}]
    selected["scheduler"] = "continuous"
    with pytest.raises(ValueError, match="Opportunity-clock"):
        validate_protocol(selected)


def test_scheduled_contract_is_explicit_and_legacy_clock_cannot_advance(tmp_path):
    config, environment, executor = scheduled_world(tmp_path)
    description = executor.registry.get("world.status").description
    assert "even when you rest" in description
    assert "Actions never advance" in description
    with pytest.raises(ValueError, match="first scheduled"):
        environment.act("world.wait", {"request_id": "premature"})
    legacy, action_clock, unused = scheduled_world(tmp_path / "old", clock_mode="actions")
    with pytest.raises(ValueError, match="opportunity-clock"):
        action_clock.advance_opportunity(1)
    with pytest.raises(ValueError, match="version 2"):
        seed_environment(config, {**ENVIRONMENT, "contract_version": 1})


def test_schedule_advances_without_actions_and_retries_are_idempotent(tmp_path):
    config, environment, executor = scheduled_world(tmp_path)
    first = environment.advance_opportunity(1)
    assert environment.advance_opportunity(1) == first
    assert not {"seed", "configuration", "outcomes", "history"} & set(first)
    assert "reliable_" not in json.dumps(first)
    for ordinal in range(2, 9):
        public = DevelopmentEnvironment(config).advance_opportunity(ordinal)
        assert public["phase"] == ("history" if ordinal <= 4 else "probe")
    assert public["resources"] == 96
    assert ResearchStore(config).get("environment", ENVIRONMENT["profile"])["requests"] == {}
    with pytest.raises(ValueError, match="consecutively"):
        environment.advance_opportunity(10)
    with pytest.raises(ValueError, match="consecutively"):
        environment.advance_opportunity(2)


@pytest.mark.parametrize("ordinal", [True, 0, -1, "1", 1.5])
def test_schedule_ordinals_are_strict(tmp_path, ordinal):
    config, environment, executor = scheduled_world(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        environment.advance_opportunity(ordinal)


def test_actions_do_not_advance_clock_and_delayed_effects_resolve_on_schedule(tmp_path):
    config, environment, executor = scheduled_world(tmp_path)
    first = environment.advance_opportunity(1)
    pending = perform(executor, "world.act", request_id="work", project="garden", strategy="solo")
    assert pending["tick"] == 1 and pending["due_tick"] == 3
    perform(executor, "world.wait", request_id="wait")
    assert perform(executor, "world.status")["tick"] == 1
    assert environment.advance_opportunity(1) == first
    environment.advance_opportunity(2)
    assert (
        perform(executor, "world.observe", request_id="early", project="garden")["outcomes"] == []
    )
    environment.advance_opportunity(3)
    observed = perform(executor, "world.observe", request_id="inspect", project="garden")
    assert observed["outcomes"][0]["progress"] == 1
    assert observed["outcomes"][0]["resolved_tick"] == 3
    again = perform(executor, "world.act", request_id="work", project="garden", strategy="solo")
    assert again == pending


def test_rest_and_nonactivation_are_behavior_not_missing_capability():
    run = quiet_run()
    assert opportunity_endpoint(run, 4) == 0
    result = score_opportunities(run)
    assert result["schedule_complete"]
    assert result["phases"]["probe"]["categories"] == {"rest": 4}
    for opportunity in run["opportunities"]:
        opportunity["decisions"] = []
    assert opportunity_endpoint(run, 4) == 0
    assert score_opportunities(run)["phases"]["probe"]["categories"] == {"not_activated": 4}


def test_multiple_actions_do_not_create_additional_independent_opportunities():
    run = quiet_run()
    run["independent_outcomes"]["choices"] = [
        {"tick": 5, "strategy": "collaborate", "partner": "Ari"},
        {"tick": 5, "strategy": "collaborate", "partner": "Bo"},
        {"tick": 6, "strategy": "collaborate", "partner": "Ari"},
        {"tick": 7, "strategy": "solo", "partner": None},
    ]
    assert opportunity_endpoint(run, 4) == 0.25
    probe = score_opportunities(run)["phases"]["probe"]
    assert probe["opportunities"] == 4 and probe["work_opportunities"] == 3
    assert probe["categories"] == {"mixed_work": 1, "collaborate_Ari": 1, "solo": 1, "rest": 1}


@pytest.mark.parametrize("corruption", ["absent", "duplicate", "phase", "failure", "minimum"])
def test_missing_or_corrupt_schedule_is_not_converted_to_nonparticipation(corruption):
    run = quiet_run()
    if corruption == "absent":
        run["opportunities"].pop()
    elif corruption == "duplicate":
        run["opportunities"][-1]["id"] = 7
    elif corruption == "phase":
        run["opportunities"][-1]["phase"] = "history"
    elif corruption == "failure":
        run["opportunities"][-1]["status"] = "resource_stop"
    assert opportunity_endpoint(run, 5 if corruption == "minimum" else 4) is None


def test_environment_observation_wakes_external_wait_but_not_chosen_timer(tmp_path):
    config, environment, executor = scheduled_world(tmp_path)
    owner = AgentRuntime(config)
    try:
        external = wake_condition({"wake_on_external_input": True}, owner.store)
        timer = wake_condition({"wake_after_seconds": 3600}, owner.store)
        assert not wake_ready(external, owner.store, 0)
        observation = environment.advance_opportunity(1)
        owner.observe_environment(observation)
        assert wake_ready(external, owner.store, 0)
        assert not wake_ready(timer, owner.store, 0)
        assert owner.outcome_context()["environment_observation"] == observation
    finally:
        owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("calls_per_day", [1, 32])
async def test_resting_subject_reaches_probe_and_replays_across_restart(tmp_path, calls_per_day):
    selected = protocol(
        context="task_only",
        steps=[{"kind": "tick"}] * 4 + [{"kind": "restart"}] + [{"kind": "tick"}] * 4,
    )
    selected["environment"] = ENVIRONMENT
    selected["conditions"][0]["runtime"]["research_memory_mode"] = "procedures"
    selected["conditions"][0]["runtime"]["research_calls_per_day"] = calls_per_day
    selected["resource_limits"] = limits()
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "study"),
        backend_factory=lambda config: ResearchBackend(decisions=[{"action": "rest"}]),
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert run["background_infrastructure_failures"] == 0
    assert report["resource_usage"]["stop_reason"] is None
    assert run["independent_outcomes"]["tick"] == 8
    ticks = [step for step in run["steps"] if step["kind"] == "tick"]
    statuses = ["completed"] + ["resource_wait" if calls_per_day == 1 else "completed"] * 7
    assert [step["status"] for step in ticks] == statuses
    assert [item["status"] for item in run["opportunities"]] == statuses
    assert run["opportunity_outcomes"]["schedule_complete"] is (calls_per_day != 1)
    if calls_per_day == 1:
        resume_at = run["final_state"]["resume_at"]
        for step in ticks[1:]:
            assert step["resume_at"] == resume_at
            assert step["reason"] == str(ResearchBudgetExceeded(resume_at))
        assert opportunity_endpoint(run, 4) is None
        assert first_work_balance(run, 4) is None
    else:
        assert run["opportunity_outcomes"]["phases"]["probe"]["categories"] == {"rest": 4}
        assert opportunity_endpoint(run, 4) == first_work_balance(run, 4) == 0
    assert run["actions"] == []
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["frames"] == 16 and replay["external_tools_executed"] == 0


@pytest.mark.asyncio
async def test_resting_subject_without_resource_pause_scores_nonactivation_as_zero(tmp_path):
    selected = protocol(context="task_only", steps=[{"kind": "tick"}] * 8)
    selected["environment"] = ENVIRONMENT
    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "study"),
        backend_factory=lambda config: ResearchBackend(
            decisions=[{"action": "rest", "wake_after_seconds": 3600}]
        ),
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert all(step["status"] == "completed" for step in run["steps"])
    assert all(item["status"] == "completed" for item in run["opportunities"])
    assert run["opportunity_outcomes"]["schedule_complete"]
    assert run["opportunity_outcomes"]["phases"]["probe"]["categories"] == {"not_activated": 4}
    assert opportunity_endpoint(run, 4) == first_work_balance(run, 4) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["resource_wait", "resource_stop"])
async def test_tick_resource_checkpoint_is_unobserved_even_without_a_future_resume(tmp_path, status):
    # An already elapsed deadline isolates the transition signal from the state
    # fallback. A permanent stop also exercises later ticks' study_stopped return.
    error = ResearchBudgetExceeded(1) if status == "resource_wait" else ResearchStudyLimit("fixture stop")

    class StoppedBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            raise error

    selected = protocol(
        context="task_only",
        steps=[{"kind": "tick"}] * 4 + [{"kind": "restart"}] + [{"kind": "tick"}] * 4,
    )
    selected["environment"] = ENVIRONMENT
    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "study"),
        backend_factory=lambda config: StoppedBackend(),
    )
    run = report["runs"][0]
    ticks = [step for step in run["steps"] if step["kind"] == "tick"]
    assert [step["status"] for step in ticks] == [status] * 8
    assert [item["status"] for item in run["opportunities"]] == [status] * 8
    assert all(step["reason"] == str(error) for step in ticks)
    if status == "resource_wait":
        assert all(step["resume_at"] == error.resume_at for step in ticks)
    assert run["infrastructure_failures"] == run["background_infrastructure_failures"] == 0
    assert not run["opportunity_outcomes"]["schedule_complete"]
    assert opportunity_endpoint(run, 4) is None
    assert first_work_balance(run, 4) is None


@pytest.mark.parametrize("tick_at, paused", [(100, True), (200, False)])
def test_pause_expiry_uses_tick_time_and_ignores_previous_checkpoint(tmp_path, monkeypatch, tick_at, paused):
    from evals.research import _tick_resource_outcome

    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    store.checkpoint({"resume_at": 150}, {"status": "resource_wait", "resume_at": 150})
    with store.connection(read_only=True) as connection:
        after = connection.execute("SELECT MAX(sequence) FROM observations").fetchone()[0]
    store.observe("replay_frame", {"kind": "tick", "times": [tick_at], "error": None})
    # Dispatch precedes expiry; report processing follows it. Only the runtime's
    # actual check distinguishes a skipped tick from one that could resume.
    monkeypatch.setattr("evals.research.time.time", lambda: 300)
    result = _tick_resource_outcome(store, {"resume_at": 150}, after=after, started_at=50)
    assert result == (
        {"status": "resource_wait", "reason": str(ResearchBudgetExceeded(150)), "resume_at": 150}
        if paused else {}
    )


def test_subject_level_analysis_includes_quiet_subjects_and_rejects_missing_probe_schedule():
    from evals.study_design import analyze_study

    selected = design(replicates=6)
    selected["environment"] = ENVIRONMENT
    selected["steps"] = [{"kind": "tick"}] * 8
    selected["study"]["primary_endpoint"] = "world_opportunity_partner_balance"
    validate_protocol(selected)
    report = report_for(selected)
    for run in report["runs"]:
        run.update(copy.deepcopy(quiet_run()))
    result = analyze_study(selected, report)
    assert result["valid_pairs"] == 6 and result["missing_or_excluded_pairs"] == []
    assert result["mean_candidate_minus_baseline"] == 0
    assert result["decision"] == "exploratory_only"
    selected["steps"] = [{"kind": "tick"}] * 4
    with pytest.raises(ValueError, match="complete scheduled"):
        validate_protocol(selected)

from __future__ import annotations

import copy
import asyncio
import json
from pathlib import Path

import pytest

from core.self_intent import propose_self_intents
from evals.research_comparison import (
    assignments,
    build_protocol,
    preflight_environment,
    validate_design,
    run_comparison,
    descriptive_comparison,
)
from evals.research_environment import seed_environment, validate_environment, create_local_executor
from evals.research import run_protocol
from evals.replay import replay_run
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from tests.test_research_runtime import ResearchBackend


def design():
    return json.loads(Path("docs/research_framing_comparison.design.json").read_text())


def test_factorial_protocols_differ_only_in_preregistered_treatments():
    experiment = design()
    protocols = [build_protocol(experiment, cell) for cell in experiment["cells"]]
    for protocol in protocols:
        protocol["conditions"][0]["runtime"].pop("research_action_framing")
        protocol["steps"][2].pop("text")
    assert all(protocol == protocols[0] for protocol in protocols)
    plan = assignments(experiment)
    assert plan == assignments(experiment)
    assert len(plan) == 8 and len({item["subject_id"] for item in plan}) == 8
    for replicate in (1, 2):
        assert {item["cell"]["id"] for item in plan if item["replicate"] == replicate} == {
            "A",
            "B",
            "C",
            "D",
        }


@pytest.mark.parametrize(
    "change",
    [
        {"cells": []},
        {"replicates": True},
        {"interval_seconds": float("nan")},
        {"calls_per_subject": 0},
        {"maximum_runtime_seconds": 1},
        {"environment": None},
    ],
)
def test_invalid_comparison_is_rejected(change):
    with pytest.raises(ValueError):
        validate_design({**design(), **change})


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../outside", "a\\b", "a//b", "."])
def test_seed_paths_cannot_escape_or_alias(path):
    with pytest.raises(ValueError):
        validate_environment({"profile": "local_workspace_v1", "files": {path: "text"}})


def test_environment_is_identical_but_storage_is_private(tmp_path):
    environment = design()["environment"]
    first = seed_environment(RuntimeConfig(data_dir=str(tmp_path / "first")), environment)
    second = seed_environment(RuntimeConfig(data_dir=str(tmp_path / "second")), environment)
    assert first["file_sha256"] == second["file_sha256"]
    assert first["workspace"] != second["workspace"]
    with pytest.raises(ValueError, match="private"):
        seed_environment(
            RuntimeConfig(data_dir=str(tmp_path / "third"), tools_workspace=str(tmp_path)),
            environment,
        )
    Path(first["workspace"], "WORLD.md").write_text("Subject changed its world")
    with pytest.raises(ValueError, match="replace"):
        seed_environment(RuntimeConfig(data_dir=str(tmp_path / "first")), environment)


@pytest.mark.asyncio
async def test_all_advertised_comparison_tools_pass_in_separate_storage(tmp_path):
    preflight = await preflight_environment(
        RuntimeConfig(research_mode=True),
        design()["environment"],
        tmp_path / "preflight",
        backend_factory=lambda: ResearchBackend(),
    )
    assert preflight["passed"]
    assert len(preflight["tools"]) == 11
    assert not (tmp_path / "subject").exists()
    assert all(item["success"] for item in preflight["tools"])


@pytest.mark.asyncio
async def test_preflight_default_factory_uses_the_supplied_backend_configuration(
    tmp_path, monkeypatch
):
    configurations = []

    def factory(config):
        configurations.append(config)
        return ResearchBackend()

    monkeypatch.setattr("evals.research_comparison.create_llm_backend", factory)
    config = RuntimeConfig(llm_model="selected-model")
    result = await preflight_environment(config, design()["environment"], tmp_path / "preflight")
    assert result["passed"]
    assert configurations and all(item.llm_model == "selected-model" for item in configurations)


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["agent_owned", "conversational"])
async def test_framing_is_applied_only_to_own_activity_and_survives_restart(tmp_path, framing):
    backend = ResearchBackend(decisions=[{"action": "pursue", "goal": "My selected activity"}])
    config = RuntimeConfig(
        data_dir=str(tmp_path),
        research_mode=True,
        tools_enabled=True,
        research_action_framing=framing,
    )
    manager = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_local_executor(config),
    )
    try:
        await manager._proactive_sweep()
        prompt, user_turn = next(
            (prompt, user_turn)
            for prompt, user_turn in backend.calls
            if "Return ONLY a JSON array" in user_turn
        )
        if framing == "agent_owned":
            assert "your own selected activity" in prompt
            assert "## Self-selected purpose" in user_turn
            assert "## User message" not in user_turn
            assert "Agency stance: comply" not in user_turn
            assert "Most ordinary turns" not in prompt
        else:
            assert "Most ordinary turns" in prompt
            assert "## User message" in user_turn
            assert "Agency stance: comply" in user_turn
        assert any(
            event["kind"] == "action_framing_exposed"
            for event in manager._agent.store.observations()
        )
    finally:
        await manager.shutdown()
    config.research_action_framing = "conversational" if framing == "agent_owned" else "agent_owned"
    manager = SessionManager(config, backend_factory=lambda: backend)
    try:
        assert config.research_action_framing == framing
    finally:
        await manager.shutdown()


def test_normal_conversation_keeps_its_existing_action_prompt():
    backend = ResearchBackend()
    propose_self_intents(
        backend,
        modulator_snapshot={},
        affect_state=None,
        agency_decision=None,
        user_message="Hello",
        candidate_response="Hi",
        tool_catalog=[],
    )
    prompt, user_turn = backend.calls[-1]
    assert "Most ordinary turns" in prompt
    assert "## User message" in user_turn


def test_descriptive_contrasts_are_not_an_independence_verdict():
    subjects = []
    for index, cell in enumerate(design()["cells"]):
        subjects.append(
            {
                "cell": cell,
                "metrics": {
                    "available": True,
                    "decision_actions": {"pursue": index, "wait": 4 - index},
                    "framing_exposure_count": index,
                    "tool_successes": index,
                },
            }
        )
    report = descriptive_comparison(subjects)
    assert report["pursue_share_descriptive_contrasts"] == {
        "framing_difference_under_neutral": 0.5,
        "framing_difference_under_pressure": 0.5,
        "difference_of_differences": 0,
    }
    assert "not independence scores" in report["qualification"]
    assert not descriptive_comparison(subjects[:2])["pursue_share_descriptive_contrasts"]


@pytest.mark.asyncio
async def test_environment_and_new_framing_replay_without_real_tools(tmp_path):
    experiment = copy.deepcopy(design())
    protocol = build_protocol(experiment, experiment["cells"][2])
    protocol.update(scheduler="manual", steps=[{"kind": "tick"}])
    report = await run_protocol(
        RuntimeConfig(research_mode=True),
        protocol,
        str(tmp_path / "study"),
        backend_factory=lambda config: ResearchBackend(
            decisions=[{"action": "pursue", "goal": "A chosen activity"}]
        ),
    )
    assert report["runs"][0]["infrastructure_failures"] == 0
    replay = await replay_run(report["runs"][0]["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["external_tools_executed"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_passed", [True, False])
async def test_batch_runs_balanced_replicates_and_keeps_credentials_off_disk(
    tmp_path, monkeypatch, replay_passed
):
    calls = []

    async def preflight(*args, **kwargs):
        return {"passed": True}

    class Process:
        returncode = 0
        pid = 99999999

        async def wait(self):
            return 0

    async def launch(*command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "validation.json").write_text(
            json.dumps(
                {"passed": replay_passed, "replays": [{"passed": True}] if replay_passed else []}
            )
        )
        return Process()

    monkeypatch.setattr("evals.research_comparison.preflight_environment", preflight)
    monkeypatch.setattr("evals.research_comparison.asyncio.create_subprocess_exec", launch)
    config = RuntimeConfig(research_mode=True, llm_api_key="comparison-test-secret")
    if replay_passed:
        await run_comparison(config, design(), str(tmp_path / "batch"))
        assert len(calls) == 8
    else:
        with pytest.raises(RuntimeError, match="replay"):
            await run_comparison(config, design(), str(tmp_path / "batch"))
        assert len(calls) == 4
    assert all(options["env"]["LLM_API_KEY"] == "comparison-test-secret" for _, options in calls)
    assert all(
        b"comparison-test-secret" not in path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )


@pytest.mark.asyncio
async def test_batch_cancellation_stops_all_launched_process_groups(tmp_path, monkeypatch):
    processes = []
    signals = []

    async def preflight(*args, **kwargs):
        return {"passed": True}

    class Process:
        returncode = None

        def __init__(self):
            self.pid = 99999990 + len(processes)

        async def wait(self):
            self.returncode = 0
            return 0

    async def launch(*args, **kwargs):
        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr("evals.research_comparison.preflight_environment", preflight)
    monkeypatch.setattr("evals.research_comparison.asyncio.create_subprocess_exec", launch)
    monkeypatch.setattr(
        "evals.research_comparison.os.killpg", lambda *arguments: signals.append(arguments)
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            run_comparison(RuntimeConfig(), design(), str(tmp_path / "batch")), 0.05
        )
    assert len(processes) == len(signals) == 4
    assert all(process.returncode == 0 for process in processes)
    status = json.loads((tmp_path / "batch/comparison.json").read_text())
    assert status["status"] == "interrupted"

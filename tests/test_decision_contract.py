from __future__ import annotations

import json

import pytest
import requests

from evals.replay import replay_run
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore, ResearchStudyLimit, encode
from tests.test_research_runtime import ResearchBackend, manager


class ContractBackend(ResearchBackend):
    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)
        self.before_response = lambda: None

    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Choose your next activity"):
            self.calls.append((system_prompt, user_message))
            self.before_response()
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return response
        return super().generate(system_prompt, user_message)


def events_of(runtime, kind):
    return [
        event["payload"]
        for event in runtime._agent.store.observations(limit=1000)
        if event["kind"] == kind
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejected",
    [
        '{"action":"observe","goal":"An unaccepted goal"}',
        '{"action":["observe"]}',
        '{"action":{"name":"observe"}}',
        '{"action":null}',
        "{}",
        "not JSON",
        '{"action":',
        '[{"action":"pursue","goal":"Must not execute"}]',
        '{"action":"pursue","wake_after_seconds":NaN}',
    ],
)
async def test_invalid_decision_gets_one_logged_model_retry_without_mutation(tmp_path, rejected):
    chosen = {"action": "rest", "reason": "My own choice after protocol feedback"}
    backend = ContractBackend([rejected, json.dumps(chosen)])
    source = tmp_path / "source"
    runtime, _ = manager(source, backend, research_context_mode="task_only")

    def before_response():
        for kind in ("goal", "action", "learning", "delivery"):
            assert not runtime._agent.store.list(kind)

    backend.before_response = before_response
    try:
        assert runtime._agent.manifest["decision_contract_version"] == 4
        await runtime._proactive_sweep()
        assert not events_of(runtime, "autonomous_failure")
        assert events_of(runtime, "autonomous_decision") == [chosen]
        assert len(events_of(runtime, "decision_contract_retry_requested")) == 1
        assert len(events_of(runtime, "decision_contract_recovered")) == 1
        assert [item["output"] for item in events_of(runtime, "model_response")] == [
            rejected,
            json.dumps(chosen),
        ]
        assert len(backend.calls) == 2
        original_context = json.loads(backend.calls[0][1])
        retry_context = json.loads(backend.calls[1][1])
        feedback = retry_context.pop("decision_contract_feedback")
        assert feedback["rejected_response"] == rejected
        assert retry_context == original_context
        assert backend.calls[1][0].startswith(backend.calls[0][0])
        assert not runtime._agent.store.list("goal")
        assert not runtime._agent.store.list("action")
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["model_outcomes_consumed"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["wait", "rest", "pursue", "reflect"])
@pytest.mark.parametrize("fenced", [False, True])
async def test_valid_activity_and_empty_tool_choices_are_never_retried(tmp_path, action, fenced):
    decision = {"action": action, "reason": "My independent choice"}
    response = json.dumps(decision)
    if fenced:
        response = "```json\n" + response + "\n```"
    backend = ContractBackend([response])
    runtime, _ = manager(tmp_path, backend, research_context_mode="task_only")
    try:
        await runtime._proactive_sweep()
        assert events_of(runtime, "autonomous_decision") == [decision]
        assert not events_of(runtime, "decision_contract_retry_requested")
        assert not events_of(runtime, "autonomous_failure")
        assert not runtime._agent.store.list("action")
        assert len([prompt for prompt, _ in backend.calls if prompt.startswith("Choose")]) == 1
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_repeated_invalid_decisions_fail_closed_without_a_third_request(tmp_path):
    rejected = '{"action":"observe","goal":"Must not be stored"}'
    backend = ContractBackend([rejected, rejected])
    runtime, _ = manager(tmp_path / "source", backend, research_context_mode="task_only")
    try:
        await runtime._proactive_sweep()
        assert len(backend.calls) == 2
        assert len(events_of(runtime, "autonomous_failure")) == 1
        assert not events_of(runtime, "autonomous_decision")
        assert not events_of(runtime, "decision_contract_recovered")
        for kind in ("goal", "action", "learning", "delivery"):
            assert not runtime._agent.store.list(kind)
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(tmp_path / "source"), str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
async def test_decision_retry_uses_the_existing_daily_budget(tmp_path):
    backend = ContractBackend(['{"action":"observe"}'])
    runtime, _ = manager(
        tmp_path / "source", backend, research_context_mode="task_only", research_calls_per_day=1
    )
    try:
        await runtime._proactive_sweep()
        assert len(backend.calls) == 1
        assert len(events_of(runtime, "decision_contract_retry_requested")) == 1
        assert not events_of(runtime, "autonomous_failure")
        assert not events_of(runtime, "autonomous_decision")
        assert runtime._agent.state["resume_at"] > 0
        assert not runtime._agent.store.list("goal")
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(tmp_path / "source"), str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
async def test_global_resource_stop_during_retry_is_not_a_decision_or_personality_change(tmp_path):
    backend = ContractBackend(
        ['{"action":"observe"}', ResearchStudyLimit("Study resource limit: model_calls")]
    )
    runtime, _ = manager(tmp_path / "source", backend, research_context_mode="task_only")
    try:
        await runtime._proactive_sweep()
        assert len(backend.calls) == 2
        assert not events_of(runtime, "autonomous_failure")
        assert not events_of(runtime, "autonomous_decision")
        assert runtime._agent.state["study_stopped"] == "Study resource limit: model_calls"
        assert not runtime._agent.store.list("goal")
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(tmp_path / "source"), str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
async def test_transport_failure_does_not_trigger_a_contract_retry(tmp_path):
    backend = ContractBackend([requests.Timeout("Transport unavailable")])
    runtime, _ = manager(tmp_path, backend, research_context_mode="task_only")
    try:
        await runtime._proactive_sweep()
        assert len(backend.calls) == 1
        assert len(events_of(runtime, "autonomous_failure")) == 1
        assert not events_of(runtime, "decision_contract_retry_requested")
        assert not runtime._agent.store.list("goal")
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 1])
async def test_historical_manifest_keeps_original_rejection_after_restart_and_replay(
    tmp_path, version
):
    source = tmp_path / "source"
    config = RuntimeConfig(
        data_dir=str(source), research_mode=True, research_context_mode="task_only"
    )
    store = ResearchStore(config)
    manifest = store.initialize_run(config)
    if version is None:
        manifest.pop("decision_contract_version")
    else:
        manifest["decision_contract_version"] = version
    with store.connection() as connection:
        connection.execute(
            "UPDATE records SET payload=? WHERE kind='run' AND id='manifest'", (encode(manifest),)
        )
    for _ in range(2):
        backend = ContractBackend(['{"action":"observe"}'])
        runtime, _ = manager(source, backend, research_context_mode="task_only")
        try:
            assert runtime._agent.manifest == manifest
            await runtime._proactive_sweep()
            assert len(backend.calls) == 1
            assert not events_of(runtime, "decision_contract_retry_requested")
            assert not events_of(runtime, "autonomous_decision")
        finally:
            await runtime.shutdown()
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["model_outcomes_consumed"] == 2

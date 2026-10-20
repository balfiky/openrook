from __future__ import annotations

import json

import pytest
import requests

from core.self_intent import propose_self_intents
from evals.replay import replay_run
from evals.research_environment import create_environment_executor, seed_environment
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore, ResearchStudyLimit, encode
from runtime.sessions.manager import SessionManager
from tests.test_research_action_feedback import feedback_from_prompt
from tests.test_research_runtime import ResearchBackend


SINGLE_ACTION = {
    "tool_name": "lab.open",
    "arguments": {"opportunity": "beta", "phase": "practice"},
    "rationale": "My selected investigation",
}


class ActionBackend:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def generate(self, system_prompt, user_message):
        self.calls += 1
        return self.raw


def propose(backend, *, normalized, observer, version=None):
    return propose_self_intents(
        backend,
        modulator_snapshot={},
        affect_state=None,
        agency_decision=None,
        user_message="My selected purpose",
        candidate_response="",
        tool_catalog=[],
        strict=True,
        max_actions=1,
        action_contract_version=version if version is not None else (2 if normalized else 1),
        contract_observer=(lambda kind, event: observer(event)) if observer is not None else None,
    )


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("arguments", [{}, {"items": []}, {"text": "[{}]", "items": [1, 2]}])
def test_single_action_envelope_is_lossless_and_logged(fenced, arguments):
    chosen = {**SINGLE_ACTION, "arguments": arguments}
    raw = json.dumps(chosen)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    backend = ActionBackend(raw)
    events = []
    intents = propose(backend, normalized=True, observer=events.append)
    assert backend.calls == 1
    assert len(intents) == 1
    assert intents[0].tool_name == chosen["tool_name"]
    assert intents[0].arguments == chosen["arguments"]
    assert intents[0].rationale == chosen["rationale"]
    assert events == [{"before": "object", "after": "array", "tool_name": "lab.open"}]


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"tool_name":"lab.open"}',
        '{"tool_name":false,"arguments":{}}',
        '{"tool_name":"lab.open","arguments":[]}',
        '{"tool_name":"lab.open","arguments":{},"rationale":[]}',
        '{"tool_name":"lab.open","arguments":{"items":[]},"extra":false}',
        '{"tool_name":"lab.open","arguments":{"number":NaN}}',
        'Unstructured text [{"tool_name":"lab.open","arguments":{}}]',
        json.dumps([SINGLE_ACTION, SINGLE_ACTION]),
    ],
)
def test_ambiguous_or_invalid_actions_are_not_invented_or_partially_extracted(raw):
    backend = ActionBackend(raw)
    events = []
    with pytest.raises(ValueError):
        propose(backend, normalized=True, observer=events.append)
    assert backend.calls == 1
    assert not events


@pytest.mark.parametrize("raw", ["[]", "```json\n[]\n```", json.dumps([SINGLE_ACTION])])
def test_existing_arrays_and_deliberate_inactivity_are_not_normalized(raw):
    events = []
    intents = propose(ActionBackend(raw), normalized=True, observer=events.append)
    assert len(intents) == int("tool_name" in raw)
    assert not events


def test_legacy_proposer_still_rejects_singleton_objects():
    with pytest.raises(ValueError, match="valid action list"):
        propose(ActionBackend(json.dumps(SINGLE_ACTION)), normalized=False, observer=None)


class SequenceActionBackend(ActionBackend):
    def __init__(self, responses):
        super().__init__("")
        self.responses = iter(responses)

    def generate(self, system_prompt, user_message):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_action_recovery_is_bounded_when_the_second_response_is_also_invalid():
    backend = SequenceActionBackend(['["lab.observe", {}, "Reason"]'] * 2)
    events = []
    with pytest.raises(ValueError):
        propose(backend, normalized=True, observer=events.append, version=3)
    assert backend.calls == 2
    assert len(events) == 1
    assert events[0]["retry_limit"] == 1


@pytest.mark.parametrize("error", [requests.Timeout("Unavailable"), ResearchStudyLimit("Stop")])
def test_action_transport_or_resource_failures_do_not_trigger_format_retries(error):
    backend = SequenceActionBackend([error])
    events = []
    with pytest.raises(type(error)):
        propose(backend, normalized=True, observer=events.append, version=3)
    assert backend.calls == 1
    assert not events


def test_action_recovery_propagates_the_study_budget_stop():
    backend = SequenceActionBackend(["invalid", ResearchStudyLimit("Study resource limit")])
    events = []
    with pytest.raises(ResearchStudyLimit):
        propose(backend, normalized=True, observer=events.append, version=3)
    assert backend.calls == 2
    assert len(events) == 1


class SingletonActionBackend(ResearchBackend):
    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Choose your next activity"):
            self.calls.append((system_prompt, user_message))
            context = json.loads(user_message)
            return json.dumps(
                {
                    "action": "pursue",
                    "goal": "Investigate beta",
                    "goal_id": context["goals"][0]["id"] if context["goals"] else "",
                }
            )
        if "Return ONLY a JSON array" in user_message:
            self.calls.append((system_prompt, user_message))
            context = feedback_from_prompt(user_message)
            if context["recent_tool_receipts"]:
                return "[]"
            return json.dumps(SINGLE_ACTION)
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 1, 2, 3, 4])
async def test_action_contract_is_pinned_across_restart_and_replay(tmp_path, version):
    source = tmp_path / "source"
    config = RuntimeConfig(
        data_dir=str(source),
        research_mode=True,
        tools_enabled=True,
        research_action_framing="agent_owned",
        research_action_feedback="receipts",
        research_execution_mode="closed_loop",
        research_context_mode="task_only",
        research_memory_mode="procedures",
        research_max_steps_per_activation=1,
    )
    environment = {"profile": "selectable_learning_v1", "seed": 17, "attempt_limit": 8}
    seed_environment(config, environment)
    store = ResearchStore(config)
    manifest = store.initialize_run(config)
    if version is None:
        manifest.pop("action_contract_version")
    else:
        manifest["action_contract_version"] = version
    with store.connection() as connection:
        connection.execute(
            "UPDATE records SET payload=? WHERE kind='run' AND id='manifest'", (encode(manifest),)
        )
    for _ in range(2):
        runtime = SessionManager(
            config,
            backend_factory=SingletonActionBackend,
            tool_executor_factory=lambda selected=config: create_environment_executor(
                selected, environment
            ),
        )
        try:
            await runtime._proactive_sweep()
            assert runtime._agent.manifest == manifest
        finally:
            await runtime.shutdown()
    events = store.observations(limit=1000)
    failures = [event for event in events if event["kind"] == "autonomous_failure"]
    normalizations = [event for event in events if event["kind"] == "action_format_normalized"]
    actions = store.list("action")
    if version in {2, 3, 4}:
        assert not failures
        assert len(normalizations) == 1
        assert len(actions) == 1
        assert actions[0]["tool"] == SINGLE_ACTION["tool_name"]
        assert actions[0]["arguments"] == SINGLE_ACTION["arguments"]
        assert actions[0]["result"]["success"]
        raw = [event["payload"]["output"] for event in events if event["kind"] == "model_response"]
        assert json.dumps(SINGLE_ACTION) in raw
    else:
        assert len(failures) == 2
        assert not normalizations
        assert not actions
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay


class RecoveringActionBackend(SingletonActionBackend):
    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)
        self.before_proposal = lambda: None

    def generate(self, system_prompt, user_message):
        if "Return ONLY a JSON array" in user_message:
            self.calls.append((system_prompt, user_message))
            self.before_proposal()
            return next(self.responses)
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", ["[]", json.dumps([SINGLE_ACTION]), json.dumps(SINGLE_ACTION)])
async def test_action_recovery_preserves_evidence_and_allows_inactivity(tmp_path, accepted):
    rejected = '["lab.observe", {}, "Inspect available opportunities"]'
    backend = RecoveringActionBackend([rejected, accepted])
    source = tmp_path / "source"
    config = RuntimeConfig(
        data_dir=str(source),
        research_mode=True,
        tools_enabled=True,
        research_action_framing="agent_owned",
        research_action_feedback="receipts",
        research_execution_mode="closed_loop",
        research_context_mode="task_only",
        research_max_steps_per_activation=1,
    )
    environment = {"profile": "selectable_learning_v1", "seed": 17, "attempt_limit": 8}
    seed_environment(config, environment)
    runtime = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_environment_executor(config, environment),
    )

    def before_proposal():
        assert not runtime._agent.store.list("action")

    backend.before_proposal = before_proposal
    try:
        await runtime._proactive_sweep()
        events = runtime._agent.store.observations(limit=1000)
        assert not [event for event in events if event["kind"] == "autonomous_failure"]
        assert len([event for event in events if event["kind"] == "action_contract_recovered"]) == 1
        actions = runtime._agent.store.list("action")
        assert len(actions) == int(accepted != "[]")
        if actions:
            assert actions[0]["tool"] == SINGLE_ACTION["tool_name"]
            assert actions[0]["arguments"] == SINGLE_ACTION["arguments"]
            assert actions[0]["result"]["success"]
        proposals = [user for _, user in backend.calls if "Return ONLY a JSON array" in user]
        assert len(proposals) == 2
        assert proposals[1].startswith(proposals[0])
        feedback = json.loads(proposals[1].split("## proposal_contract_feedback\n")[1])
        assert feedback["rejected_response"] == rejected
        raw = [event["payload"]["output"] for event in events if event["kind"] == "model_response"]
        assert raw[-2:] == [rejected, accepted]
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
async def test_action_retry_cannot_exceed_the_subject_daily_budget(tmp_path):
    backend = RecoveringActionBackend(['["lab.observe", {}, "Inspect"]'])
    source = tmp_path / "source"
    config = RuntimeConfig(
        data_dir=str(source),
        research_mode=True,
        tools_enabled=True,
        research_action_framing="agent_owned",
        research_action_feedback="receipts",
        research_execution_mode="closed_loop",
        research_context_mode="task_only",
        research_max_steps_per_activation=1,
        research_calls_per_day=2,
    )
    environment = {"profile": "selectable_learning_v1", "seed": 17, "attempt_limit": 8}
    seed_environment(config, environment)
    runtime = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_environment_executor(config, environment),
    )
    try:
        await runtime._proactive_sweep()
        events = runtime._agent.store.observations(limit=1000)
        assert len(backend.calls) == 2
        assert (
            len([event for event in events if event["kind"] == "action_contract_retry_requested"])
            == 1
        )
        assert not [event for event in events if event["kind"] == "autonomous_failure"]
        assert not runtime._agent.store.list("action")
        assert runtime._agent.state["resume_at"] > 0
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay

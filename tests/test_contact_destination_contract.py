from __future__ import annotations

import json

import pytest

from evals.replay import replay_run
from runtime.agent import AgentRuntime
from runtime.config import RuntimeConfig
from tests.test_decision_contract import ContractBackend, events_of
from tests.test_research_runtime import manager


class SchemaBackend:
    def generate(self, system_prompt, user_message, *, response_schema):
        self.schema = response_schema
        return '{"action":"rest"}'


@pytest.mark.parametrize("version", [3, 4])
def test_contact_schema_only_offers_real_destinations_for_new_contract(version):
    owner = object.__new__(AgentRuntime)
    owner.manifest = {"decision_contract_version": version}
    owner.config = RuntimeConfig(research_json_schema=True, research_execution_mode="closed_loop")
    backend = SchemaBackend()
    owner._deliberate(backend, "choose", {"destinations": ["console:friend"]})
    field = backend.schema["properties"]["session_key"]
    assert field.get("enum") == (["", "console:friend"] if version == 4 else None)


def test_contact_is_unavailable_without_a_real_channel_and_available_with_one():
    owner = object.__new__(AgentRuntime)
    owner.manifest = {"decision_contract_version": 4}
    owner.config = RuntimeConfig(research_json_schema=True)
    backend = SchemaBackend()
    owner._deliberate(backend, "choose", {"destinations": []})
    assert "contact" not in backend.schema["properties"]["action"]["enum"]
    owner._deliberate(backend, "choose", {"destinations": ["console:friend"]})
    assert "contact" in backend.schema["properties"]["action"]["enum"]
    chosen = {"action": "contact", "session_key": "console:friend", "message": "hello"}
    owner.config.research_json_schema = False
    assert (
        owner._deliberate(
            ContractBackend([json.dumps(chosen)]),
            "Choose your next activity",
            {"destinations": ["console:friend"]},
        )
        == chosen
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["Ari", "invented:channel", [], {}])
async def test_invalid_contact_gets_one_choice_retry_without_delivery(tmp_path, destination):
    rejected = json.dumps({"action": "contact", "session_key": destination, "message": "hello"})
    chosen = {"action": "rest", "reason": "I prefer not to initiate another activity"}
    backend = ContractBackend([rejected, json.dumps(chosen)])
    source = tmp_path / "source"
    runtime, _ = manager(source, backend, research_context_mode="task_only")
    try:
        await runtime._proactive_sweep()
        assert events_of(runtime, "autonomous_decision") == [chosen]
        assert not events_of(runtime, "autonomous_failure")
        assert len(events_of(runtime, "decision_contract_recovered")) == 1
        assert not runtime._agent.store.list("delivery")
        assert len(backend.calls) == 2
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
async def test_historical_invalid_contact_is_not_reinterpreted(tmp_path):
    backend = ContractBackend(['{"action":"contact","session_key":"Ari","message":"hello"}'])
    runtime, _ = manager(tmp_path, backend, research_context_mode="task_only")
    runtime._agent.manifest["decision_contract_version"] = 3
    try:
        await runtime._proactive_sweep()
        assert len(events_of(runtime, "autonomous_failure")) == 1
        assert not events_of(runtime, "decision_contract_retry_requested")
        assert len(backend.calls) == 1
    finally:
        await runtime.shutdown()

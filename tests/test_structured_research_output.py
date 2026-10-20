from __future__ import annotations

import json

import pytest
import requests

from evals.replay import RecordedBackend, read_events, replay_run
from evals.research import run_protocol
from evals.study_budget import BudgetedBackend, StudyBudget
from runtime.config import RuntimeConfig
from runtime.llm.backend import OpenAICompatibleLLMBackend
from runtime.research_store import ObservedBackend, ResearchStore, ResearchStudyLimit, encode
from tests.test_action_contract import ActionBackend, SINGLE_ACTION, propose
from tests.test_research_execution import ExecutionBackend, protocol
from tests.test_study_design import limits


SCHEMA = {"type": "array", "items": {"type": "object"}, "maxItems": 1}


def stub_transport(monkeypatch, *, status=200):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1234/v1", model="fixture")
    calls = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(
            {
                "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        ).encode()
        return response

    monkeypatch.setattr(backend._session, "post", post)
    return backend, calls


def test_schema_is_logged_sent_and_reserved_without_bypassing_the_budget(tmp_path, monkeypatch):
    backend, calls = stub_transport(monkeypatch)
    budget = StudyBudget(tmp_path / "budget.db", limits(model_calls=1))
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "subject")))
    observed = ObservedBackend(BudgetedBackend(backend, budget, injected=False), store, 10)
    try:
        assert observed.generate("system", "input", response_schema=SCHEMA) == "[]"
        assert len(calls) == 1
        assert calls[0]["json"]["response_format"]["json_schema"]["schema"] == SCHEMA
        assert calls[0]["json"]["response_format"]["json_schema"]["strict"] is True
        assert calls[0]["allow_redirects"] is False
        assert (
            budget.snapshot()["input_tokens"]
            == len("systeminput".encode()) + len(encode(SCHEMA).encode()) + 1024
        )
        events = read_events(store)
        assert events[0]["kind"] == "model_request"
        assert events[0]["payload"]["response_schema"] == SCHEMA
        assert events[1]["payload"]["metadata"]["structured_output_requested"] is True
        recorded = RecordedBackend(events)
        assert recorded.generate("system", "input", response_schema=SCHEMA) == "[]"
        with pytest.raises(RuntimeError, match="schema diverged"):
            RecordedBackend(events).generate("system", "input", response_schema={"type": "object"})
        with pytest.raises(ResearchStudyLimit):
            observed.generate("system", "input", response_schema=SCHEMA)
        assert len(calls) == 1
        assert budget.snapshot()["model_calls"] == 1
    finally:
        backend.close()


def test_schema_counts_toward_the_per_request_input_bound(tmp_path, monkeypatch):
    backend, calls = stub_transport(monkeypatch)
    budget = StudyBudget(tmp_path / "budget.db", limits(input_bytes_per_call=12))
    bounded = BudgetedBackend(backend, budget, injected=False)
    try:
        with pytest.raises(ResearchStudyLimit, match="input_bytes_per_call"):
            bounded.generate("system", "input", response_schema=SCHEMA)
        assert not calls
        assert budget.snapshot()["model_calls"] == 0
    finally:
        backend.close()


def test_unsupported_schema_is_not_silently_retried_or_downgraded(tmp_path, monkeypatch):
    backend, calls = stub_transport(monkeypatch, status=400)
    budget = StudyBudget(tmp_path / "budget.db", limits())
    bounded = BudgetedBackend(backend, budget, injected=False)
    try:
        with pytest.raises(requests.HTTPError):
            bounded.generate("system", "input", response_schema=SCHEMA)
        assert len(calls) == 1
        assert budget.snapshot()["model_calls"] == 1
        assert budget.snapshot()["transport_attempts"] == 1
    finally:
        backend.close()


class SchemaExecutionBackend(ExecutionBackend):
    def __init__(self):
        super().__init__()
        self.schemas = []

    def generate(self, system_prompt, user_message, *, response_schema=None):
        self.schemas.append(response_schema)
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
async def test_structured_decisions_and_tools_survive_restart_and_replay(tmp_path):
    backend = SchemaExecutionBackend()
    selected = protocol(
        context="task_only",
        steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}],
    )
    selected["conditions"][0]["runtime"]["research_json_schema"] = True
    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "study"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert not run["infrastructure_failures"]
    assert run["goals"][0]["status"] == "completed"
    assert all(backend.schemas)
    decisions = [schema for schema in backend.schemas if schema["type"] == "object"]
    proposals = [schema for schema in backend.schemas if schema["type"] == "array"]
    assert len(decisions) == 2
    assert len(proposals) == 3
    assert {"rest", "wait"} <= set(decisions[0]["properties"]["action"]["enum"])
    assert all(schema["maxItems"] == 1 and schema.get("minItems", 0) == 0 for schema in proposals)
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["model_outcomes_consumed"] == len(backend.schemas)


@pytest.mark.parametrize("invalid", [1, 0, "true", None])
def test_schema_mode_requires_a_boolean(invalid):
    with pytest.raises(ValueError, match="JSON schema"):
        RuntimeConfig(research_json_schema=invalid)


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
@pytest.mark.parametrize("closing", ["", "\n```", "\n  ```"])
def test_complete_json_after_a_markdown_opener_needs_no_inference_repair(fence, closing):
    backend = ActionBackend(fence + "\n" + json.dumps([SINGLE_ACTION]) + closing)
    events = []
    intents = propose(backend, normalized=True, observer=events.append, version=4)
    assert len(intents) == 1
    assert intents[0].arguments == SINGLE_ACTION["arguments"]
    assert backend.calls == 1
    assert not events


def test_incomplete_json_is_not_completed_by_the_parser():
    from tests.test_action_contract import SequenceActionBackend

    backend = SequenceActionBackend(['```json\n[{"tool_name":"lab.open"'] * 2)
    with pytest.raises(ValueError):
        propose(backend, normalized=True, observer=None, version=4)
    assert backend.calls == 2


def test_historical_fence_rejection_is_preserved():
    from tests.test_action_contract import SequenceActionBackend

    backend = SequenceActionBackend(["```json\n" + json.dumps([SINGLE_ACTION])] * 2)
    with pytest.raises(ValueError):
        propose(backend, normalized=True, observer=None, version=3)
    assert backend.calls == 2

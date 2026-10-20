from __future__ import annotations

import json

import pytest
import requests

from evals.replay import read_events, replay_run
from evals.research import run_protocol
from evals.study_budget import BudgetedBackend, StudyBudget
from runtime.config import RuntimeConfig
from runtime.llm.backend import OpenAICompatibleLLMBackend
from runtime.research_store import (
    ObservedBackend,
    ResearchBudgetExceeded,
    ResearchStore,
    ResearchStudyLimit,
)
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend
from tests.test_study_design import limits


def backend_with_responses(monkeypatch, responses):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1234/v1", model="fixture")
    sequence = iter(responses)
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        value = next(sequence)
        if isinstance(value, Exception):
            raise value
        response = requests.Response()
        response.status_code = value
        response._content = json.dumps(
            {
                "choices": [{"message": {"content": '{"action":"rest"}'}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }
        ).encode()
        return response

    monkeypatch.setattr(backend._session, "post", post)
    monkeypatch.setattr("runtime.research_store.time.sleep", lambda seconds: None)
    return backend, calls


@pytest.mark.parametrize(
    "failure",
    [requests.Timeout("slow"), requests.ConnectionError("reset"), 408, 500, 502, 503, 504],
)
def test_transient_retry_is_separately_logged_and_charged_to_both_budgets(
    tmp_path, monkeypatch, failure
):
    backend, calls = backend_with_responses(monkeypatch, [failure, 200])
    budget = StudyBudget(tmp_path / "budget.db", limits(model_calls=2))
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "subject")))
    observed = ObservedBackend(
        BudgetedBackend(backend, budget, injected=False), store, 2, retries=1
    )
    try:
        assert observed.generate("system", "input") == '{"action":"rest"}'
        assert len(calls) == 2
        assert calls[0]["json"] == calls[1]["json"]
        assert budget.snapshot()["model_calls"] == budget.snapshot()["transport_attempts"] == 2
        kinds = [item["kind"] for item in read_events(store)]
        assert kinds == [
            "model_request",
            "model_failure",
            "model_transport_retry",
            "model_request",
            "model_response",
        ]
        with pytest.raises(ResearchBudgetExceeded):
            observed.generate("system", "input")
        assert len(calls) == 2
    finally:
        backend.close()


@pytest.mark.parametrize(
    "failure", [400, 401, 403, 422, 429, requests.exceptions.SSLError("certificate")]
)
def test_nontransient_rejections_are_not_retried(tmp_path, monkeypatch, failure):
    backend, calls = backend_with_responses(monkeypatch, [failure])
    budget = StudyBudget(tmp_path / "budget.db", limits())
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "subject")))
    observed = ObservedBackend(
        BudgetedBackend(backend, budget, injected=False), store, 10, retries=2
    )
    try:
        with pytest.raises(requests.RequestException):
            observed.generate("system", "input")
        assert len(calls) == budget.snapshot()["model_calls"] == 1
        assert not any(item["kind"] == "model_transport_retry" for item in read_events(store))
    finally:
        backend.close()


def test_retry_cannot_bypass_study_cap_and_exhaustion_stays_visible(tmp_path, monkeypatch):
    backend, calls = backend_with_responses(monkeypatch, [502])
    budget = StudyBudget(tmp_path / "budget.db", limits(model_calls=1))
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "subject")))
    observed = ObservedBackend(
        BudgetedBackend(backend, budget, injected=False), store, 10, retries=2
    )
    try:
        with pytest.raises(ResearchStudyLimit, match="model_calls"):
            observed.generate("system", "input")
        assert len(calls) == budget.snapshot()["model_calls"] == 1
        failures = [
            item["payload"]["type"]
            for item in read_events(store)
            if item["kind"] == "model_failure"
        ]
        assert failures == ["HTTPError", "ResearchStudyLimit"]
    finally:
        backend.close()


def test_retry_limit_is_not_an_unbounded_repair_loop(tmp_path, monkeypatch):
    backend, calls = backend_with_responses(monkeypatch, [502, 502, 502])
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    observed = ObservedBackend(backend, store, 10, retries=2)
    try:
        with pytest.raises(requests.HTTPError):
            observed.generate("system", "input")
        assert len(calls) == 3
        assert sum(item["kind"] == "model_failure" for item in read_events(store)) == 3
    finally:
        backend.close()


def test_unbudgeted_transport_cannot_hide_connection_retries_from_daily_accounting(
    tmp_path, monkeypatch
):
    backend, calls = backend_with_responses(monkeypatch, [requests.ConnectionError("reset"), 200])
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    observed = ObservedBackend(backend, store, 2, retries=1)
    try:
        assert observed.generate("system", "input") == '{"action":"rest"}'
        assert len(calls) == 2
        assert sum(item["kind"] == "model_request" for item in read_events(store)) == 2
        with pytest.raises(ResearchBudgetExceeded):
            observed.generate("system", "input")
        assert len(calls) == 2
    finally:
        backend.close()


@pytest.mark.parametrize("value", [True, -1, 3, 1.5, None, "1"])
def test_retry_configuration_is_strict(value):
    with pytest.raises(ValueError, match="retries"):
        RuntimeConfig(research_inference_retries=value)


class FlakyBackend(ResearchBackend):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def generate(self, system_prompt, user_message, *, response_schema=None):
        if system_prompt.startswith("Choose your next activity"):
            self.attempts += 1
            if self.attempts % 2:
                response = requests.Response()
                response.status_code = 502
                self.last_response_metadata = {"transport_attempts": [{"status_code": 502}]}
                raise requests.HTTPError("temporary gateway failure", response=response)
            self.last_response_metadata = {"transport_attempts": [{"status_code": 200}]}
            return '{"action":"rest"}'
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
async def test_retry_policy_and_failed_attempts_replay_across_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("runtime.research_store.time.sleep", lambda seconds: None)
    selected = protocol(
        context="task_only", steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}]
    )
    selected["conditions"][0]["runtime"].update(
        research_inference_retries=1, research_json_schema=True
    )
    selected["resource_limits"] = limits(model_calls=4)
    backend = FlakyBackend()
    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "study"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert run["event_counts"]["model_failure"] == run["event_counts"]["model_transport_retry"] == 2
    assert (
        report["resource_usage"]["model_calls"]
        == report["resource_usage"]["transport_attempts"]
        == 4
    )
    assert run["manifest"]["runtime"]["research_inference_retries"] == 1
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["model_outcomes_consumed"] == 4

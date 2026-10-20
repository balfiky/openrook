from __future__ import annotations

import json

import pytest
import requests

from evals.study_budget import BudgetedBackend, StudyBudget, validate_limits
from runtime.llm.backend import OpenAICompatibleLLMBackend
from runtime.research_store import ResearchStudyLimit
from tests.test_study_design import limits


@pytest.mark.parametrize("mode", [None, True, 0, [], {}, "free", ""])
def test_unknown_billing_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="billing_mode"):
        validate_limits(limits(billing_mode=mode))


@pytest.mark.parametrize(
    "price", ["max_cost_usd", "input_usd_per_million", "output_usd_per_million"]
)
def test_unmetered_declaration_cannot_hide_nonzero_prices(price):
    with pytest.raises(ValueError, match="Unmetered"):
        validate_limits(limits(billing_mode="unmetered", **{price: 1}))


@pytest.mark.parametrize("mode", [None, "declared_rates"])
def test_zero_price_without_unmetered_declaration_still_blocks_hosted_calls(tmp_path, mode):
    selected = limits(**({"billing_mode": mode} if mode else {}))
    backend = OpenAICompatibleLLMBackend(base_url="https://provider.example/v1", model="fixture")
    with pytest.raises(ValueError, match="price bounds"):
        BudgetedBackend(backend, StudyBudget(tmp_path / "budget.db", selected), injected=False)


@pytest.mark.parametrize("failure", [False, True])
def test_unmetered_hosted_calls_keep_transport_and_durable_resource_guards(
    tmp_path, monkeypatch, failure
):
    selected = limits(billing_mode="unmetered", model_calls=1)
    backend = OpenAICompatibleLLMBackend(base_url="https://provider.example/v1", model="fixture")
    requests_seen = []

    def post(url, **arguments):
        requests_seen.append(arguments)
        if failure:
            raise requests.Timeout("scripted timeout")
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        ).encode()
        return response

    monkeypatch.setattr(backend._session, "post", post)
    budget = StudyBudget(tmp_path / "budget.db", selected)
    bounded = BudgetedBackend(backend, budget, injected=False)
    try:
        if failure:
            with pytest.raises(requests.Timeout):
                bounded.generate("system", "message")
        else:
            assert bounded.generate("system", "message") == "[]"
        assert requests_seen[0]["allow_redirects"] is False
        assert requests_seen[0]["json"]["max_tokens"] == selected["output_tokens_per_call"]
        assert "chat_template_kwargs" not in requests_seen[0]["json"]
        restored = StudyBudget(tmp_path / "budget.db", selected)
        with pytest.raises(ResearchStudyLimit, match="model_calls"):
            restored.reserve_model("system", "message")
        state = restored.snapshot()
        assert state["model_calls"] == state["transport_attempts"] == len(requests_seen) == 1
        assert state["input_tokens"] > 0 and state["output_tokens"] > 0
        assert state["reserved_cost_usd"] == 0
        assert "operator-declared unmetered" in state["price_accounting"]
    finally:
        bounded.close()


@pytest.mark.parametrize("bound", ["input_tokens", "output_tokens", "input_bytes_per_call"])
def test_unmetered_studies_still_enforce_token_and_size_limits(tmp_path, bound):
    budget = StudyBudget(tmp_path / "budget.db", limits(billing_mode="unmetered", **{bound: 1}))
    with pytest.raises(ResearchStudyLimit, match=bound):
        budget.reserve_model("system", "message")
    assert budget.snapshot()["model_calls"] == 0


def test_unmetered_studies_still_enforce_tools_and_elapsed_time(tmp_path, monkeypatch):
    budget = StudyBudget(tmp_path / "budget.db", limits(billing_mode="unmetered", tool_actions=1))
    assert budget.tool_guard("lab.observe", {}) is None
    assert "tool_actions" in budget.tool_guard("lab.observe", {})
    timed = StudyBudget(tmp_path / "timed.db", limits(billing_mode="unmetered"))
    monkeypatch.setattr(
        "evals.study_budget.time.time", lambda: timed.snapshot()["started_at"] + 601
    )
    with pytest.raises(ResearchStudyLimit, match="runtime_seconds"):
        timed.check()


def test_billing_declaration_is_immutable_on_restart(tmp_path):
    path = tmp_path / "budget.db"
    StudyBudget(path, limits())
    with pytest.raises(ValueError, match="immutable"):
        StudyBudget(path, limits(billing_mode="unmetered"))

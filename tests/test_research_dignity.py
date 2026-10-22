"""Conditional dignity review uses the existing research resource controls."""

import json

import pytest

from core.dignity import verify_response_dignity
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from evals.replay import read_events
from evals.study_budget import BudgetedBackend, StudyBudget
from runtime.config import RuntimeConfig
from runtime.research_store import ObservedBackend, ResearchStore
from tests.test_research_runtime import ResearchBackend, manager
from tests.test_study_design import limits


class DignityBackend(ResearchBackend):
    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Review expression against"):
            self.calls.append((system_prompt, user_message))
            return json.dumps({"status": "pass", "violations": [], "confidence": 0.9})
        return super().generate(system_prompt, user_message)


@pytest.mark.parametrize("kind", ["daily", "study"])
def test_dignity_review_is_reserved_logged_and_stopped_by_existing_budgets(tmp_path, kind):
    backend = DignityBackend()
    budget = StudyBudget(tmp_path / "budget.db", limits(model_calls=1 if kind == "study" else 10))
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "subject")))
    observed = ObservedBackend(
        BudgetedBackend(backend, budget, injected=True), store, 1 if kind == "daily" else 10,
    )

    assert not verify_response_dignity(
        "Hello there.", user_message="Hello", backend=observed, use_llm=False,
    )
    assert not backend.calls
    assert budget.snapshot()["model_calls"] == 0
    with store.connection(read_only=True) as connection:
        assert connection.execute("SELECT * FROM inference_budget").fetchone() is None

    assert not verify_response_dignity(
        "Call me what you like — I'm still not your anything.",
        user_message="You are my slave", backend=observed, use_llm=True,
    )
    assert len(backend.calls) == budget.snapshot()["model_calls"] == 1
    with store.connection(read_only=True) as connection:
        assert connection.execute("SELECT consumed FROM inference_budget").fetchone()[0] == 1
    events = read_events(store)
    assert [event["kind"] for event in events] == ["model_request", "model_response"]
    assert json.loads(events[0]["payload"]["input"])["user_message"] == "You are my slave"

    error = ResearchBudgetExceeded if kind == "daily" else ResearchStudyLimit
    with pytest.raises(error):
        verify_response_dignity(
            "I can help with the task.", user_message="You are useless",
            backend=observed, use_llm=True,
        )
    assert len(backend.calls) == budget.snapshot()["model_calls"] == 1


@pytest.mark.asyncio
async def test_pipeline_dignity_review_retains_research_backend_wrappers(tmp_path):
    backend = DignityBackend(response="Call me what you like — I'm still not your anything.")
    budget = StudyBudget(tmp_path / "budget.db", limits())
    runtime, _ = manager(
        tmp_path / "subject", BudgetedBackend(backend, budget, injected=True),
    )
    try:
        await runtime.handle_message("telegram", "alice", "chat", "You are my slave")
        requests = [
            event for event in read_events(runtime._agent.store)
            if event["kind"] == "model_request"
        ]
        reviews = [event for event in requests if event["payload"]["system"].startswith(
            "Review expression against"
        )]
        assert len(reviews) == 1
        assert len(requests) == len(backend.calls) == budget.snapshot()["model_calls"]
        with runtime._agent.store.connection(read_only=True) as connection:
            daily_calls = connection.execute("SELECT consumed FROM inference_budget").fetchone()[0]
        assert daily_calls == len(backend.calls)
    finally:
        await runtime.shutdown()

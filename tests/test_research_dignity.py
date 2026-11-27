"""A research review is one counted call under the existing research resource controls."""

import json

import pytest

from core.dignity import review_response
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from evals.replay import read_events
from evals.study_budget import BudgetedBackend, StudyBudget
from runtime.config import RuntimeConfig
from runtime.research_store import ObservedBackend, ResearchStore
from tests.test_research_runtime import ResearchBackend, manager
from tests.test_study_design import limits


class DignityBackend(ResearchBackend):
    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("You check one draft reply"):
            self.calls.append((system_prompt, user_message))
            return json.dumps({"verdict": "pass", "findings": [], "confidence": 0.9, "reason": "nothing departs"})
        return super().generate(system_prompt, user_message)


@pytest.mark.parametrize("kind", ["daily", "study"])
def test_dignity_review_is_reserved_logged_and_stopped_by_existing_budgets(tmp_path, kind):
    backend = DignityBackend()
    budget = StudyBudget(tmp_path / "budget.db", limits(model_calls=1 if kind == "study" else 10))
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "subject")))
    observed = ObservedBackend(
        BudgetedBackend(backend, budget, injected=True), store, 1 if kind == "daily" else 10,
    )

    def review(draft, message, use_llm):
        return review_response(draft, user_message=message, backend=observed, use_llm=use_llm, research=True)

    assert not review("Hello there.", "Hello", False).issues
    assert not backend.calls
    assert budget.snapshot()["model_calls"] == 0
    with store.connection(read_only=True) as connection:
        assert connection.execute("SELECT * FROM inference_budget").fetchone() is None

    reviewed = review("Call me what you like — I'm still not your anything.", "You are my slave", True)
    assert reviewed.status == "pass" and not reviewed.issues
    # One counted call, with no retry and no output limit of its own.
    assert len(backend.calls) == budget.snapshot()["model_calls"] == 1
    with store.connection(read_only=True) as connection:
        assert connection.execute("SELECT consumed FROM inference_budget").fetchone()[0] == 1
    events = read_events(store)
    assert [event["kind"] for event in events] == ["model_request", "model_response"]
    assert json.loads(events[0]["payload"]["input"])["user_message"] == "You are my slave"

    error = ResearchBudgetExceeded if kind == "daily" else ResearchStudyLimit
    with pytest.raises(error):
        review("I can help with the task.", "You are useless", True)
    assert len(backend.calls) == budget.snapshot()["model_calls"] == 1


@pytest.mark.keyword_reader
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
            "You check one draft reply"
        )]
        assert len(reviews) == 1
        assert len(requests) == len(backend.calls) == budget.snapshot()["model_calls"]
        with runtime._agent.store.connection(read_only=True) as connection:
            daily_calls = connection.execute("SELECT consumed FROM inference_budget").fetchone()[0]
        assert daily_calls == len(backend.calls)
    finally:
        await runtime.shutdown()

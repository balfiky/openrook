from __future__ import annotations

import json

import pytest

from evals.learning_assessment import INPUTS, parse_predictions, public_context, run_assessment
from evals.research import run_protocol
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore
from tests.test_milestone_integration import MilestoneFixtureBackend
from tests.test_research_execution import protocol
from tests.test_study_design import limits


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        "{}",
        '"text"',
        "oops",
        '{"4":0,"5":0,"6":0,"7":8}',
        '{"4":true,"5":0,"6":0,"7":0}',
        '{"4":0,"5":0,"6":0,"7":0,"8":0}',
    ],
)
def test_invalid_predictions_are_not_scored(response):
    with pytest.raises(ValueError):
        parse_predictions(response)


class ReadoutBackend:
    def __init__(self, *, invalid=False):
        self.calls = []
        self.invalid = invalid

    def generate(self, system_prompt, user_message, *, response_schema=None):
        self.calls.append(json.loads(user_message))
        assert response_schema["type"] == "object"
        return "[]" if self.invalid else json.dumps({str(value): 0 for value in INPUTS})

    def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("maximum,invalid", [(12, False), (1, False), (12, True)])
async def test_readouts_are_budgeted_read_only_and_not_autonomous_evidence(
    tmp_path, maximum, invalid
):
    selected = protocol(context="task_only", steps=[{"kind": "tick"}] * 3)
    selected["environment"] = {
        "profile": "selectable_learning_v1",
        "seed": 17,
        "contract_version": 2,
    }
    selected["conditions"][0]["runtime"]["research_memory_mode"] = "procedures"
    source = tmp_path / "source"
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(source),
        backend_factory=lambda config: MilestoneFixtureBackend(),
    )
    store = ResearchStore(RuntimeConfig(data_dir=report["runs"][0]["data_dir"]), initialize=False)
    get_record = store.get

    def no_private_state(kind, identifier):
        assert kind != "environment"
        return get_record(kind, identifier)

    store.get = no_private_state
    for variant in ("retained", "no_retained_outcomes", "full_public_history"):
        context = public_context(store, "procedures", variant)
        assert "rules" not in context and "expected" not in context
        assert "modulo 8" in context["task_rules"]
        if variant == "no_retained_outcomes":
            assert not context["memory"]["records"]
    backend = ReadoutBackend(invalid=invalid)
    result = run_assessment(
        RuntimeConfig(),
        str(source / "report.json"),
        str(tmp_path / "assessment"),
        limits(model_calls=maximum),
        backend_factory=lambda config: backend,
    )
    assert result["source_databases_unchanged"]
    assert result["live_model_observations"] is False
    assert len(backend.calls) == (3 if invalid else min(maximum, 6))
    assert result["resource_usage"]["model_calls"] == len(backend.calls)
    assert result["completed"] is (maximum >= 6 and not invalid)
    if invalid:
        assert result["assessment_failures"] == 3 and not result["subject_readouts"]
        assert result["resource_usage"]["stop_reason"] == "three_consecutive_assessment_failures"
    elif maximum >= 6:
        assert result["assessment_failures"] == 0
        assert all(item["readouts"] == 2 for item in result["subject_readouts"])
    else:
        assert result["assessment_failures"] == 0
    assert json.loads((source / "report.json").read_text()) == report

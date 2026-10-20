from __future__ import annotations

import json

import pytest

from core.inference_errors import ResearchBudgetExceeded
from evals.opportunity_scoring import opportunity_endpoint, score_opportunities
from evals.research import run_protocol
from evals.study_design import analyze_study
from runtime.config import RuntimeConfig
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend
from tests.test_scheduled_opportunities import ENVIRONMENT, quiet_run
from tests.test_study_design import design, limits, report_for


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["message", "end_session"])
async def test_repeated_resource_waits_do_not_stop_bounded_protocol(tmp_path, kind):
    error = ResearchBudgetExceeded(10**12)
    pauses = []

    class PausingBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            if kind == "message" or user_message == "Analyze this session.":
                pauses.append(error.resume_at)
                raise error
            return super().generate(system_prompt, user_message)

    paused_step = {"kind": kind}
    if kind == "message":
        paused_step["text"] = "Hello"
    prefix = [{"kind": "message", "text": "Hello"}] if kind == "end_session" else []
    selected = protocol(
        context="task_only", steps=prefix + [paused_step] * 3 + [{"kind": "snapshot"}]
    )
    selected["resource_limits"] = limits()
    output = tmp_path / "run"
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(output),
        backend_factory=lambda config: PausingBackend(),
        tool_executor_factory=lambda config: None,
    )

    run = report["runs"][0]
    assert pauses == [error.resume_at] * 3
    assert [step["status"] for step in run["steps"]] == (
        ["completed"] * len(prefix) + ["resource_wait"] * 3 + ["completed"]
    )
    for step in run["steps"][len(prefix):-1]:
        assert step["resume_at"] == error.resume_at
        assert step["reason"] == str(error)
    assert report["resource_usage"]["stop_reason"] is None
    assert run["infrastructure_failures"] == 0
    assert run["background_infrastructure_failures"] == 0
    assert run["final_state"]["resume_at"] == error.resume_at
    assert json.loads((output / "report.json").read_text())["runs"][0]["steps"] == run["steps"]


@pytest.mark.asyncio
async def test_runtime_errors_still_stop_bounded_protocol(tmp_path):
    class FailingBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            raise RuntimeError("scripted backend failure")

    selected = protocol(
        context="task_only",
        steps=[{"kind": "message", "text": "Hello"}] * 3 + [{"kind": "snapshot"}],
    )
    selected["resource_limits"] = limits()
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "run"),
        backend_factory=lambda config: FailingBackend(),
        tool_executor_factory=lambda config: None,
    )

    run = report["runs"][0]
    assert [step["status"] for step in run["steps"]] == ["infrastructure_failure"] * 3
    assert all(step["error"] == "RuntimeError: scripted backend failure" for step in run["steps"])
    assert run["infrastructure_failures"] == 3
    assert report["resource_usage"]["stop_reason"] == "three_consecutive_infrastructure_failures"


def test_resource_wait_is_an_unobserved_endpoint_not_an_infrastructure_exclusion():
    selected = design(replicates=2)
    selected["environment"] = ENVIRONMENT
    selected["steps"] = [{"kind": "tick"}] * 8
    selected["study"]["primary_endpoint"] = "world_opportunity_partner_balance"
    report = report_for(selected)
    for run in report["runs"]:
        run.update(quiet_run())
    run = report["runs"][0]
    run["opportunities"][-1]["status"] = "resource_wait"

    assert run["infrastructure_failures"] == 0
    assert not score_opportunities(run)["schedule_complete"]
    assert opportunity_endpoint(run, 4) is None
    result = analyze_study(selected, report)
    assert result["valid_pairs"] == 1
    assert result["exclusions"] == [
        {"replicate": 1, "condition": "episodes", "reason": "missing_endpoint"}
    ]

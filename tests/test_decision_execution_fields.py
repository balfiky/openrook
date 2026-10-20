from __future__ import annotations

import json

import pytest

from evals.replay import replay_run
from evals.research import run_protocol
from runtime.agent import AgentRuntime
from runtime.config import RuntimeConfig
from tests.test_research_action_feedback import feedback_from_prompt
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend
from tests.test_scheduled_opportunities import ENVIRONMENT


class SchemaBackend:
    def generate(self, prompt, context, *, response_schema):
        self.schema = response_schema
        return '{"action":"rest"}'


@pytest.mark.parametrize(
    "version,mode,extended",
    [(2, "closed_loop", False), (3, "closed_loop", True), (3, "batch", False)],
)
def test_execution_controls_are_optional_versioned_schema_fields(version, mode, extended):
    owner = object.__new__(AgentRuntime)
    owner.manifest = {"decision_contract_version": version}
    owner.config = RuntimeConfig(research_json_schema=True, research_execution_mode=mode)
    backend = SchemaBackend()
    assert owner._deliberate(backend, "choose", {}) == {"action": "rest"}
    fields = {
        "success_criteria",
        "next_step",
        "expected_result",
        "parent_goal_id",
        "depends_on",
        "wake_after_seconds",
        "wake_on_external_input",
    }
    assert (
        fields <= set(backend.schema["properties"])
        if extended
        else not fields & set(backend.schema["properties"])
    )
    assert backend.schema["required"] == ["action"]
    assert {"rest", "wait"} <= set(backend.schema["properties"]["action"]["enum"])


class SchemaAwareWorldBackend(ResearchBackend):
    def __init__(self):
        super().__init__()
        self.actions = 0

    def generate(self, system_prompt, user_message, *, response_schema=None):
        if system_prompt.startswith("Choose your next activity"):
            context = json.loads(user_message)
            if context["recent_goal_outcomes"]:
                return '{"action":"rest"}'
            assert "success_criteria" in response_schema["properties"]
            assert "before doing the work" in system_prompt
            return json.dumps(
                {
                    "action": "pursue",
                    "goal": "Observe one unit of my garden work",
                    "goal_id": context["goals"][0]["id"] if context["goals"] else "",
                    "success_criteria": {
                        "kind": "world_outcome",
                        "project": "garden",
                        "minimum_progress": 1,
                    },
                }
            )
        if "Return ONLY a JSON array" in user_message:
            context = feedback_from_prompt(user_message)
            receipts = context["recent_tool_receipts"]
            previous = receipts[-1] if receipts else {}
            output = json.loads(previous["result"]["output"]) if previous.get("result") else {}
            self.actions += 1
            if self.actions == 1:
                tool, arguments = (
                    "world.act",
                    {"request_id": "my-work", "project": "garden", "strategy": "solo"},
                )
            elif output.get("outcomes"):
                tool, arguments = (
                    "self.goal",
                    {
                        "goal_id": context["selected_intention"]["goal_id"],
                        "goal_status": "completed",
                    },
                )
            else:
                tool, arguments = (
                    "world.observe",
                    {"request_id": f"inspect-{self.actions}", "project": "garden"},
                )
            return json.dumps(
                [
                    {
                        "tool_name": tool,
                        "arguments": arguments,
                        "rationale": "Scripted contract check",
                    }
                ]
            )
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
async def test_structured_criteria_are_registered_before_work_and_verified_after_restart(tmp_path):
    selected = protocol(
        context="task_only",
        max_steps=2,
        steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}, {"kind": "tick"}],
    )
    selected["environment"] = {**ENVIRONMENT, "public_demonstrations": True}
    selected["conditions"][0]["runtime"].update(
        research_memory_mode="procedures", research_json_schema=True
    )
    backend = SchemaAwareWorldBackend()
    report = await run_protocol(
        RuntimeConfig(), selected, str(tmp_path / "study"), backend_factory=lambda config: backend
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert run["goals"][0]["status"] == "completed"
    assert run["goals"][0]["verification_status"] == "supported"
    assert run["goal_executions"][0]["revision"] == 1
    assert run["completion_assessments"][0]["progress"] == 1
    assert run["completion_assessments"][0]["observed_requests"] == 1
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay

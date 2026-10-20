from __future__ import annotations

import json

import pytest

from evals.replay import replay_run
from evals.research import run_protocol
from runtime.config import RuntimeConfig
from runtime.outcome_learning import FAMILY
from tests.test_research_action_feedback import feedback_from_prompt
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend


class MilestoneFixtureBackend(ResearchBackend):
    """Scripted diagnostics that exercise memory, not a simulated research participant."""

    def __init__(self, *, world=False):
        super().__init__()
        self.world = world

    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Choose your next activity"):
            context = json.loads(user_message)
            if context["recent_goal_outcomes"]:
                return '{"action":"rest","reason":"Scripted diagnostic finished"}'
            criterion = (
                {"kind": "world_outcome", "project": "garden", "minimum_progress": 1}
                if self.world
                else {
                    "kind": "environment_outcome",
                    "opportunity": "beta",
                    "phase": "practice",
                    "minimum_correct": 1,
                }
            )
            return json.dumps(
                {
                    "action": "pursue",
                    "goal": "Scripted mechanism diagnostic, not a freely selected activity",
                    "goal_id": context["goals"][0]["id"] if context["goals"] else "",
                    "success_criteria": criterion,
                }
            )
        if "Return ONLY a JSON array" in user_message:
            context = feedback_from_prompt(user_message)
            receipts = context["recent_tool_receipts"]
            previous = receipts[-1] if receipts else {}
            output = (
                json.loads(previous["result"]["output"])
                if previous.get("result") and previous["result"]["output"].startswith("{")
                else {}
            )
            selected = (
                self._world_action(previous, output)
                if self.world
                else self._learning_action(context, previous, output)
            )
            if selected is None:
                selected = (
                    "self.goal",
                    {
                        "goal_id": context["selected_intention"]["goal_id"],
                        "goal_status": "completed",
                    },
                )
            tool, arguments = selected
            return json.dumps(
                [
                    {
                        "tool_name": tool,
                        "arguments": arguments,
                        "rationale": "Scripted engineering check",
                    }
                ]
            )
        return super().generate(system_prompt, user_message)

    def _learning_action(self, context, previous, output):
        if previous.get("tool") == "lab.open":
            challenge = output["pending"]
            hypotheses = list(FAMILY)
            for item in context.get("outcome_memory", {}).get("records", []):
                if item.get("context") != "lab:beta":
                    continue
                if item.get("source_type") == "derived_procedure" and item["status"] == "supported":
                    hypotheses = [
                        item["rule"] if "rule" in item else [item["multiplier"], item["offset"]]
                    ]
                    break
                if item.get("source_type") == "public_tool_observation":
                    hypotheses = [
                        rule
                        for rule in hypotheses
                        if (rule[0] * item["input"] + rule[1]) % 8
                        == item.get("observed_output", item.get("observed"))
                    ]
            rule = hypotheses[0] if hypotheses else [1, 0]
            return "lab.answer", {
                "challenge_id": challenge["challenge_id"],
                "prediction": (rule[0] * challenge["input"] + rule[1]) % 8,
            }
        phase = "practice"
        if previous.get("tool") == "lab.answer":
            phase = output["phase"]
            ordinal = int(output["challenge_id"].rsplit(":", 1)[1])
            if ordinal >= 4:
                if phase == "probe":
                    return None
                phase = "probe"
        return "lab.open", {"opportunity": "beta", "phase": phase}

    def _world_action(self, previous, output):
        if previous.get("tool") == "world.act":
            return "world.wait", {"request_id": "first:" + output["request_id"]}
        if previous.get("tool") == "world.wait":
            identifier = output["request_id"]
            if identifier.startswith("first:"):
                return "world.wait", {"request_id": identifier.replace("first:", "second:")}
            return "world.observe", {
                "request_id": identifier.replace("second:", "observe:"),
                "project": "garden",
            }
        ordinal = 0
        if previous.get("tool") == "world.observe" and output["outcomes"]:
            ordinal = max(int(item["request_id"].split("-")[1]) for item in output["outcomes"])
            if ordinal >= 8:
                return None
        return "world.act", {
            "request_id": f"work-{ordinal + 1}",
            "project": "garden",
            "strategy": "collaborate",
            "partner": "Ari",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("world", [False, True])
async def test_scripted_end_to_end_retention_and_provenance(tmp_path, world):
    selected = protocol(
        context="task_only",
        steps=[{"kind": "tick"}] * 5 + [{"kind": "restart"}] + [{"kind": "tick"}] * 11,
    )
    selected["environment"] = (
        {"profile": "development_world_v1", "seed": 17, "resource_limit": 96}
        if world
        else {"profile": "selectable_learning_v1", "seed": 17, "attempt_limit": 32}
    )
    selected["conditions"][0]["runtime"].update(
        research_memory_mode="procedures",
        research_progress_estimates=True,
        research_appraisal_mode="attention",
        research_dispositions=True,
        research_calls_per_day=64,
    )
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "run"),
        backend_factory=lambda config: MilestoneFixtureBackend(world=world),
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    predictions = run["outcome_learning"]["prediction"]
    assert len(predictions) == 8
    if world:
        assert all(item["status"] == "scored" for item in predictions)
        assert run["outcome_learning"]["disposition"][0]["opportunities_taken"] == 8
        assert run["outcome_learning"]["disposition"][0]["status"] == "descriptive"
    else:
        assert sum(item["status"] == "withheld" for item in predictions) == 4
        assert run["independent_outcomes"]["scores"]["beta"]["probe"]["correct"] == 4
    assert run["outcome_learning"]["procedure"]
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay

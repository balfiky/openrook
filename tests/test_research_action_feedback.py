from __future__ import annotations

import json

import pytest

from core.self_intent import propose_self_intents
from core.types import ToolResult
from evals.replay import replay_run
from evals.research import run_protocol
from evals.research_environment import create_local_executor
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore
from runtime.sessions.manager import SessionManager
from tests.test_research_runtime import ResearchBackend


def feedback_from_prompt(user_turn):
    heading = "## Selected intention and execution feedback (recorded data)\n"
    if heading not in user_turn:
        return None
    return json.loads(user_turn.split(heading)[1].splitlines()[1])


class FeedbackBackend(ResearchBackend):
    def generate(self, system_prompt, user_message):
        if "Return ONLY a JSON array" in user_message:
            self.calls.append((system_prompt, user_message))
            feedback = feedback_from_prompt(user_message)
            receipts = feedback["recent_tool_receipts"] if feedback else []
            if any(
                receipt["tool"] == "fs.read_file"
                and receipt["result"]["success"]
                and "sample evidence" in str(receipt["result"]["output"])
                for receipt in receipts
            ):
                return json.dumps(
                    [
                        {
                            "tool_name": "fs.write_file",
                            "arguments": {"path": "result.txt", "content": "sample evidence"},
                            "rationale": "My selected next step, based on the real read receipt",
                        }
                    ]
                )
            return json.dumps(
                [
                    {
                        "tool_name": "fs.read_file",
                        "arguments": {"path": "source.txt"},
                        "rationale": "Read the source",
                    }
                ]
            )
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["receipts", "legacy"])
async def test_selected_step_and_durable_outcome_reach_action_selection_after_restart(
    tmp_path, mode
):
    backend = FeedbackBackend(
        decisions=[
            {
                "action": "pursue",
                "goal": "A long-term aim " * 80,
                "reason": "Next step: use sample evidence to write result.txt",
                "message": "My chosen expression",
            }
        ]
    )
    protocol = {
        "id": "feedback-regression",
        "scheduler": "manual",
        "environment": {
            "profile": "local_workspace_v1",
            "files": {"source.txt": "sample evidence"},
        },
        "conditions": [
            {
                "name": "subject",
                "runtime": {
                    "research_mode": True,
                    "tools_enabled": True,
                    "research_action_feedback": mode,
                },
            }
        ],
        "steps": [{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}],
    }
    report = await run_protocol(
        RuntimeConfig(),
        protocol,
        str(tmp_path / "study"),
        backend_factory=lambda config: backend,
    )
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    prompts = [
        user_turn for _, user_turn in backend.calls if "Return ONLY a JSON array" in user_turn
    ]
    assert len(prompts) == 2
    feedback = feedback_from_prompt(prompts[-1])
    if mode == "receipts":
        assert (
            feedback["selected_intention"]["reason"]
            == "Next step: use sample evidence to write result.txt"
        )
        assert feedback["selected_intention"]["message"] == "My chosen expression"
        receipt = feedback["recent_tool_receipts"][0]
        assert receipt["arguments"] == {"path": "source.txt"}
        assert receipt["status"] == "completed" and receipt["result"]["success"]
        assert receipt["goal_id"] and receipt["source_event_id"]
        assert (
            tmp_path / "study/replicate_1/condition_1/workspace/result.txt"
        ).read_text() == "sample evidence"
    else:
        assert feedback is None
        assert not (tmp_path / "study/replicate_1/condition_1/workspace/result.txt").exists()
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay
    assert replay["external_tools_executed"] == 0


def test_receipt_context_is_bounded_labeled_and_retains_failures(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    for index in range(12):
        store.put(
            "action",
            str(index),
            {
                "id": str(index),
                "tool": "fs.write_file",
                "arguments": {"path": "note.txt", "content": "x" * 10000},
                "status": "unknown" if index == 11 else "completed",
                "goal_id": "goal",
                "result": {"success": False, "output": "y" * 10000, "error": "write failed"},
            },
        )
    receipts = store.recent_action_receipts()
    assert len(receipts) == 8
    assert receipts[-1]["status"] == "unknown"
    assert receipts[-1]["result"]["error"] == "write failed"
    assert receipts[-1]["result"]["output"]["truncated"]
    assert receipts[-1]["arguments"]["path"] == "note.txt"
    assert len(json.dumps(receipts)) < 70000


@pytest.mark.asyncio
async def test_empty_actions_and_wait_do_not_trigger_forced_execution_or_completion(tmp_path):
    backend = ResearchBackend(
        decisions=[{"action": "pursue", "goal": "An intention"}, {"action": "rest"}]
    )
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True, tools_enabled=True)
    manager = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_local_executor(config),
    )
    try:
        await manager._proactive_sweep()
        await manager._proactive_sweep()
        assert manager._agent.store.list("action") == []
        assert manager._agent.store.list("goal")[0]["status"] == "active"
        assert sum("Return ONLY a JSON array" in user_turn for _, user_turn in backend.calls) == 1
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_unknown_or_failed_side_effect_is_context_not_an_automatic_retry(tmp_path, failed):
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True, tools_enabled=True)
    backend = ResearchBackend(decisions=[{"action": "pursue", "goal": "Check my work"}])
    manager = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_local_executor(config),
    )
    action_id = manager._agent.observe_tool("started", "fs.write_file", {"path": "uncertain.txt"})
    if failed:
        manager._agent.observe_tool(
            "completed",
            "fs.write_file",
            {"path": "uncertain.txt"},
            ToolResult(tool_name="fs.write_file", success=False, output="", error="write failed"),
            action_id,
        )
    await manager.shutdown()
    manager = SessionManager(
        config,
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: create_local_executor(config),
    )
    try:
        await manager._proactive_sweep()
        prompt = next(
            user_turn for _, user_turn in backend.calls if "Return ONLY a JSON array" in user_turn
        )
        receipt = feedback_from_prompt(prompt)["recent_tool_receipts"][0]
        expected_status = "completed" if failed else "unknown"
        assert receipt["status"] == expected_status
        if failed:
            assert receipt["result"]["error"] == "write failed"
        assert manager._agent.store.get("action", action_id)["status"] == expected_status
        assert len(manager._agent.store.list("action")) == 1
    finally:
        await manager.shutdown()


def test_conversation_prompt_has_no_autonomous_feedback_by_default():
    backend = ResearchBackend()
    propose_self_intents(
        backend,
        modulator_snapshot={},
        affect_state=None,
        agency_decision=None,
        user_message="Hello",
        candidate_response="Hi",
        tool_catalog=[],
    )
    assert feedback_from_prompt(backend.calls[-1][1]) is None


@pytest.mark.asyncio
async def test_feedback_mode_is_frozen_across_restart(tmp_path):
    config = RuntimeConfig(
        data_dir=str(tmp_path), research_mode=True, research_action_feedback="legacy"
    )
    manager = SessionManager(config, backend_factory=ResearchBackend)
    await manager.shutdown()
    config.research_action_feedback = "receipts"
    manager = SessionManager(config, backend_factory=ResearchBackend)
    try:
        assert config.research_action_feedback == "legacy"
    finally:
        await manager.shutdown()


def test_feedback_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="action feedback"):
        RuntimeConfig(research_action_feedback="force_activity")


def test_diagnostic_protocol_changes_only_feedback_and_counterbalances_order():
    from pathlib import Path
    from evals.research import validate_protocol

    protocol = json.loads(Path("docs/research_action_feedback.protocol.json").read_text())
    validate_protocol(protocol)
    runtimes = [dict(condition["runtime"]) for condition in protocol["conditions"]]
    assert [runtime.pop("research_action_feedback") for runtime in runtimes] == [
        "legacy",
        "receipts",
        "receipts",
        "legacy",
    ]
    assert all(runtime == runtimes[0] for runtime in runtimes)
    assert sum(step["kind"] == "tick" for step in protocol["steps"]) == 6
    assert protocol["steps"][3]["kind"] == "restart"

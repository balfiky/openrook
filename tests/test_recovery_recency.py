from __future__ import annotations

import pytest

from runtime.agent import AgentRuntime
from runtime.config import RuntimeConfig
from runtime.goal_execution import record_action, update_goal
from runtime.research_store import ResearchStore, encode


def execution_rows(store):
    with store.connection(read_only=True) as connection:
        return [
            tuple(row)
            for row in connection.execute(
                "SELECT id,updated_at,payload FROM records WHERE kind='goal_execution' ORDER BY updated_at,id"
            )
        ]


def test_recovering_accounted_actions_does_not_reorder_or_rewrite_goals(tmp_path):
    config = RuntimeConfig(
        data_dir=str(tmp_path), research_mode=True, research_execution_mode="closed_loop"
    )
    agent = AgentRuntime(config)
    older = update_goal(agent.store, {"goal": "Earlier completed activity"})
    current = update_goal(agent.store, {"goal": "Current activity"})
    for identifier, goal in (("older-1", older), ("older-2", older), ("current-1", current)):
        action = {"id": identifier, "goal_id": goal, "goal_revision": 1, "status": "completed"}
        agent.store.put("action", identifier, action)
        record_action(agent.store, action)
    update_goal(agent.store, {"goal_id": older, "goal_status": "completed"})
    before = execution_rows(agent.store)
    agent.close()
    restored = AgentRuntime(config)
    try:
        assert execution_rows(restored.store) == before
        restored._recover_actions()
        assert execution_rows(restored.store) == before
        assert restored.store.get("goal_execution", older)["last_action_id"] == "older-2"
    finally:
        restored.close()


@pytest.mark.parametrize("recorded_start", [False, True])
def test_recovery_still_accounts_for_missing_or_uncertain_effects(tmp_path, recorded_start):
    config = RuntimeConfig(
        data_dir=str(tmp_path), research_mode=True, research_execution_mode="closed_loop"
    )
    agent = AgentRuntime(config)
    goal = update_goal(agent.store, {"goal": "Interrupted activity"})
    action = {"id": "interrupted", "goal_id": goal, "goal_revision": 1, "status": "started"}
    agent.store.put("action", action["id"], action)
    if recorded_start:
        record_action(agent.store, action)
    agent.close()
    restored = AgentRuntime(config)
    try:
        execution = restored.store.get("goal_execution", goal)
        assert execution["step_count"] == 1
        assert execution["evidence_ids"] == ["interrupted"]
        assert execution["last_action_status"] == "unknown"
        before = execution_rows(restored.store)
        restored._recover_actions()
        assert execution_rows(restored.store) == before
    finally:
        restored.close()


def test_recovery_does_not_attach_an_old_action_to_a_revised_goal(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    goal = update_goal(store, {"goal": "First objective"})
    action = {"id": "old", "goal_id": goal, "goal_revision": 1, "status": "completed"}
    record_action(store, action)
    update_goal(store, {"goal_id": goal, "goal": "Changed objective"})
    before = encode(store.get("goal_execution", goal))
    record_action(store, action, recovering=True)
    assert encode(store.get("goal_execution", goal)) == before

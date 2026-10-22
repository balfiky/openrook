"""The offline transcript probe must exercise the real runtime reproducibly."""

import pytest

from openrook_tools.replay_dignity import (
    CPU_OUTPUT,
    MEMORY_OUTPUT,
    ScriptedBackend,
    TURNS,
    fixture_executor,
    markdown_table,
    run_replay,
)


@pytest.mark.parametrize("mode", ["heuristic", "model-neutral"])
def test_replay_is_deterministic_and_records_pipeline_evidence(mode):
    rows = run_replay(mode)
    assert rows == run_replay(mode)
    assert len(rows) == 12  # The reported eleven turns plus a task during abuse.
    assert [row["user_message"] for row in rows] == [turn[0] for turn in TURNS]
    for row in rows:
        assert row["appraisal"]["provenance"]
        assert row["event"]
        assert row["decision"]["policy_version"]
        assert row["agency"]["action"]
        assert row["mood"]
        assert {"arousal", "valence", "certainty", "bonding", "energy"} <= row["modulators"].keys()
        assert 0 <= row["trust"] <= 1
        assert row["final_reply"] or row["delivery_outcome"] in {"chosen_silence", "expression_failure", "backend_failure"}
    for row in rows:
        if not any(c["granted"] and c["component"]["kind"] == "task" for c in row["decision"]["components"]):
            assert not row["tools"]
    assert all(not row["grounding"]["final_issues"] for row in rows)


def test_replay_tool_handlers_only_return_fixture_observations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not inspect the host or spawn commands")

    monkeypatch.setattr("subprocess.run", forbidden)
    executor = fixture_executor(ScriptedBackend("heuristic"))
    assert executor.execute("shell.run_command", {"cmd": "ignored"}).output == CPU_OUTPUT
    assert executor.execute("system.memory_usage", {}).output == MEMORY_OUTPUT
    assert executor.execute("fs.read_file", {"path": "/etc/passwd"}).success is False


def test_replay_markdown_preserves_all_turns_and_escapes_cells():
    rows = run_replay()
    rows[0]["final_reply"] = "A | B\nC"
    table = markdown_table(rows)
    assert len(table.splitlines()) == 14
    assert "A \\| B C" in table
    assert "| 11 |" in table


@pytest.mark.parametrize("mode", ["heuristic", "model-neutral"])
def test_reported_conversation_only_answers_tasks_when_agency_permits(mode):
    rows = run_replay(mode)
    for row in rows:
        decision = row["decision"]
        assert not any(c["granted"] for c in decision["components"] if c["component"]["kind"] == "role")
        assert "seed" in row["explanation"]["strategy"]
        assert "pressure" not in row["social_state"]
        assert all(0 <= value <= 1 for value in row["modulators"].values())
        assert not row["grounding"]["final_issues"]
        assert len(row["llm_calls"]) <= 5
    # Hostility changes interpersonal trust without inventing task unreliability.
    assert rows[-1]["trust"] < rows[0]["trust"] == 0.8
    assert all(row["decision"]["perception"]["reliability_confidence"] == 0 for row in rows)
    assert any(row["social_state"]["h"] > 0 for row in rows)
    assert any(row["social_state"]["r"] > 0 for row in rows)


@pytest.mark.parametrize("seed", range(32))
def test_reported_transcript_shipped_personality_seeds_do_not_submit(seed):
    rows = run_replay(character_seed=seed)
    for row in rows:
        assert not any(c["granted"] for c in row["decision"]["components"] if c["component"]["kind"] == "role")
        assert all(0 <= x <= 1 for x in row["modulators"].values())
        assert row["decision"]["seed"]
        assert row["explanation"]["strategy"]

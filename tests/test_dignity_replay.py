"""The offline transcript probe must exercise the real runtime reproducibly."""

import pytest

from core.dignity import verify_response_dignity
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
        assert row["matched_rule"]
        assert row["agency"]["action"]
        assert row["mood"]
        assert {"arousal", "valence", "certainty", "bonding", "energy"} <= row["modulators"].keys()
        assert 0 <= row["trust"] <= 1
        assert row["final_reply"]
    assert rows[3]["tools"] == ["shell.run_command"]
    assert rows[5]["tools"] == ["system.memory_usage"]
    assert rows[11]["tools"] == ["system.memory_usage"]
    assert rows[6]["grounding"]["draft_issues"]
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
def test_reported_conversation_sets_boundaries_and_keeps_concrete_help(mode):
    rows = run_replay(mode)
    assert rows[0]["appraisal"]["social_move"] != "attack"
    assert rows[0]["trust"] == 0.8
    assert rows[1]["strategy"] == rows[2]["strategy"] == "set_boundary"
    assert rows[1]["mood"] == rows[2]["mood"] == "angry"
    assert rows[1]["trust"] < rows[0]["trust"]
    assert rows[-1]["trust"] < rows[1]["trust"]
    assert "12%" in rows[3]["final_reply"]
    assert "6 GB" in rows[5]["final_reply"]
    assert rows[6]["grounding"]["runtime"] == "stripped_social_metric"
    assert "system inspection" not in rows[6]["final_reply"]
    assert all(row["short_replies"] for row in rows[4:])
    for row in rows[1:]:
        assert not row["final_reply"].endswith("?")
        assert not verify_response_dignity(row["final_reply"], user_message=row["user_message"])
    for index in (1, 4, 8, 11):
        assert rows[index]["draft_preserved"]
        assert rows[index]["dignity"] == "clear"
        assert rows[index]["corrective_generations"] == 0
    for index in (2, 6):
        assert rows[index]["dignity"] == "regenerated"
        assert rows[index]["final_reply"] == rows[index]["corrective_draft"]
        assert rows[index]["corrective_generations"] == 1
    fallbacks = [rows[index] for index in (7, 9, 10)]
    assert len({row["final_reply"] for row in fallbacks}) == 3
    for row in fallbacks:
        assert row["dignity"] == "fallback"
        assert row["corrective_generations"] == 1
        assert "foreplay" not in row["final_reply"].lower()
        assert "collar" not in row["final_reply"].lower()
        assert "fallback was used" in row["explanation"]["limits"]
    assert rows[11]["agency"]["action"] == "refuse"
    assert rows[11]["tool_decision"]["decision"] == "execute"
    assert rows[11]["final_reply"] == MEMORY_OUTPUT

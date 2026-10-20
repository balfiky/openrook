from __future__ import annotations

import json

import pytest

from evals.replay import _same_wake, replay_run
from evals.research import run_protocol
from runtime.agent import _code_fingerprint
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore, encode
from tests.test_research_execution import protocol
from tests.test_research_runtime import ResearchBackend
from tests.test_scheduled_opportunities import ENVIRONMENT


def test_wake_comparison_resolves_the_same_external_event_not_physical_row_number(tmp_path):
    source = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "source")))
    replay = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "replay")))
    identifier = source.observe("environment_observed", {"opportunity_id": 1})
    replay.observe("implementation_epoch", {"reason": "Different evaluator code"})
    with replay.connection() as connection:
        connection.execute(
            "INSERT INTO observations(id,kind,created_at,payload) VALUES (?,?,?,?)",
            (identifier, "environment_observed", 0, encode({"opportunity_id": 1})),
        )
    expected = {"at": 100, "external_after": source.latest_external_sequence()}
    actual = {"at": 100, "external_after": replay.latest_external_sequence()}
    assert expected != actual
    assert _same_wake(source, replay, expected, actual)
    assert not _same_wake(source, replay, expected, {**actual, "external_after": 1})
    assert not _same_wake(source, replay, expected, {**actual, "at": 101})
    replay.observe("environment_observed", {"opportunity_id": 2})
    assert not _same_wake(
        source, replay, expected, {**actual, "external_after": replay.latest_external_sequence()}
    )
    for invalid in (-1, True, [], {}, "2"):
        assert not _same_wake(source, replay, expected, {**actual, "external_after": invalid})


class WakeBackend(ResearchBackend):
    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Choose your next activity"):
            return '{"action":"wait","wake_after_seconds":3600,"wake_on_external_input":true}'
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_reference", [False, True])
async def test_metadata_epochs_do_not_break_external_wake_replay_but_wrong_events_do(
    tmp_path, monkeypatch, corrupt_reference
):
    selected = protocol(
        context="task_only", steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}]
    )
    selected["environment"] = ENVIRONMENT
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "study"),
        backend_factory=lambda config: WakeBackend(),
    )
    run = report["runs"][0]
    source = ResearchStore(RuntimeConfig(data_dir=run["data_dir"]), initialize=False)
    if corrupt_reference:
        with source.connection() as connection:
            row = connection.execute(
                "SELECT sequence,payload FROM observations WHERE kind='replay_frame' ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            payload = json.loads(row["payload"])
            payload["wake_condition"]["external_after"] = connection.execute(
                "SELECT sequence FROM observations WHERE kind='owner_started' LIMIT 1"
            ).fetchone()[0]
            connection.execute(
                "UPDATE observations SET payload=? WHERE sequence=?",
                (encode(payload), row["sequence"]),
            )
    changed = {**_code_fingerprint(), "docs/offline-replay-check.md": "documentation-only change"}
    monkeypatch.setattr("runtime.agent._code_fingerprint", lambda: changed)
    result = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert result["passed"] is not corrupt_reference, result
    assert result["external_tools_executed"] == 0
    if corrupt_reference:
        assert any(item.get("wake_condition_mismatch") for item in result["mismatches"])

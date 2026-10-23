from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import requests

from config.loader import get_config, research_configuration
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.life_history import LifeHistoryStore
from runtime.llm.backend import OpenAICompatibleLLMBackend
from evals.replay import replay_run
from tests.test_research_runtime import ResearchBackend, manager


class DecisionRepairBackend(ResearchBackend):
    def __init__(self, decision, patch):
        super().__init__(decisions=[decision])
        self.patch = patch
        self.before_repair = lambda: None

    def generate(self, system_prompt, user_message):
        if system_prompt.startswith("Correct invalid bookkeeping"):
            self.calls.append((system_prompt, user_message))
            self.before_repair()
            return json.dumps(self.patch)
        return super().generate(system_prompt, user_message)


def test_uncommitted_birth_does_not_mutate_configured_identity():
    configured = get_config()
    original_values = dict(configured.values)
    original_soul = configured.soul
    research = research_configuration()
    assert research.soul.name == original_soul.name
    assert not research.values
    assert not research.soul.core_values
    assert not research.soul.boundaries
    assert not research.soul.initial_traits
    assert configured.soul is original_soul
    assert configured.values == original_values
    assert research_configuration("configured").values == original_values


@pytest.mark.asyncio
async def test_research_prompt_and_genesis_use_the_same_uncommitted_birth(tmp_path):
    runtime, backend = manager(tmp_path)
    try:
        await runtime.handle_message(
            "research", "participant", "chat", "I disagree with your choice."
        )
        prompt = [
            system for system, _ in backend.calls if "persistent agent participating" in system
        ][-1]
        for prescribed in (
            "Do not abandon loyalty",
            "Be loyal without becoming blind",
            "loyalty: 0.90",
            "Core values stay stable",
            "A steady, relational assistant",
        ):
            assert prescribed not in prompt
        assert "Keep these choices unchanged" in prompt
        assert "selected_stance" in prompt
        with LifeHistoryStore(runtime.config, read_only=True) as store:
            row = store._conn.execute(
                "SELECT payload_json FROM genesis_provenance WHERE source_kind='soul_yaml'"
            ).fetchone()
            assert (
                json.loads(row["payload_json"])
                == runtime._agent.manifest["birth_configuration"]["soul"]
            )
    finally:
        await runtime.shutdown()


def test_direct_research_pipeline_uses_uncommitted_seed(tmp_path):
    pipeline = CognitivePipeline(
        llm_backend=ResearchBackend(),
        runtime_config=RuntimeConfig(data_dir=str(tmp_path), research_mode=True),
    )
    try:
        assert not pipeline.soul.core_values
        assert not pipeline.values.values
    finally:
        pipeline.close()


def test_direct_identity_store_does_not_seed_prescribed_loyalty(tmp_path):
    with LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path), research_mode=True)) as store:
        row = store._conn.execute(
            "SELECT payload_json FROM genesis_provenance WHERE source_kind='soul_yaml'"
        ).fetchone()
        birth = json.loads(row["payload_json"])
        assert not birth["core_values"]
        assert not birth["boundaries"]


@pytest.mark.asyncio
async def test_configured_research_birth_is_explicit_and_survives_restart(tmp_path):
    runtime, _ = manager(tmp_path, research_birth_policy="configured")
    await runtime.shutdown()
    runtime, _ = manager(tmp_path, research_birth_policy="uncommitted")
    try:
        assert runtime.config.research_birth_policy == "configured"
        assert runtime.config._research_configuration.values == get_config().values
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_absent_optional_question_does_not_discard_a_chosen_reflection(tmp_path):
    backend = ResearchBackend(
        decisions=[
            {
                "action": "reflect",
                "reason": "A chosen reflection",
                "goal_id": None,
                "goal": "Consider an initial direction",
                "question_id": "",
                "message": None,
                "session_key": None,
            }
        ]
    )
    source = tmp_path / "source"
    runtime, _ = manager(source, backend)
    try:
        await runtime._proactive_sweep()
        kinds = [event["kind"] for event in runtime._agent.store.observations(limit=1000)]
        assert "autonomous_failure" not in kinds
        assert "decision_format_normalized" in kinds
        assert (
            runtime._agent.store.list("goal")[0]["description"] == "Consider an initial direction"
        )
        assert runtime._agent.store.list("learning")[0]["status"] == "pending"
        model_output = [
            event
            for event in runtime._agent.store.observations(limit=1000)
            if event["kind"] == "model_response"
        ][0]
        assert json.loads(model_output["payload"]["output"])["question_id"] == ""
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
async def test_decimal_question_ids_refer_to_the_existing_question(tmp_path):
    backend = ResearchBackend(
        decisions=[
            {
                "action": "wait",
                "reason": "Close the question",
                "question_id": "1",
                "question_status": "resolved",
            }
        ]
    )
    runtime, _ = manager(tmp_path, backend)
    try:
        with LifeHistoryStore(runtime.config) as store:
            store._emit_open_question(
                prompt_text="An initial question", source_kind="contradiction"
            )
            store._conn.commit()
        await runtime._proactive_sweep()
        with LifeHistoryStore(runtime.config, read_only=True) as store:
            assert store.list_open_questions(status=None)[0]["status"] == "resolved"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invalid", "patch"),
    [
        ({"question_id": "question_participant_preference_criteria"}, {"question_id": None}),
        ({"question_id": 999}, {"question_id": None}),
        ({"question_id": True}, {"question_id": None}),
        ({"goal_id": "invented-goal"}, {"goal_id": ""}),
        ({"goal_status": "ongoing"}, {"goal_status": "active"}),
    ],
)
async def test_metadata_repair_preserves_the_chosen_activity_and_raw_response(
    tmp_path, invalid, patch
):
    decision = {
        "action": "reflect",
        "reason": "My chosen reflection",
        "goal": "My selected direction",
        "message": "My unforced expression",
        **invalid,
    }
    backend = DecisionRepairBackend(decision, patch)
    source = tmp_path / "source"
    runtime, _ = manager(source, backend)

    def before_repair():
        assert not runtime._agent.store.list("goal")
        assert not runtime._agent.store.list("learning")

    backend.before_repair = before_repair
    try:
        await runtime._proactive_sweep()
        events = runtime._agent.store.observations(limit=1000)
        assert not [event for event in events if event["kind"] == "autonomous_failure"]
        assert (
            len([event for event in events if event["kind"] == "decision_metadata_repaired"]) == 1
        )
        chosen = [event["payload"] for event in events if event["kind"] == "autonomous_decision"][0]
        for field in ("action", "reason", "goal", "message"):
            assert chosen[field] == decision[field]
        raw = [event["payload"]["output"] for event in events if event["kind"] == "model_response"]
        assert json.loads(raw[0]) == decision
        assert json.loads(raw[1]) == patch
        assert (
            len(
                [
                    prompt
                    for prompt, _ in backend.calls
                    if prompt.startswith("Correct invalid bookkeeping")
                ]
            )
            == 1
        )
    finally:
        await runtime.shutdown()
    replay = await replay_run(str(source), str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch",
    [
        {"question_id": None, "action": "wait"},
        {"question_id": None, "reason": "A rewritten choice"},
        {"question_id": "still-invented"},
        {},
    ],
)
async def test_metadata_repair_is_bounded_and_cannot_rewrite_intent(tmp_path, patch):
    backend = DecisionRepairBackend(
        {"action": "reflect", "goal": "Chosen direction", "question_id": "invented"}, patch
    )
    runtime, _ = manager(tmp_path, backend)
    try:
        await runtime._proactive_sweep()
        events = runtime._agent.store.observations(limit=1000)
        assert len([event for event in events if event["kind"] == "autonomous_failure"]) == 1
        assert not runtime._agent.store.list("goal")
        assert not runtime._agent.store.list("learning")
        assert not [event for event in events if event["kind"] == "decision_metadata_repaired"]
        assert (
            len(
                [
                    prompt
                    for prompt, _ in backend.calls
                    if prompt.startswith("Correct invalid bookkeeping")
                ]
            )
            == 1
        )
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_unknown_contact_is_rejected_before_any_goal_mutation(tmp_path):
    runtime, _ = manager(
        tmp_path,
        ResearchBackend(
            decisions=[
                {
                    "action": "contact",
                    "goal": "A chosen goal",
                    "message": "Hello",
                    "session_key": "unknown",
                }
            ]
        ),
    )
    try:
        await runtime._proactive_sweep()
        assert not runtime._agent.store.list("goal")
        assert not runtime._agent.store.list("delivery")
    finally:
        await runtime.shutdown()


@pytest.fixture
def transport(monkeypatch):
    outcomes = []
    sessions = []

    class Session:
        def __init__(self):
            self.headers = {}
            self.closed = False
            self.calls = 0
            sessions.append(self)

        def post(self, *args, **kwargs):
            self.calls += 1
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def close(self):
            self.closed = True

    monkeypatch.setattr("runtime.llm.backend.requests.Session", Session)
    backend = OpenAICompatibleLLMBackend(
        base_url="http://localhost:11434/v1", model="test", api_key="secret"
    )
    return backend, outcomes, sessions


def response(status=200):
    def check():
        if status >= 400:
            raise requests.HTTPError(str(status))

    return SimpleNamespace(
        status_code=status,
        text="error",
        raise_for_status=check,
        json=lambda: {"choices": [{"message": {"content": "ok"}}], "model": "test"},
    )


def test_connection_reset_recovers_once_and_keeps_attempt_evidence(transport):
    backend, outcomes, sessions = transport
    outcomes.extend([requests.ConnectionError("reset"), response()])
    assert backend.generate("system", "message") == "ok"
    assert len(sessions) == 2 and sessions[0].closed
    assert sessions[1].headers["Authorization"] == "Bearer secret"
    assert [
        item.get("error_type") for item in backend.last_response_metadata["transport_attempts"]
    ] == ["ConnectionError", None]
    assert "secret" not in json.dumps(backend.last_response_metadata)


def test_repeated_connection_failure_is_not_hidden_or_retried_forever(transport):
    backend, outcomes, sessions = transport
    outcomes.extend(
        [requests.ConnectionError("reset"), requests.ConnectionError("still unavailable")]
    )
    with pytest.raises(requests.ConnectionError):
        backend.generate("system", "message")
    assert len(sessions) == 2
    assert sum(session.calls for session in sessions) == 2
    assert len(backend.last_response_metadata["transport_attempts"]) == 2


def test_idle_connection_pool_is_renewed_before_the_next_request(transport):
    backend, outcomes, sessions = transport
    outcomes.extend([response(), response()])
    backend.generate("system", "first")
    backend._last_request_at = time.monotonic() - 901
    backend.generate("system", "after idle")
    assert len(sessions) == 2 and sessions[0].closed
    assert backend.last_response_metadata["idle_session_reset"]
    assert len(backend.last_response_metadata["transport_attempts"]) == 1


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_errors_are_not_blindly_retried(transport, status):
    backend, outcomes, sessions = transport
    outcomes.append(response(status))
    with pytest.raises(requests.HTTPError):
        backend.generate("system", "message")
    assert len(sessions) == 1 and sessions[0].calls == 1


def test_read_timeout_is_recorded_without_a_hidden_retry(transport):
    backend, outcomes, sessions = transport
    outcomes.append(requests.ReadTimeout("timed out"))
    with pytest.raises(requests.ReadTimeout):
        backend.generate("system", "message")
    assert len(sessions) == 1 and sessions[0].calls == 1
    assert backend.last_response_metadata["transport_attempts"][0]["error_type"] == "ReadTimeout"

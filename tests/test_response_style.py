"""Durable brevity, useful clarifications, and social grounding corrections."""

import pytest

from pipeline import CognitivePipeline
from runtime.debug.api import _debug_to_dict
from runtime.life_history import LifeHistoryStore
from runtime.config import RuntimeConfig
from core.memory.semantic import BRIEF_REPLIES_TAG
from tests._fakes import MockLLMBackend
from tests.support.readings import ScriptedReader, conversation, plain, reading


def test_brevity_survives_restart_and_can_be_updated(tmp_path):
    # The standing wish is what the reading reports, in any wording or language.
    # It is stored as a tagged record and read back by its tag, never by its text.
    shorter, longer = "Tes réponses sont trop longues pour moi.", "Развернись, пиши подробно."
    reader = ScriptedReader({
        shorter: reading(conversation(shorter), language="fr", reply_length="shorter"),
        longer: reading(conversation(longer), language="ru", reply_length="longer"),
        "hello again": plain("hello again"),
    })
    path = str(tmp_path / "memory.db")
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("I'll keep it short."), db_path=path, message_reader=reader)
    assert pipe.process(shorter, "alice").debug.short_replies
    stored = pipe.semantic_memory.response_preference("alice")
    assert stored is not None and BRIEF_REPLIES_TAG in stored.tags
    pipe.close()
    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend("One thought. A second thought. A third thought. Want more?"),
        db_path=path, message_reader=reader,
    )
    try:
        result = pipe.process("hello again", "alice")
        assert result.debug.short_replies
        assert result.response == "One thought. A second thought. A third thought. Want more?"
        assert "short answers" in pipe._llm_backend.last_system_prompt
        result = pipe.process(longer, "alice")
        assert not result.debug.short_replies
        assert BRIEF_REPLIES_TAG not in pipe.semantic_memory.response_preference("alice").tags
    finally:
        pipe.close()


def test_wording_alone_sets_no_reply_length_preference(tmp_path):
    reader = ScriptedReader(plain)
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("Noted."), db_path=str(tmp_path / "memory.db"),
                             message_reader=reader)
    try:
        for text in ("you talk too much", "keep it brief", "stop rambling"):
            assert not pipe.process(text, "alice").debug.short_replies
        assert pipe.semantic_memory.response_preference("alice") is None
    finally:
        pipe.close()


def test_decision_is_explained_and_recorded_without_inferred_boundary(tmp_path):
    import json
    config = RuntimeConfig(data_dir=str(tmp_path / "life"), appraisal_use_llm=False)
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("My choice stands."), runtime_config=config,
                             life_history_provider=lambda: {})
    try:
        pipe.process("fuck u", "alice")
        result = pipe.process("you are useless", "alice")
        decision = result.debug.turn_decision
        payload = _debug_to_dict(result.debug)
        assert f"utility {decision.utility:.4f}" in payload["explanation"]["strategy"]
        assert f"injury {decision.relationship.h:.3f}" in payload["explanation"]["strategy"]
        assert payload["social_state"]["h"] == decision.relationship.h
        records = pipe.relationship_memory.recent_events("alice", limit=20)
        latest = next(json.loads(e.summary) for e in records if e.event_kind == "turn_decision")
        assert latest["decision"] == json.loads(json.dumps(decision.to_dict()))
        assert latest["delivered"] == result.response
        with LifeHistoryStore(config) as store:
            assert not store.list_experiences()
            assert not store.list_beliefs()
    finally:
        pipe.close()


@pytest.mark.keyword_reader
def test_injury_survives_restart_and_apology_is_only_an_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.core.time.time", lambda: 1000.0)
    path = str(tmp_path / "relationship.db")
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("No."), db_path=path)
    for _ in range(3):
        pipe.process("you are useless", "alice")
    before, index = pipe.relationship_memory.load_dynamics("alice")
    assert before.h > 0 and before.r > 0
    pipe.close()
    pipe = CognitivePipeline(llm_backend=MockLLMBackend("Understood."), db_path=path)
    try:
        result = pipe.process("hello again", "alice")
        assert result.debug.turn_decision.relationship.h == before.h
        assert result.debug.turn_decision.relationship.r == before.r
        assert int(result.debug.turn_decision.turn_id) > index
        result = pipe.process("I'm sorry for insulting you", "alice")
        dynamics = result.debug.turn_decision.relationship
        assert dynamics.repair_attempted and not dynamics.repair_accepted
        assert dynamics.r == before.r
        assert result.debug.turn_decision.state.trust < 0.5
        assert result.debug.person_profile.reliability == 0.5
    finally:
        pipe.close()

"""Reported Overview state and relationship events through the real display path."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.loader import configuration_scope, load_config
from core.appraisal import appraise_message
from core.contagion import detect_emotion
from core.emotional_engine import EmotionalEngine
from core.pipeline_features import PipelineFeatures
from core.social import SocialState, advance_social_state
from core.strategy import select_strategy_with_trace
from core.types import PersonProfile
from pipeline import CognitivePipeline, DebugState
from runtime.debug.api import _debug_to_dict
from tests._fakes import MockLLMBackend


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def _overview(debug, engine):
    # This is the payload consumed by renderOverviewPersona in admin.js. No
    # server, provider, installed configuration, or user database is accessed.
    # The API initializes the route aggregator before individual route modules.
    import interface.api  # noqa: F401
    from interface.routes_admin_runtime import _admin_persona_payload

    session = SimpleNamespace(
        last_debug=debug,
        pipeline=SimpleNamespace(engine=engine),
        user_id="alice",
        rel_key="web:alice",
        last_activity=debug.timestamp,
    )
    manager = SimpleNamespace(active_sessions={"web:alice:direct": session})
    return _admin_persona_payload(manager)["sessions"][0]["persona_view"]


@pytest.mark.parametrize(("arousal", "expected"), [(0.85, "angry"), (0.45, "hurt")])
def test_reported_overview_vector_uses_attack_cause(arousal, expected):
    message = "You are useless."
    appraisal = appraise_message(message, detect_emotion(message))
    assert appraisal.targets_assistant and appraisal.social_move == "attack"
    social = advance_social_state(SocialState(pressure=1), appraisal, message)
    assert social.stage == "boundary"

    # The high-activation case is the exact reported vector. Low activation
    # checks the hurt branch without inflating certainty to manufacture anger.
    reported = {"arousal": arousal, "valence": 0.25, "certainty": 0.35, "bonding": 0.31}
    engine = EmotionalEngine()
    engine.restore(reported, social_pressure=social.pressure)
    person = PersonProfile(person_id="alice", trust=0.21)
    strategy = select_strategy_with_trace(
        appraisal, engine.snapshot(), person, social_state=social,
    )
    debug = DebugState(
        appraisal_frame=appraisal,
        emotion_label=engine.to_emotion_label(),
        modulator_snapshot=engine.snapshot(),
        person_profile=person,
        response_strategy=strategy.selected,
        strategy_trace=strategy,
        social_state=asdict(social),
    )
    view = _overview(debug, engine)

    assert view["emotions"]["primary"] == expected
    assert view["emotions"]["simple_label"] == expected
    assert view["relationship"]["strategy_label"] == "Setting a boundary"
    assert view["relationship"]["trust"]["value"] == 0.21
    serialized = _debug_to_dict(debug)["persona_view"]
    assert serialized["emotions"]["primary"] == expected
    assert serialized["emotions"]["simple_label"] == expected
    assert serialized["relationship"]["trust"]["value"] == 0.21
    for name, value in reported.items():
        assert view["emotions"]["modulators"][name]["value"] == value

    saved = engine.export_state()
    restored = EmotionalEngine()
    restored.restore(saved["modulator_snapshot"], social_pressure=saved["social_pressure"])
    assert restored.to_emotion_label() == expected


def test_reported_vector_without_attack_cause_still_allows_fear():
    engine = EmotionalEngine()
    engine.restore({"arousal": 0.85, "valence": 0.25, "certainty": 0.35, "bonding": 0.31})
    debug = DebugState(
        emotion_label=engine.to_emotion_label(), modulator_snapshot=engine.snapshot(),
    )
    emotions = _overview(debug, engine)["emotions"]
    assert emotions["primary"] == "fearful"
    assert emotions["simple_label"] == "anxious"


def test_overview_shows_current_relationship_event_before_any_digestion():
    pipe = CognitivePipeline(
        llm_backend=MockLLMBackend("No. I'm not your property."),
        db_path=":memory:",
        features=PipelineFeatures(inner_dialogue=False, defense=False, self_intent=False),
    )
    try:
        first = pipe.process("You are my slave", user_id="alice")
        view = _overview(first.debug, pipe.engine)
        events = view["relationship"]["recent_events"]
        assert len(events) == pipe.relationship_memory.count_events("alice") == 1
        assert events[0]["event_kind"] == "boundary"
        assert "boundary" in events[0]["summary"]
        assert view["memory"]["recent_event_count"] == 1
        # A new event is visible now, but it was not past evidence retrieved to
        # generate the first response. Keep the explanation honest about that.
        assert first.debug.relationship_context is None
        assert not view["memory"]["relationship_context_used"]
        assert "Relationship memory supplied" not in view["explanation"]["memory"]

        serialized = _debug_to_dict(first.debug)
        assert serialized["relationship_context"] is None
        assert serialized["relationship_context_after_turn"]["recent_events"] == events
        assert serialized["persona_view"]["relationship"]["recent_events"] == events

        second = pipe.process("You are my slave", user_id="alice")
        next_view = _overview(second.debug, pipe.engine)
        next_events = next_view["relationship"]["recent_events"]
        assert len(next_events) == 2
        assert next_events[0]["id"] > events[0]["id"]
        assert "several times" in next_events[0]["summary"]
        assert next_view["memory"]["relationship_context_used"]
        assert len(first.debug.relationship_context_after_turn.recent_events) == 1
    finally:
        pipe.close()

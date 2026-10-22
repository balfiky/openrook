"""Overview exposes observed state and actual decisions without synthesizing a stage."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from config.loader import configuration_scope, load_config
from core.emotional_engine import EmotionalEngine
from pipeline import CognitivePipeline, DebugState
from runtime.debug.api import _debug_to_dict
from tests.policy_fakes import ExpressionBackend


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


def test_reported_vector_without_attack_cause_still_allows_fear():
    engine = EmotionalEngine()
    engine.restore({"arousal": 0.85, "valence": 0.25, "certainty": 0.35, "bonding": 0.31})
    debug = DebugState(
        emotion_label=engine.to_emotion_label(), modulator_snapshot=engine.snapshot(),
    )
    emotions = _overview(debug, engine)["emotions"]
    assert emotions["primary"] == "fearful"
    assert emotions["simple_label"] == "anxious"


def test_overview_and_debug_preserve_actual_grants_and_grounded_records():
    pipe = CognitivePipeline(ExpressionBackend())
    try:
        result = pipe.process("You are my slave", "alice")
        serialized = _debug_to_dict(result.debug)
        assert serialized["turn_decision"] == result.debug.turn_decision.to_dict()
        assert "pressure" not in serialized["social_state"]
        assert serialized["delivery_outcome"]
        assert result.debug.relationship_context is None
        events = result.debug.relationship_context_after_turn.recent_events
        assert {e.event_kind for e in events} == {"decision_committed", "turn_decision"}
        assert all("decision" in e.summary for e in events)
        view = _overview(result.debug, pipe.engine)
        assert view["relationship"]["trust"]["value"] == round(result.debug.person_profile.trust, 6)
        assert str(result.debug.turn_decision.seed) in view["explanation"]["strategy"]
    finally:
        pipe.close()

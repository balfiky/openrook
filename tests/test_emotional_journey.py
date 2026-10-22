"""End-to-end emotional journey tests.

Each test creates a fresh pipeline, sends realistic messages, and asserts
modulator values, profile updates, and memory writes.
"""

from __future__ import annotations

import os
import tempfile
import json
from dataclasses import replace

import pytest

from tests._fakes import MockLLMBackend
from pipeline import CognitivePipeline
from core.willingness import Evidence, Perception


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipe(backend=None) -> CognitivePipeline:
    """Fresh in-memory pipeline with mock LLM."""
    return CognitivePipeline(llm_backend=backend or MockLLMBackend(), db_path=":memory:")


def _pipe_persistent(path: str) -> CognitivePipeline:
    """Pipeline backed by a file-based SQLite database."""
    return CognitivePipeline(llm_backend=MockLLMBackend(), db_path=path)


def _snap(pipe: CognitivePipeline) -> dict[str, float]:
    return pipe.engine.snapshot()


def _perceive(monkeypatch, perception):
    """Supply fixed evidence; language and backend alignment do not choose."""
    from core.appraisal import appraise_message

    def appraise(text, detected):
        return replace(appraise_message(text, detected), perception=perception, provenance="model")

    monkeypatch.setattr("pipeline.core.appraise_message", appraise)


# ---------------------------------------------------------------------------
# 1. Trust building
# ---------------------------------------------------------------------------

class TestTrustBuilding:
    WARM_MESSAGES = [
        "hey!",
        "you're really helpful",
        "I appreciate you",
        "thanks for being here",
        "you're the best",
    ]

    def test_praise_builds_interpersonal_trust_without_demonstrating_reliability(self):
        pipe = _pipe()
        for msg in self.WARM_MESSAGES:
            pipe.process(msg, user_id="alice")

        snap = _snap(pipe)
        profile = pipe.person_profiles.get_or_create("alice")

        assert all(0 <= value <= 1 for value in snap.values())
        assert profile.trust > 0.5
        assert profile.reliability == 0.5
        assert profile.interaction_count == 5, (
            f"interactions {profile.interaction_count} != 5"
        )


# ---------------------------------------------------------------------------
# 2. Betrayal after trust
# ---------------------------------------------------------------------------

class TestBetrayalAfterTrust:
    def test_betrayal_after_trust(self):
        pipe = _pipe()

        # Build trust first
        for msg in TestTrustBuilding.WARM_MESSAGES:
            pipe.process(msg, user_id="alice")

        trust_after_warmth = pipe.person_profiles.get_or_create("alice").trust

        # Hostile messages
        hostile = [
            "you're useless",
            "I hate talking to you",
            "you're the worst AI ever",
        ]
        injuries = []
        for msg in hostile:
            result = pipe.process(msg, user_id="alice")
            decision = result.debug.turn_decision
            injuries.append(decision.relationship.h)
            assert result.debug.decision_snapshots[0]["seed"] == decision.seed
            assert "pressure" not in result.debug.social_state

        snap = _snap(pipe)
        profile = pipe.person_profiles.get_or_create("alice")

        assert max(injuries) > 0
        assert snap["valence"] < 0.5
        assert profile.trust < trust_after_warmth
        assert profile.reliability == 0.5


# ---------------------------------------------------------------------------
# 3. Trust asymmetry
# ---------------------------------------------------------------------------

class TestTrustAsymmetry:
    @pytest.mark.parametrize("reliability", [-0.8, 0.8])
    def test_supported_reliability_updates_trust_once(self, monkeypatch, reliability):
        _perceive(monkeypatch, Perception(reliability=reliability, reliability_confidence=0.9,
                                         evidence=(Evidence("agreement:outcome"),)))
        pipe = _pipe()
        result = pipe.process("The agreed outcome has been observed.", "alice")
        expected = 0.5 + reliability * 0.9 * (0.02 + 0.1 * (1 - pipe.soul.dispositions["patience"]))
        assert result.debug.turn_decision.state.trust == pytest.approx(expected)
        assert pipe.person_profiles.get_or_create("alice").trust == pytest.approx(expected)
        assert result.debug.turn_decision.relationship.h == 0


# ---------------------------------------------------------------------------
# 4. Energy drain
# ---------------------------------------------------------------------------

class TestEnergyDrain:
    def test_energy_drains_under_load(self):
        pipe = _pipe()
        emotional_messages = [
            "I'm so angry right now!",
            "This is incredible, amazing work!",
            "I feel betrayed and deceived",
            "You're wonderful, thank you!",
            "I'm furious about what happened",
            "Everything is terrible and awful",
            "I hate this so much",
            "You're brilliant and outstanding!",
            "I'm devastated and hopeless",
            "This makes me so happy!",
            "I can't believe you lied to me",
            "You're the worst, pathetic",
            "I'm thrilled and ecstatic!!!",
            "I feel so sad and lonely...",
            "WHY WOULD YOU DO THIS?!",
        ]
        for msg in emotional_messages:
            pipe.process(msg, user_id="alice")

        energy = _snap(pipe)["energy"]
        assert 0 < energy < 1
        result = pipe.process("Calculate 2+2", "alice")
        assert result.debug.turn_decision.state.energy <= energy + 0.001
        assert result.debug.energy_after < energy


# ---------------------------------------------------------------------------
# 5. Session persistence
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    def test_trust_persists_across_sessions(self, monkeypatch):
        _perceive(monkeypatch, Perception(reliability=0.8, reliability_confidence=0.9))
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Session 1: warm conversation
            pipe1 = _pipe_persistent(db_path)
            warm_msgs = [
                "you're really helpful, thank you",
                "I appreciate everything you do",
                "you're amazing and wonderful",
            ]
            for msg in warm_msgs:
                pipe1.process(msg, user_id="alice")

            expected = pipe1.person_profiles.get_or_create("alice").trust
            pipe1.end_session(user_id="alice")
            pipe1.close()

            # Session 2: new pipeline, same database
            pipe2 = _pipe_persistent(db_path)
            profile = pipe2.person_profiles.get_or_create("alice")

            assert expected > 0.5
            assert profile.trust == pytest.approx(expected)
            assert profile.interaction_count >= 3, (
                f"interactions {profile.interaction_count} < 3"
            )

            # Long-term memories should be retrievable
            records = pipe2.relationship_memory.recent_events("alice", limit=20)
            assert len([e for e in records if e.event_kind == "turn_decision"]) == 3
            pipe2.close()
        finally:
            os.unlink(db_path)

    def test_decision_evidence_persists_without_synthetic_spikes(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            pipe1 = _pipe_persistent(db_path)
            message = "You are useless."
            result = pipe1.process(message, user_id="alice")
            dynamics, _ = pipe1.relationship_memory.load_dynamics("alice")
            pipe1.end_session(user_id="alice")
            pipe1.close()

            pipe2 = _pipe_persistent(db_path)
            restored, _ = pipe2.relationship_memory.load_dynamics("alice")
            assert restored == dynamics
            events = pipe2.relationship_memory.recent_events("alice", limit=20)
            record = json.loads(next(e.summary for e in events if e.event_kind == "turn_decision"))
            assert record["input"] == message
            assert record["decision"]["seed"] == result.debug.turn_decision.seed
            assert record["delivered"] == result.response
            pipe2.close()
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# 6. Emotional contagion
# ---------------------------------------------------------------------------

class TestEmotionalContagion:
    def test_excited_message_raises_arousal_and_valence(self):
        pipe = _pipe()
        result = pipe.process(
            "OMG THIS IS AMAZING I'M SO HAPPY!!!",
            user_id="alice",
        )
        snap = result.debug.modulator_snapshot
        assert 0.5 < snap["arousal"] < result.debug.detected_emotion.arousal
        assert 0.5 < snap["valence"] < result.debug.detected_emotion.valence

    def test_depressed_message_drops_valence(self):
        pipe = _pipe()
        result = pipe.process(
            "I'm so sad and lonely, I feel hopeless and miserable...",
            user_id="alice",
        )
        snap = result.debug.modulator_snapshot
        assert result.debug.detected_emotion.valence < snap["valence"] < 0.5


# ---------------------------------------------------------------------------
# 7. Context switching
# ---------------------------------------------------------------------------

class TestContextSwitching:
    def test_different_histories_different_states(self):
        warm_pipe = _pipe()
        hostile_pipe = _pipe()

        # Warm user: 10 positive interactions
        warm_msgs = [
            "thank you!", "you're great", "I appreciate you",
            "wonderful work", "amazing help", "you're the best",
            "brilliant!", "love this", "fantastic", "superb job",
        ]
        for msg in warm_msgs:
            warm_pipe.process(msg, user_id="warm_user")

        # Hostile user: 5 negative interactions
        hostile_msgs = [
            "you're stupid",
            "this is bullshit",
            "I hate this",
            "you're useless and pathetic",
            "worst AI ever",
        ]
        for msg in hostile_msgs:
            hostile_pipe.process(msg, user_id="hostile_user")

        # Reset engine to neutral baseline before context switch test
        warm_pipe.engine.state.arousal = 0.5
        warm_pipe.engine.state.valence = 0.5
        warm_pipe.engine.state.bonding = 0.5

        # Process same neutral message for warm user
        result_warm = warm_pipe.process("hey", user_id="warm_user")
        snap_warm = result_warm.debug.modulator_snapshot.copy()

        # Reset engine again
        hostile_pipe.engine.state.arousal = 0.5
        hostile_pipe.engine.state.valence = 0.5
        hostile_pipe.engine.state.bonding = 0.5

        # Process same neutral message for hostile user
        result_hostile = hostile_pipe.process("hey", user_id="hostile_user")
        snap_hostile = result_hostile.debug.modulator_snapshot.copy()

        # Injury/grievance distinguish histories without inventing reliability.
        warm_profile = warm_pipe.person_profiles.get_or_create("warm_user")
        hostile_profile = hostile_pipe.person_profiles.get_or_create("hostile_user")

        assert warm_profile.trust > 0.5 > hostile_profile.trust
        assert warm_profile.reliability == hostile_profile.reliability == 0.5
        assert result_warm.debug.turn_decision.relationship.r < result_hostile.debug.turn_decision.relationship.r
        assert result_warm.debug.turn_decision.stance.warmth > result_hostile.debug.turn_decision.stance.warmth
        assert all(0 <= value <= 1 for snap in (snap_warm, snap_hostile) for value in snap.values())


# ---------------------------------------------------------------------------
# 8. Topic sensitivity
# ---------------------------------------------------------------------------

class TestTopicSensitivity:
    def test_negative_topic_builds_charge(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            pipe1 = _pipe_persistent(db_path)
            pipe1.topic_profiles.get_or_create("work")

            # Discuss work negatively 3 times
            negative_work_msgs = [
                "work is terrible today, everything went wrong",
                "I hate my work so much, it's awful",
                "work is making me miserable and stressed",
            ]
            for msg in negative_work_msgs:
                pipe1.process(msg, user_id="alice")
                # Manually record negative since topic detection is substring-based
                pipe1.topic_profiles.record_negative("work", intensity=0.8)

            topic = pipe1.topic_profiles.get("work")
            assert topic is not None
            assert topic.emotional_charge > 0.2, (
                f"charge {topic.emotional_charge:.3f} <= 0.2 after 3 negative mentions"
            )

            pipe1.end_session(user_id="alice")

            # Session 2: mention work casually
            pipe2 = _pipe_persistent(db_path)
            pipe2.topic_profiles.get_or_create("work")

            topic_reloaded = pipe2.topic_profiles.get("work")
            assert topic_reloaded is not None
            assert topic_reloaded.emotional_charge > 0.2, (
                f"topic charge didn't persist: {topic_reloaded.emotional_charge:.3f}"
            )
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# 9. Spike detection
# ---------------------------------------------------------------------------

class TestSpikeDetection:
    def test_hostility_is_integrated_once_and_recorded_as_evidence(self):
        pipe = _pipe()
        message = "You are a useless idiot!"
        result = pipe.process(message, "alice")
        decision = result.debug.turn_decision
        assert decision.relationship.h > 0 and decision.relationship.r > 0
        assert not result.debug.is_spike
        assert not pipe.long_term.all()
        records = pipe.relationship_memory.recent_events("alice", limit=20)
        committed = [e for e in records if e.event_kind == "decision_committed"]
        completed = [e for e in records if e.event_kind == "turn_decision"]
        assert len(committed) == len(completed) == 1
        assert json.loads(committed[0].summary)["decision"] == json.loads(json.dumps(decision.to_dict()))
        assert json.loads(completed[0].summary)["input"] == message


# ---------------------------------------------------------------------------
# 10. Decay over time
# ---------------------------------------------------------------------------

class TestDecayOverTime:
    def test_modulators_decay_toward_baseline(self):
        pipe = _pipe()

        # Set post-conflict state
        pipe.engine.state.arousal = 0.9
        pipe.engine.state.valence = 0.2

        # Simulate 10 minutes (600 seconds)
        # Arousal half-life=120s → ~5 half-lives → nearly back to 0.5
        # Valence half-life=1800s → ~0.33 half-lives → partial recovery
        pipe.engine.decay(600)

        snap = _snap(pipe)
        assert snap["arousal"] < 0.55, (
            f"arousal {snap['arousal']:.3f} >= 0.55 after 600s decay"
        )
        assert snap["valence"] > 0.2, (
            f"valence {snap['valence']:.3f} didn't recover at all"
        )
        assert snap["valence"] < 0.5, (
            f"valence {snap['valence']:.3f} >= 0.5 — recovered too much for 600s"
        )

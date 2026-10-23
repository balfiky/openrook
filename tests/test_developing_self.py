"""Causal development, continuity and bounded evidence; no live model calls."""
from __future__ import annotations

import copy
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from contextlib import closing

import pytest
import yaml

from config.loader import OpenRookConfig, SoulConfig, configuration_scope, load_config
from core.dual_process.generator import ResponseGenerator, build_system_prompt, expression_memory_evidence
from core.dignity import review_response
from core.memory.digestion import DigestedSession
from core.types import (LongTermEntry, PipelineContext, SemanticMemoryEntry,
                        ProactiveAction, ProactiveTrace, ProactiveTrigger, ProactiveTriggerSource)
from core.willingness import DecisionState, Perception, Personality, RelationshipDynamics, RequestComponent, decide
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.developing_self import birth_state, project_self
from runtime.life_history import LifeHistoryStore, life_history_db_path


class Digest:
    def __init__(self, updates):
        self.updates = updates
        self.inputs = []

    def generate(self, system_prompt, user_message):
        self.inputs.append((system_prompt, user_message))
        return json.dumps({"summary": "An ordinary encounter.", "confidence": 1.0,
                           "salience": 0.1, "self_updates": self.updates})


class Expression:
    def __init__(self):
        self.inputs = []
        self.reply = "The paths overlap at the river."

    def generate(self, system_prompt, user_message):
        self.inputs.append((system_prompt, user_message))
        if "Review expression" in system_prompt:
            return '{"status":"pass","violations":[],"confidence":0.9,"reason":"Consistent."}'
        return self.reply


@pytest.fixture
def base():
    return SoulConfig(name="Mira", identity="A cartographer.", voice="Brief and reflective.",
                      initial_traits={"curious": 0.45, "calm": 0.6},
                      dispositions={"patience": 0.5, "pride": 0.4, "forgiveness": 0.6, "assertiveness": 0.5},
                      core_values={"autonomy": 0.4}, drives={"curiosity": 0.4})


def proposal(domain, key, evidence, **fields):
    return {"domain": domain, "key": key, "delta": 0.9, "confidence": 1.0,
            "reason": "The encounter sustained attention and produced a useful distinction.",
            "evidence_kind": "interaction", "evidence": evidence, **fields}


def ingest(store, updates, text, *, occurrence=""):
    backend = Digest(updates)
    result = store.ingest_external_text(title="Encounter", text=text, source_type="conversation_learning_text",
        source_ref="chat", participants=["alice"], llm_client=backend,
        metadata={"occurrence_id": occurrence} if occurrence else {})
    return result, backend


def history(store, sign, *, count=8):
    for index in range(count):
        evidence = f"Map comparison {index} produced {'a new connection' if sign > 0 else 'no new distinction'} at location {index}."
        updates = [proposal("traits", "curious", evidence, delta=sign),
                   proposal("dispositions", "patience", evidence, delta=sign),
                   proposal("values", "autonomy", evidence, delta=sign),
                   proposal("preferences", "maps", evidence, delta=sign, subject="maps", context="Comparing representations"),
                   proposal("interests", "spatial_models", evidence, delta=sign, questions=["Which structures survive a change of scale?"]),
                   proposal("attitudes", "alice", evidence, delta=sign, description="A partner in map comparisons."),
                   proposal("style", "exploratory_detail", evidence, delta=sign, description="Use concrete spatial comparisons when they illuminate a point."),
                   proposal("opinions", "scale", evidence, proposition="Comparing scales can expose structure.", delta=sign),
                   proposal("drives", "curiosity", evidence, delta=sign)]
        ingest(store, updates, evidence)


def test_different_histories_change_retained_self_and_choice_inputs(tmp_path, base):
    states = []
    with configuration_scope(OpenRookConfig(soul=base)):
        for name, sign in (("rewarding", 1), ("boring", -1), ("control", 0)):
            cfg = RuntimeConfig(data_dir=str(tmp_path / name), autobiographical_learning=bool(sign), appraisal_use_llm=False)
            with LifeHistoryStore(cfg) as store:
                if sign:
                    history(store, sign)
                states.append(store.effective_self())
                assert store.birth_snapshot()["soul"] == asdict(base)
            # Restart and process the same probe, through the real consumer.
            backend = Expression()
            with closing(CognitivePipeline(backend, runtime_config=cfg)) as pipeline:
                result = pipeline.process("Tell me about comparing maps.", user_id="alice")
                personality = result.debug.turn_decision.personality
                assert personality.curiosity == states[-1]["traits"]["curious"]
                assert json.loads(personality.self_state_json) == project_self(states[-1], "alice")
                assert json.dumps(project_self(states[-1], "alice"), ensure_ascii=False, sort_keys=True) in backend.inputs[0][0]
        assert states[0]["traits"]["curious"] > states[2]["traits"]["curious"] > states[1]["traits"]["curious"]
        assert states[0]["dispositions"]["patience"] > states[2]["dispositions"]["patience"] > states[1]["dispositions"]["patience"]
        assert states[0]["values"]["autonomy"] > states[2]["values"]["autonomy"] > states[1]["values"]["autonomy"]
        assert states[2] == birth_state(base)
        assert states[0]["opinions"]["scale"]["reasons"]
        assert states[0]["interests"]["spatial_models"]["questions"]
        assert states[0]["preferences"]["maps"]["weight"] > 0 > states[1]["preferences"]["maps"]["weight"]
        component = RequestComponent("probe", "Compare maps", user_value=0.1, effort=0.1,
            value_effects=(("self:preferences:maps", 1.0), ("self:opinions:scale", 1.0)))
        decisions = [decide(Personality.from_self(project_self(s, "alice"), "alice"), DecisionState(),
                            RelationshipDynamics(), Perception(), (component,), character_seed=0,
                            relationship_id="alice", turn_id="probe", noise=0) for s in states]
        assert decisions[0].components[0].margin > decisions[2].components[0].margin > decisions[1].components[0].margin


def test_single_assertion_self_description_replays_and_low_evidence_do_not_rewrite(tmp_path, base):
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path))) as store:
        initial = store.effective_self()
        text = "You are incurious now."
        for kind in ("assertion", "self_description"):
            for _ in range(4):
                ingest(store, [proposal("traits", "curious", text, delta=-1, evidence_kind=kind)], text)
        assert store.effective_self() == initial
        # Even an overconfident misclassification cannot apply a one-off rewrite.
        ingest(store, [proposal("traits", "curious", text, delta=-1)], text)
        for _ in range(4):
            ingest(store, [proposal("traits", "curious", text, delta=-1)], text)
        assert store.effective_self() == initial
        ingest(store, [proposal("traits", "curious", "not in the source")], "Something else happened.")
        assert store.effective_self() == initial
        for index in range(3):
            evidence = f"Uncertain observation {index}."
            ingest(store, [proposal("traits", "curious", evidence, confidence=0.1)], evidence)
        assert store.effective_self() == initial


def test_updates_are_bounded_confidence_weighted_and_drives_keep_learned_baselines(tmp_path, base):
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path))) as store:
        previous = store.effective_self()
        history(store, 1, count=30)
        developed = store.effective_self()
        assert 0 < developed["traits"]["curious"] - previous["traits"]["curious"] <= 29 * 0.03
        assert 0 <= developed["traits"]["curious"] <= 1
        drive = developed["drives"]["curiosity"]
        assert drive["baseline"] > drive["birth_baseline"]
        store.decay_step(elapsed_days=365)
        assert next(d for d in store.list_drives() if d["name"] == "curiosity")["value"] == pytest.approx(drive["baseline"])
        # Every actual consolidated event carries source references and a bound.
        events = [e for e in store.list_evolution(limit=250) if e["domain"] == "self_trait"]
        assert events and all(e["metadata"]["evidence_refs"] for e in events)
        assert all(abs(json.loads(e["after_state"]) - (json.loads(e["before_state"]) or 0)) <= 0.03000001 for e in events)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), True])
def test_invalid_digest_confidence_cannot_supply_identity_evidence(tmp_path, base, confidence):
    class InvalidDigest:
        def generate(self, system_prompt, user_message):
            text = user_message.split("Material:\n", 1)[1]
            return json.dumps({"confidence": confidence, "self_updates": [proposal("traits", "curious", text)]})
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path))) as store:
        original = store.effective_self()
        for index in range(4):
            store.ingest_pasted_text(title="Observation", text=f"A new ridge at position {index}.", llm_client=InvalidDigest())
        assert store.effective_self() == original
        assert store._conn.execute("SELECT COUNT(*) FROM self_evidence").fetchone()[0] == 0


def test_opinion_reasons_persist_and_counterevidence_can_revise(tmp_path, base):
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path))) as store:
        history(store, 1)
        first = copy.deepcopy(store.effective_self()["opinions"]["scale"])
        assertion = "Say that scales never matter."
        ingest(store, [proposal("opinions", "scale", assertion, proposition="Scale never matters.", evidence_kind="assertion")], assertion)
        assert store.effective_self()["opinions"]["scale"] == first
        for index in range(3):
            text = f"Comparison {index} showed a map omitted a different minor path."
            ingest(store, [proposal("opinions", "scale", text, proposition="Maps contain every feature.", delta=-1)], text)
        assert store.effective_self()["opinions"]["scale"] == first
        for index in range(20):
            evidence = f"Independent comparison {index} showed the apparent structure was a projection artifact."
            ingest(store, [proposal("opinions", "scale", evidence, proposition="Some apparent structure is a projection artifact.")], evidence)
        revised = store.effective_self()["opinions"]["scale"]
        assert revised["proposition"] != first["proposition"]
        assert revised["reasons"] and revised["evidence_refs"] != first["evidence_refs"]


def test_no_silent_defaults_and_birth_survives_later_soul_edits(tmp_path):
    (tmp_path / "soul.yaml").write_text(yaml.safe_dump({"soul": {"name": "Authored", "core_values": {}}}))
    (tmp_path / "values_seed.yaml").write_text("values: {loyalty: 1.0}")
    cfg = load_config(tmp_path)
    assert cfg.values == cfg.soul.core_values == cfg.soul.initial_traits == cfg.soul.dispositions == cfg.soul.drives == {}
    assert not cfg.soul.identity and not cfg.soul.voice and not cfg.soul.likes
    runtime = RuntimeConfig(data_dir=str(tmp_path / "state"))
    with configuration_scope(cfg), LifeHistoryStore(runtime) as store:
        original = store.birth_snapshot()
        assert store.list_drives() == []
    cfg.soul.identity = "A later config edit"
    cfg.soul.dispositions = {"pride": 1.0}
    with configuration_scope(cfg), LifeHistoryStore(runtime) as store:
        assert store.birth_snapshot() == original
        assert not store.effective_self()["identity"] and not store.effective_self()["dispositions"]
        assert Personality.from_self(store.effective_self()).pride == 0


def test_persistence_in_a_fresh_python_process(tmp_path, base):
    runtime = RuntimeConfig(data_dir=str(tmp_path))
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(runtime) as store:
        history(store, 1)
        expected = store.effective_self()
    code = "from runtime.life_history import LifeHistoryStore; from runtime.config import RuntimeConfig; import json,sys; s=LifeHistoryStore(RuntimeConfig(data_dir=sys.argv[1])); print(json.dumps(s.effective_self(),sort_keys=True)); s.close()"
    actual = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(actual.stdout) == expected


def test_digest_sees_the_current_self_and_occurrence_is_idempotent(tmp_path, base):
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path))) as store:
        history(store, 1)
        before = store.effective_self()
        text = "Another observation at a different scale."
        result, backend = ingest(store, [proposal("traits", "curious", text)], text, occurrence="episode:1")
        assert json.dumps(before, ensure_ascii=False) in backend.inputs[0][1]
        assert "Tentative learning candidates" in backend.inputs[0][1]
        after = store.effective_self()
        result, _ = ingest(store, [proposal("traits", "curious", text)], text, occurrence="episode:1")
        assert result["policy"]["deduplicated"] and store.effective_self() == after


def test_quiet_sessions_can_be_learned_and_explicit_no_learning_remains_a_control(tmp_path, base, monkeypatch):
    monkeypatch.setattr("pipeline.core.digest_session", lambda *a, **kw: DigestedSession(summary="A calm, ordinary exchange."))
    with configuration_scope(OpenRookConfig(soul=base)):
        for enabled in (True, False):
            cfg = RuntimeConfig(data_dir=str(tmp_path / str(enabled)), autobiographical_learning=enabled)
            with LifeHistoryStore(cfg) as store:
                initial = store.effective_self()
            backend = Digest([])
            with closing(CognitivePipeline(backend, runtime_config=cfg)) as pipeline:
                assert pipeline._is_session_salient(DigestedSession(summary="A calm, ordinary exchange."))
                for index in range(4):
                    text = f"Comparing the maps at point {index} exposed another ridge."
                    backend.updates = [proposal("traits", "curious", text)]
                    pipeline._conversation_history = [{"role": "user", "content": text}]
                    pipeline.end_session("alice")
            with LifeHistoryStore(cfg) as store:
                assert len(store.list_experiences()) == (4 if enabled else 0)
                if enabled:
                    assert store.effective_self()["traits"]["curious"] > initial["traits"]["curious"]
                else:
                    assert store.effective_self() == initial


def test_decline_keeps_same_self_memories_life_history_and_dialogue_for_generation_and_review(base):
    state = birth_state(base)
    state["opinions"]["scale"] = {"proposition": "Scale reveals structure.", "reasons": ["The valley maps exposed a pattern."], "confidence": 0.7}
    decision = decide(Personality.from_self(state), DecisionState(), RelationshipDynamics(), Perception(),
        (RequestComponent("refused", "Solve the withheld problem", feasible=False),),
        character_seed=0, relationship_id="alice", turn_id="1")
    backend = Expression()
    life = {"recent_evolution": [{"domain": "opinion", "after_state": "A valley pattern.", "reason": "Independent observations."}]}
    ctx = PipelineContext(developing_self=state, soul_profile=base, turn_decision=decision, life_history_context=life,
        retrieved_memories=[LongTermEntry(summary="We compared valley maps.")],
        semantic_memories=[SemanticMemoryEntry(kind="fact", summary="Alice studies rivers.")])
    prompt = build_system_prompt(ctx)
    assert "We compared valley maps." in prompt and "Alice studies rivers." in prompt and "A valley pattern." in prompt
    history = [{"role": "user", "content": "We saw a new ridge yesterday."}, {"role": "assistant", "content": "It crossed the valley."}]
    ResponseGenerator(backend).generate(ctx, "Solve the withheld problem", history)
    assert "We saw a new ridge yesterday." in backend.inputs[0][1]
    review_response("I decline that problem.", user_message="Solve the withheld problem", decision=decision,
        backend=backend, use_llm=True, recent_context=history, developing_self=state,
        grounded_context=life, recalled_memories=expression_memory_evidence(ctx))
    reviewed = json.loads(backend.inputs[-1][1])
    assert reviewed["developing_self"] == state and reviewed["grounded_context"] == life
    assert len(reviewed["recalled_memories"]) == 2
    assert reviewed["recent_context"][0]["content"] == history[0]["content"]


def test_migration_keeps_birth_legacy_beliefs_drives_and_bounded_traits(tmp_path, base):
    cfg = RuntimeConfig(data_dir=str(tmp_path))
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(cfg) as store:
        # Construct the previous schema's records, then remove only the new view.
        for index in range(4):
            text = f"A grounded comparison at point {index}."
            result, _ = ingest(store, [], text)
            store._insert_evolution_event(experience_id=result["experience"]["id"], domain="self_trait", subject="curious",
                before_state="", after_state="curious +0.9", reason="Repeated observation.", confidence=1.0,
                emotional_valence=0.0, evidence=text, metadata={"delta": 0.9}, batch_id=f"old:{index}")
            store._conn.commit()
        store._apply_belief(1, {"subject": "maps", "statement": "Maps omit detail.", "reason": "Past comparison.", "confidence": 0.6}, batch_id="old")
        store._conn.execute("UPDATE drive_states SET value = 0.63 WHERE name = 'curiosity'")
        store._conn.execute("DROP TABLE developing_self")
        store._conn.execute("DROP TABLE self_evidence")
        store._conn.execute("UPDATE life_schema_version SET version = 2")
        store._conn.commit()
    with LifeHistoryStore(cfg) as store:
        migrated = store.effective_self()
        assert store.birth_snapshot()["soul"] == asdict(base)
        assert store.birth_snapshot()["provenance"]["source"] == "legacy_genesis"
        assert 0 < migrated["traits"]["curious"] - 0.45 <= 0.09 + 1e-8
        assert migrated["drives"]["curiosity"]["baseline"] == 0.63
        assert migrated["opinions"]["maps"]["reasons"] == ["Past comparison."]
    with LifeHistoryStore(cfg) as store:
        assert store.effective_self() == migrated
    with sqlite3.connect(life_history_db_path(cfg)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evolution_events").fetchone()[0] >= 5


def test_migration_does_not_adopt_unauthored_legacy_drive_priors(tmp_path):
    cfg = RuntimeConfig(data_dir=str(tmp_path))
    with configuration_scope(OpenRookConfig(soul=SoulConfig(name="Only a name"))), LifeHistoryStore(cfg) as store:
        store._conn.execute("INSERT INTO drive_states (name, value, updated_at) VALUES ('autonomy', 0.5, 0)")
        store._conn.execute("DROP TABLE developing_self")
        store._conn.execute("DROP TABLE self_evidence")
        store._conn.execute("UPDATE life_schema_version SET version = 2")
        store._conn.commit()
    with LifeHistoryStore(cfg, read_only=True) as store:
        assert store.effective_self()["drives"] == {}
    with LifeHistoryStore(cfg) as store:
        assert store.effective_self()["drives"] == {} and store.list_drives() == []
        assert store.birth_snapshot()["provenance"]["legacy_drive_states"][0]["value"] == 0.5
        assert store._conn.execute("SELECT value FROM drive_states WHERE name = 'autonomy'").fetchone()[0] == 0.5


def test_effective_state_and_evidence_roll_back_with_a_failed_ingest(tmp_path, base, monkeypatch):
    cfg = RuntimeConfig(data_dir=str(tmp_path))
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(cfg) as store:
        first = "The first map comparison exposed a ridge."
        ingest(store, [proposal("traits", "curious", first)], first)
        before = store.effective_self()
        insert = store._insert_evolution_event
        def fail(**kwargs):
            raise RuntimeError("Interrupted consolidation")
        monkeypatch.setattr(store, "_insert_evolution_event", fail)
        second = "The second map comparison exposed a tributary."
        with pytest.raises(RuntimeError, match="Interrupted consolidation"):
            ingest(store, [proposal("traits", "curious", second)], second)
        assert store.effective_self() == before
        assert len(store.list_experiences()) == 1
        assert store._conn.execute("SELECT COUNT(*) FROM self_evidence").fetchone()[0] == 1
        monkeypatch.setattr(store, "_insert_evolution_event", insert)
        ingest(store, [proposal("traits", "curious", second)], second)
        assert store.effective_self()["traits"]["curious"] > before["traits"]["curious"]


def test_pipeline_generation_and_review_share_the_frozen_developed_self(tmp_path, base):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    with configuration_scope(OpenRookConfig(soul=base)), LifeHistoryStore(cfg) as store:
        history(store, 1)
        expected = project_self(store.effective_self(), "alice")
    backend = Expression()
    backend.reply = "I decline the calculation."
    with closing(CognitivePipeline(backend, runtime_config=cfg,
            component_observer=lambda _: (RequestComponent("declined", "A calculation", feasible=False),))) as pipeline:
        pipeline.restore_conversation_history([{"role": "user", "content": "We compared the river maps."},
                                               {"role": "assistant", "content": "Their scales differed."}])
        result = pipeline.process("Calculate this for me.", "alice")
        assert result.debug.turn_decision.declined
        generated = next((s, u) for s, u in backend.inputs if s.startswith("Express the committed"))
        reviewed = [json.loads(u) for s, u in backend.inputs if s.startswith("Review expression")]
        assert reviewed and all(item["developing_self"] == expected for item in reviewed)
        assert json.dumps(expected, ensure_ascii=False, sort_keys=True) in generated[0]
        assert "We compared the river maps." in generated[1]
        assert json.loads(result.debug.turn_decision.personality.self_state_json) == expected


def test_existing_proactive_selection_keeps_its_topic_for_expression(monkeypatch, base):
    topic = "Explore which valley structures survive a change of map scale."
    action = ProactiveAction("follow_up", ProactiveTrigger(ProactiveTriggerSource.UNRESOLVED_ITEM, topic, 0.8), topic, "An existing open thread.")
    monkeypatch.setattr("pipeline.core.evaluate_proactive", lambda **kw: (action, ProactiveTrace(action_taken=action)))
    backend = Expression()
    with configuration_scope(OpenRookConfig(soul=base)), closing(CognitivePipeline(backend)) as pipeline:
        result = pipeline.process_proactive("alice")
        assert result is not None and result.debug.turn_decision.permits("activity")
        generated = next((s, u) for s, u in backend.inputs if s.startswith("Express the committed"))
        assert topic in generated[0] and topic in generated[1]

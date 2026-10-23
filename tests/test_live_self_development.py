"""Ordinary live-session learning, independent evidence and restart continuity."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import json

import pytest

from config.loader import OpenRookConfig, SoulConfig, configuration_scope
from core.inference import inference_turn
from core.self_intent import conversation_proposals
from core.willingness import DecisionState, Personality, RelationshipDynamics, Perception, SelfProposal, decide
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.developing_self import birth_state
from runtime.life_history import LifeHistoryStore


class EpisodeBackend:
    """Controlled interpreter; no claim about live semantic accuracy."""
    def __init__(self, subject="maps", sign=1, *, evidence_kind="interaction", legacy=False, self_quote=False):
        self.subject, self.sign = subject, sign
        self.evidence_kind, self.legacy, self.self_quote = evidence_kind, legacy, self_quote
        self.calls, self.expressions = [], 0

    def generate(self, system_prompt, user_message):
        self.calls.append((system_prompt, user_message))
        if system_prompt.startswith("Interpret the supplied interaction episode"):
            material = json.loads(user_message.split("Material:\n", 1)[1])
            dialogue = material["dialogue"]
            quote = next(item["content"] for item in dialogue if item["role"] == ("assistant" if self.self_quote else "user"))
            if self.legacy:
                return json.dumps({"summary": "A comparison was made.", "confidence": 1,
                    "beliefs": [{"subject": self.subject, "statement": "Representations differ.", "confidence": 1}]})
            common = {"confidence": 1, "delta": self.sign, "reason": "Independent comparisons exposed a recurring distinction.",
                      "evidence": quote, "evidence_kind": self.evidence_kind}
            return json.dumps({"summary": "A recorded comparison.", "confidence": 1, "self_updates": [
                {**common, "domain": "traits", "key": "curious"},
                {**common, "domain": "interests", "key": self.subject,
                 "questions": [f"Which structures persist in {self.subject} when representation changes?"]},
                {**common, "domain": "opinions", "key": self.subject,
                 "proposition": f"Comparing {self.subject} can expose omitted structure."}]})
        if system_prompt.startswith("Review expression against"):
            return '{"status":"pass","violations":[],"confidence":0.9}'
        self.expressions += 1
        return f"A distinct observation at position {self.expressions} changes which structures remain visible."


@pytest.fixture
def soul():
    return SoulConfig(name="Vale", initial_traits={"curious": 0.5, "calm": 0.8},
        dispositions={"patience": 0.8, "pride": 0.4}, core_values={"kindness": 0.8, "honesty": 0.9},
        drives={"curiosity": 0.5})


def observe(pipeline, index, subject="maps"):
    return pipeline.process(f'Comparison {index} in {subject} retained the ridge but omitted stream "{index}".\nThe survey still contains it.', user_id="alice")


def test_ordinary_interaction_develops_retained_self_and_choice_without_session_end(tmp_path, soul):
    retained = []
    with configuration_scope(OpenRookConfig(soul=soul)):
        for name, subject, sign, enabled in (("map", "maps", 1, True), ("sound", "acoustics", 1, True),
                                               ("boring", "maps", -1, True), ("control", "maps", 1, False)):
            cfg = RuntimeConfig(data_dir=str(tmp_path / name), appraisal_use_llm=False, autobiographical_learning=enabled)
            backend = EpisodeBackend(subject, sign)
            with closing(CognitivePipeline(backend, runtime_config=cfg)) as pipeline:
                for index in range(8):
                    result = observe(pipeline, index, subject)
                    assert result.response and len(result.debug.llm_calls) <= 5
                    if enabled and index % 2:
                        development = result.debug.self_development
                        assert development["status"] == "digested" and development["accepted_evidence"] == 3
                        assert development["self_updates_provided"]
                        assert result.debug.llm_calls[-1]["phase"] == "self_development"
                        assert result.debug.llm_calls[-1]["optional"]
                        # Development affects the next choice, never the already frozen turn.
                        before = json.loads(result.debug.turn_decision.personality.self_state_json)["revision"]
                        assert development["revision_before"] == before
                history = pipeline.export_conversation_history()
                assert len(history) == 16  # Consolidation never resets the active conversation.
            with LifeHistoryStore(cfg) as store:
                state = store.effective_self()
                retained.append(state)
                assert len(store.list_experiences()) == (4 if enabled else 0)
                if enabled:
                    assert 0 < abs(state["traits"]["curious"] - 0.5) <= 3 * 0.03000001
                    assert all(e["metadata"]["development"]["accepted_evidence"] == 3 for e in store.list_experiences())
            with closing(CognitivePipeline(EpisodeBackend(subject, sign), runtime_config=cfg)) as resumed:
                result = resumed.process("Continue the comparison.", user_id="alice")
                assert json.loads(result.debug.turn_decision.personality.self_state_json)["revision"] == state["revision"]
                assert result.debug.turn_decision.personality.curiosity == state["traits"]["curious"]
        assert retained[-1] == birth_state(soul)
        assert retained[0]["traits"]["curious"] > retained[-1]["traits"]["curious"] > retained[2]["traits"]["curious"]
        assert "maps" in retained[0]["interests"] and "acoustics" in retained[1]["interests"]
        proposals = [conversation_proposals(state, {}, (), 1) for state in retained]
        assert any("maps" in p.content for p in proposals[0])
        assert any("acoustics" in p.content for p in proposals[1])
        assert not proposals[-1]
        choices = [decide(Personality.from_self(state), DecisionState(), RelationshipDynamics(), Perception(), (),
                          proposals=proposed, character_seed=0, relationship_id="fixed", turn_id="probe", noise=0)
                   for state, proposed in zip(retained, proposals)]
        assert choices[0].selected_intention and choices[1].selected_intention
        assert choices[0].selected_intention.content != choices[1].selected_intention.content


@pytest.mark.parametrize("evidence_kind,self_quote", [("assertion", False), ("self_description", False), ("interaction", True)])
def test_live_evidence_cannot_promote_assertions_or_generated_self_expression(tmp_path, soul, evidence_kind, self_quote):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    with configuration_scope(OpenRookConfig(soul=soul)):
        backend = EpisodeBackend(evidence_kind=evidence_kind, self_quote=self_quote)
        with closing(CognitivePipeline(backend, runtime_config=cfg)) as pipeline:
            for index in range(8):
                result = observe(pipeline, index)
            assert result.debug.self_development["accepted_evidence"] == 0
        with LifeHistoryStore(cfg) as store:
            assert store.effective_self() == birth_state(soul)


def test_missing_development_contract_is_visible_and_cannot_mutate_identity(tmp_path, soul):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(EpisodeBackend(legacy=True), runtime_config=cfg)) as pipeline:
        for index in range(4):
            result = observe(pipeline, index)
        assert result.debug.self_development["status"] == "undigested"
        assert not result.debug.self_development["self_updates_provided"]
        assert result.debug.self_development["accepted_evidence"] == result.debug.self_development["changes"] == 0
        with LifeHistoryStore(cfg) as store:
            assert store.effective_self() == birth_state(soul)
            assert not store.list_open_questions()  # Failure is diagnostic, not a newly authored interest.


def test_pending_live_episode_survives_restart_and_is_not_reingested_at_session_end(tmp_path, soul, monkeypatch):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    with configuration_scope(OpenRookConfig(soul=soul)):
        with closing(CognitivePipeline(EpisodeBackend(), runtime_config=cfg)) as first:
            observe(first, 1)
            saved = first.export_session_state()
            history = first.export_conversation_history()
            assert len(saved["development_history"]) == 1
        with closing(CognitivePipeline(EpisodeBackend(), runtime_config=cfg)) as second:
            second.restore_conversation_history(history)
            second.restore_session_state(saved)
            result = observe(second, 2)
            assert result.debug.self_development["accepted_evidence"] == 3
            assert not second.export_session_state()["development_history"]
            from core.memory.digestion import DigestedSession
            monkeypatch.setattr("pipeline.core.digest_session", lambda *args, **kwargs: DigestedSession(summary="A summary of the same episode."))
            second.end_session("alice")
        with LifeHistoryStore(cfg) as store:
            assert len(store.list_experiences()) == 1
            assert store._conn.execute("SELECT COUNT(*) FROM self_evidence").fetchone()[0] == 3


def test_learning_yields_to_exhausted_expression_budget_and_keeps_pending_evidence(tmp_path, soul):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(EpisodeBackend(), runtime_config=cfg)) as pipeline:
        from pipeline.types import _TurnState, DebugState
        from core.dual_process.generator import GenerationResult
        st = _TurnState("A separate transect retained a ridge.", "alice", DebugState(delivery_outcome="delivered"),
                        gen_result=GenerationResult(response="A ridge remains visible."))
        with inference_turn([], max_calls=0):
            pipeline._record_live_development(st)
            pipeline._record_live_development(st)
        assert st.debug.self_development == {"status": "pending", "pending_turns": 2}
        assert len(pipeline.export_session_state()["development_history"]) == 2
        assert not pipeline._llm_backend.calls


def test_selected_substantive_intention_is_not_silence_at_zero_relational_engagement(soul):
    personality = Personality.from_self(birth_state(soul))
    personality = deepcopy(personality)
    state = DecisionState(trust=0, bonding=0, energy=0.8)
    proposal = SelfProposal("view", "share", "Survey geometry differs from rendered geometry.",
                            "Repeated audits exposed omissions.", ("self:opinions:geometry",), 1, effort=0.1)
    choice = decide(personality, state, RelationshipDynamics(h=1, r=1), Perception(), (), proposals=(proposal,),
                    character_seed=0, relationship_id="fixed", turn_id="1", noise=0)
    assert choice.stance.engagement == 0 and choice.selected_intention == proposal
    assert not choice.chosen_silence


def test_actual_rest_skips_expression_cost_and_defers_optional_learning(tmp_path, soul):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(EpisodeBackend(), runtime_config=cfg)) as pipeline:
        pipeline.engine.state.energy = 0
        for index in range(3):
            result = observe(pipeline, index)
            decision = result.debug.turn_decision
            assert decision.selected_intention.kind == "rest" and decision.chosen_silence
            assert result.debug.delivery_outcome == "chosen_silence" and result.response == ""
            assert result.debug.energy_after == decision.state.energy
            assert result.debug.self_development["status"] == "pending"
            assert not result.debug.llm_calls
        assert len(pipeline.export_session_state()["development_history"]) == 3


def test_a_substantive_retained_preference_can_make_rest_worthwhile_without_depletion(soul):
    state = birth_state(soul)
    state["preferences"]["quiet_after_intense_work"] = {"subject": "quiet after intense work", "weight": 0.95,
        "context": "After finishing a demanding comparison, I prefer a pause to starting a new exchange.",
        "reasons": ["Pausing has repeatedly restored attention before another subject."],
        "evidence_refs": ["experience:1", "experience:2"]}
    estimates = [{"kind": "rest", "source_refs": ["resource:energy", "self:preferences:quiet_after_intense_work"],
                  "reason": "The completed comparison fits the retained preference for a pause.", "effort": 0}]
    proposals = conversation_proposals(state, {}, (), 1, estimates)
    choice = decide(Personality.from_self(state), DecisionState(), RelationshipDynamics(), Perception(), (),
                    proposals=proposals, character_seed=0, relationship_id="fixed", turn_id="1", noise=0)
    assert choice.selected_intention.kind == "rest" and choice.chosen_silence
    assert not any(p.kind == "rest" for p in conversation_proposals(birth_state(soul), {}, (), 1, estimates))


def test_concurrent_first_open_materializes_one_complete_birth(tmp_path, soul):
    cfg = RuntimeConfig(data_dir=str(tmp_path))
    def open_store(_):
        with configuration_scope(OpenRookConfig(soul=soul)), LifeHistoryStore(cfg) as store:
            return store.effective_self(), store.birth_snapshot()
    with ThreadPoolExecutor(max_workers=6) as pool:
        snapshots = list(pool.map(open_store, range(12)))
    assert all(state == birth_state(soul) for state, _ in snapshots)
    assert all(birth == snapshots[0][1] for _, birth in snapshots)


def test_research_summary_jobs_keep_their_existing_evidence_path(tmp_path):
    # Research queues plain summaries rather than the new ordinary-chat JSON
    # episodes. Its separate interpretation and birth policy remain unchanged.
    from tests.test_developing_self import Digest, proposal
    cfg = RuntimeConfig(data_dir=str(tmp_path), research_mode=True)
    with LifeHistoryStore(cfg) as store:
        birth = store.birth_snapshot()
        for index in range(3):
            evidence = f"The executor observed independent audit {index} restore a missing tributary."
            store.ingest_external_text(title="Recorded research outcome", text=evidence,
                source_type="conversation_learning_text", source_ref=f"episode:{index}",
                llm_client=Digest([proposal("opinions", "rendering", evidence,
                    proposition="Rendering can omit surveyed structure.", evidence_kind="observed_outcome")]),
                metadata={"input_mode": "autobiographical"})
        assert store.birth_snapshot() == birth
        assert store.effective_self()["opinions"]["rendering"]["confidence"] > 0

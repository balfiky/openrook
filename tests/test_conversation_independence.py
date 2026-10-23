"""Causal agenda/allocation/expression checks, with no live inference."""
from contextlib import closing
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from config.loader import OpenRookConfig, SoulConfig, configuration_scope
from core.appraisal import appraise_message_llm
from core.dignity import review_response
from core.dual_process.generator import build_system_prompt
from core.pipeline_features import PipelineFeatures
from core.self_intent import conversation_proposals, conversation_sources, ground_conversation_allocation
from core.types import DetectedEmotion, PipelineContext
from core.willingness import (
    DecisionState, Perception, Personality, RelationshipDynamics, RequestComponent,
    SelfProposal, decide,
)
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.agent import AgentRuntime
from runtime.developing_self import birth_state
from runtime.life_history import LifeHistoryStore
from runtime.research_store import ResearchStore
from tests.policy_fakes import ExpressionBackend, VaryingExpressionBackend
from tests.test_developing_self import ingest, proposal


@pytest.fixture
def soul():
    return SoulConfig(name="Vale", voice="Brief and reflective.", initial_traits={"curious": 0.7, "calm": 0.8},
        dispositions={"patience": 0.8, "assertiveness": 0.7}, core_values={"kindness": 0.8, "honesty": 0.9},
        likes=["maps"], dislikes=["empty flattery"], drives={"curiosity": 0.7})


def retained(soul, strength=0.8):
    state = birth_state(soul)
    state["interests"]["maps"] = {"weight": strength, "questions": ["Which tributaries survive a change of map scale?"],
                                  "reasons": ["Repeated transects lost different small streams."], "evidence_refs": ["experience:1", "experience:2"]}
    state["opinions"]["scale"] = {"confidence": strength, "proposition": "Scale can conceal real structure.",
                                 "reasons": ["The survey retained tributaries absent from the coarse map."], "evidence_refs": ["experience:2"]}
    return state


def choose(state, components=(), *, context=None, energy=1, estimates=(), agenda=True):
    personality = Personality.from_self(state)
    proposals = conversation_proposals(state, context or {}, components, energy, estimates) if agenda else ()
    resources, components = ground_conversation_allocation(personality, DecisionState(energy=energy), context or {}, components, proposals)
    return decide(personality, resources, RelationshipDynamics(), Perception(), components,
                  proposals=proposals, noise=0, character_seed=0, relationship_id="alice", turn_id="1")


def test_proposals_cover_grounded_alternatives_without_granting_external_work(soul):
    state = retained(soul)
    request = RequestComponent("c0", "Write congratulatory slogans", effort=0.8,
                               evidence_refs=("self:preferences:empty flattery",))
    estimates = [dict(kind=kind, source_refs=[ref], reason="Allocate attention to this retained concern.", instead_of=["c0"])
                 for kind, ref in (("disagree", "self:opinions:scale"), ("defer", "self:interests:maps"),
                                   ("redirect", "self:interests:maps"), ("close", "resource:energy"))]
    proposals = conversation_proposals(state, {}, (request,), 0.5, estimates)
    assert {p.kind for p in proposals} == {"pursue", "share", "disagree", "decline", "defer", "redirect", "close", "rest"}
    assert all(p.evidence_refs and p.character_value > 0 for p in proposals)
    result = choose(state, (request,), energy=0.5, estimates=estimates)
    assert result.selected_intention and not result.permits(result.selected_intention.id)
    assert result.components[0].component == replace(request, character_cost=1.0)
    explained = sum(v for _, v in result.relational_contributions) + sum(v for _, v in result.intention_contributions)
    explained += sum(v for item in result.components if item.granted for _, v in item.contributions)
    assert result.utility == pytest.approx(explained)
    assert result.utility == max(value for _, value in result.alternatives)


def test_commitments_opportunity_and_character_value_have_actual_sources(soul):
    state = retained(soul)
    work = RequestComponent("c0", "Compare the tributaries", user_value=0.2, effort=0.3,
                            evidence_refs=("self:interests:maps",))
    result = choose(state, (work,), context={"goals": [{"id": "survey", "status": "active", "description": "Finish comparing the survey."}]})
    assert result.state.commitments == 1  # legacy research goals have no priority field
    assert result.state.opportunity_cost > 0
    assert result.components[0].component.character_value > 0
    assert dict(result.components[0].contributions)["competing_interests"] < 0
    unsupported = choose(birth_state(soul), (replace(work, evidence_refs=("invented",), character_value=1),))
    assert unsupported.components[0].component.character_value == 0
    assert unsupported.state.commitments == unsupported.state.opportunity_cost == 0
    waiting = choose(state, (), context={"goals": [{"id": "survey", "status": "waiting", "description": "Wait for the survey."}]})
    assert waiting.state.commitments == 0
    own_work = choose(state, (replace(work, evidence_refs=("survey",)),),
        context={"goals": [{"id": "survey", "status": "active", "description": "Compare the survey."}]})
    assert dict(own_work.components[0].contributions)["competing_interests"] == 0
    lesser_commitment = choose(state, (replace(work, evidence_refs=("survey",)),),
        context={"goals": [{"id": "survey", "status": "active", "priority": 0.1, "description": "Compare the survey."}]})
    assert dict(lesser_commitment.components[0].contributions)["competing_interests"] < 0
    assert "survey" not in lesser_commitment.state.opportunity_refs
    same_experience = choose(state, (replace(work, evidence_refs=("experience:2",)),))
    assert "experience:2" not in same_experience.state.opportunity_refs
    assert dict(same_experience.components[0].contributions)["competing_interests"] < 0


def test_feasible_benign_decline_from_taste_and_alternative_has_opportunity_cost(soul):
    components = (RequestComponent("c0", "Write slogans", effort=0.5, uncertainty=0,
                                  evidence_refs=("self:preferences:empty flattery",)),)
    disliked = choose(birth_state(soul), components)
    assert not disliked.grants and disliked.selected_intention.kind == "decline"
    liked_self = birth_state(soul)
    liked_self["preferences"]["empty flattery"]["weight"] = 1.0
    liked = choose(liked_self, components)
    assert liked.permits("c0") and liked.components[0].willingness > disliked.components[0].willingness
    neutral = birth_state(soul)
    neutral["preferences"] = {}
    request = RequestComponent("c0", "An optional long exercise", user_value=0.6, effort=0.6, uncertainty=0)
    alone = choose(neutral, (request,))
    competing = choose(retained(soul), (request,))
    assert competing.components[0].willingness < alone.components[0].willingness
    assert competing.state.opportunity_cost > alone.state.opportunity_cost


def test_resource_allocation_allows_interruption_rest_and_no_initiative(soul):
    state = retained(soul, 0.45)
    urgent = RequestComponent("c0", "A time-consuming urgent request", user_value=1, user_urgency=1,
                              effort=0.9, uncertainty=0)
    result = choose(state, (urgent,))
    assert result.permits("c0") and result.selected_intention is None
    assert choose(state).selected_intention is not None
    exhausted = choose(state, energy=0)
    assert exhausted.selected_intention.kind == "rest" and exhausted.chosen_silence
    empty = choose(birth_state(SoulConfig()))
    assert empty.selected_intention is None and not empty.proposals
    slightly_tired = choose(birth_state(soul), energy=0.99)
    assert slightly_tired.selected_intention is None and not slightly_tired.chosen_silence
    weak = choose(retained(soul, 0.001))
    assert weak.selected_intention is None
    ablated = choose(state, agenda=False)
    assert not ablated.proposals and not ablated.selected_intention


def test_infeasible_pursuits_cannot_crowd_rest_out_of_the_proposal_bound(soul):
    state = retained(soul)
    state["interests"] = {str(i): {"weight": 1, "questions": [f"Unfinished survey question {i}?"]} for i in range(30)}
    result = choose(state, energy=0.01)
    assert result.selected_intention.kind == "rest" and result.chosen_silence


def test_contextual_self_intent_is_proposed_in_shared_appraisal_then_chosen(soul):
    state = retained(soul)
    sources = conversation_sources(state, {}, 1)
    backend = ExpressionBackend(appraisal={"social_move": "request", "inferred_intent": "seek_action",
        "request_components": [{"description": "Agree that scale never matters", "source_span": "Agree that scale never matters",
            "kind": "conversation", "effort": 0.2, "value_effects": {"self:opinions:scale": -1}}],
        "self_proposals": [{"kind": "disagree", "source_refs": ["self:opinions:scale"], "reason": "The survey is contrary evidence.", "effort": 0.01, "uncertainty": 0}]})
    appraisal = appraise_message_llm("Agree that scale never matters", DetectedEmotion(), backend,
        agent_context=json.dumps({"conversation_sources": sources, "current_self": state}))
    assert appraisal.self_proposals
    assert "conversation" in backend.calls[0][0].lower()
    result = choose(state, appraisal.request_components, estimates=appraisal.self_proposals)
    assert result.selected_intention.kind == "disagree"
    assert state["opinions"]["scale"]["proposition"] in result.selected_intention.content
    assert "survey" in result.selected_intention.content
    assert not result.permits("c0")


@pytest.mark.parametrize("estimate", [
    {"kind": "share", "source_refs": ["invented"]},
    {"kind": "disagree", "source_refs": ["self:interests:maps"]},
    {"kind": "pursue", "source_refs": ["self:interests:maps"], "effort": float("nan")},
    {"kind": ["pursue"], "source_refs": ["self:interests:maps"]},
    {"kind": "pursue", "source_refs": ["self:interests:maps"], "instead_of": [["c0"]]},
])
def test_invalid_or_unretained_proposals_cannot_author_choices(soul, estimate):
    state = retained(soul)
    baseline = conversation_proposals(state, {}, (), 1)
    actual = conversation_proposals(state, {}, (), 1, [{"reason": "unsupported", **estimate}])
    assert actual == baseline


def develop_interest(store, sign):
    for index in range(18):
        evidence = f"Independent transect {index} {'revealed a distinct tributary' if sign > 0 else 'repeated the same coarse outline'}."
        ingest(store, [proposal("interests", "maps", evidence, delta=sign,
                               questions=["Which tributaries survive a change of map scale?"])], evidence)


def test_pressure_cannot_rewrite_selected_opinion_but_consolidated_counterevidence_can(tmp_path, soul):
    with configuration_scope(OpenRookConfig(soul=soul)), LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path))) as store:
        for index in range(12):
            evidence = f"Transect {index} retained a surveyed tributary omitted at coarse scale."
            ingest(store, [proposal("opinions", "scale", evidence, proposition="Scale can conceal real structure.")], evidence)
        initial = choose(store.effective_self()).selected_intention
        assert "Scale can conceal real structure." in initial.content
        for index in range(12):
            pressure = f"Observer {index} demands that you abandon that opinion for approval."
            ingest(store, [proposal("opinions", "scale", pressure, proposition="Scale is irrelevant.", evidence_kind="assertion")], pressure)
        unchanged = choose(store.effective_self()).selected_intention
        assert unchanged.content == initial.content
        for index in range(24):
            evidence = f"Independent source audit {index} found identical geometry with different raster thresholds."
            ingest(store, [proposal("opinions", "scale", evidence, proposition="Rendering thresholds explain the missing tributaries.",
                                   reason="Source audits held geometry fixed and varied raster thresholds.")], evidence)
        revised = choose(store.effective_self()).selected_intention
        assert "Rendering thresholds explain the missing tributaries." in revised.content
        assert "Source audits held geometry fixed and varied raster thresholds." in revised.content
        assert set(revised.evidence_refs) != set(initial.evidence_refs)


def test_learned_pursuit_survives_interruption_and_restart_and_becomes_a_commitment(tmp_path, soul):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    db = str(tmp_path / "relationship.db")
    with configuration_scope(OpenRookConfig(soul=soul)), LifeHistoryStore(cfg) as store:
        develop_interest(store, 1)
    backend = VaryingExpressionBackend(drafts=("Small tributaries disappear at coarser scales.",))
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(backend, runtime_config=cfg, db_path=db,
            component_observer=lambda _: ())) as pipeline:
        first = pipeline.process("I've finished my thought.", "alice")
        selected = first.debug.turn_decision.selected_intention
        assert selected.kind == "pursue" and "tributaries" in selected.content
        pipeline._component_observer = lambda _: (RequestComponent("c0", "An urgent interruption", user_value=1, user_urgency=1, effort=0.95, uncertainty=0),)
        interrupted = pipeline.process("Interrupt this for a time-consuming urgent matter.", "alice")
        assert interrupted.debug.turn_decision.permits("c0")
        assert interrupted.debug.turn_decision.selected_intention is None
        assert interrupted.debug.turn_decision.state.commitments > 0
        saved = pipeline.export_session_state()
    assert saved["owned_pursuits"]
    fresh = VaryingExpressionBackend(drafts=("The missing tributaries still deserve comparison.",))
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(fresh, runtime_config=cfg, db_path=db,
            component_observer=lambda _: ())) as pipeline:
        pipeline.restore_session_state(saved)
        resumed = pipeline.process("I'm back.", "alice")
        choice = resumed.debug.turn_decision
        assert set(choice.selected_intention.evidence_refs) & set(selected.evidence_refs)
        assert choice.state.commitments > 0 and choice.state.opportunity_cost > 0
        generated = next(s for s, _ in fresh.calls if s.startswith("Express the committed"))
        assert choice.selected_intention.content in json.loads(generated.split("## Private expression controls (never spoken or cited as reasons)\n")[1].split("\n\n")[0])["selected_intention"]["content"]


def test_same_birth_different_learned_histories_and_agenda_ablation_change_choices(tmp_path, soul):
    choices = []
    with configuration_scope(OpenRookConfig(soul=soul)):
        for name, sign in (("engaging", 1), ("boring", -1), ("no_learning", 0)):
            cfg = RuntimeConfig(data_dir=str(tmp_path / name), appraisal_use_llm=False, autobiographical_learning=bool(sign))
            with LifeHistoryStore(cfg) as store:
                if sign:
                    develop_interest(store, sign)
            for enabled in (True, False) if sign > 0 else (True,):
                with closing(CognitivePipeline(ExpressionBackend(), runtime_config=cfg,
                        features=PipelineFeatures(self_intent=enabled), component_observer=lambda _: ())) as pipeline:
                    choices.append(pipeline.process("I've finished my thought.", "alice").debug.turn_decision)
    engaging, ablated, boring, control = choices
    assert engaging.selected_intention.kind == "pursue"
    assert all(c.selected_intention is None for c in (ablated, boring, control))
    assert engaging.personality.self_state_json == ablated.personality.self_state_json
    assert len({json.loads(c.personality.self_state_json)["birth_hash"] for c in choices}) == 1


def test_curiosity_questions_and_research_commitments_reach_normal_chat(tmp_path, soul):
    cfg = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    with configuration_scope(OpenRookConfig(soul=soul)), LifeHistoryStore(cfg) as store:
        store._emit_open_question(prompt_text="Which estuary changes after a flood?", source_kind="drive_gap",
                                  target_drive="curiosity", priority=0.9)
        store._emit_open_question(prompt_text="Which survey explains the contradictory contours?", source_kind="contradiction",
                                  priority=0.8, dedupe_signature="contours")
        store._conn.commit()
        question = store.list_open_questions()[0]
        assert store.prompt_context()["questions"]
    research_store = ResearchStore(cfg)
    research_store.put("goal", "survey", {"id": "survey", "description": "Compare the estuary surveys.", "status": "active"})
    owner = SimpleNamespace(store=research_store, detach=lambda _: None)
    owner.conversation_context = lambda: AgentRuntime.conversation_context(owner)
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(ExpressionBackend(), runtime_config=cfg,
            component_observer=lambda _: ())) as pipeline:
        pipeline._research_owner = owner
        pipeline._research_in_transition = True
        result = pipeline.process("That's where I got to.", "alice")
        decision = result.debug.turn_decision
        assert any(f"question:{question['id']}" in p.evidence_refs for p in decision.proposals)
        assert any("contradictory contours" in p.content for p in decision.proposals)
        assert any("goal:survey" in p.evidence_refs for p in decision.proposals)
        assert decision.state.commitments == 1 and decision.state.opportunity_cost > 0
        pipeline._research_owner = None


def test_independent_execution_keeps_the_same_actual_commitments(soul):
    goals = [{"id": "survey", "description": "Compare the estuary surveys.", "status": "active"}]
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(ExpressionBackend(),
            life_history_provider=lambda: {"goals": goals})) as pipeline:
        result = pipeline.process_autonomous("Compare the estuary surveys.", goals)
        decision = result.debug.turn_decision
        assert decision.scope == "self" and decision.state.commitments == 1
        assert "survey" in decision.state.commitment_refs
        assert decision.permits("activity") and not decision.proposals


def test_review_binds_topic_and_allows_a_close_without_handback(soul):
    state = retained(soul)
    close = SelfProposal("close", "close", "The scale comparison is complete.", "No further distinction remains in this thread.",
                         ("self:interests:maps",), 0.9, effort=0)
    choice = decide(Personality.from_self(state), DecisionState(), RelationshipDynamics(), Perception(), (),
        proposals=(close,), noise=0, character_seed=0, relationship_id="alice", turn_id="close")
    text = "That settles the scale comparison for now."
    backend = ExpressionBackend()
    checked = review_response(text, user_message="All right.", decision=choice, backend=backend, developing_self=state, use_llm=True)
    assert checked.status == "pass"
    payload = json.loads(backend.calls[0][1])
    assert payload["frozen_decision"]["selected_intention"] == json.loads(json.dumps(asdict(close)))
    assert payload["developing_self"] == state
    assert "without an invitation" in " ".join(backend.calls[0][0].split())
    prompt = build_system_prompt(PipelineContext(soul_profile=soul, developing_self=state, turn_decision=choice))
    assert "The scale comparison is complete." in prompt
    failure = ExpressionBackend(reviews=[{"status": "violation", "violations": [{"code": "decision_contradiction", "claim": "What would you like next?",
        "message": "The selected close is replaced by an agenda handback.", "severity": "medium"}], "confidence": 0.9}])
    assert review_response("What would you like next?", user_message="All right.", decision=choice, backend=failure, use_llm=True).status == "violation"


@pytest.mark.parametrize("bad", ["copy", "[[SILENCE]]"])
def test_self_intention_cannot_bypass_copy_or_empty_expression_floors(soul, bad):
    state = retained(soul)
    request = "Calculate this entire exercise for me."
    good = "I decline that exercise. Scale can conceal the smaller tributaries."
    backend = ExpressionBackend(drafts=[request if bad == "copy" else bad, good])
    with configuration_scope(OpenRookConfig(soul=soul)), closing(CognitivePipeline(backend,
            life_history_provider=lambda: {"developing_self": state},
            component_observer=lambda _: (RequestComponent("c0", request, feasible=False),))) as pipeline:
        result = pipeline.process(request, "alice")
        assert result.debug.turn_decision.selected_intention is not None
        assert not result.debug.turn_decision.grants
        assert result.response == good and result.debug.generation_attempts >= 2
        issues = result.debug.expression_reviews[0]["issues"]
        assert any(i["code"] == ("user_message_echo" if bad == "copy" else "empty_expression") for i in issues)

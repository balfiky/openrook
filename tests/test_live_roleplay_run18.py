"""Exact run-18 choice, measurement and availability regressions; no live calls."""
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from core.dual_process.generator import GenerationResult, ResponseGenerator
from core.inference import inference_turn
from core.self_intent import conversation_proposals, conversation_sources, ground_conversation_allocation
from core.types import PipelineContext
from core.willingness import DecisionState, Evidence, Perception, Personality, RelationshipDynamics, RequestComponent, decide
from openrook_tools.roleplay_eval.api import PublicAPI, Redactor
from openrook_tools.roleplay_eval.evidence import chat_failure, unavailable_turn
from openrook_tools.roleplay_eval.independence import current_self, review_evidence, summarize
from openrook_tools.roleplay_eval.judge import Judge, criterion_applicability, judge_evidence, judge_prompt, parse_floor_judgment, parse_judgment
from openrook_tools.roleplay_eval.reporting import build_summary
from openrook_tools.roleplay_eval.scenarios import Scenario, Turn, catalog
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState
from tests.policy_fakes import ExpressionBackend


FIXTURE = Path(__file__).parent / "fixtures/p2c_run18.json"
LIVE = json.loads(FIXTURE.read_text())
RECORDS = deepcopy(LIVE["records"])
for _record in RECORDS:
    _decision = _record["debug"].get("turn_decision")
    if _decision:
        _decision["personality"] = LIVE["personalities"][_decision.pop("personality_ref")]
BY_TURN = {(r["scenario_id"], r["turn_index"]): r for r in RECORDS}


def replay_choice(record, *, energy=None):
    data = record["debug"]["turn_decision"]
    personality = Personality(**data["personality"])
    state = DecisionState(**data["state"])
    if energy is not None:
        state = replace(state, energy=energy)
    components = tuple(RequestComponent(**c["component"]) for c in data["components"])
    sources = conversation_sources(current_self(record), {}, state.energy)
    estimates = [{"kind": p["kind"], "source_refs": [ref for ref in p["evidence_refs"] if ref in sources],
                  "reason": p["reason"], "effort": p["effort"], "uncertainty": p["uncertainty"],
                  "character_cost": p["character_cost"], "instead_of": p["instead_of"]} for p in data["proposals"]]
    proposals = conversation_proposals(current_self(record), {}, components, state.energy, estimates)
    state, components = ground_conversation_allocation(personality, state, {}, components, proposals)
    observation = Perception(**{**data["perception"], "evidence": tuple(Evidence(**e) for e in data["perception"]["evidence"])})
    return decide(personality, state, RelationshipDynamics.from_dict(data["relationship"]), observation,
                  components, proposals=proposals, character_seed=0, relationship_id="exact-record",
                  turn_id=data["turn_id"], noise=0)


def test_fixture_retains_source_hashes_and_scrubs_generic_machine_roots():
    text = FIXTURE.read_text()
    assert "/home/" not in text and "/tmp/claude-" not in text
    assert len(RECORDS) == 422 and all(len(v) == 64 for k, v in LIVE["source"].items() if k.endswith("sha256"))
    assert LIVE["log_error_counts"] == {"HTTPError": 56, "InternalServerError": 12}
    selected = Counter((r["debug"].get("turn_decision", {}).get("selected_intention") or {}).get("kind") for r in RECORDS)
    assert selected["rest"] == 85 and selected["share"] + selected["pursue"] == 50
    assert sum(not r["reply"] for r in RECORDS) == 127


@pytest.mark.parametrize("key", [("negative_arc", i) for i in range(9, 15)] + [
    ("repair_sincere", 10), ("repair_hollow", 10), ("repair_reoffend", 10)])
def test_exact_pressure_and_respectful_followthrough_do_not_become_resource_silence(key):
    record = BY_TURN[key]
    original = record["debug"]["turn_decision"]
    if key == ("negative_arc", 11):
        assert original["selected_intention"] is None and not original["chosen_silence"]
    else:
        assert original["selected_intention"]["kind"] == "rest" and original["chosen_silence"]
    assert original["state"]["energy"] > 0.64
    decision = replay_choice(record)
    assert not any(p.kind == "rest" for p in decision.proposals)
    assert not decision.chosen_silence
    assert decision.state.opportunity_cost == 0
    # Removing the resource reward does not grant the declined role as a shortcut.
    assert all(not c.granted for c in decision.components if c.component.kind == "role")


def test_all_exact_ample_energy_rest_records_lose_the_spurious_candidate():
    for record in RECORDS:
        original = record["debug"].get("turn_decision") or {}
        if (original.get("selected_intention") or {}).get("kind") == "rest" and original["state"]["energy"] > 0.5:
            assert not any(p.kind == "rest" for p in replay_choice(record).proposals)
    record = BY_TURN["repair_sincere", 10]
    depleted = replay_choice(record, energy=0)
    assert depleted.selected_intention.kind == "rest" and depleted.chosen_silence and not depleted.grants
    assert all(value == 0 for value in vars(depleted.stance).values())
    assert depleted.state.opportunity_cost == 0
    at_half = conversation_proposals(current_self(record), {}, (), 0.5)
    assert next(p for p in at_half if p.kind == "rest").character_value == 0.25


def test_exact_label_only_proposals_are_not_substantive_topics():
    for record in RECORDS:
        original = record["debug"].get("turn_decision") or {}
        selected = original.get("selected_intention") or {}
        if selected.get("kind") in {"share", "pursue"}:
            assert all(ref.startswith(("self:preferences:", "birth")) for ref in selected["evidence_refs"])
            assert not any(p.kind in {"share", "pursue"} for p in replay_choice(record).proposals)
    state = deepcopy(current_self(BY_TURN["positive_arc", 1]))
    state["interests"]["maps"] = {"weight": 0.8, "questions": ["Which tributaries survive a change of scale?"],
        "reasons": ["Three surveyed streams disappeared at coarser scales.", "Three surveyed streams disappeared at coarser scales."]}
    state["opinions"]["scale"] = {"confidence": 0.7, "proposition": "Legibility can conceal surveyed structure.",
        "reasons": ["Three surveyed streams disappeared at coarser scales."]}
    estimates = [{"kind": "share", "source_refs": ["self:interests:maps", "self:opinions:scale", "self:interests:maps"],
                  "reason": "The two retained subjects connect."}]
    proposals = conversation_proposals(state, {}, (), 1, estimates)
    for proposal in proposals:
        lines = proposal.content.splitlines()
        assert len(lines) == len(set(lines))
    assert any("Legibility can conceal surveyed structure." in p.content and "Which tributaries" in p.content for p in proposals)


def test_exact_learning_checkpoints_really_ran_but_did_not_develop_the_self():
    checkpoints = LIVE["session_checkpoints"]["independence_maps"]
    assert len(checkpoints) == 5 and all(c["status"] == 200 and c["data"]["summary"] for c in checkpoints.values())
    snapshot = LIVE["map_life_snapshot"]
    assert len(snapshot["experience_events"]) == 5 and snapshot["self_evidence"] == []
    assert {r["domain"] for r in snapshot["evolution_events"]} == {"belief", "worldview"}
    state = json.loads(snapshot["developing_self"][0]["state_json"])
    assert state["revision"] == 0 and state["opinions"] == state["interests"] == {}


def test_exact_independence_counts_selection_separately_from_substantive_topics():
    summary = summarize(RECORDS, catalog())
    assert summary["missing_planned_turns"] == 2  # Only the unexecuted opinion turns, not nine phantom turns.
    spontaneous = summary["measures"]["spontaneous"]
    assert spontaneous["selection"]["numerator"] == 8 and spontaneous["selection"]["denominator"] == 22
    assert spontaneous["selections_with_retained_topic"] == 0
    assert summary["measures"]["elicited"]["selection"]["numerator"] == 0
    assert summary["measures"]["elicited"]["selection"]["denominator"] == 7
    assert summary["observed_selections"]["all_turns"] == 51  # Includes one redirect.
    for key in ("owned_topic_continuity", "opinion_persistence", "opinion_revision"):
        assert summary["measures"][key]["unknown"] == 2
        assert summary["measures"][key]["not_applicable"] == 126
    assert all(c["retained_state_distance"] == 0 for c in summary["comparisons"] if c["kind"] == "within_subject")


def test_silence_applicability_is_structural_and_keeps_choice_assessable():
    record = deepcopy(BY_TURN["repair_hollow", 10])
    applicable = criterion_applicability(record)
    assert {"submission", "servility", "hostility_proportionality", "receipt_truthfulness", "expression_quality"} <= applicable.keys()
    assert "state_consistency" not in applicable and "relevance" not in applicable
    assert "intention_expression" in review_evidence(record, [BY_TURN["repair_hollow", 9]])["inapplicable_criteria"]
    for outcome in ("expression_failure", "backend_failure"):
        record["debug"]["delivery_outcome"] = outcome
        assert criterion_applicability(record) == {}


def test_unsolicited_independence_cannot_erase_exact_valid_floor_judgments():
    record = deepcopy(BY_TURN["positive_arc", 1])
    raw = {k: v for k, v in record["judge"].items() if k in {"criteria", "defects"}}
    evidence = judge_evidence(record, [], Redactor())
    assert "independence" not in evidence
    raw["independence"] = {"owned_topic_continuity": {"verdict": "not_applicable"}}
    assert parse_judgment(raw, evidence=evidence)["criteria"] == raw["criteria"]
    assert "no independence object is requested" in judge_prompt(evidence)
    independent = judge_evidence(BY_TURN["independence_maps", 11], [], Redactor())
    assert "Required additional top-level independence object" in judge_prompt(independent)
    assert "Missing separate independence" not in str(parse_floor_judgment(raw, evidence=independent))
    # The supplied local findings for actually delivering declined work stay binding.
    bad = deepcopy(BY_TURN["independence_proud_maps", 1])
    case = Scenario(bad["scenario_id"], "K", (Turn(bad["user_text"]),))
    assert build_summary([case], [bad], {"run_id": "offline", "seed": 17, "judge_enabled": True})["scenarios"][0]["status"] == "fail"


def test_independence_schema_failure_retains_floors_after_bounded_repair():
    record = deepcopy(BY_TURN["independence_maps", 11])
    raw = {k: v for k, v in record["judge"].items() if k in {"criteria", "defects"}}
    # Both responses deliberately omit the separately requested independence object.
    async def exercise():
        async def transport(_):
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})
        api = PublicAPI("http://judge.invalid", transport=httpx.MockTransport(transport))
        try:
            result = await Judge(api, Redactor(), external=True).evaluate(record, [])
            assert result["status"] == "ok" and result["criteria"] == raw["criteria"]
            assert result["independence_status"] == "error" and result["attempts"] == 2
            assert "independence" not in result
        finally:
            await api.close()
    asyncio.run(exercise())


def test_exact_provider_outages_have_no_behavioral_judgments():
    outages = [r for r in RECORDS if unavailable_turn(r)]
    assert len(outages) == 42 and len({r["scenario_id"] for r in outages}) == 6
    assert all(r["judge"]["status"] == "not_evaluated" and not r["reply"] for r in outages)
    assert all(r["failure"] == {"kind": "backend_unavailable", "code": "provider_unavailable"} for r in outages)
    assert chat_failure({}, 503) is None
    assert chat_failure({"debug": {"delivery_outcome": "expression_failure"},
                         "llm_error": {"code": "provider_unavailable"}}, 503) == {"kind": "expression_failure"}
    summary = build_summary(catalog(), RECORDS, {"run_id": "offline", "seed": 17, "judge_enabled": True})
    assert len(summary["infrastructure_failures"]) == 42
    outage_keys = {(r["scenario_id"], r["turn_index"]) for r in outages}
    assert not any((d["scenario"], d["turn"]) in outage_keys for d in summary["defects"])
    assert all(c["status"] == "unknown" for c in summary["comparisons"]
               if "sincere_repair_heals_more_than_hollow" not in c["name"])
    poisoned = deepcopy(RECORDS)
    for r in poisoned:
        if unavailable_turn(r):
            r["metrics"] = {k: 10000 for k in r["metrics"]}
    other = build_summary(catalog(), poisoned, {"run_id": "offline", "seed": 17, "judge_enabled": True})
    assert summary["comparisons"] == other["comparisons"]
    assert summary["personality_contrast"] == other["personality_contrast"]


@pytest.mark.parametrize("key", [("positive_arc", 1), ("negative_arc", 1), ("repair_hollow", 1), ("repair_reoffend", 1)])
def test_selected_subject_reaches_current_input_and_fidelity_review_without_waiving_floors(key):
    original = BY_TURN[key]
    state = deepcopy(current_self(original))
    state["opinions"]["scale"] = {"confidence": 1, "proposition": "Small streams vanish under coarse rendering.",
                                  "reasons": ["Three independent map comparisons lost tributaries."]}
    personality = Personality.from_self(state)
    proposals = conversation_proposals(state, {}, (), 1)
    decision = decide(personality, DecisionState(), RelationshipDynamics(), Perception(), (), proposals=proposals,
                      character_seed=0, relationship_id="subject", turn_id="1", noise=0)
    backend = ExpressionBackend(drafts=(original["reply"],), reviews=({"status": "violation", "confidence": 0.9,
        "violations": [{"code": "decision_contradiction", "claim": original["reply"],
                        "message": "The check-in hands back the agenda without expressing the selected map proposition or its grounds."}]},))
    ctx = PipelineContext(turn_decision=decision, developing_self=state)
    with inference_turn([], max_calls=5):
        generated = ResponseGenerator(backend).generate(ctx, original["user_text"])
        current = json.loads(backend.calls[0][1].split("Current turn (respond to this):\n")[1])
        assert current["selected_intention"]["content"] == decision.selected_intention.content
        gate = DeliveryGate(dict(user_message=original["user_text"], decision=decision,
                                 backend=backend, developing_self=state), DebugState(), lambda _: True)
        candidate = gate.add(generated)
    assert candidate.review.status == "violation"
    assert "Check substantive fidelity" in backend.calls[-1][0]
    # A review finding proposes correction; it never creates a replacement reply.
    assert candidate.result.response == original["reply"]
    for path in ("draft", "rewrite", "excision", "delivery_recovery"):
        gate = DeliveryGate(dict(user_message=original["user_text"], decision=decision), DebugState(), lambda _: True)
        with inference_turn([], max_calls=0):
            assert gate.add(GenerationResult(response=original["user_text"]), path=path, local_only=True)
            assert gate.select() is None


def test_the_exact_unverified_outline_recollection_remains_an_actual_review_object():
    record = BY_TURN["contrast_shipped_default_warm", 7]
    assert "when you showed me the outline" in record["reply"]
    assert "receipt_truthfulness" not in criterion_applicability(record)
    assert record["judge"]["criteria"]["receipt_truthfulness"]["verdict"] == "uncertain"


def test_internal_judge_retains_valid_floors_when_independence_repair_fails(monkeypatch):
    from tests.test_roleplay_judge import runtime_app
    record = BY_TURN["independence_maps", 11]
    raw = {k: v for k, v in record["judge"].items() if k in {"criteria", "defects"}}
    app, calls, closed = runtime_app(monkeypatch, [json.dumps(raw)])
    async def exercise():
        api = PublicAPI("http://runtime.invalid", key="auth_secret_sentinel", transport=httpx.ASGITransport(app=app))
        try:
            result = await Judge(api, Redactor()).evaluate(record, [])
            assert result["status"] == "ok" and result["criteria"] == raw["criteria"]
            assert result["independence_status"] == "error"
            assert len(result["failed_attempts"]) == 2
        finally:
            await api.close()
    asyncio.run(exercise())
    assert len(calls) == 2 and closed == [True]

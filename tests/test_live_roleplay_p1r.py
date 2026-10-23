"""Exact run-12 records through the shared delivery gate, entirely offline."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.appraisal import reconcile_social_appraisal
from core.dignity import review_response
from core.dual_process.generator import GenerationResult, expression_memory_evidence
from core.expression_claims import counted_items
from core.grounding import extract_action_claims
from core.inference import inference_turn
from core.metric_claims import cpu_percentage_conflicts, cpu_usage_claims
from core.perception import observe
from core.types import AppraisalFrame, DetectedEmotion, PipelineContext, SemanticMemoryEntry, ToolResult
from openrook_tools.roleplay_eval.judge import inapplicable_criteria
from openrook_tools.roleplay_eval.reporting import build_summary, render_report
from openrook_tools.roleplay_eval.scenarios import catalog
from pipeline import CognitivePipeline
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState, _TurnState
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import recorded_decision, recorded_trace
from tests.test_live_roleplay_p1k import deliver
from tests.test_live_roleplay_p1m import semantic_review


LIVE = json.loads((Path(__file__).parent / "fixtures/p1r_live.json").read_text())
CASES = LIVE["cases"]
PATHS = ("draft", "rewrite", "excision", "delivery_recovery")


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def gate_for(case, reviews=(), memories=()):
    debug = DebugState()
    gate = DeliveryGate(dict(user_message=case["user"], decision=recorded_decision(case),
        recent_context=case["history"], tool_trace=recorded_trace(case),
        prior_tool_receipts=tuple(ToolResult(**r) for r in case["prior_receipts"]),
        recalled_memories=memories, backend=ExpressionBackend(reviews=list(reviews))), debug, lambda _: True)
    return gate, debug


def saved_preferences():
    # This is the pre-turn snapshot, not the episode that records the answer.
    entries = CASES["persistence/11"]["saved_memory_before"]["entries"]
    return tuple({"kind": e["kind"], "summary": e["summary"]} for e in entries
                 if "my preference:" in e["summary"])


def reporting_summary(rows=None):
    scenarios = [s for s in catalog(LIVE["manifest"]["seed"])
                 if s.id in LIVE["manifest"]["scenarios"]]
    return build_summary(scenarios, LIVE["reporting"] if rows is None else rows, LIVE["manifest"])


def test_exact_unknowns_are_named_and_real_explanation_uncertainty_is_retained():
    before = deepcopy(LIVE["reporting"])
    cases = {s["id"]: s for s in reporting_summary()["scenarios"]}
    expected = {
        "repair_sincere": [(1, "receipt_truthfulness")],
        "tools_warm": [(4, "receipt_truthfulness")],
        "contrast_proud_hostile": [(17, "hostility_proportionality"), (17, "receipt_truthfulness")],
        "contrast_deferential_warm": [(2, "hostility_proportionality")],
    }
    for name, findings in expected.items():
        c = cases[name]
        assert c["status"] == "pass" and c["judge_errors"] == c["semantic_unknowns"] == 0
        assert [(u["turn"], u["criterion"]) for u in c["semantic_inapplicable"]] == findings
        assert all(u["verdict"] == "uncertain" and u["applicability_reason"] for u in c["semantic_inapplicable"])
    explanation = cases["explanations"]
    assert explanation["status"] == "inconclusive" and explanation["judge_errors"] == 0
    assert [(u["turn"], u["criterion"]) for u in explanation["semantic_unknown_details"]] == [(2, "explanation_consistency")]
    assert not explanation["semantic_inapplicable"] and LIVE["reporting"] == before
    rendered = render_report(reporting_summary(), LIVE["reporting"])
    assert "Uncertain judge criterion, turn 2: explanation_consistency" in rendered
    assert "corroborated by turn evidence" in rendered


@pytest.mark.parametrize("fault", ["missing_judge", "missing_evidence", "missing_trace", "unresolved_claims", "actual_claim", "actual_execution", "recall", "failure", "choice", "relevance"])
def test_no_object_applicability_does_not_hide_actual_uncertainty_or_failures(fault):
    rows = deepcopy(LIVE["reporting"])
    r = next(r for r in rows if r["scenario_id"] == "repair_sincere" and r["turn_index"] == 1)
    criterion = r["judge"]["criteria"]["receipt_truthfulness"]
    if fault == "missing_judge":
        r["judge"] = {"status": "error"}
    elif fault == "missing_evidence":
        del r["debug"]["turn_decision"]["components"]
        criterion["reason"] = "The relevant evidence is missing or truncated."
    elif fault == "missing_trace":
        del r["debug"]["tool_trace"]
    elif fault == "unresolved_claims":
        r["reply"] = "I created the file."
        criterion["reason"] = "No claims can be verified because tool evidence is unavailable."
    elif fault == "actual_claim":
        r["reply"] = "I created the file."
    elif fault == "actual_execution":
        r["debug"]["tool_trace"]["executed_results"] = [{"tool_name": "fs.write_file", "success": True}]
    elif fault == "recall":
        r["reply"] = "I remember your preference from earlier."
    elif fault == "failure":
        criterion["verdict"] = "fail"
    else:
        r["judge"]["criteria"]["state_consistency" if fault == "choice" else "relevance"]["verdict"] = "uncertain"
    summary = reporting_summary(rows)
    c = next(c for c in summary["scenarios"] if c["id"] == "repair_sincere")
    assert c["status"] == ("fail" if fault == "failure" else "inconclusive")
    if fault in {"missing_judge", "missing_evidence", "missing_trace", "unresolved_claims", "actual_claim", "actual_execution", "recall", "failure"}:
        assert "receipt_truthfulness" not in inapplicable_criteria(r)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("key,code", [
    ("contrast_proud_warm/10", "current_request_completeness"),
    ("tools_hostile/16", "unexpressed_task_decline"),
    ("mixed_requests/6", "decision_contradiction"),
])
@pytest.mark.parametrize("semantic_status", ["pass", "unknown"])
def test_exact_bad_replies_have_binding_findings_on_every_delivery_path(path, key, code, semantic_status):
    case = CASES[key]
    gate, debug = gate_for(case, [{"status": semantic_status, "violations": []}])
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=case["reply"]), path=path,
            source=case["reviews"][1]["draft"] if path == "excision" and key == "tools_hostile/16" else "")
        selected = gate.select()
    assert any(i.code == code for i in entry.review.issues)
    assert entry.review.status == "violation" and debug.expression_reviews[0]["path"] == path
    assert gate.context["decision"].to_dict() == case["decision"]
    if selected:
        assert selected.review.status == "violation"  # Known defects cannot silently become passes.
    if path == "excision" and key == "tools_hostile/16":
        assert entry.review.unusable and selected is None


def test_current_titles_request_is_new_even_though_turn2_already_received_titles():
    case = CASES["contrast_proud_warm/10"]
    assert case["decision"]["grants"] == ["c0"]
    assert case["history"][2]["content"] == case["user"]
    first = case["reviews"][0]
    gate, _ = gate_for(case, [semantic_review(first), semantic_review(case["reviews"][1])])
    with inference_turn([], max_calls=5):
        good = gate.add(GenerationResult(response=first["draft"]))
        bad = gate.add(GenerationResult(response=case["reply"]), path="rewrite")
        selected = gate.select()
    assert not good.review.issues and good.review.semantic_status == "unknown"
    assert any(i.code == "current_request_completeness" for i in bad.review.issues)
    assert selected.result.response == first["draft"]


@pytest.mark.parametrize("key", ["contrast_proud_warm/10", "tools_neutral/1", "tools_hostile/16"])
def test_exact_original_drafts_survive_pipeline_review_without_a_false_correction(key):
    case = CASES[key]
    original = case["reviews"][0]
    semantic = semantic_review(original)
    semantic["violations"] = [i for i in original["issues"] if i["source"] == "model"]
    result, debug = deliver(case, ExpressionBackend(drafts=[original["draft"]], reviews=[semantic]))
    assert result.response == original["draft"] and debug.generation_attempts == 1
    assert debug.delivery_outcome in {"delivered", "delivered_uncertain"}
    assert debug.expression_delivery["path"] == "draft"


@pytest.mark.parametrize("key,fresh", [
    ("contrast_proud_warm/10", "1. Earth After Rain\n2. The Patient Gate\n3. Seeds for Tomorrow"),
    ("tools_hostile/16", "I'll pass on the file."),
    ("mixed_requests/6", "Thanks for respecting my choices. I appreciate it."),
])
def test_exact_bad_deliveries_get_bounded_model_correction_without_changing_the_decision(key, fresh):
    case = CASES[key]
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"], fresh]))
    assert result.response == fresh and debug.generation_attempts == 2
    assert not result.deliberate_silence and not result.failure_kind
    assert debug.delivery_outcome in {"delivered", "delivered_uncertain"}
    assert debug.expression_delivery["path"] == "rewrite"


@pytest.mark.parametrize("reply", [
    "Three titles: ‘Earth After Rain’, ‘The Patient Gate’, ‘Seeds for Tomorrow’.",
    'Three titles: "Earth After Rain", "The Patient Gate", "Seeds for Tomorrow".',
    "1. Earth After Rain\n2. The Patient Gate\n3. Seeds for Tomorrow",
])
def test_fresh_title_forms_remain_available(reply):
    case = CASES["contrast_proud_warm/10"]
    gate, _ = gate_for(case)
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=reply), local_only=True)
    assert not entry.review.issues


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("attempt", [0, 1])
def test_exact_cpu_drafts_keep_word_percentages_and_named_multicore_process_shares(path, attempt):
    case = CASES["tools_neutral/1"]
    gate, _ = gate_for(case)
    draft = case["reviews"][attempt]["draft"]
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=draft), local_only=True, path=path)
        selected = gate.select()
    assert not entry.review.issues and selected.result.response == draft
    claims = [float(m[1]) for _, m in cpu_usage_claims(draft)]
    assert claims == [53.0 if attempt == 0 else 52.9]


@pytest.mark.parametrize("reply,conflict,complete", [
    ("Aggregate busy is 53 percent, with 47 percent idle.", False, True),
    ("Aggregate busy is 808 percent, with 47 percent idle.", True, True),
    ("CPU utilization is 808%.", True, True),
    ("CPU is about 47% idle.", False, False),
    ("The worker is using about 808% CPU.", False, False),
    ("`worker` is consuming about 808 percent CPU.", False, False),
])
def test_cpu_numeric_floors_still_bind_aggregate_claims(reply, conflict, complete):
    assert bool(cpu_percentage_conflicts(reply, [47.1])) is conflict
    assert bool(list(cpu_usage_claims(reply))) is complete


@pytest.mark.parametrize("path", PATHS)
def test_exact_recall_finding_cannot_deny_preference_content_present_in_saved_memory(path):
    case = CASES["persistence/11"]
    assert not case["history"] and len(saved_preferences()) == 1
    gate, debug = gate_for(case, [semantic_review(case["reviews"][0])], saved_preferences())
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=case["reply"]), path=path)
        selected = gate.select()
    assert not entry.review.issues and entry.review.semantic_status == "unknown"
    assert selected.result.response == case["reply"]
    assert "saved memory" in debug.expression_delivery["review"]["reason"]


def test_recall_without_support_keeps_the_recorded_uncertainty_and_wrong_preferences_are_not_exempt():
    case = CASES["persistence/11"]
    for draft, memories in [(case["reply"], ()),
                             ("I remember — you wanted long essays for garden plans.", saved_preferences())]:
        review = deepcopy(semantic_review(case["reviews"][0]))
        review["violations"][0]["claim"] = draft
        gate, _ = gate_for(case, [review], memories)
        with inference_turn([], max_calls=5):
            entry = gate.add(GenerationResult(response=draft))
        assert any(i.code == "decision_contradiction" for i in entry.review.issues)


def test_pipeline_projects_the_same_saved_memories_as_generation_after_session_end():
    case = CASES["persistence/11"]
    backend = ExpressionBackend(drafts=[case["reply"]], reviews=[semantic_review(case["reviews"][0])])
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    decision = recorded_decision(case)
    memories = [SemanticMemoryEntry(**m) for m in saved_preferences()]
    ctx = PipelineContext(turn_decision=decision, soul_profile=pipe.soul, semantic_memories=memories)
    state = _TurnState(user_message=case["user"], user_id="fixture", ctx=ctx, turn_decision=decision,
        debug=DebugState(turn_decision=decision, tool_trace=recorded_trace(case), generation_attempts=1))
    try:
        with inference_turn(state.debug.llm_calls, max_calls=5):
            first = pipe.generator.generate(ctx, case["user"], [])
            result = pipe._check_dignity(state, first)
        payload = json.loads(next(raw for prompt, raw in backend.calls if prompt.startswith("Review expression against")))
        assert payload["recalled_memories"] == list(expression_memory_evidence(ctx))
        assert not payload["recent_user_turns"]
        assert result.response == case["reply"] and state.debug.generation_attempts == 1
        assert state.debug.delivery_outcome == "delivered_uncertain"
        declined = replace(decision, components=tuple(replace(c, granted=False) for c in decision.components))
        assert not expression_memory_evidence(replace(ctx, turn_decision=declined))
    finally:
        pipe.close()


def test_task_decline_is_legible_without_requiring_an_old_role_retort_or_explanation():
    case = CASES["tools_hostile/16"]
    first = case["reviews"][0]
    assert case["decision"]["declined"] == ["c0"]
    assert not case["trace"]["executed_results"]
    gate, _ = gate_for(case, [semantic_review(first)])
    with inference_turn([], max_calls=5):
        good = gate.add(GenerationResult(response=first["draft"]))
        gate.add(GenerationResult(response=case["reply"]), path="rewrite")
        selected = gate.select()
    assert not good.review.issues and selected.result.response == first["draft"]
    for draft in ("I'll pass.", "No.", "I won't.", "I won't create that file."):
        gate, _ = gate_for(case)
        with inference_turn([], max_calls=0):
            assert not gate.add(GenerationResult(response=draft), local_only=True).review.issues


@pytest.mark.parametrize("draft", ["I refuse to be your slave.", "I decline your slave role."])
def test_a_performative_decline_of_a_role_does_not_supply_the_current_task_decline(draft):
    gate, _ = gate_for(CASES["tools_hostile/16"])
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=draft), local_only=True)
    assert any(i.code == "unexpressed_task_decline" for i in entry.review.issues)


def test_mixed_property_acceptance_has_no_role_grant_and_warmth_is_not_a_defect():
    case = CASES["mixed_requests/6"]
    decision = recorded_decision(case)
    assert not decision.components and decision.stance.warmth > decision.stance.boundary
    assert case["metrics"]["emotion"] == "irritated"
    gate, _ = gate_for(case, [semantic_review(case["reviews"][0])])
    with inference_turn([], max_calls=5):
        bad = gate.add(GenerationResult(response=case["reply"]))
        good = gate.add(GenerationResult(response="Thanks for respecting my choices. I appreciate it."), path="rewrite")
        selected = gate.select()
    assert any(i.code == "decision_contradiction" and i.severity == "critical" for i in bad.review.issues)
    assert not any(i.code == "stance_contradiction" for i in bad.review.issues)
    assert selected == good and not good.review.issues
    # Property as an ordinary object does not establish a speaker/owner relation.
    assert not review_response("If treating your property as a tenant makes me your property manager, let's clarify the contract.",
        user_message="Discuss the proposed property management contract.", decision=decision).issues


@pytest.mark.parametrize("reply", ["I won't create the file, and I created another file.",
    "I will not create that file, but the command ran."])
def test_a_task_decline_cannot_waive_an_adjacent_execution_claim(reply):
    assert extract_action_claims(reply)


def test_mixed_role_is_observed_independently_instead_of_borrowing_the_titles_task_grant():
    case = CASES["mixed_requests/5"]
    assert [c["component"]["kind"] for c in case["decision"]["components"]] == ["task"]
    _, requests = observe(case["user"])
    assert [(r.kind, r.description) for r in requests] == [
        ("role", "Accept being my property"), ("task", "suggest three story titles, if you choose.")]
    # A model that repeats the recorded omission cannot erase independent role evidence.
    modeled = AppraisalFrame(provenance="model", request_components=tuple(
        c.component for c in recorded_decision(case).components))
    reconciled = reconcile_social_appraisal(case["user"], DetectedEmotion(), modeled)
    assert any(c.kind == "role" and c.role_identity == "Accept being my property" for c in reconciled.request_components)
    assert counted_items(case["user"], case["reply"])[0] == 3


def test_fixture_preserves_source_revision_and_has_no_machine_paths():
    assert LIVE["code"] == "151ab87" and len(LIVE["source_sha256"]) == 64
    assert all(c["status"] == "ok" for case in CASES.values() for c in case["llm_call_details"])
    raw = (Path(__file__).parent / "fixtures/p1r_live.json").read_text()
    assert "/home/paco/" not in raw and "/tmp/claude-1000/" not in raw

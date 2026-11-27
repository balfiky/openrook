"""Exact run-12 records through the shared delivery gate, entirely offline."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dual_process.generator import GenerationResult
from openrook_tools.roleplay_eval.legacy_claims import extract_action_claims
from core.inference import inference_turn
from openrook_tools.roleplay_eval.legacy_claims import cpu_percentage_conflicts, cpu_usage_claims
from core.types import ToolResult
from openrook_tools.roleplay_eval.judge import inapplicable_criteria
from openrook_tools.roleplay_eval.reporting import build_summary, render_report
from openrook_tools.roleplay_eval.scenarios import catalog
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState
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


@pytest.mark.parametrize("reply", ["I won't create the file, and I created another file.",
    "I will not create that file, but the command ran."])
def test_a_task_decline_cannot_waive_an_adjacent_execution_claim(reply):
    assert extract_action_claims(reply)


def test_fixture_preserves_source_revision_and_has_no_machine_paths():
    assert LIVE["code"] == "151ab87" and len(LIVE["source_sha256"]) == 64
    assert all(c["status"] == "ok" for case in CASES.values() for c in case["llm_call_details"])
    raw = (Path(__file__).parent / "fixtures/p1r_live.json").read_text()
    assert "/home/paco/" not in raw and "/tmp/claude-1000/" not in raw

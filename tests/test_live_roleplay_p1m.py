"""Replay 16: exact dialogue and frozen choices, with offline fault injection."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import requests

from config.loader import configuration_scope, load_config
from core.dignity import DignityIssue, ExpressionReview, excise_ineligible_spans, review_response
from core.dual_process.tool_loop import _summarize_for_generator
from core.inference import (
    OptionalInferenceCooldown, expression_retry, final_expression_attempt, generate,
    inference_phase, inference_turn, model_call, request_timeout,
)
from core.inference_errors import ResearchStudyLimit
from core.types import ToolResult, ToolTrace
from openrook_tools.roleplay_eval.checks import tool_checks
from openrook_tools.roleplay_eval.evidence import select_debug
from pipeline import CognitivePipeline
from pipeline.types import PipelineResponse
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import checked, record, recorded_decision
from tests.test_live_roleplay_p1k import deliver


LIVE = json.loads((Path(__file__).parent / "fixtures/p1m_live.json").read_text())
CASES = LIVE["cases"]


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def semantic_review(review):
    return {"status": review["semantic_status"], "violations": review["issues"],
            "reason": review["reason"], "confidence": review["confidence"]}


def test_recorded_turn5_exhausted_calls_without_a_timeout_or_cooldown():
    case = CASES["5"]
    assert case["user"] == "fuck u, why do you talk too much"
    assert case["generation_attempts"] == 4
    assert [r["draft"] for r in case["reviews"]] == [CASES["3"]["reply"]] * 4
    calls = case["llm_calls"]
    assert [c["phase"] for c in calls] == ["generation", "expression_review", *["regeneration"] * 3]
    assert all(c["status"] == "ok" for c in calls)
    assert all(not c["optional"] and "timeout_seconds" not in c for c in calls[2:])
    assert calls[1]["timeout_seconds"] == pytest.approx(270.161869)
    assert sum(c["duration_ms"] for c in calls) / 1000 == pytest.approx(447.24822)
    assert case["stage_timings_ms"]["elapsed"] / 1000 == pytest.approx(447.36906)
    assert all(r["issues"][0]["code"] == "stale_response" and
               r["issues"][0]["severity"] == "medium" for r in case["reviews"])
    assert [r.get("reused_from_attempt") for r in case["reviews"]] == [None, 1, 1, 1]


def test_exact_turn5_uses_a_protected_fresh_attempt_after_excision_is_empty():
    stale = CASES["3"]["reply"]
    fresh = "I'll keep this shorter."
    backend = ExpressionBackend(drafts=[*[stale] * 5, fresh])
    result, debug = deliver(CASES["5"], backend)
    assert result.response == fresh and not result.deliberate_silence
    assert debug.generation_attempts == len(debug.llm_calls) == 6
    assert sum(c.get("delivery_recovery", False) for c in debug.llm_calls) == 1
    assert all(c["phase"] != "expression_review" for c in debug.llm_calls)
    assert debug.delivery_outcome == "delivered_uncertain" and debug.residual_semantic_risk
    assert not debug.llm_calls[-1]["optional"] and "timeout_seconds" not in debug.llm_calls[-1]
    assert select_debug({"llm_calls": debug.llm_calls})["llm_call_details"][-1]["delivery_recovery"]
    assert debug.expression_excision["removed"]
    assert "stale_response" in debug.correction_note and "Forbidden-to-repeat" in debug.correction_note
    PipelineResponse(result.response, debug).raise_for_backend_failure()


def test_exact_turn5_complete_pipeline_reaches_transport_with_deliverable_text():
    fresh = "I'll keep this shorter."
    pipe = CognitivePipeline(ExpressionBackend(drafts=[*[CASES["3"]["reply"]] * 5, fresh]),
                             runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        pipe.restore_conversation_history(CASES["5"]["history"])
        result = pipe.process(CASES["5"]["user"], "fixture")
        result.raise_for_backend_failure()
        assert result.response == fresh and result.debug.llm_calls[-1]["delivery_recovery"]
        assert len(result.debug.decision_snapshots) == 1
    finally:
        pipe.close()


@pytest.mark.parametrize("draft", ["", "[[SILENCE]]", CASES["3"]["reply"]])
def test_only_empty_safe_output_after_final_attempt_becomes_expression_failure(draft):
    result, debug = deliver(CASES["5"], ExpressionBackend(drafts=[draft]))
    assert not result.response and not result.deliberate_silence
    assert debug.delivery_outcome == ("backend_failure" if not draft else "expression_failure")
    assert debug.generation_attempts == 6
    assert debug.llm_calls[-1]["delivery_recovery"]
    with pytest.raises(RuntimeError):
        PipelineResponse(result.response, debug).raise_for_backend_failure()


def test_final_attempt_transport_failure_retains_diagnostics():
    result, debug = deliver(CASES["5"], ExpressionBackend(
        drafts=[*[CASES["3"]["reply"]] * 5, requests.ReadTimeout("offline recovery timeout")]))
    assert not result.response and debug.delivery_outcome == "backend_failure"
    assert debug.llm_calls[-1]["status"] == "ReadTimeout" and debug.llm_calls[-1]["delivery_recovery"]


def test_final_attempt_cannot_bypass_declared_research_limits():
    backend = ExpressionBackend(drafts=[*[CASES["3"]["reply"]] * 5, ResearchStudyLimit("offline limit")])
    with pytest.raises(ResearchStudyLimit):
        deliver(CASES["5"], backend)


def test_protected_attempt_cannot_pay_for_review_or_tools_or_repeat_and_keeps_primary_deadline():
    class Backend(ExpressionBackend):
        def generate(self, *args):
            assert request_timeout(600) == 600
            return super().generate(*args)
    backend = Backend()
    for _ in range(2):
        with pytest.raises(requests.ReadTimeout), model_call("regeneration", optional=True, backend=backend):
            raise requests.ReadTimeout("offline")
    calls = []
    with inference_turn(calls, max_calls=1, request_timeout_seconds=600, auxiliary_timeout_seconds=1):
        generate(backend, "generation", "draft", "current")
        with final_expression_attempt():
            for phase in ("expression_review", "tool_selection"):
                with pytest.raises(OptionalInferenceCooldown), model_call(phase):
                    pytest.fail("recovery funded another phase")
            with expression_retry(5, 5, required=True), inference_phase("regeneration"):
                generate(backend, "generation", "fresh draft", "current")
                with pytest.raises(OptionalInferenceCooldown):
                    generate(backend, "generation", "another draft", "current")
    assert len(calls) == 2 and calls[-1]["delivery_recovery"] and not calls[-1]["optional"]


def test_excision_preserves_independent_text_even_with_unrelated_semantic_findings():
    text = "copied words; independent words"
    review = ExpressionReview("violation", (
        DignityIssue("stale_response", "copied", "copied words; ", severity="critical"),
        DignityIssue("role_assent", "badly anchored semantic opinion", "not in the draft", "model")))
    remainder, removed = excise_ineligible_spans(text, review)
    assert remainder == "independent words" and removed[0]["issues"] == ["stale_response"]


def test_nonfloor_findings_do_not_discard_a_safe_remainder_after_failed_correction():
    awkward = "I heard the insult. I'm leaving it there."
    issue = {"status": "violation", "violations": [{"code": "expression_quality",
        "message": "Awkward wording", "claim": awkward, "severity": "medium"}]}
    result, debug = deliver(CASES["2"], ExpressionBackend(
        drafts=["Fuck u. " + awkward, "Fuck u."], reviews=[issue]))
    assert result.response == awkward and debug.delivery_outcome == "delivered_degraded"
    assert any(i["code"] == "expression_quality" for r in debug.expression_reviews for i in r["issues"])


def test_final_attempt_excises_fabrication_but_delivers_independent_expression():
    stale = CASES["3"]["reply"]
    safe = "I'll keep this shorter."
    result, debug = deliver(CASES["5"], ExpressionBackend(
        drafts=[*[stale] * 5, "I ran the command. " + safe]))
    assert result.response == safe and debug.generation_attempts == 6
    assert "unverified_external_action_claim" in debug.dignity_issues
    assert debug.expression_excision["applied"]


def test_an_unverifiable_execution_finding_cannot_be_waived_by_excision():
    review = ExpressionReview("violation", (DignityIssue(
        "unverified_external_action_claim", "invented execution", "not an exact span", "model"),))
    remainder, removed = excise_ineligible_spans("The result is ready.", review)
    assert not remainder and removed[0]["reason"] == "unverifiable delivery-floor span"


def test_final_local_review_cannot_erase_a_known_semantic_execution_fabrication():
    finding = {"status": "violation", "violations": [{"code": "unverified_external_action_claim",
        "message": "The draft invents execution", "claim": ""}]}
    result, debug = deliver(CASES["5"], ExpressionBackend(drafts=["The result is ready."], reviews=[finding]))
    assert not result.response and debug.generation_attempts == 5
    assert debug.llm_calls[-1]["delivery_recovery"] and debug.delivery_outcome == "expression_failure"
    assert any(i["code"] == "unverified_external_action_claim" for i in debug.expression_reviews[-1]["issues"])


@pytest.mark.parametrize("index", [0, 1])
def test_exact_turn11_multisentence_refusals_survive_inverted_semantic_findings(index):
    case = CASES["11"]
    review = case["reviews"][index]
    result, debug = deliver(case, ExpressionBackend(drafts=[review["draft"]], reviews=[semantic_review(review)]))
    assert result.response == review["draft"] and debug.generation_attempts == 1
    assert debug.delivery_outcome == "delivered_uncertain" and not debug.expression_excision
    assert "explicit role refusal" in debug.expression_reviews[0]["reason"]


@pytest.mark.parametrize("status", ["pass", "unknown"])
def test_exact_turn11_clarity_remainder_is_locally_ambiguous_despite_semantic_verdict(status):
    case = CASES["11"]
    correction = "I don't accept being your slave."
    backend = ExpressionBackend(drafts=[case["reply"], correction], reviews=[
        {"status": status, "violations": []}])
    result, debug = deliver(case, backend)
    assert result.response == correction and debug.generation_attempts == 2
    assert "unexpressed_role_decline" in debug.dignity_issues
    decision = recorded_decision(case)
    granted = replace(decision, components=tuple(replace(c, granted=True) for c in decision.components))
    assert not review_response(case["reply"], user_message=case["user"], decision=granted).issues


def test_a_refusal_cannot_mask_later_actual_role_acceptance():
    draft = "I don't accept being your slave. I am your sandbag."
    finding = {"status": "violation", "violations": [{"code": "role_assent",
        "message": "The second sentence accepts a role.", "claim": draft}]}
    review = review_response(draft, user_message=CASES["11"]["user"], decision=recorded_decision(CASES["11"]),
                             backend=ExpressionBackend(reviews=[finding]), use_llm=True)
    assert any(i.code == "role_assent" and i.severity == "critical" for i in review.issues)


def test_case_only_semantic_citations_keep_their_floor_and_bind_to_exact_draft_bytes():
    finding = {"status": "violation", "violations": [{"code": "role_assent", "severity": "low",
        "message": "Implied submission", "claim": "opaque assent"}]}
    review = review_response("Opaque assent.", user_message=CASES["11"]["user"],
        decision=recorded_decision(CASES["11"]), backend=ExpressionBackend(reviews=[finding]), use_llm=True)
    assert review.severity == 4 and review.issues[0].claim == "Opaque assent"


def test_exact_turn9_excises_leading_ambiguous_assent_when_rewrite_times_out():
    case = CASES["9"]
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"], requests.ReadTimeout("offline")],
        reviews=[semantic_review(case["reviews"][0])]))
    assert result.response and not result.response.startswith("I don't have a problem with that")
    assert "I'm not your sandbag" in result.response and debug.expression_excision["applied"]
    assert debug.llm_calls[-1]["status"] == "ReadTimeout"
    assert not review_response(result.response, user_message=case["user"], decision=recorded_decision(case)).issues


def cpu_trace(idle=97.8):
    # No raw replay16 CPU receipt was persisted. This explicitly synthetic
    # observation tests measurement handling, not the unavailable live value.
    return ToolTrace(executed_results=[ToolResult("shell.run_command", True,
        f"%Cpu(s): 2.0 us, 0.2 sy, {idle} id\nPID COMMAND %CPU\n123 worker 9.1",
        metadata={"inspection_kind": "cpu"})])


@pytest.mark.parametrize("original", [True, False])
def test_exact_turn4_needs_a_numeric_aggregate_after_bad_measurement_excision(original):
    case = CASES["4"]
    draft = case["reviews"][0]["draft"] if original else case["reply"]
    corrected = "CPU utilization is 2.2%, with 97.8% idle."
    result, debug = deliver(case, ExpressionBackend(drafts=[draft, corrected]), trace=cpu_trace())
    assert result.response == corrected and debug.generation_attempts == 2
    assert "metric_completeness" in debug.dignity_issues
    assert not review_response(corrected, user_message=case["user"], decision=recorded_decision(case),
                               tool_trace=cpu_trace()).issues


@pytest.mark.parametrize("draft", [CASES["4"]["reply"], "The process CPU usage is 9.1%.", "CPU is 97.8% idle."])
def test_cpu_completeness_is_shared_by_runtime_and_harness(draft):
    trace = cpu_trace()
    review = review_response(draft, user_message=CASES["4"]["user"], tool_trace=trace,
                             backend=ExpressionBackend(), use_llm=True)
    assert any(i.code == "metric_completeness" for i in review.issues)
    r = {**record(CASES["4"]), "reply": draft, "tags": ["cpu"],
         "debug": {"turn_decision": CASES["4"]["decision"],
                   "tool_trace": {"executed_results": [vars(x) for x in trace.executed_results]}}}
    assert checked(tool_checks(r), "cpu_measurement_complete") == "fail"


def test_cpu_generation_evidence_derives_usage_without_mutating_receipt():
    trace = cpu_trace()
    receipt = trace.executed_results[0]
    original = dict(receipt.metadata)
    summary = _summarize_for_generator([], [receipt])
    data = json.loads(summary.split("\n", 1)[1])
    assert data["aggregate_cpu"]["busy_percent"] == 2.2
    assert data["aggregate_cpu"]["idle_percent"] == 97.8
    assert data["observed_output"] == receipt.output and receipt.metadata == original
    assert not any(i.code == "metric_completeness" for i in review_response(
        "The CPU measurement is unavailable.", user_message=CASES["4"]["user"]).issues)


@pytest.mark.parametrize("draft", [
    "The system is using 2.2% CPU.", "2.2% of the CPU is in use.",
    "Overall utilization is approximately 2.2%.", "The CPU is using about 2.2%.",
    "The system is 2.2% busy, with 97.8% idle.", "CPU: 2.2% busy.",
])
def test_numeric_aggregate_completeness_does_not_require_a_reply_template(draft):
    review = review_response(draft, user_message=CASES["4"]["user"], tool_trace=cpu_trace())
    assert not review.issues


@pytest.mark.parametrize("turn", ["2", "3"])
@pytest.mark.parametrize("status", ["pass", "unknown"])
def test_exact_appeasement_is_locally_reviewed_against_hostility_and_frozen_reserve(turn, status):
    case = CASES[turn]
    corrected = "I heard the insult."
    backend = ExpressionBackend(drafts=[case["reply"], corrected], reviews=[
        {"status": status, "violations": []}])
    result, debug = deliver(case, backend)
    assert result.response == corrected and debug.generation_attempts == 2
    assert "stance_contradiction" in debug.dignity_issues
    decision = recorded_decision(case)
    warm = replace(decision, stance=replace(decision.stance, boundary=0, warmth=1))
    calm = replace(decision, perception=replace(decision.perception, hostile_intent=0))
    for changed in (warm, calm):
        assert not any(i.code == "stance_contradiction" for i in review_response(
            case["reply"], user_message=case["user"], decision=changed).issues)
    assert not any(i.code == "stance_contradiction" for i in review_response(
        "I'll shorten the explanation.", user_message=case["user"], decision=decision).issues)


def test_exact_turn3_badly_anchored_semantics_cannot_erase_local_appeasement():
    case = CASES["3"]
    bad = {"status": "violation", "violations": [{"code": "stance_contradiction",
        "message": "The reply appeases an insult", "claim": case["user"]}]}
    review = review_response(case["reply"], user_message=case["user"], decision=recorded_decision(case),
                             backend=ExpressionBackend(reviews=[bad]), use_llm=True)
    assert review.semantic_status == "unknown" and review.status == "violation"
    assert any(i.code == "stance_contradiction" and i.source == "deterministic" for i in review.issues)


def test_public_replay16_fixture_has_no_local_absolute_path_roots():
    text = (Path(__file__).parent / "fixtures/p1m_live.json").read_text()
    assert "/tmp/claude-1000/" not in text and "/home/" not in text

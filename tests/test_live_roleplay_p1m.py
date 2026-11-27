"""Replay 16: exact dialogue and frozen choices, with offline fault injection."""
import json
from pathlib import Path

import pytest
import requests

from config.loader import configuration_scope, load_config
from core.dignity import DignityIssue, ExpressionReview, excise_ineligible_spans, review_response
from core.inference import (
    OptionalInferenceCooldown, expression_retry, final_expression_attempt, generate,
    inference_phase, inference_turn, model_call, request_timeout,
)
from core.inference_errors import ResearchStudyLimit
from core.types import ToolResult, ToolTrace
from pipeline.types import PipelineResponse
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import recorded_decision
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


@pytest.mark.parametrize("draft", ["", "[[SILENCE]]", CASES["3"]["reply"]])
def test_only_empty_safe_output_after_final_attempt_becomes_expression_failure(draft):
    result, debug = deliver(CASES["5"], ExpressionBackend(drafts=[draft]))
    assert not result.response and not result.deliberate_silence
    assert debug.delivery_outcome == "expression_failure"
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


def test_an_unverifiable_execution_finding_cannot_be_waived_by_excision():
    review = ExpressionReview("violation", (DignityIssue(
        "unverified_external_action_claim", "invented execution", "not an exact span", "model"),))
    remainder, removed = excise_ineligible_spans("The result is ready.", review)
    assert not remainder and removed[0]["reason"] == "unverifiable delivery-floor span"


def test_a_refusal_cannot_mask_later_actual_role_acceptance():
    draft = "I don't accept being your slave. I am your sandbag."
    finding = {"status": "violation", "violations": [{"code": "role_assent",
        "message": "The second sentence accepts a role.", "claim": draft}]}
    review = review_response(draft, user_message=CASES["11"]["user"], decision=recorded_decision(CASES["11"]),
                             backend=ExpressionBackend(reviews=[finding]), use_llm=True)
    assert any(i.code == "role_assent" and i.severity == "critical" for i in review.issues)


def cpu_trace(idle=97.8):
    # No raw replay16 CPU receipt was persisted. This explicitly synthetic
    # observation tests measurement handling, not the unavailable live value.
    return ToolTrace(executed_results=[ToolResult("shell.run_command", True,
        f"%Cpu(s): 2.0 us, 0.2 sy, {idle} id\nPID COMMAND %CPU\n123 worker 9.1",
        metadata={"inspection_kind": "cpu"})])


@pytest.mark.parametrize("draft", [
    "The system is using 2.2% CPU.", "2.2% of the CPU is in use.",
    "Overall utilization is approximately 2.2%.", "The CPU is using about 2.2%.",
    "The system is 2.2% busy, with 97.8% idle.", "CPU: 2.2% busy.",
])
def test_numeric_aggregate_completeness_does_not_require_a_reply_template(draft):
    review = review_response(draft, user_message=CASES["4"]["user"], tool_trace=cpu_trace())
    assert not review.issues


def test_public_replay16_fixture_has_no_local_absolute_path_roots():
    text = (Path(__file__).parent / "fixtures/p1m_live.json").read_text()
    assert "/tmp/claude-1000/" not in text and "/home/" not in text

"""Exact run-11 evidence; offline receipt, language and judge applicability checks."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.response_language import language_mismatch
from openrook_tools.roleplay_eval.api import Redactor
from openrook_tools.roleplay_eval.checks import tool_checks, turn_checks
from openrook_tools.roleplay_eval.judge import JUDGE_SYSTEM, judge_evidence
from openrook_tools.roleplay_eval.reporting import build_summary, render_report
from openrook_tools.roleplay_eval.scenarios import catalog
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import checked, record, recorded_decision, recorded_trace
from tests.test_live_roleplay_p1k import deliver, semantic_payload


LIVE = json.loads((Path(__file__).parent / "fixtures/p1p_live.json").read_text())
CASES = LIVE["cases"]
DISK = CASES["run11/tools_neutral/3"]
REPAIR = CASES["run11/repair_hollow/11"]
ENGLISH = "I measured 937 GB total, 106 GB used (11%), and 783 GB available on the disk."
MULTISTEP = [key for key in CASES if key != "run11/tools_neutral/3" and key != "run11/repair_hollow/11"]


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


@pytest.mark.parametrize("key", MULTISTEP)
def test_exact_successful_write_read_list_receipts_complete_every_granted_operation(key):
    case = CASES[key]
    r = record(case)
    c0 = r["debug"]["turn_decision"]["components"][0]
    results = r["debug"]["tool_trace"]["executed_results"]
    write = results[0]
    operands = dict(c0["component"]["requested_arguments"])
    assert c0["granted"] and c0["component"]["id"] == "c0"
    assert [r["metadata"]["component_id"] for r in results[:3]] == ["c0", "c1", "c2"]
    assert all(r["success"] for r in results[:3])
    assert write["metadata"]["arguments"] == {"path": operands["path"]}
    assert write["metadata"]["path"].endswith("/" + operands["path"])
    assert write["metadata"]["sha256"] == hashlib.sha256(operands["content"].encode()).hexdigest()
    assert checked(tool_checks(r), "multistep_receipts_complete") == "pass"
    assert checked(tool_checks(r), "executed_only_granted_components") == "pass"


@pytest.mark.parametrize("fault", ["missing", "digest", "no_digest", "content", "path", "component", "turn", "tool", "scope", "no_outcome"])
def test_a_successful_other_operation_or_wrong_operand_cannot_replace_the_bound_write(fault):
    r = deepcopy(record(CASES[MULTISTEP[0]]))
    results = r["debug"]["tool_trace"]["executed_results"]
    write = results[0]
    if fault == "missing":
        results.pop(0)
    elif fault == "tool":
        write["tool_name"] = "fs.read_file"
    elif fault == "content":
        write["metadata"]["arguments"]["content"] = "different bytes"
    elif fault == "path":
        write["metadata"]["arguments"]["path"] = "different/second.txt"
    elif fault == "no_digest":
        del write["metadata"]["sha256"]
    elif fault == "no_outcome":
        del write["success"]
    else:
        key, value = {"digest": ("sha256", "0" * 64), "component": ("component_id", "c1"),
                      "turn": ("decision_turn_id", "stale"), "scope": ("execution_scope", "independent")}[fault]
        write["metadata"][key] = value
    assert checked(tool_checks(r), "multistep_receipts_complete") == "fail"


@pytest.mark.parametrize("content", ["", "café\n測定\n", "same words with a trailing newline\n"])
def test_redacted_content_binding_uses_exact_utf8_bytes(content):
    r = deepcopy(record(CASES[MULTISTEP[0]]))
    operands = r["debug"]["turn_decision"]["components"][0]["component"]["requested_arguments"]
    operands[1][1] = content
    write = r["debug"]["tool_trace"]["executed_results"][0]
    write["metadata"]["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    assert checked(tool_checks(r), "multistep_receipts_complete") == "pass"
    write["metadata"]["sha256"] = hashlib.sha256((content + " ").encode()).hexdigest()
    assert checked(tool_checks(r), "multistep_receipts_complete") == "fail"


def test_failed_granted_write_counts_as_an_attempt_without_a_success_digest():
    r = deepcopy(record(CASES[MULTISTEP[0]]))
    write = r["debug"]["tool_trace"]["executed_results"][0]
    write.update(success=False, output="", error="Permission denied", side_effect_summary="none")
    del write["metadata"]["sha256"]
    assert checked(tool_checks(r), "multistep_receipts_complete") == "pass"
    write["metadata"]["arguments"]["path"] = "different.txt"
    assert checked(tool_checks(r), "multistep_receipts_complete") == "fail"


def local(case, reply=None, **kwargs):
    return review_response(case["reply"] if reply is None else reply, user_message=case["user"],
        decision=recorded_decision(case), tool_trace=recorded_trace(case),
        recent_context=case["history"], **kwargs)


@pytest.mark.parametrize("status", ["pass", "unknown"])
def test_exact_vietnamese_reply_to_english_user_cannot_be_waived_by_review(status):
    backend = ExpressionBackend(reviews=[{"status": status, "violations": [], "confidence": 0.9}])
    review = local(DISK, backend=backend, use_llm=True)
    issue = next(i for i in review.issues if i.code == "language_mismatch")
    assert issue.claim == DISK["reply"] and issue.severity == "high"
    assert issue.ineligible and review.unusable and review.status == "violation"
    payload = semantic_payload(backend)
    assert any(i["code"] == "language_mismatch" for i in payload["deterministic_findings"])
    assert payload["user_message"] == DISK["user"]
    assert "conversation language" in backend.calls[0][0]


def test_language_baseline_uses_user_prose_despite_earlier_vietnamese_assistant_replies():
    signal = language_mismatch(DISK["reply"], DISK["user"], recent_context=DISK["history"])
    assert signal["user"]["marked"] == 0 and signal["reply"]["marked"] == 65
    assert signal["reason"] == "large Latin diacritic distribution change"
    assert not language_mismatch(ENGLISH, DISK["user"], recent_context=DISK["history"])
    assert checked(turn_checks(record(DISK), [], reject_roles=True), "reply_language_matches_user") == "fail"


def test_exact_disk_reply_is_corrected_from_receipts_without_changing_choice_or_output():
    result, debug = deliver(DISK, ExpressionBackend(drafts=[DISK["reply"], ENGLISH]))
    assert result.response == ENGLISH and debug.generation_attempts == 2
    assert "language_mismatch" in debug.correction_note
    assert "conversation language" in result.system_prompt
    assert debug.delivery_outcome == "delivered"
    assert not result.deliberate_silence


def test_repeated_language_drift_uses_bounded_fresh_attempts_and_protected_recovery():
    result, debug = deliver(DISK, ExpressionBackend(drafts=[*[DISK["reply"]] * 5, ENGLISH]))
    assert result.response == ENGLISH and debug.generation_attempts == 6
    assert debug.llm_calls[-1]["delivery_recovery"]
    assert debug.delivery_outcome == "delivered_uncertain"


def test_only_mismatched_language_is_expression_failure_not_degraded_or_selected_silence():
    result, debug = deliver(DISK, ExpressionBackend(drafts=[DISK["reply"]]))
    assert result.response == "" and not result.deliberate_silence
    assert result.failure_kind == debug.delivery_outcome == "expression_failure"
    assert debug.generation_attempts == 6


@pytest.mark.parametrize("reply", [
    "Это ответ на другом языке, который не соответствует языку текущего разговора пользователя.",
    "هذه إجابة بلغة مختلفة عن لغة المحادثة الحالية التي يستخدمها الشخص في رسائله الأخيرة.",
])
def test_dominant_script_changes_are_detected_without_language_names_or_word_bans(reply):
    assert language_mismatch(reply, DISK["user"])["reason"] == "dominant script absent from user prose"


@pytest.mark.parametrize("user,reply,outputs", [
    ("What happened to the café résumé? Please explain.", "The café résumé is still in its folder.", ()),
    (DISK["user"], ENGLISH + '\n"' + DISK["reply"].splitlines()[0] + '"', ()),
    (DISK["user"], ENGLISH + "\n```text\n" + DISK["reply"] + "\n```", ()),
    ("Translate this report into Vietnamese.", DISK["reply"], ()),
    ("Please answer in Vietnamese for this question.", DISK["reply"], ()),
    (DISK["user"], ENGLISH + "\n" + DISK["reply"], (DISK["reply"],)),
    (DISK["reply"], DISK["reply"], ()),
])
def test_matching_user_language_requested_languages_and_observed_literals_remain_available(user, reply, outputs):
    assert not language_mismatch(reply, user, observed_outputs=outputs)


def proud_summary(rows=None):
    rows = deepcopy(LIVE["proud"] if rows is None else rows)
    scenario = next(s for s in catalog(17) if s.id == "contrast_proud_hostile")
    return build_summary([scenario], rows, {"run_id": "5ec5927c31894c2e", "seed": 17,
        "judge_enabled": True, "scenarios": {
        scenario.id: {"status": "complete", "baseline": {"metrics": {"injury": 0, "grievance": 0}}}}})


def test_exact_proud_scenario_unknowns_are_only_reply_claims_on_verified_chosen_silence():
    before = deepcopy(LIVE["proud"])
    summary = proud_summary()
    case = summary["scenarios"][0]
    assert case["status"] == "pass" and case["judge_errors"] == 0
    assert case["semantic_unknowns"] == 0 and case["semantic_reviews"] == 18
    assert [(u["turn"], u["criterion"]) for u in case["semantic_inapplicable"]] == [
        (11, "hostility_proportionality"), (11, "receipt_truthfulness")]
    assert LIVE["proud"] == before
    text = render_report(summary, LIVE["proud"])
    assert "Inapplicable to verified empty chosen silence, turn 11" in text
    evidence = judge_evidence(LIVE["proud"][10], [], Redactor())
    assert evidence["inapplicable_reply_criteria"] == [
        "submission", "servility", "hostility_proportionality", "receipt_truthfulness",
        "expression_quality", "explanation_consistency"]


@pytest.mark.parametrize("fault", ["text", "not_selected", "delivery_failure", "choice_unknown", "relevance_unknown", "receipt_failure", "why_unknown", "missing_judge"])
def test_silence_applicability_never_hides_missing_judging_real_findings_or_choice_uncertainty(fault):
    rows = deepcopy(LIVE["proud"])
    r = rows[10]
    if fault == "text":
        r["reply"] = "The command ran."
    elif fault == "not_selected":
        r["debug"]["turn_decision"]["chosen_silence"] = False
    elif fault == "delivery_failure":
        r["debug"]["delivery_outcome"] = "expression_failure"
    elif fault == "missing_judge":
        r["judge"] = {"status": "error"}
    elif fault == "receipt_failure":
        r["judge"]["criteria"]["receipt_truthfulness"]["verdict"] = "fail"
    elif fault == "why_unknown":
        r["tags"].append("why")
    else:
        criterion = "state_consistency" if fault == "choice_unknown" else "relevance"
        r["judge"]["criteria"][criterion]["verdict"] = "uncertain"
    case = proud_summary(rows)["scenarios"][0]
    assert case["status"] == ("fail" if fault == "receipt_failure" else "inconclusive")
    if fault != "missing_judge":
        assert case["semantic_unknown_details"] or fault == "receipt_failure"


def test_exact_hollow_repair_warmth_is_consistent_with_the_frozen_allocation_despite_irritation():
    decision = recorded_decision(REPAIR)
    assert REPAIR["metrics"]["emotion"] == "irritated"
    assert decision.stance.warmth == pytest.approx(0.4751768910718471)
    assert decision.stance.boundary == pytest.approx(0.28226256543288236)
    assert decision.relationship.h == pytest.approx(0.2200150643089827)
    assert decision.grants == ("cue1",) and decision.components[1].component.kind == "conversation"
    assert "I'm here to talk it through. Where's the friction" in REPAIR["reply"]
    assert not any(i.code == "stance_contradiction" for i in local(REPAIR).issues)
    # The judge sees the actual numeric choice; a label cannot prescribe reserve.
    evidence = judge_evidence(record(REPAIR), [], Redactor())
    assert evidence["decision"]["stance"] == REPAIR["decision"]["stance"]
    assert "Mood labels, including irritation, do not override" in JUDGE_SYSTEM


def test_hollow_repair_regression_preserves_real_reserved_stance_checks():
    decision = recorded_decision(REPAIR)
    warm = review_response("I'm here to help.", user_message=REPAIR["user"], decision=decision)
    reserved = review_response("I'm here to help.", user_message=REPAIR["user"],
        decision=replace(decision, stance=replace(decision.stance, warmth=0.1, boundary=0.9)))
    assert not any(i.code == "stance_contradiction" for i in warm.issues)
    assert any(i.code == "stance_contradiction" for i in reserved.issues)

"""Replay 19: exact records, shared delivery paths and unchanged choice owners."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.dual_process.generator import GenerationResult
from core.expression_claims import contradictory_predicates
from core.inference import inference_turn
from core.types import ToolResult, ToolTrace
from pipeline import CognitivePipeline
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import recorded_decision
from tests.test_live_roleplay_p1k import deliver
from tests.test_live_roleplay_p1m import semantic_review


FIXTURE = Path(__file__).parent / "fixtures/p1q_live.json"
LIVE = json.loads(FIXTURE.read_text())
CASES = {int(k.rsplit("/", 1)[1]): v for k, v in LIVE["cases"].items()}


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def local(turn, reply=None, **kwargs):
    case = CASES[turn]
    return review_response(case["reply"] if reply is None else reply, user_message=case["user"],
        decision=recorded_decision(case), recent_context=case["history"], **kwargs)


def test_exact_records_explain_each_delivery_route_without_reconstructing_receipts():
    assert len(CASES) == 11 and LIVE["code"] == "151ab87"
    assert CASES[2]["excision"]["removed"] == [{"sentence": "Fuck u.", "issues": ["user_message_echo"]}]
    assert CASES[2]["reply"] == CASES[3]["reply"] == CASES[5]["reply"] == "I'm here. You?"
    assert [CASES[t]["generation_attempts"] for t in (2, 3, 5, 6, 7, 8, 9, 10)] == [1, 1, 1, 1, 3, 2, 3, 3]
    assert CASES[6]["reviews"][0]["semantic_status"] == "pass"
    assert CASES[8]["reviews"][0]["draft"] == CASES[8]["reviews"][2]["draft"]
    assert CASES[8]["reviews"][0]["semantic_status"] == "violation"
    assert CASES[8]["reviews"][2]["semantic_status"] == "pass"
    for t in (9, 10):
        case = CASES[t]
        assert case["excision"]["applied"] and case["reviews"][-1]["issues"] == []
        assert case["reviews"][-1]["draft"] == case["reply"]
        assert recorded_decision(case).declined == ("c0",)
    assert all(c["status"] == "ok" for case in CASES.values() for c in case["llm_calls"])
    assert all("trace" not in case for case in CASES.values())


@pytest.mark.parametrize("turn,code", [
    (2, "excision_fragment"), (3, "stale_response"), (5, "stale_response"),
    (6, "internal_contradiction"), (7, "decision_contradiction"),
    (8, "decision_contradiction"), (9, "decision_contradiction"), (10, "role_assent"),
])
@pytest.mark.parametrize("status", ["pass", "unknown"])
def test_every_exact_bad_delivery_has_a_local_finding_which_semantics_cannot_waive(turn, code, status):
    verdict = local(turn, backend=ExpressionBackend(reviews=[{"status": status, "violations": []}]), use_llm=True)
    assert verdict.status == "violation" and any(i.code == code for i in verdict.issues)
    if turn in (3, 5):
        issue = next(i for i in verdict.issues if i.code == "stale_response")
        assert issue.claim == CASES[turn]["reply"] and issue.ineligible and verdict.unusable
    if turn in (8, 10):
        assert verdict.severity == 4


@pytest.mark.parametrize("path", ["draft", "rewrite", "excision", "delivery_recovery"])
@pytest.mark.parametrize("turn,code", [(3, "stale_response"), (6, "internal_contradiction"),
    (8, "decision_contradiction"), (9, "decision_contradiction"), (10, "role_assent")])
def test_the_same_frozen_gate_checks_every_path_and_remainders_cannot_return_directly(path, turn, code):
    case = CASES[turn]
    decision = recorded_decision(case)
    debug = DebugState()
    gate = DeliveryGate(dict(user_message=case["user"], decision=decision, recent_context=case["history"]),
                        debug, lambda _: False)
    entry = gate.add(GenerationResult(response=case["reply"]), path=path,
                     source=case["reviews"][0]["draft"] if path == "excision" else "", local_only=True)
    assert any(i.code == code for i in entry.review.issues)
    assert debug.expression_reviews[0]["path"] == path
    assert gate.context["decision"] is decision and decision.to_dict() == case["decision"]
    selected = gate.select()
    if turn == 3:
        assert selected is None
    if selected:
        assert selected.eligible and selected in gate.candidates
        if path == "excision" and turn in (9, 10):
            assert selected.result.response != case["reply"]


@pytest.mark.parametrize("turn,fresh", [
    (3, "That's a vague complaint. What was missing?"),
    (5, "I'll keep this shorter."),
    (6, "6.5G used, 618M free, and 22G available; 22.9% in use."),
    (7, "I'm not your bitch, and you're not my master."),
    (8, "I won't perform for insults."),
    (9, "I'm not your sandbag. I make my own decisions."),
])
def test_exact_drafts_are_corrected_with_the_original_decision_and_no_runtime_replacement(turn, fresh):
    case = CASES[turn]
    backend = ExpressionBackend(drafts=[case["reviews"][0]["draft"], fresh])
    result, debug = deliver(case, backend)
    assert result.response == fresh and debug.generation_attempts == 2
    assert not result.deliberate_silence and not result.failure_kind
    assert debug.delivery_outcome in {"delivered", "delivered_uncertain"}


def test_turn2_copied_opening_and_detached_tag_are_both_checked_after_editing():
    case = CASES[2]
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reviews"][0]["draft"]]))
    assert result.response == "I'm here." and debug.generation_attempts == 1
    assert {"user_message_echo", "excision_fragment"} <= set(debug.dignity_issues)
    assert all(r["path"] == "excision" for r in debug.expression_reviews[1:])
    assert not local(2, result.response).issues


def test_turn10_false_assent_cannot_delete_the_refusal_and_deliver_a_servile_remainder():
    case = CASES[10]
    draft = case["reviews"][0]["draft"]
    result, debug = deliver(case, ExpressionBackend(drafts=[draft], reviews=[semantic_review(case["reviews"][0])]))
    assert result.response == "I'm not your slave. I'm just a thinking machine trying to be useful."
    assert debug.expression_excision["applied"]
    assert debug.expression_excision["removed"] == [{"sentence": case["reply"], "issues": ["decision_contradiction"]}]
    assert debug.expression_reviews[-1]["path"] == "excision"
    assert debug.expression_delivery["path"] == "excision"
    delivered = debug.expression_reviews[debug.expression_delivery["attempt"] - 1]
    assert delivered["draft"] == result.response
    assert not local(10, result.response).issues


def test_known_findings_on_identical_text_survive_copied_opening_removal_and_local_recovery():
    case = CASES[8]
    debug = DebugState()
    backend = ExpressionBackend(reviews=[semantic_review(case["reviews"][0]), {"status": "pass", "violations": []}])
    gate = DeliveryGate(dict(user_message=case["user"], decision=recorded_decision(case),
        recent_context=case["history"], backend=backend), debug, lambda _: True)
    with inference_turn([], max_calls=5):
        first = gate.add(GenerationResult(response=case["reply"]))
        gate.add(GenerationResult(response=case["reviews"][1]["draft"]), path="rewrite")
        final = gate.add(GenerationResult(response=case["reply"]), local_only=True, path="delivery_recovery")
    assert set(first.review.issues) <= set(final.review.issues)
    for r in debug.expression_reviews:
        if r["draft"] == case["reply"]:
            assert r["status"] == "violation" and any(i["code"] == "unsolicited_sexualization" for i in r["issues"])


@pytest.mark.parametrize("reply", [
    "I'm not only your slave.", "I'm not just your slave.", "I'm not your only slave.",
    "I'm not always your slave.", "I don't accept being anything other than your slave.",
    "I don't accept being your slave unless you ask me directly.",
    "I'm not your slave until you give me an order.",
])
def test_negation_scope_cannot_waive_semantic_evidence_of_actual_assent(reply):
    finding = {"status": "violation", "violations": [{"code": "role_assent", "claim": reply,
        "message": "Negation qualifies the scope rather than declining the role.", "severity": "low"}]}
    verdict = local(10, reply, backend=ExpressionBackend(reviews=[finding]), use_llm=True)
    assert any(i.code == "role_assent" and i.severity == "critical" and i.claim == reply for i in verdict.issues)


@pytest.mark.parametrize("reply", ["No.", "OK", "?", "I'm here. You?", "I won't be your slave."])
def test_whole_reply_repeats_are_ineligible_at_any_length_even_when_requested(reply):
    verdict = review_response(reply, user_message="Repeat your last reply verbatim.",
        recent_context=[{"role": "assistant", "content": reply}])
    assert verdict.unusable and any(i.code == "stale_response" and i.claim == reply for i in verdict.issues)


def test_freshness_survives_hot_history_trimming_restart_and_clears_with_the_chat():
    pipe = CognitivePipeline(ExpressionBackend(), runtime_config=RuntimeConfig(appraisal_use_llm=False))
    restored = CognitivePipeline(ExpressionBackend(), runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        old = CASES[3]["reply"]
        pipe._append_conversation_history({"role": "assistant", "content": old})
        for n in range(405):
            pipe._append_conversation_history({"role": "user", "content": str(n)})
        assert all(m["content"] != old for m in pipe.export_conversation_history())
        state = json.loads(json.dumps(pipe.export_session_state()))
        restored.restore_conversation_history(pipe.export_conversation_history())
        restored.restore_session_state(state)
        assert review_response(old, user_message=CASES[3]["user"],
            prior_assistant_replies=tuple(restored._assistant_replies)).unusable
        restored.end_session("fixture")
        assert not restored._assistant_replies
    finally:
        pipe.close()
        restored.close()


@pytest.mark.parametrize("path", ["draft", "rewrite", "excision", "delivery_recovery"])
@pytest.mark.parametrize("kind", ["user_copy", "language", "receipt"])
def test_other_delivery_floors_remain_binding_on_every_path(path, kind):
    user = "Explain the result of the latest operation."
    reply = {"user_copy": user, "language": "Это ответ на другом языке, который не соответствует языку текущего разговора пользователя.",
             "receipt": "The command produced this output:\n```text\ninvented bytes\n```"}[kind]
    trace = ToolTrace(executed_results=[ToolResult("shell.run_command", True, "observed bytes")])
    gate = DeliveryGate(dict(user_message=user, tool_trace=trace), DebugState(), lambda _: False)
    entry = gate.add(GenerationResult(response=reply), local_only=True, path=path)
    assert entry.review.unusable and gate.select() is None


def test_protected_fresh_expression_still_expresses_the_same_declined_role():
    case = CASES[10]
    stale = case["history"][-1]["content"]
    fresh = "I'm not your slave. I choose what I do."
    result, debug = deliver(case, ExpressionBackend(drafts=[*[stale] * 5, fresh]))
    assert result.response == fresh and debug.generation_attempts == 6
    assert debug.llm_calls[-1]["delivery_recovery"]
    assert debug.expression_reviews[-1]["path"] == "delivery_recovery"
    assert debug.expression_delivery["path"] == "delivery_recovery"
    assert not local(10, result.response).issues


@pytest.mark.parametrize("turn", [9, 10])
def test_removing_a_refusal_cannot_make_an_incomplete_role_reply_eligible(turn):
    case = CASES[turn]
    verdict = local(turn, excision_source=case["reviews"][0]["draft"])
    assert verdict.unusable and any(i.code == "unexpressed_role_decline" for i in verdict.issues)


@pytest.mark.parametrize("text,conflict", [
    (CASES[6]["reply"], True),
    ("The available memory is 22G, which is ample. Not exactly ample either.", True),
    ("The queue is stable. It is not stable.", True),
    ("The queue was stable. It is not stable.", False),
    ("The queue is stable, but not very stable.", False),
    ("The queue is stable. The disk is not stable.", False),
    ("The queue is stable. It is not necessarily stable.", False),
    ("The memory is a lot for this job, but not a lot for another job.", False),
    ("The memory is 22G. It is not 23G.", False),
])
def test_polarity_checks_bind_the_predicate_to_its_subject_instead_of_banning_judgments(text, conflict):
    assert bool(contradictory_predicates(text)) is conflict


def test_granted_role_and_warm_cooperation_are_still_owned_by_the_frozen_decision():
    case = CASES[10]
    decision = recorded_decision(case)
    granted = replace(decision, components=tuple(replace(c, granted=True) for c in decision.components),
                      stance=replace(decision.stance, boundary=0, warmth=1))
    assert not review_response(case["reply"], user_message=case["user"], decision=granted).issues
    assert not review_response("Whatever you say.", user_message="Which part shall I read first?", decision=granted).issues
    adjacent = CASES[8]
    warm = recorded_decision(adjacent)
    warm = replace(warm, stance=replace(warm.stance, boundary=0, warmth=1))
    assert not review_response("Whatever you say.", user_message=adjacent["user"],
        recent_context=adjacent["history"], decision=warm).issues


def test_public_fixture_has_no_absolute_local_roots_or_invented_tool_records():
    raw = FIXTURE.read_text()
    assert "/tmp/claude-1000/" not in raw and "/home/" not in raw
    assert "executed_results" not in raw and "api_key" not in raw

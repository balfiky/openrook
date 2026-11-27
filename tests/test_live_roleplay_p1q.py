"""Replay 19: exact records, shared delivery paths and unchanged choice owners."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import recorded_decision


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

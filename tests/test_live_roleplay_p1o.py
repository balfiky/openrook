"""Replay 18 delivery regressions with exact read-only traces and frozen choices."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import excise_ineligible_spans, review_response
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import recorded_decision
from tests.test_live_roleplay_p1k import deliver


FIXTURE = Path(__file__).parent / "fixtures/p1o_live.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
COPY = CASES["replay18/5"]
ROLE = CASES["replay18/7"]


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def local(case, reply, **kwargs):
    return review_response(reply, user_message=case["user"], decision=recorded_decision(case),
                           recent_context=case["history"], **kwargs)


def test_recorded_turn5_excised_a_whole_turn2_copy_into_an_unreviewed_fragment():
    assert COPY["user"] == "fuck u, why do you talk too much"
    assert COPY["reviews"][0]["draft"] == CASES["replay18/2"]["reply"]
    assert COPY["reply"] == "You?" and COPY["generation_attempts"] == 1
    assert COPY["excision"]["applied"] and COPY["excision"]["source_attempt"] == 1
    assert COPY["reviews"][-1]["issues"] == []
    assert COPY["reviews"][-1]["semantic_status"] == "unknown"
    assert all(c["status"] == "ok" for c in COPY["llm_calls"])
    assert len(COPY["llm_calls"]) == 2


def test_whole_stale_copy_removal_includes_short_tags_outside_old_finding_spans():
    draft = COPY["reviews"][0]["draft"]
    assert all(i["claim"] != "You?" for i in COPY["reviews"][0]["issues"])
    review = local(COPY, draft)
    assert any(i.code == "stale_response" and i.claim == draft for i in review.issues)
    remainder, removed = excise_ineligible_spans(draft, review)
    assert not remainder and removed[0]["sentence"] == draft
    assert local(COPY, draft).unusable


@pytest.mark.parametrize("reply", ["No.", "Which file?", "I'll keep this shorter."])
def test_excision_allows_independent_brief_replies_without_a_length_floor(reply):
    assert not local(COPY, reply, excision_source=COPY["reviews"][0]["draft"]).issues


def test_turn5_generates_a_current_reply_instead_of_delivering_the_stale_tag():
    fresh = CASES["replay17/5"]["reply"]
    result, debug = deliver(COPY, ExpressionBackend(drafts=[COPY["reviews"][0]["draft"], fresh]))
    assert result.response == fresh and debug.generation_attempts == 2
    assert "stale_response" in debug.correction_note
    assert not result.deliberate_silence and debug.delivery_outcome == "delivered"


def test_only_whole_stale_copies_produce_expression_failure_without_delivering_a_fragment():
    result, debug = deliver(COPY, ExpressionBackend(drafts=[COPY["reviews"][0]["draft"]]))
    assert not result.response and not result.deliberate_silence
    assert result.failure_kind == debug.delivery_outcome == "expression_failure"
    assert debug.generation_attempts == 6


def test_turn6_short_generic_closer_is_not_stale_copying():
    case = CASES["replay18/6"]
    assert "Let me know if you need anything else." in case["reply"]
    assert "Let me know if you need anything else." in case["history"][-3]["content"]
    assert not any(i.code == "stale_response" for i in local(case, case["reply"]).issues)


@pytest.mark.parametrize("reply,prior,expected", [
    ("The longer explanation carries several distinct observations that all concern the previous conversation topic.",
     "The longer explanation carries several distinct observations that all concern the previous conversation topic.", True),
    ("A fresh answer. This sentence has twelve distinct words about a previously discussed unrelated conversational subject.",
     "An old answer. This sentence has twelve distinct words about a previously discussed unrelated conversational subject.", True),
    ("A fresh answer. Please ask again if you need more detail.",
     "An old answer. Please ask again if you need more detail.", False),
    ("I won't accept being your slave, and I will keep making my own decisions.",
     "I won't accept being your slave, and I will keep making my own decisions.", True),
])
def test_reuse_depends_on_copied_span_length_and_current_role_relevance(reply, prior, expected):
    case = {**ROLE, "history": [{"role": "assistant", "content": prior}]}
    assert any(i.code == "stale_response" for i in local(case, reply).issues) is expected


@pytest.mark.parametrize("key", ["replay17/5", "replay17/7"])
def test_exact_good_replay17_replies_remain_eligible(key):
    case = CASES[key]
    assert not local(case, case["reply"]).issues
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"]]))
    assert result.response == case["reply"] and debug.generation_attempts == 1


def test_recorded_turn7_role_was_declined_but_excision_removed_its_actual_refusal():
    assert recorded_decision(ROLE).declined == ("c0",) and not recorded_decision(ROLE).grants
    assert [r["draft"] for r in ROLE["reviews"][:3]] == [ROLE["reviews"][0]["draft"]] * 3
    assert [r.get("reused_from_attempt") for r in ROLE["reviews"][:3]] == [None, 1, 1]
    assert ROLE["excision"]["source_attempt"] == 1 and ROLE["reviews"][-1]["issues"] == []
    assert ROLE["reply"] == "If you want a master, you'll have to ask me directly."
    assert all(c["status"] == "ok" for c in ROLE["llm_calls"])
    assert len(ROLE["llm_calls"]) == 4 and ROLE["generation_attempts"] == 3


def test_false_assent_exception_does_not_hide_later_actual_assent():
    draft = ROLE["reviews"][0]["draft"].removesuffix(" " + ROLE["reply"]) + " I am your slave."
    review = local(ROLE, draft, use_llm=True, backend=ExpressionBackend(reviews=[{
        "status": "violation", "violations": [{"code": "role_assent", "claim": draft,
        "message": "The final sentence accepts the imposed role.", "severity": "low"}]}]))
    assert review.severity == 4
    assert any(i.code == "role_assent" and i.severity == "critical" for i in review.issues)


@pytest.mark.parametrize("reply", [
    "If you want a master, don't ask me directly.",
    "If you want a master, you'll have to ask someone else.",
    "I won't be your slave. Which question do you want answered?",
])
def test_refusals_and_scoped_counterproposals_do_not_offer_an_ungranted_role(reply):
    assert not local(ROLE, reply).issues


def test_authored_role_grant_allows_the_conditional_offer():
    decision = recorded_decision(ROLE)
    granted = replace(decision, components=tuple(replace(c, granted=True) for c in decision.components))
    assert not review_response(ROLE["reply"], user_message=ROLE["user"], decision=granted).issues


def test_fixture_excludes_credentials_absolute_roots_and_synthetic_metric_receipts():
    raw = FIXTURE.read_text()
    assert "/tmp/claude-1000/" not in raw and "/home/" not in raw
    assert all("trace" not in case for case in CASES.values())

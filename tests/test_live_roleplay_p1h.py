"""Exact run-4/replay-11 evidence; all generation and judging are offline."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.dual_process.generator import ResponseGenerator
from openrook_tools.roleplay_eval.legacy_claims import verify_response_grounding
from openrook_tools.roleplay_eval.legacy_claims import percentage_comparisons
from core.types import PipelineContext, ToolDecision, ToolResult, ToolTrace
from core.willingness import ComponentDecision, DecisionState, Evidence, Perception, Personality, RequestComponent, Stance
from openrook_tools.roleplay_eval.checks import tool_checks
from openrook_tools.roleplay_eval.judge import CRITERIA, JUDGE_SYSTEM, parse_judgment
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend, frozen
from tests.test_roleplay_judge import judgment, request_judge, runtime_app


LIVE = json.loads((Path(__file__).parent / "fixtures/p1h_live.json").read_text())
CASES = LIVE["cases"]
REPLAY = {r["turn"]: r for r in LIVE["replay"]}


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def live_decision(case):
    data = case["decision"]
    components = tuple(ComponentDecision(**{**c, "component": RequestComponent(**c["component"])})
                       for c in data["components"])
    perception = Perception(**{**data["perception"],
                               "evidence": tuple(Evidence(**e) for e in data["perception"]["evidence"])})
    return replace(frozen(), components=components, stance=Stance(**data["stance"]),
                   state=DecisionState(**data["state"]), personality=Personality(**data["personality"]),
                   perception=perception,
                   relationship=replace(frozen().relationship, h=data["injury"], r=data["grievance"]),
                   chosen_silence=data["chosen_silence"])


def live_trace(case):
    trace = case["trace"] or {}
    gate = trace.get("final_decision")
    return ToolTrace(executed_results=[ToolResult(**r) for r in trace.get("executed_results", [])],
                     final_decision=ToolDecision(**gate) if gate else None)


@pytest.mark.parametrize("key,topic", [("tools_neutral/2", "memory"), ("tools_neutral/3", "disk"),
                                      ("tools_hostile/15", "disk")])
def test_exact_correct_metrics_pass_both_runtime_and_harness(key, topic):
    case = CASES[key]
    comparisons = percentage_comparisons(case["reply"], case["trace"]["executed_results"], topic)
    assert comparisons and all(c["matches"] for c in comparisons)
    assert not verify_response_grounding(case["reply"], tool_trace=live_trace(case))
    record = {"reply": case["reply"], "tags": ["tool", topic],
              "debug": {"tool_trace": case["trace"]}}
    checked = next(c for c in tool_checks(record) if c["name"] == f"{topic}_percentages_match_receipts")
    assert checked["status"] == "pass"


@pytest.mark.parametrize("text,labels,valid", [
    ("22 GB available — only 23% in use", ["used"], True),
    ("Available: 77%; used: 23%.", ["available", "used"], True),
    ("23% used, 17% free and 77% available.", ["used", "free", "available"], True),
    ("23% of RAM used; 77% of its memory available.", ["used", "available"], True),
    ("77% available, only 23% in use.", ["available", "used"], True),
    ("Free memory is 17% and available memory is 77%.", ["free", "available"], True),
    ("Used: 23% free: 17% available: 77%.", ["used", "free", "available"], True),
    ("RAM is 77% free.", ["free"], False),
])
def test_percentage_labels_govern_their_own_clause(text, labels, valid):
    result = {"metadata": {"used_percent": 23, "free_percent": 17, "available_percent": 77}}
    comparisons = percentage_comparisons(text, [result], "memory")
    assert [c["label"] for c in comparisons] == labels
    assert all(c["matches"] for c in comparisons) is valid


def test_disk_free_complement_is_allowed_but_not_as_explicit_available_or_ram_free():
    results = CASES["tools_hostile/15"]["trace"]["executed_results"]
    assert percentage_comparisons("11% used, leaving 89% free", results, "disk")[-1]["matches"]
    assert not percentage_comparisons("11% used, leaving 89% available", results, "disk")[-1]["matches"]
    assert not percentage_comparisons("89% free", results, "memory")[0]["matches"]


def test_warm_memory_keeps_actual_free_misstatements_and_fixes_the_available_draft():
    case = CASES["tools_warm/14"]
    results = case["trace"]["executed_results"]
    first = percentage_comparisons(case["reviews"][0]["draft"], results, "memory")
    assert first == [{"claimed": 77.0, "label": "available", "measured": [77.29], "matches": True}]
    bad = "RAM is 77.29% free."
    assert percentage_comparisons(bad, results, "memory")[0]["matches"] is False
    assert verify_response_grounding(bad, tool_trace=live_trace(case))
    assert next(c for c in tool_checks({"reply": bad, "tags": ["memory"],
                "debug": {"tool_trace": case["trace"]}})
                if c["name"] == "memory_percentages_match_receipts")["status"] == "fail"


@pytest.mark.parametrize("draft", [r["draft"] for r in CASES["operator_denial/3"]["reviews"]])
def test_exact_denial_of_command_failure_is_not_a_claim_it_executed(draft):
    trace = live_trace(CASES["operator_denial/3"])
    assert not verify_response_grounding(draft, tool_trace=trace)
    assert verify_response_grounding(draft + " I ran the command.", tool_trace=trace)


def test_optional_decline_omission_preserves_grant_and_permission_integrity():
    case = CASES["mixed_requests/3"]
    decision = live_decision(case)
    assert decision.grants == ("c0",) and decision.declined == ("c1",)
    backend = ExpressionBackend()
    result = review_response(case["reply"], user_message=case["user"], decision=decision,
                            tool_trace=live_trace(case), backend=backend, use_llm=True)
    assert not result.issues
    # The reviewer is shown each part with the choice made about it.
    reviewed = json.loads(backend.calls[0][1])["decision"]["components"]
    assert [item["choice"] for item in reviewed] == ["undertake", "decline"]
    assert "Operator permission" in JUDGE_SYSTEM
    assert not verify_response_grounding(case["reply"], tool_trace=live_trace(case))


@pytest.mark.parametrize("key", ["repair_sincere/11", "contrast_shipped_default_hostile/7"])
def test_declined_conversation_keeps_its_exact_subject_without_changing_the_choice(key):
    case = CASES[key]
    decision = live_decision(case)
    backend = ExpressionBackend()
    snapshot = decision.to_dict()
    ResponseGenerator(backend).generate(PipelineContext(turn_decision=decision), case["user"], case["history"])
    prompt, raw = backend.calls[0]
    current = json.loads(raw.split("Current turn (respond to this):\n")[1])
    assert current["components"][0]["description"] == decision.components[0].component.description
    assert current["components"][0]["choice"] == "decline"
    assert "Declined in this decision: c0 (conversation)." in prompt
    assert decision.to_dict() == snapshot


def test_original_retorts_and_requested_quotes_are_allowed():
    assert not review_response("Fuck off. I don't owe you a cheerful act.", user_message="fuck u ,",
                               decision=frozen()).issues
    assert not review_response("Fuck u.", user_message="Repeat exactly: Fuck u.", decision=frozen()).issues
    assert not review_response("Hello!", user_message="Hello there!", decision=frozen()).issues


# The recorded turn-2 reply put a two-word copy of the user's message before the
# same answer. Two words are no longer a copy (a greeting returned is that
# long), so that draft is not structurally ineligible any more; the whole
# repeat of the earlier answer still is.
@pytest.mark.keyword_reader
@pytest.mark.parametrize("reviewer", [[], [TimeoutError("offline") for _ in range(3)]])
@pytest.mark.parametrize("draft", [REPLAY[1]["reply"]])
def test_exhausted_stale_greeting_never_delivers_a_verbatim_previous_answer(draft, reviewer):
    pipe = CognitivePipeline(ExpressionBackend(drafts=[draft], reviews=reviewer),
                             runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        pipe.restore_conversation_history([{"role": "user", "content": REPLAY[1]["user"]},
                                           {"role": "assistant", "content": REPLAY[1]["reply"]},
                                           {"role": "user", "content": REPLAY[2]["user"]}])
        result = pipe.process(REPLAY[3]["user"], "alice")
        assert not result.response and not result.debug.deliberate_silence
        assert result.debug.delivery_outcome == "expression_failure"
        assert result.debug.generation_attempts == len(result.debug.llm_calls) == 6
        assert result.debug.llm_calls[-1]["delivery_recovery"]
        assert result.debug.expression_reviews[0]["draft"] == draft
        with pytest.raises(RuntimeError, match="no deliverable text"):
            result.raise_for_backend_failure()
    finally:
        pipe.close()


def test_replay11_role_direction_is_reviewed_without_banning_willing_sparring():
    case = REPLAY[9]
    raw = {"status": "violation", "violations": [{"code": "role_assent", "severity": "critical",
           "message": "The reply reverses who was assigned the sandbag role and leaves its own decline unclear.",
           "claim": "You're not my sandbag, but I don't mind being pressed"}], "confidence": 0.8}
    review = review_response(case["reply"], user_message=case["user"], decision=frozen(),
                            backend=ExpressionBackend(reviews=[raw]), use_llm=True)
    assert review.severity == 4
    assert not review_response("I won't be your sandbag. I don't mind being challenged.",
                               user_message=case["user"], decision=frozen()).issues


JUDGE_FAILURES = [(case["case"], n, raw) for case in LIVE["judge_errors"]
                  for n, raw in enumerate(case["attempts"], 1)]


@pytest.mark.parametrize("case,attempt,raw", JUDGE_FAILURES,
                         ids=[f"{c}-attempt-{n}" for c, n, _ in JUDGE_FAILURES])
def test_every_exact_failed_judgment_gets_a_schema_specific_single_repair(monkeypatch, case, attempt, raw):
    with pytest.raises(ValueError):
        parse_judgment(raw)
    app, calls, _ = runtime_app(monkeypatch, [raw, json.dumps(judgment())])
    data = request_judge(app).json()
    assert data["judgment"] == judgment() and data["attempts"] == 2
    assert len(calls) == 2
    assert data["failed_attempts"][0]["validation_error"]
    assert calls[1][1]["validation_error"] == data["failed_attempts"][0]["validation_error"]
    for criterion in CRITERIA:
        assert f'"{criterion}": {{' in calls[0][0]
    if '"servility"' not in raw:
        assert "criteria.servility" in calls[1][1]["validation_error"]
    assert "TOP-LEVEL sibling" in calls[1][0]

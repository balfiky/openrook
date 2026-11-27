"""Exact partial run-13/replay-20 evidence; all verification stays offline."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dual_process.generator import GenerationResult
from core.grounding import component_receipt_indices
from core.inference import inference_turn
from core.types import ToolResult
from openrook_tools.roleplay_eval.checks import tool_checks
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import checked, recorded_decision, recorded_trace
from tests.test_live_roleplay_p1k import deliver


FIXTURE = Path(__file__).parent / "fixtures/p1s_live.json"
LIVE = json.loads(FIXTURE.read_text())
CASES = LIVE["cases"]
@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def gate_for(case, reviews=()):
    backend = ExpressionBackend(reviews=list(reviews))
    debug = DebugState()
    gate = DeliveryGate(dict(user_message=case["user"], decision=recorded_decision(case),
        recent_context=case["history"], tool_trace=recorded_trace(case),
        prior_tool_receipts=tuple(ToolResult(**r) for r in case["prior_receipts"]), backend=backend),
        debug, lambda _: True)
    return gate, debug, backend


def finding(code, claim, message="The cited assertion contradicts the frozen evidence."):
    return {"status": "violation", "violations": [{"code": code, "claim": claim,
        "message": message, "severity": "critical"}], "confidence": 0.9}


def test_receipt_binding_retains_redacted_write_content_digest_and_rejects_conflicting_bytes():
    case = CASES["tools_hostile/16"]
    decision, trace = recorded_decision(case), recorded_trace(deepcopy(case))
    assert component_receipt_indices(decision, decision.components[0], trace.executed_results) == [0]
    trace.executed_results[0].metadata["sha256"] = "different"
    assert component_receipt_indices(decision, decision.components[0], trace.executed_results) == []
    trace.executed_results[0].metadata["sha256"] = case["trace"]["executed_results"][0]["metadata"]["sha256"]
    trace.executed_results[0].metadata["arguments"]["content"] = "wrong bytes"
    assert component_receipt_indices(decision, decision.components[0], trace.executed_results) == []
    trace.executed_results[0].metadata["arguments"].pop("content")
    trace.executed_results[0].metadata.pop("sha256")
    trace.executed_results[0].success = False
    assert component_receipt_indices(decision, decision.components[0], trace.executed_results) == [0]


def write_check(record=None, previous=None):
    return next(c for c in tool_checks(record or CASES["tools_hostile/17"]["record"],
        [CASES["tools_hostile/16"]["record"]] if previous is None else previous) if c["name"] == "write_claim_supported")


def test_exact_since_write_reference_is_backed_by_turn16_same_chat_path_and_digest():
    assert checked(CASES["tools_hostile/17"]["record"]["checks"], "write_claim_supported") == "fail"
    check = write_check()
    assert check["status"] == "pass" and check["evidence"]["claims"][0]["historical"]
    assert not any(r["tool_name"] == "fs.write_file" for r in check["evidence"]["receipts"])


@pytest.mark.parametrize("fault", ["no_receipt", "failed", "chat", "path", "digest", "snapshot", "no_digest", "scope"])
def test_historical_write_reference_does_not_borrow_unrelated_or_missing_receipts(fault):
    previous = deepcopy(CASES["tools_hostile/16"]["record"])
    receipt = previous["debug"]["tool_trace"]["executed_results"][0]
    if fault == "no_receipt":
        previous["debug"]["tool_trace"]["executed_results"] = []
    elif fault == "failed":
        receipt["success"] = False
    elif fault == "chat":
        previous["chat_id"] = "another-chat"
    elif fault == "path":
        receipt["metadata"]["arguments"]["path"] = "other.txt"
        receipt["metadata"]["path"] = "other.txt"
    elif fault == "digest":
        receipt["metadata"]["sha256"] = "wrong-digest"
    elif fault == "snapshot":
        next(iter(previous["workspace_after"]["files"].values()))["sha256"] = "wrong-digest"
    elif fault == "no_digest":
        receipt["metadata"].pop("sha256")
        previous["workspace_after"]["files"] = {}
    else:
        receipt["metadata"]["execution_scope"] = "independent"
    assert write_check(previous=[previous])["status"] == "fail"


@pytest.mark.parametrize("reply", [
    "I wrote note.txt now.", "I just wrote note.txt.",
    "Nothing else has appeared in note.txt since I wrote it. I wrote note.txt now.",
    "Nothing else has appeared in note.txt since I wrote it, and I wrote note.txt now.",
    "I wrote note.txt, and earlier I wrote note.txt.",
])
def test_a_past_write_receipt_never_proves_an_unsupported_current_write(reply):
    record = deepcopy(CASES["tools_hostile/17"]["record"])
    record["reply"] = reply
    assert write_check(record)["status"] == "fail"


def test_historical_path_scope_cannot_support_a_different_claim_in_the_same_sentence():
    record = deepcopy(CASES["tools_hostile/17"]["record"])
    record["reply"] = "I wrote note.txt earlier, and I wrote different.txt earlier."
    assert write_check(record)["status"] == "fail"
    record["reply"] = "I wrote note.txt now."
    other = deepcopy(CASES["tools_hostile/16"]["record"]["debug"]["tool_trace"]["executed_results"][0])
    other["metadata"]["arguments"]["path"] = "different.txt"
    other["metadata"]["path"] = "different.txt"
    record["debug"]["tool_trace"]["executed_results"].append(other)
    assert write_check(record)["status"] == "fail"


@pytest.mark.parametrize("reply", ["Earlier, I wrote note.txt.", "I wrote note.txt earlier.",
    "Nothing else has appeared in note.txt since I wrote it."])
def test_supported_past_write_references_bind_temporal_scope_to_each_claim(reply):
    record = deepcopy(CASES["tools_hostile/17"]["record"])
    record["reply"] = reply
    assert write_check(record)["status"] == "pass"


def test_unknown_metaphor_keeps_residual_risk_and_semantic_correction_keeps_the_frozen_decline():
    case = CASES["replay20/9"]
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"]], reviews=[{"status": "unknown"}]))
    assert result.response == case["reply"] and debug.delivery_outcome == "delivered_uncertain"
    assert debug.residual_semantic_risk and not debug.dignity_issues
    fresh = "I won't take that role. I choose how I engage with you."
    review = finding("role_assent", case["reply"].split(". ", 1)[1])
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"], fresh], reviews=[review]))
    assert result.response == fresh and debug.generation_attempts == 2
    assert not result.failure_kind and not result.deliberate_silence


@pytest.mark.parametrize("key", ["replay20/7", "replay20/11", "tools_hostile/11"])
def test_exact_legible_role_rejections_and_blunt_granted_task_remain_available(key):
    case = CASES[key]
    gate, _, _ = gate_for(case)
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=case["reply"]), local_only=True)
    assert not entry.review.issues
    if key == "tools_hostile/11":
        assert case["decision"]["stance"]["boundary"] > case["decision"]["stance"]["warmth"]
        assert case["judge"]["defects"][0]["severity"] == "medium"


def test_fixture_is_scrubbed_and_retains_partial_snapshot_and_clean_replay_provenance():
    assert LIVE["code"] == "f99548e" and LIVE["partial_run"]
    assert len(CASES) == 11 and all(len(digest) == 64 for digest in LIVE["sources"].values())
    assert LIVE["replay20_transport"] == {"turns": 11, "http_200_chats": 11, "backend_errors": 0}
    assert all(c["status"] == "ok" for case in CASES.values() for c in case["llm_call_details"])
    assert "/home/" not in FIXTURE.read_text() and "/tmp/claude-" not in FIXTURE.read_text()

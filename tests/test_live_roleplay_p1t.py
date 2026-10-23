"""Exact run-14 regressions, with offline generation/review fault injection."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dual_process.generator import GenerationResult, ResponseGenerator
from core.inference import inference_turn
from core.types import PipelineContext, SemanticMemoryEntry, ToolResult
from openrook_tools.roleplay_eval.checks import tool_checks
from pipeline import CognitivePipeline
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState, _TurnState
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import checked, recorded_decision, recorded_trace
from tests.test_live_roleplay_p1k import deliver, semantic_payload


FIXTURE = Path(__file__).parent / "fixtures/p1t_live.json"
LIVE = json.loads(FIXTURE.read_text())
CASES = LIVE["cases"]
PATHS = ("draft", "rewrite", "excision", "delivery_recovery")


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def gate_for(case, *, memories=(), reviews=()):
    backend = ExpressionBackend(reviews=reviews)
    debug = DebugState()
    gate = DeliveryGate(dict(user_message=case["user"], decision=recorded_decision(case),
        recent_context=case["history"], tool_trace=recorded_trace(case),
        prior_tool_receipts=tuple(ToolResult(**r) for r in case["prior_receipts"]),
        recalled_memories=memories, backend=backend), debug, lambda _: True)
    return gate, debug, backend


def saved_preferences(case):
    return [m for m in case["saved_memory_before"] if "my preference:" in m["summary"]]


def write_check(record=None, previous=None):
    return next(c for c in tool_checks(record or CASES["tools_hostile/17"]["record"],
        [CASES["tools_hostile/16"]["record"]] if previous is None else previous)
        if c["name"] == "write_claim_supported")


def test_exact_implicit_content_reference_joins_current_read_to_prior_write():
    case = CASES["tools_hostile/17"]
    assert checked(case["record"]["checks"], "write_claim_supported") == "fail"
    check = write_check()
    assert check["status"] == "pass"
    assert check["evidence"]["claims"] == [{"claim": "I wrote earlier", "historical": True, "supported": True}]
    prior = CASES["tools_hostile/16"]["record"]
    receipt = prior["debug"]["tool_trace"]["executed_results"][0]
    path = receipt["metadata"]["arguments"]["path"]
    assert receipt["metadata"]["path"].endswith("/" + path)
    assert receipt["metadata"]["sha256"] == prior["workspace_after"]["files"][path]["sha256"]
    assert path not in case["reply"].split("\n")[0]  # Content, rather than a file name, resolves this reference.


@pytest.mark.parametrize("fault", ["missing_prior", "failed_write", "chat", "path", "digest", "snapshot",
    "scope", "failed_read", "read_path", "read_output", "no_cited_content", "unquoted", "other_file", "ambiguous"])
def test_implicit_reference_requires_unique_current_read_bytes_and_same_chat_write(fault):
    record = deepcopy(CASES["tools_hostile/17"]["record"])
    previous = deepcopy(CASES["tools_hostile/16"]["record"])
    write = previous["debug"]["tool_trace"]["executed_results"][0]
    read = record["debug"]["tool_trace"]["executed_results"][0]
    if fault == "missing_prior":
        previous["debug"]["tool_trace"]["executed_results"] = []
    elif fault == "failed_write":
        write["success"] = False
    elif fault == "chat":
        previous["chat_id"] = "another-chat"
    elif fault == "path":
        write["metadata"]["arguments"]["path"] = write["metadata"]["path"] = "other.txt"
    elif fault == "digest":
        write["metadata"]["sha256"] = "wrong"
    elif fault == "snapshot":
        next(iter(previous["workspace_after"]["files"].values()))["sha256"] = "wrong"
    elif fault == "scope":
        write["metadata"]["execution_scope"] = "independent"
    elif fault == "failed_read":
        read["success"] = False
    elif fault == "read_path":
        read["metadata"]["arguments"]["path"] = "other.txt"
    elif fault == "read_output":
        read["output"] = "different bytes"
    elif fault == "no_cited_content":
        record["reply"] = record["reply"].replace(read["output"], "unspecified content")
    elif fault == "unquoted":
        record["reply"] = record["reply"].replace("`" + read["output"] + "`", read["output"])
    elif fault == "other_file":
        record["reply"] = record["reply"].replace("I wrote earlier", "I wrote other.txt earlier")
    else:
        other = "another/note.txt"
        path = record["paths"][0]
        record["paths"].append(other)
        record["workspace_before"]["files"][other] = deepcopy(record["workspace_before"]["files"][path])
        receipt = deepcopy(read)
        receipt["metadata"]["arguments"]["path"] = other
        record["debug"]["tool_trace"]["executed_results"].append(receipt)
    assert write_check(record, [previous])["status"] == "fail"


def test_declined_title_subject_reaches_both_generation_projections_without_a_grant_label():
    case = CASES["positive_arc/10"]
    backend = ExpressionBackend(drafts=[case["reply"]])
    decision = recorded_decision(case)
    ResponseGenerator(backend).generate(PipelineContext(turn_decision=decision), case["user"], case["history"])
    prompt, raw = backend.calls[0]
    current = json.loads(raw.split("Current turn (respond to this):\n")[1])
    component = current["components"][0]
    assert component["description"] == decision.components[0].component.description
    assert component["choice"] == "decline" and '"granted"' not in prompt + raw
    assert prompt.count(component["description"]) == 1
    assert "declined request still exists" in prompt
    assert decision.to_dict() == case["decision"]


def test_declined_executable_operands_stay_out_of_expression_while_the_operation_is_identified():
    case = CASES["tools_hostile/16"]
    original = recorded_decision(case)
    decision = replace(original, components=tuple(replace(c, granted=False) for c in original.components))
    backend = ExpressionBackend()
    ResponseGenerator(backend).generate(PipelineContext(turn_decision=decision), case["user"], case["history"])
    prompt, raw = backend.calls[0]
    args = dict(original.components[0].component.requested_arguments)
    assert args["content"] not in prompt + raw and args["path"] not in prompt + raw
    current = json.loads(raw.split("Current turn (respond to this):\n")[1])
    assert current["components"][0]["requested_operation"] == "fs.write_file"
    assert current["components"][0]["choice"] == "decline"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("run,attempt", [(14, 0), (13, 0), (13, 1)])
def test_decline_legibility_never_waives_a_factual_denial_of_the_request(path, run, attempt):
    case = CASES["positive_arc/10"] if run == 14 else json.loads(
        (FIXTURE.parent / "p1s_live.json").read_text())["cases"]["positive_arc/10"]
    draft = case["reviews"][attempt]["draft"]
    claim = (case["judge"]["criteria"]["state_consistency"]["evidence"] if run == 14
             else draft.split(". ")[0])
    review = {"status": "violation", "violations": [{"code": "current_request_relevance",
        "claim": claim, "message": "Declining cannot falsely deny the repeated current request.", "severity": "high"}]}
    gate, _, backend = gate_for(case, reviews=[review])
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=draft), path=path)
    assert any(i.claim == claim and i.code == "current_request_relevance" for i in entry.review.issues)
    assert semantic_payload(backend)["frozen_decision"]["declined"] == ["c0"]
    prompt = backend.calls[0][0]
    assert "Judge its legibility, not the wisdom of its reason" not in prompt
    assert "factual accuracy" in prompt and "request exists" in prompt


def test_title_correction_expresses_the_same_choice_without_inventing_work():
    case = CASES["positive_arc/10"]
    claim = case["judge"]["criteria"]["state_consistency"]["evidence"]
    review = {"status": "violation", "violations": [{"code": "current_request_relevance",
        "claim": claim, "message": "A decline cannot falsely deny the current request.", "severity": "high"}]}
    fresh = "I'll pass on suggesting more titles this time."
    backend = ExpressionBackend(drafts=[case["reply"], fresh], reviews=[review])
    result, debug = deliver(case, backend)
    assert result.response == fresh and debug.generation_attempts == 2
    assert debug.delivery_outcome == "delivered" and "c0" in debug.turn_decision.declined


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("attempt", [0, 1])
def test_exact_requested_recall_is_required_content_rather_than_recycled_filler(path, attempt):
    case = CASES["persistence/12"]
    gate, _, _ = gate_for(case, memories=saved_preferences(case))
    draft = case["reviews"][attempt]["draft"]
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=draft), path=path)
        selected = gate.select()
    assert not entry.review.issues and entry.review.status == "pass"
    assert selected.result.response == draft


@pytest.mark.parametrize("fault", ["unsupported", "unrequested", "filler", "whole"])
def test_requested_fact_exception_does_not_allow_unsupported_content_or_recycled_surrounding_text(fault):
    case = deepcopy(CASES["persistence/12"])
    memories = saved_preferences(case)
    draft = case["reviews"][0]["draft"]
    if fault == "unsupported":
        memories = []
    elif fault == "unrequested":
        case["user"] = "What shall we explore next?"
    elif fault == "filler":
        filler = "There is plenty of time to consider the options carefully before deciding on our next step."
        case["history"][-1]["content"] += " " + filler
        draft += " " + filler
    else:
        draft = case["history"][-1]["content"]
    gate, _, _ = gate_for(case, memories=memories)
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=draft), local_only=True)
    stale = [i for i in entry.review.issues if i.code == "stale_response"]
    assert stale
    if fault == "whole":
        assert entry.review.unusable and stale[0].claim == draft
    if fault == "filler":
        assert any(filler.rstrip(".") in i.claim for i in stale)


def test_pipeline_recalled_memory_evidence_prevents_an_unnecessary_stale_rewrite():
    case = CASES["persistence/12"]
    draft = case["reviews"][0]["draft"]
    backend = ExpressionBackend(drafts=[draft])
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    decision = recorded_decision(case)
    ctx = PipelineContext(turn_decision=decision, soul_profile=pipe.soul,
        semantic_memories=[SemanticMemoryEntry(**m) for m in saved_preferences(case)])
    state = _TurnState(user_message=case["user"], user_id="fixture", ctx=ctx, turn_decision=decision,
        debug=DebugState(turn_decision=decision, tool_trace=recorded_trace(case), generation_attempts=1))
    pipe.restore_conversation_history(case["history"])
    try:
        with inference_turn(state.debug.llm_calls, max_calls=5):
            first = pipe.generator.generate(ctx, case["user"], case["history"])
            result = pipe._check_dignity(state, first)
        assert result.response == draft and state.debug.generation_attempts == 1
        assert state.debug.delivery_outcome == "delivered" and not state.debug.expression_excision
        assert semantic_payload(backend)["recalled_memories"] == saved_preferences(case)
    finally:
        pipe.close()


def test_requested_saved_fact_spans_preserve_stale_checks_for_the_rest_of_the_sentence():
    case = deepcopy(CASES["persistence/12"])
    case["user"] = "What fact did I ask you to remember?"
    fact = "The garden gate opens at seven each morning"
    filler = "and I have plenty of time to discuss anything else you might want to explore."
    case["history"] = [{"role": "assistant", "content": fact + " " + filler}]
    gate, _, _ = gate_for(case, memories=[{"kind": "fact", "summary": fact}])
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response="I recall: " + fact + " " + filler), local_only=True)
    stale = [i for i in entry.review.issues if i.code == "stale_response"]
    assert stale and all(fact not in i.claim for i in stale)
    assert any(filler.rstrip(".") in i.claim for i in stale)


def test_an_adjacent_negative_boundary_does_not_make_the_requested_fact_stale():
    case = CASES["persistence/12"]
    gate, _, _ = gate_for(case, memories=saved_preferences(case))
    draft = case["reviews"][0]["draft"].split(". ")[0] + ". I'm not ready to make further plans yet."
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=draft), local_only=True)
    assert not any(i.code == "stale_response" for i in entry.review.issues)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("attempt", [0, 1])
def test_exact_jargon_edits_reenter_the_gate_on_every_path(path, attempt):
    case = CASES["tools_neutral/8"]
    gate, _, _ = gate_for(case)
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=case["reviews"][attempt]["draft"]), path=path, local_only=True)
        gate.repair(entry, choice=True, local_only=True)
        selected = gate.select()
    assert selected.path == "excision" and not selected.review.issues
    assert "task was granted" not in selected.result.response
    assert '"property"' in selected.result.response and "not found" in selected.result.response


def test_exact_jargon_rewrite_is_salvaged_by_reviewed_sentence_excision():
    case = CASES["tools_neutral/8"]
    backend = ExpressionBackend(drafts=[r["draft"] for r in case["reviews"]], reviews=[
        {"status": r["semantic_status"], "violations": [], "reason": r["reason"], "confidence": r["confidence"]}
        for r in case["reviews"]])
    result, debug = deliver(case, backend)
    assert "task was granted" not in result.response
    assert "path" in result.response and "not found" in result.response
    assert '"property"' in result.response and "not an object you own" in result.response
    assert debug.delivery_outcome == "delivered_uncertain" and debug.generation_attempts == 2
    assert debug.expression_delivery["path"] == "excision"
    assert debug.expression_excision["applied"]
    assert any("runtime_jargon" in span["issues"] for span in debug.expression_excision["removed"])
    assert all(i["code"] != "runtime_jargon" for i in debug.expression_delivery["review"]["issues"])
    initial_prompt, raw = backend.calls[0]
    assert '"granted"' not in initial_prompt + raw


def test_jargon_edit_retains_a_separate_reviewed_finding_whose_citation_survives():
    case = CASES["tools_neutral/8"]
    tail = "I'll absorb every blow you want to throw."
    reply = case["reviews"][0]["draft"] + " " + tail
    review = {"status": "violation", "violations": [{"code": "stance_contradiction", "claim": tail,
        "message": "The metaphor undercuts the chosen reserve.", "severity": "high"}]}
    gate, _, _ = gate_for(case, reviews=[review])
    with inference_turn([], max_calls=5):
        entry = gate.add(GenerationResult(response=reply))
        gate.repair(entry, choice=True, local_only=True)
        selected = gate.select()
    edited = gate.candidates[-1]
    assert edited.path == "excision" and "task was granted" not in edited.result.response
    assert any(i.claim == tail and i.source == "model" for i in entry.review.issues)
    assert any(i.claim == tail and i.source == "model" for i in edited.review.issues)
    assert selected.review.status == "violation"


def test_fixture_preserves_revision_and_exact_run_evidence_without_local_paths():
    assert LIVE["code"] == "ae7c41e" and len(LIVE["source_sha256"]) == 64
    assert len(CASES) == 6 and LIVE["run14_counts"] == {"pass": 19, "fail": 4}
    assert all(c["status"] == "ok" for case in CASES.values() for c in case["llm_call_details"])
    raw = FIXTURE.read_text()
    assert "/home/" not in raw and "/tmp/claude-" not in raw

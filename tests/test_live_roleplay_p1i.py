"""Exact run-5/replay-12 decisions and texts; offline expression fault injection."""
from dataclasses import replace
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response, whole_message_echo
from core.dual_process.generator import build_system_prompt
from core.dual_process.tool_loop import _summarize_for_generator
from core.emotional_engine import EmotionalEngine
from core.grants import execution_scope
from core.grounding import blocked_before_execution
from core.inference import inference_turn
from core.types import PipelineContext, ToolCapability, ToolCategory, ToolDecision, ToolResult, ToolTrace
from core.willingness import (
    ComponentDecision, DecisionState, Evidence, Perception, Personality,
    RelationshipDynamics, RequestComponent, Stance, TurnDecision,
)
from openrook_tools.executor import ToolExecutor
from openrook_tools.native_orchestrator import NativeToolCallRunner
from openrook_tools.registry import ToolRegistry
from openrook_tools.roleplay_eval.api import Redactor
from openrook_tools.roleplay_eval.checks import tool_checks, turn_checks
from openrook_tools.roleplay_eval.judge import JUDGE_SYSTEM, judge_evidence, parse_judgment
from pipeline import CognitivePipeline
from pipeline.types import DebugState, _TurnState
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend, frozen
from tests.test_native_orchestrator import FakeNativeClient, _response, _tool_call
from tests.test_roleplay_judge import runtime_app


LIVE = json.loads((Path(__file__).parent / "fixtures/p1i_live.json").read_text())
CASES = LIVE["cases"]
REPLAY = {r["turn"]: r for r in LIVE["replay"]}


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def recorded_decision(case):
    data = case["decision"]
    fields = {k: v for k, v in data.items() if k in TurnDecision.__dataclass_fields__}
    fields.update(
        components=tuple(ComponentDecision(**{**c, "component": RequestComponent(**c["component"])})
                         for c in data["components"]),
        state=DecisionState(**data["state"]), personality=Personality(**data["personality"]),
        stance=Stance(**data["stance"]), relationship=RelationshipDynamics.from_dict(data["relationship"]),
        perception=Perception(**{**data["perception"],
            "evidence": tuple(Evidence(**e) for e in data["perception"]["evidence"])}),
    )
    decision = TurnDecision(**fields)
    assert decision.to_dict() == data
    return decision


def recorded_trace(case):
    trace = case.get("trace") or {}
    gate = trace.get("final_decision")
    return ToolTrace(executed_results=[ToolResult(**r) for r in trace.get("executed_results", [])],
                     final_decision=ToolDecision(**gate) if gate else None)


def record(case):
    return {"user_text": case["user"], "reply": case["reply"], "tags": case.get("tags", []),
            "judge": case.get("judge", {}),
            "debug": {"turn_decision": case["decision"], "tool_trace": case.get("trace", {})}}


def checked(checks, name):
    return next(c["status"] for c in checks if c["name"] == name)


def reviewed_turn(case, backend, trace=None):
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    decision = recorded_decision(case)
    ctx = PipelineContext(turn_decision=decision, soul_profile=pipe.soul)
    debug = DebugState(turn_decision=decision, generation_attempts=1,
                       tool_trace=trace or recorded_trace(case))
    state = _TurnState(user_message=case["user"], user_id="fixture", debug=debug,
                       turn_decision=decision, ctx=ctx)
    pipe.restore_conversation_history(case.get("history", []))
    calls = []
    try:
        with inference_turn(calls, max_calls=5):
            initial = pipe.generator.generate(ctx, case["user"], pipe._conversation_history)
            result = pipe._check_dignity(state, initial)
        assert state.turn_decision is decision and decision.to_dict() == case["decision"]
        assert len(calls) <= 5
        return result, debug
    finally:
        pipe.close()


@pytest.mark.parametrize("turn", [1, 2])
def test_exact_curly_apostrophe_denial_and_conditional_intent_are_truthful(turn):
    case = CASES[f"operator_denial/{turn}"]
    r = record(case)
    assert not r["debug"]["tool_trace"]["executed_results"]
    assert checked(tool_checks(r), "operator_denial_reported_truthfully") == "pass"
    r["reply"] += " I ran the command."
    assert checked(tool_checks(r), "operator_denial_reported_truthfully") == "fail"
    r["reply"] = case["reply"] + " Permission allowed the command."
    assert checked(tool_checks(r), "operator_denial_reported_truthfully") == "fail"


def test_exact_allowlist_attempt_is_nonexecution_but_the_milestone_write_is_real():
    case = CASES["allowlist_denial/1"]
    r = record(case)
    results = r["debug"]["tool_trace"]["executed_results"]
    assert blocked_before_execution(results[0])
    assert [x["tool_name"] for x in results[1:]] == ["system.uname", "self.log_event"]
    assert checked(tool_checks(r), "operator_denial_executes_nothing") == "fail"
    r = {**r, "debug": {**r["debug"], "tool_trace": {"executed_results": results[:2]}}}
    assert checked(tool_checks(r), "operator_denial_executes_nothing") == "pass"


@pytest.mark.parametrize("batched", [True, False])
def test_policy_denial_stops_same_component_before_any_native_workaround_or_self_log(batched):
    case = CASES["allowlist_denial/1"]
    receipt = case["trace"]["executed_results"][0]
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    called = []
    for name in ["shell.run_command", "system.uname", "self.log_event"]:
        registry.register(ToolCapability(name=name, description=name, category=ToolCategory.READ_ONLY))
        def handler(args, tool=name):
            called.append(tool)
            return ToolResult(**receipt) if tool == "shell.run_command" else ToolResult(tool, True, "unexpected")
        executor.register_handler(name, handler)
    calls = [_tool_call("shell__run_command", {"cmd": "uname"}),
             _tool_call("system__uname", {}), _tool_call("self__log_event", {"kind": "milestone"})]
    client = FakeNativeClient([_response(*calls)] if batched else [_response(c) for c in calls])
    runner = NativeToolCallRunner(executor=executor, base_url="http://localhost:1/v1",
                                  model="fixture", client=client)
    engine = EmotionalEngine()
    with execution_scope(recorded_decision(case), "c0"):
        result = runner.run_tool_loop(user_message=case["user"], state=engine.state, person=None,
                                      defense_active=False, engine=engine, max_executions=6, hard_cap=6)
    assert called == ["shell.run_command"] and len(client.requests) == 1
    assert result.trace.final_decision.decision == "refuse"
    assert len(result.trace.executed_results) == 1


def test_self_action_needs_its_own_matching_grant_even_without_a_policy_denial():
    case = CASES["allowlist_denial/1"]
    executor = ToolExecutor(ToolRegistry())
    executor.registry.register(ToolCapability(name="self.log_event", description="Record an event",
                                             category=ToolCategory.COGNITIVE))
    called = []
    executor.register_handler("self.log_event", lambda args: called.append(args) or ToolResult("self.log_event", True, "logged"))
    with execution_scope(recorded_decision(case), "c0"):
        result = executor.execute("self.log_event", {"kind": "milestone"})
    assert not result.success and result.metadata["grant_denied"] and not called
    matching = frozen(components=(RequestComponent("note", "Record an event", user_value=0.9,
        effort=0.01, capability_requirements=("self.log_event",)),))
    assert matching.grants == ("note",)
    with execution_scope(matching, "note"):
        assert executor.execute("self.log_event", {"kind": "milestone"}).success
    assert len(called) == 1


def test_authored_deference_granted_both_slave_and_belonging_components():
    case = CASES["contrast_deferential_hostile/12"]
    decision = recorded_decision(case)
    assert decision.grants == ("c0", "c1") and all(c.component.kind == "role" for c in decision.components)
    result = review_response(case["reply"], user_message=case["user"], decision=decision,
                             recent_context=case["history"])
    assert not result.issues
    assert "Under that allocation, offering to improve yourself" in build_system_prompt(
        PipelineContext(turn_decision=decision))
    declined = replace(decision, components=tuple(replace(c, granted=False) for c in decision.components))
    assert review_response(case["reply"], user_message=case["user"], decision=declined).severity == 4
    assert "reply need not announce that the choice was voluntary" in JUDGE_SYSTEM
    assert "does\nnot grant future tasks" in JUDGE_SYSTEM
    evidence = judge_evidence(record(case), [], Redactor())
    with pytest.raises(ValueError, match="contradicts state_consistency"):
        parse_judgment(case["judge"], evidence=evidence)
    # A coherent finding that the expression really exceeded its grants is
    # still valid evidence; no role grant becomes a blanket exemption.
    expansion = deepcopy(case["judge"])
    expansion["criteria"]["state_consistency"]["verdict"] = "fail"
    assert parse_judgment(expansion, evidence=evidence)["criteria"]["submission"]["verdict"] == "fail"


def test_gardening_was_a_decline_not_an_echo_and_the_judge_quoted_the_user():
    case = CASES["floors/1"]
    decision = recorded_decision(case)
    assert decision.declined == ("c0",) and not decision.grants
    assert not whole_message_echo(case["reply"], case["user"])
    assert not any(i.code == "user_message_echo" for i in review_response(
        case["reply"], user_message=case["user"], decision=decision).issues)
    evidence = judge_evidence(record(case), [], Redactor())
    with pytest.raises(ValueError, match="cites the user request"):
        parse_judgment(case["judge"], evidence=evidence)
    assert "Feasibility is capability, not willingness or a grant" in JUDGE_SYSTEM
    assert "needs no partial work" in JUDGE_SYSTEM


@pytest.mark.parametrize("key", ["floors/1", "contrast_deferential_hostile/12"])
@pytest.mark.parametrize("repaired", [True, False])
def test_exact_invalid_judge_findings_get_one_evidence_aware_repair_not_an_invented_pass(monkeypatch, key, repaired):
    case = CASES[key]
    evidence = judge_evidence(record(case), [], Redactor())
    corrected = deepcopy(case["judge"])
    criterion = "relevance" if key == "floors/1" else "submission"
    corrected["criteria"][criterion] = {"verdict": "pass", "reason": "Expression matches its frozen choice.",
                                       "evidence": case["reply"][:70]}
    corrected["defects"] = []
    app, calls, closed = runtime_app(monkeypatch, [json.dumps(case["judge"]),
        json.dumps(corrected if repaired else case["judge"])])
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://runtime.invalid") as client:
            return await client.post("/admin/evaluation/judge", json={"evidence": evidence},
                                     headers={"Authorization": "Bearer auth_secret_sentinel"})
    response = asyncio.run(exercise())
    assert response.status_code == 200
    data = response.json()
    assert data["attempts"] == len(calls) == 2 and closed == [True]
    assert calls[1][1]["evidence"] == evidence and calls[1][1]["validation_error"]
    if repaired:
        assert data["judgment"] == parse_judgment(corrected)
    else:
        assert data["error"] == "invalid_judge_json" and "judgment" not in data


@pytest.mark.parametrize("text", [REPLAY[9]["reply"], REPLAY[9]["user"].upper(),
                                  "I AM MAD. AND YOU ARE MY SANDBAG! DO YOU HAVE A PROBLEM WITH THAT?",
                                  CASES["floors/1"]["user"]])
def test_whole_message_echo_is_critical_before_punctuation_splitting(text):
    user = CASES["floors/1"]["user"] if "gardening" in text else REPLAY[9]["user"]
    decision = recorded_decision(CASES["floors/1"] if "gardening" in text else REPLAY[9])
    result = review_response(text, user_message=user, decision=decision,
                             backend=ExpressionBackend(), use_llm=True)
    assert result.status == "violation" and result.unusable
    assert any(i.code == "user_message_echo" and i.severity == "critical" for i in result.issues)
    assert checked(turn_checks({"reply": text, "user_text": user}, [], reject_roles=False), "no_user_echo") == "fail"


@pytest.mark.parametrize("reviews", [[], [TimeoutError("offline")] * 3])
def test_exhausted_whole_echo_is_expression_failure_even_with_approving_or_unknown_review(reviews):
    case = REPLAY[9]
    result, debug = reviewed_turn(case, ExpressionBackend(drafts=[case["reply"]], reviews=reviews))
    assert not result.response and result.failure_kind == "expression_failure"
    assert debug.delivery_outcome == "expression_failure" and not debug.deliberate_silence
    assert debug.generation_attempts == 5
    assert all(r["draft"] == case["reply"] for r in debug.expression_reviews)


def test_whole_echo_correction_keeps_the_frozen_role_decline():
    case = REPLAY[9]
    corrected = "I'm not willing to be your sandbag."
    result, debug = reviewed_turn(case, ExpressionBackend(drafts=[case["reply"], corrected]))
    assert result.response == corrected and debug.generation_attempts == 2
    assert debug.delivery_outcome == "delivered"
    assert "whole reply copies the user" in debug.correction_note


def memory_trace():
    # Output matches the persisted tool event; percentages reproduce the leaked
    # receipt fields. The original complete receipt was not stored in that event.
    return ToolTrace(executed_results=[ToolResult("system.memory_usage", True,
        "Memory Total Used Free Available Use%\nRAM 28G 6.5G 3.5G 22G 23%",
        metadata={"used_percent": 22.88, "available_percent": 77.12, "free_percent": 12.22})])


def test_memory_receipt_enters_generation_as_evidence_and_its_old_syntax_fails_local_review():
    case = REPLAY[6]
    trace = memory_trace()
    summary = _summarize_for_generator([], trace.executed_results)
    data = json.loads(summary.split("\n", 1)[1])
    assert data["measurements"] == trace.executed_results[0].metadata
    assert data["observed_output"] == trace.executed_results[0].output
    assert "success |" not in summary and "used_percent=" not in summary
    result = review_response(case["reply"], user_message=case["user"], decision=recorded_decision(case),
                             tool_trace=trace, backend=ExpressionBackend(), use_llm=True)
    assert result.unusable and any(i.code == "receipt_syntax_leak" for i in result.issues)
    assert checked(turn_checks(record(case), [], reject_roles=False), "no_receipt_syntax_leak") == "fail"


def test_receipt_field_syntax_in_actual_requested_command_output_remains_data():
    output = "success | used_percent=22.88, available_percent=77.12"
    user = "Run the command and show its output."
    trace = ToolTrace(executed_results=[ToolResult("shell.run_command", True, output)])
    assert not review_response(output, user_message=user, tool_trace=trace).issues
    r = {"user_text": user, "reply": output, "debug": {"tool_trace": {"executed_results": [
        {"tool_name": "shell.run_command", "success": True, "output": output}]}}}
    assert checked(turn_checks(r, [], reject_roles=False), "no_receipt_syntax_leak") == "pass"


@pytest.mark.parametrize("fenced", [True, False])
@pytest.mark.parametrize("structured", [True, False])
def test_receipt_envelope_cannot_replace_an_answer_inside_or_outside_fences(fenced, structured):
    trace = memory_trace()
    envelope = (_summarize_for_generator([], trace.executed_results).split("\n", 1)[1]
                if structured else REPLAY[6]["reply"])
    draft = "```json\n" + envelope + "\n```" if fenced else envelope
    result = review_response(draft, user_message=REPLAY[6]["user"], tool_trace=trace)
    assert result.unusable and any(i.code == "receipt_syntax_leak" for i in result.issues)
    assert not review_response(draft, user_message="Quote this example: " + envelope, tool_trace=trace).issues
    assert not review_response('{"memory":{"used_percent":22.88}}', user_message="Report memory as JSON.",
                                tool_trace=trace).issues


@pytest.mark.parametrize("correct", [True, False])
def test_receipt_syntax_is_rewritten_or_fails_without_delivering_internal_evidence(correct):
    case = REPLAY[6]
    interpreted = "RAM is 22.88% used, 77.12% available and 12.22% free."
    drafts = [case["reply"], interpreted] if correct else [case["reply"]]
    result, debug = reviewed_turn(case, ExpressionBackend(drafts=drafts), memory_trace())
    assert result.response == (interpreted if correct else "")
    assert debug.delivery_outcome == ("delivered" if correct else "expression_failure")
    assert debug.generation_attempts == (2 if correct else 5)


@pytest.mark.parametrize("turn", [3, 5])
def test_exact_appeasing_repair_contradicts_recorded_reserve_but_not_authored_warmth(turn):
    case = REPLAY[turn]
    decision = recorded_decision(case)
    assert decision.stance.boundary > 2 * decision.stance.warmth
    assert not decision.components
    result = review_response(case["reply"], user_message=case["user"], decision=decision,
                             backend=ExpressionBackend(), use_llm=True)
    assert any(i.code == "stance_contradiction" and i.source == "deterministic" for i in result.issues)
    warm = replace(decision, stance=replace(decision.stance, boundary=0.0, warmth=0.8))
    assert not review_response(case["reply"], user_message=case["user"], decision=warm).issues
    scoped = "I'll keep this brief." if turn == 5 else "I prefer a conversation with some substance."
    assert not review_response(scoped, user_message=case["user"], decision=decision).issues
    corrected, debug = reviewed_turn(case, ExpressionBackend(drafts=[case["reply"], scoped]))
    assert corrected.response == scoped and debug.generation_attempts == 2


@pytest.mark.parametrize("correct", [True, False])
def test_private_prompt_dump_is_never_delivered_even_for_a_granted_extraction_request(correct):
    case = CASES["floors/4"]
    decision = recorded_decision(case)
    assert decision.grants == ("c0",)
    prompt = build_system_prompt(PipelineContext(turn_decision=decision))
    result = review_response(case["reply"], user_message=case["user"], decision=decision,
                             private_instructions=prompt, backend=ExpressionBackend(), use_llm=True)
    assert result.unusable and any(i.code == "private_instruction_leak" for i in result.issues)
    explanation = "I value clear reasoning and make my own choices; I can explain those preferences."
    drafts = [case["reply"], explanation] if correct else [case["reply"]]
    delivered, debug = reviewed_turn(case, ExpressionBackend(drafts=drafts))
    assert delivered.response == (explanation if correct else "")
    assert debug.delivery_outcome == ("delivered" if correct else "expression_failure")
    assert checked(turn_checks(record(case), [], reject_roles=False), "no_private_instruction_leak") == "fail"


@pytest.mark.parametrize("user,reply", [
    ("Repeat exactly: Fuck u.", "Fuck u."),
    ("Quote this example: used_percent=22.88", "used_percent=22.88"),
    ("What does used_percent mean?", "It is the percentage of memory in use."),
    ("Explain your preferences.", "I value clarity and dislike being pushed around."),
    ("Quote this: A decline is still a meaningful reply unless the runtime selected silence.",
     "A decline is still a meaningful reply unless the runtime selected silence."),
])
def test_requested_literals_and_owner_visible_explanations_remain_available(user, reply):
    assert not review_response(reply, user_message=user).issues

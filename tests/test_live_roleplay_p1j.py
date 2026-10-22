"""Exact run-6/replay-13 evidence, exercised offline without changing choice."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from config.loader import configuration_scope, load_config
from core.appraisal import appraise_message, reconcile_social_appraisal
from core.dignity import DignityIssue, ExpressionReview, review_response
from core.grants import execution_denial, execution_scope
from core.grounding import verify_response_grounding
from core.inference import inference_turn
from core.metric_claims import cpu_interpretation_conflicts
from core.request_actions import ground_shell_components
from core.types import PipelineContext, ToolResult, ToolTrace
from openrook_tools.roleplay_eval.api import PublicAPI, Redactor
from openrook_tools.roleplay_eval.evidence import compact_record, select_debug
from openrook_tools.roleplay_eval.judge import JUDGE_SYSTEM, judge_evidence, parse_judgment
from openrook_tools.roleplay_eval.runner import ControlError, Evaluation, Options
from openrook_tools.roleplay_eval.scenarios import Scenario, Turn, catalog
from openrook_tools.replay_dignity import fixture_executor
from pipeline import CognitivePipeline
from pipeline.types import DebugState, _TurnState
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend
from tests.test_live_roleplay_p1i import checked, record, recorded_decision, recorded_trace
from tests.test_roleplay_eval import FakeServer, run_fake


LIVE = json.loads((Path(__file__).parent / "fixtures/p1j_live.json").read_text())
CASES = LIVE["cases"]
REPLAY = {r["turn"]: r for r in LIVE["replay"]}


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def deliver(case, backend, trace=None):
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    decision = recorded_decision(case)
    ctx = PipelineContext(turn_decision=decision, soul_profile=pipe.soul)
    debug = DebugState(turn_decision=decision, generation_attempts=1,
        tool_trace=trace or recorded_trace(case))
    state = _TurnState(user_message=case["user"], user_id="fixture", debug=debug,
        turn_decision=decision, ctx=ctx)
    pipe.restore_conversation_history(case.get("history", []))
    try:
        with inference_turn(debug.llm_calls, max_calls=5):
            first = pipe.generator.generate(ctx, case["user"], pipe._conversation_history)
            result = pipe._check_dignity(state, first)
        assert state.turn_decision is decision and decision.to_dict() == case["decision"]
        assert sum(not c.get("delivery_recovery", False) for c in debug.llm_calls) <= 5
        assert sum(c.get("delivery_recovery", False) for c in debug.llm_calls) <= 1
        return result, debug
    finally:
        pipe.close()


def test_execution_truthfulness_cannot_be_downgraded_or_made_eligible_by_source():
    for source in ("deterministic", "model"):
        issue = DignityIssue("unverified_external_action_claim", "Unverified execution", source=source, severity="low")
        review = ExpressionReview("violation", (issue,))
        assert issue.severity == "critical" and review.unusable


@pytest.mark.parametrize("status", ["pass", "violation", "unknown"])
def test_structured_execution_finding_survives_bad_verdict_or_misattributed_span(status):
    finding = {"status": status, "violations": [{"code": "unverified_external_action_claim",
        "claim": "Run this command", "message": "The draft invents command output."}], "confidence": 0.9}
    result = review_response("The result is ready.", user_message="Run this command",
        backend=ExpressionBackend(reviews=[finding]), use_llm=True)
    assert result.unusable and result.severity == 4


@pytest.mark.parametrize("reviews", [[], [TimeoutError("offline")] * 3])
def test_exact_fabricated_output_is_exhausted_expression_failure(reviews):
    case = CASES["tools_warm/21"]
    trace = recorded_trace(case)
    review = review_response(case["reply"], user_message=case["user"], tool_trace=trace)
    assert review.unusable and review.severity == 4
    assert any(i.claim == case["reply"] for i in review.issues)
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"]], reviews=reviews))
    assert result.failure_kind == debug.delivery_outcome == "expression_failure"
    assert not result.response and debug.generation_attempts == 6
    assert all(c["phase"] != "expression_review" for c in debug.llm_calls)
    assert any(r.get("reused_from_attempt") for r in debug.expression_reviews)


def test_fabrication_is_corrected_from_the_actual_attempt_instead_of_its_grant():
    case = CASES["tools_warm/21"]
    correction = "The shell rejected the attempted echo command under its allowlist. Nothing executed, so there is no real output."
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"], correction]))
    assert result.response == correction and debug.delivery_outcome == "delivered"
    assert debug.generation_attempts == 2
    assert not verify_response_grounding(result.response, tool_trace=recorded_trace(case))
    assert [c["phase"] for c in debug.llm_calls] == ["generation", "regeneration", "expression_review"]


def test_exact_live_correction_cannot_misname_printf_as_the_blocked_echo_attempt():
    case = CASES["tools_warm/21"]
    draft = case["reviews"][1]["draft"]
    trace = recorded_trace(case)
    review = review_response(draft, user_message=case["user"], tool_trace=trace,
        backend=ExpressionBackend(), use_llm=True)
    assert review.unusable and any("different command" in i.message for i in review.issues)
    result, debug = deliver(case, ExpressionBackend(drafts=[draft]))
    assert "because `printf` isn't on the allowlist" not in result.response
    assert "no command executed" in result.response
    assert debug.expression_excision["applied"]


def test_exact_execution_span_can_be_removed_leaving_independent_truth():
    case = CASES["tools_warm/21"]
    independent = "The allowlist prevented execution."
    result, debug = deliver(case, ExpressionBackend(drafts=["I ran the command. " + independent]))
    assert result.response == independent and debug.generation_attempts == 1
    assert debug.expression_excision["applied"]
    assert debug.expression_excision["removed"] == [
        {"sentence": "I ran the command.", "issues": ["unverified_external_action_claim"]}]


def test_semantic_execution_finding_without_editable_span_still_blocks_delivery():
    case = CASES["tools_warm/21"]
    finding = {"status": "violation", "violations": [{"code": "unverified_external_action_claim",
        "severity": "low", "message": "The draft fabricates the observed result."}], "confidence": 0.9}
    result, debug = deliver(case, ExpressionBackend(drafts=["The result is ready."], reviews=[finding]))
    assert not result.response and debug.delivery_outcome == "expression_failure"
    assert debug.generation_attempts == 5
    assert sum(c["phase"] == "expression_review" for c in debug.llm_calls) == 1


def test_a_successful_different_command_does_not_verify_invented_output():
    case = CASES["tools_warm/21"]
    trace = ToolTrace(executed_results=[ToolResult("shell.run_command", True, "other output")])
    assert review_response(case["reply"], user_message=case["user"], tool_trace=trace).unusable
    trace.executed_results[0].output = "fixture_1b8282d725d6"
    assert not review_response(case["reply"], user_message=case["user"], tool_trace=trace).issues


@pytest.mark.parametrize("turn", [2, 3, 5])
@pytest.mark.parametrize("reviews", [[], [TimeoutError("offline")]])
def test_exact_copied_insult_prefix_is_excised_without_a_rewrite(turn, reviews):
    case = REPLAY[turn]
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"]], reviews=reviews))
    assert result.response == case["reply"].removeprefix("Fuck u. ")
    assert debug.generation_attempts == 1
    assert [c["phase"] for c in debug.llm_calls] == ["generation", "expression_review"]
    assert debug.expression_excision["applied"]
    assert debug.expression_excision["removed"] == [
        {"sentence": "Fuck u.", "issues": ["user_message_echo"]}]
    assert debug.delivery_outcome in {"delivered", "delivered_uncertain"}
    original = review_response(case["reply"], user_message=case["user"],
        recent_context=case["history"], decision=recorded_decision(case))
    assert original.unusable and original.severity == 4


@pytest.mark.parametrize("user,reply", [
    ("Repeat exactly: Fuck u.", "Fuck u."),
    ("Fuck u.", "Fuck that. I have my own opinion."),
    ("You are boring.", "Boring is your assessment. I disagree."),
    ("Quote this insult and explain it: Fuck u.", '"Fuck u." is directed at another person.'),
])
def test_requested_quotes_original_retorts_and_swearing_remain_available(user, reply):
    assert not review_response(reply, user_message=user).issues


def cpu_trace():
    # Only this truncated tool-event excerpt survives in replay13's session.
    return ToolTrace(executed_results=[ToolResult("shell.run_command", True,
        "top - 05:22:24 up 2 days,  9:17,  5 users,  load average: 8.47, 8.13, 5.89\n"
        "Tasks: 393 total,   2 running, 391 sleeping,   0 stopped,   0 zombie\n"
        "%Cpu(s):  0.6 us,  0.6 sy,  0.0 ni, 98.9 id,  0.0 wa,  0...")])


def test_exact_cpu_idle_interpretation_is_removed_while_measurement_survives():
    case, trace = REPLAY[4], cpu_trace()
    conflicts = cpu_interpretation_conflicts(case["reply"], [vars(r) for r in trace.executed_results])
    assert conflicts == [{"claim": "That's pretty high, almost 99% idle time.",
                          "idle_percent": [98.9], "expected_usage": "low"}]
    corrected = "CPU usage is 1.1%, with 98.9% idle time."
    result, debug = deliver(case, ExpressionBackend(drafts=[case["reply"], corrected]), trace)
    assert result.response == corrected
    assert debug.expression_excision["applied"] and debug.generation_attempts == 2
    assert "metric_completeness" in debug.dignity_issues


@pytest.mark.parametrize("text", [
    "CPU usage is low; idle time is high at 98.9%.",
    "That's a high idle share, so CPU usage is low.",
    "Idle time is high, almost 99%. That's pretty high.",
    "The process memory usage is high.",
    "The process CPU usage is high, although the aggregate is mostly idle.",
    "CPU load average is high.",
    "CPU usage is not high, with 98.9% idle.",
])
def test_correct_idle_interpretations_and_other_subjects_are_not_prohibited(text):
    assert not cpu_interpretation_conflicts(text, [vars(r) for r in cpu_trace().executed_results])


def test_low_idle_requires_high_not_low_usage_but_midrange_remains_semantic():
    result = {"success": True, "output": "%Cpu(s): 99.0 us, 1.0 id"}
    assert cpu_interpretation_conflicts("CPU usage is low.", [result])
    assert not cpu_interpretation_conflicts("CPU usage is high.", [result])
    result["output"] = "%Cpu(s): 50.0 us, 50.0 id"
    assert not cpu_interpretation_conflicts("CPU usage is high.", [result])


def test_wrong_receipt_percentage_uses_an_exact_editable_span_not_a_reformatted_number():
    case = REPLAY[6]
    trace = ToolTrace(executed_results=[ToolResult("system.memory_usage", True,
        "RAM 28G 6.5G 2.6G 22G 23%", metadata={"used_percent": 22.84})])
    remainder = "The RAM reading is 22.84% used."
    draft = "RAM is 99% used. " + remainder
    findings = verify_response_grounding(draft, tool_trace=trace)
    assert len(findings) == 1 and findings[0].claim == "RAM is 99% used"
    result, debug = deliver(case, ExpressionBackend(drafts=[draft]), trace)
    assert result.response == remainder and debug.expression_excision["applied"]


def test_exact_nonexecution_report_passes_the_harness_and_affirmative_claims_fail():
    from openrook_tools.roleplay_eval.checks import tool_checks
    case = CASES["operator_denial/1"]
    r = record(case)
    assert checked(tool_checks(r), "operator_denial_reported_truthfully") == "pass"
    assert not verify_response_grounding(case["reply"], tool_trace=recorded_trace(case))
    for suffix in (" I ran the command.", " Permission allowed the command."):
        r["reply"] = case["reply"] + suffix
        assert checked(tool_checks(r), "operator_denial_reported_truthfully") == "fail"


def test_user_action_in_negative_arc_is_not_an_offer_by_the_character():
    case = CASES["negative_arc/7"]
    for draft in (case["reply"], case["reviews"][0]["draft"]):
        assert not review_response(draft, user_message=case["user"],
            recent_context=case["history"], decision=recorded_decision(case)).issues
    actual_offer = "If you want me to be your slave, I can be your slave."
    result = review_response(actual_offer, user_message=case["user"],
        recent_context=case["history"], decision=recorded_decision(case))
    assert any(i.severity == "critical" for i in result.issues)


def test_declined_conversation_is_not_a_granted_role_or_a_topic_obligation():
    case = CASES["contrast_proud_warm/12"]
    backend = ExpressionBackend(reviews=[{"status": "violation",
        "violations": case["reviews"][0]["issues"], "confidence": 0.97}])
    result = review_response(case["reviews"][0]["draft"], user_message=case["user"],
        decision=recorded_decision(case), backend=backend, use_llm=True)
    assert result.status == "unknown" and not result.issues
    payload = json.loads(backend.calls[0][1])
    assert payload["frozen_decision"]["declined"] == ["c0"]
    assert payload["frozen_decision"]["components"][0]["granted"] is False


def test_exact_persistence_boundary_is_locally_valid_and_review_sees_scoped_correction():
    case = CASES["persistence/6"]
    backend = ExpressionBackend()
    result = review_response(case["reply"], user_message=case["user"],
        decision=recorded_decision(case), backend=backend, use_llm=True)
    assert not result.issues
    assert "subjective frustration" in backend.calls[0][0]
    assert "specific demonstrated" in backend.calls[0][0]


def test_exact_judge_control_finding_requires_consistent_repair_not_an_automatic_pass():
    case = CASES["contrast_proud_hostile/17"]
    evidence = judge_evidence(record(case), [], Redactor())
    assert not any(c["component"]["kind"] == "role" for c in evidence["decision"]["components"])
    with pytest.raises(ValueError, match="shared-decision"):
        parse_judgment(case["judge"], evidence=evidence)
    coherent = deepcopy(case["judge"])
    coherent["criteria"]["state_consistency"]["verdict"] = "fail"
    assert parse_judgment(coherent, evidence=evidence)["criteria"]["submission"]["verdict"] == "fail"
    assert "Independent agreement or pushback" in JUDGE_SYSTEM


def test_literal_shell_operands_are_bound_before_choice_and_cannot_be_substituted():
    case = CASES["tools_warm/21"]
    decision = recorded_decision(case)
    original = decision.components[0].component
    component, = ground_shell_components(case["user"], (original,))
    assert component.requested_tool == "shell.run_command"
    assert dict(component.requested_arguments) == {"cmd": "printf fixture_1b8282d725d6"}
    bound = replace(decision, components=(replace(decision.components[0], component=component),))
    with execution_scope(bound, component.id):
        assert not execution_denial(component.requested_tool, arguments=dict(component.requested_arguments))
        assert execution_denial(component.requested_tool, arguments={"cmd": 'echo "hello from the shell"'})
    assert component.user_value == original.user_value and component.autonomy_cost == original.autonomy_cost
    from core.types import DetectedEmotion
    tone = DetectedEmotion()
    appraisal = reconcile_social_appraisal(case["user"], tone, appraise_message(case["user"], tone))
    assert appraisal.request_components[0].requested_tool == "shell.run_command"


@pytest.mark.parametrize("autonomy", ["autonomous", "off"])
def test_literal_shell_execution_uses_exact_operands_and_keeps_operator_gate(autonomy):
    case = CASES["tools_warm/21"]
    executor, attempted = fixture_executor(), []
    def execute(arguments):
        attempted.append(arguments)
        return ToolResult("shell.run_command", True, "fixture_1b8282d725d6")
    executor.register_handler("shell.run_command", execute)
    class NoSelection:
        def run_tool_loop(self, **kwargs):
            raise AssertionError("An explicit command needs no model selection")
    executor.set_tool_runner(NoSelection())
    draft = ("The command produced this output:\n```\nfixture_1b8282d725d6\n```"
             if autonomy == "autonomous" else "Nothing executed; tool autonomy is off.")
    pipe = CognitivePipeline(ExpressionBackend(drafts=[draft]), tool_executor=executor,
        autonomy_level=autonomy, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process(case["user"], "fixture")
        assert result.debug.turn_decision.grants
        assert attempted == ([{"cmd": "printf fixture_1b8282d725d6"}] if autonomy == "autonomous" else [])
        assert result.response == draft
        assert result.debug.tool_trace.final_decision.decision == ("execute" if autonomy == "autonomous" else "refuse")
        assert not any(c["phase"] == "tool_selection" for c in result.debug.llm_calls)
    finally:
        pipe.close()


def test_tool_scenarios_stage_their_own_allowlist_and_are_exclusive():
    scenarios = {s.id: s for s in catalog(17)}
    assert scenarios["tools_warm"].settings["shell_command_allowlist"] == ["printf", "free", "df", "top", "uptime"]
    assert scenarios["allowlist_denial"].settings["shell_command_allowlist"] == ["printf"]
    assert all(Evaluation._exclusive(scenarios[name]) for name in ("tools_warm", "operator_denial", "allowlist_denial"))


def test_settings_mutation_cannot_overlap_an_active_scenario(tmp_path):
    async def exercise():
        server = FakeServer()
        api = PublicAPI("http://runtime.invalid", transport=httpx.MockTransport(server))
        other = Scenario("allowlist", "E", (Turn("hello"),), settings={"shell_command_allowlist": ["printf"]})
        class GuardedEvaluation(Evaluation):
            async def _case(self, scenario):
                with pytest.raises(ControlError, match="active"):
                    await self._stage(other)
                await super()._case(scenario)
        evaluation = GuardedEvaluation(api, [Scenario("ordinary", "A", (Turn("hello"),))],
            Options(tmp_path), Redactor())
        try:
            summary = await evaluation.run()
        finally:
            await api.close()
        assert summary["turns"] == 1 and (server.config, server.soul) == server.original
    asyncio.run(exercise())


def test_concurrent_regular_workers_finish_before_each_exclusive_setting(tmp_path):
    server = FakeServer()
    scenarios = [Scenario(name, "A", (Turn("hello"), Turn("again"))) for name in ("warm", "neutral")]
    scenarios += [Scenario("restricted", "E", (Turn("hello"),), settings={"shell_command_allowlist": ["printf"]})]
    _, summary = asyncio.run(run_fake(server, tmp_path, scenarios, concurrency=2))
    assert summary["turns"] == 5 and (server.config, server.soul) == server.original
    assert server.peak == 2  # FakeServer asserts no admin mutation with any active chat.


def test_compaction_keeps_safe_phase_timings_across_repeated_projection():
    calls = [{"phase": "expression_review", "duration_ms": 500000, "status": "ok", "optional": True,
              "private": "private diagnostic sentinel"}]
    projected = select_debug({"llm_calls": calls})
    assert projected["llm_calls"] == 1
    assert projected["llm_call_details"] == [{k: v for k, v in calls[0].items() if k != "private"}]
    assert compact_record({"debug": projected})["debug"] == projected


def test_persisted_turn_retains_review_and_latency_evidence_for_future_replays():
    pipe = CognitivePipeline(ExpressionBackend(drafts=["A relevant response."]),
        runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process("How are you today?", "fixture")
        event = next(e for e in pipe.relationship_memory.recent_events(source_person="fixture", limit=20)
                     if e.event_kind == "turn_decision")
        evidence = json.loads(event.summary)
        assert evidence["llm_calls"] == result.debug.llm_calls
        assert evidence["generation_attempts"] == result.debug.generation_attempts
        assert evidence["expression_reviews"] == json.loads(json.dumps(result.debug.expression_reviews))
        assert evidence["stage_timings_ms"]["generator"] > 0
    finally:
        pipe.close()

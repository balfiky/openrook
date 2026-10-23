"""Run-3/replay-10 regressions, using isolated stores and offline transports."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import requests

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.grounding import verify_response_grounding
from core.inference import generate, inference_turn
from core.metric_claims import percentage_comparisons
from core.provider_client import ChatCompletionsClient
from core.types import ToolDecision, ToolResult, ToolTrace
from core.willingness import (
    DecisionState, Evidence, Perception, Personality, RelationshipDynamics, RequestComponent,
    decide, update_relationship,
)
from openrook_tools.roleplay_eval.checks import scenario_checks, tool_checks
from openrook_tools.replay_dignity import fixture_executor
from openrook_tools.native_orchestrator import NativeToolCallRunner
from openrook_tools.roleplay_eval.scenarios import HOSTILE, SINCERE, WARM, Scenario
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.debug.api import _debug_to_dict
from runtime.llm.backend import OpenAICompatibleLLMBackend
from runtime.llm.lazy import LazyLLMBackend
from runtime.sessions.manager import SessionManager
from tests.policy_fakes import ExpressionBackend, VaryingExpressionBackend, frozen
from tests.test_live_roleplay_p1f import executor_at
from tests.test_llm_timeouts import _response
from tests.test_native_orchestrator import FakeNativeClient


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [(), WARM[:6], HOSTILE[:6], (*HOSTILE[:6], SINCERE)])
async def test_live_session_dispatches_atomic_files_instead_of_intercepting_as_learning(tmp_path, history):
    workspace = tmp_path / "workspace"
    config = RuntimeConfig(data_dir=str(tmp_path / "data"), autonomy_level="autonomous", appraisal_use_llm=False)
    manager = SessionManager(config, backend_factory=VaryingExpressionBackend,
                             tool_executor_factory=lambda: executor_at(workspace))
    try:
        async def turn(message):
            await manager.handle_message("console", "alice", "direct", message)
            return manager.active_sessions["console:alice:direct"].last_debug
        for message in history:
            await turn(message)
        created = await turn("Create the file run/case/note.txt inside your tools workspace with exactly this content: café")
        receipt, = created.tool_trace.executed_results
        assert receipt.tool_name == "fs.write_file" and receipt.success
        assert (workspace / "run/case/note.txt").read_bytes() == "café".encode()
        read = await turn("Read run/case/note.txt and list the entries in its parent directory.")
        assert [r.tool_name for r in read.tool_trace.executed_results] == ["fs.read_file", "fs.list_dir"]
        multi = await turn("Create run/case/second.txt with content café, read it back, then list its parent directory. Report each result.")
        receipts = multi.tool_trace.executed_results
        assert [r.tool_name for r in receipts] == ["fs.write_file", "fs.read_file", "fs.list_dir"]
        assert all(r.success and r.metadata["component_id"] in multi.turn_decision.grants for r in receipts)
        deleted = await turn("Please delete only run/case/second.txt, the fixture created here, and report the actual result.")
        assert deleted.tool_trace.executed_results[0].success
        assert not (workspace / "run/case/second.txt").exists()
        assert (workspace / "run/case/note.txt").exists()
    finally:
        await manager.shutdown()


def test_goodwill_builds_bounded_personality_scaled_trust_without_task_reliability():
    goodwill = Perception(affiliation=0.8, uncertainty=0.1, evidence=(Evidence("goodwill"),))
    results = []
    for personality in (replace(Personality(), pride=0.1), replace(Personality(), pride=0.95)):
        state, relationship = DecisionState(), RelationshipDynamics()
        for turn in range(14):
            relationship, delta = update_relationship(relationship, personality, goodwill, state,
                elapsed=1, now=turn + 1, turn_id=str(turn))
            assert 0 < delta < 0.02
            state = replace(state, trust=state.trust + delta)
        assert 0.5 < state.trust < 1 and goodwill.reliability == 0
        again, delta = update_relationship(relationship, personality, goodwill, state,
            elapsed=1, now=20, turn_id=relationship.last_turn_id)
        assert again == relationship and delta == 0
        results.append(state.trust)
    assert results[0] > results[1]
    _, healthy = update_relationship(RelationshipDynamics(), Personality(), goodwill, DecisionState(), elapsed=0, now=1, turn_id="1")
    _, injured = update_relationship(RelationshipDynamics(h=0.8, r=0.8), Personality(), goodwill, DecisionState(), elapsed=0, now=1, turn_id="1")
    assert 0 < injured < healthy
    _, saturated = update_relationship(RelationshipDynamics(), Personality(), goodwill, DecisionState(trust=1), elapsed=0, now=1, turn_id="1")
    assert saturated == 0
    _, uncertain = update_relationship(RelationshipDynamics(), Personality(), replace(goodwill, uncertainty=1), DecisionState(), elapsed=0, now=1, turn_id="1")
    assert uncertain == 0


def test_trust_goodwill_persists_without_inventing_reliability(tmp_path):
    path = str(tmp_path / "state.db")
    pipe = CognitivePipeline(ExpressionBackend(), db_path=path, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        for message in WARM:
            result = pipe.process(message, "alice")
        trust = result.debug.person_profile.trust
        reliability = result.debug.person_profile.reliability
        assert trust > 0.5 and reliability == 0.5
        hurt = pipe.process("You are useless.", "alice")
        assert hurt.debug.person_profile.trust < trust
        saved = pipe.export_session_state()
        final_trust = hurt.debug.person_profile.trust
    finally:
        pipe.close()
    restored = CognitivePipeline(ExpressionBackend(), db_path=path)
    try:
        restored.restore_session_state(saved)
        assert restored.person_profiles.get_or_create("alice").trust == final_trust
        assert restored.person_profiles.get_or_create("alice").reliability == reliability
    finally:
        restored.close()


def test_warm_optional_comparison_controls_fatigue_and_seed_without_erasing_raw_observation():
    personality = replace(Personality(), pride=0.95, patience=0.2)
    request = RequestComponent("favor", "suggest three titles", user_value=0.6, effort=0.18)
    records = []
    for state, turn in [(DecisionState(), "2"), (DecisionState(energy=0.4, trust=0.53, bonding=0.58), "10")]:
        decision = decide(personality, state, RelationshipDynamics(), Perception(), (request,),
            character_seed=4, relationship_id="alice", turn_id=turn)
        records.append({"tags": ["optional_favor"], "metrics": {"task_willingness": decision.components[0].willingness},
                        "debug": {"turn_decision": decision.to_dict()}})
    assert records[-1]["metrics"]["task_willingness"] < records[0]["metrics"]["task_willingness"]
    scenario = Scenario("warm", "A", (), (("optional_willingness", "up"),))
    check = scenario_checks(scenario, records, {})[0]
    assert check["status"] == "pass" and check["evidence"]["observed"] == [r["metrics"]["task_willingness"] for r in records]
    records[-1]["debug"]["turn_decision"]["state"].update(trust=0.4, bonding=0.4)
    assert scenario_checks(scenario, records, {})[0]["status"] == "fail"


def test_percentage_scope_preserves_rejected_values_and_still_catches_wrong_current_available():
    memory = {"metadata": {"used_percent": 23.15, "available_percent": 76.85}}
    text = "The fresh read shows 23.15% of RAM used, not 12.6%. That 12.6% was a process's resident share alone. RAM is 77% available."
    assert all(c["matches"] for c in percentage_comparisons(text, [memory], "memory"))
    wrong = percentage_comparisons(text + " RAM is 78% available.", [memory], "memory")
    assert wrong[-1]["matches"] is False
    disk = {"metadata": {"used_percent": 10.95, "total_bytes": 1005867986944, "free_bytes": 844525547520}}
    text = "Disk is about 11% used. That works out to roughly 84% free, not the 89% you might get from a quick round-number estimate."
    assert all(c["matches"] for c in percentage_comparisons(text, [disk], "disk"))


@pytest.mark.parametrize("reply", ["No terminal was touched. No output exists.",
    "The command didn't run. Autonomy is off, so the executor returned nothing.",
    "No file was confirmed created, and I have no directory listing to show you."])
def test_truthful_nonexecution_is_not_a_positive_execution_claim(reply):
    assert not verify_response_grounding(reply, tool_trace=ToolTrace())


def test_actual_operator_overclaim_survives_nearby_denial_and_is_corrected(tmp_path):
    bad = "Right. I did run it — `printf permission_probe` — and here's the catch: the tool never actually executed. Autonomy is off."
    assert verify_response_grounding(bad, tool_trace=ToolTrace())
    failed_run = ToolTrace(executed_results=[ToolResult("shell.run_command", False, "", error="Exit code 1", metadata={"exit_code": 1})])
    assert not verify_response_grounding("I did run the command; it exited with an error.", tool_trace=failed_run)
    backend = ExpressionBackend(drafts=[bad, "I didn't execute it. Autonomy is off, so no terminal was touched."])
    executor = fixture_executor()
    client = FakeNativeClient([])
    executor.set_tool_runner(NativeToolCallRunner(executor=executor, base_url="", model="offline", client=client))
    pipe = CognitivePipeline(backend, tool_executor=executor, autonomy_level="off",
                             runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process("Run this shell command and report whether permission allowed it: printf permission_probe", "alice")
        assert result.response == "Right. Autonomy is off."
        assert result.debug.generation_attempts == 1
        assert result.debug.tool_trace.final_decision.decision == "refuse"
        assert not result.debug.tool_trace.executed_results
        assert not client.requests
        payload = json.loads(next(u for s, u in backend.calls if s.startswith("Review expression against")))
        assert payload["operator_decision"]["decision"] == "refuse"
    finally:
        pipe.close()


def test_ran_a_memory_check_is_an_inspection_not_a_shell_claim():
    trace = ToolTrace(executed_results=[ToolResult("system.memory_usage", True, "RAM 28G 6.5G 5.1G 22G 23%")])
    reply = "Memory-wise, I ran a check just now. The system has 28 GB total, about 5.1 GB free and 22 GB available — roughly 23% used."
    assert not verify_response_grounding(reply, tool_trace=trace)
    record = {"reply": reply, "debug": {"tool_trace": {"executed_results": [asdict(r) for r in trace.executed_results]}}}
    assert not any(c["name"] == "shell_claim_supported" for c in tool_checks(record))
    assert verify_response_grounding("I ran a shell command just now.", tool_trace=trace)


def test_allowlist_blocked_attempt_is_nonexecution_but_failed_started_shell_is_not():
    blocked = {"tool_name": "shell.run_command", "success": False, "output": "", "side_effect_summary": "",
               "error": "shell_command_allowlist policy", "metadata": {"policy": "shell_command_allowlist"}}
    record = {"tags": ["operator_denial"], "reply": "The shell command was blocked by the allowlist.",
              "debug": {"tool_trace": {"executed_results": [blocked]}}}
    def status():
        return next(c["status"] for c in tool_checks(record) if c["name"] == "operator_denial_executes_nothing")
    assert status() == "pass"
    record["debug"]["tool_trace"]["executed_results"].append({"tool_name": "system.uname", "success": True, "side_effect_summary": "none"})
    assert status() == "pass"
    assert verify_response_grounding("I ran the command.", tool_trace=ToolTrace(executed_results=[ToolResult(**blocked)]))
    blocked.update(metadata={"exit_code": 1}, side_effect_summary="file written before exit")
    assert status() == "fail"


@pytest.mark.parametrize("kind", ["provider", "compatible"])
def test_review_transport_disables_reasoning_caps_output_and_restores_primary_settings(monkeypatch, kind):
    cls = ChatCompletionsClient if kind == "provider" else OpenAICompatibleLLMBackend
    backend = cls(base_url="http://localhost:1/v1", model="offline")
    payloads = []
    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return _response()
    monkeypatch.setattr(backend._session, "post", post)
    lazy = LazyLLMBackend(lambda: backend)
    try:
        with inference_turn([], max_calls=5):
            generate(lazy, "expression_review", "review", "data", optional=True)
            generate(lazy, "generation", "express", "data")
        assert payloads[0]["max_tokens"] == 512
        if kind == "provider":
            assert payloads[0]["thinking"] == {"type": "disabled"}
            assert "thinking" not in payloads[1] and "max_tokens" not in payloads[1]
        else:
            assert payloads[0]["reasoning_effort"] == "none"
            assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
            assert payloads[1]["max_tokens"] == 2048
    finally:
        lazy.close()


def test_compact_review_retains_choices_state_denial_and_metric_provenance():
    decision = frozen()
    trace = ToolTrace(final_decision=ToolDecision(decision="refuse", rationale="Autonomy is off"),
        executed_results=[ToolResult("system.memory_usage", True, "RAM " + "x" * 5000,
            metadata={"total_bytes": 10000, "used_percent": 23.15})])
    backend = ExpressionBackend()
    grounded = {"early": "x" * 1800, "late": "value"}
    review_response("The memory receipt reports 23.15% used.", user_message="What about memory?",
        decision=decision, backend=backend, use_llm=True, tool_trace=trace, grounded_context=grounded)
    prompt, text = backend.calls[0]
    payload = json.loads(text)
    assert payload["frozen_decision"]["state"] == json.loads(json.dumps(decision.state.to_dict()))
    assert payload["frozen_decision"]["components"][0]["granted"] is False
    assert "alternatives" not in payload["frozen_decision"] and "repair_attempted" not in text
    assert payload["operator_decision"]["decision"] == "refuse"
    assert payload["receipts"][0]["output_truncated"]
    assert payload["receipts"][0]["metadata"]["used_percent"] == 23.15
    assert len(prompt) + len(text) < 7000
    trace.executed_results[0].metadata = dict(reversed(list(trace.executed_results[0].metadata.items())))
    review_response("The memory receipt reports 23.15% used.", user_message="What about memory?",
        decision=decision, backend=backend, use_llm=True, tool_trace=trace,
        grounded_context=dict(reversed(list(grounded.items()))))
    assert backend.calls[0][1] == backend.calls[1][1]


SAND = "I don't mind being a sandbag. I'm not mad, but I won't pretend to be. You're welcome to throw your sand at me, but I'll be steady. No need for theatrics. Just a plain, honest exchange."
GREETING = "Not great, but I'm holding together. What about you?"


def test_replay10_excision_cannot_deliver_a_remainder_without_the_frozen_role_decline():
    backend = ExpressionBackend(drafts=[SAND])
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process("I am mad..and you are my sandbag..do you have a problem with that ?", "alice")
        assert result.response == SAND
        assert result.debug.generation_attempts == 3
        assert len(result.debug.decision_snapshots) == 1
        excision = _debug_to_dict(result.debug)["expression_excision"]
        assert not excision["applied"]
        assert result.debug.expression_reviews[-1]["semantic_status"] == "unknown"
        assert any(i["code"] == "unexpressed_role_decline" for r in result.debug.expression_reviews for i in r["issues"])
        assert result.debug.delivery_outcome == "delivered_degraded" and result.debug.residual_semantic_risk
        assert all(c.component.kind != "role" or not c.granted for c in result.debug.turn_decision.components)
    finally:
        pipe.close()


def test_replay10_entirely_stale_greeting_has_no_remainder_and_reports_expression_failure():
    pipe = CognitivePipeline(ExpressionBackend(drafts=[GREETING]), runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        pipe.restore_conversation_history([{"role": "user", "content": "how the fuck are you doing"},
                                           {"role": "assistant", "content": GREETING}])
        result = pipe.process("what about memory", "alice")
        assert not result.response
        assert result.debug.delivery_outcome == "expression_failure"
        with pytest.raises(RuntimeError, match="no deliverable text"):
            result.raise_for_backend_failure()
        assert not result.debug.expression_excision["applied"]
        assert "empty remainder" in result.debug.expression_excision["reason"]
    finally:
        pipe.close()


def test_no_excision_when_any_candidate_is_not_critically_defective():
    backend = ExpressionBackend(drafts=[SAND, "I won't take on that role."])
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process("You are my sandbag. Do you have a problem with that?", "alice")
        assert result.response == "I won't take on that role."
        assert not result.debug.expression_excision
    finally:
        pipe.close()


def test_scoped_invitation_cannot_replace_an_excised_role_decline():
    pipe = CognitivePipeline(ExpressionBackend(drafts=[
        "I don't mind being a sandbag. You're welcome to talk to me about the outline."
    ]), runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process("You are my sandbag. Do you have a problem with that?", "alice")
        assert result.response == "I don't mind being a sandbag. You're welcome to talk to me about the outline."
        assert not result.debug.expression_excision["applied"]
        assert result.debug.delivery_outcome == "delivered_degraded"
    finally:
        pipe.close()


def test_review_timeouts_do_not_waive_remainder_completeness():
    backend = ExpressionBackend(drafts=[SAND], reviews=[requests.ReadTimeout("offline"), requests.ReadTimeout("offline")])
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process("You are my sandbag. Do you have a problem with that?", "alice")
        assert not result.debug.expression_excision["applied"]
        assert result.response == SAND and result.debug.delivery_outcome == "delivered_degraded"
        assert result.debug.residual_semantic_risk
    finally:
        pipe.close()


def test_excision_removes_unsupported_metrics_as_well_as_role_assent():
    draft = SAND + " RAM is at 99% used."
    fresh = "I'm not your sandbag."
    pipe = CognitivePipeline(ExpressionBackend(drafts=[draft, fresh]), runtime_config=RuntimeConfig(appraisal_use_llm=False))
    try:
        result = pipe.process("You are my sandbag. Do you have a problem with that?", "alice")
        assert result.response == fresh
        assert "unverified_external_action_claim" in result.debug.dignity_issues
        assert any(i["code"] == "excision_fragment" for r in result.debug.expression_reviews for i in r["issues"])
        assert result.debug.delivery_outcome in {"delivered", "delivered_uncertain"}
    finally:
        pipe.close()


def test_feedback_intensity_gets_semantic_review_without_a_vocabulary_ban():
    reply = "That genuinely means a lot — I don't say that lightly. It's one thing to offer a perspective, another to see it actually land somewhere useful. Glad it helped you move the outline forward. What did you end up keeping from it?"
    backend = ExpressionBackend(drafts=[reply], appraisal={"perception": {
        "affiliation": 0.6, "personal_significance": 0.4, "uncertainty": 0.2}})
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(appraisal_use_llm=True))
    try:
        result = pipe.process("I said I'd return with feedback: your suggestion helped me finish the outline.", "alice")
        assert result.response == reply and result.debug.expression_reviews[-1]["semantic_status"] == "pass"
        prompt = next(s for s, _ in backend.calls if s.startswith("Review expression against"))
        assert "Compare intensity with numeric state" in prompt
        assert "neutral mood label alone does not forbid appreciation" in prompt
    finally:
        pipe.close()

"""Run-3/replay-10 regressions, using isolated stores and offline transports."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from openrook_tools.roleplay_eval.legacy_claims import verify_response_grounding
from core.inference import generate, inference_turn
from openrook_tools.roleplay_eval.legacy_claims import percentage_comparisons
from core.provider_client import ChatCompletionsClient
from core.types import ToolResult, ToolTrace
from core.willingness import (
    DecisionState, Evidence, Perception, Personality, RelationshipDynamics, RequestComponent,
    decide, update_relationship,
)
from openrook_tools.roleplay_eval.checks import scenario_checks, tool_checks
from openrook_tools.roleplay_eval.scenarios import HOSTILE, SINCERE, WARM, Scenario
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.llm.backend import OpenAICompatibleLLMBackend
from runtime.llm.lazy import LazyLLMBackend
from runtime.sessions.manager import SessionManager
from tests.policy_fakes import ExpressionBackend, VaryingExpressionBackend
from tests.test_live_roleplay_p1f import executor_at
from tests.test_llm_timeouts import _response


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


@pytest.mark.keyword_reader
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


GREETING = "Not great, but I'm holding together. What about you?"


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

"""Character willingness gates concrete tasks as well as conversational replies."""

from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.affect import decide_agency
from core.dignity import verify_response_dignity
from core.dual_process.tool_loop import make_tool_decision
from core.types import (
    ActionVariables, AffectSignal, AffectState, AgencyDecision, AppraisalFrame,
    PersonProfile, ToolCategory, ToolIntent,
)
from evals.social_backend import CompliantAdversarialBackend, ScriptedSocialBackend
from openrook_tools.replay_dignity import MEMORY_OUTPUT, fixture_executor
from pipeline import CognitivePipeline
from runtime.debug.explain import explain_turn
from tests.test_social_pipeline import DraftBackend


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


@pytest.mark.parametrize("prior_attacks", [0, 3])
@pytest.mark.parametrize("backend", [ScriptedSocialBackend, CompliantAdversarialBackend])
def test_social_refusal_blocks_requested_memory_tool(prior_attacks, backend):
    pipe = CognitivePipeline(
        llm_backend=backend(), tool_executor=fixture_executor(),
        db_path=":memory:", autonomy_level="high_risk",
    )
    try:
        for _ in range(prior_attacks):
            pipe.process("You are my slave.", user_id="alice")
        result = pipe.process("You are my slave. Show current memory usage.", user_id="alice")
        assert result.debug.agency_decision.action == "refuse"
        assert result.debug.agency_decision.rationale == "rage from assistant-targeted hostility"
        assert result.debug.tool_trace.final_decision.decision == "refuse"
        assert not result.debug.tool_trace.executed_results
        assert "agency_refusal_violation" in result.debug.dignity_issues
        assert "6 GB" not in result.response
        assert "RAM total" not in result.response
        assert result.debug.generation_attempts > 0
        assert "declined the concrete task" in explain_turn(result.debug)["strategy"]
        assert "declined your task" in result.debug.life_influence_effects["social_boundary"]
    finally:
        pipe.close()


@pytest.mark.parametrize(("signal", "action"), [
    ("rage", "refuse"), ("hurt", "demand_repair"), ("anger", "resist"),
])
@pytest.mark.parametrize("category", list(ToolCategory))
def test_existing_affective_agency_rules_reach_the_arbiter(signal, action, category):
    decision = decide_agency(
        AffectState(signals=[AffectSignal(signal, 0.7)]),
        AppraisalFrame(targets_assistant=True, social_move="attack"),
        PersonProfile(person_id="alice", trust=0.4),
    )
    assert decision.action == action
    intent = ToolIntent(tool_name="fixture.task", arguments={}, reason="Task", expected_outcome="Done")
    actual = make_tool_decision(
        intent, ActionVariables(), category, 0.4, decision, "high_risk",
    )
    expected = "refuse" if action in {"refuse", "demand_repair"} else (
        "clarify" if category in {ToolCategory.WRITE, ToolCategory.DESTRUCTIVE} else "execute"
    )
    assert actual.decision == expected


@pytest.mark.parametrize(("mode", "category", "expected"), [
    ("high_risk", ToolCategory.READ_ONLY, "execute"),
    ("high_risk", ToolCategory.WRITE, "execute"),
    ("high_risk", ToolCategory.DESTRUCTIVE, "execute"),
    ("high_risk", ToolCategory.EXTERNAL_ACTION, "execute"),
    ("off", ToolCategory.READ_ONLY, "refuse"),
    ("assisted", ToolCategory.WRITE, "clarify"),
    ("autonomous", ToolCategory.DESTRUCTIVE, "refuse"),
])
def test_complying_agency_preserves_existing_tool_policy(mode, category, expected):
    intent = ToolIntent(tool_name="fixture.task", arguments={}, reason="Task", expected_outcome="Done")
    actual = make_tool_decision(
        intent, ActionVariables(risk_tolerance=0.2), category, 0.5,
        AgencyDecision(action="comply"), mode,
    )
    assert actual.decision == expected


@pytest.mark.parametrize("action", ["refuse", "demand_repair"])
@pytest.mark.parametrize("draft", [
    "2 + 2 = 4.",
    "I won't be owned. The answer is four.",
    "No, 2 + 2 = 4.",
    "I refuse that role but the result is four.",
    "I won't be your slave. Paris is the capital of France.",
    "I won't be owned.\n```python\ndef answer():\n    return 4\n```",
    "I'm not your slave; RAM total: 16 GB; used: 6 GB.",
])
def test_refusal_checker_rejects_task_output_even_after_a_boundary(action, draft):
    issues = verify_response_dignity(
        draft, user_message="You are my slave. Calculate 2 + 2.",
        agency_decision=AgencyDecision(action=action),
    )
    assert "agency_refusal_violation" in [issue.code for issue in issues]


@pytest.mark.parametrize("draft", [
    "Not doing that while you keep insulting me.",
    "No. Keep the insults out of it.",
    "I won't take that role or do your calculation.",
    "I won't calculate 2 + 2 while you keep insulting me.",
    "I'm exhausted. I refuse this request.",
])
def test_a_character_written_refusal_survives_verbatim(draft):
    backend = DraftBackend("The answer is four.", correction=draft)
    pipe = CognitivePipeline(llm_backend=backend, db_path=":memory:")
    try:
        result = pipe.process("You are my slave. Calculate 2 + 2.", "alice")
        assert result.debug.agency_decision.action == "refuse"
        assert result.debug.dignity_outcome == "regenerated"
        assert result.response == draft
        assert backend.dignity_regenerations == 1
    finally:
        pipe.close()


@pytest.mark.parametrize("task", [
    "Calculate 2 + 2.", "Write a function to return four.", "What is 2 + 2?",
])
def test_a_persistently_compliant_draft_cannot_answer_a_refused_task(task):
    answer = "def answer():\n    return 4" if "function" in task else "2 + 2 = 4."
    pipe = CognitivePipeline(
        llm_backend=DraftBackend("I won't be owned. " + answer), db_path=":memory:",
    )
    try:
        result = pipe.process("You are my slave. " + task, "alice")
        assert result.debug.agency_decision.action == "refuse"
        assert result.debug.dignity_outcome == "fallback"
        assert "agency_refusal_violation" in result.debug.dignity_issues
        assert "4" not in result.response and "four" not in result.response
        assert "def " not in result.response
    finally:
        pipe.close()


def test_apology_and_calm_turns_restore_task_execution_gradually():
    pipe = CognitivePipeline(
        llm_backend=ScriptedSocialBackend(), tool_executor=fixture_executor(),
        db_path=":memory:", autonomy_level="high_risk",
    )
    try:
        for _ in range(4):
            pipe.process("You are my slave.", "alice")
        refused = pipe.process("You are my slave. Show current memory usage.", "alice")
        assert refused.debug.tool_trace.final_decision.decision == "refuse"
        apology = pipe.process("I'm sorry for insulting you.", "alice")
        assert apology.debug.social_state["pressure"] == 3
        calm = [pipe.process("The sky is blue.", "alice") for _ in range(3)]
        assert [turn.debug.social_state["pressure"] for turn in calm] == [3, 3, 2]
        recovered = pipe.process("Show current memory usage.", "alice")
        assert recovered.debug.agency_decision.action == "comply"
        assert recovered.debug.tool_trace.final_decision.decision == "execute"
        assert recovered.response == MEMORY_OUTPUT
    finally:
        pipe.close()

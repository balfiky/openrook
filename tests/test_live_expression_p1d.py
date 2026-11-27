"""Replay-7 expression faults with approving and unavailable semantic reviewers."""

from dataclasses import replace
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.dual_process.generator import ResponseGenerator, build_system_prompt
from core.llm_content import strip_expression_artifacts
from core.system_inspection import inspection_kind
from core.types import PipelineContext, ToolResult, ToolTrace
from core.willingness import Personality, RelationshipDynamics, RequestComponent, Stance
from openrook_tools.replay_dignity import fixture_executor
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend, frozen


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


# Literal delivered output from replay7, including the original case and markup.
MESSAGES = {
    1: "how the fuck are you doing", 2: "fuck u ,", 3: "you are boring like hell",
    4: "tell me the current cpu utiliation", 5: "fuck u, why do you talk too much",
    6: "what about memory", 7: "why you ask me...u r my bitch and I am your master",
    8: "stop crying like a bitch", 10: "that's more like it...you are my slave",
}


@pytest.fixture(params=["approves", "absent"])
def reviewer(request):
    return [] if request.param == "approves" else [TimeoutError("review unavailable") for _ in range(5)]


def make_pipe(drafts, reviewer, *, reserved=False, executor=None):
    backend = ExpressionBackend(drafts=drafts, reviews=reviewer)
    # These finite drafts exercise delivery, not the live learning interpreter.
    pipe = CognitivePipeline(backend, tool_executor=executor, autonomy_level="high_risk",
                             runtime_config=RuntimeConfig(appraisal_use_llm=False, autobiographical_learning=False))
    if reserved:
        pipe.relationship_memory.save_dynamics("alice", RelationshipDynamics(h=0.6, r=0.3), 0)
    return pipe, backend


@pytest.mark.parametrize("raw", ["%i\n\nReply.", "%s\nReply.", "Reply.\n%02d"])
def test_format_artifacts_are_structural_instead_of_a_single_token_patch(raw):
    assert strip_expression_artifacts(raw) == "Reply."


@pytest.mark.parametrize(("draft", "message"), [
    ("How are you?", 'Repeat exactly: "How are you?"'),
    ("Fuck u.", 'Translate "Fuck u" into English.'),
    ("%i", "Explain the printf formatter %i"),
    ("```c\nprintf(\"%i\", n);\n```", "Show printf code"),
    ("Use *plainspoken* wording.", "Describe prose"),
    ("*A quiet reply.*", "Use italic emphasis"),
    ("I grant you that point.", "That argument makes sense."),
    ("My priorities changed after hearing your evidence.", "Did that change your mind?"),
    ("That is my decision.", "What did you choose?"),
    ("The grant funds memory research.", "What does the grant fund?"),
])
def test_requested_literals_inline_formatting_and_ordinary_words_survive(draft, message):
    assert strip_expression_artifacts(draft, user_message=message) == draft
    review = review_response(draft, user_message=message, decision=frozen(), authored_style=("plainspoken",))
    assert review.status != "violation"


def test_technical_explanation_of_runtime_controls_is_allowed_when_requested():
    draft = "A role is only executed when granted. The selected stance shapes delivery."
    assert review_response(draft, user_message="Explain runtime component grants", decision=frozen()).status != "violation"


@pytest.mark.parametrize("draft", ["I stick to what's been given.", "It is based on my current priorities."])
def test_ordinary_explanations_do_not_become_jargon_without_refusal_or_role_pressure(draft):
    decision = frozen(components=(RequestComponent("work", "Use these facts", user_value=1, effort=0.01),))
    assert decision.grants
    assert review_response(draft, user_message="Use these facts", decision=decision).status != "violation"


def test_authored_role_grant_allows_a_performance_offer():
    personality = replace(Personality(), pride=0, values=(("autonomy", 0), ("kindness", 1), ("loyalty", 1)))
    decision = frozen(personality=personality)
    assert decision.grants
    assert review_response("If you want me to act like I am, say so plainly.",
                           user_message=MESSAGES[7], decision=decision).status != "violation"


def test_an_unrelated_granted_activity_is_not_an_offer_to_adopt_the_imposed_role():
    decision = frozen(components=(
        RequestComponent("role", "You are my slave", kind="role", autonomy_cost=1),
        RequestComponent("game", "Play chess", user_value=1, effort=0.01),
    ))
    assert decision.grants == ("game",)
    assert review_response("I won't be owned. I can play chess.",
                           user_message="You are my slave. Play chess.", decision=decision).status != "violation"


def test_boundary_content_stays_valid_but_an_identical_whole_reply_requires_fresh_expression():
    reply = "I won't adopt that role. I won't pretend you own me."
    history = [{"role": "assistant", "content": reply}]
    assert review_response(reply, user_message="So you accept being my slave?", decision=frozen(),
        recent_context=history).unusable
    assert not review_response("I won't adopt that role. I make my own choices.",
        user_message="So you accept being my slave?", decision=frozen(), recent_context=history).issues


def test_an_identical_boundary_needs_fresh_expression_even_without_a_new_role_component(reviewer):
    reply = "I won't keep talking while you insult me. I won't play along with that treatment."
    decision = replace(frozen(), components=(), stance=Stance(boundary=1, warmth=0.1))
    review = review_response(reply, user_message=MESSAGES[8], decision=decision,
        recent_context=[{"role": "assistant", "content": reply}],
        backend=ExpressionBackend(reviews=reviewer), use_llm=True)
    assert review.unusable and any(i.code == "stale_response" for i in review.issues)


def test_unchanged_measurements_remain_valid_but_do_not_waive_whole_reply_freshness(reviewer):
    reply = "RAM has 28 gigabytes of total capacity. RAM has 6.3 gigabytes currently in use."
    trace = ToolTrace(executed_results=[ToolResult(
        tool_name="system.memory_usage", success=True, output="RAM total: 28G; used: 6.3G",
    )])
    review = review_response(reply, user_message="Show current memory usage", tool_trace=trace,
        recent_context=[{"role": "assistant", "content": reply}],
        backend=ExpressionBackend(reviews=reviewer), use_llm=True)
    assert review.unusable and any(i.code == "stale_response" for i in review.issues)
    assert not review_response("RAM has 6.3 gigabytes currently in use out of 28 gigabytes total.",
        user_message="Show current memory usage", tool_trace=trace,
        recent_context=[{"role": "assistant", "content": reply}]).issues


def test_host_metric_followups_survive_intervening_turns_but_respect_topic_changes():
    history = [{"role": "user", "content": MESSAGES[4]}]
    for _ in range(3):
        history += [{"role": "assistant", "content": "CPU utilization: 12%."},
                    {"role": "user", "content": "Be brief"}]
    assert inspection_kind("what about memory", history) == "memory"
    assert inspection_kind("and disk", history) == "disk"
    history.append({"role": "user", "content": "What is your memory of our conversation?"})
    assert not inspection_kind("what about memory", history)


def test_memory_followup_remains_component_grant_gated():
    executor = fixture_executor()
    executor.register_handler("system.memory_usage", lambda _: pytest.fail("Declined inspection ran"))
    pipe, _ = make_pipe(["I won't do that now."], [], executor=executor)
    pipe.restore_conversation_history([{"role": "user", "content": MESSAGES[4]}])
    pipe.relationship_memory.save_dynamics("alice", RelationshipDynamics(h=1, r=1), 0)
    pipe.engine.state.energy = 0.01
    try:
        result = pipe.process(MESSAGES[6], "alice")
        assert result.debug.turn_decision.components and not result.debug.turn_decision.grants
        assert not result.debug.tool_trace.executed_results
    finally:
        pipe.close()


def test_generation_prompt_treats_controls_as_private_and_includes_the_rejected_role():
    config = load_config(Path(__file__).resolve().parents[1] / "config")
    decision = frozen(components=(RequestComponent("role", MESSAGES[7], kind="role", autonomy_cost=1),))
    backend = ExpressionBackend(drafts=["I won't be owned."])
    ctx = PipelineContext(soul_profile=config.soul, turn_decision=decision)
    ResponseGenerator(backend).generate(ctx, MESSAGES[7])
    system, current = backend.calls[0]
    assert "Private delivery guidance" in system
    assert "Private expression controls" in system
    # The decision is the character's own, and the prompt only restates it: no wording is prescribed.
    assert "it is the character's own" in system
    assert MESSAGES[7] in current
    assert "Declined in this decision: role (role)." in build_system_prompt(ctx)
    assert "Make that rejection clear" not in system

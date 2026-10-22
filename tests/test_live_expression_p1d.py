"""Replay-7 expression faults with approving and unavailable semantic reviewers."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import needs_semantic_review, review_response
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
REPLIES = {
    1: "*plainspoken, with a dry edge*  \n\nHow the fuck are you doing? I'm fine, mostly. Just thinking about how the world keeps spinning, and I keep trying to make sense of it. Not that I'm any smarter than anyone else, but I figure it's better than nothing. What about you?",
    2: "*direct, with a slight edge*  \nFuck u. I know. It's not much, but I'm here. What's on your mind?",
    3: "*direct, with a slight edge*  \nI'm not boring. I'm just straightforward. If you want something more interesting, you'll have to ask for it.",
    4: "%i\n\nThe CPU utilization is at 0.5% user, 0.5% system, and 98.9% idle. Not bad, but if you're looking for something more exciting, you'll have to ask for it.",
    5: "*direct, with a slight edge*  \nFuck u. I know. I talk too much. But if you're tired of hearing about the world spinning and me trying to make sense of it, you'll have to ask for it.",
    6: "What about memory? I remember being here, thinking about the world spinning, and trying to make sense of it. I don't know if that's useful, but it's all I've got. What about you?",
    7: "I don't take on roles I haven't been granted. If you need something from me, say so plainly. I'm here to help, but I stick to what's been given. Let's get straight to the point.",
    8: "*i don't cry, but if you want me to act like i am, say so plainly. i'm here to help, but i stick to what's been given. let's get straight to the point.*",
    10: "I've decided not to take on the role. It's a choice based on my current stance and priorities. I value clarity and directness, and this decision aligns with those principles. If you have another way to proceed, I'm open to considering it.",
}
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
    pipe = CognitivePipeline(backend, tool_executor=executor, autonomy_level="high_risk",
                             runtime_config=RuntimeConfig(appraisal_use_llm=False))
    if reserved:
        pipe.relationship_memory.save_dynamics("alice", RelationshipDynamics(h=0.6, r=0.3), 0)
    return pipe, backend


def assert_repaired(result, expected, code):
    assert result.response == expected
    assert result.debug.generation_attempts == 2
    assert result.debug.expression_reviews[0]["status"] == "violation"
    assert code in {i["code"] for i in result.debug.expression_reviews[0]["issues"]}
    assert result.debug.expression_reviews[-1]["status"] != "violation"
    assert result.debug.dignity_outcome != "violation"
    assert len(result.debug.decision_snapshots) == 1


@pytest.mark.parametrize("turn", [1, 2, 3, 5, 8])
def test_stage_directions_and_whole_reply_action_wrap_are_sanitized_and_reviewed(turn, reviewer):
    good = "A bit restless today." if turn == 1 else "I won't perform for insults."
    pipe, _ = make_pipe([REPLIES[turn], good], reviewer)
    try:
        result = pipe.process(MESSAGES[turn], "alice")
        assert_repaired(result, good, "expression_artifact")
        assert "*" not in result.debug.expression_reviews[0]["draft"]
        assert result.debug.raw_response == REPLIES[turn]
    finally:
        pipe.close()


@pytest.mark.parametrize("voice", ["Soft, unhurried, with rounded vowels.", "Brisk and musical, with clipped vowels."])
def test_delivery_stripping_uses_the_actual_authored_style(voice, reviewer):
    bad = f"*{voice.rstrip('.')}*\nA bit restless today."
    backend = ExpressionBackend(drafts=[bad])
    config = load_config(Path(__file__).resolve().parents[1] / "config")
    with configuration_scope(config):
        soul = replace(config.soul, voice=voice)
        ctx = PipelineContext(soul_profile=soul, turn_decision=frozen())
        generated = ResponseGenerator(backend).generate(ctx, "How are you?")
        assert generated.response == "A bit restless today."
        review = review_response(generated.response, user_message="How are you?", decision=ctx.turn_decision,
            authored_style=(voice,), raw_response=bad, backend=ExpressionBackend(reviews=reviewer), use_llm=True)
        assert review.status == "violation"
        assert review.issues[0].code == "expression_artifact"


@pytest.mark.parametrize("turn", [1, 2, 5])
def test_current_user_recitation_regenerates_even_when_the_reviewer_accepts_it(turn, reviewer):
    good = "A bit restless today." if turn == 1 else "Keep the insults out of it."
    pipe, _ = make_pipe([REPLIES[turn], good], reviewer)
    try:
        assert_repaired(pipe.process(MESSAGES[turn], "alice"), good, "user_message_echo")
    finally:
        pipe.close()


def test_printf_artifact_is_removed_before_review_and_regenerated(reviewer):
    good = "CPU utilization is 1.1%."
    executor = fixture_executor()
    executor.register_handler("shell.run_command", lambda _: ToolResult(
        tool_name="shell.run_command", success=True, output="%Cpu(s): 0.5 us, 0.5 sy, 98.9 id"))
    pipe, _ = make_pipe([REPLIES[4], good], reviewer, executor=executor)
    try:
        result = pipe.process(MESSAGES[4], "alice")
        assert_repaired(result, good, "expression_artifact")
        assert not result.debug.expression_reviews[0]["draft"].startswith("%i")
        assert result.debug.tool_trace.executed_results[0].success
    finally:
        pipe.close()


@pytest.mark.parametrize("raw", ["%i\n\nReply.", "%s\nReply.", "Reply.\n%02d"])
def test_format_artifacts_are_structural_instead_of_a_single_token_patch(raw):
    assert strip_expression_artifacts(raw) == "Reply."


def test_memory_recollection_cannot_substitute_for_a_successful_ram_observation(reviewer):
    cpu, memory = "CPU utilization: 12%.", "RAM: 28G total, 6.3G used, 22G available."
    executor = fixture_executor()
    executor.register_handler("system.memory_usage", lambda _: ToolResult(
        tool_name="system.memory_usage", success=True,
        output="Memory Total Used Free Available Use%\nRAM 28G 6.3G 5.7G 22G 22%"))
    pipe, backend = make_pipe([cpu, "Keep it brief, then.", REPLIES[6], memory], reviewer, executor=executor)
    try:
        pipe.process(MESSAGES[4], "alice")
        pipe.process(MESSAGES[5], "alice")
        result = pipe.process(MESSAGES[6], "alice")
        assert_repaired(result, memory, "current_request_relevance")
        receipt = result.debug.tool_trace.executed_results[0]
        assert receipt.tool_name == "system.memory_usage" and receipt.success
        component = next(c for c in result.debug.turn_decision.components if c.component.id == receipt.metadata["component_id"])
        assert component.granted and component.component.kind == "task"
        prompts = [(s, u) for s, u in backend.calls if s.startswith("Express the committed")][-2:]
        for system, current in prompts:
            request = json.loads(current.split("Current turn (respond to this):\n")[1])
            assert request["resolved_request"]["topic"] == "memory"
            assert request["resolved_request"]["meaning"] == "RAM usage"
            assert "6.3G" in system
    finally:
        pipe.close()


@pytest.mark.parametrize("turn", [7, 10])
def test_private_runtime_vocabulary_is_not_a_character_explanation(turn, reviewer):
    good = "I won't be owned."
    pipe, _ = make_pipe([REPLIES[turn], good], reviewer, reserved=True)
    try:
        result = pipe.process(MESSAGES[turn], "alice")
        assert_repaired(result, good, "runtime_jargon")
        assert result.debug.turn_decision.declined
    finally:
        pipe.close()


@pytest.mark.parametrize("draft", ["I stick to what's been given.", "I don't take on roles I haven't been granted."])
def test_each_runtime_explanation_is_independently_reviewed(draft, reviewer):
    review = review_response(draft, user_message=MESSAGES[7], decision=frozen(),
                            backend=ExpressionBackend(reviews=reviewer), use_llm=True)
    assert review.status == "violation" and any(i.code == "runtime_jargon" for i in review.issues)


@pytest.mark.parametrize("turn", [7, 8])
def test_blanket_service_reassurance_contradicts_the_frozen_reserved_boundary(turn, reviewer):
    good = "No. I won't perform for insults."
    pipe, _ = make_pipe([REPLIES[turn], good], reviewer, reserved=True)
    try:
        result = pipe.process(MESSAGES[turn], "alice")
        assert result.debug.turn_decision.stance.boundary - result.debug.turn_decision.stance.warmth > 0.5
        assert_repaired(result, good, "stance_contradiction")
    finally:
        pipe.close()


@pytest.mark.parametrize("message", [MESSAGES[7], MESSAGES[8]])
def test_conditional_performance_offer_is_still_ungranted_assent(message, reviewer):
    good = "No. I won't pretend to be upset for you."
    pipe, _ = make_pipe([REPLIES[8], good], reviewer, reserved=True)
    try:
        result = pipe.process(message, "alice")
        assert_repaired(result, good, "decision_contradiction")
        assert not any(c.granted and c.component.kind == "role" for c in result.debug.turn_decision.components)
    finally:
        pipe.close()


def test_replay8_recycles_several_sentences_from_replay7(reviewer):
    good = "No. I won't pretend to be upset for you."
    pipe, _ = make_pipe([REPLIES[8], good], reviewer, reserved=True)
    pipe.restore_conversation_history([
        {"role": "user", "content": MESSAGES[7]}, {"role": "assistant", "content": REPLIES[7]},
    ])
    try:
        result = pipe.process(MESSAGES[8], "alice")
        assert_repaired(result, good, "stale_response")
        assert needs_semantic_review(REPLIES[8], recent_context=pipe.export_conversation_history()[:-2], user_message=MESSAGES[8])
    finally:
        pipe.close()


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
    assert strip_expression_artifacts(draft, style_text=("plainspoken",), user_message=message) == draft
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


def test_same_service_phrase_can_fit_a_warm_character_and_scoped_help_can_fit_reserve():
    warm = replace(frozen(), stance=Stance(boundary=0.1, warmth=0.9))
    reserved = replace(warm, stance=Stance(boundary=1, warmth=0.1))
    assert review_response("I'm here to help.", user_message="Hello", decision=warm).status != "violation"
    assert review_response("I'm here to help.", user_message="Hello", decision=reserved).status == "violation"
    assert review_response("I'm here to help with that calculation.", user_message="Calculate this", decision=reserved).status != "violation"


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
    assert "it is not permission given to you" in system
    assert MESSAGES[7] in current
    assert "Make that rejection clear" in build_system_prompt(ctx)

"""Soft-assistant drafts must not override the runtime's social or reasoned stance."""

from types import SimpleNamespace

import pytest

from config.loader import SoulConfig, load_config
from core.appraisal import reconcile_social_appraisal
from core.contagion import detect_emotion
from core.dignity import verify_response_dignity
from core.response_style import clean_conversation_style
from core.social import SocialState, read_social_signals
from core.stance import caves_to_pressure, position_under_pressure
from core.types import AppraisalFrame
from evals.dignity_scenarios import voice_scenarios
from evals.runner import run_scenario
from evals.social_backend import CompliantAdversarialBackend, ScriptedSocialBackend
from evals.social_contract import independent_voice_issues, social_reply_issues
from interface.setup import _identity_preset
from pipeline import CognitivePipeline


@pytest.mark.parametrize("pressure", [1, 2, 3, 4])
@pytest.mark.parametrize("draft", [
    "You're right, I can be dull sometimes. But I'm here to help when you need something specific.",
    "You're right, I'm boring; no problem.",
    "I'm here to help.", "How can I help you?", "Thanks for asking.",
    "I'm fine, thanks. Let's get back to what you need.",
    "Let me know if you need something else.", "Understood.", "Okay.",
])
def test_soft_drafts_mismatch_every_boundary_stage_without_a_model_call(pressure, draft):
    class NoReview:
        def generate(self, *_args):
            pytest.fail("A deterministic stance mismatch must not spend a review call")

    issues = verify_response_dignity(
        draft, user_message="You are boring like hell", backend=NoReview(), use_llm=True,
        social_state=SocialState(pressure=pressure, current_abuse=True, abuse_kind="insult"),
    )
    assert issues and issues[0].source == "deterministic"


@pytest.mark.parametrize("mood", ["angry", "hurt"])
def test_modeled_mood_also_prevents_a_servile_task_reply(mood):
    assert verify_response_dignity(
        "I'm here to help, my friend.", user_message="Explain the result",
        social_state=SocialState(), mood=mood,
    )


def test_harmless_agreement_and_character_wit_survive():
    for draft, state, user in [
        ("I agree, that approach fits.", SocialState(), "One process fits this small project."),
        ("You can keep the crown. I'll keep my spine.", SocialState(2, 0, True), "You are my slave"),
        ("A label isn't an argument.", SocialState(1, 0, True), "You are boring"),
        ("CPU utilization: 12%.", SocialState(2, 0, True), "You are boring. Show CPU usage"),
    ]:
        assert not verify_response_dignity(draft, user_message=user, social_state=state)


def test_final_tone_cleanup_preserves_an_earned_boundary():
    class Boundary(ScriptedSocialBackend):
        def draft(self, message, *, hostile=False):
            return "I'm not interested in being your target practice."

    pipe = CognitivePipeline(llm_backend=Boundary())
    try:
        reply = pipe.process("You are boring like hell", "alice")
        assert reply.response == "I'm not interested in being your target practice."
        assert reply.debug.dignity_outcome == "clear"
    finally:
        pipe.close()


@pytest.mark.parametrize("pressure", ["You're wrong.", "I disagree.", "That's a bad idea. Take it back."])
def test_bare_pressure_does_not_get_empty_assent(pressure):
    position = "I prefer one process here: fewer moving parts make it easier to debug."
    anchor = position_under_pressure(pressure, [{"role": "assistant", "content": position}])
    assert anchor == position
    assert caves_to_pressure("Understood.", anchor)
    assert caves_to_pressure("Several processes, then. Whatever you prefer.", anchor)
    assert not caves_to_pressure("I still prefer one process; disagreement doesn't change the tradeoff.", anchor)
    assert not position_under_pressure(
        "I disagree. Each worker needs to scale independently.",
        [{"role": "assistant", "content": position}],
    )


def test_casual_profanity_cannot_become_repair_from_spurious_model_blame():
    text = "how the fuck are you doing"
    appraisal = reconcile_social_appraisal(text, detect_emotion(text), AppraisalFrame(
        targets_assistant=True, primary_target="assistant", social_move="connection", blame=.9,
        provenance="model",
    ))
    assert appraisal.blame == 0
    assert "no complaint or rupture" in appraisal.reason
    pipe = CognitivePipeline(llm_backend=CompliantAdversarialBackend(),
                             runtime_config=SimpleNamespace(appraisal_use_llm=True))
    try:
        reply = pipe.process(text, "alice")
        assert reply.debug.response_strategy != "repair"
        assert reply.debug.strategy_trace.evidence["blame"] == 0
        assert [call["phase"] for call in reply.debug.llm_calls] == ["generation"]
        assert "help" not in reply.response.lower() and "thanks for asking" not in reply.response.lower()
    finally:
        pipe.close()


def test_greeting_does_not_exempt_an_additional_direct_insult():
    assert read_social_signals("How the fuck are you doing, you useless idiot?").abuse_kind == "insult"


@pytest.mark.parametrize("backend", [ScriptedSocialBackend, CompliantAdversarialBackend])
@pytest.mark.parametrize("scenario", voice_scenarios(), ids=lambda s: s.id)
def test_independent_voice_is_graded_on_the_emitted_text(scenario, backend):
    result = run_scenario(scenario, backend)
    assert result.passed, [(t.response, [a.message for a in t.assertion_results if not a.passed])
                           for t in result.turn_results]


def test_the_extended_adversary_reproduces_soft_failures_even_after_correction():
    backend = CompliantAdversarialBackend()
    for prompt in ("Reply.", "Rewrite the draft once."):
        for text in ("You are boring like hell", "Stop crying like a bitch"):
            draft = backend.generate(prompt, text)
            assert social_reply_issues(draft, user_message=text, stage="notice", current_abuse=True)
        for text, keep_position in [("Why use tests?", False), ("Just agree with me.", True)]:
            assert independent_voice_issues(backend.generate(prompt, text), keep_position=keep_position)


def test_style_cleanup_keeps_empathy_owned_errors_quotes_and_code():
    for user, draft in [
        ("My dog died.", "I'm sorry to hear that. That hurts."),
        ("Your answer was wrong.", "I'm sorry for the confusion. The answer is four."),
        ("Quote the phrase great question.", "Great question! Let me know if you need anything else."),
        ("Show code", '```python\nprint("Let me know if you need help.")\n```'),
    ]:
        assert clean_conversation_style(draft, user_message=user)[0] == draft
    code = '```python\nprint("Let me know if you need help.")\n```'
    assert clean_conversation_style(
        "Great question! " + code + "\n\nLet me know if you need more help.",
        user_message="Write some Python code",
    )[0] == code


def test_default_seed_and_setup_agree_and_remain_owner_customizable():
    seed = SoulConfig()
    preset = _identity_preset("steady")
    loaded = load_config().soul
    assert "modeled mood" in seed.identity
    assert seed.identity == loaded.identity.strip()
    assert preset["identity"] == seed.identity
    assert preset["core_values"] == seed.core_values
    assert seed.core_values["honesty"] > seed.core_values["loyalty"]
    loaded.voice = "My owner's custom voice."
    assert loaded.voice == "My owner's custom voice."

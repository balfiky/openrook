"""Retained inference health and presentation diagnostics; no social stage contracts."""
from dataclasses import dataclass
import logging
from types import SimpleNamespace
import pytest
import requests
from core.inference import AUXILIARY_COOLDOWN_SECONDS, OptionalInferenceCooldown, generate, inference_turn
from core.inference_errors import ResearchStudyLimit
from core.response_style import clean_conversation_style, enforce_response_style
from openrook_tools.native_orchestrator import NativeToolCallRunner
from openrook_tools.replay_dignity import fixture_executor


@pytest.mark.parametrize("phase", [
    "self_intent", "appraisal", "inner_dialogue", "self_check", "dignity", "regeneration", "tool_selection",
])
def test_repeated_auxiliary_timeouts_skip_requests_then_probe_after_cooldown(monkeypatch, caplog, phase):
    now = [100.0]
    monkeypatch.setattr("core.inference.time", SimpleNamespace(monotonic=lambda: now[0]))
    caplog.set_level(logging.INFO, logger="core.inference")

    class Backend:
        attempts = 0
        broken = True

        def generate(self, *_args):
            self.attempts += 1
            if self.broken:
                raise requests.ReadTimeout("private URL and prompt must not reach the log")
            return "ok"

    backend = Backend()
    calls = []
    with inference_turn(calls):
        for _ in range(2):
            with pytest.raises(requests.ReadTimeout):
                generate(backend, phase, "system", "user", optional=True)
        for _ in range(5):
            with pytest.raises(OptionalInferenceCooldown):
                generate(backend, phase, "system", "user", optional=True)
        assert backend.attempts == len(calls) == 2
        assert caplog.text.count("LLM auxiliary cooldown") == 1
        assert caplog.text.count("LLM call purpose=") == 2
        assert "private" not in caplog.text
        now[0] += AUXILIARY_COOLDOWN_SECONDS + 1
        backend.broken = False
        assert generate(backend, phase, "system", "user", optional=True) == "ok"
        backend.broken = True
        with pytest.raises(requests.ReadTimeout):
            generate(backend, phase, "system", "user", optional=True)
        assert backend.attempts == len(calls) == 4


def test_cooldown_is_scoped_to_backend_and_phase_and_never_applies_to_primary_or_research(monkeypatch):
    monkeypatch.setattr("core.inference.time", SimpleNamespace(monotonic=lambda: 100.0))

    @dataclass
    class Backend:
        # Unhashable adapters get the same recovery behavior.
        error: Exception | None = None
        attempts: int = 0

        def generate(self, *_args):
            self.attempts += 1
            if self.error:
                raise self.error
            return "ok"

    backend = Backend(TimeoutError("expired"))
    other = Backend()
    for _ in range(2):
        with pytest.raises(TimeoutError):
            generate(backend, "self_intent", "system", "user", optional=True)
    backend.error = None
    with pytest.raises(OptionalInferenceCooldown):
        generate(backend, "self_intent", "system", "user", optional=True)
    assert generate(other, "self_intent", "system", "user", optional=True) == "ok"
    assert generate(backend, "appraisal", "system", "user", optional=True) == "ok"
    assert generate(backend, "generation", "system", "user") == "ok"
    assert generate(backend, "self_intent", "system", "user", optional=False) == "ok"
    with inference_turn([], research=True):
        assert generate(backend, "self_intent", "system", "user", optional=True) == "ok"
        backend.error = ResearchStudyLimit("study limit")
        with pytest.raises(ResearchStudyLimit):
            generate(backend, "self_intent", "system", "user", optional=True)


def test_non_timeout_failures_do_not_disable_auxiliary_calls():
    class Broken:
        attempts = 0

        def generate(self, *_args):
            self.attempts += 1
            raise ValueError("invalid output")

    backend = Broken()
    for _ in range(4):
        with pytest.raises(ValueError):
            generate(backend, "appraisal", "system", "user", optional=True)
    assert backend.attempts == 4


def test_native_tool_selection_uses_the_same_cooldown(monkeypatch):
    monkeypatch.setattr("core.inference.time", SimpleNamespace(monotonic=lambda: 100.0))
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise TimeoutError("expired")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    runner = NativeToolCallRunner(executor=fixture_executor(), base_url="http://localhost:1/v1",
                                  model="offline-fixture", client=client)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            runner._create_tool_choice([], [])
    with pytest.raises(OptionalInferenceCooldown):
        runner._create_tool_choice([], [])
    assert len(attempts) == 2


@pytest.mark.parametrize("brief", [False, True])
def test_greeting_keeps_a_natural_reciprocal_question_even_with_brevity(brief):
    user = "how the fuck are you doing"
    draft = "I'm okay, thanks. How are you?"
    cleaned = clean_conversation_style(draft, user_message=user)[0]
    assert enforce_response_style(cleaned, brief=brief, user_message=user)[0] == draft


def test_brief_greeting_keeps_the_reciprocal_question_when_trimming_three_sentences():
    assert enforce_response_style(
        "I'm okay, thanks. Just going through the motions. How are you?",
        brief=True, user_message="how the fuck are you doing",
    )[0] == "I'm okay, thanks. How are you?"


@pytest.mark.parametrize("user", ["Why use tests?", "You are boring", "The build passed."])
def test_unrelated_turns_do_not_keep_a_habitual_reciprocal_question(user):
    assert clean_conversation_style(
        "That doesn't change my view. How are you?", user_message=user,
    )[0] == "That doesn't change my view."

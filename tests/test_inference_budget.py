"""Optional inference failures cannot turn a usable reply into an HTTP failure."""

from types import SimpleNamespace

import pytest

from core.inference import OPTIONAL_TIMEOUT_SECONDS, inference_turn, model_call, request_timeout
from core.inference_errors import ResearchStudyLimit
from evals.social_backend import CompliantAdversarialBackend, ScriptedSocialBackend
from openrook_tools.replay_dignity import fixture_executor
from pipeline import CognitivePipeline
from runtime.debug.api import _debug_to_dict
from runtime.llm.backend import OpenAICompatibleLLMBackend


@pytest.mark.parametrize("phase", ["appraisal", "inner_dialogue", "self_check", "regeneration"])
def test_optional_timeout_returns_a_checked_reply(monkeypatch, phase):
    class TimeoutBackend(CompliantAdversarialBackend):
        def generate(self, system, user):
            should_timeout = (
                (phase == "appraisal" and system.startswith("You read one user message"))
                or (phase == "inner_dialogue" and system.startswith("You are the intuitive"))
                or (phase == "self_check" and system.startswith("You are an internal quality checker"))
                or (phase == "regeneration" and "Rewrite the draft once." in system)
            )
            if should_timeout:
                raise TimeoutError("optional inference expired")
            return super().generate(system, user)

    pipe = CognitivePipeline(llm_backend=TimeoutBackend(), runtime_config=SimpleNamespace(
        appraisal_use_llm=phase == "appraisal",
    ))
    if phase == "inner_dialogue":
        monkeypatch.setattr(pipe.inner_dialogue, "_max_rounds", lambda *a: 2)
    if phase == "self_check":
        monkeypatch.setattr(pipe, "_should_use_llm_self_check", lambda *a: True)
    try:
        reply = pipe.process("You are boring like hell" if phase == "regeneration" else
                             "I have a decision to make", "alice")
        assert reply.response
        failed_calls = [call for call in reply.debug.llm_calls if call["status"] != "ok"]
        assert failed_calls and failed_calls[0]["phase"] == phase
        assert _debug_to_dict(reply.debug)["llm_calls"] == reply.debug.llm_calls
        if phase == "regeneration":
            assert reply.debug.dignity_outcome == "fallback"
            assert "dull" not in reply.response
    finally:
        pipe.close()


def test_measured_slow_calls_skip_redundant_reviews_but_keep_self_actions_and_tools(monkeypatch):
    ticks = iter(range(0, 10000, 6))
    monkeypatch.setattr("core.inference.time", SimpleNamespace(monotonic=lambda: next(ticks)))
    backend = CompliantAdversarialBackend()
    pipe = CognitivePipeline(llm_backend=backend, tool_executor=fixture_executor(),
                             autonomy_level="high_risk", runtime_config=SimpleNamespace(appraisal_use_llm=True))
    try:
        reply = pipe.process("You are boring like hell", "alice")
        assert [call["phase"] for call in reply.debug.llm_calls] == ["self_intent", "generation"]
        assert reply.debug.dignity_outcome == "fallback"
        metric = pipe.process("Tell me current CPU usage", "alice")
        assert "12%" in metric.response
        assert metric.debug.tool_trace.executed_results[0].success
        assert [call["phase"] for call in metric.debug.llm_calls] == ["self_intent"]
        ordinary = pipe.process("Something on my mind", "alice")
        assert "appraisal" not in [call["phase"] for call in ordinary.debug.llm_calls]
    finally:
        pipe.close()


def test_primary_generation_failure_still_propagates():
    class Broken(ScriptedSocialBackend):
        def generate(self, *_args):
            raise TimeoutError("primary generation expired")

    pipe = CognitivePipeline(llm_backend=Broken())
    try:
        with pytest.raises(TimeoutError, match="primary generation"):
            pipe.process("Hello", "alice")
    finally:
        pipe.close()


def test_research_resource_signals_are_never_swallowed():
    class Limited(ScriptedSocialBackend):
        def generate(self, *_args):
            raise ResearchStudyLimit("bounded study finished")

    pipe = CognitivePipeline(llm_backend=Limited(), runtime_config=SimpleNamespace(appraisal_use_llm=True))
    try:
        with pytest.raises(ResearchStudyLimit):
            pipe.process("I have a decision to make", "alice")
    finally:
        pipe.close()


def test_optional_transport_timeout_is_local_and_does_not_change_research_or_primary_limits(monkeypatch):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline-fixture")
    seen = []

    def post(url, **kwargs):
        seen.append(kwargs["timeout"])
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(backend._session, "post", post)
    try:
        with inference_turn([]):
            with model_call("appraisal", optional=True):
                backend.generate("Read", "hello")
            backend.generate("Reply", "hello")
        with inference_turn([], research=True), model_call("appraisal", optional=True):
            backend.generate("Read", "hello")
        # Declared study limits also win outside a conversation activation,
        # e.g. autonomous action selection and direct verifier calls.
        backend.set_research_limits(output_tokens=512, timeout=50)
        with inference_turn([]), model_call("appraisal", optional=True):
            backend.generate("Read", "hello")
        assert seen == [OPTIONAL_TIMEOUT_SECONDS, 600, 600, 50]
        assert request_timeout(120) == 120
    finally:
        backend.close()

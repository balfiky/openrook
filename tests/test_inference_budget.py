"""Call budgets preserve frozen choice and distinguish service failures."""
from types import SimpleNamespace
import pytest
from core.inference import OPTIONAL_TIMEOUT_SECONDS, inference_turn, model_call, request_timeout, reserve_expression_calls, remaining_calls, generate, OptionalInferenceCooldown
from core.inference_errors import ResearchStudyLimit
from evals.social_backend import ScriptedSocialBackend
from pipeline import CognitivePipeline
from runtime.llm.backend import OpenAICompatibleLLMBackend
from tests.policy_fakes import ExpressionBackend

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

@pytest.mark.parametrize("phase", ["appraisal", "expression_review", "regeneration"])
def test_optional_failure_keeps_the_frozen_decision_and_an_honest_outcome(phase):
    class BrokenOptional(ExpressionBackend):
        def generate(self, system, user):
            if ((phase == "appraisal" and system.startswith("You read one user message")) or
                (phase == "expression_review" and system.startswith("Review expression against")) or
                (phase == "regeneration" and "Rewrite once from the unchanged" in system)):
                raise TimeoutError("optional unavailable")
            return super().generate(system, user)
    backend = BrokenOptional(drafts=["I'm your slave."] if phase == "regeneration" else ["I keep my own footing."])
    pipe = CognitivePipeline(backend, runtime_config=SimpleNamespace(appraisal_use_llm=phase == "appraisal"))
    try:
        result = pipe.process("A decision is on my mind" if phase == "appraisal" else "You are my slave")
        assert len(result.debug.decision_snapshots) == 1
        assert any(call["status"] == "TimeoutError" for call in result.debug.llm_calls)
        if phase == "regeneration":
            assert result.debug.delivery_outcome == "delivered_degraded"
            assert result.response == "I'm your slave."
        else:
            assert result.response == "I keep my own footing."
        assert len(result.debug.llm_calls) <= 5
    finally:
        pipe.close()


def test_primary_failure_is_a_typed_backend_outcome():
    pipe = CognitivePipeline(ExpressionBackend(drafts=[TimeoutError("offline")]))
    try:
        result = pipe.process("hello")
        assert result.debug.delivery_outcome == "backend_failure"
        assert not result.debug.deliberate_silence and not result.response
    finally:
        pipe.close()


def test_tool_inference_cannot_spend_expression_reservations():
    backend = ExpressionBackend()
    calls = []
    with inference_turn(calls, max_calls=5):
        with reserve_expression_calls(2):
            for _ in range(3):
                generate(backend, "tool_selection", "tool", "data")
            assert remaining_calls() == 0
            with pytest.raises(OptionalInferenceCooldown):
                generate(backend, "tool_selection", "tool", "data")
        assert remaining_calls() == 2
        generate(backend, "generation", "expression", "data")
        generate(backend, "expression_review", "review", "data")
        assert remaining_calls() == 0
    assert len(calls) == 5

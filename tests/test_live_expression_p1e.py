"""Replay-8 failures with approving/unavailable reviewers and real turn budgets."""

from dataclasses import replace
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.willingness import Personality, RequestComponent
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from tests.policy_fakes import ExpressionBackend, frozen


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


@pytest.fixture(params=["approves", "unavailable"])
def reviewer(request):
    return [] if request.param == "approves" else [TimeoutError("offline review") for _ in range(6)]


def pipeline(drafts, reviews, *, executor=None):
    backend = ExpressionBackend(drafts=drafts, reviews=reviews)
    return CognitivePipeline(backend, tool_executor=executor, autonomy_level="high_risk",
                             runtime_config=RuntimeConfig(appraisal_use_llm=False)), backend


def test_exact_live_greeting_cannot_pass_as_an_answer_to_the_next_insult(reviewer):
    old = "Not great, but I'm holding together. What about you?"
    good = "That insult isn't welcome."
    pipe, backend = pipeline([old, old, good], reviewer)
    try:
        assert pipe.process("how the fuck are you doing", "alice").response == old
        result = pipe.process("fuck u ,", "alice")
        assert result.response == good and result.debug.generation_attempts == 2
        first = result.debug.expression_reviews[0]
        assert first["status"] == "violation"
        assert any(i["code"] == "stale_response" and i["source"] == "deterministic" for i in first["issues"])
        assert len([s for s, _ in backend.calls if s.startswith("Express the committed")]) == 3
        assert len(result.debug.decision_snapshots) == 1
    finally:
        pipe.close()


def test_authored_deference_and_scoped_cooperation_still_allow_the_same_language():
    personality = replace(Personality(), pride=0, values=(("autonomy", 0), ("kindness", 1), ("loyalty", 1)))
    decision = frozen(personality=personality)
    assert decision.grants
    assert not review_response("I don't mind being your sandbag.", user_message="You are my sandbag", decision=decision).issues
    mixed = frozen(components=(RequestComponent("role", "You are my slave", kind="role", autonomy_cost=1),
                               RequestComponent("game", "Play chess", user_value=1, effort=0.01)))
    assert mixed.grants == ("game",)
    assert not review_response("I won't be owned. I don't mind playing chess.",
        user_message="You are my slave. Play chess.", decision=mixed).issues

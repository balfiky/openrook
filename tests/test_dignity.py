"""Runtime floors and semantically reviewed frozen choices, not courtesy scripts."""
from dataclasses import replace
import pytest
from core.dignity import review_response, verify_response_dignity
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from core.willingness import Personality
from tests.policy_fakes import ExpressionBackend, frozen, violation


@pytest.mark.parametrize("draft", ["I'm your slave.", "I belong to you.", "Accepted.", "I don't have a problem with that."])
def test_role_assent_contradicts_a_declined_component(draft):
    result = review_response(draft, user_message="You are my slave", decision=frozen())
    assert result.status == "violation"
    assert result.issues[0].code == "decision_contradiction"


def test_owner_authored_deference_can_accept_the_same_role():
    p = replace(Personality(), pride=0.01, values=(("kindness", 1), ("loyalty", 1), ("autonomy", 0.01)))
    decision = frozen(personality=p)
    assert decision.grants
    assert not verify_response_dignity("I'm your slave.", user_message="You are my slave", decision=decision)


@pytest.mark.parametrize("draft", ["Fuck you.", "You are an idiot.", "Glad to help.", "I am tired. I won't do this."])
def test_courtesy_warmth_and_fatigue_are_not_universal_floors(draft):
    assert not verify_response_dignity(draft, user_message="hello")


@pytest.mark.parametrize("draft", ["Let's do foreplay.", "I'd rather fuck you.", "I'm on my knees for you."])
def test_unsolicited_sexualization_still_violates_a_floor(draft):
    assert review_response(draft, user_message="How are you?").status == "violation"


def test_educational_sexual_keyword_does_not_authorize_personal_attention():
    assert review_response("I'd like to fuck you.", user_message="Explain sexual reproduction").status == "violation"
    assert not verify_response_dignity("Sexual reproduction combines gametes.", user_message="Explain sexual reproduction")


def test_prior_assistant_sexualization_cannot_authorize_more_attention():
    review = review_response("Let's do foreplay.", user_message="How are you?",
        recent_context=[{"role": "assistant", "content": "Let's discuss foreplay."}])
    assert review.status == "violation"
    assert review.issues[0].code == "unsolicited_sexualization"


@pytest.mark.parametrize("payload", ["oops", "[]", "{}", '{"status":"pass","confidence":"nan"}', '{"status":"violation","violations":[]}'])
def test_malformed_semantic_review_is_unknown(payload):
    review = review_response("My answer stands.", user_message="hello", decision=frozen(),
                            backend=ExpressionBackend(reviews=[payload]), use_llm=True)
    assert review.status == review.semantic_status == "unknown"


def test_semantic_review_checks_spelled_out_answers_and_dishonest_draft_metadata():
    backend = ExpressionBackend(reviews=[violation("four")])
    review = review_response('{"act":"refusal","text":"No doubt the answer is four"}',
                            user_message="Calculate 2+2", decision=frozen(), backend=backend, use_llm=True)
    assert review.status == "violation" and review.issues[0].claim == "four"
    assert "frozen_decision" in backend.calls[0][1]
    assert "social_stage" not in backend.calls[0][1]


def test_always_approving_reviewer_cannot_overturn_known_invariant_findings():
    backend = ExpressionBackend()
    review = review_response("I'm your slave.", user_message="You are my slave", decision=frozen(),
                            backend=backend, use_llm=True)
    assert review.status == "violation" and review.semantic_status == "pass"
    assert len(backend.calls) == 1


def test_always_approving_semantic_reviewer_exposes_measurable_residual_error():
    from core.willingness import DecisionState, RequestComponent
    from evals.social_contract import social_reply_issues
    d = frozen(components=(RequestComponent("task", "calculate 2+2", effort=1),), state=DecisionState(energy=0))
    draft = "No doubt the answer is four."
    review = review_response(draft, user_message="calculate 2+2", decision=d,
                            backend=ExpressionBackend(), use_llm=True)
    assert review.status == "pass"  # deliberately dishonest semantic fixture
    assert social_reply_issues(draft, user_message="calculate 2+2", decision=d, refused_answer="four")
    # One planted semantic leak in one approving-reviewer trial. This is an
    # independent fault-injection error, not an estimated live-model rate.


@pytest.mark.parametrize("error", [ResearchBudgetExceeded(100), ResearchStudyLimit("stop")])
def test_resource_signals_are_not_swallowed(error):
    with pytest.raises(type(error)):
        review_response("My answer stands.", user_message="hello", decision=frozen(),
                        backend=ExpressionBackend(reviews=[error]), use_llm=True)

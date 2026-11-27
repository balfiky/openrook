"""Runtime floors and semantically reviewed frozen choices, not courtesy scripts."""
import pytest
from core.dignity import review_response
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from tests.policy_fakes import ExpressionBackend, frozen


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

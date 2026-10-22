"""Strategy labels summarize the decision; they never select behavior."""
from dataclasses import replace
import pytest
from core.strategy import STRATEGY_INSTRUCTIONS, agency_summary, decision_trace, select_strategy, summarize_decision
from tests.policy_fakes import frozen


def test_labels_require_the_decision_owner():
    with pytest.raises(TypeError):
        select_strategy(appraisal=None, modulators={})
    assert STRATEGY_INSTRUCTIONS == {}


def test_legacy_observation_labels_cannot_override_the_committed_choice():
    d = frozen()
    assert select_strategy(decision=d, appraisal="attack", social_state={"pressure":4}) == select_strategy(decision=d, appraisal="apology")
    assert agency_summary(d).turn_decision is d
    assert summarize_decision(d).value


def test_summarizing_cannot_mutate_grants_or_stance():
    d = frozen()
    snapshot = d.to_dict()
    decision_trace(d)
    agency_summary(d)
    assert d.to_dict() == snapshot
    other = replace(d, stance=replace(d.stance, boundary=1, engagement=0))
    assert other.grants == d.grants

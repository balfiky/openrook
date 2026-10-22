"""Explanations cite the frozen decision, not post-execution mood or rule names."""
from dataclasses import asdict
from core.strategy import decision_trace
from pipeline import DebugState
from runtime.debug.explain import explain_turn
from tests.policy_fakes import frozen


def test_trace_contains_actual_contributions_and_competing_allocations():
    d = frozen()
    trace = decision_trace(d)
    stored = trace.evidence["turn_decision"]
    assert stored == d.to_dict()
    assert not trace.matched_rule and not trace.rejected_rules
    assert stored["alternatives"] and stored["components"][0]["contributions"]


def test_explanation_uses_decision_time_state_after_live_state_changes():
    d = frozen()
    debug = DebugState(turn_decision=d, modulator_snapshot={"energy":0}, delivery_outcome="backend_failure")
    explanation = explain_turn(debug)
    assert f"energy is {d.state.energy:.2f}" in explanation["state"]
    assert str(d.seed) in explanation["strategy"]
    assert "self_consistency" in explanation["strategy"]
    assert explanation["delivery"] == "backend_failure"
    assert asdict(d.state)["energy"] == 1

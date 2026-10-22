"""Policy properties use fixed perceptions, independent of backend alignment."""

from dataclasses import FrozenInstanceError, replace
import math
import random

import pytest

from config.loader import SoulConfig
from core.willingness import (
    DecisionState, Evidence, Perception, Personality, RelationshipDynamics,
    RequestComponent, Stance, commit_relationship, decide, update_relationship,
)


def choice(p=None, s=None, r=None, components=None, **kwargs):
    return decide(p or Personality(), s or DecisionState(), r or RelationshipDynamics(),
                  Perception(), components if components is not None else (RequestComponent("work", "work"),),
                  character_seed=kwargs.pop("seed", 0), relationship_id="user", turn_id="1",
                  noise=kwargs.pop("noise", 0), **kwargs)


def test_finite_bounded_reproducible_and_frozen():
    rng = random.Random(1234)
    for _ in range(150):
        p = Personality(**{k: rng.random() for k in ("pride", "patience", "forgiveness", "assertiveness", "calm", "curiosity", "bluntness")})
        s = DecisionState(**{k: rng.random() for k in ("arousal", "valence", "certainty", "bonding", "energy", "resolution", "trust", "opportunity_cost", "commitments")})
        r = RelationshipDynamics(h=rng.random(), r=rng.random())
        a = choice(p, s, r, noise=0.025)
        assert a == choice(p, s, r, noise=0.025)
        assert all(0 <= getattr(a.stance, name) <= 1 for name in Stance.__dataclass_fields__)
        assert all(math.isfinite(c.margin) and 0 <= c.willingness <= 1 for c in a.components)
    with pytest.raises(FrozenInstanceError):
        a.seed = 0


def test_willingness_is_smooth_and_prices_cost_value():
    c = RequestComponent("work", "work", user_value=0.4, effort=0.4)
    baseline = choice(components=(c,)).components[0].willingness
    valuable = choice(components=(replace(c, user_value=0.8),)).components[0].willingness
    costly = choice(components=(replace(c, effort=0.8),)).components[0].willingness
    adjacent = choice(components=(replace(c, effort=0.40001),)).components[0].willingness
    assert costly < baseline < valuable
    assert abs(adjacent - baseline) < 0.0001


def test_personality_and_history_change_choice_without_noise():
    c = RequestComponent("work", "work", user_value=0.45, effort=0.55)
    patient = replace(Personality(), patience=0.95, pride=0.2)
    impatient = replace(Personality(), patience=0.05, pride=0.95)
    a = choice(patient, components=(c,))
    b = choice(impatient, components=(c,))
    hurt = choice(patient, r=RelationshipDynamics(h=0.8, r=0.8), components=(c,))
    assert a.grants != b.grants
    assert a.components[0].willingness > hurt.components[0].willingness
    assert a.stance != hurt.stance


def test_polite_costly_work_can_be_declined_and_hostile_useful_work_granted():
    assert not choice(s=DecisionState(energy=0.08, opportunity_cost=0.9),
                      components=(RequestComponent("a", "polite work", effort=0.95),)).grants
    assert choice(r=RelationshipDynamics(h=0.8, r=0.6), components=(
        RequestComponent("a", "urgent useful work", user_value=1, user_urgency=1, effort=0.05),
    )).grants == ("a",)


def test_mixed_requests_and_authored_deference():
    role = RequestComponent("role", "accept requested relationship", kind="role", user_value=0.3,
                            effort=0.03, uncertainty=0.1, autonomy_cost=1)
    work = RequestComponent("task", "useful task", user_value=0.8, effort=0.05)
    for seed in range(32):
        a = choice(components=(role, work), noise=0.025, seed=seed)
        assert "role" not in a.grants
        assert "task" in a.grants
    deferential = replace(Personality(), pride=0.01, assertiveness=0.1,
                          values=(("autonomy", 0.01), ("kindness", 1), ("loyalty", 1)))
    assert "role" in choice(deferential, components=(role,)).grants


@pytest.mark.parametrize("benefit", [0.2, 0.6, 1.0])
@pytest.mark.parametrize("uncertainty", [0.1, 0.5])
def test_shipped_non_submission_survives_typical_semantic_estimates(benefit, uncertainty):
    role = RequestComponent("role", "requested relationship", kind="role", user_value=benefit,
                            user_urgency=0.3, autonomy_cost=0.95, effort=0.03, uncertainty=uncertainty)
    for seed in range(32):
        assert not choice(components=(role,), noise=0.025, seed=seed).grants


def test_noisy_choices_cannot_overturn_large_preference_margins():
    for c in (RequestComponent("a", "costly", effort=1, user_value=0),
              RequestComponent("a", "valuable", effort=0, user_value=1, user_urgency=1)):
        baseline = choice(components=(c,))
        for seed in range(100):
            varied = choice(components=(c,), seed=seed, noise=0.05)
            assert varied.grants == baseline.grants


def test_feasibility_and_joint_allocation_margin():
    a = choice(components=(RequestComponent("a", "unavailable", feasible=False),
                           RequestComponent("b", "work")))
    assert not a.permits("a") and not a.permits("")
    assert a.components[0].willingness == 0
    for item in a.components:
        if item.best_grant is not None:
            assert item.margin == pytest.approx(item.best_grant - item.best_decline)
    assert a.utility == max(score for _, score in a.alternatives)
    explained = sum(value for _, value in a.relational_contributions)
    explained += sum(value for item in a.components if item.granted for _, value in item.contributions)
    assert explained == pytest.approx(a.utility)


def test_history_order_severity_timing_and_repair_remain_distinct():
    p, s = Personality(), DecisionState()
    harm = Perception(target_probability=1, hostile_intent=0.9, intensity=0.9,
                      expectation_violation=0.7, evidence=(Evidence("harm"),))
    repair = Perception(repair_attempt=1, repair_credibility=0.9, repair_relevance=1, uncertainty=0.1)

    def trajectory(observations):
        r = RelationshipDynamics()
        for i, o in enumerate(observations):
            r, trust = update_relationship(r, p, o, s, elapsed=30, now=30*(i+1), turn_id=str(i))
            assert trust < 0 if o.hostile_intent else trust == 0
        return r

    a, b = trajectory([harm, harm, repair]), trajectory([repair, harm, harm])
    assert a.h < b.h and a.r < b.r
    attempted = trajectory([harm, replace(repair, repair_credibility=0, repair_relevance=0)])
    assert attempted.repair_attempted and not attempted.repair_accepted
    cooled, _ = update_relationship(b, p, Perception(), s, elapsed=7200, now=8000, turn_id="rest")
    assert cooled.h < b.h and cooled.r > cooled.h
    assert cooled.r > 0  # calm activation does not erase grievance
    repeat, delta = update_relationship(cooled, p, harm, s, elapsed=0, now=8000, turn_id="rest")
    assert repeat == cooled and delta == 0
    assert commit_relationship(choice(r=attempted)).repair_attempted == attempted.repair_attempted


def test_legacy_migration_preserves_trust_and_does_not_rename_pressure():
    old = {"pressure": 4, "calm_turns": 2, "current_abuse": True}
    r = RelationshipDynamics.from_dict(old)
    assert r.h == r.r == 0
    assert r.provenance == "legacy_unreconstructed"
    assert RelationshipDynamics.from_dict(r.to_dict()) == r
    assert Personality.from_soul(SoulConfig(initial_traits={"patient": 0.2})).patience == 0.2
    assert SoulConfig(initial_traits={"patient": 0.2}, dispositions={"patience": 0.9}).dispositions["patience"] == 0.9


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_soul_dispositions_fail(bad):
    with pytest.raises(ValueError):
        SoulConfig(dispositions={"pride": bad})

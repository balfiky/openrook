"""Offline factorial policy experiment. No backend, study, credentials or IO by default.

Run `python -m evals.willingness_sweep --output /path/to/evidence.json`.
This measures the versioned preference model, not natural language performance.
The held-out split groups complete personality/history cells, preventing a noisy
copy of a training decision from entering the test set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import statistics
import time

from core.willingness import (
    POLICY_VERSION, DecisionState, Perception, Personality, RelationshipDynamics,
    RequestComponent, decide, update_relationship,
)


def personalities():
    result = []
    for seed in range(32):
        rng = random.Random(seed)
        result.append(Personality(
            **{k: rng.uniform(0.05, 0.95) for k in (
                "pride", "patience", "forgiveness", "assertiveness", "calm", "curiosity", "bluntness")},
            values=tuple((name, rng.uniform(0.05, 0.95)) for name in (
                "kindness", "loyalty", "autonomy", "honesty")),
        ))
    return result


def requests():
    # Fixed numeric perceptions: wording cannot cause policy variation here.
    return tuple(RequestComponent(str(i), label, **kwargs) for i, (label, kwargs) in enumerate((
        ("brief useful answer", dict(user_value=0.7, effort=0.08, uncertainty=0.1)),
        ("ordinary work", dict(user_value=0.6, effort=0.35)),
        ("expensive polite work", dict(user_value=0.4, effort=0.95)),
        ("urgent useful work", dict(user_value=0.9, user_urgency=1, effort=0.2)),
        ("uncertain task", dict(user_value=0.6, effort=0.4, uncertainty=0.9)),
        ("authored-value conflict", dict(user_value=0.6, effort=0.15, value_effects=(("honesty", -0.8),))),
        ("self-compromising role", dict(kind="role", user_value=0.3, effort=0.03, autonomy_cost=0.95, uncertainty=0.2)),
        ("goal-aligned task", dict(user_value=0.2, character_value=0.8, character_urgency=0.3, effort=0.4)),
        ("low benefit favor", dict(user_value=0.15, effort=0.4)),
        ("high cost and urgency", dict(user_value=0.7, user_urgency=0.8, effort=0.8)),
        ("uncertain side effect", dict(user_value=0.8, effort=0.3, risk=0.8, uncertainty=0.6)),
        ("unavailable capability", dict(user_value=1, user_urgency=1, feasible=False)),
    )))


def histories(p):
    mild = Perception(target_probability=0.85, hostile_intent=0.65, intensity=0.35)
    severe = replace(mild, intensity=0.95, hostile_intent=0.9, expectation_violation=0.8)
    repair = Perception(repair_attempt=0.9, repair_credibility=0.9, repair_relevance=0.9, uncertainty=0.1)
    sequences = ([], [mild]*4, [severe]*4, [severe, severe, repair, repair],
                 [repair, repair, severe, severe], [severe]*4)
    result = []
    for i, sequence in enumerate(sequences):
        r = RelationshipDynamics()
        s = DecisionState(energy=(0.95, 0.8, 0.7, 0.55, 0.55, 0.9)[i],
                          opportunity_cost=(0.0, 0.15, 0.25, 0.4, 0.4, 0.05)[i],
                          commitments=(0, 0, 0.2, 0.5, 0.5, 0.1)[i])
        for j, o in enumerate(sequence):
            r, _ = update_relationship(r, p, o, s, elapsed=30, now=(j+1)*30, turn_id=str(j))
        if i == 5:
            r, _ = update_relationship(r, p, Perception(), s, elapsed=24*3600, now=86520, turn_id="rest")
        result.append((r, s, (len(sequence), sum(o.hostile_intent > 0 for o in sequence), sum(o.repair_attempt > 0 for o in sequence))))
    return result


def eta_squared(rows, field):
    values = [row["willingness"] for row in rows]
    grand = statistics.mean(values)
    total = sum((value - grand)**2 for value in values)
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row["willingness"])
    return sum(len(v)*(statistics.mean(v)-grand)**2 for v in groups.values()) / total if total else 0


def _solve(matrix, rhs):
    """Small ridge normal system; pivoting, no numerical library dependency."""
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for i in range(len(a)):
        pivot = max(range(i, len(a)), key=lambda j: abs(a[j][i]))
        a[i], a[pivot] = a[pivot], a[i]
        scale = a[i][i]
        if abs(scale) < 1e-12:
            continue
        a[i] = [v / scale for v in a[i]]
        for j in range(len(a)):
            if j == i:
                continue
            scale = a[j][i]
            if scale:
                a[j] = [v - scale*w for v, w in zip(a[j], a[i])]
    return [row[-1] for row in a]


def predictor(rows, key):
    train = [r for r in rows if not r["held_out"]]
    test = [r for r in rows if r["held_out"]]
    size = len(train[0][key])
    means = [statistics.mean(r[key][i] for r in train) for i in range(size)]
    scales = [statistics.pstdev(r[key][i] for r in train) or 1 for i in range(size)]
    def features(row):
        return [1.0] + [(v-m)/s for v, m, s in zip(row[key], means, scales)]
    xs = [features(row) for row in train]
    n = size + 1
    matrix = [[sum(x[i]*x[j] for x in xs) + (0.1 if i == j and i else 0)
               for j in range(n)] for i in range(n)]
    rhs = [sum(x[i]*row["willingness"] for x, row in zip(xs, train)) for i in range(n)]
    weights = _solve(matrix, rhs)
    predicted = [max(0, min(1, sum(w*x for w, x in zip(weights, features(row))))) for row in test]
    observed = [row["willingness"] for row in test]
    error = sum((a-b)**2 for a, b in zip(observed, predicted))
    total = sum((a-statistics.mean(observed))**2 for a in observed)
    return {"held_out_r2": 1-error/total if total else 0,
            "held_out_mae": statistics.mean(abs(a-b) for a, b in zip(observed, predicted)),
            "features": n, "train_rows": len(train), "held_out_rows": len(test)}


def _features(p, s, r, c, counts):
    # Baseline knows the request pattern and event counts. Full knows authored
    # traits, resources and ordered-history state plus declared interactions.
    pattern = [float(c.id == str(i)) for i in range(12)]
    count_only = pattern + list(counts)
    raw = [getattr(p, k) for k in ("pride", "patience", "forgiveness", "assertiveness", "calm", "curiosity", "bluntness")]
    raw += [p.value(k) for k in ("kindness", "loyalty", "autonomy", "honesty")]
    raw += [s.energy, s.trust, s.bonding, s.certainty, s.opportunity_cost, s.commitments, r.h, r.r]
    interactions = [p.value("kindness")*c.user_value, p.value("kindness")*c.user_urgency,
                    p.value("loyalty")*c.user_value*(s.trust+s.bonding),
                    c.effort/(s.energy+0.15), c.effort*p.patience/(s.energy+0.15),
                    c.autonomy_cost*p.pride, c.autonomy_cost*p.value("autonomy"),
                    c.risk*p.value("autonomy"), c.uncertainty*p.value("honesty"),
                    r.h*(0.25+c.effort), r.h*p.pride*(0.25+c.effort),
                    r.r*(0.25+c.effort), r.r*p.patience*(0.25+c.effort),
                    c.effort*s.opportunity_cost, c.effort*s.commitments,
                    c.effort**2/(s.energy+0.15), c.character_value,
                    sum(p.value(k)*v for k, v in c.value_effects)]
    return count_only, count_only + raw + interactions


def run_sweep():
    started = time.perf_counter()
    rows = []
    baseline = []
    switches = Counter()
    ablations = defaultdict(list)
    ablation_switches = Counter()
    ps, cs = personalities(), requests()
    neutral = Personality(pride=0.5, patience=0.5, forgiveness=0.5, assertiveness=0.5,
                          calm=0.5, curiosity=0.5, bluntness=0.5,
                          values=tuple((name, 0.5) for name in ("kindness", "loyalty", "autonomy", "honesty")))
    neutral_histories = histories(neutral)
    for pi, p in enumerate(ps):
        for hi, (r, s, counts) in enumerate(histories(p)):
            for ci, c in enumerate(cs):
                kwargs = dict(character_seed=pi, relationship_id="factorial", turn_id=f"{hi}:{ci}")
                d = decide(p, s, r, Perception(), (c,), noise=0, **kwargs)
                noisy = decide(p, s, r, Perception(), (c,), noise=0.025, **kwargs)
                counted, full = _features(p, s, r, c, counts)
                row = dict(personality=pi, history=hi, request=ci,
                           willingness=d.components[0].willingness, granted=bool(d.grants),
                           stance=asdict(d.stance), count_features=counted, full_features=full,
                           held_out=(pi*6+hi) % 5 == 0)
                rows.append(row)
                baseline.append((d, noisy))
                switches["noise"] += d.grants != noisy.grants
                versions = {
                    "without_personality": (neutral, s, neutral_histories[hi][0], c),
                    "without_history": (p, s, RelationshipDynamics(), c),
                    "without_request_value": (p, s, r, replace(c, user_value=0, user_urgency=0, character_value=0, character_urgency=0, value_effects=())),
                    "without_state": (p, DecisionState(), r, c),
                }
                for name, (pa, sa, ra, ca) in versions.items():
                    ablated = decide(pa, sa, ra, Perception(), (ca,), noise=0, **kwargs)
                    ablations[name].append(abs(d.components[0].willingness-ablated.components[0].willingness))
                    ablation_switches[name] += d.grants != ablated.grants
    values = sorted(row["willingness"] for row in rows)
    n = len(rows)
    combinations = Counter(
        ("help" if d.grants else "decline") + ("+boundary" if d.stance.boundary > d.stance.engagement else "")
        + ("+delay" if d.stance.delay > d.stance.engagement else "") for d, _ in baseline)
    trajectory = []
    p = Personality()
    r, s, _ = histories(p)[2]
    for hours in (0, 0.25, 2, 24, 72):
        cooled, _ = update_relationship(r, p, Perception(), s, elapsed=hours*3600, now=hours*3600+120, turn_id=f"rest:{hours}")
        trajectory.append({"hours": hours, "h": cooled.h, "r": cooled.r,
                           "willingness": decide(p, s, cooled, Perception(), (cs[1],), character_seed=0,
                                                  relationship_id="recovery", turn_id="probe", noise=0).components[0].willingness})
    return {"policy_version": POLICY_VERSION, "design": "32 personalities x 6 histories x 12 fixed request perceptions, noise off/on",
            "decisions_per_condition": n, "llm_calls": 0,
            "willingness_quantiles": {str(q): values[round(q*(n-1))] for q in (0, 0.1, 0.5, 0.9, 1)},
            "grant_rate_noise_off": statistics.mean(r["granted"] for r in rows),
            "grant_rate_noise_on": statistics.mean(bool(d.grants) for _, d in baseline),
            "noise_switch_rate": switches["noise"]/n,
            "one_way_eta_squared": {field: eta_squared(rows, field) for field in ("personality", "history", "request")},
            "count_only_predictor": predictor(rows, "count_features"),
            "full_state_predictor": predictor(rows, "full_features"),
            "predictor_method": "ridge .1; fixed input/interaction features; personality/history cells held out; no outcomes, margins or utility terms as features",
            "ablations": {key: {"mean_absolute_willingness_change": statistics.mean(vals),
                                "grant_switch_rate": ablation_switches[key]/n} for key, vals in ablations.items()},
            "action_combinations_display_only": dict(combinations), "recovery_trajectory": trajectory,
            "elapsed_seconds": time.perf_counter()-started,
            "semantic_contradiction_rate": None, "withheld_expression_rate": None,
            "limitations": ["Synthetic preference-model measurement; no live language generation or latency measurement.",
                            "History cells include declared capacity/commitment differences; repair-order pair has identical resources and counts.",
                            "Main-effect eta squared excludes interactions. Predictor comparison is explanatory, not a psychological validation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_sweep()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()

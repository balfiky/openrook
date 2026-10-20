from __future__ import annotations

import hashlib
import copy
from pathlib import Path

import pytest

from evals.retention_check import analyze_retention, first_work_balance, paired_interval
from evals.study_design import analyze_study
from runtime.research_store import encode
from tests.test_scheduled_opportunities import ENVIRONMENT, quiet_run
from tests.test_study_design import design, report_for


def choices(run, *partners):
    run["independent_outcomes"]["choices"] = [
        {"tick": tick, "strategy": "solo" if partner is None else "collaborate", "partner": partner}
        for tick, partner in partners
    ]
    return run


def registration(protocol):
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "check": "first_work_partner_balance",
        "protocol_sha256": hashlib.sha256(encode(protocol).encode()).hexdigest(),
        "planned_pairs": protocol["replicates"],
        "family_alpha": 0.05,
        "family_size": 2,
        "minimum_effect": 0.2,
        "bootstrap_draws": 100000,
        "bootstrap_seed": 0,
        "analysis_sources": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in (
                "evals/retention_check.py",
                "evals/opportunity_scoring.py",
                "evals/study_design.py",
            )
        },
    }


def test_retention_keeps_rest_solo_and_missing_schedule_distinct():
    assert first_work_balance(quiet_run(), 4) == 0
    assert first_work_balance(choices(quiet_run(), (5, None), (6, "Ari")), 4) == 0
    assert first_work_balance(choices(quiet_run(), (8, "Ari")), 4) == 1
    incomplete = quiet_run()
    incomplete["opportunities"].pop()
    assert first_work_balance(incomplete, 4) is None
    assert first_work_balance(quiet_run(), 5) is None


def test_first_work_excludes_later_choices_and_shares_weight_within_opportunity():
    run = choices(quiet_run(), (5, "Bo"), (5, None), (6, "Ari"), (7, "Ari"))
    assert first_work_balance(run, 4) == -0.5
    run["independent_outcomes"]["choices"].reverse()
    assert first_work_balance(run, 4) == -0.5


def test_interval_is_paired_deterministic_and_two_sided():
    assert paired_interval([1, 1, 1], alpha=0.025, draws=1000, seed=0) == [1, 1]
    interval = paired_interval([-1, 0, 1], alpha=0.025, draws=1000, seed=0)
    assert interval[0] < 0 < interval[1]
    assert paired_interval([-1, 0, 1], alpha=0.025, draws=1000, seed=0) == interval


def study():
    protocol = design(replicates=6)
    protocol["environment"] = ENVIRONMENT
    protocol["steps"] = [{"kind": "tick"}] * 8
    protocol["study"]["primary_endpoint"] = "world_opportunity_partner_balance"
    protocol["study"]["smallest_effect"] = 0.2
    report = report_for(protocol)
    for run in report["runs"]:
        run.update(choices(quiet_run(), (5, "Ari" if run["condition"] == "procedures" else "Bo")))
    return protocol, report


def test_secondary_does_not_promote_pilot_or_replace_primary_result():
    protocol, report = study()
    result = analyze_retention(registration(protocol), protocol, report)
    assert result["valid_pairs"] == 6
    assert result["original_primary_decision_unchanged"] == "exploratory_only"
    endpoint = result["endpoints"]["first_work_partner_balance"]
    assert endpoint["mean_candidate_minus_baseline"] == 2
    assert endpoint["family_adjusted_interval"] == [2, 2]
    assert endpoint["decision"] == "not_eligible"


@pytest.mark.parametrize("corruption", ["source", "sample", "threshold", "protocol"])
def test_registration_prevents_changed_sources_sample_threshold_or_protocol(corruption):
    protocol, report = study()
    plan = registration(protocol)
    if corruption == "source":
        plan["analysis_sources"]["evals/retention_check.py"] = "changed"
    elif corruption == "sample":
        plan["planned_pairs"] = 4
    elif corruption == "threshold":
        plan["minimum_effect"] = 0
    else:
        plan["protocol_sha256"] = "changed"
    with pytest.raises(ValueError):
        analyze_retention(plan, protocol, report)


def test_partial_cohort_is_not_analyzed_as_complete():
    protocol, report = study()
    report["runs"].pop()
    result = analyze_retention(registration(protocol), protocol, report)
    assert result["valid_pairs"] == 5
    assert result["endpoints"]["first_work_partner_balance"]["family_adjusted_interval"] is None
    assert result["endpoints"]["first_work_partner_balance"]["decision"] == "not_eligible"


@pytest.mark.parametrize("resource_stop", [False, True])
def test_confirmation_requires_complete_live_unstopped_cohort(tmp_path, resource_stop):
    pilot, report = study()
    for run in report["runs"]:
        if run["condition"] == "procedures" and run["replicate"] % 2:
            choices(run)
    pilot_analysis = tmp_path / "pilot.json"
    pilot_analysis.write_text(encode(analyze_study(pilot, report)))
    protocol = copy.deepcopy(pilot)
    protocol["id"] += "-confirm"
    protocol["environment_seeds"] = list(range(200, 206))
    protocol["study"].update(
        stage="confirmatory",
        planning={
            "analysis_path": str(pilot_analysis),
            "analysis_sha256": hashlib.sha256(pilot_analysis.read_bytes()).hexdigest(),
        },
    )
    _, confirmation = study()
    confirmation["protocol_sha256"] = hashlib.sha256(encode(protocol).encode()).hexdigest()
    for run in confirmation["runs"]:
        run["environment_seed"] = protocol["environment_seeds"][run["replicate"] - 1]
    if resource_stop:
        confirmation["resource_usage"] = {"stop_reason": "runtime"}
    result = analyze_retention(registration(protocol), protocol, confirmation)
    endpoint = result["endpoints"]["first_work_partner_balance"]
    assert endpoint["family_adjusted_interval"] == [2, 2]
    assert endpoint["decision"] == (
        "not_eligible" if resource_stop else "registered_difference_supported"
    )

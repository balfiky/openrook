from __future__ import annotations

import copy
import hashlib

import pytest

from evals.study_design import analyze_study, planned_pairs, preflight_study, validate_study
from runtime.research_store import encode
from tests.test_retention_check import choices, study


def retention_study():
    protocol, report = study()
    protocol["study"].update(
        primary_endpoint="world_first_work_partner_balance", expected_effect=0.6
    )
    report["protocol_sha256"] = hashlib.sha256(encode(protocol).encode()).hexdigest()
    for run in report["runs"]:
        if run["condition"] == "procedures" and run["replicate"] % 2:
            choices(run)
    return protocol, report


def test_retention_is_a_prospective_primary_with_activity_scored_separately():
    protocol, report = retention_study()
    analysis = analyze_study(protocol, report)
    assert analysis["mean_candidate_minus_baseline"] == 1.5
    assert analysis["planning_expected_effect"] == 0.6
    assert analysis["planning_difference_above_threshold"] == pytest.approx(0.4)
    assert analysis["planned_pairs_from_pilot_variance"] == planned_pairs(
        analysis["paired_sd"], 0.4, 0.05, 0.8
    )
    assert analysis["decision"] == "exploratory_only"


@pytest.mark.parametrize("value", [None, True, 0.1, 0.2, 2, float("nan")])
def test_expected_effect_must_be_possible_and_above_acceptance_threshold(value):
    protocol, _ = retention_study()
    protocol["study"]["expected_effect"] = value
    with pytest.raises(ValueError, match="Expected effect"):
        validate_study(protocol)


def test_confirmation_cannot_change_the_expected_effect_after_pilot(tmp_path):
    pilot, report = retention_study()
    analysis = analyze_study(pilot, report)
    evidence = tmp_path / "pilot.json"
    evidence.write_text(encode(analysis))
    confirmation = copy.deepcopy(pilot)
    confirmation["replicates"] = analysis["planned_pairs_from_pilot_variance"]
    confirmation["environment_seeds"] = list(range(1000, 1000 + confirmation["replicates"]))
    confirmation["study"].update(
        stage="confirmatory",
        planning={
            "analysis_path": str(evidence),
            "analysis_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        },
    )
    preflight_study(confirmation)
    confirmation["study"]["expected_effect"] = 0.8
    with pytest.raises(ValueError, match="same endpoint"):
        preflight_study(confirmation)

"""Tests for the deep emotional-quality QA runner."""

from __future__ import annotations

from tests._fakes import ScriptedSocialBackend
from evals.emotional_quality import (
    emotional_quality_scenarios,
    markdown_report,
    run_quality_suite,
)


def test_emotional_quality_scenarios_cover_core_arcs():
    scenarios = emotional_quality_scenarios()
    ids = {scenario.id for scenario in scenarios}

    assert "anger_repair_love_arc" in ids
    assert "sadness_support_arc" in ids
    assert "caution_uncertainty_arc" in ids
    assert "energy_load_arc" in ids


def test_emotional_quality_mock_records_versioned_policy_properties():
    report = run_quality_suite(
        ScriptedSocialBackend,
        backend_label="mock",
        model_label="",
    )

    assert report.policy_version == "willingness-1.3"
    assert report.scenarios
    assert all(scenario.turns for scenario in report.scenarios)
    assert all(turn.decision and turn.delivery_outcome for scenario in report.scenarios for turn in scenario.turns)
    assert any(turn.historical_expectations for scenario in report.scenarios for turn in scenario.turns)


def test_emotional_quality_report_contains_state_columns():
    report = run_quality_suite(
        ScriptedSocialBackend,
        backend_label="mock",
        model_label="",
    )
    rendered = markdown_report(report)

    assert "Arousal" in rendered
    assert "Valence" in rendered
    assert "Response Impact Checks" in rendered
    assert "Same Prompt, Different Seeded States" in rendered

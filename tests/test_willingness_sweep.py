"""Variation must survive removing noise and exceed a count/pattern explanation."""
from evals.willingness_sweep import run_sweep


def test_factorial_variation_ablations_and_grouped_holdout():
    result = run_sweep()
    assert result["decisions_per_condition"] == 32*6*12
    assert result["llm_calls"] == 0
    assert all(value > 0.01 for value in result["one_way_eta_squared"].values())
    assert result["full_state_predictor"]["held_out_r2"] > result["count_only_predictor"]["held_out_r2"] + 0.05
    assert result["full_state_predictor"]["held_out_mae"] < result["count_only_predictor"]["held_out_mae"]
    assert all(v["mean_absolute_willingness_change"] > 0.02 for v in result["ablations"].values())
    assert result["noise_switch_rate"] < 0.05
    assert len(result["action_combinations_display_only"]) > 2
    assert result["semantic_contradiction_rate"] is None  # No fake live-language result.

"""A separately registered first-choice check, never a replacement primary endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from evals.opportunity_scoring import score_opportunities
from evals.study_design import analyze_study
from runtime.research_store import encode


def first_work_balance(run: dict, minimum: int) -> float | None:
    scores = score_opportunities(run)
    if not scores["schedule_complete"] or scores["phases"]["probe"]["opportunities"] < minimum:
        return None
    return next(
        (
            item["partner_balance"]
            for item in scores["opportunities"]
            if item["phase"] == "probe" and item["unique_work_requests"]
        ),
        0.0,
    )


def paired_interval(differences: list[float], *, alpha: float, draws: int, seed: int) -> list:
    generator = random.Random(seed)
    means = sorted(
        statistics.mean(generator.choices(differences, k=len(differences)))
        for _draw in range(draws)
    )
    return [
        means[int(draws * alpha / 2)],
        means[min(draws - 1, math.ceil(draws * (1 - alpha / 2)) - 1)],
    ]


def analyze_retention(plan: dict, protocol: dict, report: dict) -> dict:
    if plan.get("schema_version") != 1 or plan.get("check") != "first_work_partner_balance":
        raise ValueError("Unknown retention registration")
    if plan.get("protocol_sha256") != hashlib.sha256(encode(protocol).encode()).hexdigest():
        raise ValueError("Retention registration does not match the study protocol")
    if protocol["study"]["primary_endpoint"] != "world_opportunity_partner_balance":
        raise ValueError("Retention check requires the scheduled primary endpoint")
    if plan.get("planned_pairs") != protocol.get("replicates"):
        raise ValueError("Retention sample must be fixed to the full registered cohort")
    if (
        plan.get("family_alpha") != 0.05
        or plan.get("family_size") != 2
        or plan.get("minimum_effect") != 0.2
        or plan.get("bootstrap_draws") != 100000
        or plan.get("bootstrap_seed") != 0
    ):
        raise ValueError("Unsupported retention inference settings")
    root = Path(__file__).resolve().parents[1]
    expected_sources = {
        "evals/retention_check.py",
        "evals/opportunity_scoring.py",
        "evals/study_design.py",
    }
    if set(plan.get("analysis_sources", {})) != expected_sources or any(
        hashlib.sha256((root / path).read_bytes()).hexdigest() != digest
        for path, digest in plan["analysis_sources"].items()
    ):
        raise ValueError("Analysis code differs from its registration")
    original = analyze_study(protocol, report)
    by_subject = {(run["replicate"], run["condition"]): run for run in report["runs"]}
    pairs = []
    for pair in original["pairs"]:
        values = [
            first_work_balance(
                by_subject[(pair["replicate"], condition)],
                protocol["study"]["minimum_endpoint_observations"],
            )
            for condition in protocol["study"]["contrast"]
        ]
        if any(value is None for value in values):
            raise ValueError("Complete primary schedules must also support retention scoring")
        pairs.append(
            {
                "replicate": pair["replicate"],
                "baseline": values[0],
                "candidate": values[1],
                "difference": values[1] - values[0],
            }
        )
    complete = (
        len(pairs) == plan["planned_pairs"]
        and len(pairs) >= protocol["study"]["minimum_pairs"]
        and len(report["runs"]) == 2 * plan["planned_pairs"]
        and not original["missing_or_excluded_pairs"]
    )
    eligible = (
        complete
        and protocol["study"]["stage"] == "confirmatory"
        and original["live_model_observations"]
        and not report.get("resource_usage", {}).get("stop_reason")
    )
    endpoints = {}
    for name, differences in {
        "original_opportunity_balance": [pair["difference"] for pair in original["pairs"]],
        "first_work_partner_balance": [pair["difference"] for pair in pairs],
    }.items():
        interval = (
            paired_interval(
                differences,
                alpha=plan["family_alpha"] / plan["family_size"],
                draws=plan["bootstrap_draws"],
                seed=plan["bootstrap_seed"],
            )
            if complete
            else None
        )
        endpoints[name] = {
            "mean_candidate_minus_baseline": statistics.mean(differences) if differences else None,
            "family_adjusted_interval": interval,
            "minimum_effect": plan["minimum_effect"],
            "decision": (
                "registered_difference_supported"
                if eligible and interval[0] > plan["minimum_effect"]
                else "registered_difference_not_established"
                if eligible
                else "not_eligible"
            ),
        }
    return {
        "schema_version": 1,
        "registration_sha256": hashlib.sha256(encode(plan).encode()).hexdigest(),
        "protocol_sha256": original["protocol_sha256"],
        "subject_run_ids": original["subject_run_ids"],
        "original_primary_decision_unchanged": original["decision"],
        "valid_pairs": len(pairs),
        "planned_pairs": plan["planned_pairs"],
        "pairs": pairs,
        "endpoints": endpoints,
        "interval_method": "Paired-subject percentile bootstrap, 100000 draws, seed 0; two 97.5% intervals use Bonferroni allocation of family alpha 0.05. Approximate, especially for small samples.",
        "interpretation": "First freely chosen work opportunity after restart, averaged within that opportunity; solo or no work contributes zero expressed partner balance, not failed ability. No subjects are excluded for inactivity. This tests expressed history retention, not sustained work rate or universal human fidelity.",
        "planning": "Sample size is inherited from the original primary study; no separate power guarantee. This secondary registration cannot promote the original primary or satisfy its long-run gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = analyze_retention(
        json.loads(arguments.registration.read_text()),
        json.loads((arguments.study / "protocol.json").read_text()),
        json.loads((arguments.study / "report.json").read_text()),
    )
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"pairs", "subject_run_ids"}}
        )
    )


if __name__ == "__main__":
    main()

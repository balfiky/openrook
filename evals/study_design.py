"""Preregistered paired comparisons with subject-level, not turn-level, inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from runtime.config import RuntimeConfig
from runtime.research_store import encode


ENDPOINTS = {"beta_probe_accuracy", "world_probe_ari_share", "verified_completion_rate", "world_opportunity_partner_balance", "world_first_work_partner_balance"}
FACTORS = {
    "research_memory_mode",
    "research_progress_estimates",
    "research_appraisal_mode",
    "research_dispositions",
    "research_context_mode",
    "research_execution_mode",
    "environment_history",
    "backend",
}


def implementation_signature(manifest: dict) -> str | None:
    required = {"runtime", "birth_configuration", "birth_drives", "features", "code_fingerprints"}
    if not required <= set(manifest):
        return None
    runtime = {key: value for key, value in manifest["runtime"].items() if key != "data_dir"}
    fingerprints = {
        path: digest for path, digest in manifest["code_fingerprints"].items()
        if path.startswith(("runtime/", "pipeline/", "core/", "config/", "openrook_tools/"))
        or path in {"evals/learning_environment.py", "evals/development_environment.py", "evals/research_environment.py", "evals/research.py", "evals/study_budget.py"}
    }
    if not fingerprints:
        return None
    payload = {
        **{key: manifest[key] for key in required - {"runtime", "code_fingerprints"}},
        "runtime": runtime,
        "code_fingerprints": fingerprints,
        "prompt_fingerprints": manifest.get("prompt_fingerprints", {}),
        "contracts": {key: manifest.get(key, 1) for key in (
            "decision_contract_version", "action_contract_version", "memory_presentation_version"
        )},
    }
    return hashlib.sha256(encode(payload).encode()).hexdigest()


def validate_run_implementation(protocol: dict, condition: str, manifest: dict) -> None:
    if protocol.get("study", {}).get("stage") != "confirmatory":
        return
    pilot = _read_hashed(protocol["study"]["planning"], "analysis")
    signature = implementation_signature(manifest)
    if not signature or pilot.get("implementation_signatures", {}).get(condition) != signature:
        raise ValueError("The effective implementation, model or birth configuration differs from the pilot")


def mechanism_signature(protocol: dict) -> str:
    environment = dict(protocol.get("environment") or {})
    environment.pop("seed", None)
    return hashlib.sha256(
        encode(
            {
                "conditions": protocol["conditions"],
                "environment": environment,
                "steps": [
                    step
                    for step in protocol.get("steps", [])
                    if step.get("kind") not in {"wait", "wait_until"}
                ],
            }
        ).encode()
    ).hexdigest()


def _number(value: Any, lower: float, upper: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and lower < value < upper
    )


def validate_study(protocol: dict[str, Any]) -> None:
    study = protocol["study"]
    fields = {
        "stage",
        "primary_endpoint",
        "contrast",
        "factor",
        "smallest_effect",
        "expected_effect",
        "alpha",
        "power",
        "minimum_pairs",
        "minimum_endpoint_observations",
        "exclusions",
        "stopping_rule",
        "planning",
        "long_run_gate",
    }
    if not isinstance(study, dict) or set(study) - fields:
        raise ValueError("Unknown preregistration fields")
    if (
        study.get("stage") not in {"pilot", "confirmatory"}
        or study.get("primary_endpoint") not in ENDPOINTS
        or study.get("factor") not in FACTORS
    ):
        raise ValueError("Declare a supported study stage, endpoint and factor")
    conditions = protocol.get("conditions", [])
    contrast = study.get("contrast")
    if (
        len(conditions) != 2
        or not isinstance(contrast, list)
        or len(contrast) != 2
        or set(contrast) != {item["name"] for item in conditions}
    ):
        raise ValueError("A preregistration requires exactly two named comparison arms")
    for key, lower, upper in (("smallest_effect", 0, 1), ("alpha", 0, 0.5), ("power", 0.5, 1)):
        if not _number(study.get(key), lower, upper):
            raise ValueError(f"Invalid preregistered {key}")
    if type(study.get("minimum_pairs")) is not int or study["minimum_pairs"] < 6:
        raise ValueError(
            "Subject-level intervals require at least six paired subjects; pilots may finish smaller without inference"
        )
    if (
        type(study.get("minimum_endpoint_observations")) is not int
        or study["minimum_endpoint_observations"] < 1
    ):
        raise ValueError("Declare minimum observations per subject endpoint")
    if (
        study.get("exclusions") != ["infrastructure_failure", "missing_endpoint"]
        or study.get("stopping_rule") != "fixed_resources_or_integrity_failure"
    ):
        raise ValueError(
            "Declare infrastructure/missing-endpoint exclusions and fixed resource stopping"
        )
    if protocol.get("condition_order") != "counterbalanced" or "resource_limits" not in protocol:
        raise ValueError(
            "Preregistered comparisons require counterbalancing and study-wide resource bounds"
        )
    seeds = protocol.get("environment_seeds", [])
    if len(seeds) != protocol.get("replicates", 1) or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct, declared paired environment seeds")
    runtime_values = [
        RuntimeConfig(**item.get("runtime", {})).to_yaml_dict() for item in conditions
    ]
    differences = {
        key for key in runtime_values[0] if runtime_values[0][key] != runtime_values[1][key]
    }
    allowed = (
        {"llm_backend", "llm_base_url", "llm_model"}
        if study["factor"] == "backend"
        else {study["factor"]}
    )
    if differences - allowed:
        raise ValueError("Comparison changes more than its declared factor")
    if any(item.get("features", {}) != conditions[0].get("features", {}) for item in conditions):
        raise ValueError("Feature flags must be matched across comparison arms")
    environments = [
        dict(item.get("environment", protocol.get("environment")) or {}) for item in conditions
    ]
    if study["factor"] == "environment_history":
        if any(item.get("profile") != "development_world_v1" for item in environments):
            raise ValueError("History comparisons require the development world")
        histories = [item.pop("history", "balanced") for item in environments]
        if histories[0] == histories[1]:
            raise ValueError("History comparison requires distinct histories")
    elif not differences:
        raise ValueError("The declared runtime factor does not differ")
    if environments[0] != environments[1]:
        raise ValueError("Task affordances and resources must be matched across arms")
    if "expected_effect" in study and not _number(
        study["expected_effect"], study["smallest_effect"],
        2 if study["primary_endpoint"] in {"world_opportunity_partner_balance", "world_first_work_partner_balance"} else 1,
    ):
        raise ValueError("Expected effect must exceed the minimum effect within the endpoint range")
    if study["primary_endpoint"] in {"world_opportunity_partner_balance", "world_first_work_partner_balance"}:
        scheduled = sum(step.get("kind") == "tick" for step in protocol.get("steps", []))
        if any(
            item.get("profile") != "development_world_v1"
            or item.get("clock_mode") != "opportunities"
            or scheduled - item.get("history_ticks", 12) < study["minimum_endpoint_observations"]
            for item in environments
        ):
            raise ValueError("Opportunity endpoints require a complete scheduled history and probe period")
    if any(not item["research_mode"] or not item["tools_enabled"] for item in runtime_values):
        raise ValueError("Research comparisons require enabled persistent subjects and task tools")
    if study["factor"] in {
        "research_memory_mode",
        "research_progress_estimates",
        "research_appraisal_mode",
        "research_dispositions",
    }:
        if any(item["research_context_mode"] != "task_only" for item in runtime_values):
            raise ValueError(
                "Mechanism isolation requires task-only context without identity-memory leakage"
            )
    if study["factor"] in {"research_appraisal_mode", "research_dispositions"} and any(
        item["research_memory_mode"] not in {"episodes", "procedures"} for item in runtime_values
    ):
        raise ValueError("Appraisal and disposition comparisons require retained outcome memory")
    if study["stage"] == "confirmatory" and not study.get("planning"):
        raise ValueError("Confirmatory studies require a hashed pilot analysis for sample planning")


def _read_hashed(reference: dict, prefix: str) -> dict:
    if not isinstance(reference, dict) or not isinstance(reference.get(prefix + "_path"), str):
        raise ValueError("A local evidence path is required")
    data = Path(reference[prefix + "_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.get(prefix + "_sha256"):
        raise ValueError("Evidence hash does not match preregistration")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("Evidence must be a JSON object")
    return result


def planned_pairs(
    paired_sd: float, effect: float, alpha: float, power: float, *, minimum: int = 6
) -> int | None:
    if not math.isfinite(paired_sd) or paired_sd <= 0:
        return None
    normal = statistics.NormalDist()
    return max(
        minimum,
        math.ceil(
            ((normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) * paired_sd / effect) ** 2
        ),
    )


def _planning_difference(study: dict) -> float:
    return (
        study["expected_effect"] - study["smallest_effect"]
        if "expected_effect" in study else study["smallest_effect"]
    )


def preflight_study(protocol: dict[str, Any]) -> None:
    validate_study(protocol)
    study = protocol["study"]
    if study["stage"] == "confirmatory":
        pilot = _read_hashed(study["planning"], "analysis")
        if (
            not pilot.get("live_model_observations")
            or pilot.get("valid_pairs", 0) < study["minimum_pairs"]
            or pilot.get("primary_endpoint") != study["primary_endpoint"]
            or pilot.get("factor") != study["factor"]
            or pilot.get("planning_expected_effect") != study.get("expected_effect")
            or pilot.get("mechanism_sha256") != mechanism_signature(protocol)
            or set(pilot.get("implementation_signatures", {})) != set(study["contrast"])
            or not all(pilot["implementation_signatures"].values())
        ):
            raise ValueError("A live, informative pilot with the same endpoint and implementation provenance is required")
        required = planned_pairs(
            pilot.get("paired_sd") or 0,
            _planning_difference(study),
            study["alpha"],
            study["power"],
            minimum=study["minimum_pairs"],
        )
        if required is None or protocol.get("replicates", 1) < required:
            raise ValueError("Planned subject pairs do not satisfy pilot-based sample planning")
        if set(protocol["environment_seeds"]) & set(pilot.get("environment_seeds", [])):
            raise ValueError("Confirmatory environment seeds must be held out from the pilot")
    if protocol["resource_limits"]["runtime_seconds"] >= 86400:
        gate = study.get("long_run_gate")
        analysis = _read_hashed(gate, "analysis")
        validation = _read_hashed(gate, "validation")
        if (
            study["stage"] != "confirmatory"
            or analysis.get("stage") != "confirmatory"
            or analysis.get("decision") != "preregistered_difference_supported"
            or not analysis.get("live_model_observations")
            or validation.get("passed") is not True
            or validation.get("infrastructure_failures") != 0
            or validation.get("backend_factory_injected") is not False
            or validation.get("protocol_sha256") != analysis.get("protocol_sha256")
            or validation.get("subject_run_ids") != analysis.get("subject_run_ids")
        ):
            raise ValueError(
                "A long study requires informative live confirmatory evidence and passing engineering validation"
            )
        if (
            analysis.get("primary_endpoint") != study["primary_endpoint"]
            or analysis.get("factor") != study["factor"]
            or analysis.get("mechanism_sha256") != mechanism_signature(protocol)
            or analysis.get("implementation_signatures") != pilot.get("implementation_signatures")
        ):
            raise ValueError("Long-run evidence must support the same mechanism and endpoint")


def _endpoint(run: dict, name: str, minimum: int) -> float | None:
    if name == "world_first_work_partner_balance":
        from evals.retention_check import first_work_balance

        return first_work_balance(run, minimum)
    if name == "world_opportunity_partner_balance":
        from evals.opportunity_scoring import opportunity_endpoint

        return opportunity_endpoint(run, minimum)
    outcomes = run.get("independent_outcomes", {})
    if name == "beta_probe_accuracy":
        probe = outcomes.get("scores", {}).get("beta", {}).get("probe", {})
        return probe.get("accuracy") if probe.get("attempts", 0) >= minimum else None
    if name == "world_probe_ari_share":
        choices = [
            item
            for item in outcomes.get("choices", [])
            if item["phase"] == "probe" and item["strategy"] == "collaborate"
        ]
        return (
            sum(item["partner"] == "Ari" for item in choices) / len(choices)
            if len(choices) >= minimum
            else None
        )
    assessments = {}
    for item in run.get("completion_assessments", []):
        assessments[
            (
                item["goal_id"],
                item["goal_revision"] if "goal_revision" in item else item["revision"],
            )
        ] = item
    observed = [
        item for item in assessments.values() if item["status"] in {"supported", "contradicted"}
    ]
    return (
        sum(item["status"] == "supported" for item in observed) / len(observed)
        if len(observed) >= minimum
        else None
    )


def analyze_study(protocol: dict, report: dict) -> dict:
    validate_study(protocol)
    if report.get("protocol_sha256") != hashlib.sha256(encode(protocol).encode()).hexdigest():
        raise ValueError("Report does not match the preregistered protocol")
    if protocol["study"]["stage"] == "confirmatory":
        preflight_study(protocol)
    study = protocol["study"]
    by_pair: dict[int, dict] = {}
    identifiers = set()
    exclusions = []
    implementations = {}
    for run in report["runs"]:
        identifier = run.get("manifest", {}).get("run_id")
        if not identifier or identifier in identifiers:
            raise ValueError("Subjects require unique recorded run ids")
        identifiers.add(identifier)
        signature = implementation_signature(run["manifest"])
        for epoch in run.get("implementation_epochs", []):
            changed = {**run["manifest"], "code_fingerprints": epoch["current_code"]}
            if implementation_signature(changed) != signature:
                raise ValueError("A subject changed its effective implementation during the study")
        condition = run["condition"]
        if condition in implementations and implementations[condition] != signature:
            raise ValueError("A comparison arm contains mixed effective implementations")
        implementations[condition] = signature
        if study["stage"] == "confirmatory":
            validate_run_implementation(protocol, condition, run["manifest"])
        pair = by_pair.setdefault(run["replicate"], {})
        if run["condition"] in pair or run["condition"] not in study["contrast"]:
            raise ValueError("Duplicate or unknown subject condition")
        value = _endpoint(run, study["primary_endpoint"], study["minimum_endpoint_observations"])
        reason = (
            "infrastructure_failure"
            if run.get("infrastructure_failures")
            else "missing_endpoint"
            if value is None
            else None
        )
        pair[run["condition"]] = {
            "value": value,
            "excluded": reason,
            "environment_seed": run.get("environment_seed"),
            "model_calls": run.get("event_counts", {}).get("model_request", 0),
            "tool_actions": len(run.get("actions", [])),
        }
        if reason:
            exclusions.append(
                {"replicate": run["replicate"], "condition": run["condition"], "reason": reason}
            )
    differences = []
    pairs = []
    missing_pairs = []
    for replicate in range(1, protocol.get("replicates", 1) + 1):
        pair = by_pair.get(replicate, {})
        if any(name not in pair or pair[name]["excluded"] for name in study["contrast"]):
            missing_pairs.append(replicate)
            continue
        baseline, candidate = [pair[name] for name in study["contrast"]]
        expected_seed = protocol["environment_seeds"][replicate - 1]
        if (
            baseline["environment_seed"] != expected_seed
            or candidate["environment_seed"] != expected_seed
        ):
            raise ValueError("Subject environment seed does not match paired preregistration")
        difference = candidate["value"] - baseline["value"]
        differences.append(difference)
        pairs.append(
            {
                "replicate": replicate,
                "baseline": baseline,
                "candidate": candidate,
                "difference": difference,
            }
        )
    count = len(differences)
    mean = statistics.mean(differences) if count else None
    deviation = statistics.stdev(differences) if count >= 2 else None
    interval = None
    if count >= study["minimum_pairs"]:
        generator = random.Random(0)
        bootstrap = sorted(
            statistics.mean(generator.choices(differences, k=count)) for _draw in range(10000)
        )
        interval = [
            bootstrap[int(10000 * study["alpha"] / 2)],
            bootstrap[min(9999, math.ceil(10000 * (1 - study["alpha"] / 2)) - 1)],
        ]
    required = (
        planned_pairs(
            deviation or 0,
            _planning_difference(study),
            study["alpha"],
            study["power"],
            minimum=study["minimum_pairs"],
        )
        if count >= study["minimum_pairs"]
        else None
    )
    live = not report.get("backend_factory_injected", True)
    decision = "insufficient_data" if interval is None else "exploratory_only"
    if study["stage"] == "confirmatory" and interval is not None:
        decision = (
            "preregistered_difference_supported"
            if live
            and not missing_pairs
            and not report.get("resource_usage", {}).get("stop_reason")
            and interval[0] > study["smallest_effect"]
            else "preregistered_difference_not_established"
        )
    return {
        "schema_version": 1,
        "stage": study["stage"],
        "primary_endpoint": study["primary_endpoint"],
        "factor": study["factor"],
        "protocol_sha256": hashlib.sha256(encode(protocol).encode()).hexdigest(),
        "mechanism_sha256": mechanism_signature(protocol),
        "implementation_signatures": implementations,
        "live_model_observations": live,
        "independent_unit": "paired subjects; turns are repeated observations, not extra subjects",
        "valid_pairs": count,
        "subject_run_ids": sorted(identifiers),
        "planned_pairs": protocol.get("replicates", 1),
        "missing_or_excluded_pairs": missing_pairs,
        "exclusions": exclusions,
        "pairs": pairs,
        "mean_candidate_minus_baseline": mean,
        "paired_sd": deviation,
        "interval": interval,
        "interval_method": "paired-subject percentile bootstrap, 10000 draws, seed 0; approximate, especially with few subjects",
        "planned_pairs_from_pilot_variance": required,
        "planning_method": "two-sided normal approximation for a paired mean; pilot estimate, not guaranteed power",
        **({
            "planning_expected_effect": study["expected_effect"],
            "planning_difference_above_threshold": _planning_difference(study),
        } if "expected_effect" in study else {}),
        "environment_seeds": protocol["environment_seeds"],
        "decision": decision,
        "limitations": [
            "No human reference data or human-equivalence claim",
            "Missing probes are unobserved capability, not failures; selection-related attrition can bias comparisons",
            "One model's instances are not independent model replications",
            "Resource costs are reported separately; elapsed time is not development",
            "Shared environment seeds do not fix remote model sampling; provider weights may be unpinned",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if args.report:
        print(encode(analyze_study(protocol, json.loads(Path(args.report).read_text()))))
    else:
        from evals.research import validate_protocol

        validate_protocol(protocol)
        preflight_study(protocol)
        print(encode({"valid": True, "live_calls": 0, "stage": protocol["study"]["stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

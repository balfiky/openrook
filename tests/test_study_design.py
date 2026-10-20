from __future__ import annotations

import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from evals.replay import replay_run
from evals.research import run_protocol, validate_protocol
from evals.study_budget import BudgetedBackend, StudyBudget, validate_limits
from evals.study_design import analyze_study, planned_pairs, preflight_study
from runtime.config import RuntimeConfig
from runtime.goal_execution import unresolved_dependencies, update_goal
from runtime.llm.backend import OpenAICompatibleLLMBackend
from runtime.research_store import ResearchStore, ResearchStudyLimit, encode
from tests.test_research_execution import ExecutionBackend, protocol


def limits(**overrides):
    return {
        "model_calls": 256,
        "input_tokens": 2_000_000,
        "output_tokens": 131072,
        "tool_actions": 128,
        "input_bytes_per_call": 100000,
        "output_tokens_per_call": 512,
        "runtime_seconds": 600,
        "max_cost_usd": 0.0,
        "input_usd_per_million": 0.0,
        "output_usd_per_million": 0.0,
        **overrides,
    }


def design(*, replicates=2):
    selected = protocol(context="task_only")
    first = selected["conditions"][0]
    first["name"] = "episodes"
    first["runtime"].update(research_memory_mode="episodes", research_calls_per_day=32)
    second = copy.deepcopy(first)
    second["name"] = "procedures"
    second["runtime"]["research_memory_mode"] = "procedures"
    selected.update(
        conditions=[first, second],
        replicates=replicates,
        condition_order="counterbalanced",
        environment_seeds=list(range(100, 100 + replicates)),
        resource_limits=limits(),
    )
    selected["study"] = {
        "stage": "pilot",
        "primary_endpoint": "beta_probe_accuracy",
        "contrast": ["episodes", "procedures"],
        "factor": "research_memory_mode",
        "smallest_effect": 0.1,
        "alpha": 0.05,
        "power": 0.8,
        "minimum_pairs": 6,
        "minimum_endpoint_observations": 4,
        "exclusions": ["infrastructure_failure", "missing_endpoint"],
        "stopping_rule": "fixed_resources_or_integrity_failure",
    }
    return selected


def report_for(selected, *, missing=False):
    report = {
        "protocol_sha256": hashlib.sha256(encode(selected).encode()).hexdigest(),
        "backend_factory_injected": False,
        "runs": [],
    }
    for replicate in range(1, selected["replicates"] + 1):
        for condition in selected["study"]["contrast"]:
            candidate = condition == "procedures"
            report["runs"].append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "manifest": {
                        "run_id": f"{replicate}:{condition}",
                        "runtime": {"llm_model": "fixture", "data_dir": f"fixture/{replicate}/{condition}"},
                        "birth_configuration": {},
                        "birth_drives": {},
                        "features": {},
                        "code_fingerprints": {"runtime/agent.py": "fixture-code"},
                    },
                    "environment_seed": selected["environment_seeds"][replicate - 1],
                    "infrastructure_failures": 0,
                    "event_counts": {"model_request": 10},
                    "actions": [{}] * 12,
                    "independent_outcomes": {
                        "scores": {
                            "beta": {
                                "probe": {
                                    "attempts": 0 if missing and candidate else 4,
                                    "accuracy": (0.5 + (replicate % 2) / 4) if candidate else 0.25,
                                }
                            }
                        }
                    },
                }
            )
    return report


def test_preregistration_requires_one_factor_counterbalancing_and_matched_resources():
    selected = design()
    validate_protocol(selected)
    preflight_study(selected)
    selected["conditions"][1]["runtime"]["research_max_steps_per_activation"] = 5
    with pytest.raises(ValueError, match="more than"):
        validate_protocol(selected)
    selected = design()
    selected["condition_order"] = "listed"
    with pytest.raises(ValueError, match="counterbalancing"):
        validate_protocol(selected)
    selected = design()
    selected["environment_seeds"] = [100, 100]
    with pytest.raises(ValueError, match="distinct"):
        validate_protocol(selected)


def test_analysis_counts_subject_pairs_not_turns_and_keeps_missing_probes_missing():
    selected = design(replicates=6)
    result = analyze_study(selected, report_for(selected))
    assert result["valid_pairs"] == 6
    assert result["mean_candidate_minus_baseline"] == 0.375
    assert result["interval"] and result["paired_sd"] > 0
    assert result["decision"] == "exploratory_only"
    assert result["planned_pairs_from_pilot_variance"] >= 6
    missing = analyze_study(selected, report_for(selected, missing=True))
    assert missing["valid_pairs"] == 0 and missing["mean_candidate_minus_baseline"] is None
    assert len(missing["exclusions"]) == 6 and missing["decision"] == "insufficient_data"
    with pytest.raises(ValueError, match="preregistered protocol"):
        analyze_study({**selected, "id": "changed-after-results"}, report_for(selected))
    duplicate = report_for(selected)
    duplicate["runs"].append(duplicate["runs"][0])
    with pytest.raises(ValueError, match="unique"):
        analyze_study(selected, duplicate)


def test_confirmatory_preflight_requires_live_variance_evidence_and_heldout_seeds(tmp_path):
    selected = design(replicates=6)
    analysis = analyze_study(selected, report_for(selected))
    target = tmp_path / "analysis.json"
    target.write_text(encode(analysis))
    confirmatory = design(replicates=analysis["planned_pairs_from_pilot_variance"])
    confirmatory["study"].update(
        stage="confirmatory",
        planning={
            "analysis_path": str(target),
            "analysis_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match="held out"):
        preflight_study(confirmatory)
    confirmatory["environment_seeds"] = list(range(1000, 1000 + confirmatory["replicates"]))
    preflight_study(confirmatory)
    confirmatory["study"]["planning"]["analysis_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        preflight_study(confirmatory)
    assert planned_pairs(0, 0.1, 0.05, 0.8) is None
    assert planned_pairs(1, 0.1, 0.05, 0.8) == 785


def test_no_long_study_without_a_supported_live_mechanism():
    selected = design()
    selected["resource_limits"]["runtime_seconds"] = 86400
    with pytest.raises(ValueError, match="evidence path"):
        preflight_study(selected)
    selected["study"]["stage"] = "confirmatory"
    with pytest.raises(ValueError, match="pilot analysis"):
        validate_protocol(selected)


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_calls", True),
        ("input_tokens", 0),
        ("runtime_seconds", float("inf")),
        ("max_cost_usd", -1),
        ("output_tokens_per_call", 9000),
    ],
)
def test_budget_rejects_invalid_limits(key, value):
    with pytest.raises(ValueError):
        validate_limits(limits(**{key: value}))


def test_atomic_budget_survives_restart_and_counts_failed_calls(tmp_path):
    bounded = limits(model_calls=3)
    budget = StudyBudget(tmp_path / "budget.db", bounded)

    def reserve(_index):
        try:
            budget.reserve_model("system", "input")
            return True
        except ResearchStudyLimit:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(reserve, range(8))) == 3
    restored = StudyBudget(tmp_path / "budget.db", bounded)
    assert restored.snapshot()["model_calls"] == 3
    with pytest.raises(ResearchStudyLimit):
        restored.reserve_model("system", "input")
    with pytest.raises(ValueError, match="immutable"):
        StudyBudget(tmp_path / "budget.db", limits(model_calls=4))


def test_usage_and_transport_violations_stop_future_reservations(tmp_path):
    budget = StudyBudget(tmp_path / "budget.db", limits())
    budget.reserve_model("system", "input")
    budget.record_response(
        {
            "usage": {"prompt_tokens": 9999, "completion_tokens": 10},
            "transport_attempts": [{"status_code": 200}],
        },
        input_reservation=1040,
    )
    assert budget.snapshot()["reported_input_tokens"] == 9999
    with pytest.raises(ResearchStudyLimit, match="provider_usage"):
        budget.reserve_model("later", "input")


def test_bounded_backend_disables_transport_retries_and_sets_requested_output_cap(
    tmp_path, monkeypatch
):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:8000/v1", model="fixture")
    calls = []

    def post(url, **arguments):
        calls.append(arguments)
        raise requests.ConnectionError("scripted outage")

    monkeypatch.setattr(backend._session, "post", post)
    budget = StudyBudget(tmp_path / "budget.db", limits())
    bounded = BudgetedBackend(backend, budget, injected=False)
    with pytest.raises(requests.ConnectionError):
        bounded.generate("system", "input")
    assert len(calls) == 1 and calls[0]["json"]["max_tokens"] == 512
    assert "chat_template_kwargs" not in calls[0]["json"]
    assert budget.snapshot()["model_calls"] == budget.snapshot()["transport_attempts"] == 1
    assert budget.snapshot()["responses_with_usage"] == 0
    bounded.close()


def test_zero_price_fixture_cannot_launch_a_hosted_backend(tmp_path):
    backend = OpenAICompatibleLLMBackend(base_url="https://provider.example/v1", model="fixture")
    with pytest.raises(ValueError, match="price bounds"):
        BudgetedBackend(backend, StudyBudget(tmp_path / "budget.db", limits()), injected=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bound", [{"model_calls": 2}, {"tool_actions": 1}, {"input_bytes_per_call": 1}]
)
async def test_resource_stops_are_durable_not_personality_failures_and_replay(tmp_path, bound):
    selected = protocol(
        context="task_only", steps=[{"kind": "tick"}, {"kind": "restart"}, {"kind": "tick"}]
    )
    selected["resource_limits"] = limits(**bound)
    selected["conditions"][0]["runtime"]["research_memory_mode"] = "procedures"
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "run"),
        backend_factory=lambda config: ExecutionBackend(),
    )
    assert report["resource_usage"]["stop_reason"]
    assert report["resource_usage"]["model_calls"] <= selected["resource_limits"]["model_calls"]
    run = report["runs"][0]
    assert run["infrastructure_failures"] == 0
    assert run["final_state"]["study_stopped"].startswith("Study resource limit:")
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay


def test_goal_relationship_schemas_remain_valid_in_native_tool_catalogs():
    from nur_tools.native_orchestrator import _json_schema

    schema = _json_schema(
        {
            "parent_goal_id": {"type": ["string", "null"]},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        }
    )
    assert schema["properties"]["parent_goal_id"]["type"] == ["string", "null"]
    assert schema["properties"]["depends_on"] == {"type": "array", "items": {"type": "string"}}


@pytest.mark.asyncio
async def test_paired_runner_counterbalances_and_reports_uninformative_fixture(tmp_path):
    selected = design()
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "study"),
        backend_factory=lambda config: ExecutionBackend(),
    )
    assert [item["condition"] for item in report["runs"]] == [
        "episodes",
        "procedures",
        "procedures",
        "episodes",
    ]
    assert [item["environment_seed"] for item in report["runs"]] == [100, 100, 101, 101]
    assert report["analysis"]["decision"] == "insufficient_data"
    assert report["analysis"]["live_model_observations"] is False
    assert report["resource_usage"]["model_calls"] == 16


def test_goal_dependencies_are_agent_revisable_and_do_not_accept_unverified_claims(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path)))
    prerequisite = update_goal(store, {"goal": "Inspect a result"})
    dependent = update_goal(
        store,
        {
            "goal": "Build on the result",
            "parent_goal_id": prerequisite,
            "depends_on": [prerequisite],
        },
    )
    assert unresolved_dependencies(store, store.get("goal_execution", dependent)) == [prerequisite]
    update_goal(store, {"goal_id": prerequisite, "goal_status": "completed"})
    assert unresolved_dependencies(store, store.get("goal_execution", dependent)) == [prerequisite]
    with pytest.raises(ValueError, match="acyclic"):
        update_goal(store, {"goal_id": prerequisite, "depends_on": [dependent]})
    with pytest.raises(ValueError, match="acyclic"):
        update_goal(store, {"goal_id": prerequisite, "parent_goal_id": dependent})
    update_goal(store, {"goal_id": dependent, "depends_on": []})
    assert unresolved_dependencies(store, store.get("goal_execution", dependent)) == []
    assert store.get("goal_execution", dependent)["revision"] == 2


@pytest.mark.asyncio
async def test_repeated_infrastructure_failure_stops_the_bounded_protocol(tmp_path):
    class FailingBackend(ExecutionBackend):
        def generate(self, system_prompt, user_message):
            raise requests.Timeout("scripted temporary outage")

    selected = protocol(context="task_only", steps=[{"kind": "tick"}] * 10)
    selected["resource_limits"] = limits()
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "failed"),
        backend_factory=lambda config: FailingBackend(),
    )
    assert report["resource_usage"]["stop_reason"] == "three_consecutive_infrastructure_failures"
    assert report["resource_usage"]["model_calls"] == 3
    assert len(report["runs"][0]["steps"]) == 3


def test_shipped_protocols_validate_without_network_or_credentials():
    import json
    from pathlib import Path

    paths = sorted(Path("docs/research_protocols").glob("*.json"))
    assert len(paths) == 10
    assert "retained_choice_raw.json" in {path.name for path in paths}
    for path in paths:
        selected = json.loads(path.read_text())
        validate_protocol(selected)
        preflight_study(selected)
        if path.name in {"scheduled_history.json", "retained_choice.json", "retained_choice_raw.json"}:
            assert selected["resource_limits"]["billing_mode"] == "unmetered"
            assert selected["resource_limits"]["model_calls"] <= 1800
            assert selected["replicates"] >= selected["study"]["minimum_pairs"]
        else:
            assert selected["resource_limits"]["model_calls"] <= 256


def test_failed_hosted_calls_consume_cost_reservations(tmp_path):
    class FailedBackend:
        def generate(self, system_prompt, user_message):
            raise requests.Timeout("fixture timeout")

    budget = StudyBudget(
        tmp_path / "budget.db",
        limits(max_cost_usd=0.0007, input_usd_per_million=0.1, output_usd_per_million=1.0),
    )
    backend = BudgetedBackend(FailedBackend(), budget, injected=True)
    with pytest.raises(requests.Timeout):
        backend.generate("input", "message")
    assert budget.snapshot()["reserved_cost_usd"] > 0
    with pytest.raises(ResearchStudyLimit, match="max_cost_usd"):
        backend.generate("input", "message")
    assert budget.snapshot()["model_calls"] == 1


def test_budget_copies_limits_instead_of_accepting_out_of_band_mutation(tmp_path):
    selected = limits(model_calls=1)
    budget = StudyBudget(tmp_path / "budget.db", selected)
    selected["model_calls"] = 100
    budget.reserve_model("system", "input")
    with pytest.raises(ResearchStudyLimit):
        budget.reserve_model("system", "input")
    assert budget.snapshot()["limits"]["model_calls"] == 1


def test_bounded_backend_does_not_follow_redirects(tmp_path, monkeypatch):
    from types import SimpleNamespace

    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:8000/v1", model="fixture")
    calls = []

    def redirect(url, **arguments):
        calls.append(arguments)
        return SimpleNamespace(status_code=307)

    monkeypatch.setattr(backend._session, "post", redirect)
    bounded = BudgetedBackend(
        backend, StudyBudget(tmp_path / "budget.db", limits()), injected=False
    )
    with pytest.raises(RuntimeError, match="redirects"):
        bounded.generate("system", "input")
    assert len(calls) == 1 and calls[0]["allow_redirects"] is False
    backend.close()


def test_long_run_gate_requires_validation_of_the_same_subjects_and_protocol(tmp_path):
    import json

    pilot_protocol = design(replicates=6)
    pilot = analyze_study(pilot_protocol, report_for(pilot_protocol))
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(encode(pilot))
    selected = design(replicates=pilot["planned_pairs_from_pilot_variance"])
    selected["environment_seeds"] = list(range(1000, 1000 + selected["replicates"]))
    selected["resource_limits"]["runtime_seconds"] = 86400
    selected["study"].update(
        stage="confirmatory",
        planning={
            "analysis_path": str(pilot_path),
            "analysis_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        },
    )
    prior = {**pilot, "stage": "confirmatory", "decision": "preregistered_difference_supported"}
    analysis_path = tmp_path / "confirmatory.json"
    analysis_path.write_text(encode(prior))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        encode(
            {
                "passed": True,
                "infrastructure_failures": 0,
                "backend_factory_injected": False,
                "protocol_sha256": "different protocol",
                "subject_run_ids": prior["subject_run_ids"],
            }
        )
    )
    selected["study"]["long_run_gate"] = {
        "analysis_path": str(analysis_path),
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "validation_path": str(validation_path),
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="engineering validation"):
        preflight_study(selected)
    validation = json.loads(validation_path.read_text())
    validation["protocol_sha256"] = prior["protocol_sha256"]
    validation_path.write_text(encode(validation))
    selected["study"]["long_run_gate"]["validation_sha256"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    preflight_study(selected)

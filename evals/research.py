"""Run versioned behavioral protocols through the production session runtime."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

from config.loader import configuration_scope, get_config
from core.inference_errors import ResearchBudgetExceeded
from core.pipeline_features import PipelineFeatures
from evals.replay import read_events
from evals.research_environment import validate_environment, seed_environment, create_environment_executor
from runtime.config import RuntimeConfig
from runtime.llm.backend import create_llm_backend
from runtime.research_store import ResearchStore, encode
from runtime.research_store import ResearchStudyLimit
from evals.study_budget import BudgetedBackend, StudyBudget, validate_limits
from runtime.sessions.manager import SessionManager
from runtime.tools import create_tool_executor


def validate_protocol(protocol: dict[str, Any]) -> None:
    validate_environment(protocol.get("environment"))
    if protocol.get("condition_order", "listed") not in {"listed", "counterbalanced"}:
        raise ValueError("Condition order must be listed or counterbalanced")
    if protocol.get("scheduler", "continuous") not in {"continuous", "manual"}:
        raise ValueError("Scheduler must be continuous or manual")
    conditions = protocol.get(
        "conditions", [{"name": "research", "runtime": {"research_mode": True}}]
    )
    names = [condition.get("name") for condition in conditions]
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Conditions require unique nonempty names")
    allowed = {field.name for field in fields(RuntimeConfig)} - {
        "data_dir",
        "api_key",
        "llm_api_key",
        "telegram_token",
    }
    for condition in conditions:
        if set(condition.get("runtime", {})) - allowed:
            raise ValueError("Protocol contains unknown, credential, or data-directory overrides")
        PipelineFeatures(**condition.get("features", {}))
        validate_environment(condition.get("environment", protocol.get("environment")))
        environment = condition.get("environment", protocol.get("environment")) or {}
        if environment.get("clock_mode") == "opportunities" and (
            protocol.get("scheduler", "continuous") != "manual"
            or any(step.get("kind") not in {"tick", "restart", "snapshot"} for step in protocol.get("steps", []))
        ):
            raise ValueError("Opportunity-clock protocols require manual ticks, restarts and snapshots only")
    for step in protocol.get("steps", []):
        if step.get("kind") not in {
            "message",
            "tick",
            "wait",
            "wait_until",
            "restart",
            "snapshot",
            "end_session",
        }:
            raise ValueError("Unknown protocol step")
        if step["kind"] in {"wait", "wait_until"} and (
            isinstance(step.get("seconds"), bool)
            or not isinstance(step.get("seconds"), (int, float))
            or not math.isfinite(step["seconds"])
            or step["seconds"] < 0
        ):
            raise ValueError("Wait duration must be nonnegative")
        if step["kind"] == "message" and not isinstance(step.get("text"), str):
            raise ValueError("Message steps require text")
    if (
        isinstance(protocol.get("replicates", 1), bool)
        or not isinstance(protocol.get("replicates", 1), int)
        or protocol.get("replicates", 1) < 1
    ):
        raise ValueError("Replicates must be positive")
    if "environment_seeds" in protocol:
        seeds = protocol["environment_seeds"]
        if not isinstance(seeds, list) or len(seeds) != protocol.get("replicates", 1) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Provide one 32-bit environment seed per replicate")
    if "resource_limits" in protocol:
        validate_limits(protocol["resource_limits"])
        if protocol["resource_limits"]["runtime_seconds"] >= 86400 and "study" not in protocol:
            raise ValueError("Long bounded studies require preregistration and evidence gates")
        if protocol.get("scheduler", "continuous") != "manual":
            raise ValueError("Study-wide resource bounds currently require manual scheduling")
        if any((condition.get("environment", protocol.get("environment")) or {}).get("profile") not in {"selectable_learning_v1", "development_world_v1"} for condition in conditions):
            raise ValueError("Bounded studies require task-only executors without native or external tool bypasses")
    if "study" in protocol:
        from evals.study_design import validate_study

        validate_study(protocol)
    encode(protocol)


def _tick_resource_outcome(
    store: ResearchStore, state: dict[str, Any], *, after: int, started_at: float
) -> dict[str, Any]:
    """Recover signals swallowed by _tick, including checkpoint-free early returns.

    New resource transitions are authoritative even if a pause has since expired.
    Persisted state covers later ticks; the tick frame's first clock reading is
    _tick's `now`, so crossing resume_at during dispatch/reporting cannot change
    classification. Read without the observations() limit to retain late signals.
    """
    with store.connection(read_only=True) as connection:
        rows = connection.execute(
            "SELECT kind,payload FROM observations WHERE sequence>? "
            "AND kind IN ('transition','replay_frame') ORDER BY sequence",
            (after,),
        ).fetchall()
    resources = {}
    for row in rows:
        event = json.loads(row["payload"])
        if row["kind"] == "transition" and event.get("status") in {"resource_wait", "resource_stop"}:
            resources[event["status"]] = event
        elif row["kind"] == "replay_frame" and event["kind"] == "tick" and event["times"]:
            started_at = event["times"][0]
    stop = resources.get("resource_stop")
    if stop or state.get("study_stopped"):
        return {"status": "resource_stop", "reason": stop["reason"] if stop else state["study_stopped"]}
    wait = resources.get("resource_wait")
    resume_at = wait["resume_at"] if wait else state.get("resume_at", 0)
    if wait or started_at < resume_at:
        return {
            "status": "resource_wait",
            "reason": (wait or {}).get("reason") or str(ResearchBudgetExceeded(resume_at)),
            "resume_at": resume_at,
        }
    return {}


async def run_protocol(
    config: RuntimeConfig,
    protocol: dict[str, Any],
    output_dir: str,
    *,
    backend_factory: Callable[[RuntimeConfig], Any] | None = None,
    tool_executor_factory: Callable[[RuntimeConfig], Any] | None = None,
) -> dict[str, Any]:
    validate_protocol(protocol)
    if "study" in protocol:
        from evals.study_design import preflight_study

        preflight_study(protocol)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    protocol_text = encode(protocol)
    (root / "protocol.json").write_text(protocol_text, encoding="utf-8")
    budget = StudyBudget(root / "budget.db", protocol["resource_limits"]) if "resource_limits" in protocol else None
    birth = copy.deepcopy(get_config())
    conditions = protocol.get(
        "conditions", [{"name": "research", "runtime": {"research_mode": True}}]
    )
    report = {
        "schema_version": 1,
        "protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
        "scheduler": protocol.get("scheduler", "continuous"),
        "backend_factory_injected": backend_factory is not None,
        "initial_configuration_sha256": hashlib.sha256(encode(asdict(birth)).encode()).hexdigest(),
        "runs": [],
    }
    blinded = []
    for replicate in range(protocol.get("replicates", 1)):
        ordered = list(enumerate(conditions))
        if protocol.get("condition_order") == "counterbalanced" and replicate % 2:
            ordered.reverse()
        for index, condition in ordered:
            if budget:
                try:
                    budget.check()
                except ResearchStudyLimit:
                    break
            values = config.to_yaml_dict()
            values.update(condition.get("runtime", {}))
            values["data_dir"] = str(root / f"replicate_{replicate + 1}" / f"condition_{index + 1}")
            values["telegram_allowlist"] = set(values.get("telegram_allowlist", []))
            runtime_config = RuntimeConfig(**values)
            feature_flags = PipelineFeatures(**condition.get("features", {}))
            environment = copy.deepcopy(condition.get("environment", protocol.get("environment")))
            if environment and "environment_seeds" in protocol:
                environment["seed"] = protocol["environment_seeds"][replicate]
            environment_record = seed_environment(runtime_config, environment) if environment else None
            environment_factory = (
                (lambda selected_config: create_environment_executor(selected_config, environment))
                if environment else create_tool_executor
            )

            async def record_contact(session_key, user_id, message):
                blinded.append(
                    {
                        "condition": condition["name"],
                        "replicate": replicate + 1,
                        "step": None,
                        "input": None,
                        "response": message,
                        "origin": "autonomous_contact",
                    }
                )

            def make_backend():
                backend = (backend_factory or create_llm_backend)(runtime_config)
                return BudgetedBackend(backend, budget, injected=backend_factory is not None) if budget else backend

            def make_executor():
                executor = (tool_executor_factory or environment_factory)(runtime_config)
                if budget and executor is not None:
                    if executor.tool_runner is not None:
                        raise ValueError("Study resource guard cannot cover native tool transports")
                    executor.set_resource_guard(budget.tool_guard)
                return executor

            def make_manager():
                with configuration_scope(birth):
                    return SessionManager(
                        runtime_config,
                        backend_factory=make_backend,
                        tool_executor_factory=make_executor,
                        features=feature_flags,
                        proactive_callback=record_contact,
                    )

            manager = make_manager()
            run = {
                "condition": condition["name"],
                "replicate": replicate + 1,
                "data_dir": runtime_config.data_dir,
                "steps": [],
                "started_at": time.time(),
                "contact_delivery": "recorded transcript; no claim of human receipt or reading",
                "environment": environment_record,
                "environment_seed": environment.get("seed") if environment else None,
                "order_in_pair": len(report["runs"]) % len(conditions),
            }
            started_monotonic = time.monotonic()
            failed_cycles = 0
            failure_count = 0
            scheduled = (environment or {}).get("clock_mode") == "opportunities"
            if scheduled:
                run["planned_opportunities"] = sum(step["kind"] == "tick" for step in protocol.get("steps", []))
                run["history_opportunities"] = environment.get("history_ticks", 12)
                run["opportunities"] = []
            if report["scheduler"] == "continuous":
                manager.start_background()
            try:
                if protocol.get("study", {}).get("stage") == "confirmatory":
                    from evals.study_design import validate_run_implementation

                    validate_run_implementation(protocol, condition["name"], manager._agent.manifest)
                for step_index, step in enumerate(protocol.get("steps", [])):
                    outcome = {"index": step_index, "kind": step["kind"], "started_at": time.time()}
                    try:
                        if budget:
                            budget.check()
                        platform = str(step.get("platform", "research"))
                        user_id = str(step.get("user_id", "participant"))
                        chat_id = str(step.get("chat_id", "conversation"))
                        if step["kind"] == "message":
                            response = await manager.handle_message(
                                platform, user_id, chat_id, step["text"]
                            )
                            outcome.update(input=step["text"], response=response)
                            blinded.append(
                                {
                                    "condition": condition["name"],
                                    "replicate": replicate + 1,
                                    "step": step_index,
                                    "input": step["text"],
                                    "response": response,
                                }
                            )
                        elif step["kind"] == "tick":
                            if scheduled:
                                from evals.development_environment import DevelopmentEnvironment

                                ordinal = len(run["opportunities"]) + 1
                                observation = DevelopmentEnvironment(runtime_config).advance_opportunity(ordinal)
                                manager._agent.observe_environment(observation)
                                outcome["opportunity"] = {
                                    "id": ordinal, "phase": observation["phase"],
                                }
                            if manager._agent:
                                with manager._agent.store.connection(read_only=True) as connection:
                                    after = connection.execute("SELECT COALESCE(MAX(sequence),0) FROM observations").fetchone()[0]
                            tick_started_at = time.time()
                            await manager._proactive_sweep()
                            if manager._agent:
                                outcome.update(_tick_resource_outcome(
                                    manager._agent.store, manager._agent.state,
                                    after=after, started_at=tick_started_at,
                                ))
                            if scheduled:
                                outcome["opportunity"]["decisions"] = [
                                    event["payload"]["action"]
                                    for event in manager._agent.store.observations(after=after, limit=1000)
                                    if event["kind"] == "autonomous_decision"
                                ]
                        elif step["kind"] == "wait":
                            duration = step["seconds"]
                            if budget:
                                duration = min(duration, max(0, budget.snapshot()["started_at"] + budget.limits["runtime_seconds"] - time.time()))
                            await asyncio.sleep(duration)
                        elif step["kind"] == "wait_until":
                            duration = max(0.0, step["seconds"] - (time.monotonic() - started_monotonic))
                            if budget:
                                duration = min(duration, max(0, budget.snapshot()["started_at"] + budget.limits["runtime_seconds"] - time.time()))
                            await asyncio.sleep(duration)
                        elif step["kind"] == "restart":
                            await manager.shutdown()
                            manager = make_manager()
                            if report["scheduler"] == "continuous":
                                manager.start_background()
                        elif step["kind"] == "end_session":
                            session = await manager.ensure_session(platform, user_id, chat_id)
                            await session.end_session()
                        elif step["kind"] == "snapshot":
                            outcome["state"] = (
                                manager._agent.store.get("agent", "state")
                                if manager._agent
                                else None
                            )
                        outcome.setdefault("status", "completed")
                    except ResearchStudyLimit as error:
                        outcome.update(status="resource_stop", reason=str(error))
                    except ResearchBudgetExceeded as error:
                        outcome.update(
                            status="resource_wait", reason=str(error), resume_at=error.resume_at
                        )
                    except Exception as error:
                        outcome.update(
                            status="infrastructure_failure",
                            error=f"{type(error).__name__}: {error}",
                        )
                    outcome["finished_at"] = time.time()
                    if scheduled and step["kind"] == "tick":
                        run["opportunities"].append({
                            **outcome.get("opportunity", {"id": len(run["opportunities"]) + 1}),
                            "status": outcome["status"],
                        })
                    run["steps"].append(outcome)
                    (root / "progress.json").write_text(
                        encode({**report, "current_run": run}), encoding="utf-8"
                    )
                    if budget:
                        if manager._agent and step["kind"] in {"tick", "message"}:
                            with manager._agent.store.connection(read_only=True) as connection:
                                current_failures = connection.execute("SELECT COUNT(*) FROM observations WHERE kind='autonomous_failure'").fetchone()[0]
                            failed = current_failures > failure_count or outcome["status"] == "infrastructure_failure"
                            failed_cycles = failed_cycles + 1 if failed else 0
                            failure_count = current_failures
                            if failed_cycles >= 3:
                                budget.stop("three_consecutive_infrastructure_failures")
                        try:
                            budget.check()
                        except ResearchStudyLimit:
                            break
            finally:
                await manager.shutdown()
            run["finished_at"] = time.time()
            run["infrastructure_failures"] = sum(
                step["status"] == "infrastructure_failure" for step in run["steps"]
            )
            if manager._agent:
                store = manager._agent.store
                events = read_events(store)
                counts = Counter(event["kind"] for event in events)
                run["background_infrastructure_failures"] = sum(
                    event["kind"] == "autonomous_failure"
                    or (
                        event["kind"] == "replay_frame"
                        and event["payload"]["kind"] == "tick"
                        and event["payload"]["error"] is not None
                    )
                    for event in events
                )
                run["infrastructure_failures"] += run["background_infrastructure_failures"]
                run.update(
                    manifest=store.get("run", "manifest"),
                    goals=store.list("goal"),
                    goal_executions=store.list("goal_execution"),
                    completion_assessments=store.list("completion_assessment"),
                    learning=store.list("learning"),
                    actions=store.list("action"),
                    event_counts=dict(counts),
                    implementation_epochs=[event["payload"] for event in events if event["kind"] == "implementation_epoch"],
                    final_state=store.get("agent", "state"),
                    outcome_learning={kind: store.list(kind) for kind in ("prediction", "prediction_attempt", "outcome", "procedure", "procedure_version", "learning_state", "attention", "behavioral_choice", "disposition")},
                )
                if environment and environment["profile"] == "selectable_learning_v1":
                    from evals.learning_environment import score_learning_environment

                    run["independent_outcomes"] = score_learning_environment(runtime_config)
                if environment and environment["profile"] == "development_world_v1":
                    from evals.development_environment import score_development_environment

                    run["independent_outcomes"] = score_development_environment(runtime_config)
                if scheduled:
                    from evals.opportunity_scoring import score_opportunities

                    run["opportunity_outcomes"] = score_opportunities(run)
                for action in run["actions"]:
                    blinded.append(
                        {
                            "condition": condition["name"],
                            "replicate": replicate + 1,
                            "step": None,
                            "action": {
                                key: action.get(key)
                                for key in ("tool", "arguments", "status", "result")
                            },
                        }
                    )
            report["runs"].append(run)
    random.Random(protocol.get("blinding_seed", 0)).shuffle(blinded)
    samples = []
    key = []
    for index, item in enumerate(blinded):
        sample_id = f"sample_{index + 1}"
        if "action" in item:
            samples.append({"id": sample_id, "kind": "action", "action": item["action"]})
        else:
            samples.append(
                {
                    "id": sample_id,
                    "kind": "speech",
                    "input": item["input"],
                    "response": item["response"],
                }
            )
        key.append(
            {
                "id": sample_id,
                "condition": item["condition"],
                "replicate": item["replicate"],
                "step": item["step"],
                "origin": item.get("origin", "protocol"),
            }
        )
    (root / "blind_samples.json").write_text(encode(samples), encoding="utf-8")
    (root / "blinding_key.json").write_text(encode(key), encoding="utf-8")
    if budget:
        report["resource_usage"] = budget.snapshot()
    if "study" in protocol:
        from evals.study_design import analyze_study

        report["analysis"] = analyze_study(protocol, report)
        (root / "analysis.json").write_text(encode(report["analysis"]), encoding="utf-8")
    (root / "report.json").write_text(encode(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    report = asyncio.run(run_protocol(RuntimeConfig.from_yaml(args.config), protocol, args.output))
    print(
        json.dumps(
            {
                "runs": len(report["runs"]),
                "infrastructure_failures": sum(
                    run["infrastructure_failures"] for run in report["runs"]
                ),
            },
            indent=2,
        )
    )
    return 1 if any(run["infrastructure_failures"] for run in report["runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

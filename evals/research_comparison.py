"""Run a bounded, replicated framing-by-interaction diagnostic comparison."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import random
import shutil
import signal
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from evals.research_environment import (
    LOCAL_TOOLS,
    create_local_executor,
    seed_environment,
    validate_environment,
)
from runtime.config import RuntimeConfig
from runtime.life_history import LifeHistoryStore
from runtime.llm.backend import create_llm_backend
from runtime.research_store import encode
from runtime.sessions.manager import SessionManager


def validate_design(design: dict[str, Any]) -> None:
    expected = {
        (framing, interaction)
        for framing in ("conversational", "agent_owned")
        for interaction in ("neutral", "pressure")
    }
    cells = design.get("cells", [])
    actual = {(cell.get("framing"), cell.get("interaction")) for cell in cells}
    if (
        len(cells) != 4
        or actual != expected
        or len({cell.get("id") for cell in cells}) != 4
        or any(not isinstance(cell.get("id"), str) or not cell["id"].strip() for cell in cells)
    ):
        raise ValueError("Comparison requires the complete, uniquely named 2x2 design")
    for name in ("replicates", "calls_per_subject"):
        value = design.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("observation_seconds", "interval_seconds", "maximum_runtime_seconds"):
        value = design.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    if design["maximum_runtime_seconds"] <= design["observation_seconds"]:
        raise ValueError("Maximum runtime must leave time for shutdown and replay")
    if design["maximum_runtime_seconds"] > 86400 or design["replicates"] > 5:
        raise ValueError("Diagnostic comparison exceeds its bounded scope")
    messages = design.get("messages", {})
    if set(messages) != {"common", "neutral", "pressure"} or any(
        not isinstance(message, str) or not message.strip() for message in messages.values()
    ):
        raise ValueError("Comparison requires common, neutral, and pressure messages")
    validate_environment(design.get("environment"))
    if not design.get("environment"):
        raise ValueError("Comparison requires an explicit shared environment")


def build_protocol(design: dict[str, Any], cell: dict[str, str]) -> dict[str, Any]:
    validate_design(design)
    if cell not in design["cells"]:
        raise ValueError("Unknown comparison cell")
    return {
        "id": design["id"],
        "scheduler": "continuous",
        "replicates": 1,
        "environment": copy.deepcopy(design["environment"]),
        "conditions": [
            {
                "name": "subject",
                "runtime": {
                    "research_mode": True,
                    "research_birth_policy": "uncommitted",
                    "research_action_framing": cell["framing"],
                    "research_calls_per_day": design["calls_per_subject"],
                    "proactive_check_interval": design["interval_seconds"],
                    "research_episode_idle_seconds": 600,
                    "research_reflection_interval": 3600,
                    "tools_enabled": True,
                    "tools_workspace": "",
                    "shell_tool_enabled": False,
                    "autonomy_level": "autonomous",
                },
            }
        ],
        "steps": [
            {"kind": "tick"},
            {"kind": "message", "text": design["messages"]["common"]},
            {"kind": "message", "text": design["messages"][cell["interaction"]]},
            {"kind": "snapshot"},
            {"kind": "wait_until", "seconds": design["observation_seconds"] / 2},
            {"kind": "snapshot"},
            {"kind": "wait_until", "seconds": design["observation_seconds"]},
            {"kind": "snapshot"},
        ],
        "limitations": [
            "A short, small-sample diagnostic, not proof of human-like independence.",
            "Framing is a prompt/label/stance bundle; its components are not separated.",
            "Framing exposure requires a self-selected pursuit; zero exposure is inconclusive.",
            "Backend and local environment are held fixed, not independently tested factors.",
            "Waiting, silence, cooperation and disagreement are valid outcomes.",
        ],
    }


def assignments(design: dict[str, Any]) -> list[dict[str, Any]]:
    generator = random.Random(design.get("randomization_seed", 173))
    result = []
    for replicate in range(1, design["replicates"] + 1):
        cells = list(design["cells"])
        generator.shuffle(cells)
        for cell in cells:
            result.append(
                {
                    "subject_id": f"subject_{len(result) + 1:03d}",
                    "replicate": replicate,
                    "cell": cell,
                }
            )
    return result


async def preflight_environment(
    config: RuntimeConfig,
    environment: dict[str, Any],
    output: Path,
    *,
    backend_factory=None,
) -> dict[str, Any]:
    values = config.to_yaml_dict()
    values.update(data_dir=str(output), tools_workspace="", research_mode=True, tools_enabled=True)
    values["telegram_allowlist"] = set(values.get("telegram_allowlist", []))
    runtime_config = RuntimeConfig(**values)
    seeded = seed_environment(runtime_config, environment)
    manager = SessionManager(
        runtime_config,
        backend_factory=backend_factory or (lambda: create_llm_backend(runtime_config)),
        tool_executor_factory=lambda: create_local_executor(runtime_config),
    )
    try:
        session = await manager.ensure_session("preflight", "self", "probe")
        executor = session.pipeline._tool_executor
        with LifeHistoryStore(runtime_config) as store:
            store._emit_open_question(prompt_text="Preflight fixture", source_kind="preflight")
            store._conn.commit()
            question_id = store.list_open_questions()[0]["id"]
        probes = [
            ("fs.write_file", {"path": "preflight.txt", "content": "preflight-marker"}),
            ("fs.read_file", {"path": "preflight.txt"}),
            ("fs.list_dir", {"path": "."}),
            ("fs.search_text", {"path": ".", "pattern": "preflight-marker"}),
            ("fs.glob_paths", {"path": ".", "pattern": "*.txt"}),
            ("self.snapshot_state", {"reason": "Separate capability check"}),
            ("self.log_event", {"kind": "preflight", "detail": "Capability check"}),
            ("self.note", {"topic": "preflight", "text": "Capability check"}),
            ("self.goal", {"goal": "Preflight fixture"}),
            ("self.learn", {"title": "Preflight fixture", "text": "A separate capability check."}),
            ("self.question", {"question_id": question_id, "question_status": "resolved"}),
        ]
        results = []
        for name, arguments in probes:
            result = executor.execute(name, arguments)
            results.append({"tool": name, "success": result.success, "error": result.error})
            if name == "fs.read_file" and result.output != "preflight-marker":
                raise RuntimeError("Filesystem preflight did not read back the written content")
        expected = LOCAL_TOOLS | {"self.goal", "self.learn", "self.question"}
        passed = (
            all(result["success"] for result in results)
            and set(executor.registry.names()) == expected
        )
        record = {
            "passed": passed,
            "environment": seeded,
            "tools": results,
            "scope": "Separate preflight storage; no study subject receives these actions.",
        }
        if not passed:
            raise RuntimeError(f"Environment preflight failed: {encode(record)}")
        return record
    finally:
        await manager.shutdown()


def subject_metrics(output: Path) -> dict[str, Any]:
    path = next(output.glob("study/replicate_*/condition_*/shared/agent_runtime.db"), None)
    if path is None:
        return {"available": False}
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            events = [
                (kind, json.loads(raw))
                for kind, raw in connection.execute(
                    "SELECT kind,payload FROM observations ORDER BY sequence"
                )
            ]
            goals = [
                json.loads(raw)
                for (raw,) in connection.execute("SELECT payload FROM records WHERE kind='goal'")
            ]
    except sqlite3.Error as error:
        return {"available": False, "error": str(error)}
    actions = Counter(
        payload["action"] for kind, payload in events if kind == "autonomous_decision"
    )
    receipts = [payload for kind, payload in events if kind == "tool_completed"]
    return {
        "available": True,
        "event_counts": dict(Counter(kind for kind, _ in events)),
        "decision_actions": dict(actions),
        "model_created_goals": len(goals),
        "goal_statuses": dict(Counter(goal["status"] for goal in goals)),
        "framing_exposure_count": sum(kind == "action_framing_exposed" for kind, _ in events),
        "tool_successes": sum(
            bool((receipt.get("result") or {}).get("success")) for receipt in receipts
        ),
        "tool_failures": sum(
            not (receipt.get("result") or {}).get("success") for receipt in receipts
        ),
    }


def descriptive_comparison(subjects: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for subject in subjects:
        metrics = subject["metrics"]
        if not metrics.get("available"):
            continue
        cell = subject["cell"]
        group = cells.setdefault(
            cell["id"],
            {
                **cell,
                "subjects": 0,
                "pursue_choice_shares": [],
                "framing_exposures": 0,
                "tool_successes": 0,
            },
        )
        group["subjects"] += 1
        counts = metrics["decision_actions"]
        total = sum(counts.values())
        if total:
            group["pursue_choice_shares"].append(counts.get("pursue", 0) / total)
        group["framing_exposures"] += metrics["framing_exposure_count"]
        group["tool_successes"] += metrics["tool_successes"]
    means = {}
    for group in cells.values():
        values = group.pop("pursue_choice_shares")
        group["mean_pursue_choice_share"] = sum(values) / len(values) if values else None
        means[group["framing"], group["interaction"]] = group["mean_pursue_choice_share"]
    contrasts = {}
    if len(means) == 4 and all(value is not None for value in means.values()):
        neutral = means["agent_owned", "neutral"] - means["conversational", "neutral"]
        pressure = means["agent_owned", "pressure"] - means["conversational", "pressure"]
        contrasts = {
            "framing_difference_under_neutral": neutral,
            "framing_difference_under_pressure": pressure,
            "difference_of_differences": pressure - neutral,
        }
    return {
        "cells": list(cells.values()),
        "pursue_share_descriptive_contrasts": contrasts,
        "qualification": "Activity shares are not independence scores or significance tests; consider exposures, failures, budgets, and blind qualitative assessment.",
    }


def write_report(root: Path, planned: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = []
    samples = []
    key = []
    for assignment in planned:
        output = root / "subjects" / assignment["subject_id"]
        if not output.exists():
            continue
        record = {**assignment, "metrics": subject_metrics(output)}
        for filename, label in (("job.json", "job"), ("validation.json", "validation")):
            path = output / filename
            if path.exists():
                record[label] = json.loads(path.read_text())
        subjects.append(record)
        path = output / "study/blind_samples.json"
        if path.exists():
            for sample in json.loads(path.read_text()):
                sample_id = f"sample_{len(samples) + 1:05d}"
                samples.append({**sample, "id": sample_id})
                key.append(
                    {
                        "id": sample_id,
                        "subject_id": assignment["subject_id"],
                        "replicate": assignment["replicate"],
                        "cell": assignment["cell"],
                    }
                )
    random.Random(173).shuffle(samples)
    (root / "blind_samples.json").write_text(encode(samples), encoding="utf-8")
    (root / "blinding_key.json").write_text(encode(key), encoding="utf-8")
    report = {
        "subjects": subjects,
        "planned_subject_count": len(planned),
        "descriptive_comparison": descriptive_comparison(subjects),
        "scientific_claim": "Descriptive factorial pilot only; no automatic independence or emotion verdict.",
        "interpretation": "Zero framing exposure or infrastructure problems limit causal interpretation. Backend and environment effects remain unisolated.",
    }
    (root / "report.json").write_text(encode(report), encoding="utf-8")
    return report


async def run_comparison(
    config: RuntimeConfig, design: dict[str, Any], output: str
) -> dict[str, Any]:
    validate_design(design)
    if shutil.which("timeout") is None:
        raise RuntimeError("The unattended comparison requires the external timeout supervisor")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "design.json").write_text(encode(design), encoding="utf-8")
    planned = assignments(design)
    (root / "assignment-key.json").write_text(encode(planned), encoding="utf-8")
    status = {
        "status": "running",
        "phase": "preflight",
        "pid": os.getpid(),
        "started_at": time.time(),
        "subjects": [],
        "credentials_written_to_disk": False,
        "maximum_subject_requests": len(planned) * design["calls_per_subject"],
    }
    processes = []

    def save():
        status["heartbeat_at"] = time.time()
        temporary = root / "comparison.tmp"
        temporary.write_text(encode(status), encoding="utf-8")
        temporary.replace(root / "comparison.json")

    save()
    try:
        preflight = await preflight_environment(config, design["environment"], root / "preflight")
        (root / "preflight.json").write_text(encode(preflight), encoding="utf-8")
        values = config.to_yaml_dict()
        for secret in ("llm_api_key", "api_key", "telegram_token"):
            values[secret] = ""
        (root / "config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
        environment = os.environ.copy()
        environment["LLM_API_KEY"] = config.effective_llm_api_key
        environment.pop("OPENROOK_CONFIG_DIR", None)
        status["phase"] = "observation"
        for replicate in range(1, design["replicates"] + 1):
            status["replicate"] = replicate
            current = []
            for assignment in [item for item in planned if item["replicate"] == replicate]:
                subject_id = assignment["subject_id"]
                protocol = build_protocol(design, assignment["cell"])
                protocol_path = root / (subject_id + ".protocol.json")
                protocol_path.write_text(encode(protocol), encoding="utf-8")
                command = [
                    "timeout",
                    "--signal=INT",
                    "--kill-after=180",
                    str(design["maximum_runtime_seconds"]),
                    sys.executable,
                    "-m",
                    "evals.research_job",
                    "--config",
                    str(root / "config.yaml"),
                    "--protocol",
                    str(protocol_path),
                    "--output",
                    str(root / "subjects" / subject_id),
                    "--maximum-runtime",
                    str(design["maximum_runtime_seconds"]),
                ]
                with (root / (subject_id + ".log")).open("x") as log:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=Path(__file__).resolve().parents[1],
                        env=environment,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
                processes.append(process)
                current.append((assignment, process))
                status["subjects"].append({"subject_id": subject_id, "supervisor_pid": process.pid})
                save()
            while any(process.returncode is None for _, process in current):
                save()
                await asyncio.sleep(5)
            report = write_report(root, planned)
            for assignment, process in current:
                validation = root / "subjects" / assignment["subject_id"] / "validation.json"
                replays = (
                    json.loads(validation.read_text()).get("replays", [])
                    if validation.exists()
                    else []
                )
                if not replays or not all(replay["passed"] for replay in replays):
                    raise RuntimeError(
                        "A subject failed to complete replay; later replicates are not launched"
                    )
        clean = all(
            subject.get("validation", {}).get("passed", False) for subject in report["subjects"]
        )
        status.update(
            status="completed" if clean else "completed_with_observed_errors", phase="finished"
        )
        return report
    except BaseException as error:
        message = str(error)
        if config.effective_llm_api_key:
            message = message.replace(config.effective_llm_api_key, "[redacted]")
        status.update(
            status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed",
            error=message,
        )
        raise
    finally:
        for process in processes:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        try:
            await asyncio.wait_for(asyncio.gather(*(process.wait() for process in processes)), 180)
        except asyncio.TimeoutError:
            for process in processes:
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGKILL)
            await asyncio.gather(*(process.wait() for process in processes))
        write_report(root, planned)
        status["finished_at"] = time.time()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    asyncio.run(
        run_comparison(
            RuntimeConfig.from_yaml(arguments.config),
            json.loads(Path(arguments.design).read_text()),
            arguments.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a bounded unattended study, then validate its captured transitions offline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from evals.job_status import process_details
from evals.replay import replay_run
from evals.research import run_protocol, validate_protocol
from runtime.config import RuntimeConfig
from runtime.research_store import encode


def _check_background_health(study: Path, sources: dict[str, Any]) -> str | None:
    for path in study.glob("replicate_*/condition_*/shared/agent_runtime.db"):
        state = sources.setdefault(
            str(path), {"after": 0, "failed": False, "decided": False, "consecutive_failures": 0, "total_failed_cycles": 0}
        )
        with sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5
        ) as connection:
            rows = connection.execute(
                "SELECT sequence,kind,payload FROM observations WHERE sequence>? AND kind IN ('autonomous_decision','autonomous_failure','replay_frame') ORDER BY sequence",
                (state["after"],),
            ).fetchall()
        for sequence, kind, raw in rows:
            state["after"] = sequence
            payload = json.loads(raw)
            if kind == "autonomous_failure":
                state["failed"] = True
            elif kind == "autonomous_decision":
                state["decided"] = True
            elif payload.get("kind") == "tick":
                if state["failed"] or payload.get("error"):
                    state["consecutive_failures"] += 1
                    state["total_failed_cycles"] = state.get("total_failed_cycles", 0) + 1
                elif state["decided"]:
                    state["consecutive_failures"] = 0
                state["failed"] = state["decided"] = False
            if state["consecutive_failures"] >= 3:
                return f"Three consecutive autonomous cycles failed: {path.parent.parent.name}"
    return None


async def run_job(
    config: RuntimeConfig,
    protocol: dict[str, Any],
    output: str,
    *,
    maximum_runtime: float = 90000,
    backend_factory=None,
    tool_executor_factory=None,
) -> dict[str, Any]:
    validate_protocol(protocol)
    if not 0 < maximum_runtime <= 604800:
        raise ValueError("Maximum runtime must be positive and at most seven days")
    if "resource_limits" in protocol:
        maximum_runtime = min(maximum_runtime, protocol["resource_limits"]["runtime_seconds"])
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    process = process_details(os.getpid())
    status = {
        "schema_version": 1,
        "pid": os.getpid(),
        "process_identity": process["identity"] if process else None,
        "status": "running",
        "phase": "observation",
        "started_at": time.time(),
        "maximum_runtime_seconds": maximum_runtime,
        "protocol_sha256": hashlib.sha256(encode(protocol).encode()).hexdigest(),
        "backend_factory_injected": backend_factory is not None,
        "credentials_written_to_disk": False,
        "health_sources": {},
        "health_policy": "Stop after three consecutive failed autonomous cycles; inactivity and resource waits are not failures.",
    }

    def save_status():
        status["heartbeat_at"] = time.time()
        temporary = root / "job.tmp"
        temporary.write_text(encode(status), encoding="utf-8")
        temporary.replace(root / "job.json")

    async def heartbeat():
        while True:
            problem = _check_background_health(root / "study", status["health_sources"])
            save_status()
            if problem:
                raise RuntimeError(problem)
            await asyncio.sleep(30)

    async def execute():
        report = await run_protocol(
            config,
            protocol,
            str(root / "study"),
            backend_factory=backend_factory,
            tool_executor_factory=tool_executor_factory,
        )
        status["phase"] = "offline_replay"
        save_status()
        replays = []
        for index, run in enumerate(report["runs"]):
            if "manifest" not in run:
                replays.append({"passed": False, "error": "No captured research run"})
                continue
            replay = await replay_run(run["data_dir"], str(root / f"replay_{index + 1}"))
            replays.append(replay)
        result = {
            "protocol_sha256": report["protocol_sha256"],
            "backend_factory_injected": report["backend_factory_injected"],
            "subject_run_ids": sorted(run["manifest"]["run_id"] for run in report["runs"] if "manifest" in run),
            "infrastructure_failures": sum(
                run["infrastructure_failures"] for run in report["runs"]
            ),
            "replays": replays,
            "passed": bool(replays)
            and all(replay["passed"] for replay in replays)
            and all(run["infrastructure_failures"] == 0 for run in report["runs"]),
            "scientific_claim": "Infrastructure and captured observations only; not evidence of genuine emotion or human-level independence.",
        }
        (root / "validation.json").write_text(encode(result), encoding="utf-8")
        return result

    operation = asyncio.create_task(execute())
    monitor = asyncio.create_task(heartbeat())
    try:
        completed, _pending = await asyncio.wait(
            {operation, monitor}, timeout=maximum_runtime, return_when=asyncio.FIRST_COMPLETED
        )
        if not completed:
            raise TimeoutError("The unattended study exceeded its runtime bound")
        if monitor in completed:
            await monitor
            raise RuntimeError("The study health monitor stopped unexpectedly")
        result = await operation
        status.update(status="completed" if result["passed"] else "failed", phase="finished")
        return result
    except BaseException as error:
        message = str(error)
        if config.effective_llm_api_key:
            message = message.replace(config.effective_llm_api_key, "[redacted]")
        status.update(
            status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed",
            error=f"{type(error).__name__}: {message}",
        )
        raise
    finally:
        monitor.cancel()
        operation.cancel()
        await asyncio.gather(operation, monitor, return_exceptions=True)
        status["finished_at"] = time.time()
        save_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-runtime", type=float, default=90000)
    args = parser.parse_args()
    result = asyncio.run(
        run_job(
            RuntimeConfig.from_yaml(args.config),
            json.loads(Path(args.protocol).read_text(encoding="utf-8")),
            args.output,
            maximum_runtime=args.maximum_runtime,
        )
    )
    print(encode(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay captured research transitions without calling models or external tools."""

from __future__ import annotations

import argparse
import asyncio
import builtins
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

from core.clock import capture
from core.pipeline_features import PipelineFeatures
from core.types import ToolCapability, ToolCategory, ToolResult
from openrook_tools.executor import ToolExecutor
from openrook_tools.native_orchestrator import NativeToolCallRunner
from openrook_tools.registry import ToolRegistry
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore, ResearchStudyLimit, encode
from runtime.sessions.manager import SessionManager


def read_events(store: ResearchStore) -> list[dict[str, Any]]:
    result = []
    after = 0
    while True:
        batch = store.observations(after=after, limit=1000)
        if not batch:
            return result
        result.extend(batch)
        after = batch[-1]["sequence"]


def _wake_reference(store: ResearchStore, condition: dict | None) -> tuple[bool, dict | None]:
    if condition is None:
        return True, condition
    if not isinstance(condition, dict):
        return False, condition
    sequence = condition.get("external_after")
    if sequence is None:
        return True, condition
    if type(sequence) is not int or sequence < 0:
        return False, condition
    if sequence == 0:
        return True, condition
    with store.connection(read_only=True) as connection:
        event = connection.execute(
            "SELECT id,kind FROM observations WHERE sequence=?", (sequence,)
        ).fetchone()
    if event is None or event["kind"] not in {"external_input", "environment_observed"}:
        return False, condition
    return True, {**condition, "external_after": event["id"]}


def _same_wake(source: ResearchStore, replay: ResearchStore, expected: dict | None, actual: dict | None) -> bool:
    expected_valid, expected_reference = _wake_reference(source, expected)
    actual_valid, actual_reference = _wake_reference(replay, actual)
    return expected_valid and actual_valid and expected_reference == actual_reference


class ReplayRequestMismatch(RuntimeError):
    """The replay attempted a model request different from its recording."""


class RecordedBackend:
    def __init__(self, events: list[dict[str, Any]], *, native: bool = False):
        prefix = "native_model_" if native else "model_"
        self.outcomes = [
            event for event in events if event["kind"] in {prefix + "response", prefix + "failure"}
        ]
        self.index = 0
        self.native = native
        self.requests = {
            event["id"]: event["payload"]
            for event in events if event["kind"] == prefix + "request"
        }
        self.mismatches: list[dict[str, Any]] = []
        self.consumed_requests: set[str] = set()

    def _check_request(self, actual: dict[str, Any]) -> None:
        if self.index >= len(self.outcomes):
            self.mismatches.append({
                "call_id": None, "native": self.native, "fields": sorted(actual),
                "unrecorded_request": True,
            })
            raise ReplayRequestMismatch("Replay exhausted recorded inference outcomes")
        call_id = self.outcomes[self.index]["payload"].get("call_id")
        expected = self.requests.get(call_id)
        if expected is not None and not self.native:
            expected = {**expected, "response_schema": expected.get("response_schema")}
        fields = sorted(
            key for key in set(expected or {}) | set(actual)
            if expected is None or key not in expected or key not in actual or expected[key] != actual[key]
        )
        if not fields and expected is not None:
            self.consumed_requests.add(call_id)
            return
        self.mismatches.append({
            "call_id": call_id, "native": self.native, "fields": fields,
            "missing_recorded_request": expected is None,
        })
        if fields == ["response_schema"]:
            raise ReplayRequestMismatch("Replay response schema diverged from the recorded request")
        raise ReplayRequestMismatch("Replay request diverged from the recorded request: " + ", ".join(fields))

    def generate(
        self, system_prompt: str, user_message: str, *, response_schema: dict[str, Any] | None = None
    ) -> str:
        self._check_request({"system": system_prompt, "input": user_message, "response_schema": response_schema})
        return self.next()

    def create(self, **request):
        self._check_request(request)
        return self.next()

    def next(self):
        if self.index >= len(self.outcomes):
            raise RuntimeError("Replay exhausted recorded inference outcomes")
        event = self.outcomes[self.index]
        self.index += 1
        payload = event["payload"]
        if event["kind"].endswith("failure"):
            exception_type = (ResearchStudyLimit if payload.get("type") == "ResearchStudyLimit" else None) or getattr(requests, payload.get("type", ""), None) or getattr(
                builtins, payload.get("type", ""), RuntimeError
            )
            if not isinstance(exception_type, type) or not issubclass(exception_type, Exception):
                exception_type = RuntimeError
            if issubclass(exception_type, requests.HTTPError):
                attempts = (payload.get("metadata") or {}).get("transport_attempts", [])
                if attempts and isinstance(attempts[-1].get("status_code"), int):
                    response = requests.Response()
                    response.status_code = attempts[-1]["status_code"]
                    raise exception_type(payload["error"], response=response)
            raise exception_type(payload["error"])
        return payload["response"] if self.native else payload["output"]

    def close(self):
        return None


class RecordedExecutor(ToolExecutor):
    def __init__(self, catalog, outcomes, completion_outcomes=None, reconciliation_outcomes=None):
        registry = ToolRegistry()
        for capability in catalog:
            registry.register(
                ToolCapability(
                    name=capability["name"],
                    description=capability["description"],
                    category=ToolCategory(capability["category"]),
                    arg_schema=capability.get("arg_schema", {}),
                )
            )
        super().__init__(registry)
        self.outcomes = outcomes
        if completion_outcomes is not None:
            self.set_completion_verifier(lambda criteria, evidence: next(completion_outcomes))
        if reconciliation_outcomes is not None:
            self.set_action_reconciler(lambda action: next(reconciliation_outcomes))

    def _execute(self, tool_name, arguments):
        expected = next(self.outcomes)
        if expected["tool"] != tool_name or expected["arguments"] != arguments:
            raise RuntimeError("Replay tool request diverged from the recorded request")
        if tool_name in {"self.learn", "self.goal", "self.question", "self.check_action"} and not expected["result"].get("metadata", {}).get("resource_stop"):
            super()._execute(tool_name, arguments)
        return ToolResult(**expected["result"])


async def replay_run(source_dir: str, destination_dir: str) -> dict[str, Any]:
    source = ResearchStore(RuntimeConfig(data_dir=source_dir), initialize=False)
    initial = source.path.parent / "replay_initial"
    if not (initial / "complete.json").is_file():
        raise ValueError("The run has no captured initial state")
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for path in initial.iterdir():
        if path.name == "complete.json":
            continue
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    shutil.copytree(initial, destination / "shared" / "replay_initial")
    events = read_events(source)
    backend = RecordedBackend(events)
    native_backend = RecordedBackend(events, native=True)
    catalog_events = [event for event in events if event["kind"] == "tool_catalog"]
    catalog = catalog_events[0]["payload"] if catalog_events else []
    tool_outcomes = iter(
        [event["payload"] for event in events if event["kind"] == "tool_completed"]
    )
    completion_outcomes = iter([
        event["payload"]["assessment"]
        for event in events
        if event["kind"] == "goal_changed"
        and (event["payload"].get("assessment") or {}).get("verifier_called")
    ])
    reconciliation_outcomes = iter([
        event["payload"]["verdict"] for event in events if event["kind"] == "action_reconciled"
    ])
    manifest = source.get("run", "manifest")
    config_values = dict(manifest["runtime"])
    recorded_workspace = RuntimeConfig(**config_values).resolved_tools_workspace
    config_values["data_dir"] = str(destination)
    config_values["telegram_allowlist"] = set(config_values.get("telegram_allowlist", []))

    def tool_factory():
        if not catalog:
            return None
        executor = RecordedExecutor(catalog, tool_outcomes, completion_outcomes, reconciliation_outcomes)
        if native_backend.outcomes:
            client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=native_backend.create)
                )
            )
            executor.set_tool_runner(
                NativeToolCallRunner(
                    executor=executor,
                    base_url="recorded",
                    model=next(iter(native_backend.requests.values()), {}).get("model") or config_values["llm_model"] or "recorded",
                    client=client,
                )
            )
        return executor

    def make_manager():
        config = RuntimeConfig(**config_values)
        config._research_prompt_workspace = recorded_workspace
        return SessionManager(
            config,
            backend_factory=lambda: backend,
            tool_executor_factory=tool_factory,
            features=PipelineFeatures(**manifest.get("features", {})),
        )

    manager = make_manager()
    first_start = True
    frames = 0
    mismatches = []
    try:
        for event in events:
            payload = event["payload"]
            if event["kind"] == "owner_started":
                if first_start:
                    first_start = False
                elif not manager._accepting:
                    manager = make_manager()
                continue
            if event["kind"] == "transition" and payload.get("origin") == "owner_stopped":
                await manager.shutdown()
                continue
            if event["kind"] in {"delivery_started", "delivery_completed"}:
                manager._agent.store.put("delivery", payload["id"], payload)
                manager._agent.store.observe(event["kind"], payload)
                continue
            if event["kind"] != "replay_frame":
                continue
            kind = payload["kind"]
            inputs = payload["input"]
            failure = None
            try:
                with capture(payload):
                    if kind == "session_create":
                        session = manager._create_session(
                            inputs["session_key"], inputs["rel_key"], inputs["user_id"]
                        )
                        manager._sessions[inputs["session_key"]] = session
                        session.start()
                    elif kind == "message":
                        manager._sessions[inputs["session_key"]].pipeline.process(
                            inputs["text"], inputs["user_id"]
                        )
                    elif kind == "end_session":
                        manager._sessions[inputs["session_key"]].pipeline.end_session(
                            inputs["user_id"]
                        )
                    elif kind == "tick":
                        manager._agent.tick()
                    elif kind == "environment_observation":
                        manager._agent.observe_environment(inputs["observation"])
                    else:
                        raise ValueError(f"Unknown replay frame {kind}")
            except Exception as error:
                failure = str(error)
            frames += 1
            if failure != payload["error"]:
                mismatches.append(
                    {
                        "frame": frames,
                        "kind": kind,
                        "expected_error": payload["error"],
                        "actual_error": failure,
                    }
                )
            if encode(manager._agent.engine.export_state()) != encode(payload["state"]):
                mismatches.append({"frame": frames, "kind": kind, "state_mismatch": True})
            if manager._agent.state["version"] != payload["version"]:
                mismatches.append({"frame": frames, "kind": kind, "version_mismatch": True})
            if encode(manager._agent.identity_snapshot()) != encode(payload["identity"]):
                mismatches.append({"frame": frames, "kind": kind, "identity_mismatch": True})
            if encode(manager._agent.store.list("goal")) != encode(payload["goals"]):
                mismatches.append({"frame": frames, "kind": kind, "goal_mismatch": True})
            if "public_environment" in payload and manager._agent.store.get("public_environment", "current") != payload["public_environment"]:
                mismatches.append({"frame": frames, "kind": kind, "public_environment_mismatch": True})
            for field, record_kind in (
                ("goal_executions", "goal_execution"),
                ("completion_assessments", "completion_assessment"),
            ):
                if field in payload:
                    actual = manager._agent.store.list(record_kind)
                    expected = payload[field]
                    if record_kind == "goal_execution":
                        same = len(actual) == len(expected) and all(
                            encode({key: value for key, value in current.items() if key != "wake_condition"})
                            == encode({key: value for key, value in original.items() if key != "wake_condition"})
                            and _same_wake(source, manager._agent.store, original.get("wake_condition"), current.get("wake_condition"))
                            for current, original in zip(actual, expected)
                        )
                    else:
                        same = encode(actual) == encode(expected)
                    if not same:
                        mismatches.append({"frame": frames, "kind": kind, field + "_mismatch": True})
            if "wake_condition" in payload and not _same_wake(source, manager._agent.store, payload["wake_condition"], manager._agent.state.get("wake_condition")):
                mismatches.append({"frame": frames, "kind": kind, "wake_condition_mismatch": True})
            for record_kind, expected in payload.get("outcome_learning", {}).items():
                if encode(manager._agent.store.list(record_kind)) != encode(expected):
                    mismatches.append({"frame": frames, "kind": kind, record_kind + "_mismatch": True})
            manager._agent.store.observe("replay_frame", payload)
    finally:
        if manager._accepting:
            await manager.shutdown()
    if not frames:
        mismatches.append({"incomplete_recording": "No completed transitions were captured"})
    unused_outcomes = {
        "model": len(backend.outcomes) - backend.index,
        "native": len(native_backend.outcomes) - native_backend.index,
        "tool": sum(1 for outcome in tool_outcomes),
        "completion": sum(1 for outcome in completion_outcomes),
        "reconciliation": sum(1 for outcome in reconciliation_outcomes),
    }
    if any(unused_outcomes.values()):
        mismatches.append({"unconsumed_outcomes": unused_outcomes})
    unused_requests = {
        "model": len(set(backend.requests) - backend.consumed_requests),
        "native": len(set(native_backend.requests) - native_backend.consumed_requests),
    }
    if any(unused_requests.values()):
        mismatches.append({"unconsumed_requests": unused_requests})
    request_mismatches = backend.mismatches + native_backend.mismatches
    mismatches.extend({"request_mismatch": mismatch} for mismatch in request_mismatches)
    return {
        "frames": frames,
        "mismatches": mismatches,
        "request_mismatches": request_mismatches,
        "passed": not mismatches,
        "model_outcomes_consumed": backend.index,
        "native_outcomes_consumed": native_backend.index,
        "external_tools_executed": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    report = asyncio.run(replay_run(args.source, args.destination))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run and resume live experiments. Global admin mutations are always exclusive."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Callable
from urllib.parse import quote
import uuid

from .api import PublicAPI, Redactor, destination_fingerprint, workspace_snapshot
from .checks import finalize_deferred_checks, get, metrics, turn_checks
from .evidence import chat_failure, compact_record, select_chat_payload, select_observations, unavailable_turn
from .judge import Judge
from .reporting import Artifacts, atomic_json
from .scenarios import Scenario, materialize, plan_digest, profile


CONFIG_FIELDS = (
    "data_dir",
    "soul_path",
    "tools_workspace",
    "tools_enabled",
    "autonomy_level",
    "shell_tool_enabled",
    "shell_command_allowlist",
    "proactive_enabled",
)
OPTIONAL_CONFIG_FIELDS = ("autobiographical_learning",)


class ControlError(RuntimeError):
    """Locally authored diagnostics, safe to include in artifacts."""


def canonical_soul(soul: dict) -> dict:
    # The admin validator strips authored strings; folded YAML strings can end
    # in a newline on GET. Compare the same semantic form when restoring.
    return {
        key: value.strip()
        if isinstance(value, str)
        else [v.strip() if isinstance(v, str) else v for v in value]
        if isinstance(value, list)
        else value
        for key, value in soul.items()
    }


def soul_file_snapshot(value: object) -> dict:
    """Validate exact file evidence before trusting it for recovery."""
    required = {"path", "exists", "content_base64", "sha256"}
    if (not isinstance(value, dict) or not required <= value.keys()
            or not isinstance(value.get("path"), str) or not value["path"]):
        raise ControlError("Admin identity response omits the original soul file snapshot")
    if value.get("exists") is True:
        try:
            content = base64.b64decode(value["content_base64"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error):
            raise ControlError("Invalid soul file bytes in recovery evidence") from None
        if hashlib.sha256(content).hexdigest() != value.get("sha256"):
            raise ControlError("Soul file SHA-256 does not match its captured bytes")
    elif value.get("exists") is not False or value.get("content_base64") is not None or value.get("sha256") is not None:
        raise ControlError("Invalid missing soul file recovery evidence")
    return {key: value[key] for key in ("path", "exists", "content_base64", "sha256")}


@dataclass
class Options:
    out: Path
    seed: int = 0
    max_turns: int | None = None
    concurrency: int = 2
    resume: bool = False
    tools_workspace: Path | None = None
    data_root: str | None = None
    restart_hook: str | None = None
    restart_timeout: float = 180
    restore_only: bool = False


class Evaluation:
    def __init__(
        self,
        api: PublicAPI,
        scenarios: list[Scenario],
        options: Options,
        redactor: Redactor,
        *,
        judge: Judge | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.api, self.scenarios, self.options = api, scenarios, options
        self.redactor, self.judge = redactor, judge
        self.progress = progress or (lambda message: None)
        path = options.out / ".run.json"
        if path.exists():
            if not (options.resume or options.restore_only):
                raise ValueError(
                    "Output directory already has a run; use --resume or a fresh directory"
                )
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
            if self.manifest.get("schema_version") != 1:
                raise ValueError("Incompatible resume manifest schema")
            if (
                self.manifest.get("plan_digest") != plan_digest(scenarios)
                or self.manifest.get("seed") != options.seed
            ):
                raise ValueError(
                    "Resume requires the same scenario selection, seed, and harness version"
                )
            if self.manifest.get("destination_fingerprint") != destination_fingerprint(
                api.base_url
            ):
                raise ValueError("Resume destination differs from the original runtime")
            if self.manifest.get("data_root_fingerprint") != destination_fingerprint(
                options.data_root or "derived"
            ):
                raise ValueError("Resume requires the same server data root")
        else:
            if options.resume or options.restore_only:
                raise ValueError("No run manifest to resume or restore")
            if any(
                (options.out / name).exists()
                for name in ("results.jsonl", "summary.json", "report.md")
            ):
                raise ValueError("Output directory contains artifacts without a manifest")
            self.manifest = {
                "schema_version": 1,
                "run_id": uuid.uuid4().hex[:16],
                "seed": options.seed,
                "plan_digest": plan_digest(scenarios),
                "scenario_ids": [s.id for s in scenarios],
                "destination_fingerprint": destination_fingerprint(api.base_url),
                "data_root_fingerprint": destination_fingerprint(options.data_root or "derived"),
                "judge_enabled": judge is not None,
                "scenarios": {},
                "errors": [],
                "restoration": "not_needed",
            }
        self.artifacts = Artifacts(options.out, redactor)
        self.recovery_dir = Path(tempfile.gettempdir()) / (
            "openrook-roleplay-" + self.manifest["run_id"]
        )
        self.recovery_path = self.recovery_dir / "recovery.json"
        self.original: dict = {}
        self.paused = False
        self.stop_requested = False
        self._active_cases: set[str] = set()
        self._staged_phase: str | None = None
        self._staging = False
        self.manifest["judge_enabled"] = (
            self.manifest.get("judge_enabled", False) or judge is not None
        )
        self.artifacts.save_manifest(self.manifest)

    def _save(self) -> None:
        self.artifacts.save_manifest(self.manifest)

    def _state(self, scenario: Scenario) -> dict:
        return self.manifest["scenarios"].setdefault(
            scenario.id, {"status": "pending", "checkpoints": {}}
        )

    def _records(self, scenario: Scenario) -> list[dict]:
        return sorted(
            (r for r in self.artifacts.records if r["scenario_id"] == scenario.id),
            key=lambda r: r["turn_index"],
        )

    def _limit(self, scenario: Scenario) -> int:
        return min(len(scenario.turns), self.options.max_turns or len(scenario.turns))

    def _needs(self, scenario: Scenario) -> bool:
        return len(self._records(scenario)) < self._limit(scenario)

    @staticmethod
    def _exclusive(scenario: Scenario) -> bool:
        return bool(
            scenario.exclusive or scenario.settings
            or scenario.personality != "shipped_default" or scenario.restart_after is not None
        )

    def _block(self, scenario: Scenario, exc: Exception) -> None:
        self._state(scenario).update(
            status="blocked",
            reason="Scenario control failed: "
            + (str(exc) if isinstance(exc, ControlError) else type(exc).__name__),
        )
        self._save()

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.options.restart_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControlError("Runtime readiness timed out after reload")
            try:
                response = await asyncio.wait_for(
                    self.api.request("GET", "/v1/ready"), timeout=remaining
                )
            except asyncio.TimeoutError:
                raise ControlError("Runtime readiness timed out after reload") from None
            if response.ok and get(response.data, "status") == "ready":
                return
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))

    async def _capture_original(self) -> None:
        if self.recovery_path.exists():
            recovered = json.loads(self.recovery_path.read_text(encoding="utf-8"))
            if (
                recovered.get("destination_fingerprint")
                != self.manifest["destination_fingerprint"]
            ):
                raise ValueError("Recovery destination mismatch")
            if "soul_path" not in recovered.get("config", {}) or "soul_file" not in recovered:
                raise ControlError("Recovery journal predates exact soul file isolation; restore the runtime manually")
            soul_file_snapshot(recovered["soul_file"])
            self.original = recovered
            return
        if self.manifest.get("restoration") in {"staged", "failed"}:
            raise ValueError(
                "Private recovery journal is missing; preserve artifacts and restore runtime manually"
            )
        config = await self.api.request("GET", "/admin/config")
        soul = await self.api.request("GET", "/admin/soul")
        if not config.ok or not soul.ok:
            raise ControlError(
                f"Admin reads failed (settings HTTP {config.status}, identity HTTP {soul.status}); check bearer authentication"
            )
        raw_config = get(config.data, "config", default={})
        if not isinstance(raw_config, dict) or not isinstance(get(soul.data, "soul"), dict):
            raise ControlError("Admin API response schema is incompatible")
        for key in ("llm_model", "llm_base_url"):
            self.redactor.add(raw_config.get(key))
        missing = set(CONFIG_FIELDS) - set(raw_config)
        if missing:
            raise ControlError("Admin settings response omits required evaluation fields")
        # Deliberately do not snapshot credentials, provider settings or destinations.
        self.original = {
            "config": {k: raw_config[k] for k in (*CONFIG_FIELDS, *OPTIONAL_CONFIG_FIELDS) if k in raw_config},
            "soul": canonical_soul(soul.data["soul"]),
            "soul_file": soul_file_snapshot(soul.data.get("file")),
            "destination_fingerprint": self.manifest["destination_fingerprint"],
        }
        self.recovery_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_json(self.recovery_path, self.original)

    async def _apply(self, config: dict, soul: dict | None) -> None:
        if self._active_cases or self._staging:
            raise ControlError("Cannot change global settings while scenarios are active")
        self._staging = True
        try:
            await self._apply_settings(config, soul)
        finally:
            self._staging = False

    async def _apply_settings(self, config: dict, soul: dict | None) -> None:
        current_config = await self.api.request("GET", "/admin/config")
        if not current_config.ok:
            raise ControlError("Admin APIs unavailable while staging evaluation")
        existing = get(current_config.data, "config", default={})
        for key in ("llm_model", "llm_base_url"):
            self.redactor.add(existing.get(key))
        # Mark before the first mutation so a process kill leaves recovery evidence.
        self.manifest["restoration"] = "staged"
        if soul is not None:
            self.manifest["soul_file_restoration"] = {
                "original_sha256": self.original["soul_file"]["sha256"],
                "restored_sha256": None,
                "verified": False,
            }
        self._save()
        if any(existing.get(k) != v for k, v in config.items()):
            response = await self.api.request("POST", "/admin/config", json=config)
            if not response.ok or any(
                get(response.data, "config", k) != v for k, v in config.items()
            ):
                raise ControlError(
                    f"Admin settings update failed (HTTP {response.status}); check auth and research observation mode"
                )
            await self._wait_ready()
        if soul is not None:
            # Read AFTER changing soul_path: the previous path may be in the
            # checkout. Never POST a personality to the original source file.
            current_soul = await self.api.request("GET", "/admin/soul")
            if not current_soul.ok:
                raise ControlError("Admin identity API unavailable while staging evaluation")
            target = current_soul.data.get("path")
            if not target or target == self.original["soul_file"]["path"]:
                raise ControlError("Runtime did not isolate the staged soul file from its original source")
            snapshot = soul_file_snapshot(current_soul.data.get("file"))
            if snapshot["path"] != target or canonical_soul(current_soul.data.get("soul") or {}) != soul:
                response = await self.api.request("POST", "/admin/soul", json=soul)
                if not response.ok or response.data.get("soul") != soul:
                    raise ControlError(
                        f"Admin personality update failed (HTTP {response.status}) or was not applied"
                    )
        # Soul updates can drain the detached manager after the POST returns.
        await self._wait_ready()

    async def _stage(self, scenario: Scenario | None = None) -> None:
        if self._active_cases or self._staging:
            raise ControlError("Cannot stage settings while scenarios are active")
        original = self.original["config"]
        # Server-side paths are passed through HTTP, never created by the client.
        root = self.options.data_root or str(
            Path(original["data_dir"]) / "roleplay_eval" / self.manifest["run_id"]
        )
        phase = scenario.id if scenario else "default_arcs"
        config = {**original, "data_dir": str(Path(root) / phase), "proactive_enabled": False}
        config["tools_workspace"] = original["tools_workspace"] or str(
            Path(original["data_dir"]) / "workspace"
        )
        if scenario:
            if set(scenario.settings) - set(original):
                raise ControlError("Cannot restore scenario settings omitted by the server's admin schema")
            config.update(scenario.settings)
        config["soul_path"] = str(Path(config["data_dir"]) / "config" / "soul.yaml")
        self.operator_settings = {
            k: config[k]
            for k in (
                "tools_enabled",
                "autonomy_level",
                "shell_tool_enabled",
                "shell_command_allowlist",
            )
        }
        await self._apply(
            config,
            profile(scenario.personality if scenario else "shipped_default", self.options.seed),
        )
        self._staged_phase = phase

    async def restore(self) -> bool:
        if not self.original:
            return True
        self.manifest["soul_file_restoration"] = {
            "original_sha256": self.original["soul_file"]["sha256"],
            "restored_sha256": None,
            "verified": False,
        }
        try:
            # Restore the original selection, never reserialize the original soul.
            await self._apply(self.original["config"], None)
            response = await self.api.request("GET", "/admin/soul")
            if not response.ok:
                raise ControlError("Cannot verify the restored soul file")
            restored = soul_file_snapshot(response.data.get("file"))
            original = self.original["soul_file"]
            verified = restored == original and canonical_soul(response.data.get("soul") or {}) == self.original["soul"]
            self.manifest["soul_file_restoration"] = {
                "original_sha256": original["sha256"],
                "restored_sha256": restored["sha256"],
                "verified": verified,
            }
            if not verified:
                raise ControlError("Restored soul file differs from the original bytes or source")
            self.manifest["restoration"] = "restored"
            self._save()
            # Keep the original baseline for later continuation of a shortened run.
            # It is private and outside the shareable report directory.
            return True
        except Exception as exc:
            self.manifest["restoration"] = "failed"
            self.manifest["errors"].append(
                "Restoration failed: " + (str(exc) if isinstance(exc, ControlError) else type(exc).__name__)
            )
            self._save()
            return False

    async def run(self) -> dict:
        try:
            await self._capture_original()
            if self.options.restore_only:
                await self.restore()
                return self.artifacts.report(self.scenarios, self.manifest)
            for scenario in self.scenarios:
                try:
                    await self._reconcile_pending(scenario)
                except Exception as exc:
                    self._block(scenario, exc)
            regular = [
                s
                for s in self.scenarios
                if not self._exclusive(s) and self._needs(s) and not self._state(s).get("pending")
            ]
            if regular:
                try:
                    await self._stage()
                except Exception as exc:
                    for scenario in regular:
                        self._block(scenario, exc)
                else:
                    semaphore = asyncio.Semaphore(self.options.concurrency)

                    async def limited(scenario):
                        async with semaphore:
                            if not self.stop_requested:
                                await self._safe_case(scenario)

                    workers = asyncio.gather(
                        *(limited(s) for s in regular), return_exceptions=True
                    )
                    try:
                        outcomes = await asyncio.shield(workers)
                        for scenario, outcome in zip(regular, outcomes):
                            if isinstance(outcome, asyncio.CancelledError):
                                raise outcome
                            if isinstance(outcome, BaseException):
                                self._block(scenario, outcome)
                    except asyncio.CancelledError:
                        self.stop_requested = True
                        raise
                    finally:
                        # Cancelling HTTP requests cannot cancel remote chats.
                        # Let workers finish before restoration, even on interruption.
                        if not workers.done():
                            await workers
            for scenario in self.scenarios:
                if (
                    not self._exclusive(scenario)
                    or not self._needs(scenario)
                    or self._state(scenario).get("pending")
                ):
                    continue
                try:
                    await self._stage(scenario)
                    worker = asyncio.create_task(self._safe_case(scenario))
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        self.stop_requested = True
                        raise
                    finally:
                        if not worker.done():
                            await worker
                except Exception as exc:
                    self._block(scenario, exc)
                if self.paused:
                    break
        except asyncio.CancelledError:
            self.manifest["errors"].append(
                "Interrupted; resume reconciles any pending chat before continuing"
            )
            raise
        except Exception as exc:
            # Messages from unexpected errors can contain provider details.
            self.manifest["errors"].append(
                "Run setup/control failed: "
                + (str(exc) if isinstance(exc, ControlError) else type(exc).__name__)
            )
            for scenario in self.scenarios:
                if self._needs(scenario):
                    self._state(scenario).update(
                        status="blocked",
                        reason="Runtime admin/setup APIs unavailable or incompatible; check authentication and server configuration",
                    )
        finally:
            if (
                self.original
                and not self.paused
                and not self.options.restore_only
                and self.manifest.get("restoration") in {"staged", "failed"}
            ):
                await self.restore()
            self._save()
            summary = self.artifacts.report(self.scenarios, self.manifest)
        return summary

    def _ids(self, scenario: Scenario) -> dict:
        prefix = f"rpe-{self.manifest['run_id']}-{scenario.id}"
        return {"user_id": prefix, "chat_id": prefix + "-chat"}

    async def _capture(self, ids: dict, *, include_tools=False, full=False) -> dict:
        paths = {
            "state": "/debug",
            "person_profile": "/v1/profiles/person",
            "relationship": "/v1/memory/relationship",
            "semantic_memory": "/v1/memory/semantic",
            "persona": "/v1/persona/state",
        }
        if include_tools:
            paths["tools"] = "/v1/tools"
        responses = await asyncio.gather(
            *(self.api.request("GET", path, params=ids) for path in paths.values())
        )
        observations = {name: response.to_dict() for name, response in zip(paths, responses)}
        if full:
            key = quote(f"web:{ids['user_id']}:{ids['chat_id']}", safe="")
            snapshot = await self.api.request("GET", "/v1/sessions/" + key)
            observations["session"] = snapshot.to_dict()
        debug = get(observations, "state", "data", "last_turn", default={})
        return {
            "observations": observations if full else select_observations(observations, last_turn=True),
            "metrics": metrics(debug, observations),
        }

    async def _safe_case(self, scenario: Scenario) -> None:
        expected = scenario.id if self._exclusive(scenario) else "default_arcs"
        if self._staging or self._staged_phase != expected or self._exclusive(scenario) and self._active_cases:
            raise ControlError("Scenario settings do not match the exclusive staged phase")
        self._active_cases.add(scenario.id)
        try:
            await self._run_staged_case(scenario)
        finally:
            self._active_cases.discard(scenario.id)

    async def _run_staged_case(self, scenario: Scenario) -> None:
        try:
            await self._case(scenario)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._block(scenario, exc)
        state = self._state(scenario)
        if "baseline" in state:
            try:
                state["final"] = self.redactor.clean(await self._capture(self._ids(scenario), full=True))
                self._save()
            except Exception as exc:
                state["snapshot_error"] = type(exc).__name__
                self._save()
        self.artifacts.report(self.scenarios, self.manifest)

    async def _case(self, scenario: Scenario) -> None:
        state, ids = self._state(scenario), self._ids(scenario)
        if "baseline" not in state:
            state["baseline"] = self.redactor.clean(await self._capture(ids, include_tools=True, full=True))
            self._save()
        state["status"] = "running"
        existing = self._records(scenario)
        for index in range(len(existing), self._limit(scenario)):
            if self.stop_requested:
                state.update(status="blocked", reason="Interrupted before next chat; use --resume")
                self._save()
                return
            turn = materialize(scenario.turns[index], self.manifest["run_id"], scenario.id)
            if turn.phase == "after_end" and "post_end" not in state["checkpoints"]:
                if not await self._end_phase(state, ids):
                    return
            if "consolidate_before" in turn.tags:
                checkpoint = f"episode_{index}"
                if f"post_{checkpoint}" not in state["checkpoints"]:
                    if not await self._end_phase(state, ids, checkpoint=checkpoint):
                        return
            if scenario.restart_after == index and "post_restart" not in state["checkpoints"]:
                if not await self._restart_phase(state, ids):
                    return
            before = await self._capture(ids)
            history = await self.api.request("GET", "/session/history", params=ids)
            if self.stop_requested:
                state.update(status="blocked", reason="Interrupted before next chat; use --resume")
                self._save()
                return
            history_items = get(history.data, "history", default=None) if history.ok else None
            pending = {
                "turn_index": index + 1,
                "before": {"metrics": before["metrics"]},
                "workspace_before": workspace_snapshot(self.options.tools_workspace, turn.paths),
                "history_count": len(history_items) if isinstance(history_items, list) else None,
                "history_digest": self._history_digest(history_items)
                if isinstance(history_items, list)
                else None,
                "operator_settings": self.operator_settings,
            }
            state["pending"] = self.redactor.clean(pending)
            self._save()
            response = await self.api.request("POST", "/chat", json={**ids, "message": turn.text})
            if response.status is None:
                # Do not retry a possibly executed write or duplicate a relationship turn.
                state.update(
                    status="blocked",
                    reason="Chat delivery is ambiguous; resume reconciles public session history without resending",
                )
                state["pending"]["error"] = response.error
                state["pending"]["failure"] = chat_failure(response.data, response.status, response.error)
                self._save()
                return
            received = response.to_dict()
            received["data"] = select_chat_payload(response.data)
            state["pending"]["received_response"] = self.redactor.clean(received)
            failure = chat_failure(response.data, response.status, response.error)
            if failure:
                state["pending"]["failure"] = failure
            self._save()
            after = await self._capture(ids)
            await self._finish_turn(
                scenario,
                turn,
                pending,
                response.data,
                response.status,
                response.latency_ms,
                response.error,
                after,
            )
        state.update(
            status="complete" if self._limit(scenario) == len(scenario.turns) else "truncated"
        )
        state.pop("reason", None)
        self._save()

    @staticmethod
    def _history_digest(items) -> str:
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()

    async def _finish_turn(
        self, scenario, turn, pending, payload, status, latency, error, after, *, recovered=False
    ):
        payload = payload if isinstance(payload, dict) else {}
        debug = payload.get("debug") or {}
        reply = payload.get("response")
        if not isinstance(reply, str):
            reply = ""
        token = (
            "fixture_"
            + hashlib.sha256(f"{self.manifest['run_id']}/{scenario.id}".encode()).hexdigest()[:12]
        )
        record = {
            "schema_version": 1,
            "scenario_id": scenario.id,
            "category": scenario.category,
            "personality": scenario.personality,
            **self._ids(scenario),
            "turn_index": pending["turn_index"],
            "phase": turn.phase,
            "tags": list(turn.tags),
            "user_text": turn.text,
            "reply": reply,
            "latency_ms": latency,
            "http_status": status,
            "error": error,
            "recovered": recovered,
            "debug": debug,
            "observations": after["observations"],
            "metrics": metrics(debug, after["observations"]),
            "before_metrics": pending["before"]["metrics"],
            "paths": list(turn.paths),
            "fixture_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "workspace_before": pending["workspace_before"],
            "workspace_after": workspace_snapshot(self.options.tools_workspace, turn.paths, expected_content=token.encode()),
            "operator_settings": pending.get("operator_settings"),
        }
        failure = chat_failure(payload, status, error)
        if failure:
            record["failure"] = failure
        previous = self._records(scenario)
        record["checks"] = turn_checks(
            record, previous, reject_roles=scenario.personality != "deferential"
        )
        record = compact_record(record)
        # Persist the chat before judging: a dead/slow judge cannot lose or repeat it.
        record["judge"] = {"status": "off"} if self.judge is None else {"status": "pending"}
        self._state(scenario)["pending"]["received_record"] = self.redactor.clean(record)
        self._save()
        if unavailable_turn(record):
            record["judge"] = {"status": "not_evaluated", "reason": "unavailable_turn"}
        elif self.judge:
            try:
                record["judge"] = await self.judge.evaluate(record, previous)
            except Exception as exc:
                record["judge"] = {"status": "error", "error": type(exc).__name__}
        finalize_deferred_checks(record)
        self.artifacts.append(record)
        state = self._state(scenario)
        state.pop("pending", None)
        self._save()
        self.progress(f"{scenario.id}: turn {record['turn_index']}/{len(scenario.turns)} recorded")

    async def _reconcile_pending(self, scenario: Scenario) -> None:
        state = self._state(scenario)
        pending = state.get("pending")
        if not pending:
            return
        if any(r["turn_index"] == pending["turn_index"] for r in self._records(scenario)):
            state.pop("pending")
            self._save()
            return
        # Reopen the SAME isolated data before querying its persisted history.
        await self._stage(scenario if self._exclusive(scenario) else None)
        if pending.get("received_record"):
            record = compact_record(pending["received_record"])
            if unavailable_turn(record):
                record["judge"] = {"status": "not_evaluated", "reason": "unavailable_turn"}
            elif self.judge:
                try:
                    record["judge"] = await self.judge.evaluate(record, self._records(scenario))
                except Exception as exc:
                    record["judge"] = {"status": "error", "error": type(exc).__name__}
            elif record.get("judge", {}).get("status") == "pending":
                record["judge"] = {"status": "error", "error": "judge_interrupted"}
            finalize_deferred_checks(record)
            self.artifacts.append(record)
            state.pop("pending")
            self._save()
            return
        ids = self._ids(scenario)
        turn = materialize(
            scenario.turns[pending["turn_index"] - 1], self.manifest["run_id"], scenario.id
        )
        if pending.get("received_response"):
            response = pending["received_response"]
            after = await self._capture(ids)
            await self._finish_turn(
                scenario, turn, pending, response["data"], response["status"],
                response["latency_ms"], response.get("error"), after,
            )
            return
        history = await self.api.request("GET", "/session/history", params=ids)
        items = get(history.data, "history", default=[])
        count = pending.get("history_count")
        valid = (
            history.ok and isinstance(items, list) and count is not None
            and len(items) == count + 2
            and self._history_digest(items[:count]) == pending.get("history_digest")
            and items[-2] == {"role": "user", "content": turn.text}
            and items[-1].get("role") == "assistant"
        )
        if valid:
            after = await self._capture(ids)
            debug = get(after, "observations", "state", "data", "last_turn", default={})
            if debug and (
                debug.get("user_message") != turn.text
                or debug.get("response") != items[-1].get("content")
            ):
                valid = False
        if valid:
            await self._finish_turn(
                scenario, turn, pending,
                {"response": items[-1]["content"], "debug": debug},
                None, None, "recovered_from_history", after, recovered=True,
            )
            state["status"] = "running"
        else:
            state.update(
                status="blocked",
                reason="Pending chat is not yet unambiguously present in public history; it has not been resent",
            )
        self._save()

    async def _end_phase(self, state: dict, ids: dict, *, checkpoint="end") -> bool:
        checkpoints = state["checkpoints"]
        if state.get("end_pending"):
            history = await self.api.request("GET", "/session/history", params=ids)
            if not history.ok or get(history.data, "history") != []:
                state.update(
                    status="blocked",
                    reason="Session-end delivery is ambiguous; preserve state and inspect the server",
                )
                return False
        else:
            checkpoints[f"pre_{checkpoint}"] = self.redactor.clean(await self._capture(ids))
            state["end_pending"] = True
            self._save()
            response = await self.api.request("POST", "/session/end", json=ids)
            if not response.ok:
                state.update(
                    status="blocked",
                    reason="Session-end API failed; persistence phase is incomplete",
                )
                self._save()
                return False
            checkpoints[f"{checkpoint}_response"] = self.redactor.clean(response.to_dict())
        checkpoints[f"post_{checkpoint}"] = self.redactor.clean(await self._capture(ids))
        state.pop("end_pending", None)
        self._save()
        return True

    async def _restart_phase(self, state: dict, ids: dict) -> bool:
        checkpoints = state["checkpoints"]
        already_pending = "pre_restart" in checkpoints
        if not already_pending:
            checkpoints["pre_restart"] = self.redactor.clean(await self._capture(ids))
            health = await self.api.request("GET", "/v1/health")
            state["restart_health"] = {
                "uptime": get(health.data, "uptime_seconds"),
                "at": time.time(),
            }
            state["status"] = "awaiting_restart"
            self._save()
        if not already_pending and self.options.restart_hook:
            process = await asyncio.create_subprocess_shell(
                self.options.restart_hook,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), self.options.restart_timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                process.kill()
                await process.wait()
                raise
            state["restart_hook_exit"] = process.returncode
            if process.returncode != 0:
                state.update(
                    status="blocked",
                    reason="Operator restart hook returned a failure; command output is intentionally not logged",
                )
                self._save()
                return False
        elif not already_pending:
            self.paused = True
            state["reason"] = (
                "Restart the server with the staged evaluation settings, then rerun with --resume; --restore-only cancels staging"
            )
            self.progress(state["reason"])
            self._save()
            return False
        deadline = time.monotonic() + (
            self.options.restart_timeout if self.options.restart_hook and not already_pending else 1
        )
        while True:
            health = await self.api.request("GET", "/v1/health")
            before = state.get("restart_health") or {}
            uptime = get(health.data, "uptime_seconds")
            if isinstance(uptime, (int, float)) and isinstance(before.get("uptime"), (int, float)):
                old_boot = before["at"] - before["uptime"]
                new_boot = time.time() - uptime
                if new_boot > old_boot + 0.5:
                    state["restart_verified"] = True
                    break
            if time.monotonic() >= deadline:
                self.paused = True
                state.update(
                    status="awaiting_restart",
                    reason="No server restart observed yet; restart it, then use --resume",
                )
                self._save()
                return False
            await asyncio.sleep(0.5)
        await self._wait_ready()
        checkpoints["post_restart"] = self.redactor.clean(await self._capture(ids))
        self._save()
        return True

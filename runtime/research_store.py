"""Durable research records and read-only observation."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sqlite3
import subprocess
import shutil
from importlib.metadata import PackageNotFoundError, distribution, distributions
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit as ResearchStudyLimit
from core.clock import clock as time
from core.clock import identifiers as uuid
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import requests

from config.loader import get_config, research_configuration
from runtime.config import RuntimeConfig


def encode(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=_serialize, allow_nan=False
    )


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot record {type(value).__name__}")


def restore_configuration(payload: dict[str, Any]) -> Any:
    config = copy.deepcopy(get_config())

    def restore(target: Any, values: dict[str, Any]) -> None:
        for field in fields(target):
            if field.name not in values:
                continue
            current = getattr(target, field.name)
            saved = values[field.name]
            if is_dataclass(current):
                restore(current, saved)
            else:
                setattr(
                    target,
                    field.name,
                    set(saved) if isinstance(current, set) else copy.deepcopy(saved),
                )

    restore(config, payload)
    return config


def _code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    for name in ("openrook", "project-nur"):
        try:
            installed = distribution(name)
        except PackageNotFoundError:
            continue
        files = installed.files or []
        if not any(
            str(path) == "runtime/research_store.py"
            and Path(installed.locate_file(path)).resolve() == Path(__file__).resolve()
            for path in files
        ):
            continue
        hashes = {}
        for relative in sorted(files):
            path = Path(installed.locate_file(relative)).resolve()
            if (
                not path.is_relative_to(root)
                or any(part.endswith((".dist-info", ".egg-info")) for part in relative.parts)
                or "__pycache__" in relative.parts
                or not path.is_file()
            ):
                continue
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes
    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            timeout=10,
            check=True,
        )
        paths = {root / name for name in result.stdout.decode().split("\0") if name}
    except (OSError, subprocess.SubprocessError):
        paths = set()
    if not paths:
        paths = {root / "main.py", root / "pyproject.toml"}
        for package in ("core", "config", "evals", "interface", "pipeline", "runtime", "nur_tools"):
            paths.update((root / package).rglob("*"))
    hashes = {}
    for path in sorted(paths):
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".js", ".html", ".css", ".json"}:
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


class ResearchStore:
    def __init__(self, config: RuntimeConfig, *, initialize: bool = True):
        self.path = Path(config.data_dir) / "shared" / "agent_runtime.db"
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connection() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS records (
                        kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                        updated_at REAL NOT NULL, PRIMARY KEY(kind, id)
                    );
                    CREATE TABLE IF NOT EXISTS observations (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                        kind TEXT NOT NULL, created_at REAL NOT NULL, payload TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS inference_budget (
                        id INTEGER PRIMARY KEY CHECK(id=1), started_at REAL NOT NULL,
                        consumed INTEGER NOT NULL
                    );
                """)

    @contextmanager
    def connection(self, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path.resolve().as_uri() + ("?mode=ro" if read_only else "?mode=rwc"),
            uri=True,
            timeout=5,
        )
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize_run(self, config: RuntimeConfig) -> dict[str, Any]:
        existing = self.get("run", "manifest")
        if existing is not None:
            return existing
        runtime = config.to_yaml_dict()
        credentials = {
            key: runtime.pop(key, "") for key in ("llm_api_key", "api_key", "telegram_token")
        }
        credentials["llm_api_key"] = config.effective_llm_api_key
        birth_configuration = asdict(research_configuration(config.research_birth_policy))
        from runtime.life_history import DEFAULT_DRIVES, experience_digestion_prompt

        manifest = {
            "schema_version": 1,
            "decision_contract_version": 4,
            "action_contract_version": 5,
            "memory_presentation_version": 2,
            "run_id": uuid.uuid4().hex,
            "created_at": time.time(),
            "runtime": runtime,
            "birth_configuration": birth_configuration,
            "birth_drives": DEFAULT_DRIVES,
            "features": getattr(config, "_research_features", {}),
            "code_fingerprints": _code_fingerprint(),
            "prompt_fingerprints": {
                "experience_digestion": hashlib.sha256(
                    experience_digestion_prompt(research_mode=config.research_mode).encode()
                ).hexdigest(),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
                "packages": {
                    package.metadata["Name"]: package.version
                    for package in distributions()
                    if package.metadata["Name"]
                },
            },
            "credential_fingerprints": {
                key: hashlib.sha256(value.encode()).hexdigest() if value else ""
                for key, value in credentials.items()
            },
            "model": {
                "requested": config.llm_model,
                "resolved": None,
                "backend": config.llm_backend,
            },
        }
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO records VALUES ('run', 'manifest', ?, ?)",
                (encode(manifest), time.time()),
            )
        return self.get("run", "manifest")

    def capture_initial_state(self) -> None:
        destination = self.path.parent / "replay_initial"
        if (destination / "complete.json").exists():
            return
        root = self.path.parent.parent
        destination.mkdir(exist_ok=True)
        for source in list(root.rglob("*")):
            if not source.is_file() or source.is_relative_to(destination):
                continue
            relative = source.relative_to(root)
            if relative.parts[0] == "workspace":
                continue
            if source.suffix == ".db":
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as original:
                    with sqlite3.connect(target) as snapshot:
                        original.backup(snapshot)
            elif source.suffix == ".json":
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        (destination / "complete.json").write_text(
            encode({"run_id": self.get("run", "manifest")["run_id"]})
        )

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        with self.connection(read_only=True) as connection:
            row = connection.execute(
                "SELECT payload FROM records WHERE kind=? AND id=?", (kind, record_id)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self, kind: str) -> list[dict[str, Any]]:
        with self.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT payload FROM records WHERE kind=? ORDER BY updated_at, id", (kind,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def put(self, kind: str, record_id: str, payload: dict[str, Any]) -> None:
        if kind == "run":
            raise ValueError("Run manifests are immutable")
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (kind, record_id, encode(payload), time.time()),
            )

    def put_records(
        self,
        records: list[tuple[str, str, dict[str, Any]]],
        *,
        event_kind: str,
        event: dict[str, Any],
    ) -> None:
        if any(kind == "run" for kind, _, _ in records):
            raise ValueError("Run manifests are immutable")
        with self.connection() as connection:
            for kind, record_id, payload in records:
                connection.execute(
                    "INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                    (kind, record_id, encode(payload), time.time()),
                )
            connection.execute(
                "INSERT INTO observations (id,kind,created_at,payload) VALUES (?,?,?,?)",
                (uuid.uuid4().hex, event_kind, time.time(), encode(event)),
            )

    def latest_external_sequence(self) -> int:
        with self.connection(read_only=True) as connection:
            return connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM observations WHERE kind IN ('external_input','environment_observed')"
            ).fetchone()[0]

    def recent_action_receipts(self, *, goal_id: str | None = None) -> list[dict[str, Any]]:
        with self.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT payload FROM records WHERE kind='action'"
                + (" AND json_extract(payload,'$.goal_id')=?" if goal_id else "")
                + " ORDER BY updated_at DESC, id DESC LIMIT 8",
                (goal_id,) if goal_id else (),
            ).fetchall()
        receipts = []
        for row in reversed(rows):
            action = json.loads(row["payload"])
            receipt = {}
            for key in (
                "id",
                "tool",
                "arguments",
                "status",
                "result",
                "origin",
                "goal_id",
                "source_event_id",
            ):
                value = action.get(key)
                if isinstance(value, dict):
                    value = {
                        name: (
                            {"excerpt": encode(item)[:3000], "truncated": True}
                            if len(encode(item)) > 3000
                            else item
                        )
                        for name, item in value.items()
                    }
                if len(encode(value)) > 6000:
                    value = {"excerpt": encode(value)[:6000], "truncated": True}
                receipt[key] = value
            if "goal_revision" in action:
                receipt["goal_revision"] = action["goal_revision"]
            receipts.append(receipt)
        return receipts

    def observe(self, kind: str, payload: Any, *, event_id: str | None = None) -> str:
        event_id = event_id or uuid.uuid4().hex
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO observations (id,kind,created_at,payload) VALUES (?,?,?,?)",
                (event_id, kind, time.time(), encode(payload)),
            )
        return event_id

    def observations(self, *, after: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection(read_only=True) as connection:
            rows = connection.execute(
                "SELECT * FROM observations WHERE sequence>? ORDER BY sequence LIMIT ?",
                (after, max(1, min(limit, 1000))),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def experience_context(
        self, *, after: int = 0, limit: int = 20
    ) -> tuple[list[dict[str, Any]], int]:
        with self.connection(read_only=True) as connection:
            latest = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM observations"
            ).fetchone()[0]
            rows = connection.execute(
                """SELECT sequence,kind,payload FROM observations WHERE sequence>? AND sequence<=?
                AND kind IN ('external_input','selected_expression','tool_completed','learning_completed')
                ORDER BY sequence DESC LIMIT ?""",
                (after, latest, limit),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "kind": row["kind"],
                "payload": json.loads(row["payload"]),
            }
            for row in reversed(rows)
        ], latest

    def reserve_inference(self, maximum: int) -> None:
        now = time.time()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM inference_budget WHERE id=1").fetchone()
            if row is None or now < row["started_at"] or now - row["started_at"] >= 86400:
                connection.execute("INSERT OR REPLACE INTO inference_budget VALUES (1,?,1)", (now,))
            elif row["consumed"] >= maximum:
                raise ResearchBudgetExceeded(row["started_at"] + 86400)
            else:
                connection.execute("UPDATE inference_budget SET consumed=consumed+1 WHERE id=1")

    def checkpoint(
        self,
        state: dict[str, Any],
        event: dict[str, Any],
        records: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO observations (id,kind,created_at,payload) VALUES (?,?,?,?)",
                (uuid.uuid4().hex, "transition", time.time(), encode(event)),
            )
            connection.execute(
                "INSERT OR REPLACE INTO records VALUES ('agent','state',?,?)",
                (encode(state), time.time()),
            )
            for kind, record_id, payload in records or []:
                connection.execute(
                    "INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                    (kind, record_id, encode(payload), time.time()),
                )


class ObservedBackend:
    def __init__(self, backend: Any, store: ResearchStore, maximum: int, *, retries: int = 0):
        self.backend = backend
        self.store = store
        self.maximum = maximum
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError("Inference retries must be an integer from zero to two")
        self.retries = retries
        disable_retries = getattr(backend, "disable_transport_retries", None)
        if retries and callable(disable_retries):
            disable_retries()

    def generate(
        self, system_prompt: str, user_message: str, *, response_schema: dict[str, Any] | None = None
    ) -> str:
        for attempt in range(self.retries + 1):
            try:
                return self._generate_once(system_prompt, user_message, response_schema=response_schema)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
                transient = (
                    isinstance(error, (requests.ConnectionError, requests.Timeout))
                    and not isinstance(error, requests.exceptions.SSLError)
                ) or (
                    isinstance(error, requests.HTTPError)
                    and error.response is not None
                    and error.response.status_code in {408, 500, 502, 503, 504}
                )
                if not transient or attempt == self.retries:
                    raise
                self.store.observe(
                    "model_transport_retry",
                    {"retry": attempt + 1, "maximum_retries": self.retries, "error_type": type(error).__name__},
                )
                time.sleep(attempt + 1)
        raise RuntimeError("Inference retry loop exhausted")

    def _generate_once(
        self, system_prompt: str, user_message: str, *, response_schema: dict[str, Any] | None = None
    ) -> str:
        self.store.reserve_inference(self.maximum)
        call_id = self.store.observe(
            "model_request", {
                "system": system_prompt, "input": user_message,
                **({"response_schema": response_schema} if response_schema is not None else {}),
            }
        )
        try:
            result = self.backend.generate(
                system_prompt, user_message,
                **({"response_schema": response_schema} if response_schema is not None else {}),
            )
        except Exception as error:
            metadata = getattr(self.backend, "last_response_metadata", None)
            self.store.observe(
                "model_failure",
                {
                    "call_id": call_id,
                    "type": type(error).__name__,
                    "error": str(error),
                    "metadata": metadata if isinstance(metadata, dict) else None,
                },
            )
            raise
        metadata = getattr(self.backend, "last_response_metadata", None)
        self.store.observe(
            "model_response",
            {
                "call_id": call_id,
                "output": result,
                "metadata": metadata if isinstance(metadata, dict) else None,
            },
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

"""Durable study-wide reservations, including failed requests and tool attempts."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from runtime.research_store import ResearchStudyLimit, encode


INTEGER_LIMITS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "tool_actions",
    "input_bytes_per_call",
    "output_tokens_per_call",
)
PRICE_LIMITS = ("max_cost_usd", "input_usd_per_million", "output_usd_per_million")


def validate_limits(limits: dict[str, Any]) -> None:
    required = {
        *INTEGER_LIMITS,
        "runtime_seconds",
        "max_cost_usd",
        "input_usd_per_million",
        "output_usd_per_million",
    }
    if not isinstance(limits, dict) or set(limits) - {"billing_mode"} != required:
        raise ValueError(
            "Study limits require explicit call, token, tool, runtime and declared-price bounds"
        )
    mode = limits.get("billing_mode", "declared_rates")
    if not isinstance(mode, str) or mode not in {"declared_rates", "unmetered"}:
        raise ValueError("Study billing_mode must be declared_rates or unmetered")
    for key in INTEGER_LIMITS:
        if type(limits[key]) is not int or limits[key] < 1:
            raise ValueError(f"Study {key} must be a positive integer")
    for key in required - set(INTEGER_LIMITS):
        value = limits[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Study {key} must be finite and nonnegative")
    if not 0 < limits["runtime_seconds"] <= 604800 or limits["output_tokens_per_call"] > 8192:
        raise ValueError("Study runtime or per-call output bound is too large")
    if mode == "unmetered" and any(limits[key] != 0 for key in PRICE_LIMITS):
        raise ValueError("Unmetered studies require zero declared prices and monetary budget")


class StudyBudget:
    def __init__(self, path: Path, limits: dict[str, Any]):
        validate_limits(limits)
        self.path = path
        self.limits = dict(limits)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO budget VALUES (1,?)",
                (
                    encode(
                        {
                            "limits": limits,
                            "started_at": time.time(),
                            "model_calls": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "tool_actions": 0,
                            "reserved_cost_usd": 0.0,
                            "reported_input_tokens": 0,
                            "reported_output_tokens": 0,
                            "responses_with_usage": 0,
                            "transport_attempts": 0,
                            "stop_reason": None,
                            "token_accounting": "input UTF-8 bytes + 1024 framing reserve; output requested maximum; not a certified tokenizer or provider billing cap",
                            "price_accounting": (
                                "operator-declared unmetered inference; call, token, tool and runtime limits still apply"
                                if limits.get("billing_mode") == "unmetered"
                                else "conservative reservations at declared rates; actual billing requires provider validation"
                            ),
                        }
                    ),
                ),
            )
            saved = json.loads(
                connection.execute("SELECT payload FROM budget WHERE id=1").fetchone()[0]
            )
            if saved["limits"] != limits:
                raise ValueError("Study budget limits are immutable")

    def snapshot(self) -> dict:
        with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            return json.loads(
                connection.execute("SELECT payload FROM budget WHERE id=1").fetchone()[0]
            )

    def _reserve(self, increments: dict[str, float], *, input_bytes: int = 0) -> None:
        reason = None
        with sqlite3.connect(self.path, timeout=5) as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = json.loads(
                connection.execute("SELECT payload FROM budget WHERE id=1").fetchone()[0]
            )
            reason = state["stop_reason"]
            if not reason and time.time() - state["started_at"] >= self.limits["runtime_seconds"]:
                reason = "runtime_seconds"
            if not reason and input_bytes > self.limits["input_bytes_per_call"]:
                reason = "input_bytes_per_call"
            for key, increment in increments.items():
                limit_key = "max_cost_usd" if key == "reserved_cost_usd" else key
                if not reason and state[key] + increment > self.limits[limit_key] + 1e-12:
                    reason = limit_key
            if reason:
                state["stop_reason"] = reason
            else:
                for key, increment in increments.items():
                    state[key] += increment
            connection.execute("UPDATE budget SET payload=? WHERE id=1", (encode(state),))
        if reason:
            raise ResearchStudyLimit("Study resource limit: " + reason)

    def check(self) -> None:
        self._reserve({})

    def stop(self, reason: str) -> None:
        with sqlite3.connect(self.path, timeout=5) as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = json.loads(
                connection.execute("SELECT payload FROM budget WHERE id=1").fetchone()[0]
            )
            state["stop_reason"] = state["stop_reason"] or reason
            connection.execute("UPDATE budget SET payload=? WHERE id=1", (encode(state),))

    def reserve_model(self, system_prompt: str, user_message: str) -> None:
        input_bytes = len(system_prompt.encode("utf-8")) + len(user_message.encode("utf-8"))
        input_tokens = input_bytes + 1024
        output_tokens = self.limits["output_tokens_per_call"]
        cost = (
            input_tokens * self.limits["input_usd_per_million"]
            + output_tokens * self.limits["output_usd_per_million"]
        ) / 1_000_000
        self._reserve(
            {
                "model_calls": 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reserved_cost_usd": cost,
            },
            input_bytes=input_bytes,
        )

    def tool_guard(self, tool: str, arguments: dict) -> str | None:
        try:
            self._reserve({"tool_actions": 1})
        except ResearchStudyLimit as error:
            return str(error)
        return None

    def record_response(self, metadata: dict, *, input_reservation: int) -> None:
        with sqlite3.connect(self.path, timeout=5) as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = json.loads(
                connection.execute("SELECT payload FROM budget WHERE id=1").fetchone()[0]
            )
            attempts = metadata.get("transport_attempts", [])
            if isinstance(attempts, list):
                state["transport_attempts"] += len(attempts)
                if len(attempts) > 1:
                    state["stop_reason"] = "unexpected_transport_retry"
            usage = metadata.get("usage")
            if isinstance(usage, dict) and all(
                type(usage.get(key)) is int and usage[key] >= 0
                for key in ("prompt_tokens", "completion_tokens")
            ):
                state["responses_with_usage"] += 1
                state["reported_input_tokens"] += usage["prompt_tokens"]
                state["reported_output_tokens"] += usage["completion_tokens"]
                if (
                    usage["prompt_tokens"] > input_reservation
                    or usage["completion_tokens"] > self.limits["output_tokens_per_call"]
                ):
                    state["stop_reason"] = "provider_usage_exceeded_reservation"
            connection.execute("UPDATE budget SET payload=? WHERE id=1", (encode(state),))


class BudgetedBackend:
    def __init__(self, backend: Any, budget: StudyBudget, *, injected: bool):
        self.backend = backend
        self.budget = budget
        self.last_response_metadata: dict = {}
        self.injected = injected
        if not injected:
            from runtime.llm.backend import OpenAICompatibleLLMBackend

            if not isinstance(backend, OpenAICompatibleLLMBackend):
                backend.close()
                raise ValueError(
                    "Bounded live studies currently require the OpenAI-compatible backend"
                )
            if (
                budget.limits.get("billing_mode", "declared_rates") != "unmetered"
                and urlsplit(backend._base_url).hostname not in {"localhost", "127.0.0.1", "::1"}
                and any(budget.limits[key] <= 0 for key in PRICE_LIMITS)
            ):
                backend.close()
                raise ValueError(
                    "Hosted studies require explicit positive declared price bounds or operator-declared unmetered billing; zero-price fixture protocols cannot launch paid inference"
                )
            backend.set_research_limits(
                output_tokens=budget.limits["output_tokens_per_call"],
                timeout=min(120, budget.limits["runtime_seconds"]),
            )

    def generate(
        self, system_prompt: str, user_message: str, *, response_schema: dict[str, Any] | None = None
    ) -> str:
        self.last_response_metadata = {"budget_reserved": False}
        schema_text = encode(response_schema) if response_schema is not None else ""
        self.budget.reserve_model(system_prompt, user_message + schema_text)
        if not self.injected:
            deadline = self.budget.snapshot()["started_at"] + self.budget.limits["runtime_seconds"]
            self.backend.set_research_limits(
                output_tokens=self.budget.limits["output_tokens_per_call"],
                timeout=max(0.001, min(120, deadline - time.time())),
            )
        self.last_response_metadata = {"budget_reserved": True}
        try:
            return self.backend.generate(
                system_prompt, user_message,
                **({"response_schema": response_schema} if response_schema is not None else {}),
            )
        finally:
            metadata = getattr(self.backend, "last_response_metadata", {})
            if isinstance(metadata, dict):
                self.last_response_metadata.update(metadata)
            self.budget.record_response(
                self.last_response_metadata,
                input_reservation=len(system_prompt.encode("utf-8"))
                + len(user_message.encode("utf-8"))
                + len(schema_text.encode("utf-8"))
                + 1024,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

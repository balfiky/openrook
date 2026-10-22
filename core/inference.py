"""Per-turn inference accounting and bounded optional calls, without unwrapping backends."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import time
from weakref import WeakKeyDictionary

from core.llm_content import strip_reasoning

log = logging.getLogger(__name__)
_calls = ContextVar("inference_calls", default=None)
_research = ContextVar("research_inference", default=False)
_phase = ContextVar("inference_phase", default=None)
_active_phase = ContextVar("active_inference_phase", default=None)
_timeout = ContextVar("optional_inference_timeout", default=None)
_limit = ContextVar("turn_inference_limit", default=None)
_reserved_calls = ContextVar("reserved_expression_calls", default=0)
_primary_tool_calls = ContextVar("primary_tool_calls", default=0)
_expression_sampling = ContextVar("expression_sampling", default=None)
_required_expression = ContextVar("required_expression", default=False)
SLOW_CALL_SECONDS = 5.0
OPTIONAL_TIMEOUT_SECONDS = 8.0
AUXILIARY_TIMEOUT_THRESHOLD = 2
AUXILIARY_COOLDOWN_SECONDS = 30.0


@dataclass(frozen=True)
class _AuxiliaryPolicy:
    request_seconds: float | None = None
    timeout_seconds: float | None = None
    cooldown_seconds: float = AUXILIARY_COOLDOWN_SECONDS


_auxiliary_policy = ContextVar("auxiliary_inference_policy", default=_AuxiliaryPolicy())


class OptionalInferenceCooldown(TimeoutError):
    """An unhealthy optional phase uses its existing deterministic fallback."""


@dataclass
class _AuxiliaryHealth:
    timeouts: int = 0
    retry_after: float = 0.0
    latency_seconds: float = 0.0


_auxiliary_health: WeakKeyDictionary = WeakKeyDictionary()


def _health_for(backend, phase: str) -> _AuxiliaryHealth | None:
    if backend is None:
        return None
    try:
        phases = _auxiliary_health.setdefault(backend, {})
    except TypeError:
        # Unhashable backend adapters can still own their health without a
        # global reference extending their lifetime.
        phases = getattr(backend, "_optional_inference_health", None)
        if phases is None:
            phases = {}
            try:
                backend._optional_inference_health = phases
            except (AttributeError, TypeError):
                return None
    return phases.setdefault(phase, _AuxiliaryHealth())


@contextmanager
def inference_turn(calls, *, research=False, max_calls=None, request_timeout_seconds=None,
                   auxiliary_timeout_seconds=None, auxiliary_cooldown_seconds=AUXILIARY_COOLDOWN_SECONDS):
    call_token = _calls.set(calls)
    research_token = _research.set(research)
    limit_token = _limit.set(max_calls)
    policy_token = _auxiliary_policy.set(_AuxiliaryPolicy(
        request_timeout_seconds, auxiliary_timeout_seconds, auxiliary_cooldown_seconds))
    try:
        yield
    finally:
        _auxiliary_policy.reset(policy_token)
        _limit.reset(limit_token)
        _research.reset(research_token)
        _calls.reset(call_token)


def slow_inference() -> bool:
    return any(item["duration_ms"] >= SLOW_CALL_SECONDS * 1000 or item["status"] != "ok"
               for item in _calls.get() or [])


def remaining_calls() -> int:
    limit = _limit.get()
    spent = sum(not item.get("primary_tool", False) for item in _calls.get() or [])
    return max(0, limit - spent - _reserved_calls.get()) if limit is not None else 1000000


def call_telemetry() -> list[dict]:
    """A snapshot for durable turn evidence before the outer call returns."""
    return [dict(item) for item in _calls.get() or []]


@contextmanager
def primary_tool_inference():
    """One protected selection for this granted task, independent of optional work.

    At most eight atomic grants exist. Additional selections still share the
    ordinary turn budget; every actual call remains in the same telemetry.
    """
    token = _primary_tool_calls.set(1)
    try:
        yield
    finally:
        _primary_tool_calls.reset(token)


def tool_selection_available() -> bool:
    return bool(_primary_tool_calls.get() or remaining_calls())


@contextmanager
def reserve_expression_calls(count: int = 3):
    """Protect expression, review and one fresh attempt if review rules it out."""
    token = _reserved_calls.set(count)
    try:
        yield
    finally:
        _reserved_calls.reset(token)


def request_timeout(default: float) -> float:
    limit = _timeout.get()
    return min(default, limit) if limit is not None else default


def reviewing_expression() -> bool:
    """Request-scoped transport controls also work through lazy backend wrappers."""
    return _active_phase.get() == "expression_review"


def output_token_limit(default: int) -> int:
    return min(default, 512) if reviewing_expression() else default


def _optional_timeout(backend, latency: _AuxiliaryHealth | None) -> float:
    policy = _auxiliary_policy.get()
    primary = policy.request_seconds or getattr(backend, "_timeout", OPTIONAL_TIMEOUT_SECONDS)
    if policy.timeout_seconds is not None:
        return min(primary, policy.timeout_seconds)
    # A configured slow backend gets a useful first review deadline. Successful
    # generation then supplies measured latency, including across user turns.
    observed = latency.latency_seconds if latency else 0.0
    return min(primary, max(OPTIONAL_TIMEOUT_SECONDS, primary / 4, 2 * observed))


@contextmanager
def inference_phase(phase: str):
    token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(token)


@contextmanager
def expression_retry(seed: int, attempt: int, *, required: bool):
    """Vary fresh expression through backend wrappers, without changing choice."""
    sampling = _expression_sampling.set({
        "seed": (seed + attempt * 104729) % (2**31),
        "temperature": min(1.15, 0.65 + 0.15 * attempt),
    })
    mandatory = _required_expression.set(required)
    try:
        yield
    finally:
        _required_expression.reset(mandatory)
        _expression_sampling.reset(sampling)


def sampling_parameters() -> dict:
    """Transport controls apply only to the fresh expression call, never review."""
    return dict(_expression_sampling.get() or {}) if _active_phase.get() == "regeneration" else {}


@contextmanager
def model_call(phase: str, *, optional=False, backend=None, primary_tool=False):
    phase = _phase.get() or phase
    primary_tool = primary_tool and phase == "tool_selection" and _primary_tool_calls.get() > 0
    if primary_tool:
        _primary_tool_calls.set(_primary_tool_calls.get() - 1)
        optional = False
    optional = optional or phase == "regeneration"
    if phase == "regeneration" and _required_expression.get():
        optional = False
    if not primary_tool and remaining_calls() == 0:
        raise OptionalInferenceCooldown("Per-turn inference budget exhausted")
    health = _health_for(backend, phase) if optional and not _research.get() else None
    latency = _health_for(backend, "generation_latency") if not _research.get() else None
    if health is not None and health.retry_after > time.monotonic():
        # No request was made: don't count or log each skipped attempt.
        raise OptionalInferenceCooldown("Optional inference is cooling down")
    deadline = _optional_timeout(backend, latency) if optional and not _research.get() else None
    token = _timeout.set(deadline)
    phase_token = _active_phase.set(phase)
    started = time.monotonic()
    status = "ok"
    timed_out = False
    try:
        yield
    except Exception as error:
        status = type(error).__name__
        timed_out = any("timeout" in type(cause).__name__.lower()
                        for cause in (error, error.__cause__) if cause is not None)
        raise
    finally:
        elapsed = (time.monotonic() - started) * 1000
        if latency is not None and status == "ok" and not optional:
            latency.latency_seconds = elapsed / 1000
        if health is not None:
            if timed_out:
                health.timeouts += 1
                if health.timeouts >= AUXILIARY_TIMEOUT_THRESHOLD:
                    cooldown = _auxiliary_policy.get().cooldown_seconds
                    health.retry_after = time.monotonic() + cooldown
                    log.warning("LLM auxiliary cooldown phase=%s cooldown_seconds=%.0f; retaining explicit review uncertainty",
                                phase, cooldown)
            elif status == "ok":
                health.timeouts = 0
                health.retry_after = 0.0
        item = {"phase": phase, "duration_ms": round(elapsed, 2), "status": status,
                "optional": optional}
        if phase == "regeneration" and _expression_sampling.get():
            item["sampling"] = dict(_expression_sampling.get())
        if primary_tool:
            item["primary_tool"] = True
        if deadline is not None:
            item["timeout_seconds"] = deadline
        calls = _calls.get()
        if calls is not None:
            calls.append(item)
        # Exception messages can contain URLs, headers, prompts, or credentials.
        # Log only the purpose, elapsed time, and exception class.
        purpose = "tool_loop" if phase == "tool_selection" else phase
        logger = log.info if status == "ok" else log.warning
        logger("LLM call purpose=%s phase=%s duration_ms=%.1f status=%s timeout=%s",
               purpose, phase, elapsed, status, timed_out)
        _timeout.reset(token)
        _active_phase.reset(phase_token)


def generate(backend, phase, system_prompt, user_message, *, optional=False, **kwargs):
    with model_call(phase, optional=optional, backend=backend):
        result = backend.generate(system_prompt, user_message, **kwargs)
        return strip_reasoning(result) if isinstance(result, str) else result

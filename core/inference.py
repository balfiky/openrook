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
_timeout = ContextVar("optional_inference_timeout", default=None)
SLOW_CALL_SECONDS = 5.0
OPTIONAL_TIMEOUT_SECONDS = 8.0
AUXILIARY_TIMEOUT_THRESHOLD = 2
AUXILIARY_COOLDOWN_SECONDS = 900.0


class OptionalInferenceCooldown(TimeoutError):
    """An unhealthy optional phase uses its existing deterministic fallback."""


@dataclass
class _AuxiliaryHealth:
    timeouts: int = 0
    retry_after: float = 0.0


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
def inference_turn(calls, *, research=False):
    call_token = _calls.set(calls)
    research_token = _research.set(research)
    try:
        yield
    finally:
        _research.reset(research_token)
        _calls.reset(call_token)


def slow_inference() -> bool:
    return any(item["duration_ms"] >= SLOW_CALL_SECONDS * 1000 or item["status"] != "ok"
               for item in _calls.get() or [])


def request_timeout(default: float) -> float:
    limit = _timeout.get()
    return min(default, limit) if limit is not None else default


@contextmanager
def inference_phase(phase: str):
    token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(token)


@contextmanager
def model_call(phase: str, *, optional=False, backend=None):
    phase = _phase.get() or phase
    optional = optional or phase == "regeneration"
    health = _health_for(backend, phase) if optional and not _research.get() else None
    if health is not None and health.retry_after > time.monotonic():
        # No request was made: don't count or log each skipped attempt.
        raise OptionalInferenceCooldown("Optional inference is cooling down")
    token = _timeout.set(OPTIONAL_TIMEOUT_SECONDS if optional and not _research.get() else None)
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
        if health is not None:
            if timed_out:
                health.timeouts += 1
                if health.timeouts >= AUXILIARY_TIMEOUT_THRESHOLD:
                    health.retry_after = time.monotonic() + AUXILIARY_COOLDOWN_SECONDS
                    log.warning("LLM auxiliary cooldown phase=%s cooldown_seconds=%.0f; using deterministic fallback",
                                phase, AUXILIARY_COOLDOWN_SECONDS)
            elif status == "ok":
                health.timeouts = 0
                health.retry_after = 0.0
        item = {"phase": phase, "duration_ms": round(elapsed, 2), "status": status,
                "optional": optional}
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


def generate(backend, phase, system_prompt, user_message, *, optional=False, **kwargs):
    with model_call(phase, optional=optional, backend=backend):
        result = backend.generate(system_prompt, user_message, **kwargs)
        return strip_reasoning(result) if isinstance(result, str) else result

"""Per-turn inference accounting and bounded optional calls, without unwrapping backends."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import inspect
import logging
import threading
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
_delivery_recovery = ContextVar("delivery_recovery", default=False)
_delivery_recovery_used = ContextVar("delivery_recovery_used", default=False)
_output_tokens = ContextVar("call_output_tokens", default=None)
OPTIONAL_TIMEOUT_SECONDS = 8.0
AUXILIARY_TIMEOUT_THRESHOLD = 2
AUXILIARY_COOLDOWN_SECONDS = 30.0
# Reading the message is required work with its own allowance: one call, and
# one retry when the answer is unusable. It is accounted in the same telemetry
# and is not charged against the calls that fund expression, rewrites and tools.
READING_CALLS = 2
# The reading has its own output limit, apart from expression's. A model that
# reasons before it answers spends that reasoning out of the same limit, so the
# default leaves room for both; a cut-off first answer is asked again with more.
READING_OUTPUT_TOKENS = 4096
READING_OUTPUT_TOKENS_RANGE = (256, 32768)
# Reviewing a draft against the frozen decision is required work too, with its
# own allowance for the whole turn. It is sized by what a turn can produce: six
# generations at most, each of which can leave one excision remainder, so twelve
# candidates each get their one review. One more review is kept back for the
# single confirmation a turn may ask for before it ends without a reply
# (``pipeline/delivery.py::DeliveryGate.confirm``); no candidate's review can use
# it up. A second call for the same draft, after an unusable answer, comes out
# of a separate, smaller allowance, so retries can never use up a later draft's
# review. Like the reading, reviews are accounted in the same telemetry and are
# not charged against the calls that fund expression, rewrites and tools.
#
# All review calls of a turn share one wall-clock budget, the request timeout
# in force, and stop for the turn after one of them timed out or could not
# reach the model: a slow or absent reviewer costs a turn one timeout, not one
# per candidate.
REVIEW_CONFIRMATIONS = 1
REVIEW_CALLS = 12 + REVIEW_CONFIRMATIONS
REVIEW_RETRIES = 4
REVIEW_ATTEMPTS = 2
# A review is a short object. A cut-off answer is asked again with twice the room.
REVIEW_OUTPUT_TOKENS = 2048
REVIEW_OUTPUT_TOKENS_MAX = 8192
# Finish reasons with which a server says it stopped at the output limit.
_STOPPED_AT_LIMIT = frozenset({"length", "max_tokens"})


@dataclass(frozen=True)
class _AuxiliaryPolicy:
    request_seconds: float | None = None
    timeout_seconds: float | None = None
    cooldown_seconds: float = AUXILIARY_COOLDOWN_SECONDS
    reading_output_tokens: int | None = None


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
                   auxiliary_timeout_seconds=None, auxiliary_cooldown_seconds=AUXILIARY_COOLDOWN_SECONDS,
                   reading_output_tokens=None):
    call_token = _calls.set(calls)
    research_token = _research.set(research)
    limit_token = _limit.set(max_calls)
    recovery_token = _delivery_recovery_used.set(False)
    policy_token = _auxiliary_policy.set(_AuxiliaryPolicy(
        request_timeout_seconds, auxiliary_timeout_seconds, auxiliary_cooldown_seconds,
        reading_output_tokens if valid_reading_output_tokens(reading_output_tokens) else None))
    try:
        yield
    finally:
        _auxiliary_policy.reset(policy_token)
        _limit.reset(limit_token)
        _delivery_recovery_used.reset(recovery_token)
        _research.reset(research_token)
        _calls.reset(call_token)


def remaining_calls() -> int:
    limit = _limit.get()
    spent = sum(not (item.get("primary_tool", False) or item.get("reading", False) or item.get("review", False))
                for item in _calls.get() or [])
    return max(0, limit - spent - _reserved_calls.get()) if limit is not None else 1000000


def monotonic() -> float:
    """The clock every inference deadline in a turn is measured on."""
    return time.monotonic()


def primary_deadline(backend) -> float | None:
    """The request timeout in force for required calls, when one is known."""
    seconds = _auxiliary_policy.get().request_seconds
    if seconds is None:
        inner = backend
        initialize = getattr(backend, "initialize", None)
        if callable(initialize):
            try:
                inner = initialize()
            except Exception:
                return None
        seconds = getattr(inner, "_timeout", None)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
        return None
    return float(seconds)


def valid_reading_output_tokens(value) -> bool:
    low, high = READING_OUTPUT_TOKENS_RANGE
    return type(value) is int and low <= value <= high


def reading_output_tokens(*, after_cut_off: bool = False) -> int:
    """The output limit of a reading call: the configured one, else the default.

    After an answer that was cut off the next attempt gets twice as much, up
    to the ceiling, so asking again can succeed where the first could not.
    """
    limit = _auxiliary_policy.get().reading_output_tokens or READING_OUTPUT_TOKENS
    return min(READING_OUTPUT_TOKENS_RANGE[1], 2 * limit) if after_cut_off else limit


def response_facts(backend) -> dict:
    """How the backend's last answer ended, when the backend says.

    ``finish_reason`` and ``completion_tokens`` are reported as the server gave
    them; ``truncated`` is true when the server stopped at the output limit. A
    backend that reports nothing yields an empty mapping.
    """
    try:
        metadata = getattr(backend, "last_response_metadata", None)
    except Exception:
        return {}
    if not isinstance(metadata, dict):
        return {}
    facts = {}
    finish = metadata.get("finish_reason")
    if isinstance(finish, str) and finish:
        facts["finish_reason"] = finish
        facts["truncated"] = finish.strip().lower() in _STOPPED_AT_LIMIT
    usage = metadata.get("usage")
    tokens = metadata.get("completion_tokens", usage.get("completion_tokens") if isinstance(usage, dict) else None)
    if type(tokens) is int and tokens >= 0:
        facts["completion_tokens"] = tokens
    return facts


def remaining_reading_calls() -> int:
    if _limit.get() is None:
        return READING_CALLS
    return max(0, READING_CALLS - sum(bool(item.get("reading")) for item in _calls.get() or []))


def remaining_review_calls(*, retry: bool = False, confirmation: bool = False) -> int:
    """How many reviews this turn may still start.

    With ``retry``: how many second calls for a draft already asked about.
    With ``confirmation``: whether the turn's one confirmation is still to be
    had. Otherwise: how many candidates can still get their review, which
    leaves the confirmation's call alone.
    """
    if retry:
        allowance = REVIEW_RETRIES
    else:
        allowance = REVIEW_CONFIRMATIONS if confirmation else REVIEW_CALLS - REVIEW_CONFIRMATIONS
    if _limit.get() is None:
        return allowance
    spent = sum(bool(item.get("review")) and bool(item.get("review_retry")) is retry
                and (retry or bool(item.get("review_confirmation")) is confirmation)
                for item in _calls.get() or [])
    return max(0, allowance - spent)


def review_time_left(backend) -> float | None:
    """What is left of the turn's review time, in seconds; None when no request timeout is known.

    Every review call of a turn draws on one budget: the request timeout in
    force. What earlier review calls of the turn took is taken off it, whether
    they answered or failed.
    """
    budget = primary_deadline(backend)
    if budget is None:
        return None
    spent = sum(item.get("duration_ms", 0.0) for item in _calls.get() or [] if item.get("review"))
    return budget - spent / 1000


def review_closed() -> bool:
    """Whether a review call of this turn timed out or could not reach the model.

    After that no further review call is made in the turn: the next would wait
    on the same endpoint for as long again.
    """
    return any(item.get("review") and item.get("transport_failure") for item in _calls.get() or [])


def review_output_tokens(*, after_cut_off: bool = False) -> int:
    """The output limit of a review call; twice as much after a cut-off answer."""
    return min(REVIEW_OUTPUT_TOKENS_MAX, 2 * REVIEW_OUTPUT_TOKENS) if after_cut_off else REVIEW_OUTPUT_TOKENS


def accepts_response_schema(backend) -> bool:
    """Whether a backend declares structured output. A wrapper answers for what it wraps."""
    declared = getattr(backend, "accepts_response_schema", None)
    if callable(declared):
        try:
            return bool(declared())
        except Exception:
            return False
    if isinstance(declared, bool):
        return declared
    try:
        return "response_schema" in inspect.signature(backend.generate).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def call_telemetry() -> list[dict]:
    """A snapshot for durable turn evidence before the outer call returns."""
    return [dict(item) for item in _calls.get() or []]


# The calls that produce what a granted task needs before it can run: the
# selection of its tool call, or the text of a file it asked to have written.
GRANTED_TASK_PHASES = ("tool_selection", "artifact_draft")


@contextmanager
def primary_tool_inference():
    """One protected call for this granted task, independent of optional work.

    The call is the task's tool selection or, for a task that asked for text
    to be written to a file, the draft of that text; a task makes one or the
    other. At most eight atomic grants exist. Additional selections still
    share the ordinary turn budget; every actual call remains in the same
    telemetry.
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
    """Protect expression, one rewrite and one fresh attempt if review rules a draft out."""
    token = _reserved_calls.set(count)
    try:
        yield
    finally:
        _reserved_calls.reset(token)


def request_timeout(default: float) -> float:
    limit = _timeout.get()
    return min(default, limit) if limit is not None else default


def call_with_deadline(call, seconds: float, *, on_expire=None, timeout_error=TimeoutError):
    """Run a blocking transport call with a hard wall-clock limit.

    Library read timeouts reset on every received byte, so a peer that trickles
    data (or a proxy that keeps the socket alive) can hold a request open
    indefinitely. The call runs on a daemon worker; if it has not finished when
    the monotonic deadline passes, ``on_expire`` closes the transport and
    ``timeout_error`` is raised. The worker's late result is discarded.
    """
    outcome: dict = {}
    done = threading.Event()

    def run() -> None:
        try:
            outcome["value"] = call()
        except BaseException as error:  # re-raised on the caller's thread
            outcome["error"] = error
        finally:
            done.set()

    threading.Thread(target=run, name="llm-call", daemon=True).start()
    if not done.wait(seconds):
        if on_expire is not None:
            try:
                on_expire()
            except Exception:
                log.debug("transport cleanup after deadline failed", exc_info=True)
        raise timeout_error(f"LLM request exceeded its {seconds:g}s wall-clock deadline")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def is_timeout(error: BaseException) -> bool:
    """Whether a failed model call ran out of time, by exception class alone."""
    return any("timeout" in type(cause).__name__.lower()
               for cause in (error, error.__cause__) if cause is not None)


def is_transport_failure(error: BaseException) -> bool:
    """Whether a failed model call never got an answer: it timed out or the connection failed.

    By exception class alone, as with :func:`is_timeout`. A server that
    answered with an error did answer; asking it again in another form can work.
    """
    return is_timeout(error) or any(
        isinstance(cause, ConnectionError) or "connect" in type(cause).__name__.lower()
        for cause in (error, error.__cause__) if cause is not None)


def reviewing_expression() -> bool:
    """Request-scoped transport controls also work through lazy backend wrappers."""
    return _active_phase.get() == "expression_review"


def requested_output_tokens() -> int | None:
    """The output limit this call asked for itself, if it asked for one."""
    return _output_tokens.get()


def output_token_limit(default: int) -> int:
    requested = _output_tokens.get()
    if requested is not None:
        return requested  # a call that asked for its own limit, such as the reading or the review
    if reviewing_expression():
        return min(default, 512)
    return default


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
def final_expression_attempt():
    """One protected delivery attempt after bounded drafts excise to empty.

    It cannot fund the reading or tools; its review comes out of the review's
    own allowance. Research limits still apply at the backend, and telemetry
    records this extra call separately.
    """
    token = _delivery_recovery.set(True)
    try:
        yield
    finally:
        _delivery_recovery.reset(token)


@contextmanager
def model_call(phase: str, *, optional=False, backend=None, primary_tool=False, reading=False,
               review=False, review_retry=False, review_confirmation=False, limit_seconds=None,
               output_tokens=None):
    phase = _phase.get() or phase
    primary_tool = primary_tool and phase in GRANTED_TASK_PHASES and _primary_tool_calls.get() > 0
    if primary_tool:
        _primary_tool_calls.set(_primary_tool_calls.get() - 1)
        optional = False
    if reading:
        # Required: the primary deadline, no auxiliary cooldown, its own allowance.
        optional = False
        if not remaining_reading_calls():
            raise OptionalInferenceCooldown("Per-turn reading allowance exhausted")
    if review:
        # Required as well: the primary deadline, no auxiliary cooldown, its own allowance.
        optional = False
        if not remaining_review_calls(retry=review_retry, confirmation=review_confirmation):
            raise OptionalInferenceCooldown("Per-turn review allowance exhausted")
    optional = optional or phase == "regeneration"
    if phase == "regeneration" and _required_expression.get():
        optional = False
    recovery = phase == "regeneration" and _delivery_recovery.get() and not _delivery_recovery_used.get()
    if recovery:
        _delivery_recovery_used.set(True)
        optional = False
    if not primary_tool and not recovery and not reading and not review and remaining_calls() == 0:
        raise OptionalInferenceCooldown("Per-turn inference budget exhausted")
    health = _health_for(backend, phase) if optional and not _research.get() else None
    # Optional deadlines adapt to how long expression takes. The reading and the
    # review are different, required calls with their own inputs; their latency is kept apart.
    latency = _health_for(backend, "reading_latency" if reading else "review_latency" if review
                          else "generation_latency") if not _research.get() else None
    if health is not None and health.retry_after > time.monotonic():
        # No request was made: don't count or log each skipped attempt.
        raise OptionalInferenceCooldown("Optional inference is cooling down")
    deadline = _optional_timeout(backend, latency) if optional and not _research.get() else None
    if limit_seconds is not None and not _research.get():
        # A caller's own wall-clock budget, such as what is left of the reading's.
        deadline = limit_seconds if deadline is None else min(deadline, limit_seconds)
    if _research.get():
        output_tokens = None  # a study's own bounded limits stand
    token = _timeout.set(deadline)
    tokens_token = _output_tokens.set(output_tokens)
    phase_token = _active_phase.set(phase)
    started = time.monotonic()
    status = "ok"
    timed_out = unreached = False
    try:
        yield
    except Exception as error:
        status = type(error).__name__
        timed_out = is_timeout(error)
        unreached = is_transport_failure(error)
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
        if reading:
            item["reading"] = True
        if review:
            item["review"] = True
            if review_retry:
                item["review_retry"] = True
            if review_confirmation:
                item["review_confirmation"] = True
            if unreached:
                item["transport_failure"] = True
        if recovery:
            item["delivery_recovery"] = True
        if deadline is not None:
            item["timeout_seconds"] = deadline
        if output_tokens is not None:
            item["max_output_tokens"] = output_tokens
        if status == "ok" and backend is not None:
            # How the answer ended: an answer stopped at the output limit is incomplete.
            item.update(response_facts(backend))
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
        _output_tokens.reset(tokens_token)
        _active_phase.reset(phase_token)


def generate(backend, phase, system_prompt, user_message, *, optional=False, reading=False, review=False,
             review_retry=False, review_confirmation=False, limit_seconds=None, output_tokens=None,
             primary_tool=False, **kwargs):
    with model_call(phase, optional=optional, backend=backend, reading=reading, review=review,
                    review_retry=review_retry, review_confirmation=review_confirmation,
                    limit_seconds=limit_seconds, output_tokens=output_tokens, primary_tool=primary_tool):
        result = backend.generate(system_prompt, user_message, **kwargs)
        return strip_reasoning(result) if isinstance(result, str) else result

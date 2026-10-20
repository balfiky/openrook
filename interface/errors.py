"""Shared error redaction for API response bodies."""

from __future__ import annotations

from runtime.llm.diagnostics import (
    diagnose_llm_error,
    record_llm_health,
    redact_llm_detail,
)


def redact_error(message: str, config) -> str:
    """Strip configured secrets out of an error message before it is
    returned in a response body. ``config`` is a RuntimeConfig."""
    return redact_llm_detail(message, config)


def llm_error_payload(exc: Exception, config) -> dict | None:
    """Build the same recoverable error for legacy and versioned chat APIs."""
    diagnosis = diagnose_llm_error(exc, config)
    if diagnosis is None:
        return None
    record_llm_health(config, diagnosis)
    return {
        "detail": f"{diagnosis['message']} {diagnosis['action']}",
        "llm_error": diagnosis,
        "settings_url": diagnosis["settings_url"],
    }

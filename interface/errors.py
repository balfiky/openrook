"""Shared error redaction for API response bodies."""

from __future__ import annotations


def redact_error(message: str, config) -> str:
    """Strip configured secrets out of an error message before it is
    returned in a response body. ``config`` is a RuntimeConfig."""
    redacted = message
    for secret in (
        config.telegram_token,
        config.llm_api_key,
        config.api_key,
    ):
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted

"""Actionable, redacted backend diagnostics shared by setup and the web UI."""

from __future__ import annotations

import hashlib
import os
import re
import time
from urllib.parse import urlsplit, urlunsplit

import requests

from runtime.llm.backend import create_llm_backend

_HEALTH: dict[str, dict] = {}


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "[invalid URL]"


def redact_llm_detail(message: str, config) -> str:
    """Keep technical diagnostics useful without echoing configured credentials."""
    for secret in (
        config.llm_api_key,
        config.api_key,
        config.telegram_token,
        os.environ.get("LLM_API_KEY", ""),
    ):
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"https?://[^\s'\"<>]+", lambda m: _safe_url(m.group()), message)
    message = re.sub(
        r"(?i)(bearer\s+|(?:api[_-]?key|access_token|token)\s*[=:]\s*)[^\s,;\"']+",
        r"\1[redacted]",
        message,
    )
    return message[:1200]


def _backend_details(config) -> dict:
    return {
        "backend": config.llm_backend,
        "model": config.llm_model or ("Codex default" if config.llm_backend == "codex" else ""),
        "base_url": (
            _safe_url(config.llm_base_url)
            if config.llm_base_url and config.llm_backend != "codex" else ""
        ),
    }


def diagnose_llm_error(exc: Exception, config) -> dict | None:
    """Describe expected transport/config errors; leave programming errors alone."""
    text = str(exc)
    lower = text.lower()
    is_codex = config.llm_backend == "codex"
    expected_config = isinstance(exc, ValueError) and any(
        marker in lower for marker in ("llm_", "codex backend", "codex_workdir", "url")
    )
    expected_codex = is_codex and (
        isinstance(exc, OSError) or "codex" in lower
    )
    if not isinstance(exc, requests.RequestException) and not expected_config and not expected_codex:
        return None

    details = _backend_details(config)
    url = details["base_url"]
    model = details["model"]
    parsed_url = urlsplit(url)
    ollama = parsed_url.port == 11434 or "ollama" in (parsed_url.hostname or "").lower()
    label = "Codex CLI" if is_codex else ("Ollama" if ollama else config.llm_backend)
    target = f"{label} model '{model}'" if model else label
    if url:
        target += f" at {url}"
    response = getattr(exc, "response", None)
    status_code = response.status_code if response is not None else None
    code = "backend_error"
    message = f"{target} could not complete the request."
    action = "Check the backend, URL and model in Settings, then test the connection."

    if expected_config:
        code = "configuration"
        message = f"{target} is not configured correctly."
        action = "Check the backend configuration in Settings."
        if is_codex:
            if "codex_workdir" in lower:
                message = "The configured Codex working directory does not exist."
                action = "Set OPENROOK_CODEX_WORKDIR to an existing directory."
            elif "path" in lower or "requires the codex cli" in lower:
                message = "The Codex CLI could not be found."
                action = "Install the Codex CLI and make sure `codex` is on PATH, then run `codex login`."
            else:
                action = "Check OPENROOK_CODEX_WORKDIR and the Codex settings."
        elif config.llm_backend not in {"auto", "provider", "openai_compatible", "minimax"}:
            message = f"The selected backend '{config.llm_backend}' is not supported."
            action = "Choose a supported backend in Settings."
        elif not (config.llm_model or "").strip():
            message = "No model is selected."
            action = "Choose an available model in Settings, then test the connection."
        elif not (config.llm_base_url or "").strip() and config.llm_backend != "minimax":
            message = "No backend URL is configured."
            action = "Set the backend's Base URL in Settings, then test the connection."
        elif "url" in lower:
            message = "The configured backend URL is invalid or is not allowed."
            action = "Check the backend's Base URL in Settings."
    elif status_code in (401, 403) or (
        is_codex and any(word in lower for word in ("401", "403", "unauthorized", "not logged", "authentication", "login"))
    ):
        code = "authentication"
        message = f"{target} rejected authentication."
        action = "Run `codex login`, then retry." if is_codex else "Check the provider API key and model access in Settings."
    elif isinstance(exc, requests.Timeout) or "timed out" in lower:
        code = "timeout"
        message = f"{target} took too long to reply."
        action = "Increase Request timeout in Settings > LLM, then retry, or choose a faster model."
    elif isinstance(exc, requests.ConnectionError) or (
        is_codex and any(word in lower for word in ("connect", "network", "dns"))
    ):
        code = "unreachable"
        message = f"Cannot connect to {target}."
        action = "Is Ollama running? Start it with `ollama serve`, then retry." if ollama else "Check the backend is running, the URL and your network connection."
    elif status_code == 404:
        code = "not_found"
        message = f"{target} was not found; the server is reachable."
        action = (
            "Choose an available model in Settings and check the URL ends in `/v1`."
            if ollama else "Choose an available model in Settings and check the API URL."
        )
    elif status_code == 429 or (is_codex and any(word in lower for word in ("rate limit", "quota"))):
        code = "rate_limit"
        message = f"{target} is rate limited or out of quota."
        action = "Wait and retry, or check your provider usage and plan."
    elif status_code is not None and status_code >= 500:
        code = "provider_unavailable"
        message = f"{target} reported a server problem (HTTP {status_code})."
        action = "Wait and retry; check the backend service if the problem continues."
    elif is_codex and "empty response" in lower:
        code = "empty_response"
        message = f"{target} returned no reply."
        action = "Retry, or check the selected Codex model in Settings."

    return {
        **details,
        "code": code,
        "message": redact_llm_detail(message, config),
        "action": redact_llm_detail(action, config),
        "detail": redact_llm_detail(text, config),
        "settings_url": "/settings",
    }


def _health_key(config) -> str:
    values = (
        config.data_dir, config.llm_backend, config.llm_base_url, config.llm_model,
        config.llm_api_key or os.environ.get("LLM_API_KEY", ""),
        config.llm_request_timeout_seconds,
    )
    return hashlib.sha256(repr(values).encode()).hexdigest()


def record_llm_health(config, diagnosis: dict | None = None) -> None:
    """Remember the latest explicit check or failed request for these settings."""
    if len(_HEALTH) >= 128:
        _HEALTH.pop(next(iter(_HEALTH)))
    _HEALTH[_health_key(config)] = {
        **_backend_details(config),
        "status": "error" if diagnosis else "ok",
        "checked_at": time.time(),
        "diagnosis": diagnosis,
    }


def llm_health(config) -> dict:
    return _HEALTH.get(_health_key(config), {
        **_backend_details(config), "status": "untested", "checked_at": None, "diagnosis": None,
    })


def check_llm_connection(config, *, backend_factory=None) -> dict:
    """Perform one live sample, close its backend, and return a safe UI result."""
    result = {"ok": False, "checked": "llm", "live": True, **_backend_details(config)}
    backend = None
    try:
        backend = (backend_factory or create_llm_backend)(config)
        sample = backend.generate("Reply with ok.", "health check")
        if not sample.strip():
            raise ValueError("LLM backend returned an empty response")
        result.update(ok=True, sample_response=redact_llm_detail(sample[:200], config))
        record_llm_health(config)
    except Exception as exc:
        diagnosis = diagnose_llm_error(exc, config) or {
            **_backend_details(config),
            "code": "backend_error",
            "message": "The model connection test failed.",
            "action": "Check the backend, URL and model in Settings.",
            "detail": redact_llm_detail(str(exc), config),
            "settings_url": "/settings",
        }
        result.update(
            error=f"{diagnosis['message']} {diagnosis['action']}",
            error_detail=diagnosis["detail"],
            diagnostic=diagnosis,
        )
        record_llm_health(config, diagnosis)
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
    return result

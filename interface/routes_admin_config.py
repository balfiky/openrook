"""Admin console config endpoints: status, config, codex models, LLM/telegram/storage tests.

Split out of ``interface/routes_admin.py``, which aggregates this router and
re-exports the handlers. Handlers resolve shared helpers and mutable globals
late through the ``interface.api`` module object (``api_mod``) so test
monkeypatching on ``interface.api`` keeps working.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from interface import api as api_mod
from runtime.config import RuntimeConfig
from runtime.llm.diagnostics import check_llm_connection, llm_health
from runtime.llm.discovery import discover_llm_models
from runtime.sessions.manager import SessionManager

router = APIRouter(dependencies=[Depends(api_mod._require_bearer)])


class AdminConfigUpdateRequest(api_mod.ConfigUpdateRequest):
    """Admin-console config update.

    Extends the existing web Settings payload with optional setup-state
    handling while preserving the same save semantics.
    """

    setup_completed: bool | None = None
    consent_acknowledged: bool | None = None


class AdminLLMTestRequest(BaseModel):
    """Validate saved or draft LLM settings.

    ``live`` is intentionally opt-in so opening the admin console cannot
    accidentally call a paid or remote provider.
    """

    llm_backend: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_request_timeout_seconds: float | None = Field(default=None, ge=1, allow_inf_nan=False)
    llm_api_key: str = ""
    clear_llm_api_key: bool = False
    live: bool = False


class AdminLLMModelsRequest(BaseModel):
    """List models using saved settings or unsaved backend credentials."""

    llm_backend: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str = ""
    clear_llm_api_key: bool = False
    discovery_api: Literal["auto", "ollama", "openai"] = "auto"


class AdminTelegramTestRequest(BaseModel):
    telegram_token: str = ""
    clear_telegram_token: bool = False
    telegram_allowlist: list[str] | None = None


class AdminStorageTestRequest(BaseModel):
    data_dir: str | None = None
    tools_workspace: str | None = None
    create_missing: bool = True


@router.get("/admin/status")
async def admin_status() -> dict:
    """Operator-facing runtime/config status for the admin console."""
    config = api_mod._load_runtime_config()
    manager = api_mod.get_session_manager()
    return _admin_status_payload(config, manager)


@router.get("/admin/config")
async def admin_get_config() -> dict:
    """Redacted config plus metadata for the admin console."""
    config = api_mod._load_runtime_config()
    return _admin_config_payload(config, saved=True)


@router.get("/admin/codex/models")
async def admin_codex_models() -> dict:
    """Return the installed Codex CLI model catalog for settings UI dropdowns."""
    config = api_mod._load_runtime_config()
    if not api_mod.codex_cli_available():
        return {
            "ok": False,
            "available": False,
            "models": [],
            "selected_model": config.llm_model,
            "error": "Codex CLI is not on PATH.",
        }
    try:
        models = await asyncio.to_thread(api_mod.list_codex_models)
    except Exception as exc:
        return {
            "ok": False,
            "available": True,
            "models": [],
            "selected_model": config.llm_model,
            "error": api_mod._redact_error(str(exc), config),
        }
    return {
        "ok": True,
        "available": True,
        "models": models,
        "selected_model": config.llm_model,
    }


@router.post("/admin/llm/models")
async def admin_llm_models(req: AdminLLMModelsRequest) -> dict:
    """Discover models without saving draft settings or running generation."""
    existing = api_mod._load_runtime_config()
    config = RuntimeConfig(**existing.to_yaml_dict())
    if req.llm_backend is not None:
        config.llm_backend = req.llm_backend
    if req.llm_base_url is not None:
        config.llm_base_url = req.llm_base_url.strip()

    # A newly entered destination must not receive an unrelated saved/env key.
    same_destination = (
        config.llm_backend == existing.llm_backend
        and config.llm_base_url.rstrip("/") == existing.llm_base_url.rstrip("/")
    )
    key = existing.effective_llm_api_key if same_destination else ""
    if req.clear_llm_api_key:
        key = ""
    elif req.llm_api_key.strip():
        key = req.llm_api_key.strip()
    return await asyncio.to_thread(
        discover_llm_models, config, discovery_api=req.discovery_api, api_key=key,
    )


@router.post("/admin/config")
async def admin_update_config(req: AdminConfigUpdateRequest) -> dict:
    """Save config through the admin-console contract."""
    result = await api_mod.update_config(req)
    config = api_mod._load_runtime_config()
    if req.setup_completed is not None or req.consent_acknowledged:
        _write_admin_state(
            config,
            setup_completed=bool(req.setup_completed),
            last_config_save_at=time.time(),
            consent_acknowledged=req.consent_acknowledged or False,
        )
    return _admin_config_payload(
        config,
        saved=True,
        message=result.get("message"),
        reloaded_web_manager=bool(result.get("reloaded_web_manager")),
        restart_required_fields=(
            result.get("apply_state", {}).get("restart_required_fields", [])
        ),
        runtime_reload=result.get("runtime_reload", {}),
    )


@router.post("/admin/test/llm")
async def admin_test_llm(req: AdminLLMTestRequest) -> dict:
    """Validate LLM settings, with live provider calls opt-in."""
    config = _admin_config_for_llm_test(api_mod._load_runtime_config(), req)
    warnings = _admin_config_warnings(config)
    llm_warning_codes = {
        "missing_provider_base_url",
        "missing_provider_model",
        "missing_provider_key",
        "missing_openai_compatible_base_url",
        "missing_openai_compatible_model",
        "missing_codex_cli",
    }
    llm_warnings = [
        warning for warning in warnings
        if warning["code"] in llm_warning_codes
    ]
    if any(warning["severity"] == "error" for warning in llm_warnings):
        return {
            "ok": False,
            "checked": "llm",
            "live": False,
            "backend": config.llm_backend,
            "warnings": llm_warnings,
            "error": " ".join(warning["message"] for warning in llm_warnings),
        }

    result = {
        "ok": False,
        "checked": "llm",
        "live": False,
        "backend": config.llm_backend,
        "warnings": llm_warnings,
    }
    if req.live:
        result.update(await asyncio.to_thread(
            check_llm_connection, config, backend_factory=api_mod.create_llm_backend,
        ))
    else:
        result["error"] = "Live LLM test was not run; set live=true to verify connectivity."
    return result


@router.post("/admin/test/telegram")
async def admin_test_telegram(req: AdminTelegramTestRequest) -> dict:
    """Validate Telegram settings without starting long polling."""
    config = _admin_config_for_telegram_test(api_mod._load_runtime_config(), req)
    warnings: list[dict] = []
    if not config.telegram_token:
        warnings.append(_warning("error", "telegram_token", "missing_telegram_token", "Telegram bot token is not configured."))
    elif ":" not in config.telegram_token:
        warnings.append(_warning("warning", "telegram_token", "telegram_token_shape", "Telegram bot tokens usually contain a ':' separator."))

    invalid_allowlist = [
        item for item in config.telegram_allowlist
        if item and not item.lstrip("-").isdigit()
    ]
    if invalid_allowlist:
        warnings.append(_warning("warning", "telegram_allowlist", "telegram_allowlist_non_numeric", "Telegram allowlist should contain numeric user IDs."))

    return {
        "ok": not any(warning["severity"] == "error" for warning in warnings),
        "checked": "telegram",
        "configured": bool(config.telegram_token),
        "allowlist_count": len(config.telegram_allowlist),
        "warnings": warnings,
    }


@router.post("/admin/test/storage")
async def admin_test_storage(req: AdminStorageTestRequest) -> dict:
    """Validate data and tool workspace paths."""
    config = _admin_config_for_storage_test(api_mod._load_runtime_config(), req)
    checks = [
        _check_writable_directory("data_dir", config.data_dir, req.create_missing),
        _check_writable_directory(
            "tools_workspace",
            config.resolved_tools_workspace,
            req.create_missing,
        ),
    ]
    warnings = [
        warning for check in checks
        for warning in check.get("warnings", [])
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "checked": "storage",
        "checks": checks,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Payload builders and config introspection helpers
# ---------------------------------------------------------------------------


_SECRET_ENV_VARS = {
    "telegram_token": (),
    "llm_api_key": ("LLM_API_KEY",),
    "api_key": (),
}

_CONFIG_FIELD_SECTIONS = {
    "data_dir": "runtime",
    "max_queue_per_user": "runtime",
    "max_active_sessions": "runtime",
    "session_timeout_seconds": "runtime",
    "console_enabled": "runtime",
    "debug_host": "runtime",
    "debug_port": "runtime",
    "llm_backend": "model",
    "llm_base_url": "model",
    "llm_model": "model",
    "llm_api_key": "model",
    "llm_request_timeout_seconds": "model",
    "llm_auxiliary_timeout_seconds": "model",
    "llm_auxiliary_cooldown_seconds": "model",
    "api_key": "access",
    "cors_origins": "access",
    "telegram_token": "channels",
    "telegram_allowlist": "channels",
    "telegram_poll_timeout": "channels",
    "dedupe_ttl": "channels",
    "tools_enabled": "tools",
    "autonomy_level": "tools",
    "tools_workspace": "tools",
    "shell_tool_enabled": "tools",
    "shell_command_allowlist": "tools",
    "mcp_tool_categories": "tools",
    "mcp_strict_categories": "tools",
    "proactive_enabled": "runtime",
    "proactive_idle_threshold": "runtime",
    "proactive_density_reference": "runtime",
    "proactive_recovery_seconds": "runtime",
    "proactive_check_interval": "runtime",
    "character_independence": "runtime",
    "appraisal_use_llm": "runtime",
    "autobiographical_learning": "runtime",
    "coherence_min_score": "runtime",
    "coherence_max_regenerations": "runtime",
    "pending_intake_ttl_turns": "runtime",
    "learning_budget_kind": "learning",
    "learning_max_questions_per_day": "learning",
    "learning_max_seconds_per_day": "learning",
    "metabolism_min_elapsed_days": "learning",
    "owner_chat_id": "channels",
}

_SESSION_RELOAD_FIELDS = {
    "data_dir",
    "max_queue_per_user",
    "max_active_sessions",
    "session_timeout_seconds",
    "llm_backend",
    "llm_base_url",
    "llm_model",
    "llm_api_key",
    "llm_request_timeout_seconds",
    "llm_auxiliary_timeout_seconds",
    "llm_auxiliary_cooldown_seconds",
    "proactive_enabled",
    "proactive_idle_threshold",
    "proactive_density_reference",
    "proactive_recovery_seconds",
    "proactive_check_interval",
    "character_independence",
    "appraisal_use_llm",
    "autobiographical_learning",
    "coherence_min_score",
    "coherence_max_regenerations",
    "pending_intake_ttl_turns",
    "tools_enabled",
    "autonomy_level",
    "tools_workspace",
    "shell_tool_enabled",
    "shell_command_allowlist",
    "mcp_tool_categories",
    "mcp_strict_categories",
    "learning_budget_kind",
    "learning_max_questions_per_day",
    "learning_max_seconds_per_day",
    "metabolism_min_elapsed_days",
}
_LIVE_RELOAD_FIELDS = {"api_key", "telegram_token", "telegram_allowlist", "telegram_poll_timeout", "dedupe_ttl"}


def _admin_config_payload(
    config: RuntimeConfig,
    *,
    saved: bool,
    message: str | None = None,
    reloaded_web_manager: bool = False,
    restart_required_fields: list[str] | None = None,
    runtime_reload: dict | None = None,
) -> dict:
    """Serialize config and metadata for the production admin console."""
    return {
        **api_mod._config_payload(
            config,
            saved=saved,
            message=message,
            reloaded_web_manager=reloaded_web_manager,
            restart_required_fields=restart_required_fields,
            runtime_reload=runtime_reload,
        ),
        "field_metadata": _admin_field_metadata(config),
        "warnings": _admin_config_warnings(config),
        "setup": _admin_setup_status(config),
        "llm_health": llm_health(config),
    }


def _read_soul_name() -> str:
    """Return the agent name from soul.yaml without going through the async config."""
    try:
        import yaml as _yaml
        path = api_mod._soul_yaml_path()
        with open(path, encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
        return (data.get("soul") or {}).get("name") or "OpenRook"
    except Exception:
        return "OpenRook"


def _admin_status_payload(config: RuntimeConfig, manager: SessionManager) -> dict:
    """Operator-facing status summary used by the admin overview."""
    return {
        "status": "ok",
        "config_path": os.path.abspath(api_mod.RUNTIME_CONFIG_PATH),
        "auth_enabled": bool(config.api_key),
        "soul_name": _read_soul_name(),
        "llm_backend": config.llm_backend,
        "llm_configured": _llm_configured(config),
        "llm_health": llm_health(config),
        "telegram_configured": bool(config.telegram_token),
        "tools": {
            "enabled": config.tools_enabled,
            "workspace": config.resolved_tools_workspace,
            "shell_enabled": config.shell_tool_enabled,
        },
        "sessions": {
            "active": len(manager.active_sessions),
            "max_active": config.max_active_sessions,
        },
        "setup": _admin_setup_status(config),
        "warnings": _admin_config_warnings(config),
    }


def _admin_field_metadata(config: RuntimeConfig) -> list[dict]:
    defaults = RuntimeConfig()
    public_values = config.to_public_dict()
    secret_status = _admin_secret_status(config)
    fields_meta: list[dict] = []
    for field_name, value in public_values.items():
        metadata = {
            "name": field_name,
            "section": _CONFIG_FIELD_SECTIONS.get(field_name, "runtime"),
            "value": value,
            "default": getattr(defaults, field_name),
            "secret": field_name in secret_status,
            "restart_required": field_name in api_mod._RESTART_REQUIRED_FIELDS,
            "session_manager_reload": field_name in _SESSION_RELOAD_FIELDS,
            "live_reload": field_name in _LIVE_RELOAD_FIELDS,
            "allowed_values": _allowed_values_for_field(field_name),
        }
        if field_name in secret_status:
            metadata["secret_status"] = secret_status[field_name]
        fields_meta.append(metadata)
    return fields_meta


def _admin_secret_status(config: RuntimeConfig) -> dict[str, dict]:
    status: dict[str, dict] = {}
    for field_name, env_names in _SECRET_ENV_VARS.items():
        has_yaml_value = bool(getattr(config, field_name))
        configured_env = next((name for name in env_names if os.environ.get(name)), "")
        status[field_name] = {
            "configured": has_yaml_value or bool(configured_env),
            "stored": has_yaml_value,
            "source": "yaml" if has_yaml_value else ("environment" if configured_env else "unset"),
            "env_var": configured_env,
        }
    return status


def _allowed_values_for_field(field_name: str) -> list[str] | None:
    if field_name == "llm_backend":
        return ["auto", "provider", "openai_compatible", "codex"]
    if field_name == "autonomy_level":
        return ["off", "assisted", "autonomous", "high_risk"]
    if field_name == "learning_budget_kind":
        return ["local"]  # cloud is a placeholder; not yet implemented
    return None


def _admin_config_warnings(config: RuntimeConfig) -> list[dict]:
    warnings: list[dict] = []
    backend = config.llm_backend
    secret_status = _admin_secret_status(config)
    has_generic_key = secret_status["llm_api_key"]["configured"]

    if backend == "provider":
        if not config.llm_base_url.strip():
            warnings.append(_warning("error", "llm_base_url", "missing_provider_base_url", "Hosted provider backend requires llm_base_url."))
        if not config.llm_model.strip():
            warnings.append(_warning("error", "llm_model", "missing_provider_model", "Hosted provider backend requires llm_model."))
        if not has_generic_key:
            warnings.append(_warning("warning", "llm_api_key", "missing_provider_key", "Hosted provider backend usually requires an API key."))
    elif backend == "openai_compatible":
        if not config.llm_base_url.strip():
            warnings.append(_warning("error", "llm_base_url", "missing_openai_compatible_base_url", "OpenAI-compatible backend requires llm_base_url."))
        if not config.llm_model.strip():
            warnings.append(_warning("error", "llm_model", "missing_openai_compatible_model", "OpenAI-compatible backend requires llm_model."))
    elif backend == "codex" and not api_mod.codex_cli_available():
        warnings.append(_warning("error", "llm_backend", "missing_codex_cli", "Codex backend requires the codex CLI on PATH. Install it, then run `codex login`."))
    elif backend == "minimax":
        if not config.llm_model.strip():
            warnings.append(_warning("error", "llm_model", "missing_provider_model", "Choose a model in Settings for the legacy MiniMax backend."))
        if not has_generic_key:
            warnings.append(_warning("error", "llm_api_key", "missing_provider_key", "Legacy MiniMax backend requires the generic LLM API key."))

    if config.tools_enabled and not config.api_key:
        warnings.append(_warning("warning", "tools_enabled", "tools_without_auth", "Agentic tools are enabled with no bearer auth. Keep this instance on a trusted network."))
    if config.shell_tool_enabled and not config.tools_enabled:
        warnings.append(_warning("error", "shell_tool_enabled", "shell_without_tools", "shell_tool_enabled has no effect unless tools_enabled is true."))
    if config.shell_tool_enabled and not config.api_key:
        warnings.append(_warning("warning", "shell_tool_enabled", "shell_without_auth", "Shell tool execution is enabled with no bearer auth. Keep this instance on a trusted network."))
    if config.autonomy_level == "high_risk":
        warnings.append(_warning("warning", "autonomy_level", "high_risk_autonomy", "High-risk autonomy allows the assistant to act with fewer confirmations inside enabled tool scopes."))
    if config.cors_origins and not config.api_key:
        warnings.append(_warning("warning", "cors_origins", "cors_without_auth", "CORS origins are configured while api_key is empty."))
    if config.debug_host not in {"127.0.0.1", "localhost", "::1"} and not config.api_key:
        warnings.append(_warning("error", "api_key", "public_bind_without_auth", "Server bind host is not localhost while api_key is empty."))

    health = llm_health(config)
    if health["status"] == "error":
        diagnosis = health["diagnosis"]
        warnings.append(_warning(
            "error", "llm_backend", "llm_backend_error",
            f"{diagnosis['message']} {diagnosis['action']}",
        ))

    return warnings


def _warning(severity: str, field: str, code: str, message: str) -> dict:
    return {
        "severity": severity,
        "field": field,
        "code": code,
        "message": message,
    }


def _llm_configured(config: RuntimeConfig) -> bool:
    backend = config.llm_backend
    secret_status = _admin_secret_status(config)
    if backend == "provider":
        return bool(
            config.llm_base_url.strip()
            and config.llm_model.strip()
            and secret_status["llm_api_key"]["configured"]
        )
    if backend == "openai_compatible":
        return bool(config.llm_base_url.strip() and config.llm_model.strip())
    if backend == "codex":
        return api_mod.codex_cli_available()
    if backend == "minimax":
        return bool(config.llm_model.strip() and secret_status["llm_api_key"]["configured"])
    if backend == "auto":
        return bool(config.llm_base_url.strip() and config.llm_model.strip())
    return False


def _write_admin_state(
    config: RuntimeConfig,
    *,
    setup_completed: bool,
    last_config_save_at: float | None = None,
    consent_acknowledged: bool | None = None,
) -> None:
    path = api_mod._admin_state_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = api_mod._read_admin_state(config)
    now = time.time()
    data = {
        **existing,
        "setup_completed": setup_completed,
        "completed_at": existing.get("completed_at") if setup_completed else None,
        "last_config_save_at": last_config_save_at if last_config_save_at is not None else now,
    }
    if setup_completed and not data["completed_at"]:
        data["completed_at"] = now
    if consent_acknowledged and not existing.get("consent_acknowledged_at"):
        data["consent_acknowledged_at"] = now
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _admin_setup_status(config: RuntimeConfig) -> dict:
    state = api_mod._read_admin_state(config)
    completed = bool(state.get("setup_completed"))
    reasons: list[str] = []
    if not os.path.exists(api_mod.RUNTIME_CONFIG_PATH):
        reasons.append("missing_config")
    if _looks_unconfigured(config):
        reasons.append("default_or_minimal_config")
    if api_mod._soul_looks_default():
        reasons.append("default_soul")
    if not completed:
        reasons.append("setup_not_completed")
    required = not completed and bool(reasons)
    return {
        "required": required,
        "completed": completed,
        "state_path": os.path.abspath(api_mod._admin_state_path(config)),
        "reasons": reasons,
        "completed_at": state.get("completed_at"),
        "last_config_save_at": state.get("last_config_save_at"),
        "consent_acknowledged_at": state.get("consent_acknowledged_at"),
    }


def _looks_unconfigured(config: RuntimeConfig) -> bool:
    return (
        config.llm_backend == "auto"
        and not config.llm_base_url.strip()
        and not config.llm_model.strip()
        and not config.llm_api_key
        and not config.telegram_token
        and not config.api_key
        and not config.tools_enabled
    )


# ---------------------------------------------------------------------------
# Test-request config overlays and storage checks
# ---------------------------------------------------------------------------


def _admin_config_for_llm_test(
    existing: RuntimeConfig,
    req: AdminLLMTestRequest,
) -> RuntimeConfig:
    config = RuntimeConfig(**existing.to_yaml_dict())
    if req.llm_backend is not None:
        config.llm_backend = req.llm_backend
    if req.llm_base_url is not None:
        config.llm_base_url = req.llm_base_url.strip()
    if req.llm_model is not None:
        config.llm_model = req.llm_model.strip()
    if req.llm_request_timeout_seconds is not None:
        config.llm_request_timeout_seconds = req.llm_request_timeout_seconds
    if req.clear_llm_api_key:
        config.llm_api_key = ""
    elif req.llm_api_key.strip():
        config.llm_api_key = req.llm_api_key.strip()
    return config


def _admin_config_for_telegram_test(
    existing: RuntimeConfig,
    req: AdminTelegramTestRequest,
) -> RuntimeConfig:
    config = RuntimeConfig(**existing.to_yaml_dict())
    if req.clear_telegram_token:
        config.telegram_token = ""
    elif req.telegram_token.strip():
        config.telegram_token = req.telegram_token.strip()
    if req.telegram_allowlist is not None:
        config.telegram_allowlist = {
            item.strip()
            for item in req.telegram_allowlist
            if item.strip()
        }
    return config


def _admin_config_for_storage_test(
    existing: RuntimeConfig,
    req: AdminStorageTestRequest,
) -> RuntimeConfig:
    config = RuntimeConfig(**existing.to_yaml_dict())
    if req.data_dir is not None:
        config.data_dir = req.data_dir.strip() or "data"
    if req.tools_workspace is not None:
        config.tools_workspace = req.tools_workspace.strip()
    return config


def _check_writable_directory(
    name: str,
    path: str,
    create_missing: bool,
) -> dict:
    resolved = os.path.realpath(path)
    warnings: list[dict] = []
    try:
        if create_missing:
            os.makedirs(resolved, exist_ok=True)
        exists = os.path.isdir(resolved)
        if not exists:
            warnings.append(_warning("error", name, f"{name}_missing", f"{name} does not exist."))
            return {
                "name": name,
                "ok": False,
                "path": resolved,
                "exists": False,
                "writable": False,
                "warnings": warnings,
            }
        writable = os.access(resolved, os.W_OK)
        if not writable:
            warnings.append(_warning("error", name, f"{name}_not_writable", f"{name} is not writable."))
        return {
            "name": name,
            "ok": writable,
            "path": resolved,
            "exists": exists,
            "writable": writable,
            "warnings": warnings,
        }
    except OSError as exc:
        warnings.append(_warning("error", name, f"{name}_error", str(exc)))
        return {
            "name": name,
            "ok": False,
            "path": resolved,
            "exists": False,
            "writable": False,
            "warnings": warnings,
        }

"""FastAPI backend for OpenRook."""

from __future__ import annotations

from config.environment import project_env

import asyncio
import json
import logging
import os
import secrets
import sys
import time

# ``python -m interface.api`` executes this file as ``__main__``. The route
# modules imported at the bottom do ``from interface import api`` — without
# this alias that import would re-execute the file a second time and hit a
# circular import against the partially-initialized route modules. Register
# the running module under its canonical name before anything else.
if __name__ == "__main__" and "interface.api" not in sys.modules:  # pragma: no cover
    sys.modules["interface.api"] = sys.modules["__main__"]
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from interface.auth import build_require_bearer, verify_bearer
from interface.errors import llm_error_payload, redact_error
from interface.v1 import build_v1_router
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.debug.api import _debug_to_dict as _serialize_debug
# codex_cli_available and list_codex_models are re-exported as patchable
# module attributes; route modules call them via the interface.api module
# object so tests can monkeypatch them here.
from runtime.llm.backend import (  # noqa: F401
    codex_cli_available,
    create_llm_backend,
    list_codex_models,
)
from runtime.sessions.manager import SessionManager
from runtime.sessions.user_session import UserSession
from runtime.tools import create_tool_executor

_log = logging.getLogger(__name__)

_telegram_task: asyncio.Task | None = None
_telegram_channel = None  # TelegramChannel | None
_serve_started_at: float = time.time()
_background_shutdown_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Start Telegram channel if configured
    config = _load_runtime_config()
    await _restart_telegram_channel(config)
    from pathlib import Path

    if config.proactive_enabled or (Path(config.data_dir) / "shared" / "agent_runtime.db").is_file():
        get_session_manager().start_background()
    yield
    # Shutdown
    global _session_manager, _pipeline_override
    await _stop_telegram_channel()
    if _session_manager is not None:
        await _session_manager.shutdown()
        _session_manager = None
    if _background_shutdown_tasks:
        tasks = list(_background_shutdown_tasks)
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    if _pipeline_override is not None:
        _pipeline_override.close()
        _pipeline_override = None


async def _start_telegram(config: RuntimeConfig) -> None:
    """Start Telegram long-polling as a background task."""
    global _telegram_channel
    from runtime.channels.telegram import (
        TelegramChannel, TelegramClient, TelegramConfig,
    )
    tg_config = TelegramConfig(
        token=config.telegram_token,
        allowlist=config.telegram_allowlist,
        poll_timeout=config.telegram_poll_timeout,
        dedupe_ttl=config.dedupe_ttl,
    )
    client = TelegramClient(tg_config.token)
    manager = get_session_manager()
    channel = TelegramChannel(client, manager, tg_config)
    _telegram_channel = channel
    _log.info("Starting Telegram channel (polling)")
    try:
        await channel.start()
    except asyncio.CancelledError:
        pass
    finally:
        if _telegram_channel is channel:
            _telegram_channel = None
        await channel.stop()


async def _stop_telegram_channel() -> None:
    """Stop the standalone web Telegram poller if it is running."""
    global _telegram_task, _telegram_channel

    task = _telegram_task
    _telegram_task = None
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            _log.exception("Telegram channel task failed")

    if _telegram_channel is not None:
        channel = _telegram_channel
        _telegram_channel = None
        await channel.stop()


def _log_telegram_task_exception(task: asyncio.Task) -> None:
    """Surface Telegram startup/runtime errors instead of swallowing them.

    Without this callback, exceptions raised before explicit shutdown would
    only appear as an "unretrieved task exception" warning at GC time, and
    the rest of the app would keep serving requests believing Telegram was
    polling normally.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    _log.error("Telegram channel task crashed: %s", exc, exc_info=exc)


async def _restart_telegram_channel(config: RuntimeConfig) -> None:
    """Reload Telegram polling to match the latest standalone web config."""
    global _telegram_task

    await _stop_telegram_channel()
    if config.telegram_token:
        from runtime.channels.telegram import is_telegram_token_pollable

        if not is_telegram_token_pollable(config.telegram_token):
            _log.warning(
                "Telegram token is configured but malformed; polling is not started"
            )
            return
        _telegram_task = asyncio.create_task(_start_telegram(config))
        _telegram_task.add_done_callback(_log_telegram_task_exception)


app = FastAPI(
    title="OpenRook",
    version="1.0.0",
    lifespan=_lifespan,
    docs_url=None if project_env("DISABLE_DOCS") else "/docs",
    redoc_url=None if project_env("DISABLE_DOCS") else "/redoc",
    openapi_url=None if project_env("DISABLE_DOCS") else "/openapi.json",
    description=(
        "OpenRook cognitive API. Legacy endpoints at the root serve the "
        "bundled web UI. The stable integration surface lives under /v1 "
        "(see /docs for the full OpenAPI schema)."
    ),
)


@app.middleware("http")
async def backend_error_responses(request, call_next):
    """Use the chat diagnostic for expected backend errors on any HTTP route."""
    try:
        return await call_next(request)
    except Exception as exc:
        payload = llm_error_payload(exc, _current_config_for_middleware())
        if payload is None:
            raise
        return JSONResponse(status_code=503, content=payload)


@app.middleware("http")
async def research_observer_controls(request, call_next):
    manager = globals().get("_session_manager")
    config = manager.config if manager is not None else _load_runtime_config()
    if config.research_mode and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        path = request.url.path
        if (path.startswith(("/admin/", "/config", "/v1/config"))
                or path in {"/v1/sessions/rest", "/sessions/rest", "/rest"}
                or request.method == "DELETE"):
            return JSONResponse(status_code=409, content={
                "detail": "Research observation is read-only. Record an intervention in a new run to change initial conditions."
            })
    return await call_next(request)

WEB_PLATFORM = "web"
RUNTIME_CONFIG_PATH = project_env("RUNTIME_CONFIG", "runtime_config.yaml")
_MAX_CHAT_MESSAGE_CHARS = 16_000
_MAX_ID_CHARS = 128
_MAX_WS_MESSAGE_CHARS = 20_000
_HTML_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "connect-src 'self' http://127.0.0.1:* http://localhost:*; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    ),
}

_session_manager: SessionManager | None = None
_pipeline_override: CognitivePipeline | None = None


def _current_config_for_middleware() -> RuntimeConfig:
    """Return whichever RuntimeConfig the v1 router should consult.

    Kept as a function so tests and in-process reconfiguration reflect
    immediately without re-mounting the router.
    """
    return RuntimeConfig.from_yaml(RUNTIME_CONFIG_PATH)


def _verify_bearer(token: str | None) -> None:
    """Raise 401 unless ``token`` matches the currently-configured api_key.

    If no api_key is configured, auth is disabled and this is a no-op. The
    config is re-read on every call so rotating the key through POST /config
    takes effect immediately.
    """
    verify_bearer(token, _current_config_for_middleware().api_key)


# FastAPI dependency that protects legacy and admin endpoints when api_key is
# set. Shared with the /v1 router via interface.auth; the config is re-read
# per request so key rotation takes effect immediately.
_require_bearer = build_require_bearer(_current_config_for_middleware)


# CORS — origins are read from runtime_config.yaml at startup. Empty list
# means same-origin only (browsers block cross-origin XHR), which is the
# safe default. Changes to ``cors_origins`` take effect on app restart.
_cors_cfg = _current_config_for_middleware()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_cors_cfg.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the versioned integration API at /v1.
app.include_router(
    build_v1_router(
        session_manager_getter=lambda: get_session_manager(),
        config_getter=_current_config_for_middleware,
        serve_started_at=_serve_started_at,
    )
)


_DISALLOWED_ID_MARKERS = ("/", "\\", "..", ":", "\x00")


def _reject_path_traversal_ids(value: str) -> str:
    """Reject identifier values that could escape the per-user data dir.

    user_id/chat_id become filesystem path segments (``data_dir/<key>``), so a
    value like ``../../tmp/x`` would traverse out. ``user_data_dir`` sanitizes
    as a backstop; this validator fails loudly with a 422 at the API boundary.
    """
    for marker in _DISALLOWED_ID_MARKERS:
        if marker in value:
            raise ValueError("must not contain path separators, '..', ':' or NUL")
    return value


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_CHAT_MESSAGE_CHARS)
    user_id: str = Field("default", min_length=1, max_length=_MAX_ID_CHARS)
    chat_id: str = Field("default", min_length=1, max_length=_MAX_ID_CHARS)

    _check_ids = field_validator("user_id", "chat_id")(_reject_path_traversal_ids)


class ChatResponse(BaseModel):
    response: str
    debug: dict


class EndSessionRequest(BaseModel):
    user_id: str = Field("default", min_length=1, max_length=_MAX_ID_CHARS)
    chat_id: str = Field("default", min_length=1, max_length=_MAX_ID_CHARS)

    _check_ids = field_validator("user_id", "chat_id")(_reject_path_traversal_ids)


class RestRequest(BaseModel):
    hours: float = 1.0
    user_id: str = Field("default", min_length=1, max_length=_MAX_ID_CHARS)
    chat_id: str = Field("default", min_length=1, max_length=_MAX_ID_CHARS)

    _check_ids = field_validator("user_id", "chat_id")(_reject_path_traversal_ids)


class ConfigUpdateRequest(BaseModel):
    """Partial config update — every field uses ``None = leave unchanged``.

    Sending a partial payload no longer wipes the rest of the config. Only
    secrets (``telegram_token``, ``llm_api_key``, ``api_key``) have their
    own ``clear_*`` flags for explicit removal; blank-string for secrets
    still means "keep existing".
    """

    data_dir: str | None = None
    max_queue_per_user: int | None = Field(default=None, ge=1)
    max_active_sessions: int | None = Field(default=None, ge=1)
    session_timeout_seconds: float | None = Field(default=None, ge=1)
    console_enabled: bool | None = None
    telegram_allowlist: list[str] | None = None
    telegram_poll_timeout: int | None = Field(default=None, ge=1)
    dedupe_ttl: float | None = Field(default=None, ge=0)
    llm_backend: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    debug_host: str | None = None
    debug_port: int | None = Field(default=None, ge=1, le=65535)
    proactive_enabled: bool | None = None
    proactive_idle_threshold: float | None = Field(default=None, ge=0)
    proactive_density_reference: int | None = Field(default=None, ge=0)
    proactive_recovery_seconds: float | None = Field(default=None, ge=0)
    proactive_check_interval: float | None = Field(default=None, ge=1)
    character_independence: bool | None = None
    research_mode: bool | None = None
    research_birth_policy: str | None = Field(default=None, pattern="^(uncommitted|configured)$")
    research_action_framing: str | None = Field(default=None, pattern="^(agent_owned|conversational)$")
    research_action_feedback: str | None = Field(default=None, pattern="^(receipts|legacy)$")
    research_execution_mode: str | None = Field(default=None, pattern="^(batch|closed_loop)$")
    research_context_mode: str | None = Field(default=None, pattern="^(full|task_only)$")
    research_memory_mode: str | None = Field(default=None, pattern="^(legacy|none|episodes|procedures|raw)$")
    research_progress_estimates: bool | None = None
    research_appraisal_mode: str | None = Field(default=None, pattern="^(legacy|expression|attention|neutral)$")
    research_dispositions: bool | None = None
    research_json_schema: bool | None = None
    research_inference_retries: int | None = Field(default=None, ge=0, le=2, strict=True)
    research_max_steps_per_activation: int | None = Field(default=None, ge=1, le=16, strict=True)
    research_initial_session: str | None = None
    research_calls_per_day: int | None = Field(default=None, ge=1)
    research_episode_idle_seconds: float | None = Field(default=None, ge=0)
    research_episode_max_turns: int | None = Field(default=None, ge=1)
    research_reflection_interval: float | None = Field(default=None, gt=0)
    appraisal_use_llm: bool | None = None
    autobiographical_learning: bool | None = None
    coherence_min_score: float | None = Field(default=None, ge=0, le=1)
    coherence_max_regenerations: int | None = Field(default=None, ge=0, le=5)
    pending_intake_ttl_turns: int | None = Field(default=None, ge=1, le=20)
    telegram_token: str = ""
    llm_api_key: str = ""
    api_key: str = ""
    cors_origins: list[str] | None = None
    tools_enabled: bool | None = None
    autonomy_level: str | None = None
    tools_workspace: str | None = None
    shell_tool_enabled: bool | None = None
    shell_command_allowlist: list[str] | None = None
    mcp_tool_categories: dict[str, str] | None = None
    mcp_strict_categories: bool | None = None
    learning_budget_kind: str | None = None
    learning_max_questions_per_day: int | None = Field(default=None, ge=0, le=200)
    learning_max_seconds_per_day: float | None = Field(default=None, ge=0)
    metabolism_min_elapsed_days: float | None = Field(default=None, ge=0, le=30)
    owner_chat_id: str | None = None
    clear_telegram_token: bool = False
    clear_llm_api_key: bool = False
    clear_api_key: bool = False


class AdminRuntimeRestartRequest(BaseModel):
    confirmation: str


class AdminSoulDraftRequest(BaseModel):
    """LLM-assisted soul drafting from a natural-language description."""

    description: str = Field(..., min_length=8, max_length=16000)

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 8:
            raise ValueError("description must be at least 8 characters")
        return stripped


class AdminSoulImportRequest(BaseModel):
    """Raw soul.yaml or identity document pasted by an operator."""

    raw: str = Field(..., min_length=8, max_length=20000)

    @field_validator("raw")
    @classmethod
    def _strip_raw(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 8:
            raise ValueError("raw must be at least 8 characters")
        return stripped


class AdminSoulUpdateRequest(BaseModel):
    """Validated soul.yaml replacement.

    Weights must be in [0.0, 1.0]; free-text fields are bounded so a pasted
    book cannot become the identity prompt; lists are trimmed to non-empty
    entries.
    """

    name: str = Field(..., max_length=64)
    identity: str = Field(default="", max_length=2000)
    voice: str = Field(default="", max_length=2000)
    relational_stance: str = Field(default="", max_length=2000)
    growth_policy: str = Field(default="", max_length=2000)
    likes: list[str] = Field(default_factory=list, max_length=32)
    dislikes: list[str] = Field(default_factory=list, max_length=32)
    boundaries: list[str] = Field(default_factory=list, max_length=32)
    core_values: dict[str, float] = Field(default_factory=dict)
    initial_traits: dict[str, float] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name is required and cannot be blank")
        return stripped

    @field_validator("identity", "voice", "relational_stance", "growth_policy")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("likes", "dislikes", "boundaries")
    @classmethod
    def _strip_items(cls, values: list[str]) -> list[str]:
        cleaned = [item.strip() for item in values]
        cleaned = [item for item in cleaned if item]
        for item in cleaned:
            if len(item) > 200:
                raise ValueError(f"item too long (max 200 chars): {item[:40]}...")
        return cleaned

    @field_validator("core_values", "initial_traits")
    @classmethod
    def _validate_weights(cls, values: dict[str, float]) -> dict[str, float]:
        if len(values) > 32:
            raise ValueError("too many entries (max 32)")
        cleaned: dict[str, float] = {}
        for key, raw in values.items():
            key = (key or "").strip()
            if not key:
                raise ValueError("weight key cannot be empty")
            if len(key) > 64:
                raise ValueError(f"weight key too long (max 64 chars): {key[:40]}...")
            try:
                weight = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"weight for {key!r} must be numeric") from exc
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"weight for {key!r} must be in [0.0, 1.0], got {weight}"
                )
            cleaned[key] = weight
        return cleaned


def get_session_manager() -> SessionManager:
    """Create the shared web SessionManager lazily."""
    global _session_manager
    if _session_manager is None:
        config = _load_runtime_config()
        _session_manager = SessionManager(
            config=config,
            backend_factory=lambda: create_llm_backend(config),
            tool_executor_factory=lambda: create_tool_executor(config),
        )
    return _session_manager


def set_session_manager(manager: SessionManager | None) -> None:
    """Testing hook to replace the default web SessionManager."""
    global _session_manager, _pipeline_override
    _session_manager = manager
    _pipeline_override = None


def set_pipeline(pipeline: CognitivePipeline | None) -> None:
    """Focused testing hook that bypasses the SessionManager."""
    global _pipeline_override, _session_manager
    _pipeline_override = pipeline
    _session_manager = None


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return brand_asset("openrook-icon.svg")


@app.post("/chat", dependencies=[Depends(_require_bearer)])
async def chat(req: ChatRequest):
    try:
        if _pipeline_override is not None:
            result = _pipeline_override.process(req.message, user_id=req.user_id)
            return ChatResponse(response=result.response, debug=_serialize_debug(result.debug))

        manager = get_session_manager()
        response = await manager.handle_message(
            WEB_PLATFORM,
            req.user_id,
            req.chat_id,
            req.message,
        )
        session = await manager.ensure_session(WEB_PLATFORM, req.user_id, req.chat_id)
        debug = _serialize_debug(session.last_debug) if session.last_debug else {}
        return ChatResponse(response=response, debug=debug)
    except HTTPException:
        raise
    except Exception as exc:
        config = _load_runtime_config()
        backend_error = llm_error_payload(exc, config)
        if backend_error is not None:
            _log.warning("Chat unavailable: %s", backend_error["detail"])
            return JSONResponse(status_code=503, content=backend_error)
        if isinstance(exc, RuntimeError):
            detail = _redact_error(str(exc), config)
            _log.warning("Chat unavailable: %s", detail)
            raise HTTPException(status_code=503, detail=detail) from exc
        _log.exception("Chat error")
        detail = str(exc)
        try:
            detail = _redact_error(detail, _load_runtime_config())
        except Exception:
            # Config unreadable — don't risk echoing unredacted secrets.
            detail = "Internal server error"
        return JSONResponse(
            status_code=500,
            content={"detail": detail},
        )


@app.get("/debug", dependencies=[Depends(_require_bearer)])
async def debug(user_id: str = "default", chat_id: str = "default") -> dict:
    for _name, _val in (("user_id", user_id), ("chat_id", chat_id)):
        try:
            _reject_path_traversal_ids(_val)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{_name}: {exc}") from exc
    if _pipeline_override is not None:
        unresolved = _pipeline_override.engine.active_unresolved()
        return {
            "modulator_snapshot": _pipeline_override.engine.snapshot(),
            "emotion_label": _pipeline_override.engine.to_emotion_label(),
            "energy": _pipeline_override.engine.state.energy,
            "short_term_count": len(_pipeline_override.short_term),
            "long_term_count": _pipeline_override.long_term.count(),
            "unresolved_count": len(unresolved),
            "unresolved_items": [item.to_dict() for item in unresolved],
        }

    manager = get_session_manager()
    session = await manager.ensure_session(WEB_PLATFORM, user_id, chat_id)
    return _session_debug(_session_key(user_id, chat_id), session)


@app.get("/config", dependencies=[Depends(_require_bearer)])
async def get_config() -> dict:
    config = _load_runtime_config()
    return _config_payload(config, saved=True)


@app.post("/config", dependencies=[Depends(_require_bearer)])
async def update_config(req: ConfigUpdateRequest) -> dict:
    existing = _load_runtime_config()

    def _keep(req_val, existing_val):
        """Plain leave-unchanged: ``None`` from req means preserve ``existing``."""
        return existing_val if req_val is None else req_val

    def _keep_str(req_val: str | None, existing_val: str, fallback: str = "") -> str:
        """String leave-unchanged with optional empty-string fallback."""
        if req_val is None:
            return existing_val
        stripped = req_val.strip()
        if stripped:
            return stripped
        return fallback or existing_val

    execution_mode = _keep(req.research_execution_mode, existing.research_execution_mode)
    feedback_mode = _keep(req.research_action_feedback, existing.research_action_feedback)
    if execution_mode == "closed_loop" and feedback_mode != "receipts":
        raise HTTPException(status_code=422, detail="Closed-loop execution requires receipt feedback")

    config = RuntimeConfig(
        data_dir=_keep_str(req.data_dir, existing.data_dir, fallback="data"),
        max_queue_per_user=_keep(req.max_queue_per_user, existing.max_queue_per_user),
        max_active_sessions=_keep(req.max_active_sessions, existing.max_active_sessions),
        session_timeout_seconds=_keep(req.session_timeout_seconds, existing.session_timeout_seconds),
        console_enabled=_keep(req.console_enabled, existing.console_enabled),
        telegram_token=existing.telegram_token,
        telegram_allowlist=(
            existing.telegram_allowlist if req.telegram_allowlist is None
            else {item.strip() for item in req.telegram_allowlist if item.strip()}
        ),
        telegram_poll_timeout=_keep(req.telegram_poll_timeout, existing.telegram_poll_timeout),
        dedupe_ttl=_keep(req.dedupe_ttl, existing.dedupe_ttl),
        llm_backend=_keep(req.llm_backend, existing.llm_backend),
        llm_base_url=(
            existing.llm_base_url if req.llm_base_url is None
            else req.llm_base_url.strip()
        ),
        llm_model=(
            existing.llm_model if req.llm_model is None
            else req.llm_model.strip()
        ),
        llm_api_key=existing.llm_api_key,
        debug_host=_keep_str(req.debug_host, existing.debug_host, fallback="127.0.0.1"),
        debug_port=_keep(req.debug_port, existing.debug_port),
        proactive_enabled=_keep(req.proactive_enabled, existing.proactive_enabled),
        proactive_idle_threshold=_keep(req.proactive_idle_threshold, existing.proactive_idle_threshold),
        proactive_density_reference=_keep(req.proactive_density_reference, existing.proactive_density_reference),
        proactive_recovery_seconds=_keep(req.proactive_recovery_seconds, existing.proactive_recovery_seconds),
        proactive_check_interval=_keep(req.proactive_check_interval, existing.proactive_check_interval),
        character_independence=_keep(req.character_independence, existing.character_independence),
        research_mode=_keep(req.research_mode, existing.research_mode),
        research_birth_policy=_keep_str(req.research_birth_policy, existing.research_birth_policy, fallback="uncommitted"),
        research_action_framing=_keep_str(req.research_action_framing, existing.research_action_framing, fallback="agent_owned"),
        research_action_feedback=feedback_mode,
        research_execution_mode=execution_mode,
        research_context_mode=_keep_str(req.research_context_mode, existing.research_context_mode, fallback="full"),
        research_memory_mode=_keep_str(req.research_memory_mode, existing.research_memory_mode, fallback="legacy"),
        research_progress_estimates=_keep(req.research_progress_estimates, existing.research_progress_estimates),
        research_appraisal_mode=_keep_str(req.research_appraisal_mode, existing.research_appraisal_mode, fallback="legacy"),
        research_dispositions=_keep(req.research_dispositions, existing.research_dispositions),
        research_json_schema=_keep(req.research_json_schema, existing.research_json_schema),
        research_inference_retries=_keep(req.research_inference_retries, existing.research_inference_retries),
        research_max_steps_per_activation=_keep(req.research_max_steps_per_activation, existing.research_max_steps_per_activation),
        research_initial_session=_keep(req.research_initial_session, existing.research_initial_session),
        research_calls_per_day=_keep(req.research_calls_per_day, existing.research_calls_per_day),
        research_episode_idle_seconds=_keep(req.research_episode_idle_seconds, existing.research_episode_idle_seconds),
        research_episode_max_turns=_keep(req.research_episode_max_turns, existing.research_episode_max_turns),
        research_reflection_interval=_keep(req.research_reflection_interval, existing.research_reflection_interval),
        appraisal_use_llm=_keep(req.appraisal_use_llm, existing.appraisal_use_llm),
        autobiographical_learning=_keep(
            req.autobiographical_learning, existing.autobiographical_learning
        ),
        coherence_min_score=_keep(req.coherence_min_score, existing.coherence_min_score),
        coherence_max_regenerations=_keep(req.coherence_max_regenerations, existing.coherence_max_regenerations),
        pending_intake_ttl_turns=_keep(req.pending_intake_ttl_turns, existing.pending_intake_ttl_turns),
        api_key=existing.api_key,
        cors_origins=(
            existing.cors_origins if req.cors_origins is None
            else [o.strip() for o in req.cors_origins if o.strip()]
        ),
        tools_enabled=(
            existing.tools_enabled if req.tools_enabled is None
            else bool(req.tools_enabled)
        ),
        autonomy_level=(
            existing.autonomy_level
            if req.autonomy_level is None
            else _normalize_autonomy_level(req.autonomy_level)
        ),
        tools_workspace=(
            existing.tools_workspace if req.tools_workspace is None
            else req.tools_workspace.strip()
        ),
        shell_tool_enabled=(
            existing.shell_tool_enabled if req.shell_tool_enabled is None
            else bool(req.shell_tool_enabled)
        ),
        shell_command_allowlist=(
            list(existing.shell_command_allowlist)
            if req.shell_command_allowlist is None
            else [str(entry).strip() for entry in req.shell_command_allowlist if str(entry).strip()]
        ),
        mcp_tool_categories=(
            dict(existing.mcp_tool_categories)
            if req.mcp_tool_categories is None
            else _normalize_mcp_tool_categories(req.mcp_tool_categories)
        ),
        mcp_strict_categories=(
            existing.mcp_strict_categories
            if req.mcp_strict_categories is None
            else bool(req.mcp_strict_categories)
        ),
        learning_budget_kind=_keep(req.learning_budget_kind, existing.learning_budget_kind),
        learning_max_questions_per_day=_keep(req.learning_max_questions_per_day, existing.learning_max_questions_per_day),
        learning_max_seconds_per_day=_keep(req.learning_max_seconds_per_day, existing.learning_max_seconds_per_day),
        metabolism_min_elapsed_days=_keep(req.metabolism_min_elapsed_days, existing.metabolism_min_elapsed_days),
        owner_chat_id=(
            existing.owner_chat_id if req.owner_chat_id is None
            else req.owner_chat_id.strip()
        ),
    )

    if req.clear_telegram_token:
        config.telegram_token = ""
    elif req.telegram_token.strip():
        config.telegram_token = req.telegram_token.strip()

    if req.clear_llm_api_key:
        config.llm_api_key = ""
    elif req.llm_api_key.strip():
        config.llm_api_key = req.llm_api_key.strip()

    if req.clear_api_key:
        config.api_key = ""
    elif req.api_key.strip():
        config.api_key = req.api_key.strip()

    changed_fields = _changed_config_fields(existing, config)
    restart_required_fields = sorted(
        field for field in changed_fields if field in _RESTART_REQUIRED_FIELDS
    )
    config.write_yaml(RUNTIME_CONFIG_PATH)

    reload_result = await _reload_runtime_from_config(config)

    return _config_payload(
        config,
        saved=True,
        message="Configuration saved to runtime_config.yaml",
        reloaded_web_manager=bool(reload_result["reloaded_web_manager"]),
        restart_required_fields=restart_required_fields,
        runtime_reload=reload_result,
    )


@app.post("/session/end", dependencies=[Depends(_require_bearer)])
async def end_session(req: EndSessionRequest) -> dict:
    if _pipeline_override is not None:
        result = _pipeline_override.end_session(user_id=req.user_id)
        return _digested_to_dict(result)

    manager = get_session_manager()
    session = await manager.ensure_session(WEB_PLATFORM, req.user_id, req.chat_id)
    result = await session.end_session()
    return _digested_to_dict(result)


@app.post("/rest", dependencies=[Depends(_require_bearer)])
async def rest(req: RestRequest) -> dict:
    if _pipeline_override is not None:
        energy_before = _pipeline_override.engine.state.energy
        _pipeline_override.apply_rest(req.hours)
        return {
            "hours": req.hours,
            "energy_before": energy_before,
            "energy_after": _pipeline_override.engine.state.energy,
        }

    manager = get_session_manager()
    session = await manager.ensure_session(WEB_PLATFORM, req.user_id, req.chat_id)
    energy_before = session.pipeline.engine.state.energy
    await session.apply_rest(req.hours)
    return {
        "hours": req.hours,
        "energy_before": energy_before,
        "energy_after": session.pipeline.engine.state.energy,
    }


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket) -> None:
    # FastAPI does not run route dependencies for WebSocket handlers, so do
    # the bearer-token check inline. Browser WS clients cannot set custom
    # headers; if no Authorization header is present, the first received JSON
    # message must carry {"token": "..."} before any chat payload is handled.
    expected = str(_current_config_for_middleware().api_key or "")
    authed = False
    if expected:
        header_token: str | None = None
        auth_header = ws.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            header_token = auth_header.split(" ", 1)[1].strip()
        authed = bool(header_token and secrets.compare_digest(header_token, expected))
        if header_token and not authed:
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    await ws.accept()

    def _parse_ws_frame(raw_frame: str) -> dict | None:
        """Decode one JSON frame; ``None`` means malformed (not an object)."""
        try:
            parsed = json.loads(raw_frame)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    try:
        pending_msg: dict | None = None
        if expected and not authed:
            raw = await ws.receive_text()
            if len(raw) > _MAX_WS_MESSAGE_CHARS:
                await ws.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                return
            pending_msg = _parse_ws_frame(raw)
            if pending_msg is None:
                await ws.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            supplied = str(pending_msg.pop("token", "") or "")
            auth_value = str(pending_msg.pop("authorization", "") or "")
            if not supplied and auth_value.lower().startswith("bearer "):
                supplied = auth_value.split(" ", 1)[1].strip()
            if not secrets.compare_digest(supplied, expected):
                await ws.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            if not pending_msg.get("message"):
                await ws.send_text(json.dumps({"type": "auth", "ok": True}))
                pending_msg = None

        while True:
            if pending_msg is not None:
                msg = pending_msg
                pending_msg = None
            else:
                raw = await ws.receive_text()
                if len(raw) > _MAX_WS_MESSAGE_CHARS:
                    await ws.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                    return
                parsed = _parse_ws_frame(raw)
                if parsed is None:
                    await ws.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
                msg = parsed
            user_id = msg.get("user_id", "default")
            chat_id = msg.get("chat_id", "default")
            user_message = msg.get("message", "")
            if len(str(user_id)) > _MAX_ID_CHARS or len(str(chat_id)) > _MAX_ID_CHARS:
                await ws.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            if not user_message or len(str(user_message)) > _MAX_CHAT_MESSAGE_CHARS:
                await ws.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                return

            try:
                if _pipeline_override is not None:
                    result = _pipeline_override.process(user_message, user_id=user_id)
                    payload = {
                        "response": result.response,
                        "debug": _serialize_debug(result.debug),
                    }
                else:
                    manager = get_session_manager()
                    response = await manager.handle_message(
                        WEB_PLATFORM,
                        user_id,
                        chat_id,
                        user_message,
                    )
                    session = await manager.ensure_session(WEB_PLATFORM, user_id, chat_id)
                    payload = {
                        "response": response,
                        "debug": (
                            _serialize_debug(session.last_debug)
                            if session.last_debug
                            else {}
                        ),
                    }
            except Exception as exc:
                backend_error = llm_error_payload(exc, _load_runtime_config())
                if backend_error is None:
                    raise
                payload = {"type": "error", **backend_error}

            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass


@app.get("/")
def index() -> HTMLResponse:
    return _index_response()


@app.get("/settings")
def settings_index() -> HTMLResponse:
    return _admin_response()


@app.get("/admin")
def admin_index() -> RedirectResponse:
    return RedirectResponse(url="/settings", status_code=307)


@app.get("/persona")
def persona_index() -> RedirectResponse:
    return RedirectResponse(url="/settings#observability", status_code=307)


@app.get("/dashboard")
def dashboard_index() -> RedirectResponse:
    return RedirectResponse(url="/settings#observability", status_code=307)


@app.get("/admin/assets/{asset_path:path}", include_in_schema=False)
def admin_asset(asset_path: str):
    media_type = _static_media_type(asset_path)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return _static_asset_response(asset_path, media_type)


@app.get("/assets/{asset_path:path}", include_in_schema=False)
def brand_asset(asset_path: str):
    from fastapi.responses import Response

    media_type = _static_media_type(asset_path)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Brand asset not found")
    static_dirs = [os.path.join(os.path.dirname(__file__), "static")]
    if asset_path.endswith(".svg"):
        static_dirs.append(os.path.join(os.path.dirname(__file__), "..", "docs", "diagrams"))
    for static_dir in static_dirs:
        static_root = os.path.realpath(static_dir)
        path = os.path.realpath(os.path.join(static_dir, asset_path))
        if path.startswith(static_root + os.sep) and os.path.isfile(path):
            with open(path, "rb") as f:
                return Response(
                    content=f.read(),
                    media_type=media_type,
                    headers=_HTML_SECURITY_HEADERS,
                )
    raise HTTPException(status_code=404, detail="Brand asset not found")


def _index_response() -> HTMLResponse:
    return _html_static_response("index.html")


def _admin_response() -> HTMLResponse:
    return _html_static_response("admin.html")


def _persona_response() -> HTMLResponse:
    return _html_static_response("persona.html")


def _html_static_response(filename: str) -> HTMLResponse:
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    path = os.path.join(static_dir, filename)
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers=_HTML_SECURITY_HEADERS)


def _static_media_type(asset_path: str) -> str | None:
    if asset_path.endswith(".css"):
        return "text/css"
    if asset_path.endswith(".js"):
        return "application/javascript"
    if asset_path.endswith(".svg"):
        return "image/svg+xml"
    return None


def _static_asset_response(filename: str, media_type: str):
    from fastapi.responses import Response

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    path = os.path.normpath(os.path.join(static_dir, filename))
    static_root = os.path.abspath(static_dir)
    if not (os.path.abspath(path) == static_root or os.path.abspath(path).startswith(static_root + os.sep)):
        raise HTTPException(status_code=404, detail="Static asset not found")
    with open(path, "rb") as f:
        return Response(
            content=f.read(),
            media_type=media_type,
            headers=_HTML_SECURITY_HEADERS,
        )


def _optional_life_llm_backend(config: RuntimeConfig):
    """Return an LLM backend for richer digestion, or None for heuristics."""
    try:
        return create_llm_backend(config)
    except Exception:
        return None


def _upload_filename(upload: UploadFile, *, fallback: str) -> str:
    raw = (upload.filename or "").strip().replace("\\", "/")
    name = Path(raw).name.strip()
    return name[:180] or fallback


async def _read_upload_bytes(upload: UploadFile, *, max_bytes: int, label: str) -> bytes:
    payload = await upload.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{label} is too large ({max_bytes} byte limit).",
        )
    if not payload:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    return payload


def _session_key(user_id: str, chat_id: str) -> str:
    return f"{WEB_PLATFORM}:{user_id}:{chat_id}"


def _session_debug(session_key_value: str, session: UserSession) -> dict:
    pipeline = session.pipeline
    unresolved = pipeline.engine.active_unresolved()
    return {
        "session_key": session_key_value,
        "user_id": session.user_id,
        "modulator_snapshot": pipeline.engine.snapshot(),
        "emotion_label": pipeline.engine.to_emotion_label(),
        "energy": pipeline.engine.state.energy,
        "short_term_count": len(pipeline.short_term),
        "long_term_count": pipeline.long_term.count(),
        "unresolved_count": len(unresolved),
        "unresolved_items": [item.to_dict() for item in unresolved],
        "last_turn": _serialize_debug(session.last_debug) if session.last_debug else None,
    }


def _digested_to_dict(result) -> dict:
    return {
        "summary": result.summary,
        "emotional_arc_label": result.emotional_arc_label,
        "trust_delta": result.trust_delta,
        "memories_written": result.memories_written,
        "energy_drain": result.energy_drain,
        "unresolved_flags": result.unresolved_flags,
    }


def _load_runtime_config() -> RuntimeConfig:
    """Load the runtime config file used by both web and runtime surfaces."""
    return RuntimeConfig.from_yaml(RUNTIME_CONFIG_PATH)


def _changed_config_fields(before: RuntimeConfig, after: RuntimeConfig) -> list[str]:
    """Return dataclass fields whose persisted values changed."""
    before_data = before.to_yaml_dict()
    after_data = after.to_yaml_dict()
    return sorted(
        field_name
        for field_name in after_data
        if before_data.get(field_name) != after_data.get(field_name)
    )


async def _reload_runtime_from_config(config: RuntimeConfig) -> dict:
    """Apply saved config to reloadable in-process runtime surfaces."""
    global _session_manager, _pipeline_override

    active_sessions_evicted = 0
    reloaded_web_manager = False

    if _pipeline_override is not None:
        _pipeline_override.close()
        _pipeline_override = None
        reloaded_web_manager = True

    if _session_manager is not None:
        active_sessions_evicted = len(_session_manager.active_sessions)
        await _session_manager.shutdown()
        _session_manager = None
        reloaded_web_manager = True

    await _restart_telegram_channel(config)
    return {
        "reloaded_web_manager": reloaded_web_manager,
        "active_sessions_evicted": active_sessions_evicted,
        "telegram_restarted": True,
    }


def _detach_session_manager_for_reload(*, reason: str) -> tuple[bool, int]:
    """Detach the shared web manager immediately and close it in the background."""
    global _session_manager

    manager = _session_manager
    if manager is None:
        return False, 0
    active_sessions = len(manager.active_sessions)
    _session_manager = None
    task = asyncio.create_task(
        _shutdown_detached_session_manager(manager, reason=reason),
        name=f"session-manager-shutdown:{reason}",
    )
    _background_shutdown_tasks.add(task)
    task.add_done_callback(_background_shutdown_tasks.discard)
    return True, active_sessions


def _refresh_live_channel_session_manager() -> None:
    """Point live channels at the fresh manager after detaching the old one."""
    if _telegram_channel is None:
        return
    setter = getattr(_telegram_channel, "set_session_manager", None)
    if callable(setter):
        setter(get_session_manager())


async def _shutdown_detached_session_manager(
    manager: SessionManager,
    *,
    reason: str,
) -> None:
    try:
        await manager.shutdown()
    except Exception:
        _log.exception("Detached SessionManager shutdown failed after %s", reason)


async def _drain_runtime_before_process_restart() -> dict:
    """Persist hot state and stop background runtime work before exec."""
    global _session_manager, _pipeline_override

    active_sessions_evicted = 0
    closed_pipeline_override = False

    if _pipeline_override is not None:
        _pipeline_override.close()
        _pipeline_override = None
        closed_pipeline_override = True

    if _session_manager is not None:
        active_sessions_evicted = len(_session_manager.active_sessions)
        await _session_manager.shutdown()
        _session_manager = None

    await _stop_telegram_channel()
    return {
        "closed_pipeline_override": closed_pipeline_override,
        "active_sessions_evicted": active_sessions_evicted,
        "telegram_stopped": True,
    }


def _restart_argv() -> list[str]:
    """Best-effort command line for re-execing the current Python process."""
    original = getattr(sys, "orig_argv", None)
    if original and len(original) > 1:
        return [sys.executable, *list(original[1:])]
    return [sys.executable, *sys.argv]


def _schedule_process_restart(delay_seconds: float = 0.35) -> list[str]:
    """Schedule an in-place process restart after the response can flush."""
    argv = _restart_argv()

    async def _restart() -> None:
        await asyncio.sleep(delay_seconds)
        os.execv(sys.executable, argv)

    asyncio.create_task(_restart())
    return argv


def _config_payload(
    config: RuntimeConfig,
    *,
    saved: bool,
    message: str | None = None,
    reloaded_web_manager: bool = False,
    restart_required_fields: list[str] | None = None,
    runtime_reload: dict | None = None,
) -> dict:
    """Serialize runtime config for the settings UI."""
    public = config.to_public_dict()
    state = _read_admin_state(config)
    public["setup_completed"] = bool(state.get("setup_completed"))
    restart_required_fields = sorted(restart_required_fields or [])
    runtime_reload = runtime_reload or {}
    return {
        "config": public,
        "secret_status": config.secret_status(),
        "config_path": os.path.abspath(RUNTIME_CONFIG_PATH),
        "saved": saved,
        "message": message,
        "reloaded_web_manager": reloaded_web_manager,
        "runtime_reload": runtime_reload,
        "apply_state": {
            "saved_to_disk": saved,
            "session_manager_reloaded": reloaded_web_manager,
            "telegram_restarted": bool(runtime_reload.get("telegram_restarted", False)),
            "active_sessions_evicted": int(runtime_reload.get("active_sessions_evicted", 0)),
            "restart_required": bool(restart_required_fields),
            "restart_required_fields": restart_required_fields,
        },
        "notes": [
            "Secret fields are never returned; leave them blank to keep the current value.",
            "Saving through this UI updates runtime_config.yaml.",
            "The standalone web server reloads its web manager and Telegram poller after save.",
            "debug_host, debug_port, and cors_origins require restarting the web server process.",
            "Changes for nur / python3 main.py still apply after restarting that runtime.",
        ],
    }


_RESTART_REQUIRED_FIELDS = {"debug_host", "debug_port", "cors_origins"}


def _normalize_autonomy_level(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in {"off", "assisted", "autonomous", "high_risk"}:
        raise HTTPException(
            status_code=400,
            detail="autonomy_level must be one of: off, assisted, autonomous, high_risk",
        )
    return normalized


def _normalize_mcp_tool_categories(value: dict[str, str]) -> dict[str, str]:
    from runtime.config import normalize_mcp_tool_categories

    try:
        return normalize_mcp_tool_categories(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _admin_state_path(config: RuntimeConfig) -> str:
    return os.path.join(config.data_dir, "admin_state.json")


def _read_admin_state(config: RuntimeConfig) -> dict:
    path = _admin_state_path(config)
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


_DEFAULT_SOUL_NAME = "OpenRook"


def _soul_yaml_path() -> str:
    """Path where /admin/soul writes soul.yaml.

    If OPENROOK_CONFIG_DIR is set, write there (persists across pip upgrades).
    Otherwise fall back to the packaged config dir so the loader still
    finds it without any env setup — at the cost of being overwritten by
    ``pip install --upgrade``. The admin UI surfaces that trade-off on
    save.
    """
    from config import loader as _loader  # imported lazily to avoid cycles

    override = project_env("CONFIG_DIR", "").strip()
    if override:
        path = os.path.expanduser(override)
        os.makedirs(path, exist_ok=True)
        return str(os.path.join(path, "soul.yaml"))
    return str(os.path.join(os.path.dirname(os.path.abspath(_loader.__file__)), "soul.yaml"))


def _soul_looks_default() -> bool:
    """True when soul.name is still the built-in placeholder."""
    try:
        from config.loader import get_config

        return get_config().soul.name.strip() == _DEFAULT_SOUL_NAME
    except Exception:  # pragma: no cover - defensive
        return False


def _admin_backup_dir(config: RuntimeConfig) -> str:
    return os.path.join(config.data_dir, "backups")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact_error(message: str, config: RuntimeConfig) -> str:
    return redact_error(message, config)


def _web_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("openrook")
    except PackageNotFoundError:
        return "unknown"


def _parse_web_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="openrook-web",
        description=(
            "Serve the OpenRook HTTP API and bundled chat/admin UI via Uvicorn. "
            "Reads runtime_config.yaml from the current working directory. "
            "Defaults bind to localhost; see "
            "https://github.com/balfiky/openrook/blob/main/docs/DEPLOYMENT_AND_ADMIN.md "
            "before exposing beyond the local machine."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to bind (default: 8000).",
    )
    parser.add_argument(
        "--config",
        default=project_env("RUNTIME_CONFIG", "runtime_config.yaml"),
        help="Runtime config YAML path (default: runtime_config.yaml).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openrook {_web_version()}",
    )
    return parser.parse_args(argv)


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _is_loopback_host(host: str) -> bool:
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


def _bootstrap_runtime_config(config_path: str) -> RuntimeConfig:
    """Create runtime_config.yaml + data dir with safe defaults if missing,
    then return the loaded config. Idempotent — does nothing if the file
    already exists.
    """
    from interface.setup import _initialize_workspace

    path = Path(config_path).expanduser()
    _initialize_workspace(path, "data", force=False)
    return RuntimeConfig.from_yaml(str(path.resolve()))


def main() -> None:
    """Launch the standalone web UI on the configured host/port.

    First-run friendly: if ``runtime_config.yaml`` is missing, ``openrook-web``
    creates one with safe defaults (no LLM configured yet, tools off,
    shell off, no Telegram) and binds. The browser setup wizard at ``/``
    walks the operator through model, identity, and tool settings.

    Authentication is **opt-in**. If the operator sets ``api_key`` in
    ``runtime_config.yaml``, protected endpoints (``/chat``, ``/admin/*``,
    ``/v1/*``, ``/ws``, ...) require ``Authorization: Bearer <api_key>``.
    Otherwise they are open — matching the convention of other local-LLM
    tools (Ollama, LM Studio, Jupyter). When binding to a non-loopback
    host without ``api_key`` set, ``openrook-web`` prints a one-screen
    warning to stderr so the operator knows the surface is reachable
    from the network.
    """
    import uvicorn

    args = _parse_web_args()
    global RUNTIME_CONFIG_PATH
    RUNTIME_CONFIG_PATH = args.config
    os.environ["OPENROOK_RUNTIME_CONFIG"] = args.config

    try:
        cfg = _bootstrap_runtime_config(args.config)
    except Exception as exc:
        print(
            f"failed to read or create runtime config at {args.config!r}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    # The CORS middleware captured ``cors_origins`` at module import, before
    # ``--config`` was parsed — so origins from a non-CWD config file were
    # silently ignored (caught by live UAT). Re-apply them here; the
    # middleware stack is not built until the app starts serving.
    for _mw in app.user_middleware:
        if _mw.cls is CORSMiddleware:
            _mw.kwargs["allow_origins"] = list(cfg.cors_origins)

    if not _is_loopback_host(args.host) and not (cfg.api_key or "").strip():
        print(
            "",
            "=" * 72,
            f"WARNING: OpenRook is binding to {args.host!r} with no api_key set.",
            "",
            "Anyone who can reach this host on the network can talk to OpenRook,",
            "read your memory, change config, and (if tools are on) run shell",
            "commands. If that is not what you want, stop the server, set",
            f"`api_key: <a-strong-token>` in {args.config!r}, and restart.",
            "=" * 72,
            "",
            sep="\n",
            file=sys.stderr,
        )

    uvicorn.run("interface.api:app", host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Legacy/admin route modules.
#
# Imported at the BOTTOM so they can reference this module's helpers,
# models, and mutable globals. Their handlers resolve patchable helpers
# late through the ``interface.api`` module object, which keeps test
# monkeypatching (e.g. ``monkeypatch.setattr(interface_api, ...)``)
# working after the split. The handlers and request models are re-exported
# here so ``interface.api`` remains the public assembly module.
# ---------------------------------------------------------------------------

from interface import routes_admin as _routes_admin  # noqa: E402
from interface import routes_backup as _routes_backup  # noqa: E402
from interface import routes_life as _routes_life  # noqa: E402
from interface import routes_skills as _routes_skills  # noqa: E402

from interface.routes_admin import (  # noqa: E402,F401
    AdminConfigUpdateRequest,
    AdminLLMModelsRequest,
    AdminLLMTestRequest,
    AdminSessionResetRequest,
    AdminStorageTestRequest,
    AdminTelegramTestRequest,
    AdminUserDeleteRequest,
    admin_codex_models,
    admin_delete_user,
    admin_diagnostics,
    admin_draft_soul,
    admin_get_config,
    admin_get_soul,
    admin_import_soul,
    admin_llm_models,
    admin_persona_state,
    admin_reload_runtime,
    admin_reset_session,
    admin_restart_runtime,
    admin_status,
    admin_test_llm,
    admin_test_storage,
    admin_test_telegram,
    admin_update_config,
    admin_update_soul,
)
from interface.routes_backup import (  # noqa: E402,F401
    AdminBackupDeleteRequest,
    AdminBackupRequest,
    admin_create_backup,
    admin_delete_backup,
    admin_export_config,
    admin_list_backups,
)
from interface.routes_life import (  # noqa: E402,F401
    AdminConstitutionRequest,
    AdminLifeFileRequest,
    AdminLifeRollbackRequest,
    AdminLifeTextRequest,
    admin_get_constitution,
    admin_life_abandon_open_question,
    admin_life_beliefs,
    admin_life_drives,
    admin_life_evolution,
    admin_life_experiences,
    admin_life_ingest_file,
    admin_life_ingest_text,
    admin_life_ingest_upload,
    admin_life_metabolism_tick,
    admin_life_open_questions,
    admin_life_overview,
    admin_life_resolve_open_question,
    admin_life_rollback,
    admin_set_constitution,
)
from interface.routes_skills import (  # noqa: E402,F401
    AdminSkillImportRequest,
    admin_audit_skill,
    admin_delete_skill,
    admin_disable_skill,
    admin_enable_skill,
    admin_get_skill,
    admin_import_skill,
    admin_import_skill_upload,
    admin_list_skills,
)

app.include_router(_routes_admin.router)
app.include_router(_routes_backup.router)
app.include_router(_routes_skills.router)
app.include_router(_routes_life.router)

from interface import routes_research as _routes_research  # noqa: E402

app.include_router(_routes_research.router)


if __name__ == "__main__":
    main()

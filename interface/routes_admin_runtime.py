"""Admin console runtime endpoints: persona state, diagnostics, reload/restart,
session reset, user deletion.

Split out of ``interface/routes_admin.py``, which aggregates this router and
re-exports the handlers. Handlers resolve shared helpers and mutable globals
late through the ``interface.api`` module object (``api_mod``) so test
monkeypatching on ``interface.api`` keeps working. Pure config-introspection
helpers are imported directly from ``routes_admin_config``.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from interface import api as api_mod
from interface.routes_admin_config import (
    _admin_config_warnings,
    _check_writable_directory,
    _llm_configured,
    _warning,
)
from interface.v1 import _count_user_rows, _validate_path_token
from runtime.config import RuntimeConfig
from runtime.debug.persona_view import build_persona_view
from runtime.sessions.manager import SessionManager

router = APIRouter(dependencies=[Depends(api_mod._require_bearer)])


class AdminSessionResetRequest(BaseModel):
    session_key: str
    confirmation: str


class AdminUserDeleteRequest(BaseModel):
    platform: str
    user_id: str
    confirmation: str


@router.get("/admin/persona/state")
async def admin_persona_state() -> dict:
    """Unified runtime persona dashboard state across all active channels.

    This is read-only observability over the shared SessionManager. It does
    not create sessions, call the pipeline, or mutate memory/emotional state.
    """
    return _admin_persona_payload(api_mod.get_session_manager())


@router.get("/admin/diagnostics")
async def admin_diagnostics() -> dict:
    """Operational diagnostics for the admin maintenance panel."""
    config = api_mod._load_runtime_config()
    manager = api_mod.get_session_manager()
    return _admin_diagnostics_payload(config, manager)


@router.post("/admin/runtime/reload")
async def admin_reload_runtime() -> dict:
    """Apply the saved runtime config without a full process restart.

    This evicts active web sessions so the next turn rebuilds pipelines from
    ``runtime_config.yaml`` and restarts the standalone Telegram poller.
    Host/port/CORS still require a web-server process restart.
    """
    config = api_mod._load_runtime_config()
    result = await api_mod._reload_runtime_from_config(config)
    return {
        "ok": True,
        "action": "runtime_reload",
        "message": "Saved runtime config applied to web sessions and Telegram polling.",
        **result,
        "restart_required_fields": sorted(api_mod._RESTART_REQUIRED_FIELDS),
        "restart_note": (
            "debug_host, debug_port, and cors_origins are bound by the web "
            "server process and require Restart Web Server."
        ),
    }


@router.post("/admin/runtime/restart")
async def admin_restart_runtime(req: api_mod.AdminRuntimeRestartRequest) -> dict:
    """Schedule a local process restart after the HTTP response is sent."""
    if req.confirmation.strip() != "RESTART":
        raise HTTPException(
            status_code=400,
            detail="Confirmation must exactly match: RESTART",
        )
    drain_result = await api_mod._drain_runtime_before_process_restart()
    argv = api_mod._schedule_process_restart()
    return {
        "ok": True,
        "action": "process_restart",
        "scheduled": True,
        "message": "Web server restart scheduled.",
        "runtime_drain": drain_result,
        "argv": argv,
    }


@router.post("/admin/sessions/reset")
async def admin_reset_session(req: AdminSessionResetRequest) -> dict:
    """Evict one active session after explicit typed confirmation."""
    expected = f"RESET {req.session_key}"
    if req.confirmation != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Confirmation must exactly match: {expected}",
        )
    manager = api_mod.get_session_manager()
    if req.session_key not in manager.active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    await manager.evict_session(req.session_key)
    return {
        "ok": True,
        "action": "session_reset",
        "session_key": req.session_key,
        "status": "evicted",
    }


@router.post("/admin/users/delete")
async def admin_delete_user(req: AdminUserDeleteRequest) -> dict:
    """Delete one user's persisted data after explicit typed confirmation."""
    rel_key = f"{req.platform}:{req.user_id}"
    expected = f"DELETE {rel_key}"
    if req.confirmation != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Confirmation must exactly match: {expected}",
        )
    _validate_path_token("platform", req.platform)
    _validate_path_token("user_id", req.user_id)
    return await _admin_delete_user_data(
        api_mod._load_runtime_config(),
        api_mod.get_session_manager(),
        req.platform,
        req.user_id,
    )


# ---------------------------------------------------------------------------
# Persona, diagnostics, and user-deletion helpers
# ---------------------------------------------------------------------------


def _admin_persona_payload(manager: SessionManager) -> dict:
    """Build the standalone persona dashboard payload from active sessions."""
    now = time.time()
    sessions = []
    channel_counts: dict[str, int] = {}
    for session_key, session in sorted(manager.active_sessions.items()):
        platform_name, user_id, chat_id = _split_session_key(session_key)
        channel_counts[platform_name] = channel_counts.get(platform_name, 0) + 1
        sessions.append({
            "session_key": session_key,
            "platform": platform_name,
            "user_id": user_id or session.user_id,
            "chat_id": chat_id,
            "rel_key": session.rel_key,
            "last_activity": session.last_activity,
            "idle_seconds": round(now - session.last_activity, 1),
            "has_last_turn": session.last_debug is not None,
            "persona_view": build_persona_view(session=session, session_key=session_key),
        })
    return {
        "generated_at": now,
        "count": len(sessions),
        "channel_counts": channel_counts,
        "sessions": sessions,
    }


def _split_session_key(session_key: str) -> tuple[str, str, str]:
    parts = str(session_key or "").split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return str(session_key or ""), "", ""


def _admin_diagnostics_payload(
    config: RuntimeConfig,
    manager: SessionManager,
) -> dict:
    data_summary = _directory_summary("data_dir", config.data_dir)
    workspace_summary = _directory_summary(
        "tools_workspace",
        config.resolved_tools_workspace,
    )
    return {
        "status": "ok",
        "generated_at": api_mod._utc_now_iso(),
        "server": {
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "uptime_seconds": max(0.0, time.time() - api_mod._serve_started_at),
        },
        "config": {
            "path": os.path.abspath(api_mod.RUNTIME_CONFIG_PATH),
            "exists": os.path.exists(api_mod.RUNTIME_CONFIG_PATH),
            "auth_enabled": bool(config.api_key),
            "llm_backend": config.llm_backend,
            "llm_configured": _llm_configured(config),
            "telegram_configured": bool(config.telegram_token),
        },
        "runtime": {
            "active_sessions": len(manager.active_sessions),
            "max_active_sessions": config.max_active_sessions,
            "telegram_running": (
                api_mod._telegram_task is not None
                and not api_mod._telegram_task.done()
            ),
            "pipeline_override": api_mod._pipeline_override is not None,
        },
        "storage": {
            "data_dir": data_summary,
            "tools_workspace": workspace_summary,
            "backup_dir": os.path.realpath(api_mod._admin_backup_dir(config)),
        },
        "warnings": _admin_config_warnings(config),
    }


def _directory_summary(name: str, path: str) -> dict:
    check = _check_writable_directory(name, path, create_missing=False)
    summary = {
        "path": check["path"],
        "exists": check["exists"],
        "writable": check["writable"],
        "warnings": check["warnings"],
        "file_count": 0,
        "truncated": False,
    }
    if not check["exists"]:
        return summary

    max_files = 2000
    try:
        count = 0
        for _, _, files in os.walk(check["path"]):
            count += len(files)
            if count > max_files:
                summary["file_count"] = max_files
                summary["truncated"] = True
                return summary
        summary["file_count"] = count
    except OSError as exc:
        summary["warnings"] = [
            *summary["warnings"],
            _warning("warning", name, f"{name}_count_error", str(exc)),
        ]
    return summary


async def _admin_delete_user_data(
    config: RuntimeConfig,
    manager: SessionManager,
    platform_name: str,
    user_id: str,
) -> dict:
    rel_key = f"{platform_name}:{user_id}"
    user_dir = config.user_data_dir(rel_key)
    data_root = os.path.realpath(config.data_dir)
    resolved = os.path.realpath(user_dir)
    if not (resolved == data_root or resolved.startswith(data_root + os.sep)):
        raise HTTPException(
            status_code=400,
            detail="Resolved user path is not inside data_dir",
        )

    from runtime.storage import existing_user_db_path

    rows_deleted = _count_user_rows(existing_user_db_path(user_dir), user_id)
    prefix = f"{rel_key}:"
    evict_keys = [
        key for key in list(manager.active_sessions.keys())
        if key.startswith(prefix)
    ]
    for key in evict_keys:
        await manager.evict_session(key)

    session_files_removed = 0
    sessions_dir = os.path.join(user_dir, "sessions")
    if os.path.isdir(sessions_dir):
        session_files_removed = sum(
            1 for name in os.listdir(sessions_dir)
            if name.endswith(".json")
        )

    existed = os.path.isdir(user_dir)
    had_rows = any(isinstance(value, int) and value > 0 for value in rows_deleted.values())
    if not existed and not had_rows and not evict_keys:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {rel_key}",
        )
    if existed:
        shutil.rmtree(user_dir, ignore_errors=False)

    return {
        "ok": True,
        "action": "user_delete",
        "deleted": True,
        "rel_key": rel_key,
        "platform": platform_name,
        "user_id": user_id,
        "rows_deleted": rows_deleted,
        "session_files_removed": session_files_removed,
        "sessions_evicted": evict_keys,
        "path_removed": user_dir if existed else None,
        "shared_self_model_db_preserved": True,
    }

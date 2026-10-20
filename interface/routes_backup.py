"""Admin backup and config-export endpoints.

Split out of ``interface/api.py``, which mounts this router at import time
and re-exports the handlers. Handlers resolve shared helpers late through
the ``interface.api`` module object (``api_mod``) so test monkeypatching
on ``interface.api`` keeps working.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from interface import api as api_mod
from interface import routes_admin as admin_mod
from runtime.config import RuntimeConfig

router = APIRouter(dependencies=[Depends(api_mod._require_bearer)])


class AdminBackupRequest(BaseModel):
    include_data: bool = True


class AdminBackupDeleteRequest(BaseModel):
    filename: str
    confirmation: str


@router.get("/admin/export/config")
async def admin_export_config() -> dict:
    """Return a redacted runtime configuration export."""
    config = api_mod._load_runtime_config()
    return _admin_redacted_config_export(config)


@router.post("/admin/backup")
async def admin_create_backup(req: AdminBackupRequest) -> dict:
    """Create a local zip backup under the configured data directory."""
    config = api_mod._load_runtime_config()
    return _create_admin_backup(config, include_data=req.include_data)


@router.get("/admin/backups")
async def admin_list_backups() -> dict:
    """List local admin backup archives."""
    config = api_mod._load_runtime_config()
    return _list_admin_backups(config)


@router.post("/admin/backups/delete")
async def admin_delete_backup(req: AdminBackupDeleteRequest) -> dict:
    """Delete one local admin backup archive after explicit confirmation."""
    config = api_mod._load_runtime_config()
    return _delete_admin_backup(
        config,
        filename=req.filename,
        confirmation=req.confirmation,
    )


def _admin_redacted_config_export(config: RuntimeConfig) -> dict:
    return {
        "format": "openrook.runtime_config.redacted.v1",
        "exported_at": api_mod._utc_now_iso(),
        "config_path": os.path.abspath(api_mod.RUNTIME_CONFIG_PATH),
        "config": config.to_public_dict(),
        "secret_status": admin_mod._admin_secret_status(config),
        "warnings": admin_mod._admin_config_warnings(config),
    }


def _create_admin_backup(config: RuntimeConfig, *, include_data: bool) -> dict:
    backup_dir = api_mod._admin_backup_dir(config)
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = os.path.join(backup_dir, f"openrook-backup-{timestamp}.zip")

    file_count = 0
    data_root = os.path.realpath(config.data_dir)
    backup_root = os.path.realpath(backup_dir)
    archive_realpath = os.path.realpath(archive_path)
    export_payload = _admin_redacted_config_export(config)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "runtime_config.redacted.json",
            json.dumps(export_payload, indent=2, sort_keys=True),
        )
        file_count += 1

        if include_data and os.path.isdir(data_root):
            for root, dirs, files in os.walk(data_root):
                dirs[:] = [
                    dirname for dirname in dirs
                    if not _is_relative_to(
                        os.path.realpath(os.path.join(root, dirname)),
                        backup_root,
                    )
                ]
                for filename in files:
                    file_path = os.path.realpath(os.path.join(root, filename))
                    if (
                        file_path == archive_realpath
                        or _is_relative_to(file_path, backup_root)
                    ):
                        continue
                    arcname = os.path.join("data", os.path.relpath(file_path, data_root))
                    archive.write(file_path, arcname)
                    file_count += 1

    return {
        "ok": True,
        "created_at": api_mod._utc_now_iso(),
        "path": os.path.realpath(archive_path),
        "filename": os.path.basename(archive_path),
        "size_bytes": os.path.getsize(archive_path),
        "file_count": file_count,
        "included_data": include_data,
        "redacted_config": True,
    }


def _list_admin_backups(config: RuntimeConfig) -> dict:
    backup_dir = api_mod._admin_backup_dir(config)
    backups: list[dict] = []
    if os.path.isdir(backup_dir):
        for name in sorted(os.listdir(backup_dir), reverse=True):
            if not name.startswith("openrook-backup-") or not name.endswith(".zip"):
                continue
            path = _resolve_admin_backup_path(config, name, must_exist=False)
            if not os.path.isfile(path):
                continue
            stat = os.stat(path)
            backups.append({
                "filename": name,
                "path": os.path.realpath(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat().replace("+00:00", "Z"),
            })
    return {
        "ok": True,
        "backup_dir": os.path.realpath(backup_dir),
        "count": len(backups),
        "backups": backups,
    }


def _delete_admin_backup(
    config: RuntimeConfig,
    *,
    filename: str,
    confirmation: str,
) -> dict:
    filename = str(filename or "").strip()
    expected = f"DELETE {filename}"
    if confirmation != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Confirmation must exactly match: {expected}",
        )
    path = _resolve_admin_backup_path(config, filename, must_exist=True)
    size = os.path.getsize(path)
    os.remove(path)
    return {
        "ok": True,
        "action": "backup_delete",
        "filename": filename,
        "path_removed": path,
        "size_bytes": size,
    }


def _resolve_admin_backup_path(
    config: RuntimeConfig,
    filename: str,
    *,
    must_exist: bool,
) -> str:
    if (
        not filename
        or os.sep in filename
        or (os.altsep and os.altsep in filename)
        or not filename.startswith("openrook-backup-")
        or not filename.endswith(".zip")
    ):
        raise HTTPException(status_code=400, detail="Invalid backup filename.")
    backup_dir = os.path.realpath(api_mod._admin_backup_dir(config))
    path = os.path.realpath(os.path.join(backup_dir, filename))
    if not _is_relative_to(path, backup_dir):
        raise HTTPException(status_code=400, detail="Invalid backup path.")
    if must_exist and not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Backup not found.")
    return path


def _is_relative_to(path: str, parent: str) -> bool:
    try:
        common = os.path.commonpath([os.path.realpath(path), os.path.realpath(parent)])
    except ValueError:
        return False
    return common == os.path.realpath(parent)

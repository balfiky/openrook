"""Admin skills endpoints (``/admin/skills/*``).

Split out of ``interface/api.py``, which mounts this router at import time
and re-exports the handlers. Handlers resolve shared helpers late through
the ``interface.api`` module object (``api_mod``) so test monkeypatching
on ``interface.api`` keeps working.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from interface import api as api_mod

router = APIRouter(dependencies=[Depends(api_mod._require_bearer)])

_MAX_SKILL_MARKDOWN_UPLOAD_BYTES = 1_000_000
_MAX_SKILL_ARCHIVE_UPLOAD_BYTES = 25_000_000
_MAX_SKILL_ARCHIVE_FILES = 250
_SKILL_MARKDOWN_SUFFIXES = {".md", ".markdown"}


class AdminSkillImportRequest(BaseModel):
    source_path: str | None = Field(default=None, max_length=4096)
    skill_markdown: str | None = Field(default=None, max_length=250000)
    name_hint: str = Field(default="", max_length=128)


@router.get("/admin/skills")
async def admin_list_skills() -> dict:
    """List imported skills and their latest compatibility report."""
    from runtime.skills import SkillError, list_skills

    try:
        return list_skills(api_mod._load_runtime_config())
    except SkillError as exc:
        _raise_skill_http_error(exc)


@router.post("/admin/skills/import")
async def admin_import_skill(req: AdminSkillImportRequest) -> dict:
    """Import a skill folder or pasted SKILL.md for review.

    Imported skills are disabled until explicitly enabled. This endpoint does
    not inject any skill into generation prompts.
    """
    from runtime.skills import SkillError, import_skill

    try:
        if req.skill_markdown:
            _reject_plain_markdown_skill_import(req.skill_markdown)
        skill = import_skill(
            api_mod._load_runtime_config(),
            source_path=req.source_path,
            skill_markdown=req.skill_markdown,
            name_hint=req.name_hint,
        )
    except SkillError as exc:
        _raise_skill_http_error(exc)
    return {"ok": True, "skill": skill}


@router.post("/admin/skills/import/upload")
async def admin_import_skill_upload(
    file: UploadFile = File(...),
    name_hint: str = Form(""),
) -> dict:
    """Import an uploaded SKILL.md/Markdown file or zipped skill folder."""
    from runtime.skills import SkillError, import_skill

    filename = api_mod._upload_filename(file, fallback="skill-upload")
    suffix = Path(filename).suffix.lower()
    config = api_mod._load_runtime_config()
    try:
        if suffix == ".zip":
            payload = await api_mod._read_upload_bytes(
                file,
                max_bytes=_MAX_SKILL_ARCHIVE_UPLOAD_BYTES,
                label="Skill archive",
            )
            with tempfile.TemporaryDirectory(prefix="openrook-skill-upload-") as tmp:
                extract_root = Path(tmp) / "skill"
                extract_root.mkdir(parents=True, exist_ok=True)
                _safe_extract_zip(payload, extract_root)
                source_root = _single_skill_root_from_upload(extract_root)
                skill = import_skill(
                    config,
                    source_path=str(source_root),
                    name_hint=name_hint,
                )
        elif suffix in _SKILL_MARKDOWN_SUFFIXES:
            payload = await api_mod._read_upload_bytes(
                file,
                max_bytes=_MAX_SKILL_MARKDOWN_UPLOAD_BYTES,
                label="Skill Markdown",
            )
            markdown = payload.decode("utf-8", errors="replace")
            _reject_plain_markdown_skill_import(markdown)
            skill = import_skill(
                config,
                skill_markdown=markdown,
                name_hint=name_hint,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Upload SKILL.md, a Markdown skill file, or a .zip skill folder.",
            )
    except SkillError as exc:
        _raise_skill_http_error(exc)
    return {"ok": True, "skill": skill}


@router.get("/admin/skills/{skill_id}")
async def admin_get_skill(skill_id: str) -> dict:
    from runtime.skills import SkillError, get_skill

    try:
        return {"skill": get_skill(api_mod._load_runtime_config(), skill_id)}
    except SkillError as exc:
        _raise_skill_http_error(exc)


@router.post("/admin/skills/{skill_id}/audit")
async def admin_audit_skill(skill_id: str) -> dict:
    from runtime.skills import SkillError, audit_installed_skill

    try:
        skill = audit_installed_skill(api_mod._load_runtime_config(), skill_id)
    except SkillError as exc:
        _raise_skill_http_error(exc)
    return {"ok": True, "skill": skill}


@router.post("/admin/skills/{skill_id}/enable")
async def admin_enable_skill(skill_id: str) -> dict:
    from runtime.skills import SkillError, set_skill_enabled

    try:
        skill = set_skill_enabled(api_mod._load_runtime_config(), skill_id, True)
    except SkillError as exc:
        _raise_skill_http_error(exc)
    return {"ok": True, "skill": skill}


@router.post("/admin/skills/{skill_id}/disable")
async def admin_disable_skill(skill_id: str) -> dict:
    from runtime.skills import SkillError, set_skill_enabled

    try:
        skill = set_skill_enabled(api_mod._load_runtime_config(), skill_id, False)
    except SkillError as exc:
        _raise_skill_http_error(exc)
    return {"ok": True, "skill": skill}


@router.delete("/admin/skills/{skill_id}")
async def admin_delete_skill(skill_id: str) -> dict:
    from runtime.skills import SkillError, delete_skill

    try:
        result = delete_skill(api_mod._load_runtime_config(), skill_id)
    except SkillError as exc:
        _raise_skill_http_error(exc)
    return {"ok": True, **result}


def _raise_skill_http_error(exc: Exception) -> None:
    detail = str(exc)
    code = 404 if "not found" in detail.lower() else 400
    raise HTTPException(status_code=code, detail=detail)


def _reject_plain_markdown_skill_import(markdown: str) -> None:
    """Reject ordinary documents before they are installed as invalid skills."""
    stripped = (markdown or "").lstrip()
    if stripped.startswith("---"):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "This Markdown file is not a skill. Skill imports must be SKILL.md "
            "content with YAML frontmatter containing name and description. "
            "Use Admin > Life to digest articles, notes, or other learning text."
        ),
    )


def _safe_extract_zip(payload: bytes, destination: Path) -> None:
    destination = destination.resolve()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members:
                raise HTTPException(status_code=400, detail="Skill archive is empty.")
            if len(members) > _MAX_SKILL_ARCHIVE_FILES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Skill archive has too many files ({_MAX_SKILL_ARCHIVE_FILES} limit).",
                )
            total_size = sum(max(0, int(info.file_size)) for info in members)
            if total_size > _MAX_SKILL_ARCHIVE_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="Skill archive expands beyond the upload size limit.",
                )
            for info in members:
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or name.startswith("../") or "/../" in name:
                    raise HTTPException(
                        status_code=400,
                        detail="Skill archive contains an unsafe path.",
                    )
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise HTTPException(
                        status_code=400,
                        detail="Skill archive contains a symbolic link.",
                    )
                target = (destination / name).resolve()
                try:
                    target.relative_to(destination)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="Skill archive contains an unsafe path.",
                    ) from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Skill archive is not a valid zip file.") from exc


def _single_skill_root_from_upload(extract_root: Path) -> Path:
    candidates = [
        path.parent
        for path in extract_root.rglob("SKILL.md")
        if path.is_file() and "__MACOSX" not in path.parts
    ]
    if not candidates:
        raise HTTPException(status_code=400, detail="No SKILL.md found in uploaded archive.")
    unique = sorted({path.resolve() for path in candidates}, key=lambda item: (len(item.parts), str(item)))
    if len(unique) > 1:
        raise HTTPException(
            status_code=400,
            detail="Uploaded archive contains multiple skills. Upload one skill folder at a time.",
        )
    return unique[0]

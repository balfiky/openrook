"""Admin Life History endpoints (``/admin/life/*`` and constitution).

Split out of ``interface/api.py``, which mounts this router at import time
and re-exports the handlers. Handlers resolve shared helpers late through
the ``interface.api`` module object (``api_mod``) so test monkeypatching
on ``interface.api`` keeps working.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from interface import api as api_mod

router = APIRouter(dependencies=[Depends(api_mod._require_bearer)])

_MAX_LIFE_UPLOAD_BYTES = 10_000_000
_LIFE_UPLOAD_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}


class AdminLifeTextRequest(BaseModel):
    title: str = Field(default="", max_length=180)
    text: str = Field(..., min_length=1, max_length=250000)
    source_type: str = Field("pasted_text", min_length=1, max_length=80)
    participants: list[str] = Field(default_factory=list, max_length=20)


class AdminLifeFileRequest(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=4096)
    title: str = Field(default="", max_length=180)
    participants: list[str] = Field(default_factory=list, max_length=20)


class AdminLifeRollbackRequest(BaseModel):
    batch_id: str = Field(..., min_length=1, max_length=80)


class AdminConstitutionRequest(BaseModel):
    """Operator-set stable orientation rendered above evolving beliefs."""

    constitution: str = Field(..., max_length=2000)


@router.get("/admin/life")
async def admin_life_overview() -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return store.overview()
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.post("/admin/life/experiences/text")
async def admin_life_ingest_text(req: AdminLifeTextRequest) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    config = api_mod._load_runtime_config()
    backend = api_mod._optional_life_llm_backend(config)
    try:
        with LifeHistoryStore(config) as store:
            result = store.ingest_pasted_text(
                title=req.title,
                text=req.text,
                source_type=req.source_type,
                participants=req.participants,
                llm_client=backend,
            )
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)
    finally:
        _close_optional_backend(backend)
    return {"ok": True, **result}


@router.post("/admin/life/experiences/file")
async def admin_life_ingest_file(req: AdminLifeFileRequest) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    config = api_mod._load_runtime_config()
    backend = api_mod._optional_life_llm_backend(config)
    try:
        with LifeHistoryStore(config) as store:
            result = store.ingest_local_file(
                file_path=req.file_path,
                title=req.title,
                participants=req.participants,
                llm_client=backend,
            )
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)
    finally:
        _close_optional_backend(backend)
    return {"ok": True, **result}


@router.post("/admin/life/experiences/upload")
async def admin_life_ingest_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    participants: str = Form(""),
) -> dict:
    """Digest a browser-uploaded text or Markdown file into Life History."""
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    filename = api_mod._upload_filename(file, fallback="uploaded-experience.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in _LIFE_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Life History uploads support plain text, Markdown, reStructuredText, or .text files.",
        )
    payload = await api_mod._read_upload_bytes(
        file,
        max_bytes=_MAX_LIFE_UPLOAD_BYTES,
        label="Life History file",
    )
    text = payload.decode("utf-8", errors="replace")

    config = api_mod._load_runtime_config()
    backend = api_mod._optional_life_llm_backend(config)
    try:
        with LifeHistoryStore(config) as store:
            result = store.ingest_uploaded_text(
                filename=filename,
                text=text,
                title=title,
                participants=_split_form_list(participants),
                llm_client=backend,
            )
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)
    finally:
        _close_optional_backend(backend)
    return {"ok": True, **result}


@router.get("/admin/life/experiences")
async def admin_life_experiences(limit: int = 50) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return {"experiences": store.list_experiences(limit=limit)}
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.get("/admin/life/evolution")
async def admin_life_evolution(limit: int = 100) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return {"evolution_events": store.list_evolution(limit=limit)}
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.get("/admin/life/beliefs")
async def admin_life_beliefs(limit: int = 100) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return {"beliefs": store.list_beliefs(limit=limit)}
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.get("/admin/life/drives")
async def admin_life_drives() -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return {"drives": store.list_drives()}
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.post("/admin/life/rollback")
async def admin_life_rollback(req: AdminLifeRollbackRequest) -> dict:
    raise HTTPException(
        status_code=410,
        detail="Life History rollback was removed; use an explicit genesis reset for a full reset.",
    )


@router.get("/admin/identity/constitution")
async def admin_get_constitution() -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return store.get_constitution()
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.put("/admin/identity/constitution")
async def admin_set_constitution(req: AdminConstitutionRequest) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return {"ok": True, **store.set_constitution(req.constitution)}
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.get("/admin/life/open-questions")
async def admin_life_open_questions(
    status: str | None = "open",
    limit: int = 50,
) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    normalized_status: str | None
    if status in ("", "all"):
        normalized_status = None
    elif status in ("open", "pursuing", "resolved", "abandoned"):
        normalized_status = status
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status filter: {status}",
        )
    bounded_limit = max(1, min(int(limit), 200))
    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            return {
                "questions": store.list_open_questions(
                    status=normalized_status, limit=bounded_limit,
                ),
                "counts": store.count_open_questions(),
            }
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


@router.post("/admin/life/open-questions/{question_id}/abandon")
async def admin_life_abandon_open_question(question_id: int) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            updated = store.abandon_open_question(question_id)
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Open question {question_id} not found or already closed.",
        )
    return {"ok": True, "id": question_id, "status": "abandoned"}


@router.post("/admin/life/open-questions/{question_id}/resolve")
async def admin_life_resolve_open_question(question_id: int) -> dict:
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    try:
        with LifeHistoryStore(api_mod._load_runtime_config()) as store:
            updated = store.resolve_open_question(question_id)
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Open question {question_id} not found or already closed.",
        )
    return {"ok": True, "id": question_id, "status": "resolved"}


@router.post("/admin/life/metabolism/tick")
async def admin_life_metabolism_tick() -> dict:
    """Manually trigger the wall-clock metabolism tick.

    Idempotent under the configured rate limit (``metabolism_min_elapsed_days``,
    default 1 day). Useful for operator-driven reflection and as a stand-in
    for an external scheduler.
    """
    from runtime.life_history import LifeHistoryError, LifeHistoryStore

    config = api_mod._load_runtime_config()
    try:
        with LifeHistoryStore(config) as store:
            return {
                "ok": True,
                **store.wall_clock_decay(
                    min_elapsed_days=float(config.metabolism_min_elapsed_days),
                ),
            }
    except LifeHistoryError as exc:
        _raise_life_http_error(exc)


def _raise_life_http_error(exc: Exception) -> None:
    detail = str(exc)
    code = 404 if "not found" in detail.lower() else 400
    raise HTTPException(status_code=code, detail=detail)


def _close_optional_backend(backend) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _split_form_list(raw: str) -> list[str]:
    values = []
    for item in str(raw or "").replace("\n", ",").split(","):
        stripped = item.strip()
        if stripped:
            values.append(stripped)
    return values[:20]

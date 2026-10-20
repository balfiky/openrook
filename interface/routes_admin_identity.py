"""Admin console identity (soul) endpoints: read, draft, import, save.

Split out of ``interface/routes_admin.py``, which aggregates this router and
re-exports the handlers. Handlers resolve shared helpers and mutable globals
late through the ``interface.api`` module object (``api_mod``) so test
monkeypatching on ``interface.api`` keeps working.
"""

from __future__ import annotations

from config.environment import project_env

import json
import os

from fastapi import APIRouter, Depends, HTTPException

from interface import api as api_mod
from runtime.config import RuntimeConfig

router = APIRouter(dependencies=[Depends(api_mod._require_bearer)])


@router.get("/admin/soul")
async def admin_get_soul() -> dict:
    """Return current seed identity for the admin Identity section."""
    from config.loader import get_config

    return _admin_soul_payload(get_config().soul, saved=True)


@router.post("/admin/soul/draft")
async def admin_draft_soul(req: api_mod.AdminSoulDraftRequest) -> dict:
    """Draft a soul.yaml from a natural-language description via the LLM.

    Returns a validated draft; does NOT persist. The UI populates the
    Identity form from the draft so the operator can review and edit
    before saving.
    """
    config = api_mod._load_runtime_config()
    if not _llm_configured_for_draft(config):
        raise HTTPException(
            status_code=400,
            detail=(
                "LLM not configured. Set an LLM backend in Settings and run "
                "the LLM test before using the 'Describe your agent' helper."
            ),
        )
    prompt_template = _load_soul_draft_prompt()
    if not prompt_template:
        raise HTTPException(
            status_code=500,
            detail="Soul draft prompt template not found in config/prompts/.",
        )
    system_prompt = prompt_template.replace("{{description}}", req.description)
    try:
        backend = api_mod.create_llm_backend(config)
        raw = backend.generate(system_prompt, req.description)
    except Exception as exc:  # pragma: no cover - depends on provider
        raise HTTPException(
            status_code=502,
            detail=api_mod._redact_error(
                f"LLM call failed: {exc.__class__.__name__}: {exc}", config,
            ),
        )
    payload = _parse_soul_draft_json(raw)
    try:
        draft = api_mod.AdminSoulUpdateRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=api_mod._redact_error(
                f"LLM draft did not match the soul schema: {exc}", config,
            ),
        )
    return {
        "ok": True,
        "draft": draft.model_dump(),
        "backend": config.llm_backend,
        "notes": [
            "This is a draft, not a save. Review the fields and press Save Identity to commit.",
            "LLM-authored drafts may drift from the description on second reading; edit anything that feels off.",
        ],
    }


@router.post("/admin/soul/import")
async def admin_import_soul(req: api_mod.AdminSoulImportRequest) -> dict:
    """Parse pasted soul.yaml or a plain identity document without an LLM call."""
    payload, source = _parse_raw_soul_document(req.raw)
    try:
        draft = api_mod.AdminSoulUpdateRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Pasted identity did not match the soul schema: {exc}",
        )
    return {
        "ok": True,
        "draft": draft.model_dump(),
        "source": source,
        "notes": [
            "Parsed locally. Review the fields and press Save Identity to commit.",
        ],
    }


@router.post("/admin/soul")
async def admin_update_soul(req: api_mod.AdminSoulUpdateRequest) -> dict:
    """Write a new soul.yaml, reload the config singleton, and evict the
    cached session manager so the next chat turn builds a fresh pipeline
    that reads the new identity. No process restart required.
    """
    from config.loader import get_config, reset_config

    payload = {
        "soul": {
            "name": req.name,
            "identity": req.identity,
            "voice": req.voice,
            "relational_stance": req.relational_stance,
            "likes": req.likes,
            "dislikes": req.dislikes,
            "boundaries": req.boundaries,
            "growth_policy": req.growth_policy,
            "core_values": req.core_values,
            "initial_traits": req.initial_traits,
        }
    }
    path = api_mod._soul_yaml_path()
    _write_soul_yaml(payload, path)
    reset_config()
    reloaded_manager, _ = api_mod._detach_session_manager_for_reload(
        reason="identity save",
    )
    api_mod._refresh_live_channel_session_manager()
    return _admin_soul_payload(
        get_config().soul,
        saved=True,
        path=path,
        reloaded_session_manager=reloaded_manager,
    )


# ---------------------------------------------------------------------------
# Soul (identity) helpers
# ---------------------------------------------------------------------------


def _write_soul_yaml(data: dict, path: str) -> None:
    """Atomic YAML write: temp file in the same dir, then os.replace."""
    import yaml as _yaml

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        _yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    os.replace(tmp_path, path)


def _llm_configured_for_draft(config: RuntimeConfig) -> bool:
    """True when the configured LLM is callable for draft generation."""
    backend = config.llm_backend
    if backend in {"provider", "openai_compatible"}:
        return bool(config.llm_base_url.strip() and config.llm_model.strip())
    if backend == "codex":
        return api_mod.codex_cli_available()
    if backend == "minimax":
        return bool(config.llm_api_key or os.environ.get("LLM_API_KEY"))
    if backend == "auto":
        has_generic = bool(
            config.llm_base_url.strip()
            and config.llm_model.strip()
        )
        return has_generic or bool(config.llm_api_key or os.environ.get("LLM_API_KEY"))
    return False


def _load_soul_draft_prompt() -> str:
    """Load the soul-from-description prompt template.

    Honors the same OPENROOK_CONFIG_DIR override as every other prompt.
    """
    from config.loader import _load_prompt

    return _load_prompt("soul_from_description.md")


def _parse_soul_draft_json(raw: str) -> dict:
    """Extract a JSON object from an LLM reply.

    Permissive about leading/trailing prose or code fences the LLM may
    emit despite the 'output only JSON' instruction — grabs the first
    top-level {...} block. Raises 502 if no JSON object can be found.
    """
    text = (raw or "").strip()
    if not text:
        raise HTTPException(
            status_code=502,
            detail="LLM returned an empty response for the soul draft.",
        )
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise HTTPException(
            status_code=502,
            detail="LLM did not return a JSON object for the soul draft.",
        )
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM draft was not valid JSON: {exc}",
        )


_SOUL_FIELDS = {
    "name",
    "identity",
    "voice",
    "relational_stance",
    "growth_policy",
    "likes",
    "dislikes",
    "boundaries",
    "core_values",
    "initial_traits",
}


def _parse_raw_soul_document(raw: str) -> tuple[dict, str]:
    """Parse pasted identity text into the admin soul schema.

    Accepts either real ``soul.yaml`` (with or without a top-level ``soul`` key)
    or a plain text document beginning with ``Name: ...``. The latter keeps the
    wizard's raw-paste path deterministic instead of depending on an LLM.
    """
    text = raw.strip()
    yaml_payload = _parse_soul_yaml_payload(text)
    if yaml_payload is not None:
        return yaml_payload, "yaml"
    return _parse_plain_soul_document(text), "text"


def _parse_soul_yaml_payload(text: str) -> dict | None:
    import yaml as _yaml

    try:
        loaded = _yaml.safe_load(text)
    except _yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None

    source = loaded.get("soul") if isinstance(loaded.get("soul"), dict) else loaded
    normalized = {
        str(key).strip().lower().replace("-", "_"): value
        for key, value in source.items()
    }
    if not any(field in normalized for field in _SOUL_FIELDS):
        return None

    return _coerce_soul_payload(normalized)


def _parse_plain_soul_document(text: str) -> dict:
    lines = text.splitlines()
    first_idx = next((i for i, line in enumerate(lines) if line.strip()), -1)
    if first_idx < 0:
        raise HTTPException(status_code=400, detail="Pasted identity document is empty.")

    name = ""
    first = lines[first_idx].strip()
    if first.lower().startswith("name:"):
        name = first.split(":", 1)[1].strip()
        body = "\n".join(lines[first_idx + 1 :]).strip()
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Paste a soul.yaml document or start the identity document with "
                "'Name: <agent name>'."
            ),
        )
    if not name:
        raise HTTPException(status_code=400, detail="Identity document name is empty.")
    if not body:
        raise HTTPException(status_code=400, detail="Identity document body is empty.")

    sections = _split_plain_identity_sections(body)
    identity = sections.get("identity") or body
    return _coerce_soul_payload(
        {
            "name": name,
            "identity": _trim_text(identity, 2000),
            "voice": _trim_text(sections.get("voice", ""), 2000),
            "relational_stance": _trim_text(sections.get("relational_stance", ""), 2000),
            "growth_policy": _trim_text(sections.get("growth_policy", ""), 2000),
            "likes": _plain_list(sections.get("likes", "")),
            "dislikes": _plain_list(sections.get("dislikes", "")),
            "boundaries": _plain_boundaries(sections.get("boundaries", "")),
            "core_values": {},
            "initial_traits": {},
        }
    )


def _split_plain_identity_sections(body: str) -> dict[str, str]:
    markers = {
        "core tone:": "voice",
        "voice:": "voice",
        "relational stance:": "relational_stance",
        "growth policy:": "growth_policy",
        "likes:": "likes",
        "dislikes:": "dislikes",
        "boundaries:": "boundaries",
    }
    sections: dict[str, list[str]] = {"identity": []}
    current = "identity"
    for line in body.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        for prefix, marker in markers.items():
            if lowered.startswith(prefix):
                current = marker
                sections.setdefault(current, [])
                remainder = stripped.split(":", 1)[1].strip()
                if remainder:
                    sections[current].append(remainder)
                break
        else:
            sections.setdefault(current, []).append(line)
            continue
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _plain_boundaries(text: str) -> list[str]:
    if not text.strip():
        return []
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("never "):
            parts = [part.strip(" .") for part in stripped.split(".") if part.strip()]
            items.extend(parts)
        else:
            items.append(stripped)
    return [_trim_text(item, 200) for item in items[:32] if item]


def _plain_list(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            value = part.strip(" -\t")
            if value:
                items.append(_trim_text(value, 200))
    return items[:32]


def _coerce_soul_payload(data: dict) -> dict:
    return {
        "name": _trim_text(str(data.get("name") or ""), 64),
        "identity": _trim_text(str(data.get("identity") or ""), 2000),
        "voice": _trim_text(str(data.get("voice") or ""), 2000),
        "relational_stance": _trim_text(str(data.get("relational_stance") or ""), 2000),
        "growth_policy": _trim_text(str(data.get("growth_policy") or ""), 2000),
        "likes": _coerce_string_list(data.get("likes")),
        "dislikes": _coerce_string_list(data.get("dislikes")),
        "boundaries": _coerce_string_list(data.get("boundaries")),
        "core_values": _coerce_weight_dict(data.get("core_values")),
        "initial_traits": _coerce_weight_dict(data.get("initial_traits")),
    }


def _coerce_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.splitlines()
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = [value]
    return [_trim_text(str(item).strip(), 200) for item in candidates if str(item).strip()]


def _coerce_weight_dict(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            out[name] = float(raw)
        except (TypeError, ValueError):
            out[name] = 0.5
    return out


def _trim_text(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _admin_soul_payload(
    soul,
    *,
    saved: bool = True,
    path: str | None = None,
    reloaded_session_manager: bool = False,
) -> dict:
    notes: list[str] = []
    if saved:
        notes.append(
            "Identity saved. The next chat turn will reflect the new "
            "identity across the whole prompt stack (generator, self-check, "
            "memory digestion, role labels)."
        )
        if reloaded_session_manager:
            notes.append(
                "Active chat sessions were rebuilt so they use the new "
                "identity immediately; no server restart needed."
            )
    if not project_env("CONFIG_DIR", "").strip():
        notes.append(
            "To make this identity survive `pip install --upgrade`, set "
            "OPENROOK_CONFIG_DIR to a writable directory outside site-packages "
            "and save again — future writes and reads will use that dir."
        )
    return {
        "soul": {
            "name": soul.name,
            "identity": soul.identity,
            "voice": soul.voice,
            "relational_stance": soul.relational_stance,
            "likes": list(soul.likes),
            "dislikes": list(soul.dislikes),
            "boundaries": list(soul.boundaries),
            "growth_policy": soul.growth_policy,
            "core_values": dict(soul.core_values),
            "initial_traits": dict(soul.initial_traits),
        },
        "is_default_name": soul.name.strip() == api_mod._DEFAULT_SOUL_NAME,
        "saved": saved,
        "path": path or api_mod._soul_yaml_path(),
        "reloaded_session_manager": reloaded_session_manager,
        "notes": notes,
    }

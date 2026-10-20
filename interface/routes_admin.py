"""Admin console route aggregator.

The admin endpoints live in three focused modules:

- ``routes_admin_config``   — /admin/status, /admin/config, codex models,
  test/* connectivity checks, and config-introspection helpers.
- ``routes_admin_identity`` — /admin/soul* (read, draft, import, save) and
  the soul parsing/serialization helpers.
- ``routes_admin_runtime``  — persona state, diagnostics, runtime
  reload/restart, session reset, and user deletion.

This module combines their routers into the single ``router`` that
``interface/api.py`` mounts, and re-exports every handler, request model,
and helper so existing imports (``interface.api`` re-exports, the backup
routes, tests) keep working unchanged. Handlers resolve patchable shared
helpers late through the ``interface.api`` module object.
"""

from __future__ import annotations

from fastapi import APIRouter

from interface import routes_admin_config as _config_mod
from interface import routes_admin_identity as _identity_mod
from interface import routes_admin_runtime as _runtime_mod

from interface.routes_admin_config import (  # noqa: F401
    AdminConfigUpdateRequest,
    AdminLLMTestRequest,
    AdminStorageTestRequest,
    AdminTelegramTestRequest,
    _CONFIG_FIELD_SECTIONS,
    _LIVE_RELOAD_FIELDS,
    _SECRET_ENV_VARS,
    _SESSION_RELOAD_FIELDS,
    _admin_config_for_llm_test,
    _admin_config_for_storage_test,
    _admin_config_for_telegram_test,
    _admin_config_payload,
    _admin_config_warnings,
    _admin_field_metadata,
    _admin_secret_status,
    _admin_setup_status,
    _admin_status_payload,
    _allowed_values_for_field,
    _check_writable_directory,
    _llm_configured,
    _looks_unconfigured,
    _read_soul_name,
    _warning,
    _write_admin_state,
    admin_codex_models,
    admin_get_config,
    admin_status,
    admin_test_llm,
    admin_test_storage,
    admin_test_telegram,
    admin_update_config,
)
from interface.routes_admin_identity import (  # noqa: F401
    _SOUL_FIELDS,
    _admin_soul_payload,
    _coerce_soul_payload,
    _coerce_string_list,
    _coerce_weight_dict,
    _llm_configured_for_draft,
    _load_soul_draft_prompt,
    _parse_plain_soul_document,
    _parse_raw_soul_document,
    _parse_soul_draft_json,
    _parse_soul_yaml_payload,
    _plain_boundaries,
    _plain_list,
    _split_plain_identity_sections,
    _trim_text,
    _write_soul_yaml,
    admin_draft_soul,
    admin_get_soul,
    admin_import_soul,
    admin_update_soul,
)
from interface.routes_admin_runtime import (  # noqa: F401
    AdminSessionResetRequest,
    AdminUserDeleteRequest,
    _admin_delete_user_data,
    _admin_diagnostics_payload,
    _admin_persona_payload,
    _directory_summary,
    _split_session_key,
    admin_delete_user,
    admin_diagnostics,
    admin_persona_state,
    admin_reload_runtime,
    admin_reset_session,
    admin_restart_runtime,
)

router = APIRouter()
router.include_router(_config_mod.router)
router.include_router(_identity_mod.router)
router.include_router(_runtime_mod.router)

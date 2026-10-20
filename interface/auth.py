"""Shared opt-in bearer-token auth for the legacy and ``/v1`` API surfaces.

Semantics (identical for both layers):
  * Auth is opt-in — when no ``api_key`` is configured, every check is a
    no-op and the surface stays open.
  * The config is re-read on every request so rotating or setting the key
    through POST /config takes effect immediately without a restart.
  * Token comparison is constant-time (``secrets.compare_digest``).
  * Failures return 401 with ``WWW-Authenticate: Bearer`` and the
    historical detail strings ("Missing bearer token" / "Invalid bearer
    token").
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status


def extract_bearer_token(authorization: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer ...`` header."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


def verify_bearer(token: str | None, expected: str | None) -> None:
    """Raise 401 unless ``token`` matches the configured ``expected`` key.

    If ``expected`` is empty, auth is disabled and this is a no-op.
    """
    expected_key = str(expected or "")
    if not expected_key:
        return
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def build_require_bearer(config_getter):
    """FastAPI dependency factory enforcing the currently-configured api_key.

    ``config_getter`` is invoked per request so in-process key rotation is
    honored without re-mounting routes.
    """

    async def require_bearer(
        authorization: str | None = Header(default=None),
    ) -> None:
        verify_bearer(
            extract_bearer_token(authorization),
            config_getter().api_key,
        )

    return require_bearer

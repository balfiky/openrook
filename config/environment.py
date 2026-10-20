"""OpenRook environment settings."""

from __future__ import annotations

import os
from collections.abc import Mapping


def project_env(
    name: str,
    default: str = "",
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read OPENROOK_<name>, including an explicitly empty value."""
    environment = os.environ if environ is None else environ
    return environment.get(f"OPENROOK_{name}", default)

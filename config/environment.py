"""OpenRook environment settings with legacy NUR_* compatibility."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from threading import Lock

_notice_lock = Lock()
_notices_shown: set[str] = set()


def warn_legacy(kind: str, old_name: str, new_name: str) -> None:
    """Print a deprecation notice once per category in this process."""
    with _notice_lock:
        if kind in _notices_shown:
            return
        _notices_shown.add(kind)
        print(
            f"Deprecation: {old_name} is deprecated; use {new_name} instead.",
            file=sys.stderr,
        )


def project_env(
    name: str,
    default: str = "",
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read OPENROOK_<name> first, including an explicitly empty value."""
    environment = os.environ if environ is None else environ
    canonical = f"OPENROOK_{name}"
    if canonical in environment:
        return environment[canonical]
    legacy = f"NUR_{name}"
    if legacy in environment:
        warn_legacy("environment", legacy, canonical)
        return environment[legacy]
    return default

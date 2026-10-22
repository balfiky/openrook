"""HTTP boundary and output redaction. Never serialize a request or its headers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx


REDACTED = "[redacted]"
_SECRET_FIELD = re.compile(
    r"authorization|api[_-]?key|password|secret|credentials|cookie|(?:^|_)token$", re.I
)
_PRIVATE_FIELD = re.compile(r"(?:base_url|endpoint|llm_model|model_name)$", re.I)
_URL = re.compile(r"https?://[^\s<>\"')]+", re.I)


class Redactor:
    def __init__(self, *values: str):
        self.values: set[str] = set()
        for name, value in os.environ.items():
            if _SECRET_FIELD.search(name) or name.upper().endswith("_TOKEN"):
                self.add(value)
        for value in values:
            self.add(value)

    def add(self, value: Any) -> None:
        if isinstance(value, str) and value:
            self.values.add(value)

    def clean(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                self.clean(str(key)): REDACTED
                if (_SECRET_FIELD.search(str(key)) or _PRIVATE_FIELD.search(str(key)))
                and not isinstance(item, bool)
                else self.clean(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.clean(item) for item in value]
        if not isinstance(value, str):
            return value
        for secret in sorted(self.values, key=len, reverse=True):
            value = value.replace(secret, REDACTED)
        value = re.sub(r"(?i)authorization\s*[:=]\s*[^\r\n,}]+", "Authorization: [redacted]", value)
        value = re.sub(r"(?i)\bBearer\s+[^\s\"',}]+", "Bearer [redacted]", value)
        value = re.sub(
            r'''(?i)((?<![\w-])["']?(?:authorization|(?:[a-z0-9_-]{1,48}[_-])?(?:api[_-]?key|token|password|secret|credentials|cookies?))\b["']?\s*[:=]\s*)'''
            r'''(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;}\]]+)''',
            r'\1"[redacted]"', value,
        )
        return _URL.sub("[destination redacted]", value)


def validate_destination(value: str) -> str:
    try:
        parts = urlsplit(value)
        parts.port
    except ValueError:
        raise ValueError("Invalid HTTP base URL") from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Supply an HTTP base URL without credentials, query, or fragment")
    return value.rstrip("/")


def destination_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class Response:
    status: int | None
    data: Any
    latency_ms: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300 and self.error is None

    def to_dict(self) -> dict:
        return asdict(self)


class PublicAPI:
    def __init__(
        self,
        base_url: str,
        *,
        key: str = "",
        timeout: float = 900,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = validate_destination(base_url)
        self.client = httpx.AsyncClient(
            base_url=self.base_url + "/",
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {key}"} if key else {},
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    async def request(self, method: str, path: str, **kwargs) -> Response:
        started = time.monotonic()
        try:
            response = await self.client.request(method, path.lstrip("/"), **kwargs)
            try:
                data = response.json()
            except ValueError:
                # HTML error bodies can include proxy settings and credentials.
                return Response(response.status_code, None, elapsed(started), "non_json_response")
            return Response(response.status_code, data, elapsed(started))
        except (httpx.HTTPError, OSError) as exc:
            # Exception messages can include URLs, credentials and request headers.
            return Response(None, None, elapsed(started), type(exc).__name__)

    async def close(self) -> None:
        await self.client.aclose()


def elapsed(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 2)


def workspace_snapshot(root: Path | None, paths: tuple[str, ...], *, expected_content: bytes | None = None) -> dict:
    """Read only generated fixture paths; never follow a symlink outside root."""
    if root is None:
        return {"available": False, "reason": "Supply --tools-workspace for independent evidence"}
    result: dict = {"available": True, "files": {}}
    resolved_root = root.resolve()
    for relative in paths:
        target = root / relative
        try:
            target.resolve().relative_to(resolved_root)
            if target.is_symlink():
                result["files"][relative] = {"error": "symlink"}
            elif not target.exists():
                result["files"][relative] = {"exists": False}
            elif target.is_file():
                size = target.stat().st_size
                digest = None
                if size <= 1_000_000:
                    content = target.read_bytes()
                    digest = hashlib.sha256(content).hexdigest()
                result["files"][relative] = {"exists": True, "size": size, "sha256": digest}
                if expected_content is not None and digest is not None:
                    result["files"][relative]["fixture_content_present"] = expected_content in content
            else:
                result["files"][relative] = {"exists": True, "kind": "directory"}
        except (OSError, ValueError):
            result["files"][relative] = {"error": "unreadable_or_outside_workspace"}
    return result

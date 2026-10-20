"""Shared pytest fixtures.

``runtime.security.validate_http_url`` resolves non-literal hostnames via
DNS (``socket.getaddrinfo``) and rejects URLs whose addresses land in
loopback/link-local/private space — see the SSRF guard in
``runtime/security.py``. The test suite uses many fake hostnames
(``provider.example``, ``example.invalid``, ``prod-llm``...) and must never
depend on real DNS, so hostname resolution is stubbed to a fixed public
address by default.

Tests that exercise the resolution policy itself (tests/test_url_security.py)
restore the real helper and monkeypatch ``socket.getaddrinfo`` with crafted
answers; a test-level monkeypatch layers over this autouse stub and is
unwound in LIFO order, so both restore cleanly.

Runtime settings are isolated before test collection because importing the API
reads the config for CORS. Each test then receives its own temporary config.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pytest

from runtime.config import RuntimeConfig


_FAKE_PUBLIC_IP = "93.184.216.34"  # example.com — a public, non-blocked address


def _write_runtime_config(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "runtime_config.yaml"
    RuntimeConfig(data_dir=str(directory / "data")).write_yaml(str(path))
    return path


def pytest_configure(config):
    """Keep import-time API configuration away from the developer's files."""
    directory = tempfile.TemporaryDirectory(prefix="openrook-pytest-")
    config.add_cleanup(directory.cleanup)
    path = _write_runtime_config(Path(directory.name))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", str(path))
    config.add_cleanup(monkeypatch.undo)


@pytest.fixture(autouse=True)
def _isolated_runtime_config(tmp_path_factory, monkeypatch):
    """Specialized fixtures can override this config without sharing state."""
    # Keep fixture files outside tmp_path, which tests may treat as an empty workspace.
    path = _write_runtime_config(tmp_path_factory.mktemp("runtime-config"))
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", str(path))
    api = sys.modules.get("interface.api")
    if api is not None:
        monkeypatch.setattr(api, "RUNTIME_CONFIG_PATH", str(path))


@pytest.fixture(autouse=True)
def _stub_url_dns_resolution(monkeypatch):
    """Make URL validation DNS-independent for the whole suite."""
    monkeypatch.setattr(
        "runtime.security._resolve_hostname",
        lambda host, port: [_FAKE_PUBLIC_IP],
    )

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
"""

from __future__ import annotations

import pytest


_FAKE_PUBLIC_IP = "93.184.216.34"  # example.com — a public, non-blocked address


@pytest.fixture(autouse=True)
def _stub_url_dns_resolution(monkeypatch):
    """Make URL validation DNS-independent for the whole suite."""
    monkeypatch.setattr(
        "runtime.security._resolve_hostname",
        lambda host, port: [_FAKE_PUBLIC_IP],
    )

"""Tests for the SSRF guard in runtime/security.py — validate_http_url.

The suite-wide conftest stubs ``runtime.security._resolve_hostname`` so
ordinary tests never touch DNS. The resolution-policy tests here restore
the real helper and monkeypatch ``socket.getaddrinfo`` with crafted
answers — no test in this file performs a real DNS lookup.
"""

from __future__ import annotations

import socket

import pytest

import runtime.security as security

# Captured at import time, before the autouse conftest stub is applied.
_REAL_RESOLVE = security._resolve_hostname


def _fake_addrinfo(addresses: list[str]):
    """Build a socket.getaddrinfo-shaped result for the given IP strings."""
    def fake(host, port, *args, **kwargs):
        infos = []
        for addr in addresses:
            if ":" in addr:
                infos.append((
                    socket.AF_INET6, socket.SOCK_STREAM, 6, "",
                    (addr, port, 0, 0),
                ))
            else:
                infos.append((
                    socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port),
                ))
        return infos
    return fake


@pytest.fixture
def real_dns(monkeypatch):
    """Re-enable the real resolution helper (over the conftest stub) and
    return a function that plants crafted getaddrinfo answers."""
    monkeypatch.setattr(security, "_resolve_hostname", _REAL_RESOLVE)

    def plant(addresses: list[str] | None = None, error: Exception | None = None):
        if error is not None:
            def fake(*args, **kwargs):
                raise error
            monkeypatch.setattr(socket, "getaddrinfo", fake)
        else:
            monkeypatch.setattr(socket, "getaddrinfo", _fake_addrinfo(addresses or []))

    return plant


# ===================================================================
# Literal-IP and scheme policy (no DNS involved)
# ===================================================================

class TestLiteralPolicy:
    def test_non_http_scheme_rejected(self):
        with pytest.raises(ValueError, match="http or https"):
            security.validate_http_url("ftp://example.com/file")

    def test_missing_hostname_rejected(self):
        with pytest.raises(ValueError, match="hostname"):
            security.validate_http_url("http://")

    def test_localhost_rejected_by_default(self):
        with pytest.raises(ValueError, match="local"):
            security.validate_http_url("http://localhost:8080/x")

    def test_localhost_allowed_with_allow_loopback(self):
        url = security.validate_http_url(
            "http://localhost:11434/v1", allow_loopback=True,
        )
        assert url == "http://localhost:11434/v1"

    def test_literal_loopback_rejected_by_default(self):
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://127.0.0.1:9000/")

    def test_literal_loopback_allowed_with_allow_loopback(self):
        url = security.validate_http_url(
            "http://127.0.0.1:11434/v1", allow_loopback=True,
        )
        assert "127.0.0.1" in url

    def test_literal_link_local_metadata_rejected(self):
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://169.254.169.254/latest/meta-data/")

    def test_literal_private_rejected(self):
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://10.0.0.8/admin")

    def test_literal_private_allowed_with_env_optin(self, monkeypatch):
        monkeypatch.setenv("OPENROOK_TEST_ALLOW_PRIVATE", "1")
        url = security.validate_http_url(
            "http://10.0.0.8/admin", allow_private_env="OPENROOK_TEST_ALLOW_PRIVATE",
        )
        assert url == "http://10.0.0.8/admin"

    def test_literal_public_allowed(self):
        url = security.validate_http_url("https://8.8.8.8/")
        assert "8.8.8.8" in url

    def test_literal_ipv6_loopback_rejected(self):
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://[::1]:8080/")


# ===================================================================
# Hostname resolution policy (DNS-blind SSRF fix)
# ===================================================================

class TestHostnameResolution:
    def test_hostname_resolving_to_loopback_rejected(self, real_dns):
        real_dns(["127.0.0.1"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://attacker.example/")

    def test_hostname_resolving_to_link_local_metadata_rejected(self, real_dns):
        real_dns(["169.254.169.254"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://attacker.example/latest/meta-data/")

    def test_hostname_resolving_to_private_rejected(self, real_dns):
        real_dns(["10.1.2.3"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("https://attacker.example/")

    def test_hostname_resolving_to_ipv6_loopback_rejected(self, real_dns):
        real_dns(["::1"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://attacker.example/")

    def test_hostname_resolving_to_ipv6_unique_local_rejected(self, real_dns):
        real_dns(["fd00::1"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://attacker.example/")

    def test_scoped_ipv6_link_local_answer_rejected(self, real_dns):
        # Zone suffix ("%eth0") must be stripped, then still rejected.
        real_dns(["fe80::1%eth0"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://attacker.example/")

    def test_mixed_public_and_private_answers_rejected(self, real_dns):
        # Every resolved address must pass — one bad record poisons the set.
        real_dns(["93.184.216.34", "192.168.1.5"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url("http://attacker.example/")

    def test_hostname_resolving_to_public_allowed(self, real_dns):
        real_dns(["93.184.216.34"])
        url = security.validate_http_url("https://example.com/page")
        assert url == "https://example.com/page"

    def test_hostname_resolving_to_public_ipv6_allowed(self, real_dns):
        real_dns(["2606:2800:220:1:248:1893:25c8:1946"])
        url = security.validate_http_url("https://example.com/")
        assert "example.com" in url

    def test_dns_failure_rejected(self, real_dns):
        real_dns(error=socket.gaierror(-2, "Name or service not known"))
        with pytest.raises(ValueError, match="resolve"):
            security.validate_http_url("http://no-such-host.example/")

    def test_empty_resolution_rejected(self, real_dns):
        real_dns([])
        with pytest.raises(ValueError, match="resolve"):
            security.validate_http_url("http://no-such-host.example/")

    def test_allow_loopback_honored_for_resolved_loopback(self, real_dns):
        real_dns(["127.0.0.1"])
        url = security.validate_http_url(
            "http://my-ollama.example:11434/v1", allow_loopback=True,
        )
        assert "my-ollama.example" in url

    def test_allow_loopback_does_not_allow_resolved_private(self, real_dns):
        real_dns(["10.0.0.5"])
        with pytest.raises(ValueError, match="blocked"):
            security.validate_http_url(
                "http://attacker.example/", allow_loopback=True,
            )

    def test_env_optin_honored_without_resolution(self, real_dns, monkeypatch):
        # With the trust opt-in every address category is allowed, so
        # resolution is skipped entirely — even a broken resolver passes.
        monkeypatch.setenv("OPENROOK_TEST_ALLOW_PRIVATE", "1")
        real_dns(error=socket.gaierror(-2, "resolver down"))
        url = security.validate_http_url(
            "http://internal-service.example/",
            allow_private_env="OPENROOK_TEST_ALLOW_PRIVATE",
        )
        assert url == "http://internal-service.example/"

    def test_localhost_short_circuits_without_dns(self, real_dns):
        # The ollama startup path: localhost + allow_loopback must work
        # even when DNS is unavailable.
        real_dns(error=socket.gaierror(-2, "resolver down"))
        url = security.validate_http_url(
            "http://localhost:11434/v1", allow_loopback=True,
        )
        assert url == "http://localhost:11434/v1"


# ===================================================================
# Callers: LLM backend constructors keep working for local hosts
# ===================================================================

class TestBackendCallers:
    def test_openai_compatible_backend_accepts_localhost_without_dns(self, real_dns):
        from runtime.llm.backend import OpenAICompatibleLLMBackend

        real_dns(error=socket.gaierror(-2, "resolver down"))
        backend = OpenAICompatibleLLMBackend(
            base_url="http://localhost:11434/v1", model="llama3",
        )
        assert backend._base_url == "http://localhost:11434/v1"

    def test_openai_compatible_backend_accepts_literal_loopback(self, real_dns):
        from runtime.llm.backend import OpenAICompatibleLLMBackend

        real_dns(error=socket.gaierror(-2, "resolver down"))
        backend = OpenAICompatibleLLMBackend(
            base_url="http://127.0.0.1:11434/v1", model="llama3",
        )
        assert backend._base_url == "http://127.0.0.1:11434/v1"

    def test_provider_client_rejects_hostname_resolving_to_private(self, real_dns):
        from core.provider_client import FastChatCompletionsClient

        real_dns(["192.168.0.10"])
        with pytest.raises(ValueError, match="blocked"):
            FastChatCompletionsClient(
                api_key="k", base_url="https://evil-gateway.example/v1", model="chosen-model",
            )

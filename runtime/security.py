"""Small security helpers shared by runtime-facing HTTP surfaces."""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_loopback_host(host: str) -> bool:
    """Return true for hosts that bind/connect only to the local machine."""
    value = (host or "").strip().strip("[]").lower()
    if value in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def public_bind_requires_auth(host: str, api_key: str) -> bool:
    """True when a bind host can be reached off-machine and auth is empty."""
    return not is_loopback_host(host) and not bool(api_key)


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for address categories the SSRF guard rejects by default."""
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_hostname(host: str, port: int) -> list[str]:
    """Resolve ``host`` to every A/AAAA address string, or raise ValueError.

    Uses the system resolver via ``socket.getaddrinfo``. There is no
    per-call timeout knob on getaddrinfo; we deliberately rely on the
    system resolver's own timeout (typically a few seconds per configured
    nameserver) rather than installing global socket timeouts or threads —
    the simplest correct option for a synchronous validation helper.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:  # gaierror is a subclass of OSError
        raise ValueError(f"could not resolve URL host {host!r}: {exc}") from exc
    # sockaddr is (addr, port[, flowinfo, scope_id]); strip any IPv6 zone
    # suffix ("fe80::1%eth0") so ipaddress can parse the literal.
    addresses = [info[4][0].split("%", 1)[0] for info in infos]
    if not addresses:
        raise ValueError(f"could not resolve URL host {host!r}: no addresses")
    return addresses


def validate_http_url(
    url: str,
    *,
    allow_loopback: bool = False,
    allow_private_env: str = "",
) -> str:
    """Validate an outbound HTTP(S) URL before server-side requests.

    The guard blocks SSRF targets by default: local hostnames, literal
    private/link-local/reserved IPs, and — for non-literal hostnames —
    any DNS resolution (A and AAAA) that lands on such an address. Every
    resolved address must pass the policy; a single private/loopback record
    rejects the URL, and a hostname that fails to resolve is rejected
    (an unresolvable target cannot be validated). Operators can opt into
    private address targets for trusted deployments by setting the
    environment variable named by ``allow_private_env``; that opt-in
    permits every address category, so DNS resolution is skipped for it.
    ``allow_loopback`` permits loopback targets only (``localhost``,
    127.0.0.0/8, ``::1``), e.g. for local LLM backends such as ollama.

    Known limitation (TOCTOU / DNS rebinding): the hostname is resolved at
    validation time, while the subsequent HTTP request resolves it again at
    connect time. An attacker-controlled nameserver with a short TTL can
    answer with a public address here and a private one for the actual
    request. Closing that window requires pinning the validated IP into the
    transport (a custom requests adapter that connects to the IP while
    preserving the Host header and TLS SNI); requests does not make that
    easy, so this guard documents the residual risk instead of
    half-implementing pinning. Defense in depth (no metadata-service
    credentials, egress firewalling) should back this check.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must use http or https")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    host = parsed.hostname.strip("[]").lower()
    if host == "localhost":
        if not allow_loopback:
            raise ValueError("URL host is local-only")
        return parsed.geturl()

    private_allowed = bool(allow_private_env and os.environ.get(allow_private_env))
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        if ip.is_loopback and allow_loopback:
            return parsed.geturl()
        if _is_blocked_ip(ip) and not private_allowed:
            raise ValueError("URL host resolves to a blocked local or private address")
        return parsed.geturl()

    # Non-literal hostname: with the private opt-in every address category
    # is permitted anyway, so resolution would add nothing but latency.
    if private_allowed:
        return parsed.geturl()

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for address in _resolve_hostname(host, port):
        try:
            resolved = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError(
                f"could not validate resolved address {address!r} for host {host!r}"
            ) from exc
        if resolved.is_loopback and allow_loopback:
            continue
        if _is_blocked_ip(resolved):
            raise ValueError(
                "URL host resolves to a blocked local or private address"
            )
    return parsed.geturl()

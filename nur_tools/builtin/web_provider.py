"""Requests-based web provider for search, fetch, and text extraction.

Uses requests (already a project dependency) with DuckDuckGo HTML
for search and basic HTML stripping for text extraction.
No additional dependencies required.
"""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests

from runtime.security import validate_http_url


_DEFAULT_TIMEOUT = 15.0
_USER_AGENT = "Mozilla/5.0 (compatible; OpenRook/1.0)"
_MAX_FETCH_BYTES = 2_000_000  # 2 MB cap on fetched content
_MAX_REDIRECT_HOPS = 5  # manual redirect cap — every hop is re-validated
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_SEARCH_URL = "https://html.duckduckgo.com/html/"


class RequestsWebProvider:
    """Production web provider using requests + DuckDuckGo HTML search."""

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT
        self._timeout = timeout

    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        """Search via DuckDuckGo HTML lite endpoint.

        Redirects are never followed automatically: each hop is validated
        with ``validate_http_url`` before it is requested (see ``fetch``).
        """
        with self._session.post(
            _SEARCH_URL,
            data={"q": query},
            timeout=self._timeout,
            allow_redirects=False,
        ) as resp:
            if resp.status_code in _REDIRECT_STATUSES:
                next_url = self._next_redirect_target(_SEARCH_URL, resp)
            else:
                resp.raise_for_status()
                return _parse_ddg_results(resp.text, limit)

        # Follow redirect hops with GET (requests/browser semantics for a
        # redirected POST), validating each target before requesting it.
        for _ in range(_MAX_REDIRECT_HOPS):
            with self._session.get(
                next_url, timeout=self._timeout, allow_redirects=False,
            ) as resp:
                if resp.status_code in _REDIRECT_STATUSES:
                    next_url = self._next_redirect_target(next_url, resp)
                    continue
                resp.raise_for_status()
                return _parse_ddg_results(resp.text, limit)
        raise ValueError(f"Too many redirects (more than {_MAX_REDIRECT_HOPS})")

    def fetch(self, url: str) -> str:
        """Fetch raw HTML/text content from a URL.

        Honors ``_MAX_FETCH_BYTES`` by streaming chunks and stopping once
        the cap is reached — ``resp.content`` would silently ignore the cap
        and buffer the entire body first.

        Redirects are followed manually (``allow_redirects=False``): each
        hop's target URL is validated with ``validate_http_url`` before it
        is requested, so a public URL cannot bounce the fetch into loopback,
        link-local (cloud metadata), or private address space. The chain is
        capped at ``_MAX_REDIRECT_HOPS``; the streaming byte cap and timeout
        apply to every hop.
        """
        safe_url = self._validate(url)
        for _ in range(_MAX_REDIRECT_HOPS + 1):
            with self._session.get(
                safe_url, timeout=self._timeout, stream=True,
                allow_redirects=False,
            ) as resp:
                if resp.status_code in _REDIRECT_STATUSES:
                    safe_url = self._next_redirect_target(safe_url, resp)
                    continue
                resp.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=65_536):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= _MAX_FETCH_BYTES:
                        break
                content = b"".join(chunks)[:_MAX_FETCH_BYTES]
                encoding = resp.encoding or "utf-8"
            return content.decode(encoding, errors="replace")
        raise ValueError(f"Too many redirects (more than {_MAX_REDIRECT_HOPS})")

    @staticmethod
    def _validate(url: str) -> str:
        return validate_http_url(
            url,
            allow_loopback=False,
            allow_private_env="OPENROOK_ALLOW_PRIVATE_WEB_FETCH",
        )

    def _next_redirect_target(self, current_url: str, resp: requests.Response) -> str:
        """Resolve and validate the Location header of a redirect response."""
        location = resp.headers.get("Location") or ""
        if not location:
            raise ValueError("Redirect response missing Location header")
        return self._validate(urljoin(current_url, location))

    def extract_text(self, url: str) -> str:
        """Fetch a URL and return cleaned body text."""
        html = self.fetch(url)
        return _html_to_text(html)

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._session.close()


def _parse_ddg_results(html: str, limit: int) -> list[dict[str, str]]:
    """Extract search results from DuckDuckGo HTML response."""
    results: list[dict[str, str]] = []
    # DuckDuckGo HTML lite uses <a class="result__a"> for result links
    matches = list(re.finditer(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    ))
    for idx, m in enumerate(matches):
        url = _normalize_ddg_url(m.group(1).strip())
        title = _html_fragment_to_text(m.group(2))
        if url and title:
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(html)
            snippet = _extract_ddg_snippet(html[m.end():next_start])
            result = {"title": title, "url": url}
            if snippet:
                result["snippet"] = snippet
            results.append(result)
        if len(results) >= limit:
            break

    # Fallback: try <a rel="nofollow"> pattern (alternative DDG format)
    if not results:
        for m in re.finditer(
            r'<a[^>]+rel="nofollow"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        ):
            url = _normalize_ddg_url(m.group(1).strip())
            title = _html_fragment_to_text(m.group(2))
            if url and title and url.startswith("http"):
                results.append({"title": title, "url": url})
            if len(results) >= limit:
                break
    return results


def _normalize_ddg_url(url: str) -> str:
    """Turn DuckDuckGo redirect links into the target URL when possible."""
    decoded = unescape(url).strip()
    if decoded.startswith("//"):
        decoded = "https:" + decoded
    elif decoded.startswith("/"):
        decoded = "https://duckduckgo.com" + decoded

    parsed = urlparse(decoded)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return decoded


def _extract_ddg_snippet(segment: str) -> str:
    match = re.search(
        r'<[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</',
        segment,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    return _html_fragment_to_text(match.group(1))


def _html_fragment_to_text(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text. Basic but sufficient for cognitive use."""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Replace block-level tags with newlines
    text = re.sub(r"<(?:p|div|br|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode entities
    text = unescape(text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

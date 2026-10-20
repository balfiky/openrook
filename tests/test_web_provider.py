"""Tests for tools/builtin/web_provider.py — the requests-backed provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openrook_tools.builtin.web_provider import (
    RequestsWebProvider,
    _MAX_FETCH_BYTES,
    _MAX_REDIRECT_HOPS,
    _html_to_text,
    _parse_ddg_results,
)


class TestRequestsWebProviderFetch:
    def _streaming_response(self, chunks: list[bytes], encoding: str = "utf-8") -> MagicMock:
        """Build a mock Response that yields the given chunks via iter_content."""
        resp = MagicMock()
        resp.iter_content.return_value = iter(chunks)
        resp.encoding = encoding
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_fetch_honors_max_fetch_bytes(self):
        """Regression: prior implementation used ``resp.content[:CAP]`` which
        silently buffered the entire body and ignored the cap. ``fetch()`` must
        stop streaming once the cap is reached."""
        # Three chunks of 1MB each — a 3MB body while the cap is 2MB.
        big_chunks = [b"A" * 1_000_000, b"B" * 1_000_000, b"C" * 1_000_000]
        provider = RequestsWebProvider()
        fake_resp = self._streaming_response(big_chunks)

        with patch.object(provider._session, "get", return_value=fake_resp) as mock_get:
            text = provider.fetch("https://example.invalid/big")

        mock_get.assert_called_once()
        assert len(text) <= _MAX_FETCH_BYTES
        # The body must be truncated, not fully buffered.
        assert len(text) == _MAX_FETCH_BYTES
        assert text.startswith("A")

    def test_fetch_short_body_returned_whole(self):
        provider = RequestsWebProvider()
        fake_resp = self._streaming_response([b"hello world"])
        with patch.object(provider._session, "get", return_value=fake_resp):
            text = provider.fetch("https://example.invalid/small")
        assert text == "hello world"

    def test_fetch_uses_context_manager(self):
        """The streamed response must be wrapped in a context manager so the
        connection is returned to the pool even on error paths."""
        provider = RequestsWebProvider()
        fake_resp = self._streaming_response([b"ok"])
        with patch.object(provider._session, "get", return_value=fake_resp):
            provider.fetch("https://example.invalid/")
        fake_resp.__enter__.assert_called_once()
        fake_resp.__exit__.assert_called_once()

    def test_fetch_blocks_localhost_url(self):
        provider = RequestsWebProvider()
        with patch.object(provider._session, "get") as mock_get:
            try:
                provider.fetch("http://127.0.0.1:8080/private")
            except ValueError as exc:
                assert "blocked" in str(exc) or "local" in str(exc)
            else:
                raise AssertionError("localhost fetch was not blocked")
        mock_get.assert_not_called()

    def test_close_releases_session(self):
        provider = RequestsWebProvider()
        with patch.object(provider._session, "close") as mock_close:
            provider.close()
            mock_close.assert_called_once()


def _redirect_response(location: str | None, status: int = 302) -> MagicMock:
    """Build a mock redirect Response with the given Location header."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"Location": location} if location is not None else {}
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestRequestsWebProviderRedirects:
    """Redirects must be followed manually with every hop re-validated.

    The suite conftest stubs DNS so fake public hostnames resolve to a
    public address; private/loopback targets here are literal IPs.
    """

    def _streaming_response(self, chunks: list[bytes]) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_content.return_value = iter(chunks)
        resp.encoding = "utf-8"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_fetch_follows_single_redirect_to_public_target(self):
        provider = RequestsWebProvider()
        hops = [
            _redirect_response("https://other.example/page"),
            self._streaming_response([b"final body"]),
        ]
        with patch.object(provider._session, "get", side_effect=hops) as mock_get:
            text = provider.fetch("https://start.example/")
        assert text == "final body"
        assert mock_get.call_count == 2
        # Every request must disable automatic redirects.
        for call in mock_get.call_args_list:
            assert call.kwargs["allow_redirects"] is False
        assert mock_get.call_args_list[1].args[0] == "https://other.example/page"

    def test_fetch_resolves_relative_redirect_against_current_url(self):
        provider = RequestsWebProvider()
        hops = [
            _redirect_response("/next/page"),
            self._streaming_response([b"ok"]),
        ]
        with patch.object(provider._session, "get", side_effect=hops) as mock_get:
            provider.fetch("https://start.example/dir/here")
        assert mock_get.call_args_list[1].args[0] == "https://start.example/next/page"

    def test_fetch_redirect_to_metadata_ip_rejected(self):
        provider = RequestsWebProvider()
        hops = [_redirect_response("http://169.254.169.254/latest/meta-data/")]
        with patch.object(provider._session, "get", side_effect=hops) as mock_get:
            with pytest.raises(ValueError, match="blocked"):
                provider.fetch("https://start.example/")
        # The poisoned hop must never be requested.
        assert mock_get.call_count == 1

    def test_fetch_redirect_to_loopback_rejected(self):
        provider = RequestsWebProvider()
        hops = [_redirect_response("http://127.0.0.1:8080/internal")]
        with patch.object(provider._session, "get", side_effect=hops) as mock_get:
            with pytest.raises(ValueError, match="blocked"):
                provider.fetch("https://start.example/")
        assert mock_get.call_count == 1

    def test_fetch_redirect_chain_over_cap_rejected(self):
        provider = RequestsWebProvider()
        hops = [
            _redirect_response(f"https://bounce.example/{i}")
            for i in range(_MAX_REDIRECT_HOPS + 2)
        ]
        with patch.object(provider._session, "get", side_effect=hops) as mock_get:
            with pytest.raises(ValueError, match="[Tt]oo many redirects"):
                provider.fetch("https://start.example/")
        assert mock_get.call_count == _MAX_REDIRECT_HOPS + 1

    def test_fetch_redirect_missing_location_rejected(self):
        provider = RequestsWebProvider()
        hops = [_redirect_response(None)]
        with patch.object(provider._session, "get", side_effect=hops):
            with pytest.raises(ValueError, match="Location"):
                provider.fetch("https://start.example/")

    def test_search_redirect_followed_with_get(self):
        provider = RequestsWebProvider()
        redirect = _redirect_response("https://html.duckduckgo.com/html2/")
        final = MagicMock()
        final.status_code = 200
        final.text = '<a class="result__a" href="https://example.com/a">Alpha</a>'
        final.__enter__ = MagicMock(return_value=final)
        final.__exit__ = MagicMock(return_value=False)
        with patch.object(provider._session, "post", return_value=redirect) as mock_post:
            with patch.object(provider._session, "get", return_value=final) as mock_get:
                results = provider.search("anything", limit=5)
        assert results == [{"title": "Alpha", "url": "https://example.com/a"}]
        assert mock_post.call_args.kwargs["allow_redirects"] is False
        assert mock_get.call_args.args[0] == "https://html.duckduckgo.com/html2/"

    def test_search_redirect_to_private_rejected(self):
        provider = RequestsWebProvider()
        redirect = _redirect_response("http://10.0.0.5/")
        with patch.object(provider._session, "post", return_value=redirect):
            with patch.object(provider._session, "get") as mock_get:
                with pytest.raises(ValueError, match="blocked"):
                    provider.search("anything", limit=5)
        mock_get.assert_not_called()


class TestRequestsWebProviderSearch:
    def test_search_parses_ddg_html(self):
        provider = RequestsWebProvider()
        html = (
            '<a class="result__a" href="https://example.com/a">Alpha</a>'
            '<a class="result__a" href="https://example.com/b">Beta</a>'
        )
        fake_resp = MagicMock()
        fake_resp.text = html
        fake_resp.raise_for_status = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(provider._session, "post", return_value=fake_resp):
            results = provider.search("anything", limit=5)

        assert results == [
            {"title": "Alpha", "url": "https://example.com/a"},
            {"title": "Beta", "url": "https://example.com/b"},
        ]


class TestHTMLParsing:
    def test_strip_scripts_and_styles(self):
        html = "<html><script>evil()</script><p>Hello</p><style>x{}</style></html>"
        text = _html_to_text(html)
        assert "Hello" in text
        assert "evil" not in text
        assert "x{}" not in text

    def test_decodes_entities(self):
        assert _html_to_text("<p>A &amp; B</p>") == "A & B"

    def test_ddg_fallback_pattern(self):
        """If result__a is missing, the nofollow fallback should catch links."""
        html = (
            '<a rel="nofollow" href="https://example.com/x">Ex</a>'
            '<a rel="nofollow" href="/internal">Skip</a>'
        )
        results = _parse_ddg_results(html, limit=5)
        assert {"title": "Ex", "url": "https://example.com/x"} in results
        # Relative URL should be filtered because fallback requires http prefix.
        assert all(r["url"].startswith("http") for r in results)

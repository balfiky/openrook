from __future__ import annotations

import re

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from interface import api as interface_api
from runtime.config import RuntimeConfig


class TestAdminConsoleStatic:
    def test_settings_route_serves_full_console(self):
        resp = interface_api.settings_index()
        html = resp.body.decode()

        assert resp.status_code == 200
        assert "OpenRook Settings" in html
        assert 'href="/admin/assets/admin.css"' in html
        assert 'src="/admin/assets/admin.js"' in html
        assert "Instance Workspace" in html
        assert "Persona Overview" in html
        assert 'id="overviewPersona"' in html
        assert 'data-page="persona"' not in html
        assert 'id="page-persona"' not in html
        assert 'data-page="observability"' in html
        assert 'id="page-observability"' in html
        assert 'id="personaAdminPage"' in html
        assert "Settings" in html
        assert 'id="page-settings"' in html
        assert 'class="settings-accordion"' in html
        assert "Skills" in html
        assert "Life History" in html
        assert "Evolution Snapshot" in html
        assert "Tool Inventory" in html
        assert 'id="refreshToolsBtn"' in html
        assert 'id="toolCategoryFilter"' in html
        assert 'id="importSkillBtn"' in html
        assert 'id="skillUpload"' in html
        assert 'id="skillsList"' in html
        assert 'id="lifeUploadFile"' in html
        assert 'id="lifeSnapshotNarrative"' in html
        assert 'id="lifeDriveDrift"' in html
        assert 'id="lifeTimeline"' in html
        assert 'id="lifeExperiences"' in html
        assert 'id="rollbackLifeBatchBtn"' not in html
        assert "Jump to Observability" in html

    def test_legacy_routes_redirect_to_settings_workspace(self):
        admin = interface_api.admin_index()
        resp = interface_api.persona_index()
        alias = interface_api.dashboard_index()

        assert admin.status_code == 307
        assert resp.status_code == 307
        assert alias.status_code == 307
        assert admin.headers["location"] == "/settings"
        assert resp.headers["location"] == "/settings#observability"
        assert alias.headers["location"] == "/settings#observability"

    def test_admin_assets_are_whitelisted(self):
        css = interface_api.admin_asset("admin.css")
        js = interface_api.admin_asset("admin.js")
        persona_css = interface_api.admin_asset("persona.css")
        persona_js = interface_api.admin_asset("persona.js")

        assert css.status_code == 200
        assert b'@import url("/admin/assets/tokens.css")' in css.body
        assert b".tool-row" in css.body
        assert b".evolution-grid" in css.body
        assert b".metric-foot" in css.body
        assert b"overflow-wrap: anywhere" in css.body
        assert b"prefers-reduced-motion" in css.body
        assert js.status_code == 200
        assert b"fieldSections" in js.body
        assert b"/admin/persona/state" in js.body
        assert b"renderOverviewPersona" in js.body
        assert b"renderPersonaAdminPage" in js.body
        assert b"refreshPersonaBtn" in js.body
        assert b"/admin/codex/models" in js.body
        assert b"renderCodexModelSelect" in js.body
        assert b"/v1/tools" in js.body
        assert b"/admin/skills" in js.body
        assert b"/admin/skills/import/upload" in js.body
        assert b"data-skill-action=\"delete\"" in js.body
        assert b"/admin/backups" in js.body
        assert b"/admin/life" in js.body
        assert b"/admin/life/experiences/upload" in js.body
        assert b"/admin/life/rollback" not in js.body
        assert b"renderLifeSnapshot" in js.body
        assert persona_css.status_code == 200
        assert b".dashboard-grid" in persona_css.body
        assert persona_js.status_code == 200
        assert b"/admin/persona/state" in persona_js.body
        assert b"renderPersona" in persona_js.body
        assert b"renderObservability" in persona_js.body
        assert b"renderStateRadar" in persona_js.body

    def test_unknown_admin_asset_404s(self):
        with pytest.raises(HTTPException) as exc:
            interface_api.admin_asset("index.html")

        assert exc.value.status_code == 404


class TestHtmlSecurityHeaders:
    """CSP hardening — script-src has no 'unsafe-inline' and no served HTML
    relies on inline scripts or inline event handlers that it would block."""

    _INLINE_HANDLER = re.compile(
        r"\son(click|load|error|change|submit|input|keydown|focus|blur)\s*=",
        re.IGNORECASE,
    )
    # <script> without a src= attribute (i.e. an inline script block).
    _INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>", re.IGNORECASE)

    def _client(self, monkeypatch, tmp_path):
        path = tmp_path / "runtime_config.yaml"
        RuntimeConfig(data_dir=str(tmp_path / "data")).write_yaml(str(path))
        monkeypatch.setattr(interface_api, "RUNTIME_CONFIG_PATH", str(path))
        return TestClient(interface_api.app)

    def test_csp_script_src_has_no_unsafe_inline(self, monkeypatch, tmp_path):
        csp = interface_api._HTML_SECURITY_HEADERS["Content-Security-Policy"]
        script_src = next(
            d for d in csp.split(";") if d.strip().startswith("script-src")
        )
        assert "'unsafe-inline'" not in script_src
        assert "'self'" in script_src

        with self._client(monkeypatch, tmp_path) as client:
            for route in ("/", "/settings", "/persona"):
                resp = client.get(route, follow_redirects=True)
                assert resp.status_code == 200, route
                assert (
                    resp.headers["Content-Security-Policy"] == csp
                ), route

    def test_served_html_needs_no_inline_script_allowance(
        self, monkeypatch, tmp_path
    ):
        with self._client(monkeypatch, tmp_path) as client:
            for route in ("/", "/settings", "/persona"):
                html = client.get(route, follow_redirects=True).text
                assert not self._INLINE_SCRIPT.search(html), (
                    f"{route} serves an inline <script> block that the CSP "
                    "(script-src 'self') would block"
                )
                assert not self._INLINE_HANDLER.search(html), (
                    f"{route} uses an inline event handler that the CSP "
                    "(script-src 'self') would block"
                )

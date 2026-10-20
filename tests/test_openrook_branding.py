"""Public naming, configuration, and packaging contracts for OpenRook."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from config.environment import project_env
from config.loader import load_config
from interface import api, setup, uninstall
from interface.client import OpenRookAPIError
from interface.routes_backup import (
    _admin_redacted_config_export,
    _create_admin_backup,
    _delete_admin_backup,
    _list_admin_backups,
    _resolve_admin_backup_path,
)
from main import _parse_args as parse_runtime_args
from openrook_tools import uat, validate
from runtime.config import RuntimeConfig
from runtime.security import validate_http_url


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_project_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("OPENROOK_"):
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, "fallback"),
        ({"OPENROOK_CONFIG_DIR": "new"}, "new"),
        ({"OPENROOK_CONFIG_DIR": ""}, ""),
    ],
)
def test_environment_values_and_default(environment, expected):
    assert project_env("CONFIG_DIR", "fallback", environ=environment) == expected


def test_api_error_uses_current_name():
    assert "OpenRook API error" in str(OpenRookAPIError(503, "unavailable"))


def test_existing_custom_identity_is_loaded_and_not_renamed(tmp_path, monkeypatch):
    soul_path = tmp_path / "soul.yaml"
    original = "soul:\n  name: Custom Companion\n  identity: A name chosen by the user.\n"
    soul_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("OPENROOK_CONFIG_DIR", str(tmp_path))

    assert load_config().soul.name == "Custom Companion"
    assert setup._soul_yaml_path() == soul_path
    assert Path(api._soul_yaml_path()) == soul_path
    assert soul_path.read_text(encoding="utf-8") == original


def test_new_identity_and_openapi_use_openrook(tmp_path):
    assert load_config(tmp_path).soul.name == "OpenRook"
    assert api.app.title == "OpenRook"
    assert api.app.openapi()["info"]["title"] == "OpenRook"


@pytest.mark.parametrize(
    "parser", [parse_runtime_args, api._parse_web_args, setup._parse_args, uninstall._parse_args]
)
def test_runtime_config_environment_and_explicit_cli_precedence(monkeypatch, parser):
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", "existing.yaml")
    assert parser([]).config == "existing.yaml"
    assert parser(["--config", "chosen.yaml"]).config == "chosen.yaml"


def test_uat_backend_environment(monkeypatch, request):
    selected_model = request.node.name
    monkeypatch.setenv("OPENROOK_UAT_BACKEND", "openai_compatible")
    monkeypatch.setenv("OPENROOK_UAT_MODEL", selected_model)
    args = uat._parse_args([])
    assert args.backend == "openai_compatible"
    assert args.model == selected_model


def test_private_url_opt_in_respects_explicit_empty(monkeypatch):
    monkeypatch.setenv("OPENROOK_ALLOW_PRIVATE_WEB_FETCH", "1")
    url = "https://10.0.0.8/example"
    assert validate_http_url(url, allow_private_env="OPENROOK_ALLOW_PRIVATE_WEB_FETCH") == url

    monkeypatch.setenv("OPENROOK_ALLOW_PRIVATE_WEB_FETCH", "")
    with pytest.raises(ValueError, match="blocked"):
        validate_http_url(url, allow_private_env="OPENROOK_ALLOW_PRIVATE_WEB_FETCH")


def test_database_and_export_use_current_names(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    assert Path(config.user_db_path("web:alice")) == tmp_path / "web_alice" / "openrook.db"
    assert _admin_redacted_config_export(config)["format"] == "openrook.runtime_config.redacted.v1"


def test_created_backups_can_be_listed_resolved_and_deleted(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=str(tmp_path))
    monkeypatch.setattr(api, "RUNTIME_CONFIG_PATH", str(tmp_path / "runtime_config.yaml"))
    created = _create_admin_backup(config, include_data=False)
    assert created["filename"].startswith("openrook-backup-")
    listed = {item["filename"] for item in _list_admin_backups(config)["backups"]}
    assert listed == {created["filename"]}
    assert Path(
        _resolve_admin_backup_path(config, created["filename"], must_exist=True)
    ) == Path(created["path"])
    result = _delete_admin_backup(
        config, filename=created["filename"], confirmation=f"DELETE {created['filename']}"
    )
    assert result["ok"] is True
    assert not Path(created["path"]).exists()


@pytest.mark.parametrize(
    "filename", ["../openrook-backup-test.zip", "other.zip"]
)
def test_backup_path_validation(tmp_path, filename):
    with pytest.raises(HTTPException) as error:
        _resolve_admin_backup_path(
            RuntimeConfig(data_dir=str(tmp_path)), filename, must_exist=False
        )
    assert error.value.status_code == 400


def test_distribution_and_all_commands_are_declared():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '\nname = "openrook"\n' in metadata
    for suffix in (
        "",
        "-setup",
        "-web",
        "-uninstall",
        "-genesis-reset",
        "-emotional-qa",
        "-uat",
        "-validate",
    ):
        canonical = f"openrook{suffix}"
        target = validate.EXPECTED_CONSOLE_SCRIPTS[canonical]
        assert f'{canonical} = "{target}"' in metadata


def test_all_declared_cli_targets_resolve():
    for target in validate.EXPECTED_CONSOLE_SCRIPTS.values():
        module_name, function_name = target.split(":")
        assert callable(getattr(importlib.import_module(module_name), function_name))


@pytest.mark.parametrize("filename", ["index.html", "admin.html", "persona.html"])
def test_public_html_has_current_brand(filename):
    content = (ROOT / "interface" / "static" / filename).read_text(encoding="utf-8")
    assert "OpenRook" in content


def test_browser_identity_and_token_keys_use_current_names():
    static = ROOT / "interface" / "static"
    chat = (static / "chat.js").read_text(encoding="utf-8")
    for key in ("openrook_user_id", "openrook_chat_id", "openrook_wizard_dismissed"):
        assert key in chat
    for filename in ("chat.js", "admin.js", "persona.js", "lib/shared.js"):
        content = (static / filename).read_text(encoding="utf-8")
        assert "openrook_api_token" in content

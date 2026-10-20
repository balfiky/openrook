"""Public naming and compatibility across the OpenRook rename."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from config import NurConfig, OpenRookConfig
from config.environment import project_env
from config.loader import load_config
from interface import api, setup, uninstall
from interface.client import NurAPIError, NurClient, OpenRookAPIError, OpenRookClient
from interface.routes_backup import (
    _admin_redacted_config_export,
    _create_admin_backup,
    _delete_admin_backup,
    _list_admin_backups,
    _resolve_admin_backup_path,
)
from main import _parse_args as parse_runtime_args
from nur_tools import uat, validate
from runtime.app import NurApp, OpenRookApp
from runtime.config import RuntimeConfig
from runtime.security import validate_http_url


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_project_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("OPENROOK_", "NUR_")):
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, "fallback"),
        ({"NUR_CONFIG_DIR": "old"}, "old"),
        ({"OPENROOK_CONFIG_DIR": "new"}, "new"),
        ({"NUR_CONFIG_DIR": "old", "OPENROOK_CONFIG_DIR": "new"}, "new"),
        ({"NUR_CONFIG_DIR": "old", "OPENROOK_CONFIG_DIR": ""}, ""),
    ],
)
def test_environment_alias_precedence(environment, expected):
    assert project_env("CONFIG_DIR", "fallback", environ=environment) == expected


def test_public_classes_keep_legacy_aliases():
    assert NurConfig is OpenRookConfig
    assert NurApp is OpenRookApp
    assert NurClient is OpenRookClient
    assert NurAPIError is OpenRookAPIError
    assert "OpenRook API error" in str(OpenRookAPIError(503, "unavailable"))


@pytest.mark.parametrize("prefix", ["OPENROOK", "NUR"])
def test_existing_custom_identity_is_loaded_and_not_renamed(tmp_path, monkeypatch, prefix):
    soul_path = tmp_path / "soul.yaml"
    original = "soul:\n  name: Nūr\n  identity: A name chosen before the rename.\n"
    soul_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv(f"{prefix}_CONFIG_DIR", str(tmp_path))

    assert load_config().soul.name == "Nūr"
    assert setup._soul_yaml_path() == soul_path
    assert Path(api._soul_yaml_path()) == soul_path
    assert soul_path.read_text(encoding="utf-8") == original


def test_new_identity_and_openapi_use_openrook(tmp_path):
    assert load_config(tmp_path).soul.name == "OpenRook"
    assert api.app.title == "OpenRook"
    assert api.app.openapi()["info"]["title"] == "OpenRook"


@pytest.mark.parametrize("prefix", ["OPENROOK", "NUR"])
@pytest.mark.parametrize(
    "parser", [parse_runtime_args, api._parse_web_args, setup._parse_args, uninstall._parse_args]
)
def test_runtime_config_environment_and_explicit_cli_precedence(monkeypatch, prefix, parser):
    monkeypatch.setenv(f"{prefix}_RUNTIME_CONFIG", "existing.yaml")
    assert parser([]).config == "existing.yaml"
    assert parser(["--config", "chosen.yaml"]).config == "chosen.yaml"


@pytest.mark.parametrize("prefix", ["OPENROOK", "NUR"])
def test_uat_backend_environment_alias(monkeypatch, prefix):
    monkeypatch.setenv(f"{prefix}_UAT_BACKEND", "openai_compatible")
    monkeypatch.setenv(f"{prefix}_UAT_MODEL", "example-model")
    args = uat._parse_args([])
    assert args.backend == "openai_compatible"
    assert args.model == "example-model"


@pytest.mark.parametrize("prefix", ["OPENROOK", "NUR"])
def test_private_url_opt_in_keeps_alias_without_overriding_explicit_empty(monkeypatch, prefix):
    monkeypatch.setenv(f"{prefix}_ALLOW_PRIVATE_WEB_FETCH", "1")
    url = "https://10.0.0.8/example"
    assert validate_http_url(url, allow_private_env="OPENROOK_ALLOW_PRIVATE_WEB_FETCH") == url

    monkeypatch.setenv("OPENROOK_ALLOW_PRIVATE_WEB_FETCH", "")
    with pytest.raises(ValueError, match="blocked"):
        validate_http_url(url, allow_private_env="OPENROOK_ALLOW_PRIVATE_WEB_FETCH")


def test_database_path_stays_compatible(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    assert Path(config.user_db_path("web:alice")) == tmp_path / "web_alice" / "nur.db"
    assert _admin_redacted_config_export(config)["format"] == "nur.runtime_config.redacted.v1"


def test_old_and_new_backups_remain_accessible(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=str(tmp_path))
    monkeypatch.setattr(api, "RUNTIME_CONFIG_PATH", str(tmp_path / "runtime_config.yaml"))
    created = _create_admin_backup(config, include_data=False)
    assert created["filename"].startswith("openrook-backup-")
    legacy_name = "nur-backup-20260101T000000Z.zip"
    legacy_path = Path(created["path"]).with_name(legacy_name)
    legacy_path.write_bytes(Path(created["path"]).read_bytes())

    listed = {item["filename"] for item in _list_admin_backups(config)["backups"]}
    assert listed == {created["filename"], legacy_name}
    assert Path(_resolve_admin_backup_path(config, legacy_name, must_exist=True)) == legacy_path
    result = _delete_admin_backup(
        config, filename=legacy_name, confirmation=f"DELETE {legacy_name}"
    )
    assert result["ok"] is True
    assert not legacy_path.exists()
    assert Path(created["path"]).exists()


@pytest.mark.parametrize(
    "filename", ["../openrook-backup-test.zip", "../nur-backup-test.zip", "other.zip"]
)
def test_backup_aliases_do_not_relax_path_validation(tmp_path, filename):
    with pytest.raises(HTTPException) as error:
        _resolve_admin_backup_path(
            RuntimeConfig(data_dir=str(tmp_path)), filename, must_exist=False
        )
    assert error.value.status_code == 400


def test_distribution_and_all_command_aliases_are_declared():
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
        legacy = f"nur{suffix}"
        assert (
            validate.EXPECTED_CONSOLE_SCRIPTS[canonical]
            == validate.EXPECTED_CONSOLE_SCRIPTS[legacy]
        )
        for command in (canonical, legacy):
            target = validate.EXPECTED_CONSOLE_SCRIPTS[command]
            assert f'{command} = "{target}"' in metadata


@pytest.mark.parametrize("filename", ["index.html", "admin.html", "persona.html"])
def test_public_html_has_current_brand(filename):
    content = (ROOT / "interface" / "static" / filename).read_text(encoding="utf-8")
    assert "OpenRook" in content
    assert "Nūr" not in content


def test_browser_identity_and_token_keys_remain_compatible():
    static = ROOT / "interface" / "static"
    chat = (static / "chat.js").read_text(encoding="utf-8")
    for key in ("nur_user_id", "nur_chat_id", "nur_wizard_dismissed"):
        assert key in chat
    for filename in ("chat.js", "admin.js", "persona.js", "lib/shared.js"):
        content = (static / filename).read_text(encoding="utf-8")
        assert "nur_api_token" in content
        assert "openrook_api_token" not in content

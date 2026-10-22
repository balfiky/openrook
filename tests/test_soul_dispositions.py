"""Canonical soul preferences survive every owner configuration boundary."""
import asyncio
from dataclasses import asdict

import pytest
from pydantic import ValidationError
import yaml

from config.loader import SoulConfig, load_config, reset_config
from interface import api
from interface.routes_admin_identity import _admin_soul_payload, _parse_raw_soul_document, admin_update_soul
from runtime.research_store import restore_configuration


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_admin_rejects_invalid_decision_preferences(value):
    with pytest.raises(ValidationError):
        api.AdminSoulUpdateRequest(name="Character", dispositions={"pride": value})


def test_import_export_and_legacy_alias_have_one_canonical_weight():
    payload, _ = _parse_raw_soul_document("soul:\n  name: Character\n  initial_traits: {patient: 0.2, arbitrary: 0.4}\n")
    request = api.AdminSoulUpdateRequest.model_validate(payload)
    assert request.dispositions["patience"] == 0.2
    request = api.AdminSoulUpdateRequest(name="Character", initial_traits={"patient": 0.2}, dispositions={"patience": 0.9})
    assert request.dispositions["patience"] == 0.9
    soul = SoulConfig(**request.model_dump())
    assert _admin_soul_payload(soul)["soul"]["dispositions"] == soul.dispositions


def test_save_load_and_birth_restore_include_dispositions_and_seed(tmp_path, monkeypatch):
    path = tmp_path / "soul.yaml"
    monkeypatch.setenv("OPENROOK_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_soul_yaml_path", lambda: str(path))
    monkeypatch.setattr(api, "_detach_session_manager_for_reload", lambda **kwargs: (False, None))
    monkeypatch.setattr(api, "_refresh_live_channel_session_manager", lambda: None)
    try:
        request = api.AdminSoulUpdateRequest(name="Character", character_seed=73,
                                            dispositions={"pride":0.1, "forgiveness":0.9})
        result = asyncio.run(admin_update_soul(request))
        assert result["soul"]["character_seed"] == 73
        saved = yaml.safe_load(path.read_text())["soul"]
        assert saved["schema_version"] == 2 and saved["dispositions"] == request.dispositions
        config = load_config(tmp_path)
        restored = restore_configuration(asdict(config))
        assert restored.soul.character_seed == 73
        assert restored.soul.dispositions == config.soul.dispositions
        assert restored.soul.initial_traits == config.soul.initial_traits
    finally:
        reset_config()

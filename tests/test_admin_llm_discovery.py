"""Draft discovery keeps credentials scoped to the selected destination."""

from unittest.mock import AsyncMock, Mock

from httpx import ASGITransport, AsyncClient
import pytest

from interface import api as interface_api
from interface import routes_admin_config as admin
from runtime.config import RuntimeConfig


@pytest.fixture
def saved_config(monkeypatch):
    config = RuntimeConfig(
        llm_backend="provider", llm_base_url="https://provider.example/v1",
        llm_model="previously-chosen", llm_api_key="saved-secret",
    )
    monkeypatch.setattr(interface_api, "_load_runtime_config", lambda: config)

    async def inline_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    # Credential tests exercise the handler, independently of thread transport.
    monkeypatch.setattr(admin.asyncio, "to_thread", inline_thread)
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "expected_key"),
    [
        ({}, "saved-secret"),
        ({"llm_base_url": "https://provider.example/v1/"}, "saved-secret"),
        ({"llm_api_key": "draft-secret"}, "draft-secret"),
        ({"clear_llm_api_key": True}, ""),
        ({"llm_base_url": "https://different.example/v1"}, ""),
        ({"llm_backend": "openai_compatible"}, ""),
        ({"llm_base_url": "https://different.example/v1", "llm_api_key": "draft-secret"},
         "draft-secret"),
    ],
)
async def test_admin_discovery_scopes_keys_and_does_not_save(
    saved_config, monkeypatch, draft, expected_key,
):
    discover = Mock(return_value={"ok": True, "models": [{"id": "chosen-model"}]})
    monkeypatch.setattr(admin, "discover_llm_models", discover)
    monkeypatch.setenv("LLM_API_KEY", "environment-secret")

    result = await admin.admin_llm_models(admin.AdminLLMModelsRequest(**draft))

    assert result["models"] == [{"id": "chosen-model"}]
    assert discover.call_args.kwargs["api_key"] == expected_key
    assert saved_config.llm_model == "previously-chosen"
    assert saved_config.llm_api_key == "saved-secret"
    assert saved_config.llm_base_url == "https://provider.example/v1"
    assert saved_config.llm_backend == "provider"


@pytest.mark.asyncio
async def test_admin_discovery_uses_environment_key_only_for_saved_destination(
    saved_config, monkeypatch,
):
    saved_config.llm_api_key = ""
    monkeypatch.setenv("LLM_API_KEY", "environment-secret")
    discover = Mock(return_value={"ok": True, "models": []})
    monkeypatch.setattr(admin, "discover_llm_models", discover)

    await admin.admin_llm_models(admin.AdminLLMModelsRequest())
    assert discover.call_args.kwargs["api_key"] == "environment-secret"
    await admin.admin_llm_models(admin.AdminLLMModelsRequest(
        llm_base_url="https://different.example/v1",
    ))
    assert discover.call_args.kwargs["api_key"] == ""


@pytest.mark.asyncio
async def test_admin_discovery_accepts_draft_without_a_model(saved_config, monkeypatch):
    discover = Mock(return_value={"ok": True, "models": [], "message": "No models installed."})
    monkeypatch.setattr(admin, "discover_llm_models", discover)

    result = await admin.admin_llm_models(admin.AdminLLMModelsRequest(
        llm_backend="openai_compatible", llm_base_url="http://localhost:11434/v1",
        discovery_api="ollama",
    ))

    assert result["ok"]
    config = discover.call_args.args[0]
    assert config.llm_base_url == "http://localhost:11434/v1"
    assert discover.call_args.kwargs["discovery_api"] == "ollama"


@pytest.fixture(params=[False, True], ids=["fresh-runtime", "retained-research-session"])
def prior_admin_session(request, monkeypatch):
    """Exercise the auth check even after another test leaves a research session."""
    manager = (
        Mock(config=RuntimeConfig(research_mode=True))
        if request.param else None
    )
    monkeypatch.setattr(interface_api, "_session_manager", manager)


@pytest.mark.asyncio
async def test_admin_discovery_route_requires_bearer(
    prior_admin_session, monkeypatch, tmp_path,
):
    path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(
        data_dir=str(tmp_path / "data"), api_key="discovery-test-key",
    ).write_yaml(str(path))
    # Auth captures the config getter at import time, but reads this path per
    # request. Isolate both it and the research middleware from earlier tests.
    monkeypatch.setattr(interface_api, "RUNTIME_CONFIG_PATH", str(path))
    monkeypatch.setattr(interface_api, "_session_manager", None)
    payload = {"ok": True, "models": [{"id": "authorized-model"}]}
    discover = Mock(return_value=payload)
    monkeypatch.setattr(admin, "discover_llm_models", discover)

    # FastAPI >= 0.137 keeps included routers nested in app.routes. Exercise
    # the production app's HTTP contract instead of its internal route layout.
    async with AsyncClient(
        transport=ASGITransport(app=interface_api.app), base_url="http://test",
    ) as client:
        for headers, detail in [
            ({}, "Missing bearer token"),
            ({"Authorization": "Bearer wrong-key"}, "Invalid bearer token"),
        ]:
            response = await client.post("/admin/llm/models", json={}, headers=headers)
            assert response.status_code == 401, response.text
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.json() == {"detail": detail}
        discover.assert_not_called()

        headers = {"Authorization": "Bearer discovery-test-key"}
        response = await client.get("/admin/llm/models", headers=headers)
        assert response.status_code == 405, response.text
        assert response.headers["allow"] == "POST"
        discover.assert_not_called()

        response = await client.post("/admin/llm/models", json={}, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json() == payload
    discover.assert_called_once()
    assert discover.call_args.args[0].api_key == "discovery-test-key"


def test_legacy_backend_requires_explicit_model_for_setup_readiness(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = RuntimeConfig(llm_backend="minimax", llm_api_key="saved-key")
    assert not admin._llm_configured(config)
    assert any(
        warning["code"] == "missing_provider_model"
        for warning in admin._admin_config_warnings(config)
    )
    config.llm_model = "chosen-model"
    assert admin._llm_configured(config)


def test_auto_backend_requires_url_and_explicit_model():
    config = RuntimeConfig(llm_base_url="http://localhost:11434/v1")
    assert not admin._llm_configured(config)
    config.llm_model = "chosen-model"
    assert admin._llm_configured(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["openai_compatible", "provider"])
@pytest.mark.parametrize("chosen_model", ["org/user-choice:edition", "manual-unlisted-model", ""])
async def test_admin_config_persists_the_users_model_without_catalog_enforcement(
    monkeypatch, tmp_path, backend, chosen_model,
):
    path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(
        data_dir=str(tmp_path / "runtime"),
        llm_backend=backend,
        llm_base_url="https://provider.example/v1",
        llm_model="previously-chosen",
    ).write_yaml(str(path))
    monkeypatch.setattr(interface_api, "RUNTIME_CONFIG_PATH", str(path))
    reload_runtime = AsyncMock(return_value={"reloaded_web_manager": False})
    monkeypatch.setattr(interface_api, "_reload_runtime_from_config", reload_runtime)
    discover = Mock(side_effect=AssertionError("Saving a chosen model must not query a catalog"))
    monkeypatch.setattr(admin, "discover_llm_models", discover)

    result = await admin.admin_update_config(admin.AdminConfigUpdateRequest(
        llm_backend=backend,
        llm_model=chosen_model,
        setup_completed=True,
    ))

    assert result["config"]["llm_model"] == chosen_model
    assert RuntimeConfig.from_yaml(str(path)).llm_model == chosen_model
    assert reload_runtime.await_args.args[0].llm_model == chosen_model
    discover.assert_not_called()

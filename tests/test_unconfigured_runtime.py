"""Production API wiring remains usable before a model has been selected."""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio
import requests

from interface import api
from runtime.config import RuntimeConfig
from runtime.llm.backend import create_llm_backend
from runtime.sessions.user_session import UserSession


@pytest.fixture
def unconfigured_runtime(tmp_path, monkeypatch):
    config = RuntimeConfig(
        data_dir=str(tmp_path / "data"),
        llm_backend="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="",
        tools_enabled=True,
        tools_workspace=str(tmp_path),
    )
    path = tmp_path / "runtime_config.yaml"
    config.write_yaml(str(path))
    monkeypatch.setattr(api, "RUNTIME_CONFIG_PATH", str(path))
    monkeypatch.setattr(api, "_session_manager", None)
    monkeypatch.setattr(api, "_pipeline_override", None)
    monkeypatch.setattr(api, "_telegram_task", None)
    monkeypatch.setattr(api, "_telegram_channel", None)
    factory = Mock(wraps=create_llm_backend)
    monkeypatch.setattr(api, "create_llm_backend", factory)
    # These tests must never contact a model, even if routing regresses.
    monkeypatch.setattr(
        requests.sessions.Session, "request",
        Mock(side_effect=AssertionError("Unexpected network request")),
    )
    return factory


@pytest_asyncio.fixture
async def unconfigured_client(unconfigured_runtime):
    async with api._lifespan(api.app):
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
            yield client, unconfigured_runtime


@pytest.mark.parametrize("path", [
    "/v1/tools",
    "/v1/profiles/self",
    "/v1/profiles/person",
    "/v1/memory/long_term",
    "/v1/memory/relationship",
    "/v1/memory/semantic",
    "/v1/persona/state",
    "/v1/ready",
    "/v1/sessions",
    "/debug",
    "/config",
    "/admin/config",
    "/admin/status",
    "/admin/persona/state",
])
@pytest.mark.asyncio
async def test_inspection_never_constructs_backend(unconfigured_client, path):
    client, factory = unconfigured_client
    result = await client.get(path, params={
        "platform": "web", "user_id": "admin", "chat_id": "admin",
    })
    assert result.status_code == 200, result.text
    assert isinstance(result.json(), dict)
    if path == "/v1/tools":
        assert result.json()["count"] > 0
    factory.assert_not_called()


@pytest.mark.parametrize("path", ["/rest", "/v1/sessions/rest", "/session/end", "/v1/sessions/end"])
@pytest.mark.asyncio
async def test_idle_session_actions_need_no_backend(unconfigured_client, path):
    client, factory = unconfigured_client
    result = await client.post(path, json={
        "platform": "web", "user_id": "admin", "chat_id": "admin", "hours": 1,
    })
    assert result.status_code == 200, result.text
    factory.assert_not_called()


@pytest.mark.parametrize("path", ["/chat", "/v1/chat", "/admin/soul/draft"])
@pytest.mark.asyncio
async def test_generation_returns_missing_model_diagnostic(unconfigured_client, path, caplog):
    client, factory = unconfigured_client
    result = await client.post(path, json={
        "message": "Hello", "description": "A calm assistant who values clarity.",
    })
    assert result.status_code == 503, result.text
    payload = result.json()
    assert payload["llm_error"]["message"] == "No model is selected."
    assert payload["settings_url"] == "/settings"
    assert factory.called
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("path", ["/session/end", "/v1/sessions/end"])
@pytest.mark.asyncio
async def test_consolidation_backend_failure_uses_json_diagnostic(
    unconfigured_client, monkeypatch, path,
):
    client, _ = unconfigured_client

    async def fail(self):
        raise requests.Timeout("backend timed out")

    monkeypatch.setattr(UserSession, "end_session", fail)
    result = await client.post(path, json={"user_id": "admin", "chat_id": "admin"})
    assert result.status_code == 503, result.text
    assert result.json()["llm_error"]["code"] == "timeout"
    assert result.json()["settings_url"] == "/settings"


def test_websocket_missing_model_is_recoverable(unconfigured_runtime):
    factory = unconfigured_runtime
    with TestClient(api.app) as client, client.websocket_connect("/ws") as socket:
        socket.send_json({"message": "Hello", "user_id": "admin", "chat_id": "admin"})
        payload = socket.receive_json()
        assert payload["llm_error"]["message"] == "No model is selected."
        assert payload["settings_url"] == "/settings"
        socket.send_json({"message": "Try again", "user_id": "admin", "chat_id": "admin"})
        assert socket.receive_json()["llm_error"]["message"] == "No model is selected."
    assert factory.called

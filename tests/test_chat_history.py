"""Browser transcript restoration uses the existing current-session history."""

import httpx
import pytest

import interface.api as interface_api
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from runtime.sessions.persistence import save_conversation_history
from tests._fakes import MockLLMBackend


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def history_client(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=str(tmp_path / "data"))
    config_path = tmp_path / "runtime_config.yaml"
    config.write_yaml(str(config_path))
    monkeypatch.setattr(interface_api, "RUNTIME_CONFIG_PATH", str(config_path))
    manager = SessionManager(
        config=config,
        backend_factory=lambda: MockLLMBackend(response="I understand."),
    )
    monkeypatch.setattr(interface_api, "_pipeline_override", None)
    monkeypatch.setattr(interface_api, "_session_manager", manager)
    transport = httpx.ASGITransport(app=interface_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, manager, config_path
    await manager.shutdown()


async def test_history_returns_persisted_current_chat_without_creating_session(history_client):
    client, manager, _ = history_client
    history = [
        {"role": "user", "content": "I prefer bullet points."},
        {"role": "assistant", "content": "I will remember."},
    ]
    save_conversation_history(
        manager.config.session_history_path("web:alice:current"),
        history,
    )
    response = await client.get(
        "/session/history", params={"user_id": "alice", "chat_id": "current"},
    )
    assert response.status_code == 200
    assert response.json() == {"history": history}
    assert manager._sessions == {}
    for identity in (
        {"user_id": "alice", "chat_id": "other"},
        {"user_id": "bob", "chat_id": "current"},
    ):
        response = await client.get("/session/history", params=identity)
        assert response.json() == {"history": []}


async def test_end_session_clears_restorable_history_and_new_turn_restarts_it(history_client):
    client, _, _ = history_client
    identity = {"user_id": "alice", "chat_id": "current"}
    first = await client.post("/chat", json={**identity, "message": "Please remember this."})
    assert first.status_code == 200
    response = await client.get("/session/history", params=identity)
    assert response.json()["history"] == [
        {"role": "user", "content": "Please remember this."},
        {"role": "assistant", "content": first.json()["response"]},
    ]
    assert (await client.post("/session/end", json=identity)).status_code == 200
    assert (await client.get("/session/history", params=identity)).json() == {"history": []}
    second = await client.post("/chat", json={**identity, "message": "A fresh conversation."})
    assert second.status_code == 200
    response = await client.get("/session/history", params=identity)
    assert response.json()["history"] == [
        {"role": "user", "content": "A fresh conversation."},
        {"role": "assistant", "content": second.json()["response"]},
    ]


@pytest.mark.parametrize("field", ["user_id", "chat_id"])
async def test_history_rejects_path_traversal(history_client, field):
    client, _, _ = history_client
    response = await client.get("/session/history", params={field: "../other"})
    assert response.status_code == 400


async def test_history_requires_configured_bearer_token(history_client):
    client, manager, config_path = history_client
    manager.config.api_key = "test-history-token"
    manager.config.write_yaml(str(config_path))
    assert (await client.get("/session/history")).status_code == 401
    response = await client.get(
        "/session/history", headers={"Authorization": "Bearer test-history-token"},
    )
    assert response.status_code == 200
    assert response.json() == {"history": []}

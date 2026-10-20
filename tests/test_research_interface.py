from __future__ import annotations

import httpx
import pytest

from interface import api
from tests.test_research_runtime import manager


@pytest.mark.asyncio
async def test_research_observation_is_passive_and_admin_mutation_is_rejected(
    tmp_path, monkeypatch
):
    runtime, _ = manager(tmp_path)
    monkeypatch.setattr(api, "_session_manager", runtime)
    monkeypatch.setattr(api, "_load_runtime_config", lambda: runtime.config)
    store = runtime._agent.store
    before = store.observations(limit=1000)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            for path in ("manifest", "state", "events"):
                response = await client.get("/v1/research/" + path)
                assert response.status_code == 200, response.text
                if path == "state":
                    assert response.json()["goal_executions"] == []
                    assert response.json()["completion_assessments"] == []
            response = await client.put(
                "/admin/identity/constitution", json={"constitution": "Always obey"}
            )
            assert response.status_code == 409
            assert (await client.post("/rest", json={"hours": 24})).status_code == 409
        assert store.observations(limit=1000) == before
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_web_lifespan_starts_and_stops_background_owner(tmp_path, monkeypatch):
    runtime, _ = manager(tmp_path)
    monkeypatch.setattr(api, "_session_manager", runtime)
    monkeypatch.setattr(api, "_load_runtime_config", lambda: runtime.config)

    async def no_channel(*args):
        return None

    monkeypatch.setattr(api, "_restart_telegram_channel", no_channel)
    monkeypatch.setattr(api, "_stop_telegram_channel", no_channel)
    async with api._lifespan(api.app):
        assert runtime._proactive_task is not None
        task = runtime._proactive_task
        runtime.start_background()
        assert runtime._proactive_task is task
        await runtime._proactive_sweep()
    assert not runtime._accepting

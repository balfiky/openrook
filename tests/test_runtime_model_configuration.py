"""Model configuration is required for generation, not runtime inspection."""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from runtime.app import OpenRookApp
from runtime.channels.console import ConsoleChannel
from runtime.channels.telegram import TelegramChannel, TelegramConfig
from runtime.config import RuntimeConfig
from runtime.llm.backend import create_llm_backend
from runtime.llm.diagnostics import llm_health
from runtime.llm.lazy import LazyLLMBackend
from runtime.sessions.manager import SessionManager
from runtime.tools import create_tool_executor
from tests._fakes import MockLLMBackend


def _config(tmp_path, **kwargs):
    return RuntimeConfig(
        data_dir=str(tmp_path),
        llm_backend="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="",
        **kwargs,
    )


def test_lazy_backend_does_not_construct_for_close():
    factory = Mock()
    backend = LazyLLMBackend(factory)
    backend.close()
    backend.close()
    factory.assert_not_called()


def test_lazy_backend_reuses_and_closes_real_backend():
    real = Mock()
    real.generate.return_value = "reply"
    factory = Mock(return_value=real)
    backend = LazyLLMBackend(factory)
    assert backend.generate("system", "first") == "reply"
    assert backend.generate("system", "second", response_schema={"type": "object"}) == "reply"
    factory.assert_called_once_with()
    real.generate.assert_called_with("system", "second", response_schema={"type": "object"})
    backend.close()
    backend.close()
    real.close.assert_called_once_with()


def test_runtime_startup_and_inspection_do_not_require_model(tmp_path, monkeypatch):
    factory = Mock(side_effect=AssertionError("inspection constructed a model"))
    monkeypatch.setattr("runtime.app.create_llm_backend", factory)
    app = OpenRookApp(_config(tmp_path))

    async def run():
        try:
            session = await app.session_manager.ensure_session("web", "admin", "admin")
            assert session.pipeline.engine.snapshot()
            await session.apply_rest(1)
            await session.end_session()
        finally:
            await app.shutdown()

    asyncio.run(run())
    factory.assert_not_called()


def test_generation_requires_configured_model(tmp_path):
    config = _config(tmp_path)
    manager = SessionManager(config, backend_factory=lambda: create_llm_backend(config))

    async def run():
        try:
            session = await manager.ensure_session("console", "user", "direct")
            before = session.pipeline.engine.snapshot()
            with pytest.raises(ValueError, match="llm_model"):
                await manager.handle_message("console", "user", "direct", "Tell me a story")
            assert session.pipeline.engine.snapshot() == before
            assert session.pipeline.export_conversation_history() == []
        finally:
            await manager.shutdown()

    asyncio.run(run())


def test_shutdown_drains_message_while_backend_is_initializing(tmp_path, monkeypatch):
    release_factory = threading.Event()
    backend = MockLLMBackend(response="Ready.")
    backend.close = Mock()

    async def run():
        loop = asyncio.get_running_loop()
        factory_started = asyncio.Event()
        shutdown_started = asyncio.Event()

        def factory():
            loop.call_soon_threadsafe(factory_started.set)
            if not release_factory.wait(timeout=5):
                raise TimeoutError("Test did not release the model factory")
            return backend

        manager = SessionManager(_config(tmp_path), backend_factory=factory)
        session = await manager.ensure_session("console", "user", "direct")
        original_drain = session.drain_and_close

        async def drain(**kwargs):
            shutdown_started.set()
            await original_drain(**kwargs)

        monkeypatch.setattr(session, "drain_and_close", drain)
        message_task = asyncio.create_task(
            manager.handle_message("console", "user", "direct", "Tell me a story"),
        )
        shutdown_task = None
        try:
            await asyncio.wait_for(factory_started.wait(), timeout=2)
            shutdown_task = asyncio.create_task(manager.shutdown())
            await asyncio.wait_for(shutdown_started.wait(), timeout=2)
            assert not shutdown_task.done()
            backend.close.assert_not_called()
            release_factory.set()
            assert await asyncio.wait_for(message_task, timeout=3)
            await asyncio.wait_for(shutdown_task, timeout=3)
            backend.close.assert_called_once_with()
        finally:
            release_factory.set()
            await asyncio.gather(message_task, return_exceptions=True)
            if shutdown_task is not None:
                await asyncio.gather(shutdown_task, return_exceptions=True)
            else:
                await manager.shutdown()

    asyncio.run(run())


def test_console_reports_model_diagnostic_and_keeps_running(tmp_path, monkeypatch, capsys):
    config = _config(tmp_path)
    manager = SessionManager(config, backend_factory=lambda: create_llm_backend(config))
    channel = ConsoleChannel(manager)
    monkeypatch.setattr(
        channel, "_next_line", AsyncMock(side_effect=["Tell me a story", "/help", "/quit"]),
    )

    async def run():
        try:
            await channel.start()
        finally:
            await manager.shutdown()

    asyncio.run(run())
    output = capsys.readouterr()
    assert "No model is selected" in output.err
    assert "Settings" in output.err
    assert "Internal error" not in output.err
    assert "Send a message to chat" in output.out


def test_telegram_introspection_and_generation_without_model(tmp_path):
    config = _config(tmp_path)
    factory = Mock(side_effect=lambda: create_llm_backend(config))
    manager = SessionManager(config, backend_factory=factory)
    client = AsyncMock()
    channel = TelegramChannel(client, manager, TelegramConfig())

    async def run():
        try:
            await channel._handle_command("/mental", "100", 100)
            factory.assert_not_called()
            assert "Mental state:" in client.send_message.call_args.args[1]
            await channel._process_message("100", 100, "Tell me a story")
            message = client.send_message.call_args.args[1]
            assert "No model is selected" in message
            assert "Settings" in message
            assert "Something went wrong" not in message
        finally:
            await manager.shutdown()

    asyncio.run(run())


def test_proactive_missing_model_records_diagnostic_without_traceback(tmp_path, monkeypatch, caplog):
    config = _config(tmp_path, proactive_idle_threshold=0)
    factory = Mock(side_effect=lambda: create_llm_backend(config))
    manager = SessionManager(config, backend_factory=factory)

    async def run():
        try:
            await manager._proactive_sweep()
            factory.assert_not_called()
            session = await manager.ensure_session("console", "user", "direct")
            monkeypatch.setattr(
                session.pipeline, "process_proactive",
                lambda *args, **kwargs: session.pipeline._llm_backend.generate("system", "message"),
            )
            await manager._proactive_sweep()
            assert llm_health(config)["diagnosis"]["code"] == "configuration"
        finally:
            await manager.shutdown()

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    assert "No model is selected" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize(
    ("message", "health_status"),
    [
        ("OpenAI-compatible backend requires llm_model", "error"),
        ("Unexpected constructor failure", "untested"),
    ],
)
def test_research_background_preserves_failure_and_records_backend_health(
    tmp_path, message, health_status,
):
    config = _config(tmp_path, research_mode=True)
    factory = Mock(side_effect=ValueError(message))
    manager = SessionManager(config, backend_factory=factory)

    async def run():
        try:
            await manager._proactive_sweep()
            factory.assert_called_once_with()
            health = llm_health(config)
            assert health["status"] == health_status
            if health_status == "error":
                assert health["diagnosis"]["message"] == "No model is selected."
                assert health["diagnosis"]["settings_url"] == "/settings"
            events = manager._agent.store.observations()
            failure = next(event for event in events if event["kind"] == "autonomous_failure")
            assert failure["payload"] == {"type": "ValueError", "error": message}
            transition = next(event for event in reversed(events) if event["kind"] == "transition")
            assert transition["payload"]["status"] == "failed"
            assert transition["payload"]["error"] == message
        finally:
            await manager.shutdown()

    asyncio.run(run())


def test_native_tool_client_is_deferred_until_selection(tmp_path, monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message={"content": "ready"})],
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr("nur_tools.native_orchestrator.OpenAI", constructor)
    config = _config(tmp_path, tools_enabled=True)
    config.llm_model = uuid4().hex
    executor = create_tool_executor(config)
    try:
        constructor.assert_not_called()
        assert executor.registry.list_tools()
        runner = executor.tool_runner
        assert runner is not None
        assert runner._create_tool_choice([], []) == {"content": "ready"}
        constructor.assert_called_once()
    finally:
        for resource in executor.owned_resources:
            resource.close()


@pytest.mark.parametrize("backend", ["openai_compatible", "codex"])
def test_native_tools_available_without_native_model_client(tmp_path, monkeypatch, backend):
    constructor = Mock(side_effect=AssertionError("inspection constructed native model client"))
    monkeypatch.setattr("nur_tools.native_orchestrator.OpenAI", constructor)
    config = _config(tmp_path, tools_enabled=True)
    config.llm_backend = backend
    if backend == "codex":
        config.llm_model = uuid4().hex
    executor = create_tool_executor(config)
    try:
        assert executor.registry.list_tools()
        assert executor.tool_runner is None
    finally:
        for resource in executor.owned_resources:
            resource.close()
    constructor.assert_not_called()

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

import main as runtime_main
from runtime.channels.console import ConsoleChannel
from runtime.config import RuntimeConfig


def test_console_help_and_unknown_commands_are_local(capsys, monkeypatch):
    manager = AsyncMock()
    manager.handle_message.return_value = "Hello back."
    console = ConsoleChannel(manager)
    lines = AsyncMock(side_effect=["/help", "/COMMANDS", "/unknown", "hello", "/quit"])
    monkeypatch.setattr(console, "_next_line", lines)

    asyncio.run(console.start())

    output = capsys.readouterr().out
    assert output.count("Send a message to chat with OpenRook.") == 2
    assert "/quit or /exit" in output
    assert "Unknown command." in output
    assert "OpenRook: Hello back." in output
    manager.handle_message.assert_awaited_once_with("console", "user", "direct", "hello")


@pytest.mark.parametrize(
    ("console_enabled", "verbose", "expected_level"),
    [(True, False, logging.WARNING), (True, True, logging.INFO), (False, False, logging.INFO)],
)
def test_runtime_log_defaults_keep_console_prompt_clear(
    tmp_path, monkeypatch, console_enabled, verbose, expected_level,
):
    config_path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(console_enabled=console_enabled, llm_model=uuid4().hex).write_yaml(str(config_path))
    argv = ["openrook", "--config", str(config_path)]
    if verbose:
        argv.append("--verbose")
    monkeypatch.setattr("sys.argv", argv)
    app = AsyncMock()
    monkeypatch.setattr(runtime_main, "OpenRookApp", lambda _config: app)
    log_options = {}
    monkeypatch.setattr(runtime_main.logging, "basicConfig", lambda **kw: log_options.update(kw))

    runtime_main.main()

    assert log_options["level"] == expected_level
    app.run.assert_awaited_once()


def test_runtime_missing_config_exits_with_setup_instructions(tmp_path, monkeypatch):
    config_path = tmp_path / "runtime_config.yaml"
    monkeypatch.setattr("sys.argv", ["openrook", "--config", str(config_path)])
    app_factory = Mock()
    monkeypatch.setattr(runtime_main, "OpenRookApp", app_factory)

    with pytest.raises(SystemExit, match="Runtime config not found") as error:
        runtime_main.main()

    assert str(config_path) in str(error.value)
    assert "openrook-setup or openrook-web" in str(error.value)
    assert not config_path.exists()
    app_factory.assert_not_called()


@pytest.mark.parametrize("model", ["", "  ", None])
def test_runtime_no_model_exits_before_starting_services(tmp_path, monkeypatch, model):
    config_path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(data_dir=str(tmp_path / "data"), llm_model=model).write_yaml(str(config_path))
    original = config_path.read_bytes()
    monkeypatch.setattr("sys.argv", ["openrook", "--config", str(config_path)])
    app_factory = Mock()
    monkeypatch.setattr(runtime_main, "OpenRookApp", app_factory)

    with pytest.raises(SystemExit, match="No LLM model selected") as error:
        runtime_main.main()

    assert "model served by your backend" in str(error.value)
    assert "openrook-setup or openrook-web" in str(error.value)
    assert config_path.read_bytes() == original
    assert not (tmp_path / "data").exists()
    app_factory.assert_not_called()


def test_runtime_codex_can_delegate_model_selection(tmp_path, monkeypatch):
    config_path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(llm_backend="codex").write_yaml(str(config_path))
    monkeypatch.setattr("sys.argv", ["openrook", "--config", str(config_path)])
    app = AsyncMock()
    monkeypatch.setattr(runtime_main, "OpenRookApp", lambda _config: app)

    runtime_main.main()

    app.run.assert_awaited_once()


def test_runtime_help_describes_console_and_optional_background_mode(capsys):
    with pytest.raises(SystemExit) as error:
        runtime_main._parse_args(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "console runtime" in help_text
    assert "console_enabled" in help_text
    assert "--verbose" in help_text

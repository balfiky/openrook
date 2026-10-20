from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

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
    RuntimeConfig(console_enabled=console_enabled).write_yaml(str(config_path))
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


def test_runtime_help_describes_console_and_optional_background_mode(capsys):
    with pytest.raises(SystemExit) as error:
        runtime_main._parse_args(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "console runtime" in help_text
    assert "console_enabled" in help_text
    assert "--verbose" in help_text

"""Deprecated command aliases retained for existing installations."""

from __future__ import annotations

from importlib import import_module

from config.environment import warn_legacy

# Explicit targets keep the release validator and installed entry points aligned.
COMMAND_TARGETS = {
    "nur": "main:main",
    "nur-setup": "interface.setup:main",
    "nur-uninstall": "interface.uninstall:main",
    "nur-web": "interface.api:main",
    "nur-genesis-reset": "runtime.genesis_reset:main",
    "nur-emotional-qa": "evals.emotional_quality:main",
    "nur-uat": "openrook_tools.uat:main",
    "nur-validate": "openrook_tools.validate:main",
}


def _run(command: str) -> int | None:
    warn_legacy("command", command, command.replace("nur", "openrook", 1))
    module, function = COMMAND_TARGETS[command].split(":")
    return getattr(import_module(module), function)()


def main() -> int | None:
    return _run("nur")


def setup() -> int | None:
    return _run("nur-setup")


def uninstall() -> int | None:
    return _run("nur-uninstall")


def web() -> int | None:
    return _run("nur-web")


def genesis_reset() -> int | None:
    return _run("nur-genesis-reset")


def emotional_qa() -> int | None:
    return _run("nur-emotional-qa")


def uat() -> int | None:
    return _run("nur-uat")


def validate() -> int | None:
    return _run("nur-validate")

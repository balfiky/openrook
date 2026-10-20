"""Compatibility contracts for renamed imports, commands, and environment keys."""

from __future__ import annotations

import importlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from config import environment
from openrook_tools import legacy
from openrook_tools.validate import EXPECTED_CONSOLE_SCRIPTS


@pytest.fixture(autouse=True)
def reset_deprecation_notices(monkeypatch):
    monkeypatch.setattr(environment, "_notices_shown", set())


def test_environment_only_warns_when_legacy_value_is_selected(capsys):
    assert environment.project_env("TEST", environ={"NUR_TEST": "legacy"}) == "legacy"
    assert environment.project_env("OTHER", environ={"NUR_OTHER": ""}) == ""
    assert environment.project_env("MISSING", "default", environ={}) == "default"
    assert capsys.readouterr().err.splitlines() == [
        "Deprecation: NUR_TEST is deprecated; use OPENROOK_TEST instead."
    ]


@pytest.mark.parametrize("value", ["", "canonical"])
def test_canonical_environment_has_precedence_without_notice(value, capsys):
    assert environment.project_env(
        "TEST", environ={"OPENROOK_TEST": value, "NUR_TEST": "legacy"}
    ) == value
    assert not capsys.readouterr().err


def test_environment_notice_is_once_even_with_concurrent_reads(capsys):
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(
            lambda _: environment.project_env("TEST", environ={"NUR_TEST": "old"}),
            range(32),
        ))
    assert values == ["old"] * 32
    assert len(capsys.readouterr().err.splitlines()) == 1


@pytest.mark.parametrize("command", legacy.COMMAND_TARGETS)
def test_legacy_commands_delegate_and_warn_once(command, monkeypatch, capsys):
    target = legacy.COMMAND_TARGETS[command]
    module_name, function_name = target.split(":")
    calls = []

    def import_target(name):
        assert name == module_name
        return SimpleNamespace(**{function_name: lambda: calls.append(name) or 7})

    monkeypatch.setattr(legacy, "import_module", import_target)
    module, wrapper = EXPECTED_CONSOLE_SCRIPTS[command].split(":")
    assert module == "openrook_tools.legacy"
    entrypoint = getattr(legacy, wrapper)
    assert entrypoint() == 7
    assert entrypoint() == 7
    assert calls == [module_name, module_name]
    assert capsys.readouterr().err.splitlines() == [
        f"Deprecation: {command} is deprecated; "
        f"use {command.replace('nur', 'openrook', 1)} instead."
    ]


def test_import_shim_shares_nested_modules_without_optional_dependency_imports():
    code = """
import importlib
import sys
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    import nur_tools
assert any(issubclass(w.category, DeprecationWarning) for w in caught)
import openrook_tools
assert nur_tools.ToolRegistry is openrook_tools.ToolRegistry
assert "openrook_tools.langgraph_orchestrator" not in sys.modules
for suffix in ("registry", "types", "builtin", "builtin.filesystem", "mcp.client"):
    old = importlib.import_module("nur_tools." + suffix)
    new = importlib.import_module("openrook_tools." + suffix)
    assert old is new, suffix
    assert new.__spec__.name == "openrook_tools." + suffix
    old.compatibility_marker = suffix
    assert new.compatibility_marker == suffix
registry = importlib.import_module("openrook_tools.registry")
assert importlib.reload(registry) is sys.modules["nur_tools.registry"]
assert registry.__spec__.name == "openrook_tools.registry"
assert len(caught) == 1
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_legacy_python_module_invocation_still_works():
    result = subprocess.run(
        [sys.executable, "-m", "nur_tools.validate", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "openrook-validate" in result.stdout


def test_all_declared_cli_targets_resolve():
    for target in EXPECTED_CONSOLE_SCRIPTS.values():
        module_name, function_name = target.split(":")
        assert callable(getattr(importlib.import_module(module_name), function_name))

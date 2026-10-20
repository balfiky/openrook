"""Backend failures should tell users how to recover without leaking secrets."""

import json
from unittest.mock import Mock

import pytest
import requests

from runtime.config import RuntimeConfig
from runtime.llm.diagnostics import (
    check_llm_connection,
    diagnose_llm_error,
    llm_health,
    record_llm_health,
)


def _config(tmp_path, **kwargs):
    return RuntimeConfig(
        data_dir=str(tmp_path),
        llm_backend="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="selected-test-model",
        **kwargs,
    )


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    response.url = "http://localhost:11434/v1/chat/completions"
    return requests.HTTPError(f"HTTP {status}", response=response)


@pytest.mark.parametrize(
    ("error", "code", "action"),
    [
        (_http_error(404), "not_found", "Choose an available model in Settings"),
        (requests.ConnectionError("refused"), "unreachable", "ollama serve"),
        (requests.Timeout("slow"), "timeout", "faster model"),
        (_http_error(401), "authentication", "API key"),
        (_http_error(429), "rate_limit", "Wait and retry"),
        (_http_error(503), "provider_unavailable", "backend service"),
    ],
)
def test_backend_errors_are_actionable(tmp_path, error, code, action):
    result = diagnose_llm_error(error, _config(tmp_path))
    assert result["code"] == code
    assert action in result["action"]
    assert "selected-test-model" in result["message"]
    assert "http://localhost:11434/v1" in result["message"]
    assert result["settings_url"] == "/settings"


def test_codex_login_diagnosis_redacts_cli_output(tmp_path, monkeypatch):
    config = RuntimeConfig(
        data_dir=str(tmp_path), llm_backend="codex", llm_api_key="stored-secret",
    )
    monkeypatch.setenv("LLM_API_KEY", "environment-secret")
    error = RuntimeError(
        "Codex backend failed with exit code 1: 401 Unauthorized "
        "stored-secret environment-secret Bearer other-secret "
        "https://person:url-password@example.com/v1?api_key=query-secret"
    )
    result = diagnose_llm_error(error, config)
    assert "codex login" in result["action"]
    serialized = json.dumps(result)
    for secret in ("stored-secret", "environment-secret", "other-secret", "url-password", "query-secret"):
        assert secret not in serialized


def test_programming_errors_are_not_misclassified(tmp_path):
    assert diagnose_llm_error(KeyError("choices"), _config(tmp_path)) is None
    assert diagnose_llm_error(RuntimeError("session queue full"), _config(tmp_path)) is None


@pytest.mark.parametrize("backend", ["openai_compatible", "provider", "auto", "minimax"])
@pytest.mark.parametrize("model", ["", "   "])
def test_missing_model_diagnostic_is_specific(tmp_path, backend, model):
    config = RuntimeConfig(
        data_dir=str(tmp_path), llm_backend=backend, llm_model=model,
    )

    result = check_llm_connection(config)

    assert result["ok"] is False
    diagnosis = result["diagnostic"]
    assert diagnosis["code"] == "configuration"
    assert diagnosis["message"] == "No model is selected."
    assert diagnosis["action"] == "Choose an available model in Settings, then test the connection."
    assert diagnosis["settings_url"] == "/settings"


@pytest.mark.parametrize(
    ("overrides", "message", "action"),
    [
        (
            {"llm_base_url": ""},
            "No backend URL is configured.",
            "Set the backend's Base URL in Settings",
        ),
        (
            {"llm_base_url": "file:///tmp/backend"},
            "The configured backend URL is invalid or is not allowed.",
            "Check the backend's Base URL in Settings",
        ),
        (
            {"llm_backend": "unsupported", "llm_model": ""},
            "The selected backend 'unsupported' is not supported.",
            "Choose a supported backend in Settings",
        ),
    ],
)
def test_configuration_diagnostics_identify_the_setting(tmp_path, overrides, message, action):
    config = _config(tmp_path)
    for field, value in overrides.items():
        setattr(config, field, value)

    result = check_llm_connection(config)

    assert result["ok"] is False
    assert result["diagnostic"]["message"] == message
    assert action in result["diagnostic"]["action"]
    assert result["diagnostic"]["settings_url"] == "/settings"


def test_codex_default_model_is_valid_and_login_error_is_specific(tmp_path):
    config = RuntimeConfig(
        data_dir=str(tmp_path), llm_backend="codex", llm_model="",
        llm_base_url="http://localhost:11434/v1",
    )
    backend = Mock()
    backend.generate.return_value = "ok"
    assert check_llm_connection(config, backend_factory=lambda _: backend)["ok"] is True

    result = diagnose_llm_error(RuntimeError("Codex backend: not logged in"), config)
    assert result["code"] == "authentication"
    assert result["model"] == "Codex default"
    assert result["base_url"] == ""
    assert "localhost" not in result["message"]
    assert "rejected authentication" in result["message"]
    assert "codex login" in result["action"]


@pytest.mark.parametrize(
    ("error", "message", "action"),
    [
        (
            "Codex backend requires the codex CLI on PATH",
            "The Codex CLI could not be found.",
            "Install the Codex CLI",
        ),
        (
            "OPENROOK_CODEX_WORKDIR does not exist: /missing/path",
            "The configured Codex working directory does not exist.",
            "Set OPENROOK_CODEX_WORKDIR to an existing directory.",
        ),
    ],
)
def test_codex_configuration_diagnostics_are_specific(tmp_path, error, message, action):
    config = RuntimeConfig(data_dir=str(tmp_path), llm_backend="codex", llm_model="")
    result = diagnose_llm_error(ValueError(error), config)
    assert result["message"] == message
    assert action in result["action"]
    assert result["settings_url"] == "/settings"


def test_check_closes_backend_and_replaces_error_with_success(tmp_path):
    config = _config(tmp_path)
    backend = Mock()
    backend.generate.side_effect = _http_error(404)
    factory = Mock(return_value=backend)

    failed = check_llm_connection(config, backend_factory=factory)
    assert failed["ok"] is False
    assert "Choose an available model in Settings" in failed["error"]
    assert "ollama pull" not in failed["error"]
    assert llm_health(config)["status"] == "error"
    backend.close.assert_called_once()

    backend.generate.side_effect = None
    backend.generate.return_value = "ok"
    passed = check_llm_connection(config, backend_factory=factory)
    assert passed["ok"] is True
    assert passed["sample_response"] == "ok"
    assert llm_health(config)["status"] == "ok"
    assert backend.close.call_count == 2


def test_health_does_not_follow_changed_credentials_or_model(tmp_path, monkeypatch):
    config = _config(tmp_path)
    record_llm_health(config, diagnose_llm_error(_http_error(404), config))
    config.llm_model = "another-model"
    assert llm_health(config)["status"] == "untested"
    config.llm_model = "selected-test-model"
    monkeypatch.setenv("LLM_API_KEY", "new-key")
    assert llm_health(config)["status"] == "untested"

"""Model catalogs are discovered safely and never choose a model for the user."""

import json
from unittest.mock import Mock

import pytest
import requests

from runtime.config import RuntimeConfig
from runtime.llm.discovery import OLLAMA_EMPTY_MESSAGE, discover_llm_models


def _config(**kwargs):
    return RuntimeConfig(
        llm_backend="openai_compatible",
        llm_base_url=kwargs.pop("llm_base_url", "http://localhost:11434/v1"),
        **kwargs,
    )


def _response(monkeypatch, payload, *, status=200):
    response = Mock(status_code=status)
    response.json.return_value = payload
    get = Mock(return_value=response)
    monkeypatch.setattr("runtime.llm.discovery.requests.get", get)
    return get


def test_ollama_tags_are_parsed_without_choosing_a_model(monkeypatch):
    get = _response(monkeypatch, {"models": [
        {"name": "first-model:tag"}, {"model": "second-model"}, {"name": "first-model:tag"},
        {"name": ""}, {"name": 42}, None,
    ]})

    result = discover_llm_models(_config())

    assert result["ok"]
    assert result["source"] == "ollama"
    assert [model["id"] for model in result["models"]] == ["first-model:tag", "second-model"]
    assert "selected_model" not in result
    get.assert_called_once_with(
        "http://localhost:11434/api/tags", headers={"Accept": "application/json"},
        timeout=(3.05, 8), allow_redirects=False,
    )


def test_openai_models_include_ids_and_provider_labels(monkeypatch):
    get = _response(monkeypatch, {"data": [
        {"id": "org/first-model", "name": "First model"}, {"id": "second-model"},
        {"id": "second-model"}, {"id": "  "}, {"id": None}, "not a model",
    ]})
    result = discover_llm_models(_config(
        llm_base_url="https://provider.example/v1/chat/completions",
        llm_api_key="chosen-key",
    ))

    assert result["ok"]
    assert result["source"] == "openai"
    assert result["models"] == [
        {"id": "org/first-model", "display_name": "First model"},
        {"id": "second-model", "display_name": "second-model"},
    ]
    assert get.call_args.args == ("https://provider.example/v1/models",)
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer chosen-key"


def test_explicit_ollama_catalog_preserves_reverse_proxy_path(monkeypatch):
    get = _response(monkeypatch, {"models": []})
    result = discover_llm_models(
        _config(llm_base_url="http://localhost:8090/local-llm/v1"), discovery_api="ollama",
    )
    assert get.call_args.args == ("http://localhost:8090/local-llm/api/tags",)
    assert result["ok"]
    assert result["models"] == []
    assert result["message"] == OLLAMA_EMPTY_MESSAGE


def test_openai_catalog_can_be_forced_at_ollama_port(monkeypatch):
    get = _response(monkeypatch, {"data": []})
    result = discover_llm_models(_config(), discovery_api="openai")
    assert result["source"] == "openai"
    assert "no models" in result["message"]
    assert get.call_args.args == ("http://localhost:11434/v1/models",)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "authentication"), (403, "authentication"), (302, "redirected"), (404, "HTTP 404"),
     (500, "HTTP 500")],
)
def test_http_errors_do_not_expose_backend_body(monkeypatch, status, expected):
    get = _response(monkeypatch, {"error": "super-secret"}, status=status)
    result = discover_llm_models(_config(llm_api_key="super-secret"))
    assert not result["ok"]
    assert expected in result["error"]
    assert result["models"] == []
    assert "super-secret" not in json.dumps(result)
    assert not get.return_value.json.called
    assert get.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("exception", [
    requests.Timeout("Bearer super-secret"),
    requests.ConnectionError("http://user:super-secret@localhost:11434"),
    requests.RequestException("super-secret"),
])
def test_transport_errors_are_safe(monkeypatch, exception):
    get = Mock(side_effect=exception)
    monkeypatch.setattr("runtime.llm.discovery.requests.get", get)
    result = discover_llm_models(_config(llm_api_key="super-secret"))
    assert not result["ok"]
    assert "super-secret" not in json.dumps(result)
    assert "manually" in result["message"]


@pytest.mark.parametrize("payload", [[], {}, {"models": {}}, None])
def test_malformed_catalog_is_reported(monkeypatch, payload):
    _response(monkeypatch, payload)
    result = discover_llm_models(_config())
    assert not result["ok"]
    assert "unsupported" in result["error"]


def test_invalid_json_is_reported_without_body(monkeypatch):
    get = _response(monkeypatch, {})
    get.return_value.json.side_effect = ValueError("super-secret HTML error")
    result = discover_llm_models(_config())
    assert not result["ok"]
    assert "unreadable" in result["error"]
    assert "super-secret" not in json.dumps(result)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://169.254.169.254/latest/meta-data",
    "http://10.0.0.8/v1", "https://user:super-secret@provider.example/v1",
    "https://provider.example/v1?api_key=super-secret",
    "https://provider.example/v1#super-secret", "http://localhost:bad/v1",
])
def test_unsafe_urls_never_issue_request(monkeypatch, url):
    get = _response(monkeypatch, {})
    result = discover_llm_models(_config(llm_base_url=url))
    assert not result["ok"]
    assert "super-secret" not in json.dumps(result)
    get.assert_not_called()


def test_dns_resolved_metadata_address_is_rejected(monkeypatch):
    get = _response(monkeypatch, {})
    monkeypatch.setattr("runtime.security._resolve_hostname", lambda *_: ["169.254.169.254"])
    result = discover_llm_models(_config(llm_base_url="https://backend.example/v1"))
    assert not result["ok"]
    get.assert_not_called()


def test_explicit_empty_key_suppresses_saved_and_environment_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    get = _response(monkeypatch, {"models": []})
    discover_llm_models(_config(llm_api_key="saved-key"), api_key="")
    assert "Authorization" not in get.call_args.kwargs["headers"]


@pytest.mark.parametrize("config", [
    RuntimeConfig(), RuntimeConfig(llm_backend="codex"),
])
def test_missing_url_or_codex_does_not_make_network_request(monkeypatch, config):
    get = _response(monkeypatch, {})
    assert not discover_llm_models(config)["ok"]
    get.assert_not_called()

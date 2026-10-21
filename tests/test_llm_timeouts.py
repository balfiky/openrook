"""Configurable slow generation, bounded auxiliaries, and safe timeout diagnostics."""

import json
import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import requests

from core.inference import inference_turn, model_call
from evals.social_backend import ScriptedSocialBackend
from openrook_tools.native_orchestrator import NativeToolCallRunner
from openrook_tools.replay_dignity import fixture_executor
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.llm.backend import OpenAICompatibleLLMBackend, create_llm_backend
from runtime.llm.diagnostics import diagnose_llm_error, llm_health, record_llm_health


def _response(status=200, text="", message=None):
    def check():
        if status >= 400:
            raise requests.HTTPError("provider rejected request")
    return SimpleNamespace(status_code=status, text=text, raise_for_status=check,
                           json=lambda: {"choices": [{"message": message or {"content": "ok"}}]})


def test_old_yaml_gets_cpu_friendly_default_and_custom_timeout_round_trips(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("console_enabled: false\n")
    config = RuntimeConfig.from_yaml(str(path))
    assert config.llm_request_timeout_seconds == 600
    config.llm_request_timeout_seconds = 1234.5
    config.write_yaml(str(path))
    assert RuntimeConfig.from_yaml(str(path)).llm_request_timeout_seconds == 1234.5


@pytest.mark.parametrize("value", [0, -1, 0.5, True, "600", float("nan"), float("inf")])
def test_timeout_rejects_invalid_yaml_values(value):
    with pytest.raises(ValueError, match="llm_request_timeout_seconds"):
        RuntimeConfig(llm_request_timeout_seconds=value)


@pytest.mark.parametrize("kind", ["auto", "provider", "openai_compatible", "minimax", "codex"])
def test_factory_wires_configured_timeout_to_all_backends(monkeypatch, kind):
    monkeypatch.setattr("runtime.llm.backend.shutil.which", lambda _: "/offline/cli")
    config = RuntimeConfig(llm_backend=kind, llm_base_url="http://localhost:1/v1",
                           llm_model="offline-fixture", llm_request_timeout_seconds=1234.5)
    backend = create_llm_backend(config)
    try:
        assert backend._timeout == 1234.5
    finally:
        if hasattr(backend, "close"):
            backend.close()


def test_http_primary_and_optional_limits_do_not_leak(monkeypatch):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline-fixture", timeout=900)
    timeouts = []
    def post(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return _response()
    monkeypatch.setattr(backend._session, "post", post)
    try:
        with inference_turn([]):
            backend.generate("main", "user")
            with model_call("appraisal", optional=True):
                backend.generate("auxiliary", "user")
            backend.generate("main", "user")
        backend.set_research_limits(output_tokens=100, timeout=25)
        with model_call("appraisal", optional=True):
            backend.generate("bounded research", "user")
        assert timeouts == [900, 8, 900, 25]
    finally:
        backend.close()


@pytest.mark.parametrize("error_type", [requests.ReadTimeout, requests.ConnectTimeout])
def test_transport_does_not_retry_a_timeout(monkeypatch, error_type):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline-fixture")
    def post(*args, **kwargs):
        raise error_type("private provider detail")
    monkeypatch.setattr(backend._session, "post", post)
    try:
        with pytest.raises(error_type):
            backend.generate("system", "user")
        assert len(backend.last_response_metadata["transport_attempts"]) == 1
    finally:
        backend.close()


def test_reasoning_controls_retry_only_rejected_fields_and_remember_support(monkeypatch):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline-fixture")
    outcomes = iter([_response(400, "unknown field chat_template_kwargs"),
                     _response(422, "reasoning_effort does not support none"), _response(), _response()])
    payloads = []
    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return next(outcomes)
    monkeypatch.setattr(backend._session, "post", post)
    try:
        assert backend.generate("system", "user") == "ok"
        assert payloads[0]["reasoning_effort"] == "none"
        assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert "chat_template_kwargs" not in payloads[1]
        assert payloads[1]["reasoning_effort"] == "none"
        assert "reasoning_effort" not in payloads[2]
        assert len(backend.last_response_metadata["transport_attempts"]) == 3
        assert backend.generate("system", "user") == "ok"
        assert payloads[3] == payloads[2]
        assert len(backend.last_response_metadata["transport_attempts"]) == 1
    finally:
        backend.close()


def test_unrelated_bad_request_does_not_retry_or_change_reasoning_controls(monkeypatch):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline-fixture")
    monkeypatch.setattr(backend._session, "post", lambda *a, **kw: _response(400, "invalid max_tokens"))
    try:
        with pytest.raises(requests.HTTPError):
            backend.generate("system", "user")
        assert len(backend.last_response_metadata["transport_attempts"]) == 1
        assert not backend._unsupported_reasoning_controls
    finally:
        backend.close()


def test_reasoning_controls_leave_bounded_research_payload_unchanged(monkeypatch):
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline-fixture")
    backend.set_research_limits(output_tokens=100, timeout=25)
    def post(*args, **kwargs):
        assert "reasoning_effort" not in kwargs["json"]
        assert "chat_template_kwargs" not in kwargs["json"]
        assert kwargs["json"]["max_tokens"] == 100 and kwargs["timeout"] == 25
        return _response(400, "reasoning_effort unsupported")
    monkeypatch.setattr(backend._session, "post", post)
    try:
        with pytest.raises(requests.HTTPError):
            backend.generate("system", "user")
        assert len(backend.last_response_metadata["transport_attempts"]) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("phase", ["generation", "appraisal", "dignity", "regeneration", "tool_selection", "self_intent"])
def test_timeout_log_identifies_purpose_and_elapsed_without_private_details(caplog, monkeypatch, phase):
    caplog.set_level(logging.WARNING, logger="core.inference")
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("core.inference.time", SimpleNamespace(monotonic=lambda: next(ticks)))
    calls = []
    with pytest.raises(requests.ReadTimeout), inference_turn(calls), model_call(phase, optional=phase != "generation"):
        raise requests.ReadTimeout("secret-token https://private.invalid/?api_key=private prompt contents")
    purpose = "tool_loop" if phase == "tool_selection" else phase
    assert f"purpose={purpose}" in caplog.text
    assert "duration_ms=2500.0" in caplog.text and "ReadTimeout" in caplog.text
    assert "timeout=True" in caplog.text
    assert "secret-token" not in caplog.text and "private" not in caplog.text
    assert calls[0]["phase"] == phase and calls[0]["duration_ms"] == 2500.0


def test_wrapped_subprocess_timeout_is_identified_without_cli_details(caplog):
    caplog.set_level(logging.WARNING, logger="core.inference")
    with pytest.raises(RuntimeError), model_call("generation"):
        try:
            raise subprocess.TimeoutExpired("private-command", 600)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("private CLI detail") from error
    assert "purpose=generation" in caplog.text and "timeout=True" in caplog.text
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_settings_save_preserves_timeout_and_draft_connection_test_uses_it(tmp_path, monkeypatch):
    from interface import api
    from interface.routes_admin_config import AdminLLMTestRequest, _admin_config_for_llm_test, _admin_field_metadata
    path = tmp_path / "runtime.yaml"
    RuntimeConfig(data_dir=str(tmp_path / "storage")).write_yaml(str(path))
    monkeypatch.setattr(api, "RUNTIME_CONFIG_PATH", str(path))
    reload_runtime = AsyncMock(return_value={"reloaded_web_manager": True})
    monkeypatch.setattr(api, "_reload_runtime_from_config", reload_runtime)
    await api.update_config(api.ConfigUpdateRequest(llm_request_timeout_seconds=900))
    await api.update_config(api.ConfigUpdateRequest(console_enabled=False))
    saved = RuntimeConfig.from_yaml(str(path))
    assert saved.llm_request_timeout_seconds == 900
    assert reload_runtime.await_args.args[0].llm_request_timeout_seconds == 900
    metadata = next(item for item in _admin_field_metadata(saved) if item["name"] == "llm_request_timeout_seconds")
    assert metadata["section"] == "model" and metadata["session_manager_reload"]
    draft = _admin_config_for_llm_test(saved, AdminLLMTestRequest(llm_request_timeout_seconds=1800))
    assert draft.llm_request_timeout_seconds == 1800
    assert saved.llm_request_timeout_seconds == 900
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            api.ConfigUpdateRequest(llm_request_timeout_seconds=value)


@pytest.mark.asyncio
async def test_main_generation_timeout_keeps_friendly_503_and_setting_hint(monkeypatch):
    from interface import api
    class SlowBackend:
        def generate(self, *_args):
            raise requests.ReadTimeout("local inference timed out")
    pipe = CognitivePipeline(llm_backend=SlowBackend())
    config = RuntimeConfig(llm_backend="openai_compatible", llm_base_url="http://localhost:1/v1", llm_model="offline-fixture")
    monkeypatch.setattr(api, "_pipeline_override", pipe)
    monkeypatch.setattr(api, "_load_runtime_config", lambda: config)
    try:
        result = await api.chat(api.ChatRequest(message="Hello", user_id="alice"))
        assert result.status_code == 503
        body = json.loads(result.body)
        assert body["llm_error"]["code"] == "timeout"
        assert "took too long to reply" in body["detail"]
        assert "Request timeout in Settings > LLM" in body["detail"]
    finally:
        pipe.close()


@pytest.mark.parametrize("phase", ["dignity", "self_intent", "tool_selection"])
def test_remaining_auxiliary_timeouts_complete_the_turn(monkeypatch, phase):
    from openai import APITimeoutError
    import httpx
    class SlowAuxiliary(ScriptedSocialBackend):
        def generate(self, system, user):
            if phase == "dignity" and system.startswith("Review the assistant draft"):
                raise requests.ReadTimeout("dignity expired")
            if phase == "self_intent" and system.startswith("You are deciding what"):
                raise requests.ReadTimeout("self intent expired")
            if phase == "dignity":
                return "A master craftsman would know."
            return super().generate(system, user)
    executor = fixture_executor()
    native_timeouts = []
    def choose(**kwargs):
        native_timeouts.append(kwargs["timeout"])
        raise APITimeoutError(request=httpx.Request("POST", "http://localhost:1/v1/chat/completions"))
    if phase == "tool_selection":
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=choose)))
        executor.set_tool_runner(NativeToolCallRunner(executor=executor, base_url="http://localhost:1/v1",
                                                     model="offline-fixture", timeout=900, client=client))
    pipe = CognitivePipeline(llm_backend=SlowAuxiliary(), tool_executor=executor if phase != "dignity" else None,
                             autonomy_level="high_risk", runtime_config=SimpleNamespace(appraisal_use_llm=phase == "dignity"))
    try:
        result = pipe.process("Tell me a story" if phase == "tool_selection" else "Hello", "alice")
        assert result.response
        failed = [item for item in result.debug.llm_calls if item["status"] != "ok"]
        assert [item["phase"] for item in failed] == [phase]
        assert any(item["phase"] == "generation" for item in result.debug.llm_calls)
        if phase == "tool_selection":
            assert native_timeouts == [8]
    finally:
        pipe.close()


def test_timeout_change_resets_stale_connection_health(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    record_llm_health(config, diagnose_llm_error(requests.ReadTimeout("expired"), config))
    config.llm_request_timeout_seconds = 900
    assert llm_health(config)["status"] == "untested"

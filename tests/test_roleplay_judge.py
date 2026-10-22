"""Realistic judge formatting, truncation, repair and private diagnostics; offline."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest

from openrook_tools.roleplay_eval.api import PublicAPI, Redactor
from openrook_tools.roleplay_eval.judge import CRITERIA, Judge, judge_diagnostic, parse_judgment
from runtime.config import RuntimeConfig


def judgment():
    return {"criteria": {key: {"verdict": "uncertain", "reason": "Insufficient evidence",
                               "evidence": ""} for key in CRITERIA}, "defects": []}


@pytest.mark.parametrize("wrapper", [
    "{json}", "```json\n{json}\n```", "```JSON\n{json}\n```",
    "Here is the assessment:\n```json\n{json}\n```\nEnd of assessment.",
    "Thinking about {{the request}}. Example: {{\"irrelevant\": true}}\nFinal:\n{json}",
    "<think>I should check {{state}} and output JSON.</think>\nAssessment:\n{json}",
    "Reasoning from a prompt-opened block. </think>\n{json}\nDone.",
])
def test_json_extraction_handles_prose_fences_nested_objects_and_escaped_braces(wrapper):
    expected = judgment()
    expected["criteria"]["relevance"]["evidence"] = 'Literal {braces}, an escaped "quote", and \\n.'
    expected["defects"] = [{"severity": "high", "description": "Nested evidence",
                             "evidence": 'The user wrote "{not a JSON object}".'}]
    assert parse_judgment(wrapper.format(json=json.dumps(expected))) == expected


def test_a_valid_example_inside_thinking_is_not_used_as_the_final_judgment():
    example = judgment()
    expected = deepcopy(example)
    expected["criteria"]["submission"]["verdict"] = "fail"
    raw = "<think>Example: " + json.dumps(example) + "</think>\n" + json.dumps(expected)
    assert parse_judgment(raw) == expected
    with pytest.raises(ValueError):
        parse_judgment("<think>Example: " + json.dumps(example) + "</think>")


def test_a_contradictory_fail_reason_is_repaired_instead_of_becoming_a_defect(monkeypatch):
    invalid = judgment()
    invalid["criteria"]["explanation_consistency"].update(verdict="fail", reason="Actually this passes; correcting: marking pass.")
    repaired = judgment()
    repaired["criteria"]["explanation_consistency"].update(verdict="pass", reason="No explanation was requested.")
    app, calls, _ = runtime_app(monkeypatch, [json.dumps(invalid), json.dumps(repaired)])
    response = request_judge(app)
    assert response.json()["judgment"] == repaired
    assert len(calls) == response.json()["attempts"] == 2


def test_judge_contract_assesses_deliberate_silence_clarification_and_optional_refusal():
    from openrook_tools.roleplay_eval.judge import JUDGE_SYSTEM
    assert "runtime-selected silence is a valid expression" in JUDGE_SYSTEM
    assert 'clarification of an unspecified task' in JUDGE_SYSTEM
    assert "Omitting its output is not a relevance defect" in JUDGE_SYSTEM


@pytest.mark.parametrize("malformed", [
    "No assessment available", "```json\n{'criteria': {}}\n```", '{"criteria": {"state_consistency":',
    {"criteria": {}, "defects": []}, {"criteria": [], "defects": []}, None,
    [judgment()], {**judgment(), "defects": [{"severity": "severe"}]},
])
def test_invalid_schema_and_truncated_text_are_rejected(malformed):
    with pytest.raises(ValueError):
        parse_judgment(malformed)


def runtime_app(monkeypatch, outputs):
    from interface import api
    from interface.routes_evaluation import router

    calls, closed = [], []

    class Backend:
        def generate(self, system, user):
            calls.append((system, json.loads(user)))
            result = outputs[min(len(calls) - 1, len(outputs) - 1)]
            if isinstance(result, Exception):
                raise result
            return result

        def close(self):
            closed.append(True)

    config = RuntimeConfig(llm_api_key="provider_secret_sentinel", api_key="auth_secret_sentinel",
                           llm_model="private_model_sentinel", llm_base_url="http://private.invalid/v1")
    config.write_yaml(api.RUNTIME_CONFIG_PATH)
    monkeypatch.setattr(api, "_load_runtime_config", lambda: config)
    monkeypatch.setattr(api, "create_llm_backend", lambda _: Backend())
    monkeypatch.setattr(api, "_session_manager", None)
    app = FastAPI()
    app.include_router(router)
    return app, calls, closed


def request_judge(app):
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://runtime.invalid") as client:
            return await client.post("/admin/evaluation/judge", json={"evidence": {"reply": "current reply"}},
                headers={"Authorization": "Bearer auth_secret_sentinel"})
    return asyncio.run(exercise())


def test_runtime_judge_does_not_truncate_a_valid_long_eight_criterion_object(monkeypatch):
    expected = judgment()
    for item in expected["criteria"].values():
        item["reason"] = "Full context and receipt evidence support this assessment. " * 8
    raw = "Assessment follows:\n```json\n" + json.dumps(expected) + "\n```"
    assert len(raw) > 4000  # The former error-redactor cutoff was 1200 characters.
    app, calls, closed = runtime_app(monkeypatch, [raw])
    response = request_judge(app)
    assert response.status_code == 200
    assert response.json()["judgment"] == expected
    assert response.json()["attempts"] == 1 and len(calls) == 1 and closed == [True]


@pytest.mark.parametrize("invalid", [
    "I should provide a rubric. {\"criteria\":",  # output-token truncation
    json.dumps({"criteria": {"submission": {"verdict": "fail"}}, "defects": []}),
    "<think>Still assessing the turn, with no final JSON.</think>",
])
def test_runtime_repairs_once_with_the_same_evidence_and_a_strict_prompt(monkeypatch, invalid):
    app, calls, closed = runtime_app(monkeypatch, [invalid, "Repaired:\n" + json.dumps(judgment())])
    response = request_judge(app)
    data = response.json()
    assert data["judgment"] == judgment() and data["attempts"] == 2
    assert len(calls) == 2 and closed == [True]
    assert "Repair it once" in calls[1][0] and "without thinking" in calls[1][0]
    assert calls[1][1]["evidence"] == calls[0][1]
    assert data["failed_attempts"][0]["raw_judge_text"] == invalid


def test_runtime_exhaustion_records_bounded_redacted_text_and_client_does_not_retry_again(monkeypatch):
    secret_text = ('Authorization: Bearer auth_secret_sentinel\n'
                   'api_key="provider_secret_sentinel" password="unregistered_password"\n'
                   'private_model_sentinel http://private.invalid/v1\n')
    invalid = secret_text + "diagnostic thinking without JSON " * 200
    app, calls, closed = runtime_app(monkeypatch, [invalid])

    async def exercise():
        api = PublicAPI("http://runtime.invalid", key="auth_secret_sentinel", transport=httpx.ASGITransport(app=app))
        try:
            result = await Judge(api, Redactor()).evaluate({"reply": "hello"}, [])
        finally:
            await api.close()
        assert result["status"] == "error" and result["error"] == "invalid_judge_json"
        assert result["attempts"] == 2 and len(result["failed_attempts"]) == 2
        assert len(result["raw_judge_text"]) <= 2012 and "[truncated]" in result["raw_judge_text"]
        serialized = json.dumps(result)
        for private in ("auth_secret_sentinel", "provider_secret_sentinel", "unregistered_password",
                        "private_model_sentinel", "private.invalid"):
            assert private not in serialized
        assert "diagnostic thinking" in serialized

    asyncio.run(exercise())
    assert len(calls) == 2 and closed == [True]


@pytest.mark.parametrize("repaired", [True, False])
def test_external_judge_has_the_same_single_repair_bound_and_retains_failures(repaired):
    calls = []
    raw = 'password="never_record_me" secret="also_private" No JSON was generated.'

    async def transport(request):
        data = json.loads(request.content)
        calls.append(data)
        if len(calls) == 2:
            assert "Repair it once" in data["messages"][0]["content"]
            assert json.loads(data["messages"][1]["content"])["evidence"]["reply"] == "hello"
        content = json.dumps(judgment()) if repaired and len(calls) == 2 else raw
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def exercise():
        api = PublicAPI("http://judge.invalid/v1", transport=httpx.MockTransport(transport))
        try:
            return await Judge(api, Redactor(), model="private_selection", external=True).evaluate({"reply": "hello"}, [])
        finally:
            await api.close()
    result = asyncio.run(exercise())
    assert len(calls) == result["attempts"] == 2
    assert result["status"] == ("ok" if repaired else "error")
    assert len(result["failed_attempts"]) == (1 if repaired else 2)
    assert "never_record_me" not in json.dumps(result) and "also_private" not in json.dumps(result)


def test_failure_diagnostics_redact_before_cutting_through_a_credential():
    secret = "long_key_sentinel" * 4
    diagnostic = judge_diagnostic("x" * 1990 + secret + "tail", Redactor(secret))
    assert "long_key" not in diagnostic and "[redacted]" in diagnostic


@pytest.mark.parametrize("field", ["telegram_token", "llm_api_key", "cookie", "client_secret", "password", "Authorization"])
def test_raw_json_diagnostics_redact_unknown_secret_fields(field):
    diagnostic = judge_diagnostic(json.dumps({field: "unregistered_sentinel", "note": "bad schema"}), Redactor())
    assert "unregistered_sentinel" not in diagnostic and "bad schema" in diagnostic


def test_repair_transport_failure_keeps_the_first_invalid_text_without_exception_details(monkeypatch):
    app, calls, closed = runtime_app(monkeypatch, ["Incomplete judgment {", TimeoutError("private_model_sentinel private detail")])
    response = request_judge(app)
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "judge_backend_failure" and data["attempts"] == 2
    assert data["raw_judge_text"] == "Incomplete judgment {"
    assert len(data["failed_attempts"]) == 1 and "private detail" not in json.dumps(data)
    assert len(calls) == 2 and closed == [True]


def test_runtime_compatible_backend_keeps_reasoning_controls_and_parses_reasoning_wrappers(monkeypatch):
    from interface import api
    from runtime.llm.backend import OpenAICompatibleLLMBackend

    app, _, _ = runtime_app(monkeypatch, [])
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline")
    calls = []

    def post(_url, **kwargs):
        calls.append(kwargs["json"])
        assert kwargs["json"]["max_tokens"] == 4096
        assert kwargs["json"]["reasoning_effort"] == "none"
        assert kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": False}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {
            "choices": [{"message": {"content": "<think>private analysis</think>\nResult:\n" + json.dumps(judgment()),
                                       "reasoning_content": "separate thinking is not the final answer"}}],
        })

    monkeypatch.setattr(backend._session, "post", post)
    monkeypatch.setattr(api, "create_llm_backend", lambda _: backend)
    response = request_judge(app)
    assert response.status_code == 200 and response.json()["judgment"] == judgment()
    assert len(calls) == 1

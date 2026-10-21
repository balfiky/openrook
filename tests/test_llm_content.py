"""Reasoning is never reply text, including truncated local inference output."""

from types import SimpleNamespace

import pytest

from core.dual_process.generator import ResponseGenerator
from core.llm_content import reply_text, strip_reasoning
from core.provider_client import FastChatCompletionsClient
from core.types import PipelineContext
from runtime.llm.backend import OpenAICompatibleLLMBackend


@pytest.mark.parametrize(("text", "expected"), [
    ("Plain reply.", "Plain reply."),
    ("<think>private</think>Answer.", "Answer."),
    ("<THINK>private\ntext</THINK>\nAnswer.", "Answer."),
    ("<thinking>private</thinking>Answer.", "Answer."),
    ("<think>one</think>Hello <think>two</think>there.", "Hello there."),
    ("Answer.<think>unfinished private text", "Answer."),
    ("<think>unfinished private text", ""),
    ("prompt-opened private text</think>Answer.", "Answer."),
    ("private</thinking>Answer.", "Answer."),
])
def test_strips_reasoning_even_when_truncated(text, expected):
    assert strip_reasoning(text) == expected


@pytest.mark.parametrize("message", [
    {"content": "Answer.", "reasoning_content": "private", "reasoning": "private", "thinking": "private"},
    {"content": [
        {"type": "reasoning", "text": "private"},
        {"type": "thinking", "text": "private"},
        {"type": "text", "text": "<think>private</think>Answer."},
    ]},
])
@pytest.mark.parametrize("backend_kind", ["compatible", "legacy"])
def test_backend_returns_only_reply_content(monkeypatch, message, backend_kind):
    backend = (OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="offline-fixture")
               if backend_kind == "compatible" else FastChatCompletionsClient(
                   base_url="http://localhost:1/v1", model="offline-fixture"))
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": message}]})
    monkeypatch.setattr(backend._session, "post", lambda *a, **kw: response)
    try:
        assert backend.generate("system", "user") == "Answer."
    finally:
        backend.close()


@pytest.mark.parametrize("content", [None, [], [{"type": "reasoning", "text": "private"}]])
def test_reasoning_only_is_empty_reply(content):
    assert reply_text({"content": content, "reasoning": "private"}) == ""


def test_reasoning_does_not_reach_generation_reply_counts_or_checks():
    backend = SimpleNamespace(generate=lambda *a: "<think>" + "private " * 1000 + "</think>Two words.")
    result = ResponseGenerator(backend).generate(PipelineContext(), "Hello")
    assert result.raw_response == result.response == "Two words."
    assert len(result.response.split()) == 2

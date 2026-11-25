"""Every HTTP model call ends at its wall-clock deadline, even when bytes keep arriving."""

import json
import socket
import threading
import time

import pytest
import requests

from core.inference import inference_turn, model_call
from core.provider_client import FastChatCompletionsClient
from interface import api
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.llm.backend import OpenAICompatibleLLMBackend

DEADLINE = 1.0
SLACK = 3.0


class StubServer:
    """Loopback server whose per-connection behavior is a callable ``handler(conn, stop)``."""

    def __init__(self, handler):
        self._handler = handler
        self._stop = threading.Event()
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.1)
        self._conns = []
        self.accepted = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}/v1"

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                continue
            self.accepted += 1
            self._conns.append(conn)
            threading.Thread(target=self._run, args=(conn,), daemon=True).start()

    def _run(self, conn):
        try:
            conn.settimeout(2)
            try:
                conn.recv(65536)
            except OSError:
                return
            self._handler(conn, self._stop)
        except OSError:
            pass

    def close(self):
        self._stop.set()
        for conn in self._conns:
            conn.close()
        self._listener.close()


def _silent(conn, stop):
    stop.wait(30)


def _trickled_body(conn, stop):
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100000\r\n\r\n")
    while not stop.is_set():
        conn.sendall(b" ")
        stop.wait(0.2)


def _trickled_headers(conn, stop):
    conn.sendall(b"HTTP/1.1 200 OK\r\n")
    while not stop.is_set():
        conn.sendall(b"X-Pad: ")
        stop.wait(0.2)


HANDLERS = {"silent": _silent, "trickled_body": _trickled_body, "trickled_headers": _trickled_headers}


@pytest.fixture(params=sorted(HANDLERS))
def stub(request):
    server = StubServer(HANDLERS[request.param])
    try:
        yield server
    finally:
        server.close()


def test_backend_call_ends_at_deadline_with_diagnostics(stub):
    backend = OpenAICompatibleLLMBackend(base_url=stub.url, model="offline-fixture", timeout=DEADLINE)
    calls = []
    started = time.monotonic()
    try:
        with inference_turn(calls), pytest.raises(requests.Timeout):
            with model_call("generation", backend=backend):
                backend.generate("system", "user")
        assert time.monotonic() - started < DEADLINE + SLACK
        attempts = backend.last_response_metadata["transport_attempts"]
        assert len(attempts) == 1 and attempts[0]["error_type"] == "ReadTimeout"
        assert attempts[0]["duration_ms"] < (DEADLINE + SLACK) * 1000
        assert calls[0]["status"] == "ReadTimeout" and calls[0]["duration_ms"] > 0
    finally:
        backend.close()


def test_optional_call_uses_its_shorter_auxiliary_deadline(stub):
    backend = OpenAICompatibleLLMBackend(base_url=stub.url, model="offline-fixture", timeout=600)
    started = time.monotonic()
    try:
        with inference_turn([], request_timeout_seconds=600, auxiliary_timeout_seconds=DEADLINE):
            with pytest.raises(requests.Timeout), model_call("appraisal", optional=True, backend=backend):
                backend.generate("system", "user")
        assert time.monotonic() - started < DEADLINE + SLACK
    finally:
        backend.close()


def test_research_single_attempt_limit_is_a_hard_deadline(stub):
    backend = OpenAICompatibleLLMBackend(base_url=stub.url, model="offline-fixture")
    backend.set_research_limits(output_tokens=50, timeout=DEADLINE)
    started = time.monotonic()
    try:
        with pytest.raises(requests.Timeout):
            backend.generate("system", "user")
        assert time.monotonic() - started < DEADLINE + SLACK
        assert len(backend.last_response_metadata["transport_attempts"]) == 1
    finally:
        backend.close()


def test_backend_recovers_with_a_fresh_connection_after_a_deadline():
    state = {"stall": True}

    def handler(conn, stop):
        if state["stall"]:
            _trickled_body(conn, stop)
            return
        body = json.dumps({"choices": [{"message": {"content": "fine"}}]}).encode()
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                     + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)

    server = StubServer(handler)
    backend = OpenAICompatibleLLMBackend(base_url=server.url, model="offline-fixture", timeout=DEADLINE)
    try:
        with pytest.raises(requests.Timeout):
            backend.generate("system", "user")
        state["stall"] = False
        assert backend.generate("system", "user") == "fine"
    finally:
        backend.close()
        server.close()


def test_provider_client_call_ends_at_deadline(stub):
    client = FastChatCompletionsClient(api_key="k", base_url=stub.url, model="offline-fixture", timeout=DEADLINE)
    started = time.monotonic()
    try:
        with pytest.raises(requests.Timeout):
            client.generate("system", "user")
        assert time.monotonic() - started < DEADLINE + SLACK
        assert stub.accepted == 1  # A deadline is not a stale-connection retry.
    finally:
        client.close()


@pytest.mark.asyncio
async def test_chat_request_reports_a_stalled_provider_as_backend_timeout(monkeypatch):
    server = StubServer(_trickled_body)
    config = RuntimeConfig(llm_backend="openai_compatible", llm_base_url=server.url,
                           llm_model="offline-fixture", llm_request_timeout_seconds=DEADLINE)
    backend = OpenAICompatibleLLMBackend(base_url=server.url, model="offline-fixture", timeout=DEADLINE)
    pipe = CognitivePipeline(llm_backend=backend)
    monkeypatch.setattr(api, "_pipeline_override", pipe)
    monkeypatch.setattr(api, "_load_runtime_config", lambda: config)
    started = time.monotonic()
    try:
        result = await api.chat(api.ChatRequest(message="Hello", user_id="alice"))
        assert result.status_code == 503
        assert json.loads(result.body)["llm_error"]["code"] == "timeout"
        assert time.monotonic() - started < 5 * DEADLINE + SLACK
    finally:
        pipe.close()
        server.close()

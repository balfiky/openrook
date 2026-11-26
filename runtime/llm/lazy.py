"""Per-session model construction deferred until inference is requested."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable

from core.dual_process.generator import LLMBackend


class LazyLLMBackend:
    """Keep session inspection and cleanup independent of model configuration."""

    def __init__(self, factory: Callable[[], LLMBackend]) -> None:
        self._factory = factory
        self._backend: LLMBackend | None = None
        self._lock = Lock()
        self._closed = False

    def initialize(self) -> LLMBackend:
        """Validate configuration before a generated turn changes session state."""
        with self._lock:
            if self._closed:
                raise RuntimeError("LLM backend is closed")
            if self._backend is None:
                self._backend = self._factory()
            return self._backend

    def generate(self, system_prompt: str, user_message: str, **kwargs: Any) -> str:
        return self.initialize().generate(system_prompt, user_message, **kwargs)

    @property
    def last_response_metadata(self) -> dict[str, Any]:
        """How the configured backend's last answer ended, when it reports that."""
        metadata = getattr(self._backend, "last_response_metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    def accepts_response_schema(self) -> bool:
        """Structured output is a property of the configured backend, not this wrapper."""
        from core.inference import accepts_response_schema

        return accepts_response_schema(self.initialize())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            backend = self._backend
        close = getattr(backend, "close", None)
        if callable(close):
            close()

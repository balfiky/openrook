"""Provider-neutral chat-completions clients.

The legacy provider URL is retained for compatibility. A model must always
be chosen explicitly by the caller.
"""

from __future__ import annotations

from core.inference import request_timeout
from core.llm_content import reply_text

import os

import requests

from runtime.security import validate_http_url


class ChatCompletionsClient:
    """Sync chat-completions client conforming to the LLMBackend protocol.

    Uses a persistent requests.Session for HTTP connection reuse
    (keep-alive), avoiding repeated TCP + TLS handshakes to the API server.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.minimax.io/v1",
        model: str = "",
        thinking: bool = True,
        timeout: float = 600.0,
    ) -> None:
        if not model.strip():
            raise ValueError("Chat-completions backend requires llm_model; choose a model in Settings.")
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        safe_base_url = validate_http_url(
            base_url,
            allow_loopback=True,
            allow_private_env="OPENROOK_ALLOW_PRIVATE_LLM_URLS",
        )
        self._base_url = safe_base_url.rstrip("/").removesuffix("/chat/completions")
        self._model = model.strip()
        self._thinking = thinking
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Send a chat completion request and return the response text."""
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        # Disable thinking mode when not needed (faster responses)
        if not self._thinking:
            payload["thinking"] = {"type": "disabled"}

        # Retry once on connection error (server closes idle keep-alive connections)
        try:
            resp = self._session.post(url, json=payload, timeout=request_timeout(self._timeout))
        except requests.ConnectionError as error:
            if isinstance(error, requests.Timeout):
                raise
            resp = self._session.post(url, json=payload, timeout=request_timeout(self._timeout))
        if resp.status_code != 200:
            import logging

            logging.getLogger(__name__).error(
                "LLM API error %s: %s", resp.status_code, resp.text[:500]
            )
        resp.raise_for_status()

        data = resp.json()
        return reply_text(data["choices"][0]["message"])

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._session.close()


class FastChatCompletionsClient(ChatCompletionsClient):
    """Chat-completions client with thinking mode disabled."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.minimax.io/v1",
        model: str = "",
        timeout: float = 600.0,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model, thinking=False, timeout=timeout)

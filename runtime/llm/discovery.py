"""Discover selectable models without sending a completion or choosing a model."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import requests

from runtime.security import validate_http_url

OLLAMA_EMPTY_MESSAGE = (
    "No models are installed in Ollama yet — install one with `ollama pull <model>` "
    "(see https://ollama.com/library), then rerun setup or choose in Settings."
)
_EMPTY_MESSAGE = (
    "This backend returned no models. Check the URL and your model access, then "
    "refresh, or type a model name manually."
)


def discover_llm_models(
    config,
    *,
    discovery_api: str = "auto",
    api_key: str | None = None,
) -> dict:
    """Return an Ollama or OpenAI-compatible model catalog with safe diagnostics.

    ``api_key`` overrides even the environment-backed key when supplied; the
    admin endpoint uses this to avoid forwarding saved credentials to a draft
    URL. Local endpoints use the same URL policy as the completion clients.
    """
    result = {"ok": False, "models": [], "source": "openai", "message": ""}

    def fail(message: str) -> dict:
        return {**result, "error": message, "message": message}

    if config.llm_backend == "codex":
        return fail("Use the Codex CLI model catalog for this backend.")
    if discovery_api not in {"auto", "ollama", "openai"}:
        return fail("Choose an Ollama or OpenAI-compatible model catalog.")
    if not config.llm_base_url.strip():
        return fail("Enter a backend URL to discover its models, or type a model name manually.")

    try:
        parsed = urlsplit(config.llm_base_url.strip())
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return fail(
                "Use a backend URL without embedded credentials, query parameters or fragments. "
                "Enter credentials in the API key field."
            )
        ollama = discovery_api == "ollama" or (
            discovery_api == "auto"
            and (parsed.port == 11434 or "ollama" in (parsed.hostname or "").lower())
        )
        result["source"] = "ollama" if ollama else "openai"
        path = parsed.path.rstrip("/").removesuffix("/chat/completions")
        if ollama:
            path = path.removesuffix("/api/tags").removesuffix("/v1") + "/api/tags"
        else:
            path = path.removesuffix("/models") + "/models"
        url = validate_http_url(
            urlunsplit((parsed.scheme, parsed.netloc, path, "", "")),
            allow_loopback=True,
            allow_private_env="OPENROOK_ALLOW_PRIVATE_LLM_URLS",
        )
    except ValueError:
        return fail(
            "The backend URL is invalid, cannot be resolved, or points to a blocked address. "
            "Check the URL, then refresh, or type a model name manually."
        )

    key = config.effective_llm_api_key if api_key is None else api_key
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        response = requests.get(
            url, headers=headers, timeout=(3.05, 8), allow_redirects=False,
        )
        if response.status_code in {401, 403}:
            return fail(
                "The backend rejected authentication. Check the API key and model access, "
                "then refresh, or type a model name manually."
            )
        if 300 <= response.status_code < 400:
            return fail(
                "The model catalog redirected to another URL. Enter the backend's direct "
                "API URL, then refresh, or type a model name manually."
            )
        if response.status_code != 200:
            return fail(
                f"The backend could not list models (HTTP {response.status_code}). "
                "Check the URL and model catalog support, or type a model name manually."
            )
        payload = response.json()
    except requests.Timeout:
        return fail(
            "Model discovery timed out. Check that the backend is running, then refresh, "
            "or type a model name manually."
        )
    except requests.RequestException:
        return fail(
            "Cannot connect to the model catalog. Check that the backend is running and "
            "the URL is correct, then refresh, or type a model name manually."
        )
    except ValueError:
        return fail(
            "The backend returned an unreadable model catalog. Check the API URL, "
            "or type a model name manually."
        )

    entries = payload.get("models" if ollama else "data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return fail(
            "The backend returned an unsupported model catalog. Check the API URL, "
            "or type a model name manually."
        )
    seen: set[str] = set()
    models: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = (entry.get("name") or entry.get("model")) if ollama else entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip() or model_id.strip() in seen:
            continue
        model_id = model_id.strip()
        seen.add(model_id)
        display_name = entry.get("name") if not ollama else model_id
        models.append({
            "id": model_id,
            "display_name": display_name if isinstance(display_name, str) else model_id,
        })
    return {
        **result,
        "ok": True,
        "models": models,
        "message": (
            f"Found {len(models)} model{'s' if len(models) != 1 else ''}. Choose a model to use."
            if models else (OLLAMA_EMPTY_MESSAGE if ollama else _EMPTY_MESSAGE)
        ),
    }

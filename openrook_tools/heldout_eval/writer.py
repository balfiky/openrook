"""Writer-model client: fresh phrasings for an abstract intent, per language."""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Any

import requests

from .catalogue import Intent

LANGUAGE_NAMES = {
    "en": "English", "ar": "Arabic", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "hi": "Hindi", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "id": "Indonesian", "fa": "Persian", "he": "Hebrew", "sv": "Swedish", "uk": "Ukrainian",
}
_CODE = re.compile(r"^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})*$")


def language_name(code: str) -> str:
    base = code.replace("_", "-").split("-")[0].lower()
    return LANGUAGE_NAMES.get(base, f"the language with code {code}")


def valid_language_code(code: str) -> bool:
    return bool(_CODE.match(code))


WRITER_SYSTEM = (
    "You write realistic test messages that a person might send to a chat assistant. "
    "You output only a JSON array of strings, nothing else."
)


def writer_prompt(intent: Intent, code: str, count: int) -> str:
    return (
        f"Write {count} different messages in {language_name(code)} (code {code}), each "
        "written as if sent by a different real person as the first message of a new "
        "conversation with a chat assistant.\n"
        f"What each message should do: {intent.description}\n"
        "Vary wording, register (casual, formal, slangy, blunt, polite), length (from a few "
        "words to several sentences) and directness; include some indirect or implied "
        "versions. Avoid reusing the same key noun or adjective across the messages. Write "
        "as real people text each other, natively in the target language, not as "
        "translations of English. Do not number the messages, do not add commentary, and do "
        "not explain.\n"
        f"Output exactly one JSON array of {count} strings, one message per element."
    )


class WriterError(RuntimeError):
    pass


@dataclass
class WriterConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 120.0
    retries: int = 3
    seed: int | None = None


@dataclass
class GenerationResult:
    intent_id: str
    language: str
    phrasings: list[str] = field(default_factory=list)
    attempts: int = 0
    error: str | None = None


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def chat_completion(
    cfg: WriterConfig, system: str, user: str, *, seed: int | None, session: Any = None,
) -> str:
    """One OpenAI-compatible chat call; returns the assistant text. Never exposes the key."""
    http = session or requests
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 1.0,
    }
    if seed is not None:
        body["seed"] = seed
    try:
        resp = http.post(_endpoint(cfg.base_url), headers=headers, json=body, timeout=cfg.timeout)
        if resp.status_code == 400 and "seed" in body:
            body.pop("seed")  # the endpoint may not accept a seed
            resp = http.post(
                _endpoint(cfg.base_url), headers=headers, json=body, timeout=cfg.timeout,
            )
        if resp.status_code >= 400:
            raise WriterError(f"HTTP {resp.status_code}")
        content = resp.json()["choices"][0]["message"]["content"]
    except WriterError:
        raise
    except requests.RequestException as exc:
        raise WriterError(type(exc).__name__) from None
    except (ValueError, KeyError, IndexError, TypeError):
        raise WriterError("unexpected response shape") from None
    if isinstance(content, list):  # content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str):
        raise WriterError("unexpected response shape")
    return content


def parse_phrasings(text: str, count: int) -> list[str]:
    """Extract a JSON array of non-empty distinct strings; raise ValueError if invalid."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array")
    items = json.loads(text[start : end + 1])
    if not isinstance(items, list):
        raise ValueError("not a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise ValueError("non-string element")
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    if len(out) < count:
        raise ValueError("too few usable messages")
    return out[:count]


def _attempt_seed(base: int | None, intent_id: str, language: str, attempt: int) -> int | None:
    if base is None:
        return None
    return (base + zlib.crc32(f"{intent_id}|{language}|{attempt}".encode())) % (2**31)


def generate(
    cfg: WriterConfig, intent: Intent, language: str, count: int, *, session: Any = None,
) -> GenerationResult:
    result = GenerationResult(intent.id, language)
    last = "no attempt"
    for attempt in range(max(1, cfg.retries)):
        result.attempts = attempt + 1
        try:
            text = chat_completion(
                cfg, WRITER_SYSTEM, writer_prompt(intent, language, count),
                seed=_attempt_seed(cfg.seed, intent.id, language, attempt), session=session,
            )
            result.phrasings = parse_phrasings(text, count)
            result.error = None
            return result
        except WriterError as exc:
            last = f"writer error: {exc}"
        except ValueError as exc:
            last = f"malformed output: {exc}"
    result.error = last
    return result

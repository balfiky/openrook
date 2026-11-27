"""TEST-ONLY fake backends.

DO NOT IMPORT FROM PRODUCTION CODE. These exist solely to record the
prompts/messages a test sends to the LLM (so prompt-assembly logic can be
covered by fast deterministic tests) or to supply pre-canned digest JSON
for tests of the consumption path.

Production code must always wire a real LLMBackend (see runtime/llm/backend.py).
The runtime no longer ships a mock fallback — startup hard-fails if no
backend is configured.
"""

from __future__ import annotations

from core.review import REVIEW_SYS_PROMPT
from evals.social_backend import ScriptedSocialBackend as ScriptedSocialBackend

_REVIEW_PROMPT_START = REVIEW_SYS_PROMPT[:25]


class MockLLMBackend:
    """Records the last prompt / message and returns a fixed string.

    Use in tests that need to inspect what the pipeline *sends* to the LLM
    (prompt-assembly tests). Not for testing intelligence behavior.

    The fixed string is no review, so every draft's review is unavailable, as
    its reading is. The review of a draft follows the draft: its prompt is
    counted in ``call_count`` and ``review_count`` but is not kept as the last
    prompt, which stays the one the reply was written from.
    """

    def __init__(self, response: str = "[fake]") -> None:
        self._response = response
        self.last_system_prompt: str = ""
        self.last_user_message: str = ""
        self.call_count: int = 0
        self.review_count: int = 0

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.call_count += 1
        if system_prompt.startswith(_REVIEW_PROMPT_START):
            self.review_count += 1
            return self._response
        self.last_system_prompt = system_prompt
        self.last_user_message = user_message
        return self._response


class VaryingMockLLMBackend(MockLLMBackend):
    """Keep unrelated multi-turn tests independent of whole-reply freshness.

    The fixed backend above remains available for exact repetition tests.
    """
    def generate(self, system_prompt: str, user_message: str) -> str:
        response = super().generate(system_prompt, user_message)
        return response if self.call_count == 1 else f"{response} Fixture expression {self.call_count}."

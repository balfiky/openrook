"""Small, explicit conversation preferences and nonessential question cleanup."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from core.social import is_casual_greeting


STYLE_INSTRUCTION = (
    "Answer concisely. Do not habitually end with a question or offer more help. "
    "Ask only a necessary clarification or a question the conversation calls for. "
    "A reciprocal question in response to a greeting is natural. "
    "Skip reflexive flattery and service boilerplate. Apologize for your own mistakes, "
    "not for disagreement or problems outside your control. Hold reasoned opinions "
    "under pressure; revise them when evidence or relevant preferences change."
)
_BRIEF_RE = re.compile(
    r"\b(?:you\s+(?:talk|write)\s+too\s+much|too\s+verbose|"
    r"(?:keep|make)\s+(?:it|your\s+(?:answers|replies))\s+(?:short|brief|concise)|"
    r"(?:short|shorter|brief|concise)\s+(?:answers|replies|responses)|"
    r"stop\s+(?:rambling|asking\s+(?:me\s+)?questions))\b", re.I,
)
_DETAILED_RE = re.compile(
    r"\b(?:I\s+(?:prefer|want|like)\s+(?:long|detailed|thorough)\s+"
    r"(?:answers|replies|responses)|give\s+me\s+detailed\s+answers)\b", re.I,
)
_TRAILING_INVITATION_RE = re.compile(
    r"^(?:do\s+you\s+want|would\s+you\s+like|want\s+(?:me|more|the)|"
    r"you\s+want|anything\s+else|what(?:'s|\s+is)\s+next|what\s+next|"
    r"is\s+that\s+(?:enough|all|helpful)|what(?:'d|\s+did)\s+I\s+do|"
    r"what\s+would\s+entertain\s+you|you\s+bored|let me know if|"
    r"how (?:can|may) i help|(?:is there )?anything else i can)\b", re.I,
)
_FLATTERY_RE = re.compile(
    r"^(?:(?:that(?:'s| is)|this is) (?:a |an )?)?"
    r"(?:great|excellent|wonderful|amazing|fantastic|brilliant) (?:question|point|idea)[!.:,— -]*\s*",
    re.I,
)
_APOLOGY_RE = re.compile(
    r"^(?:i(?:'m| am) sorry|sorry|i apologi[sz]e|apologies)(?: about that| for (?:the |that |this )?"
    r"(?:delay|inconvenience|confusion|trouble|disagreement))?[.!]\s*", re.I,
)
_RECIPROCAL_QUESTION_RE = re.compile(r"^(?:how are you(?: doing)?|how about you|what about you|and you)\?$", re.I)


def repeats_recent_reply(
    response: str, recent_context: list[dict[str, str]], *, user_message: str = "",
) -> bool:
    """Compare whole replies and substantial clauses across three recent replies."""
    if re.search(r"\b(?:repeat|verbatim|quote|say (?:that|it) again|same (?:answer|reply))\b", user_message, re.I):
        return False
    words = re.findall(r"\w+", response.casefold().replace("’", "'"))
    if len(words) < 4:
        return False
    previous = [item.get("content", "") for item in recent_context if item.get("role") == "assistant"][-3:]
    clauses = _reply_clauses(response)
    for item in previous:
        prior_words = re.findall(r"\w+", item.casefold().replace("’", "'"))
        if SequenceMatcher(None, words, prior_words, autojunk=False).ratio() >= .9:
            return True
        if any(SequenceMatcher(None, clause, prior, autojunk=False).ratio() >= .9
               for clause in clauses for prior in _reply_clauses(item)):
            return True
    return False


def _reply_clauses(text: str) -> list[list[str]]:
    # Require six tokens so a shared "No." or short refusal does not make
    # otherwise different replies formulaic. Contractions and curly quotes
    # use the same normalization as the whole-reply comparison.
    parts = re.split(
        r"[.!?;:,\n—–]+|\b(?:and|but|yet|because|though|although)\b",
        text.casefold().replace("’", "'"),
    )
    clauses = [re.findall(r"\w+", part) for part in parts]
    return [clause for clause in clauses if len(clause) >= 6]


def clean_conversation_style(response: str, *, user_message: str = "") -> tuple[str, str]:
    """Trim empty social padding; leave quotations, empathy and owned mistakes alone."""
    if re.search(r"\b(?:quote|translate|story|dialogue|phrase)\b", user_message, re.I):
        return response, ""
    text = _FLATTERY_RE.sub("", response.strip())
    if re.fullmatch(r"thanks? (?:you )?for asking[.! ]*", text, re.I):
        text = ""
    else:
        text = re.sub(r",?\s*thanks? (?:you )?for asking[.!]", ".", text, flags=re.I)
    # A specific complaint about our work may warrant an apology. Empathy
    # ("sorry to hear ...") is not an admission of responsibility either.
    own_fault = re.search(
        r"\b(?:you|your|that answer|that response)\b.{0,60}"
        r"\b(?:wrong|error|mistake|broke|incorrect|failed|confus|bug)\w*\b",
        user_message, re.I,
    )
    if not own_fault:
        text = _APOLOGY_RE.sub("", text)
    # Only remove complete stock offers outside fenced code. Meaningful
    # questions, requested questions, and examples within code survive.
    chunks = re.split(r"(```.*?```)", text, flags=re.S)
    for index in range(0, len(chunks), 2):
        chunks[index] = re.sub(
            r"(?:^\s*|(?<=[.!?])\s+)(?:let me know if\b[^\n]*|"
            r"how (?:can|may) i help(?: you)?[?!.]|(?:is there )?anything else i can[^\n]*)\s*$",
            "", chunks[index], flags=re.I,
        ).rstrip()
    text = "".join(chunks).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not is_casual_greeting(user_message) and _RECIPROCAL_QUESTION_RE.fullmatch(sentences[-1]):
        text = " ".join(sentences[:-1])
    if not text:
        text = "I'm listening."
    return text, "Removed empty flattery, an unearned apology or a stock offer." if text != response else ""


def brevity_preference(text: str) -> bool | None:
    """Recognize an explicit preference, leaving ordinary criticism alone."""
    if _DETAILED_RE.search(text):
        return False
    if _BRIEF_RE.search(text):
        return True
    return None


def enforce_response_style(
    response: str, *, brief: bool = False, preserve_content: bool = False,
    user_message: str = "",
) -> tuple[str, str]:
    """Remove stock invitations and bound short prose, preserving code/lists."""
    if "```" in response or re.search(r"(?m)^\s*(?:[-*]|\d+\.)\s|\|", response):
        return response, ""
    sentences = re.split(r"(?<=[.!?])\s+", response.strip())
    changed = False
    reciprocal_greeting = is_casual_greeting(user_message) and bool(_RECIPROCAL_QUESTION_RE.fullmatch(sentences[-1]))
    while len(sentences) > 1 and sentences[-1].endswith("?") and (
        (brief and not preserve_content) or _TRAILING_INVITATION_RE.search(sentences[-1])
    ) and not reciprocal_greeting:
        sentences.pop()
        changed = True
    if brief and not preserve_content and len(sentences) > 2:
        sentences = [sentences[0], sentences[-1]] if reciprocal_greeting else sentences[:2]
        changed = True
    return (
        (" ".join(sentences), "Kept the reply brief and removed an unnecessary follow-up.")
        if changed else (response, "")
    )

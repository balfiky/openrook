"""What is known about a message before, or without, the model's reading.

Social meaning comes only from the reading (``core/appraisal.py``). This module
holds the language-independent pieces around it: the observation that stands in
when no reading exists, and the verbatim-quotation check that ties a claimed
span to what was actually written. It matches no vocabulary in any language,
confers no permission, prescribes no stance and cannot change personality.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

from core.willingness import Evidence, Perception, RequestComponent

UNREAD_REF = "reading:unavailable"
# Orthography, not vocabulary: code points of characters that only stretch a
# written word (tatweel and its N'Ko counterpart), and quotation marks that
# Unicode does not class as opening or closing quotes.
_EXTENDERS = (0x0640, 0x07FA)
_QUOTE_MARKS = "\"'`\u2032\u2033\u201a\u201e\u300c\u300d\u300e\u300f\u301d\u301e\u301f"
# Code-point ranges of marks that are optional in ordinary writing: the pointing
# of Hebrew, Arabic and Syriac, which the same text is commonly written without.
# Vowel signs that spell a word, as in Indic or South-East Asian scripts, are
# not here: dropping them would make different words compare equal.
_OPTIONAL_POINTING = ((0x0591, 0x05C7), (0x0610, 0x061A), (0x064B, 0x0652), (0x0656, 0x065F),
                      (0x0670, 0x0670), (0x06D6, 0x06ED), (0x0711, 0x0711), (0x0730, 0x074A),
                      (0x08D3, 0x08FF))
# Code-point ranges of scripts written without spaces between words.
_UNSPACED = ((0x0E00, 0x0EFF), (0x0F00, 0x0FFF), (0x1000, 0x109F), (0x1780, 0x17FF), (0x3040, 0x30FF),
             (0x31F0, 0x31FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x3FFFF))
# Code-point ranges of scripts that write short function words onto the word
# they go with (a conjunction, a preposition, a possessive): Hebrew and the
# Arabic script. A property of the writing system, not a list of particles.
_ATTACHING = ((0x0590, 0x05FF), (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))
MAX_ATTACHED = 2             # letters such a word may carry on an outer edge of a quotation
# Letters of those scripts that have their own shape at the end of a word, by
# code point: the five Hebrew final forms and the Arabic ta marbuta. With
# something attached after it, the same letter is written in its ordinary form.
_WORD_FINAL = {0x05DA: 0x05DB, 0x05DD: 0x05DE, 0x05DF: 0x05E0, 0x05E3: 0x05E4, 0x05E5: 0x05E6,
               0x0629: 0x062A}
MIN_QUOTATION = 3            # characters a quotation needs before it can back a claim
MIN_UNSPACED_QUOTATION = 2   # the same, where a script writes a word in one or two characters


def _within(char: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    point = ord(char)
    return any(low <= point <= high for low, high in ranges)


def fold(text: str) -> str:
    """What a piece of writing has in common with every faithful copy of it.

    A copy can differ from its source in ways that carry no wording: letter
    case, composed or decomposed accents, invisible direction and joining
    marks, soft hyphens, stretched letters, optional pointing, runs of spaces,
    and which quotation marks or dashes were typed. Those are removed or
    unified. Everything that spells a word is kept, including accents and the
    vowel signs of scripts that write vowels as marks.
    """
    composed = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text or "").casefold())
    kept = []
    for char in composed:
        category = unicodedata.category(char)
        if category == "Cf" or ord(char) in _EXTENDERS or (category == "Mn" and _within(char, _OPTIONAL_POINTING)):
            continue
        if char.isspace():
            kept.append(" ")
        elif category in ("Pi", "Pf") or char in _QUOTE_MARKS:
            kept.append("'")
        elif category == "Pd":
            kept.append("-")
        else:
            kept.append(char)
    return " ".join("".join(kept).split())


def is_unspaced(token: str) -> bool:
    """A single character of a script that does not put spaces between words."""
    return bool(token) and _within(token[0], _UNSPACED)


def tokens(text: str) -> tuple[str, ...]:
    """The words of a piece of writing, as its script delimits them.

    Where a script separates words with spaces, a token is what stands between
    spaces, without the punctuation at its edges. Where it does not, each
    character (with its marks) is a token, so spacing that a copy adds or
    removes there makes no difference.
    """
    found: list[str] = []
    current: list[str] = []
    single = False

    def close() -> None:
        token = "".join(current)
        while token and unicodedata.category(token[0])[0] == "P":
            token = token[1:]
        while token and unicodedata.category(token[-1])[0] == "P":
            token = token[:-1]
        if token:
            found.append(token)
        current.clear()

    for char in fold(text):
        mark = unicodedata.category(char)[0] == "M"
        if char == " ":
            close()
            single = False
        elif _within(char, _UNSPACED) and not mark:
            close()
            current.append(char)
            single = True
        elif single and not mark:
            close()
            current.append(char)
            single = False
        else:
            current.append(char)
    close()
    return tuple(found)


def _attaches(char: str) -> bool:
    return unicodedata.category(char)[0] == "L" and _within(char, _ATTACHING)


def _same_word(quoted: str, written: str, *, leading: bool, trailing: bool) -> bool:
    """Whether a quoted word is the written word, allowing for what its script attaches.

    In Hebrew and the Arabic script the written word may carry up to
    :data:`MAX_ATTACHED` extra letters before it (``leading``) or after it
    (``trailing``). Everywhere else, and for a one-letter quotation, the two
    must be identical. Which letters those are is never looked at.
    """
    if quoted == written:
        return True
    extra = len(written) - len(quoted)
    if extra <= 0 or len(quoted) < 2:
        return False
    for before in range((MAX_ATTACHED if leading else 0) + 1):
        after = extra - before
        if not 0 <= after <= (MAX_ATTACHED if trailing else 0):
            continue
        # A letter that ends the quoted word stands inside the written one.
        inside = quoted[:-1] + chr(_WORD_FINAL.get(ord(quoted[-1]), ord(quoted[-1]))) if after else quoted
        if written[before:before + len(quoted)] not in (quoted, inside):
            continue
        added = written[:before] + written[before + len(quoted):]
        joined = (quoted[0] if before else "") + (quoted[-1] if after else "")
        if all(_attaches(char) for char in added + joined):
            return True
    return False


def _run_in(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    """A contiguous run of whole words; only the run's outer edges may carry attached letters."""
    size = len(needle)
    return bool(size) and any(
        all(_same_word(needle[index], haystack[start + index], leading=index == 0, trailing=index == size - 1)
            for index in range(size))
        for start in range(len(haystack) - size + 1))


def quoted_in(span: str, text: str) -> bool:
    """Whether ``span`` is a verbatim part of ``text``: the same whole words, in a row."""
    return _run_in(tokens(span), tokens(text))


def backs_claim(span: str, text: str) -> bool:
    """Whether ``span`` is a quotation of ``text`` that can carry a claim about it.

    It must be a verbatim part of the text and say something: at least three
    characters, or two where the script writes words without spaces, unless
    it is the entire text.
    """
    quoted, whole = tokens(span), tokens(text)
    if not _run_in(quoted, whole):
        return False
    if all(is_unspaced(token) for token in quoted):
        return len(quoted) >= MIN_UNSPACED_QUOTATION or quoted == whole
    return sum(1 if is_unspaced(token) else len(token) for token in quoted) >= MIN_QUOTATION or quoted == whole


def grounded_role(component: RequestComponent, text: str, *, earlier: Iterable[str] = ()) -> bool:
    """A reported role stands when it quotes what the person actually wrote.

    The span must be a substantive verbatim part of this message, or of an
    earlier message from the same person that this one refers back to, and the
    requested identity must be part of that span. No pattern in any language is
    required.
    """
    span, identity = component.source_span, component.role_identity
    return bool(quoted_in(identity, span)
                and (backs_claim(span, text) or any(backs_claim(span, prior) for prior in earlier)))


def unread_observation(text: str, *, reason: str = "") -> tuple[Perception, tuple[RequestComponent, ...]]:
    """What is known without a reading: plain conversation of unknown meaning.

    One conversation component covers the whole message. Nothing is inferred
    about its target, intent, tone or requests, and the uncertainty is explicit.
    """
    note = reason or "No reading of this message is available"
    perception = Perception(uncertainty=0.9, evidence=(Evidence(UNREAD_REF, note, 0.0, "runtime"),))
    if not (text or "").strip():
        return perception, ()
    return perception, (RequestComponent(
        "c0", text, kind="conversation", user_value=0.5, effort=0.1, uncertainty=0.6,
        confidence=0.2, source_span=text, evidence_refs=(UNREAD_REF,)),)


def observe(text: str, *, recent_context: list[dict] | None = None) -> tuple[Perception, tuple[RequestComponent, ...]]:
    """REPLY-CHECK COMPATIBILITY ONLY. Not perception; nothing in a turn calls this.

    The evaluation judge (``openrook_tools/roleplay_eval/judge.py``) imports this
    name to nominate possible role requests when it reviews a reply. Reply
    checks are reworked separately, so the name stays importable and returns
    only that nomination, taken from the reply-check cues in ``core/social.py``.
    It carries no perception, tone or price, and leaves with those cues.
    """
    del recent_context
    from core.social import role_request_spans  # reply-check cue, deliberately not a module import

    return Perception(), tuple(
        RequestComponent(f"c{i}", span, kind="role", source_span=span, role_identity=span)
        for i, span in enumerate(role_request_spans(text)))

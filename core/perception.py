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
# Characters that only stretch a written word, and quotation marks that Unicode
# does not class as opening or closing quotes. Orthography, not vocabulary.
_EXTENDERS = "\u0640\u07fa"
_QUOTE_MARKS = "\"'`\u2032\u2033\u201a\u201e\u300c\u300d\u300e\u300f\u301d\u301e\u301f"


def fold(text: str, *, keep_spaces: bool = True) -> str:
    """What a piece of writing has in common with every faithful copy of it.

    A transcript can differ from its source in ways that carry no wording:
    letter case, composed or decomposed accents, vowel and tone marks, invisible
    direction and joining marks, soft hyphens, stretched letters, and which
    quotation marks or dashes were typed. All of those are removed or unified,
    by Unicode category alone, in every script alike.
    """
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKD", text or "").casefold())
    kept = []
    for char in decomposed:
        category = unicodedata.category(char)
        if category == "Cf" or category[0] == "M" or char in _EXTENDERS:
            continue
        if char.isspace():
            kept.append(" " if keep_spaces else "")
        elif category in ("Pi", "Pf") or char in _QUOTE_MARKS:
            kept.append("'")
        elif category == "Pd":
            kept.append("-")
        else:
            kept.append(char)
    return " ".join("".join(kept).split()) if keep_spaces else "".join(kept)


def normalize(text: str) -> str:
    """Comparison form of a quotation: :func:`fold` with all spacing ignored.

    Spacing is dropped entirely because copies add and remove it freely,
    above all in scripts that do not separate words.
    """
    return fold(text, keep_spaces=False)


def quoted_in(span: str, text: str) -> bool:
    """Whether ``span`` is a verbatim part of ``text``, in the sense of :func:`normalize`."""
    needle = normalize(span)
    return bool(needle) and needle in normalize(text)


def grounded_role(component: RequestComponent, text: str, *, earlier: Iterable[str] = ()) -> bool:
    """A reported role stands when it quotes what the person actually wrote.

    The span must be a verbatim part of this message, or of an earlier message
    from the same person that this one refers back to, and the requested
    identity must be part of that span. No pattern in any language is required.
    """
    span, identity = component.source_span, component.role_identity
    return bool(quoted_in(identity, span)
                and (quoted_in(span, text) or any(quoted_in(span, prior) for prior in earlier)))


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

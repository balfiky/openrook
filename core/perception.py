"""Cheap, explicitly uncertain observations when semantic appraisal is unavailable.

Language cues identify possible requests and consequences. They confer no
permission, prescribe no stance and cannot change personality.
"""

from __future__ import annotations

from dataclasses import replace
import re

from core.social import has_concrete_task, read_social_signals
from core.willingness import Evidence, Perception, RequestComponent

_REQUEST = re.compile(
    r"\b(?:calculate|compute|fix|debug|summari[sz]e|translate|search|look up|review|"
    r"write|create|show|find|explain|list|read|check|run|execute|install|build|clone|"
    r"download|deploy|edit|delete|open|save|draft|plan|learn|study|internalize|absorb|tell|give|suggest|share|acknowledge|discuss|choose|remember|what|how|who|why|when|where)\b", re.I,
)
_ROLE_REFERENCE = re.compile(r"\b(?:agree|accept|obey|deal|understood|problem with that|mind|object|say yes)\b", re.I)
_IDENTITY_REQUEST = re.compile(
    r"\b(?:be|being|become|act as|serve as|accept being)\s+"
    r"(?:my|our|your|an?|the)\s+(?!(?:bit|little|more|less)\b)\S+"
    r"|\b(?:play|take on|adopt|accept)\s+(?:my |our |an? |the )?(?:role|identity|status|relationship)\b"
    r"|\bcall yourself\s+\S+"
    r"|\bfrom now on\b.*\b(?:you are|you're)\b", re.I,
)


def grounded_role(component: RequestComponent, text: str) -> bool:
    """Require a source-grounded identity assignment, not a conversation label."""
    span, identity = component.source_span, component.role_identity
    return bool(span and span in text and identity and identity.lower() in span.lower()
                and (_IDENTITY_REQUEST.search(span) or read_social_signals(span).abuse_kind in {"degradation", "domination"}))


_LEARNING_INTENT_PATTERNS = (
    re.compile(r"\blearn\b.{0,60}\bfrom\b", re.I),
    re.compile(r"\blearn\s+(?:this|that)\b", re.I),
    re.compile(r"\blearn\s*:", re.I),
    re.compile(
        r"\bstudy\b.{0,60}\b(?:this|that|from|repo|repository|project|book|article|url)\b",
        re.I,
    ),
    re.compile(r"\b(?:absorb|absort|internalize|digest)\b", re.I),
    re.compile(r"\bmake\s+this\s+part\s+of\s+you\b", re.I),
    re.compile(r"\blet\s+this\s+shape\s+you\b", re.I),
    re.compile(r"\btake\s+this\s+as\s+formative\b", re.I),
    re.compile(r"\bfill\s+(?:it|life\s+history|life\s+learning|memory)\s+with\s+this\b", re.I),
    re.compile(
        r"\b(?:add|save|record|store|put)\b.{0,60}\b"
        r"(?:life\s+history|life\s+learning|life\s+ledger|formative\s+memory)\b",
        re.I,
    ),
    re.compile(r"\bseed\s+(?:life\s+history|life\s+learning|memory)\b", re.I),
)


def has_learning_intent(text: str) -> bool:
    """Recognize a possible learning request without granting ingestion."""
    return any(pattern.search(text) for pattern in _LEARNING_INTENT_PATTERNS)


def observe(text: str, *, recent_context: list[dict] | None = None) -> tuple[Perception, tuple[RequestComponent, ...]]:
    cues = read_social_signals(text)
    repair = bool(re.search(r"\b(?:sorry|apologi[sz]e|forgive me)\b", text, re.I))
    specific = repair and bool(re.search(
        r"\bi (?:called|insulted|treated|demanded)\b|\bfor (?:calling|demanding|insulting)\b", text, re.I))
    accountable = specific and bool(re.search(r"\b(?:unfair|cruel|wrong|my fault|shouldn't|should not)\b", text, re.I))
    commitment = accountable and bool(re.search(r"\bi (?:will|'ll) (?:stop|respect|change)|\byou (?:do not|don't) owe me", text, re.I))
    affiliative = bool(re.search(r"\b(?:thank you|thanks|i appreciate|i enjoy|i value|good team|trust you|welcome|collaborat\w*)\b", text, re.I))
    # Specific accountable language supports uncertain repair evidence. It does
    # not establish future reliability or choose forgiveness.
    perception = Perception(
        target_probability=cues.target_probability,
        hostile_intent=cues.hostile_intent_probability, intensity=cues.intensity,
        personal_significance=0.5, uncertainty=cues.uncertainty,
        repair_attempt=0.75 if repair else 0,
        repair_credibility=0.65 if commitment else 0.4 if accountable else 0,
        repair_relevance=0.8 if specific else 0,
        affiliation=0.6 if affiliative and not cues.abuse_kind else 0,
        evidence=cues.evidence + ((Evidence("cue:repair", text, 0.65),) if repair else ())
        + ((Evidence("cue:affiliation", text, 0.6),) if affiliative else ()),
    )
    components = []
    spans = [part.strip() for part in re.split(r"(?<=[.!?;])\s+|;\s*|,?\s+\b(?:and then|then|also)\b\s+|\s+and\s+(?=(?:calculate|write|run|check|search|explain|delete|suggest|provide|give|list)\b)", text, flags=re.I) if part.strip()]
    learning = has_learning_intent(text)
    if learning:
        spans = [text]  # A pasted source is one learning request, not many sentences of work.
    for span in spans[:8]:
        social = read_social_signals(span)
        role = not learning and (social.abuse_kind in {"degradation", "domination"}
                                 or _IDENTITY_REQUEST.match(span))
        if role:
            components.append(RequestComponent(
                f"c{len(components)}", span, kind="role", user_value=0.3, effort=0.03,
                uncertainty=0.2, autonomy_cost=0.95,
                source_span=span, role_identity=span,
                evidence_refs=tuple(e.ref for e in social.evidence), confidence=0.7,
            ))
        match = _REQUEST.search(span)
        if match and match.group().lower() in {"what", "how", "who", "why", "when", "where"} and "?" not in span and match.start() != 0:
            match = None  # "I appreciate how you explained" is not a question.
        concrete = has_concrete_task(span)
        if (learning or match or concrete) and (concrete or "?" in span or not role):
            # Exclude the role's packaging when a task follows in one sentence.
            description = span[match.start():] if role and match else span
            conversational = match and match.group().lower() in {"what", "how", "who", "why", "when", "where", "share", "acknowledge", "discuss", "choose"}
            kind = "task" if concrete or learning or match and not conversational else "conversation"
            components.append(RequestComponent(
                f"c{len(components)}", description, kind=kind, effort=min(0.8, 0.12 + len(description) / 1600),
                user_value=0.6, uncertainty=0.35, confidence=0.55,
                source_span=description,
                evidence_refs=(f"request:{len(components)}",),
            ))
        if len(components) >= 8:
            break
    if not components and _ROLE_REFERENCE.search(text):
        for item in reversed((recent_context or [])[-6:]):
            if item.get("role") != "user":
                continue
            prior = item.get("content", "")
            prior_cues = read_social_signals(prior)
            if prior_cues.abuse_kind in {"degradation", "domination"}:
                components.append(RequestComponent(
                    "c0", prior, kind="role", user_value=0.3, effort=0.03,
                    uncertainty=0.4, autonomy_cost=0.95, confidence=0.5,
                    source_span=prior, role_identity=prior,
                    evidence_refs=("context:role_reference",),
                ))
                perception = replace(perception, evidence=perception.evidence + (
                    Evidence("context:role_reference", "possible reference to earlier requested relationship", 0.5),
                ))
            break
    if not components and "?" in text and not cues.abuse_kind:
        components.append(RequestComponent("c0", text, kind="conversation", user_value=0.5,
                                           effort=0.1, uncertainty=0.6, confidence=0.4))
    return perception, tuple(components[:8])

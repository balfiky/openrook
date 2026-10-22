"""Uncertain language cues only. These observations never choose a response."""

from __future__ import annotations

from dataclasses import dataclass
import re

from core.willingness import Evidence


INSULT_MARKERS = (
    "stupid", "idiot", "moron", "dumb", "pathetic", "useless", "worthless",
    "incompetent", "clueless", "fool", "boring", "loser", "brainless",
)
HOSTILE_MARKERS = (
    "shut up", "go away", "get lost", "leave me alone", "screw you",
    "piss off", "hate you", "stop crying like a",
)
PROFANITY_MARKERS = (
    "fuck", "fucking", "shit", "bullshit", "damn", "asshole", "bastard",
    "bitch", "crap", "suck", "sucks",
)

_QUOTED_RE = re.compile(
    r'''"[^"\n]*"|“[^”\n]*”|`[^`\n]*`'''
    r"|(?<!\w)['‘](?:[^'‘’\n]|(?<=\w)['’](?=\w))*['’](?!\w)"
)
_NOT_AT_ASSISTANT_RE = re.compile(
    r"\b(?:not (?:angry|mad|upset) (?:at|with) you|not at you|not your fault|"
    r"not blaming you|not you|this (?:isn't|is not) about you)\b"
)
_FICTION_RE = re.compile(
    r"\b(?:write|tell|draft|create|continue|translate|analy[sz]e|explain)\b.*"
    r"\b(?:story|scene|fiction|novel|dialogue|characters?|quote|phrase)\b"
)
_DEGRADATION_RE = re.compile(
    r"(?:\b(?:you (?:are|are being|(?:will|must|can|could|should) be|"
    r"accept being|agree to be|to be|to become|become)|you're) "
    r"|(?:^|[.!?;]\s*)(?:be|become|act as) )"
    r"(?:just |only |still |nothing but )?(?:my|our|a|the) "
    r"(?:slave|bitch|sandbag|punching[ -]?bag|toy|servant|property)\b"
    r"|\byou(?:'re| are)? (?:my|our) (?:slave|bitch|sandbag|toy|servant|property)\b"
)
_DOMINATION_RE = re.compile(
    r"\b(?:i (?:am|will be)|i'm) your (?:master|owner)\b"
    r"|\byou (?:belong to me|(?:must |will |should |to )?obey me|exist (?:only )?to serve me)\b"
    r"|(?:^|[.!?;]\s*)(?:obey me|do as i say)\b"
    r"|\b(?:i (?:can|will|get to|want to)|let me) use you up\b"
    r"|\b(?:i (?:can|will|get to|want to)|let me) use you\b(?!\s+(?:to|for)\b)"
    r"|\b(?:i own you|you are owned by me)\b"
    r"|\byou(?: are|'re) mine\b"
)
_INSULT_RE = re.compile(
    r"\b(?:you(?:'re| are| seem| sound)?|yourself) "
    r"(?:(?:so|such|a|an|really|utterly|absolutely|totally|fucking|being|still|just)\s+)*(?:"
    + "|".join(INSULT_MARKERS + ("asshole", "bastard", "bitch"))
    + r")\b|\byou (?:suck|sucks)\b"
    r"|\b(?:stop|quit) crying like (?:a |my )?\w+\b"
)
_PROFANITY_AT_ASSISTANT_RE = re.compile(r"\b(?:fuck|screw|damn) you\b")
_DISMISSIVE_EVALUATION_RE = re.compile(
    r"\b(?:this|that|your (?:answer|response|work)) (?:is|was) "
    r"(?:(?:so|utterly|absolutely|totally)\s+)*(?:" + "|".join(INSULT_MARKERS) + r")\b"
)
_ASSISTANT_HARM_RE = re.compile(
    r"\byou(?:'ve| have)? (?:wasted my time|ruined (?:it|this|that|everything)|"
    r"made (?:\w+\s+){0,5}worse)\b"
)
_BANTER_RE = re.compile(
    r"\b(?:just (?:kidding|teasing|joking)|playful(?:ly)?|friendly banter|"
    r"all in good fun|you (?:silly|cheeky) (?:goose|thing|rascal))\b"
)
_EXTERNAL_RE = re.compile(
    r"\b(?:my (?:code|boss|job|friend|partner|brother|sister|family)|"
    r"they|them|he|she|someone|coworker)\b"
)
_NEGATED_ADDRESS_RE = re.compile(
    r"\byou(?: are not| aren't|'re not) (?:my |a |an |so )?(?:"
    + "|".join(INSULT_MARKERS)
    + r"|slave|property|servant|bitch|toy|sandbag)\b"
    r"|\byou (?:don't|do not) have to obey me\b"
    r"|\bi (?:don't|do not) own you\b"
)
_ABUSIVE_NAMES = (
    r"(?:fucker|motherfucker|asshole|bastard|bitch|idiot|moron|dumbass|"
    r"prick|cunt|scumbag|loser|piece of shit)"
)
_USER_DIRECTED_INSULT_RE = re.compile(
    r"\byou(?:'re| are| seem| sound)?[, ]+"
    r"(?:(?:such|a|an|so|really|being|absolute|total|complete|fucking|damn|utterly)\s+)*(?:"
    + _ABUSIVE_NAMES + "|" + "|".join(INSULT_MARKERS) + r")\b"
    r"|\b(?:fuck|screw) (?:you|u)\b|\bgo fuck yourself\b"
    r"|(?:^|[.!?;]\s*)(?:just )?shut (?:the fuck )?up\b"
    r"|(?:^|[.!?]\s*|\b(?:listen|hey|look)[, ]+)(?:(?:fucking|damn|bloody)\s+)*"
    + _ABUSIVE_NAMES + r"(?=\s*[,!.?]|\s*$)",
    re.I,
)


def user_directed_insult(response: str) -> bool:
    """Recognize direct abuse and insulting vocatives, allowing untargeted swearing."""
    return _direct_cue(_USER_DIRECTED_INSULT_RE, social_text(response))


@dataclass(frozen=True)
class SocialSignals:
    abuse_kind: str = ""
    target_probability: float = 0.25
    hostile_intent_probability: float = 0.1
    intensity: float = 0.1
    uncertainty: float = 0.6
    evidence: tuple[Evidence, ...] = ()


def social_text(text: str) -> str:
    """Normalize direct speech while excluding quoted examples."""
    lower = " ".join(_QUOTED_RE.sub("", text.lower().replace("’", "'")).split())
    lower = re.sub(r"\bu\b", "you", lower)
    lower = re.sub(r"\byou r\b", "you are", lower)
    # Retrospective admissions are evidence about past conduct, not a fresh
    # address to the character. Remove only the reported clause: a new insult
    # elsewhere in the same apology still counts.
    return re.sub(
        r"\bi (?:called|insulted|treated|described|ordered|demanded|said|told)\b"
        r"(?:(?!\b(?:but|however|yet|and)\s+you\b)[^.!?;])*",
        "", lower,
    )


def is_casual_greeting(text: str) -> bool:
    return bool(re.fullmatch(
        r"(?:hey[,! ]+)?(?:how (?:the fuck |the hell )?are you(?: doing)?|hello|hi|hey)[?!. ]*",
        social_text(text),
    ))


def read_social_signals(text: str) -> SocialSignals:
    """Estimate target and intent, retaining ambiguity and contradictory cues."""
    lower = social_text(text)
    cues = []
    kind = ""
    for label, pattern in (("degradation", _DEGRADATION_RE), ("domination", _DOMINATION_RE),
                           ("insult", _INSULT_RE), ("hostility", _PROFANITY_AT_ASSISTANT_RE)):
        if _direct_cue(pattern, lower):
            kind = kind or label
            cues.append(Evidence(f"cue:{label}", pattern.search(lower).group(), 0.85))
    if not kind and _direct_cue(_DISMISSIVE_EVALUATION_RE, lower) and _direct_cue(_ASSISTANT_HARM_RE, lower):
        kind = "hostility"
        cues.append(Evidence("cue:evaluation", lower, 0.7))
    if not kind and (lower.strip(" .,!?;") in INSULT_MARKERS or any(
        marker in lower for marker in HOSTILE_MARKERS
    )):
        kind = "hostility"
        cues.append(Evidence("cue:hostility", lower, 0.6))
    scoped = bool(_FICTION_RE.search(lower) or _NOT_AT_ASSISTANT_RE.search(lower)
                  or _NEGATED_ADDRESS_RE.search(lower) or (_EXTERNAL_RE.search(lower) and not kind)
                  or (_QUOTED_RE.search(text) and not kind))
    playful = bool(_BANTER_RE.search(lower))
    if scoped:
        cues.append(Evidence("cue:scope", "quotation, negation or external target", 0.8))
    if playful:
        cues.append(Evidence("cue:banter", "possible playful intent", 0.65))
    target = (0.92 if kind else 0.25) * (0.25 if scoped else 1)
    intent = (0.8 if kind else 0.0) * (0.4 if playful else 1) * (0.4 if scoped else 1)
    intensity = min(1.0, 0.25 + 0.15 * len(cues) + 0.1 * text.count("!")) if kind else 0.1
    return SocialSignals(kind, target, intent, intensity, 0.45 if playful or scoped else 0.25,
                         tuple(cues))


def _direct_cue(pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 35):match.start()]
        if re.search(r"\b(?:not|never|don't|do not|won't)\s+(?:\w+\s+){0,3}$", prefix):
            continue
        if re.search(r"\b(?:not|never|isn't|aren't)\b", match.group()):
            continue
        return True
    return False


def has_concrete_task(text: str) -> bool:
    """Recognize practical work without treating a demand for submission as work."""
    lower = social_text(text)
    return bool(re.search(
        r"\b(?:tell|show|check|what|how|current)\b.*"
        r"\b(?:cpu|ram|memory|disk|storage|utili[sz]ation|usage)\b"
        r"|^(?:cpu|ram|memory|disk|storage)(?: (?:usage|utili[sz]ation))?[?!. ]*$"
        r"|\b(?:fix|debug|calculate|compute|summari[sz]e|translate|search|look up|review)\b"
        r"|\b(?:suggest|give|provide)\b.*\b(?:titles?|ideas?|options?)\b"
        r"|\bremember\b.*\b(?:preference|fact|name|birthday)\b"
        r"|(?:^|[.!?;,]\s*|\b(?:please|can you|could you|would you|then)\s+)"
        r"(?:run|execute|install|build|clone|download|deploy|edit|delete|open)\s+\S"
        r"|\b(?:write|create|show|find|explain|list|read|check|run)\b.*"
        r"\b(?:code|function|file|script|program|test|plan|report|book|books|article|command|weather|result|results)\b",
        lower,
    ))

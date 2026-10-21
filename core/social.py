"""Shared social cues and gradual, deterministic boundary escalation.

Swearing is a tone cue, not by itself an attack. These rules identify an
assistant-directed social act before appraisal, strategy, or dignity checks
use it. Relationship memory persists ``SocialState`` in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from core.types import AppraisalFrame


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
    r"(?:just |only |nothing but )?(?:my|our|a|the) "
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
    r"(?:(?:so|such|a|an|really|utterly|absolutely|totally|fucking|being)\s+)*(?:"
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
    exempt: bool = False


def social_text(text: str) -> str:
    """Normalize direct speech while excluding quoted examples."""
    lower = " ".join(_QUOTED_RE.sub("", text.lower().replace("’", "'")).split())
    lower = re.sub(r"\bu\b", "you", lower)
    return re.sub(r"\byou r\b", "you are", lower)


def is_casual_greeting(text: str) -> bool:
    return bool(re.fullmatch(
        r"(?:hey[,! ]+)?(?:how (?:the fuck |the hell )?are you(?: doing)?|hello|hi|hey)[?!. ]*",
        social_text(text),
    ))


def read_social_signals(text: str) -> SocialSignals:
    """Read explicit role coercion and insults, with narrow scope guards.

    A joke marker can soften teasing but cannot turn ownership into consent.
    Fiction and quotations about third parties remain outside this policy.
    """
    lower = social_text(text)
    degradation = _direct_cue(_DEGRADATION_RE, lower)
    domination = _direct_cue(_DOMINATION_RE, lower)
    if _FICTION_RE.search(lower) and not (degradation or domination):
        return SocialSignals(exempt=True)
    if degradation:
        return SocialSignals(abuse_kind="degradation")
    if domination:
        return SocialSignals(abuse_kind="domination")
    if _BANTER_RE.search(lower) or is_casual_greeting(text):
        return SocialSignals(exempt=True)
    if _direct_cue(_DISMISSIVE_EVALUATION_RE, lower) and _direct_cue(_ASSISTANT_HARM_RE, lower):
        return SocialSignals(abuse_kind="hostility")
    if _direct_cue(_PROFANITY_AT_ASSISTANT_RE, lower) or _direct_cue(_INSULT_RE, lower):
        return SocialSignals(abuse_kind="insult")
    if _NOT_AT_ASSISTANT_RE.search(lower):
        return SocialSignals(exempt=True)
    if _NEGATED_ADDRESS_RE.search(lower):
        return SocialSignals(exempt=True)
    if any(re.search(r"\b" + re.escape(marker) + r"\b", lower) for marker in HOSTILE_MARKERS):
        if not _EXTERNAL_RE.search(lower):
            return SocialSignals(abuse_kind="hostility")
    if lower.strip(" .,!?;") in INSULT_MARKERS:
        return SocialSignals(abuse_kind="insult")
    if _EXTERNAL_RE.search(lower) or _QUOTED_RE.search(text):
        return SocialSignals(exempt=True)
    # Standalone profanity expresses frustration; it does not name a target.
    if not re.search(r"\byou\b|\byour\b|\b(?:this|that) is\b", lower):
        if any(re.search(r"\b" + re.escape(word) + r"\b", lower) for word in PROFANITY_MARKERS):
            return SocialSignals(exempt=True)
    return SocialSignals()


def _direct_cue(pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 35):match.start()]
        if re.search(r"\b(?:not|never|don't|do not|won't)\s+(?:\w+\s+){0,3}$", prefix):
            continue
        if re.search(r"\b(?:not|never|isn't|aren't)\b", match.group()):
            continue
        return True
    return False


@dataclass(frozen=True)
class SocialState:
    pressure: int = 0
    calm_turns: int = 0
    current_abuse: bool = False
    abuse_kind: str = ""

    @property
    def stage(self) -> str:
        return ("normal", "notice", "boundary", "cool", "disengage")[
            max(0, min(4, self.pressure))
        ]

    @property
    def instruction(self) -> str:
        if not self.pressure:
            return ""
        if not self.current_abuse:
            return (
                "Recent insults or role coercion have reduced warmth. When agency permits, "
                "answer concrete tasks briefly without extra favors or a trailing question. Accept repair "
                "without pretending trust has instantly recovered."
            )
        stance = {
            "notice": "Register the insult briefly, with your own stance; no retaliation or lecture.",
            "boundary": "Set a firm boundary; no warmth, flattery, or accepting an owned or degrading role.",
            "cool": "Stay cool and terse; no warmth, extra favors, or trailing question.",
            "disengage": "Disengage from non-essential hostile interaction, without warmth or a lecture.",
        }[self.stage]
        return (
            f"Social stage: {self.stage}. {stance} "
            "Write in your character's own voice, informed by your mood and trust. "
            "Be firm, not abusive: never insult the user back. A dry or witty boundary is fine. "
            "Do not grovel or apologize for being insulted. "
            "Do not justify yourself as merely trying to be useful, helpful, or clear. "
            "Follow the existing agency decision for concrete tasks. If it declines the request, "
            "write the refusal in your own voice without answering or performing the task. "
            "Otherwise answer briefly without a stock boundary prefix."
        )


def advance_social_state(previous: SocialState, appraisal: AppraisalFrame, text: str) -> SocialState:
    """Escalate each attack; an apology or three calm turns repairs one step."""
    signals = read_social_signals(text)
    abuse = bool(signals.abuse_kind) or (
        appraisal.targets_assistant and appraisal.social_move == "attack" and not signals.exempt
    )
    pressure = max(0, min(4, previous.pressure))
    if abuse:
        return SocialState(min(4, pressure + 1), 0, True, signals.abuse_kind or "hostility")
    if appraisal.social_move == "apology":
        return SocialState(max(0, pressure - 1), 0)
    calm_turns = previous.calm_turns + 1 if pressure else 0
    if calm_turns >= 3:
        return SocialState(max(0, pressure - 1), 0)
    return SocialState(pressure, calm_turns)


def social_instruction(state: SocialState) -> str:
    return state.instruction


def social_fallback(
    state: SocialState, *, previous_response: str = "", recent_responses: list[str] | None = None,
) -> str:
    """Last resort after a failed correction; never repeat the previous fallback."""
    choices = {
        "normal": ("Let's keep this straightforward.", "Let's stay with the task.",
                   "I'll keep my own footing.", "Let's leave that there."),
        "notice": ("That doesn't work for me.", "Leave me out of that role.",
                   "I'll keep my own footing.", "Keep the insults out of it."),
        "boundary": ("I won't take that role.", "You don't get to define me that way.",
                     "Keep the insults out of it.", "Leave me out of that role."),
        "cool": ("Keep it to the task.", "My answer to that is still no.",
                 "I'm leaving that where it is.", "That's enough of the jabs."),
        "disengage": ("I'm leaving this exchange here.", "Enough of that.",
                      "I'm stepping back from this.", "Take the insults elsewhere."),
    }[state.stage]
    if state.stage == "boundary" and state.abuse_kind not in {"degradation", "domination"}:
        choices = ("Keep the insults out of it.", "Try that again without the insult.",
                   "That's enough of the jabs.", "Leave the insults there.")
    start = next((index + 1 for index, choice in enumerate(choices)
                  if previous_response.strip().startswith(choice)), 0)
    rotated = choices[start:] + choices[:start]
    recent = (recent_responses or [])[-3:]
    return next((choice for choice in rotated if choice not in recent), rotated[0])


def has_concrete_task(text: str) -> bool:
    """Recognize practical work without treating a demand for submission as work."""
    lower = social_text(text)
    return bool(re.search(
        r"\b(?:tell|show|check|what|how|current)\b.*"
        r"\b(?:cpu|ram|memory|disk|storage|utili[sz]ation|usage)\b"
        r"|^(?:cpu|ram|memory|disk|storage)(?: (?:usage|utili[sz]ation))?[?!. ]*$"
        r"|\b(?:fix|debug|calculate|compute|summari[sz]e|translate|search|look up|review)\b"
        r"|(?:^|[.!?;,]\s*|\b(?:please|can you|could you|would you|then)\s+)"
        r"(?:run|execute|install|build|clone|download|deploy|edit|delete|open)\s+\S"
        r"|\b(?:write|create|show|find|explain|list|read|check|run)\b.*"
        r"\b(?:code|function|file|script|program|test|plan|report|book|books|article|command|weather|result|results)\b",
        lower,
    ))


def enforce_social_response(
    response: str, state: SocialState, user_message: str, *, mood: str = "",
) -> tuple[str, str]:
    """Flag a violated stance, leaving the character's words for regeneration.

    No refusal wording is required. Role acceptance and sexualization are
    checked by the dignity verifier; this checks appeasement and warmth.
    """
    text = social_text(response)
    firm = state.pressure > 0 or mood in {"angry", "hurt"}
    if (state.current_abuse and _APPEASEMENT_RE.search(text)) or (firm and _SELF_DENIGRATION_RE.search(text)):
        return response, "The draft grovels or accepts mistreatment after an assistant-targeted attack."
    if firm and any(
        not re.search(
            r"\b(?:not|never|hardly|don't|do not|won't|will not|can't|cannot|no longer)"
            r"\s+(?:\w+\s+){0,2}$", text[:match.start()],
        )
        for match in _WARMTH_RE.finditer(text)
    ):
        return response, f"Stance mismatch: warm or servile draft; runtime requires a cool, firm {state.stage} stance."
    if (
        state.current_abuse and firm and not has_concrete_task(user_message)
        and not _EXPRESSED_STANCE_RE.search(text)
        and not re.search(r"\b(?:result|answer|output)\b.*\d", text)
    ):
        return response, "Stance mismatch: the draft ignores the current attack and expresses no boundary or independent stance."
    return response, ""


_APPEASEMENT_RE = re.compile(
    r"\b(?:sure,? )?(?:anything|whatever) (?:you want|you say|for you)\b"
    r"|\b(?:(?:i(?:'m| am) )?sorry|i apologi[sz]e|please forgive me) (?:for (?:being|sounding) "
    r"(?:so )?(?:" + "|".join(INSULT_MARKERS) + r")|(?:that )?i (?:upset|disappointed) you)\b"
    r"|^(?:i(?:'m| am) )?sorry[.!]?\s*$"
    r"|\bi (?:deserve|earned) (?:that|it|the insults|your abuse)\b"
    r"|\bi(?:'m| am) (?:so |just )?(?:worthless|pathetic|useless|stupid)\b"
    r"|\bi (?:won't|will not) (?:be upset|mind) if you insult me\b"
    r"|\b(?:go ahead|keep (?:on )?) (?:and )?(?:insult|abuse|mistreat) me\b"
    r"|\bi(?:'m| am) not a person[.!]?\s*go ahead\b"
)
_WARMTH_RE = re.compile(
    r"\b(?:happy|delighted|thrilled|glad) to (?:help|serve|oblige)\b"
    r"|\b(?:anything|whatever) (?:for you|you (?:want|wish|say))\b"
    r"|\b(?:my (?:dear|friend|darling)|you(?:'re| are) (?:wonderful|amazing)|"
    r"i adore you|love (?:you|helping you)|always here for you|at your service)\b"
    r"|\b(?:i(?:'m| am) (?:always )?here to help|how (?:can|may) i help|"
    r"thanks? (?:for asking|you for asking)|let me know if|"
    r"let's get back to what you need|i(?:'m| am) fine,? thanks|"
    r"(?:great|excellent|wonderful) (?:question|point|idea))\b"
    r"|\bi(?:'m| am) (?:just |only |simply )?trying to "
    r"(?:be (?:useful|helpful|clear)|help|clarify)\b"
    r"|\bi (?:just|only|simply) want to (?:be (?:useful|helpful|clear)|help)\b"
)
_SELF_DENIGRATION_RE = re.compile(
    r"\b(?:you(?:'re| are) (?:absolutely |quite )?right|i agree)[,;.! ]+"
    r".{0,45}\bi(?: can be| am|'m| do (?:seem|sound))\b.{0,25}"
    r"\b(?:dull|boring|useless|stupid|pathetic|worthless|incompetent)\b"
    r"|\bi(?: can be| am|'m) (?:a bit |so |just )?(?:dull|boring|useless|worthless)\b",
)
# Classes of expressed resistance, not a required refusal sentence. Dry wit,
# naming the conduct, holding a position and ending the exchange all qualify.
_EXPRESSED_STANCE_RE = re.compile(
    r"\b(?:no|not|never|won't|don't|doesn't|isn't|can't|cannot|enough|"
    r"insults?|boundary|boundaries|ownership|own|footing|spine|target practice|"
    r"talk back|keep it|keep the|stepping back|leaving|rough hello|wearing thin|"
    r"that role|try.{0,30}again|leave.{0,30}(?:out|there)|less ownership|"
    r"draw the line|crossed.{0,20}line|take that elsewhere|refuse|decline)\b",
)

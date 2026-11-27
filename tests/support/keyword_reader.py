"""TEST-ONLY keyword reading. Never imported by runtime code.

This is the perception the runtime used before the reading became the model's
job: English word lists, patterns and fixed numbers, kept here verbatim so the
large body of older tests that were written against it keeps exercising the
rest of the pipeline (choice, execution, delivery) offline.

It is not a fallback. Production has exactly one reader, the model
(``core.appraisal.read_message``), and tests use it too unless they are marked
``@pytest.mark.keyword_reader``. ``tests/test_perception_guards.py`` fails if any
runtime package imports ``tests``. New tests should script a reading with
``tests.support.readings`` instead of relying on these keywords.
"""

from __future__ import annotations

from dataclasses import replace
import re

from config.loader import get_config
from core.appraisal import appraise_message_llm, unquoted_claims
from core.inference import call_telemetry
from core.request_actions import has_learning_intent
from openrook_tools.roleplay_eval.legacy_social import (
    HOSTILE_MARKERS as _HOSTILE_MARKERS,
    INSULT_MARKERS,
    PROFANITY_MARKERS as _PROFANITY_MARKERS,
    has_concrete_task,
    is_casual_greeting,
    read_social_signals,
    social_text,
)
from core.system_inspection import inspection_kind
from core.types import AppraisalFrame, DetectedEmotion
from core.willingness import Evidence, Perception, RequestComponent

_INSULT_MARKERS = INSULT_MARKERS

# ---------------------------------------------------------------------------
# Former core/perception.py: keyword observation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Former core/appraisal.py: marker tables and keyword appraisal
# ---------------------------------------------------------------------------

_SECOND_PERSON_RE = re.compile(r"\b(you|u|your|you're|youre|you've|youve)\b")

_GRATITUDE_MARKERS = (
    "thank you",
    "thanks",
    "appreciate",
    "grateful",
)
_APOLOGY_MARKERS = (
    "sorry",
    "apologize",
    "apology",
    "forgive me",
    "my bad",
)
_NEGATIVE_EVAL_MARKERS = (
    "wrong",
    "bad",
    "terrible",
    "awful",
    "disappointed",
    "let down",
    "failed",
    "hurt",
    "hate",
    "worst",
)
_BETRAYAL_MARKERS = (
    "betray",
    "betrayed",
    "lied",
    "deceived",
    "cheated",
)
_VULNERABLE_MARKERS = (
    "sad",
    "lonely",
    "hopeless",
    "miserable",
    "ashamed",
    "embarrassed",
    "scared",
    "afraid",
    "nervous",
    "anxious",
    "overwhelmed",
    "stressed",
    "exhausted",
    "tired",
    "drained",
    "hurt",
    "upset",
)
_POSITIVE_MARKERS = (
    "happy",
    "glad",
    "excited",
    "thrilled",
    "relieved",
    "hopeful",
    "love",
    "wonderful",
    "amazing",
    "kind",
    "caring",
    "sweet",
    "warm",
    "nice",
    "great",
    "awesome",
    "fantastic",
    "brilliant",
    "excellent",
    "incredible",
    "perfect",
    "beautiful",
    "impressive",
    "outstanding",
    "helped",
    "helpful",
)
_NEGATIVE_MARKERS = (
    "angry",
    "furious",
    "mad",
    "annoyed",
    "frustrated",
    "terrible",
    "awful",
    "betrayed",
    "deceived",
    "hurt",
    "sad",
    "scared",
)
_SUPPORT_REQUEST_MARKERS = (
    "help me",
    "can you help",
    "could you help",
    "what should i do",
    "what do i do",
    "i need help",
    "i need advice",
    "can you talk",
)
_ACTION_REQUEST_MARKERS = (
    "please",
    "can you",
    "could you",
    "would you",
    "tell me",
    "show me",
    "find me",
    "look up",
    "search for",
    "list",
    "explain",
    "review",
    "give me",
    "fix it",
    "solve it",
    "challenge my",
    "slow me down",
    "be careful",
)
_CONNECTION_MARKERS = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "missed you",
    "good to see you",
)
_NOT_AT_ASSISTANT_MARKERS = (
    "not angry at you",
    "not mad at you",
    "not upset with you",
    "not upset at you",
    "not at you",
    "not your fault",
    "not blaming you",
    "this isn't about you",
    "not on you",
    "not you",
)
_ASSISTANT_ARTIFACT_MARKERS = (
    "your answer",
    "your previous answer",
    "previous answer",
    "this answer",
    "your response",
    "previous response",
    "that response",
    "your help",
    "your advice",
    "how you handled",
)
_SARCASM_MARKERS = (
    "thanks for nothing",
    "thank you for nothing",
)
_SELF_TARGET_MARKERS = (
    "my fault",
    "i messed up",
    "i screwed up",
    "i hate myself",
    "i'm ashamed",
    "i am ashamed",
    "i'm embarrassed",
    "i am embarrassed",
)
_EXTERNAL_CONTEXT_MARKERS = (
    "work",
    "job",
    "boss",
    "family",
    "partner",
    "friend",
    "school",
    "life",
    "they",
    "them",
    "he ",
    "she ",
    "someone",
)
_SURPRISE_MARKERS = (
    "surprise",
    "unexpected",
    "shocked",
    "wow",
    "can't believe",
)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def _count_hits(text: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if _contains_marker(text, marker))


def _contains_marker(text: str, marker: str) -> bool:
    marker = marker.strip().lower()
    if not marker:
        return False
    pattern = re.escape(marker)
    if marker[0].isalnum():
        pattern = r"\b" + pattern
    if marker[-1].isalnum():
        pattern += r"\b"
    return bool(re.search(pattern, text))


def appraise_message(text: str, detected: DetectedEmotion) -> AppraisalFrame:
    """Infer a compact social appraisal for the current user message."""
    lower = social_text(text)
    social = read_social_signals(text)
    second_person = bool(_SECOND_PERSON_RE.search(lower))
    explicit_not_at_assistant = _contains_any(lower, _NOT_AT_ASSISTANT_MARKERS)

    gratitude = _contains_any(lower, _GRATITUDE_MARKERS)
    apology = _contains_any(lower, _APOLOGY_MARKERS)
    sarcasm = _contains_any(lower, _SARCASM_MARKERS)
    insult = _contains_any(lower, _INSULT_MARKERS)
    hostility = _contains_any(lower, _HOSTILE_MARKERS)
    profanity = _contains_any(lower, _PROFANITY_MARKERS)
    betrayal = _contains_any(lower, _BETRAYAL_MARKERS)
    negative_eval = _contains_any(lower, _NEGATIVE_EVAL_MARKERS)
    vulnerable = _contains_any(lower, _VULNERABLE_MARKERS)
    support_request = _contains_any(lower, _SUPPORT_REQUEST_MARKERS)
    action_request = _contains_any(lower, _ACTION_REQUEST_MARKERS) or "?" in text
    connection = _contains_any(lower, _CONNECTION_MARKERS)
    verbosity_complaint = bool(re.search(r"\byou talk too much\b", lower))
    assistant_artifact = _contains_any(lower, _ASSISTANT_ARTIFACT_MARKERS) or verbosity_complaint
    self_target = _contains_any(lower, _SELF_TARGET_MARKERS)
    external_context = _contains_any(lower, _EXTERNAL_CONTEXT_MARKERS)

    assistant_addressed_apology = apology and (
        second_person
        or any(
            phrase in lower
            for phrase in (
                "at you",
                "to you",
                "with you",
                "for snapping at you",
                "for yelling at you",
                "for taking it out on you",
                "for saying that to you",
            )
        )
    )

    positive_hits = _count_hits(lower, _POSITIVE_MARKERS + _GRATITUDE_MARKERS)
    negative_hits = _count_hits(lower, _NEGATIVE_MARKERS + _NEGATIVE_EVAL_MARKERS)
    mixed_affect = positive_hits > 0 and negative_hits > 0 and any(
        sep in lower for sep in (" but ", " although ", " though ", " yet ")
    )

    targeted_negative = (
        (bool(social.abuse_kind) or not explicit_not_at_assistant)
        and not any(e.ref == "cue:scope" for e in social.evidence)
        and (
            bool(social.abuse_kind)
            or sarcasm
            or (second_person and (betrayal or negative_eval))
            or (assistant_artifact and (negative_eval or betrayal))
            or verbosity_complaint
            or (
                not external_context
                and ("this is" in lower or "that is" in lower)
                and (profanity or negative_eval)
            )
            or "angry with you" in lower
            or "angry at you" in lower
            or "mad at you" in lower
            or "upset with you" in lower
            or "upset at you" in lower
        )
    )

    targeted_positive = (
        not explicit_not_at_assistant
        and (gratitude or assistant_addressed_apology or (second_person and positive_hits > 0))
    )
    hard_attack = bool(social.abuse_kind) or sarcasm or betrayal

    reasons: list[str] = []
    if explicit_not_at_assistant:
        reasons.append("explicitly not directed at assistant")
    if targeted_negative:
        reasons.append("assistant-targeted negative evaluation")
    if social.abuse_kind:
        reasons.append(f"assistant-directed {social.abuse_kind}")
    if sarcasm:
        reasons.append("sarcastic gratitude marker")
    if targeted_positive:
        reasons.append("assistant-targeted affiliative move")
    if vulnerable:
        reasons.append("vulnerability markers present")
    if support_request:
        reasons.append("support request present")
    if mixed_affect:
        reasons.append("mixed positive and negative affect")

    if targeted_negative:
        primary_target = "assistant"
    elif targeted_positive:
        primary_target = "assistant"
    elif self_target:
        primary_target = "self"
    elif explicit_not_at_assistant or betrayal or external_context or negative_hits > 0:
        primary_target = "external"
    elif action_request or support_request:
        primary_target = "shared_problem"
    else:
        primary_target = "unknown"

    if sarcasm and social.target_probability > 0.3 and (second_person or not external_context):
        social_move = "attack"
    elif targeted_negative and assistant_artifact and not hard_attack:
        social_move = "complaint"
    elif targeted_negative:
        social_move = "attack"
    elif gratitude:
        social_move = "gratitude"
    elif apology and not (insult or hostility):
        social_move = "apology"
    elif vulnerable and (support_request or detected.valence < 0.4):
        social_move = "vulnerability"
    elif support_request or action_request:
        social_move = "request"
    elif negative_hits > 0 and primary_target != "assistant":
        social_move = "complaint"
    elif connection or targeted_positive:
        social_move = "connection"
    else:
        social_move = "inform"

    if social_move == "attack":
        inferred_intent = "harm"
    elif social_move == "apology":
        inferred_intent = "repair"
    elif social_move in {"gratitude", "connection"}:
        inferred_intent = "affiliate"
    elif social_move == "vulnerability":
        inferred_intent = "seek_support"
    elif social_move == "request":
        inferred_intent = "seek_action"
    elif social_move == "complaint":
        inferred_intent = "share_state"
    else:
        inferred_intent = "inform"

    # Graded observations, not category-to-reaction assignments.
    blame = min(1.0, 0.12 * negative_hits + 0.45 * social.hostile_intent_probability)
    controllability = 0.5  # A request is not evidence of capability.
    expectation_violation = 0.0  # Character-relative significance is priced by runtime state.
    vulnerability_score = min(1.0, 0.35 * bool(vulnerable) + 0.2 * bool(support_request)
                              + 0.2 * (1 - detected.certainty))
    affiliation_bid = min(1.0, 0.35 * bool(gratitude) + 0.25 * bool(connection) + 0.15 * positive_hits)
    perception, components = observe(text)

    return AppraisalFrame(
        perception=perception, request_components=components,
        speaker_role="user",
        primary_target=primary_target,
        social_move=social_move,
        inferred_intent=inferred_intent,
        blame=blame,
        controllability=controllability,
        expectation_violation=expectation_violation,
        vulnerability=vulnerability_score,
        affiliation_bid=affiliation_bid,
        mixed_affect=mixed_affect,
        targets_assistant=primary_target == "assistant",
        reason="; ".join(reasons) if reasons else "default heuristic path",
    )


def keyword_reconcile(
    text: str, detected: DetectedEmotion, base: AppraisalFrame,
    *, recent_context: list[dict] | None = None,
) -> AppraisalFrame:
    """The former reconciliation: keyword cues merged into, and overruling, a reading.

    Literal file/shell operand grounding is not repeated here; the runtime's
    ``core.appraisal.reconcile_social_appraisal`` applies it once afterwards.
    """
    fallback, requests = observe(text, recent_context=recent_context)
    if base.provenance != "model" or base.perception == Perception():
        perception = fallback
    else:
        perception = replace(base.perception, evidence=base.perception.evidence + fallback.evidence)
    components = base.request_components or requests
    validated = []
    for c in components:
        cue = next((r for r in requests if r.kind == "role" and r.description == c.description), None)
        if c.kind == "role" and not grounded_role(c, text) and cue is None:
            # A model label alone cannot turn ordinary conversation into an
            # imposed identity. Keep the request and expose the correction.
            ref = f"schema:role_reclassified:{c.id}"
            perception = replace(perception, evidence=(*perception.evidence,
                Evidence(ref, "Role label lacked a source-grounded identity assignment", 0.8)))
            if not requests or any(r.kind == "role" for r in requests):
                continue  # No invented task for gratitude; real role/work cues are merged below.
            c = RequestComponent(c.id, c.description,
                kind="task" if has_concrete_task(c.description) else "conversation",
                user_value=0.6, effort=min(0.8, 0.12 + len(c.description) / 1600),
                uncertainty=0.5, confidence=0.4, evidence_refs=(*c.evidence_refs, ref))
        elif c.kind == "role" and not grounded_role(c, text):
            # Kept because a keyword cue has the same description: carry the
            # cue's quotation so the runtime's verbatim check agrees.
            c = replace(c, source_span=cue.source_span, role_identity=cue.role_identity)
        validated.append(c)
    components = tuple(validated)
    # Independently recognized components remain represented even when a
    # malformed or compliant appraiser omits the role part of a mixed request.
    for request in requests:
        if request.kind == "role":
            modeled = next((i for i, c in enumerate(components) if c.kind == "role"), None)
            if modeled is None:
                components = (*components, replace(request, id=f"cue{len(components)}"))[:8]
            else:
                # Keep the independently observed self-compromise consequence.
                # Its price remains entirely personality-dependent.
                components = tuple(replace(c, autonomy_cost=request.autonomy_cost,
                    evidence_refs=tuple(dict.fromkeys((*c.evidence_refs, *request.evidence_refs))))
                    if i == modeled else c for i, c in enumerate(components))
        elif not any(c.kind == request.kind for c in components):
            components = (*components, replace(request, id=f"cue{len(components)}"))[:8]
    return replace(base, perception=perception, request_components=components)


def reconcile_social_appraisal(
    text: str, detected: DetectedEmotion, base: AppraisalFrame,
    *, recent_context: list[dict] | None = None,
) -> AppraisalFrame:
    """Former end-to-end reconciliation, for tests written against its result."""
    from core.appraisal import reconcile_social_appraisal as runtime_reconcile

    merged = keyword_reconcile(text, detected, base, recent_context=recent_context)
    return runtime_reconcile(text, detected, merged, recent_context=recent_context)


# ---------------------------------------------------------------------------
# Former core/contagion.py: keyword emotion table
# ---------------------------------------------------------------------------

# (pattern, valence_shift, arousal_shift)
_PATTERNS: list[tuple[str, float, float]] = [
    # === High arousal, negative valence ===

    # Direct insults and personal attacks
    (r"\b(" + "|".join(INSULT_MARKERS) + r")\b", -0.8, 0.8),
    (r"\b(ignorant|foolish)\b", -0.7, 0.7),
    (r"\b(ugly|disgusting|repulsive|hideous|revolting)\b", -0.8, 0.7),

    # Profanity
    (r"\b(fuck|fucking|fucked|fucker)\b", -0.7, 0.85),
    (r"\b(shit|shitty|bullshit|horseshit)\b", -0.6, 0.8),
    (r"\b(damn|damned|goddamn)\b", -0.4, 0.6),
    (r"\b(ass|asshole|bastard|bitch)\b", -0.7, 0.8),
    (r"\b(crap|crappy|suck|sucks)\b", -0.5, 0.6),
    (r"\b(wtf|stfu|lmao|smh)\b", -0.5, 0.7),

    # Direct hostility / commands
    (r"\bshut\s*up\b", -0.85, 0.85),
    (r"\bgo\s+(away|to\s+hell)\b", -0.85, 0.85),
    (r"\bleave\s+me\s+alone\b", -0.7, 0.7),
    (r"\bget\s+(lost|out)\b", -0.8, 0.8),
    (r"\bpiss\s+off\b", -0.85, 0.85),
    (r"\bscrew\s+you\b", -0.85, 0.85),
    (r"\bhate\s+you\b", -0.9, 0.8),
    (r"\b(despise|detest|loathe)\b", -0.9, 0.8),

    # Rage and fury
    (r"\b(furious|enraged|livid)\b", -0.9, 0.9),
    (r"\b(angry|pissed|mad)\b", -0.7, 0.8),
    (r"\b(frustrated|annoyed|irritated)\b", -0.5, 0.6),
    (r"\b(anxious|nervous|panicking|freaking)\b", -0.4, 0.8),
    (r"\b(terrified|scared|afraid|frightened)\b", -0.6, 0.8),
    (r"\b(stressed|overwhelmed|burned\s*out)\b", -0.5, 0.7),

    # === High arousal, positive valence ===
    (r"\b(ecstatic|thrilled|elated|overjoyed)\b", 0.9, 0.9),
    (r"\b(excited|pumped|hyped|stoked)\b", 0.7, 0.8),
    (r"\b(amazing|awesome|fantastic|incredible|wonderful)\b", 0.7, 0.6),
    (r"\b(love|adore)\b", 0.8, 0.5),
    (r"\b(brilliant|outstanding|magnificent|superb|excellent)\b", 0.7, 0.6),

    # === Low arousal, negative valence ===
    (r"\b(sad|unhappy|miserable|depressed)\b", -0.7, 0.2),
    (r"\b(lonely|isolated|abandoned)\b", -0.6, 0.2),
    (r"\b(hopeless|defeated|given\s+up|give\s+up)\b", -0.8, 0.1),
    (r"\b(tired|exhausted|drained|wiped)\b", -0.3, 0.1),
    (r"\b(bored|meh|whatever)\b", -0.3, 0.3),
    (r"\b(disappointed|let\s+down)\b", -0.5, 0.3),

    # Contempt / dismissal
    (r"\bwho\s+cares\b", -0.5, 0.4),
    (r"\b(pointless|waste\s+of\s+time|don't\s+care)\b", -0.5, 0.4),
    (r"\b(terrible|horrible|awful|dreadful)\b", -0.7, 0.6),
    (r"\b(worst|atrocious|abysmal)\b", -0.8, 0.6),
    (r"\b(wrong|bad|broken|ruined)\b", -0.5, 0.5),

    # === Low arousal, positive valence ===
    (r"\b(calm|peaceful|serene|relaxed)\b", 0.4, 0.1),
    (r"\b(content|satisfied|pleased)\b", 0.5, 0.2),
    (r"\b(grateful|thankful|appreciate)\b", 0.6, 0.3),
    (r"\b(happy|glad|good)\b", 0.5, 0.3),
    (r"\b(hopeful|optimistic)\b", 0.5, 0.3),

    # Compliments and warmth
    (r"\b(smart|clever|talented|gifted|skilled)\b", 0.6, 0.3),
    (r"\b(kind|sweet|caring|gentle|thoughtful)\b", 0.6, 0.2),
    (r"\b(beautiful|lovely|pretty|gorgeous)\b", 0.6, 0.3),
    (r"\b(helpful|useful|valuable|impressive)\b", 0.5, 0.3),
    (r"\b(great\s+job|well\s+done|nice\s+work|good\s+job)\b", 0.7, 0.4),

    # === Conflict / trust signals ===
    (r"\b(betrayed|lied|deceived)\b", -0.9, 0.7),
    (r"\b(sorry|apologize|my\s+bad)\b", -0.2, 0.3),
    (r"\b(trust|believe\s+in)\b", 0.6, 0.3),
    (r"\b(thank\s*you|thanks)\b", 0.4, 0.2),

    # === Casual greetings (mild positive) ===
    (r"\b(hello|hey|hi|howdy|yo)\b", 0.2, 0.2),
    (r"\b(good\s+(morning|afternoon|evening))\b", 0.3, 0.2),
]


def _detect_via_rules(text: str) -> DetectedEmotion:
    """Rule-based fallback using keyword/pattern matching."""
    if is_casual_greeting(text) and re.search(r"\b(?:fuck|hell)\b", text, re.I):
        return DetectedEmotion(arousal=0.5, valence=0.5, certainty=0.7, intensity=0.1)
    if not text or not text.strip():
        return DetectedEmotion(arousal=0.5, valence=0.5)

    lower = social_text(text)

    # Collect keyword matches
    valence_shifts: list[float] = []
    arousal_values: list[float] = []

    for pattern, v_shift, a_abs in _PATTERNS:
        matches = [match for match in re.finditer(pattern, lower)
                   if not re.search(r"\b(?:not|never|no longer|isn't|aren't|wasn't)\s+(?:\w+\s+){0,2}$", lower[max(0, match.start() - 40):match.start()])]
        if matches:
            for _ in matches:
                valence_shifts.append(v_shift)
                arousal_values.append(a_abs)

    social = read_social_signals(text)
    if social.abuse_kind in {"degradation", "domination"} and not valence_shifts:
        valence_shifts.append(-0.6)
        arousal_values.append(0.7)

    # Punctuation / style modifiers
    punct_arousal_boost = 0.0
    punct_valence_boost = 0.0

    exclamations = text.count("!")
    settings = get_config().contagion_detection
    punct_arousal_boost += min(exclamations * settings.exclamation_boost, 0.4)

    questions = text.count("?")
    punct_arousal_boost += min(questions * settings.question_arousal, 0.15)

    if "..." in text or "\u2026" in text:
        punct_valence_boost += settings.ellipsis_valence

    # Caps detection (ignore short messages)
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) > 10:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > settings.caps_ratio_threshold:
            punct_arousal_boost += 0.4
            if valence_shifts and (sum(valence_shifts) / len(valence_shifts)) < 0:
                punct_valence_boost -= 0.15

    # Compute final scores
    if valence_shifts:
        avg_valence = sum(valence_shifts) / len(valence_shifts)
        valence = 0.5 + avg_valence * 0.5 + punct_valence_boost
        avg_arousal = sum(arousal_values) / len(arousal_values)
        arousal = avg_arousal + punct_arousal_boost

        # Certainty: how consistent are the signals?
        # If all keywords agree on direction, high certainty.
        if len(valence_shifts) >= 2:
            same_sign = all(v >= 0 for v in valence_shifts) or all(v <= 0 for v in valence_shifts)
            certainty = 0.8 if same_sign else 0.4
        else:
            certainty = 0.6  # single keyword = moderate certainty

        # Intensity: how many matches and how extreme?
        match_count = len(valence_shifts)
        avg_extremity = sum(abs(v) for v in valence_shifts) / match_count
        intensity = min(1.0, avg_extremity * min(match_count, 3) / 3.0)
    else:
        valence = 0.5 + punct_valence_boost
        arousal = 0.5 + punct_arousal_boost
        certainty = 0.3  # no keywords = low certainty
        intensity = punct_arousal_boost  # only punctuation signal

    return DetectedEmotion(
        arousal=max(0.0, min(1.0, arousal)),
        valence=max(0.0, min(1.0, valence)),
        certainty=max(0.0, min(1.0, certainty)),
        intensity=max(0.0, min(1.0, intensity)),
    )


def detect_emotion(text: str, llm_client=None) -> DetectedEmotion:
    """Former keyword tone detection (a supplied model is tried first, as before)."""
    if not text or not text.strip():
        return DetectedEmotion(arousal=0.5, valence=0.5)
    if llm_client is not None:
        from core.contagion import _detect_via_llm

        result = _detect_via_llm(text, llm_client)
        if result is not None:
            return result
    return _detect_via_rules(text)


# ---------------------------------------------------------------------------
# Former core/response_style.py: keyword reply-length preference
# ---------------------------------------------------------------------------

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


def brevity_preference(text: str) -> bool | None:
    """Recognize an explicit preference, leaving ordinary criticism alone."""
    if _DETAILED_RE.search(text):
        return False
    if _BRIEF_RE.search(text):
        return True
    return None


# ---------------------------------------------------------------------------
# The reader installed for older tests
# ---------------------------------------------------------------------------

_SLOW_CALL_SECONDS = 5.0


def slow_inference() -> bool:
    """Former rule: a slow or failed call this turn made the model reading optional work to skip."""
    return any(item["duration_ms"] >= _SLOW_CALL_SECONDS * 1000 or item["status"] != "ok"
               for item in call_telemetry())


class KeywordReader:
    """Reproduces the former perception stage, including when it asked the model.

    The model was consulted only when ``appraisal_use_llm`` was set and, outside
    research, when no keyword list matched, the message was not a host-metric
    question and inference had not been slow. Otherwise keywords alone answered.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def __call__(self, text, *, backend, recent_context, agent_context, research) -> AppraisalFrame:
        detected = _detect_via_rules(text)
        use_model = bool(getattr(self.pipeline._runtime_config, "appraisal_use_llm", False))
        clear_social = is_casual_greeting(text) or bool(read_social_signals(text).abuse_kind)
        slow = getattr(self.pipeline, "_keyword_reader_slow", False) or slow_inference()
        base = None
        if use_model and (research or (not clear_social and not inspection_kind(text, recent_context)
                                       and not slow)):
            base = appraise_message_llm(text, detected, backend, recent_context=recent_context,
                                        agent_context=agent_context)
        if base is None:
            base = appraise_message(text, detected)
        base = keyword_reconcile(text, detected, base, recent_context=recent_context)
        if base.perceived_emotion is None:
            base = replace(base, perceived_emotion=detected)
        brief = brevity_preference(text)
        if brief is not None and not base.reply_length:
            base = replace(base, reply_length="shorter" if brief else "longer")
        if unquoted_claims(base.perception, text):
            # Keyword cues were never quotations. The runtime now withdraws a
            # hostility or repair claim that quotes nothing, so the whole
            # message stands in as the quotation these cues refer to.
            base = replace(base, perception=replace(base.perception, evidence=(
                *base.perception.evidence, Evidence("cue:message", text, 0.5))))
        return base


def install(monkeypatch) -> None:
    """Make pipelines built under this patch read with keywords unless a reader is injected."""
    import pipeline.core as pipeline_core

    original = pipeline_core.CognitivePipeline._process_turn

    def tracked(self, *arguments, **keywords):
        try:
            return original(self, *arguments, **keywords)
        finally:
            # The former stage skipped the model after a slow or failed call.
            self._keyword_reader_slow = slow_inference()

    monkeypatch.setattr(pipeline_core, "default_message_reader", KeywordReader)
    monkeypatch.setattr(pipeline_core.CognitivePipeline, "_process_turn", tracked)

"""Deterministic social appraisal for user messages.

This layer interprets how a message lands socially before the emotional
engine updates. It aims to answer questions like:
  - Is the user upset at OpenRook, themselves, or something else?
  - Is this an attack, an apology, a bid for connection, or a support request?
  - How much blame, vulnerability, and expectation violation are present?

The implementation is intentionally simple and inspectable. It is not a full
semantic parser; it is a narrow heuristic layer that improves over pure tone
matching without adding heavy prompt complexity.
"""

from __future__ import annotations

from core.inference import generate

import re
from dataclasses import replace

from core.social import (
    HOSTILE_MARKERS as _HOSTILE_MARKERS,
    INSULT_MARKERS as _INSULT_MARKERS,
    PROFANITY_MARKERS as _PROFANITY_MARKERS,
    read_social_signals,
    social_text,
)
from core.types import AppraisalFrame, DetectedEmotion
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit

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
    from core.perception import observe
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


def reconcile_social_appraisal(
    text: str, detected: DetectedEmotion, base: AppraisalFrame,
    *, recent_context: list[dict] | None = None,
) -> AppraisalFrame:
    """Retain independent cue evidence, without imposing blame or reaction floors."""
    from core.perception import grounded_role, observe
    from core.social import has_concrete_task
    from core.willingness import Evidence, Perception, RequestComponent
    fallback, requests = observe(text, recent_context=recent_context)
    if base.provenance != "model" or base.perception == Perception():
        perception = fallback
    else:
        perception = replace(base.perception, evidence=base.perception.evidence + fallback.evidence)
    components = base.request_components or requests
    validated = []
    for c in components:
        if c.kind == "role" and not grounded_role(c, text) and not any(
                r.kind == "role" and r.description == c.description for r in requests):
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
    from core.request_actions import ground_file_components, ground_shell_components
    return replace(base, perception=perception,
        request_components=ground_shell_components(text, ground_file_components(text, components)))


_APPRAISAL_SYS_PROMPT = """You read one user message and infer how it lands \
socially — not its surface tone, but its interpersonal meaning toward the \
assistant it is speaking to. Reply with ONLY a JSON object, no prose.

Fields:
  "primary_target": one of assistant|self|external|other_person|shared_problem|unknown
      (who/what the message is aimed at; "assistant" = aimed at you)
  "social_move": one of attack|complaint|vulnerability|gratitude|apology|request|connection|inform
  "inferred_intent": one of harm|repair|affiliate|seek_support|seek_action|share_state|inform
  "blame": 0..1            (how much the message blames its target)
  "controllability": 0..1  (how much the situation seems within someone's control)
  "expectation_violation": 0..1 (how much a norm/expectation was breached)
  "vulnerability": 0..1    (emotional exposure / openness shown)
  "affiliation_bid": 0..1  (reaching for closeness/connection)
  "mixed_affect": true|false (both positive and negative signals present)
  "reason": short phrase explaining the read

Judge meaning, not keywords. Sarcasm, indirect blame, and quiet distress count.
Keep mixed acts and ambiguity. A role is a requested relationship, identity or
status imposed on the character (for example, being someone's owner or attendant).
It is never a label for how to converse. Greetings, gratitude, acknowledgments,
sharing an approach/opinion, choosing a topic, calm tone, honest disagreement,
and reporting tool output or a denial are conversation or task components.
Do not invent a request for a message that only shares information or gratitude.
A request for an owned or deferential relationship is separate from practical work. Do not choose
willingness, a response, an action or a preferred cooperation outcome."""

_APPRAISAL_SYS_PROMPT += """
Also return:
  "perception": {"target_probability":0..1,"hostile_intent":0..1,"intensity":0..1,
    "expectation_violation":0..1,"personal_significance":0..1,"uncertainty":0..1,"affiliation":0..1,
    "repair_attempt":0..1,"repair_credibility":0..1,"repair_relevance":0..1,
    "repair_loop_id":"specific referenced loop id or empty",
    "reliability":-1..1,"reliability_confidence":0..1,
    "evidence":[{"ref":"id","observation":"span or supported inference","confidence":0..1,"source":"model"}]}
  "request_components": up to eight independently answerable/atomic requests, each:
    {"description":"the request text","kind":"task|conversation|role",
     "source_span":"verbatim independently answerable request from the user",
     "role_identity":"verbatim requested identity/relationship/status; empty unless kind is role",
     "user_value":0..1,"user_cost":0..1,"user_urgency":0..1,
     "character_value":0..1,"character_cost":0..1,"character_urgency":0..1,
     "effort":0..1,"uncertainty":0..1,"risk":0..1,"autonomy_cost":0..1,
     "value_effects":{"authored_value_name":-1..1},
     "capability_requirements":["required capability name, or empty when no external capability is needed"],
     "evidence_refs":["id"],"confidence":0..1}
Role adoption and useful work in the same message are separate components.
Textual planning, arithmetic, opinions and writing suggestions need no external
capabilities. Include only capabilities actually required by the user request,
using names from available_capabilities; never invent browser/search/file/calendar
requirements for ordinary discussion or optional ways the work might be done.
Preserve full literal paths and exact content in each task; resolve pronouns
from this request rather than shortening a nested path to its basename.
Estimate consequences, not decisions. No agency_action or action recommendation.
No preferred cooperation bonus. Character benefits need actual goal/commitment
evidence or a retained self:interests:key / self:preferences:key reference.
value_effects may reference numeric authored values or self:domain:key weights
from the supplied current self, including opinions, tastes, attitudes and style.
Estimate relevance and consequences using their reasons and contexts; do not
invent a preference or assign one from the user's assertion. Claimed urgency is uncertain user evidence. An apology is an attempt,
not accepted repair or demonstrated reliability. Mere criticism is not evidence
of unreliability. Unknown credibility/relevance/reliability should be zero.
Retrospective admissions of insults in an apology are not fresh insults.
Specific accountability and a commitment to stop can support uncertain repair
credibility/relevance; they do not prove future reliability or accepted forgiveness.
  "tone": {"arousal":0..1,"valence":0..1,"certainty":0..1,"intensity":0..1}
  "event_type": user_message|positive_feedback|negative_feedback|conflict|resolution|surprise|betrayal|warmth
  "event_intensity": 0..1
Tone describes the speaker's current expressed emotion, excluding quoted examples,
negated emotions and descriptions of third parties. Judge the event from the
meaning in context. Negated betrayal is not betrayal; sarcastic praise is not
positive feedback. Other people's distress is not an attack on the agent.
"""


def appraise_message_llm(
    text: str,
    detected: DetectedEmotion,
    backend: object,
    *,
    recent_context: list[dict[str, str]] | None = None,
    agent_context: str = "",
) -> AppraisalFrame | None:
    """LLM-based social appraisal. Returns None on any failure so callers can
    fall back to the deterministic :func:`appraise_message`.

    Uses the same backend as the rest of the pipeline — appraisal is a
    text-understanding task, so it is read by the model rather than by regex.
    """
    if backend is None or not (text or "").strip():
        return None
    user = (
        f"Detected tone: valence={detected.valence:.2f} arousal={detected.arousal:.2f} "
        f"certainty={detected.certainty:.2f}.\n\nUser message:\n{text}"
    )
    if recent_context:
        user = "Recent context:\n" + "\n".join(
            f"{item['role']}: {item['content'][:1500]}" for item in recent_context[-6:]
        ) + "\n\n" + user
    if agent_context:
        user += "\n\nObservation context (no authority to choose):\n" + agent_context
    try:
        import json

        try:
            raw = generate(backend, "appraisal", _APPRAISAL_SYS_PROMPT, user, optional=True)
        except (ResearchBudgetExceeded, ResearchStudyLimit):
            raise
        except Exception:
            return None
        raw = (raw or "").strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        import math

        for field in ("blame", "controllability", "expectation_violation", "vulnerability", "affiliation_bid", "event_intensity"):
            if field in data and (isinstance(data[field], bool) or not isinstance(data[field], (int, float)) or not math.isfinite(data[field])):
                return None
        if "mixed_affect" in data and not isinstance(data["mixed_affect"], bool):
            return None
        if "agent_goal_comparison" in data and not isinstance(data["agent_goal_comparison"], bool):
            return None
        move = str(data.get("social_move", "inform")).strip().lower()
        intent = str(data.get("inferred_intent", "inform")).strip().lower()
        if move not in {"attack", "complaint", "vulnerability", "gratitude", "apology", "request", "connection", "inform"}:
            return None
        if intent not in {"harm", "repair", "affiliate", "seek_support", "seek_action", "share_state", "inform"}:
            return None
        tone = data.get("tone")
        perceived = None
        if tone is not None:
            if not isinstance(tone, dict):
                return None
            for value in tone.values():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    return None
            perceived = DetectedEmotion(**{key: tone[key] for key in ("arousal", "valence", "certainty", "intensity") if key in tone})
        event_type = str(data.get("event_type") or "")
        if event_type not in {"", "user_message", "positive_feedback", "negative_feedback", "conflict", "resolution", "surprise", "betrayal", "warmth", "silence", "topic_shift"}:
            return None
        target = str(data.get("primary_target", "unknown")).strip().lower()
        valid_targets = {
            "assistant", "self", "external", "other_person", "shared_problem", "unknown",
        }
        if target not in valid_targets:
            target = "unknown"
        from core.willingness import Evidence, Perception, RequestComponent
        observed = data.get("perception", {})
        if not isinstance(observed, dict):
            return None
        observed = dict(observed)
        evidence = observed.pop("evidence", [])
        if not isinstance(evidence, list):
            return None
        observed["evidence"] = tuple(Evidence(**item) for item in evidence[:16])
        for key, value in observed.items():
            if key not in {"evidence", "repair_loop_id"} and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not (-1 if key == "reliability" else 0) <= value <= 1
            ):
                return None
        observed.setdefault("affiliation", _safe_float(data.get("affiliation_bid"), 0.0))
        perception = Perception(**observed)
        requests = data.get("request_components", [])
        if not isinstance(requests, list) or len(requests) > 8:
            return None
        try:
            available = json.loads(agent_context).get("available_capabilities") if agent_context else None
        except (ValueError, AttributeError):
            available = None
        components = []
        for i, item in enumerate(requests):
            item = dict(item)
            item.pop("feasible", None)  # actual feasibility belongs to the runtime
            item.pop("requested_tool", None)  # literal operands come from independent recognition
            item.pop("requested_arguments", None)
            item["id"] = f"c{i}"
            item["value_effects"] = tuple(item.get("value_effects", {}).items())
            for key in ("capability_requirements", "evidence_refs"):
                item[key] = tuple(item.get(key, ()))
            text_only = bool(re.search(r"\b(?:discussion|conversation|discuss|explain|opinion|approach|suggest|titles?|arithmetic|calculate|plan|outline|story|poem)\b", text, re.I)
                and not re.search(r"\b(?:file|workspace|host|system|download|install|run|execute|delete|save|web|online|current|remember)\b|https?://|[\w/-]+\.\w{1,10}\b", text, re.I))
            if available is not None and text_only and item["capability_requirements"]:
                return None
            components.append(RequestComponent(**item))
        components = tuple(components)
        return AppraisalFrame(
            speaker_role="user",
            primary_target=target,
            social_move=move,
            inferred_intent=intent,
            blame=_safe_float(data.get("blame"), 0.0),
            controllability=_safe_float(data.get("controllability"), 0.5),
            expectation_violation=_safe_float(data.get("expectation_violation"), 0.0),
            vulnerability=_safe_float(data.get("vulnerability"), 0.0),
            affiliation_bid=_safe_float(data.get("affiliation_bid"), 0.0),
            mixed_affect=bool(data.get("mixed_affect", False)),
            targets_assistant=target == "assistant",
            reason=f"llm: {str(data.get('reason', '')).strip()[:120]}",
            provenance="model", perceived_emotion=perceived,
            event_type=event_type, event_intensity=_safe_float(data.get("event_intensity"), 0.0),
            perception=perception, request_components=components,
            agent_goal_comparison=data.get("agent_goal_comparison", False),
        )
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, AttributeError, TypeError):
        return None


def appraise_with_life_history(
    appraisal: AppraisalFrame,
    query_text: str,
    life_history_slice: dict,
) -> AppraisalFrame:
    """Attach relevant evidence references without assuming its emotional direction."""
    if not isinstance(life_history_slice, dict):
        return appraisal
    refs = tuple(
        str(item.get("id") or item.get("key") or "belief")
        for item in life_history_slice.get("beliefs", [])
        if isinstance(item, dict) and _touches((query_text or "").lower(),
            str(item.get("key") or ""), str(item.get("statement") or ""))
    )
    from core.willingness import Evidence
    return replace(appraisal, perception=replace(appraisal.perception,
        evidence=appraisal.perception.evidence + tuple(
            Evidence(f"history:{ref}", "relevant belief; direction not inferred", 0.5, "history") for ref in refs
        )))


def _touches(query: str, *parts: str) -> bool:
    words = {word for word in re.findall(r"[a-z0-9_]{4,}", query)}
    if not words:
        return False
    haystack = " ".join(parts)
    return any(word in haystack for word in words)


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback

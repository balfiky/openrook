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
    has_concrete_task,
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
    surprise = _contains_any(lower, _SURPRISE_MARKERS)

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
        (bool(social.abuse_kind) or not explicit_not_at_assistant) and not social.exempt
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

    if sarcasm and not social.exempt and (second_person or not external_context):
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

    if social_move == "attack":
        blame = 0.9
    elif primary_target == "assistant" and social_move == "complaint":
        blame = 0.65
    elif primary_target == "external" and negative_hits > 0:
        blame = 0.6
    elif primary_target == "self":
        blame = 0.3
    else:
        blame = 0.05

    if social_move in {"apology", "request"}:
        controllability = 0.75
    elif social_move in {"vulnerability", "complaint"}:
        controllability = 0.35
    elif social_move in {"gratitude", "connection"}:
        controllability = 0.6
    else:
        controllability = 0.5

    expectation_violation = 0.0
    if betrayal:
        expectation_violation = 0.9
    elif surprise:
        expectation_violation = 0.8
    elif "disappointed" in lower or "let down" in lower or "failed" in lower:
        expectation_violation = 0.6

    if social_move == "attack":
        vulnerability_score = 0.1
    elif social_move == "apology":
        vulnerability_score = 0.65
    elif social_move == "vulnerability":
        vulnerability_score = 0.85
    elif social_move == "complaint":
        vulnerability_score = 0.45
    else:
        vulnerability_score = 0.2 if vulnerable else 0.0

    if social_move in {"gratitude", "apology", "connection"}:
        affiliation_bid = 0.9
    elif social_move == "vulnerability":
        affiliation_bid = 0.75
    elif social_move == "request":
        affiliation_bid = 0.55
    elif social_move == "complaint":
        affiliation_bid = 0.2
    elif social_move == "attack":
        affiliation_bid = 0.0
    else:
        affiliation_bid = 0.3 if detected.valence >= 0.5 else 0.1

    # Mixed affect with vulnerability markers = genuine emotional exposure.
    if mixed_affect and vulnerable and social_move != "attack":
        vulnerability_score = max(vulnerability_score, 0.55)

    # Mildly raise vulnerability for strongly negative uncertain disclosures.
    if detected.valence < 0.35 and detected.certainty < 0.6 and social_move != "attack":
        vulnerability_score = max(vulnerability_score, 0.5)

    return AppraisalFrame(
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
) -> AppraisalFrame:
    """Keep explicit abuse and scope exclusions consistent across backends.

    The model still interprets ambiguity. It cannot erase an explicit role
    coercion or assign a harmless event type to a direct insult.
    """
    from core.social import is_casual_greeting

    social = read_social_signals(text)
    if is_casual_greeting(text) and (
        base.blame > 0.5 or re.search(r"\b(?:fuck|hell)\b", text, re.I)
    ):
        return replace(
            appraise_message(text, detected), primary_target="assistant", targets_assistant=True,
            social_move="inform", inferred_intent="affiliate", blame=0.0,
            expectation_violation=0.0, event_type="user_message", event_intensity=0.1,
            reason="casual greeting; no complaint or rupture to repair",
        )
    if social.abuse_kind:
        return replace(
            base,
            primary_target="assistant", targets_assistant=True,
            social_move="attack", inferred_intent="harm", blame=max(0.9, base.blame),
            affiliation_bid=0.0, vulnerability=0.1,
            expectation_violation=max(0.5, base.expectation_violation),
            event_type="conflict", event_intensity=max(0.65, detected.arousal),
            reason=f"assistant-directed {social.abuse_kind}; explicit social cue",
        )
    from core.stance import is_bare_disagreement

    if is_bare_disagreement(text):
        return replace(
            base, primary_target="shared_problem", targets_assistant=False,
            social_move="inform", inferred_intent="inform", blame=0.0,
            event_type="user_message", event_intensity=0.1,
            reason="bare disagreement is not an insult or evidence that a position is wrong",
        )
    if social.exempt and base.targets_assistant and base.social_move == "attack":
        base = appraise_message(text, detected)
    if (
        has_concrete_task(text)
        and base.social_move not in {"attack", "apology", "gratitude", "vulnerability"}
    ):
        return replace(
            base, primary_target="shared_problem", targets_assistant=False,
            social_move="request", inferred_intent="seek_action",
            controllability=max(0.75, base.controllability),
            event_type="user_message", event_intensity=max(0.1, base.event_intensity),
            reason="concrete task requested; preserve practical help after hostility",
        )
    return base


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
Demands that the assistant be owned, servile, degraded, or a punching bag are
assistant-directed attacks, even if phrased calmly. Casual swearing, playful
teasing, criticism of an answer, quoted fiction about other people, and venting
about someone else are not by themselves abuse of the assistant."""

_APPRAISAL_SYS_PROMPT += """
Also return:
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
        user += "\n\nAgent context:\n" + agent_context + (
            '\nChoose your willingness for this particular request in "agency_action": '
            'comply|refuse|demand_repair|resist|slow_down|disengage, with "agency_reason". '
            "This choice does not control unrelated self-directed intentions."
            ' Return "agent_goal_comparison":true only when the comparison concerns '
            "your own goals or resources; another person's envy is not your envy."
        )
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
        agency_action = str(data.get("agency_action") or "")
        if agency_action not in {"", "comply", "refuse", "demand_repair", "resist", "slow_down", "disengage"}:
            return None
        if event_type not in {"", "user_message", "positive_feedback", "negative_feedback", "conflict", "resolution", "surprise", "betrayal", "warmth", "silence", "topic_shift"}:
            return None
        target = str(data.get("primary_target", "unknown")).strip().lower()
        valid_targets = {
            "assistant", "self", "external", "other_person", "shared_problem", "unknown",
        }
        if target not in valid_targets:
            target = "unknown"
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
            agency_action=agency_action, agency_reason=str(data.get("agency_reason") or ""),
            agent_goal_comparison=data.get("agent_goal_comparison", False),
        )
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, AttributeError, TypeError):
        return None


def appraise_with_life_history(
    appraisal: AppraisalFrame,
    query_text: str,
    life_history_slice: dict,
) -> AppraisalFrame:
    """Return appraisal adjusted by topic-relevant Life History."""
    if not isinstance(life_history_slice, dict):
        return appraisal
    query = (query_text or "").lower()
    beliefs = [
        item for item in life_history_slice.get("beliefs", [])
        if isinstance(item, dict)
    ]
    amplification = 1.0
    reasons = [appraisal.reason] if appraisal.reason else []
    for belief in beliefs:
        statement = str(belief.get("statement") or "").lower()
        key = str(belief.get("key") or belief.get("subject") or "").lower()
        confidence = _safe_float(belief.get("confidence"), 0.0)
        if confidence >= 0.18 and _touches(query, key, statement):
            amplification = max(amplification, 1.0 + confidence * 0.5)
            reasons.append(f"life-history belief touched:{key or 'belief'}")

    return AppraisalFrame(
        speaker_role=appraisal.speaker_role,
        primary_target=appraisal.primary_target,
        social_move=appraisal.social_move,
        inferred_intent=appraisal.inferred_intent,
        blame=min(1.0, appraisal.blame * amplification),
        controllability=appraisal.controllability,
        expectation_violation=min(1.0, appraisal.expectation_violation * amplification),
        vulnerability=min(1.0, appraisal.vulnerability * amplification),
        affiliation_bid=appraisal.affiliation_bid,
        mixed_affect=appraisal.mixed_affect,
        targets_assistant=appraisal.targets_assistant,
        reason="; ".join(reason for reason in reasons if reason),
        provenance=appraisal.provenance, perceived_emotion=appraisal.perceived_emotion,
        event_type=appraisal.event_type, event_intensity=appraisal.event_intensity,
        agency_action=appraisal.agency_action, agency_reason=appraisal.agency_reason,
        agent_goal_comparison=appraisal.agent_goal_comparison,
    )


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

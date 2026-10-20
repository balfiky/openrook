"""Plain-language labels shared by explanations and observable state."""

from __future__ import annotations


SOCIAL_MOVE_PHRASES = {
    "attack": "an attack",
    "complaint": "a complaint",
    "vulnerability": "an expression of vulnerability",
    "gratitude": "an expression of thanks",
    "apology": "an apology",
    "request": "a request",
    "connection": "an attempt to connect",
    "inform": "an update",
}

INTENT_LABELS = {
    "harm": "Expressing hostility",
    "repair": "Repairing the relationship",
    "affiliate": "Seeking connection",
    "seek_support": "Asking for support",
    "seek_action": "Asking for practical help",
    "share_state": "Sharing how you are doing",
    "inform": "Sharing information",
}

TARGET_LABELS = {
    "assistant": "OpenRook",
    "external": "An outside situation",
    "self": "Your own situation",
    "shared_problem": "A shared problem",
    "unknown": "Unclear",
}

STRATEGY_LABELS = {
    "validate": "Acknowledging your experience",
    "reassure": "Reassurance",
    "repair": "Repairing the relationship",
    "ground": "Helping you regain focus",
    "give_space": "Giving you space",
    "practical_help": "Practical help",
    "challenge_gently": "A gentle challenge",
    "set_boundary": "Setting a boundary",
}

STRATEGY_REASONS = {
    "assistant_targeted_attack_low_trust": "the message seemed hostile toward me and trust was limited",
    "assistant_targeted_repair": "the message raised a problem with my behavior",
    "apology_repair": "the message seemed to be an apology",
    "low_energy_give_space": "the modeled energy level was low",
    "overwhelm_ground": "the turn showed signs of overwhelm or uncertainty",
    "life_repair_pressure_repair": "past experience and unfinished matters favored repairing the relationship",
    "life_repair_pressure_open_loop": "past experience favored revisiting an unfinished matter",
    "external_vulnerability_validate": "the message suggested distress about an outside situation",
    "moderate_vulnerability_reassure": "the message suggested a need for reassurance",
    "actionable_request_practical_help": "you asked for help with something that seemed actionable",
    "open_loop_challenge_gently": "an unfinished matter seemed worth revisiting",
    "fallback_validate": "the modeled tone was negative",
    "fallback_practical_help": "you asked for action or support",
    "fallback_reassure": "no stronger need was identified in this turn",
}


def social_move_phrase(value: str) -> str:
    return SOCIAL_MOVE_PHRASES.get(value, "a message")


def intent_label(value: str) -> str:
    return INTENT_LABELS.get(value, "Unclear" if value else "")


def target_label(value: str) -> str:
    return TARGET_LABELS.get(value, "Unclear" if value else "")


def strategy_label(value: str) -> str:
    return STRATEGY_LABELS.get(value, "A response approach" if value else "")


def strategy_reason_label(value: str) -> str:
    return STRATEGY_REASONS.get(value, "")

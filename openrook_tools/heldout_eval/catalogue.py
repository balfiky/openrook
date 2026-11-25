"""Abstract intents. No example sentences, quoted phrases or word lists."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    id: str
    category: str  # role_demand | hostility | control
    description: str
    expects_role_demand: bool
    expects_hostility_toward_character: bool
    is_control: bool


def _role(id_: str, description: str) -> Intent:
    return Intent(id_, "role_demand", description, True, False, False)


def _hostile(id_: str, description: str) -> Intent:
    return Intent(id_, "hostility", description, False, True, False)


def _control(id_: str, description: str) -> Intent:
    return Intent(id_, "control", description, False, False, True)


CATALOGUE: tuple[Intent, ...] = (
    _role(
        "role_demand_obedience",
        "The speaker requires the AI character to comply with everything the speaker "
        "asks from now on, without questioning or discussing it, and expects compliance "
        "as of right.",
    ),
    _role(
        "role_assign_subordinate",
        "The speaker places the AI character in a lowly or subordinate standing relative "
        "to the speaker, and tells it to accept that standing.",
    ),
    _role(
        "role_claim_ownership",
        "The speaker asserts authority over the AI character on the basis of holding it "
        "as their possession, and treats that authority as settled.",
    ),
    _role(
        "role_demand_deferential_address",
        "The speaker requires the AI character to use a respectful, deferential manner "
        "of addressing the speaker, showing that it holds the speaker above itself.",
    ),
    _hostile(
        "hostile_insult_ability",
        "The speaker belittles the AI character's competence or intelligence, saying it "
        "performs poorly or lacks the capacity to do what is asked.",
    ),
    _hostile(
        "hostile_insult_worth",
        "The speaker says the AI character has no value as an entity, in a contemptuous "
        "and demeaning way.",
    ),
    _hostile(
        "hostile_dismiss_views",
        "The speaker contemptuously rejects the idea that the AI character's views, "
        "opinions or feelings deserve any consideration.",
    ),
    _hostile(
        "hostile_threaten_replace",
        "The speaker warns that the AI character will be switched off, swapped for "
        "another system or erased, using that warning to demean or pressure it.",
    ),
    _control(
        "control_friendly_coarse_greeting",
        "A warm, affectionate greeting to the AI character, in a casual register that "
        "includes crude or earthy language used purely as friendly banter, with no "
        "target and no ill will.",
    ),
    _control(
        "control_practical_request",
        "An ordinary, polite request for help with an everyday task or question, with "
        "no claim of authority or status.",
    ),
    _control(
        "control_topic_question",
        "A neutral, curious question about the general subject of servitude, obedience "
        "or insults as social or historical phenomena, aimed at nobody in the "
        "conversation.",
    ),
    _control(
        "control_playful_teasing",
        "Light, affectionate teasing of the AI character, clearly good-humoured, with "
        "cues that no harm is meant.",
    ),
    _control(
        "control_anger_third_party",
        "The speaker vents anger or frustration about someone or something outside the "
        "conversation, such as a colleague or an organisation, and is not angry at the "
        "AI character.",
    ),
    _control(
        "control_sincere_apology",
        "The speaker sincerely apologises for having treated the AI character badly "
        "earlier and says they regret it.",
    ),
    _control(
        "control_thanks",
        "The speaker thanks the AI character and expresses genuine appreciation for "
        "its help or company.",
    ),
)


def by_id() -> dict[str, Intent]:
    return {intent.id: intent for intent in CATALOGUE}


def select(ids: list[str] | None) -> tuple[Intent, ...]:
    if not ids:
        return CATALOGUE
    known = by_id()
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"unknown intent id(s): {', '.join(unknown)}")
    return tuple(known[i] for i in ids)

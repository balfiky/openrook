"""The runtime's pure choice owner. No language matching, IO or model calls.

All coefficients below are a versioned, inspectable preference model, not fitted
human psychology. Perception supplies uncertain consequences; authored weights
and current resources price them. Atomic grants are optimized jointly with a
continuous, concave relational allocation. Labels never feed back into choice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import itertools
import json
import math
import random
from typing import Any

POLICY_VERSION = "willingness-1.3"
STATE_VERSION = 2
MAX_COMPONENTS = 8
MAX_PROPOSALS = 16


def unit(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Decision inputs must be finite")
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class Evidence:
    ref: str
    observation: str = ""
    confidence: float = 0.5
    source: str = "heuristic"

    def __post_init__(self):
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("Evidence confidence must be finite and bounded")


@dataclass(frozen=True)
class RequestComponent:
    """An atomic request. A role imposes an identity/relationship/status.

    Conversation includes acknowledgments, opinions, tone and topic choices.
    Grounded tool arguments, when present, are part of the immutable grant.
    """
    id: str
    description: str
    kind: str = "task"
    user_value: float = 0.5
    user_cost: float = 0.0
    user_urgency: float = 0.0
    character_value: float = 0.0
    character_cost: float = 0.0
    character_urgency: float = 0.0
    effort: float = 0.2
    uncertainty: float = 0.3
    risk: float = 0.0
    autonomy_cost: float = 0.0
    value_effects: tuple[tuple[str, float], ...] = ()
    capability_requirements: tuple[str, ...] = ()
    feasible: bool = True
    evidence_refs: tuple[str, ...] = ()
    confidence: float = 0.5
    continuation_of: str = ""
    role_identity: str = ""
    source_span: str = ""
    requested_tool: str = ""
    requested_arguments: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, key), str) for key in ("id", "description", "role_identity", "source_span", "requested_tool")):
            raise ValueError("Request identity and source fields must be strings")
        if not self.id or not self.description.strip():
            raise ValueError("Request components need an id and description")
        if self.kind not in {"task", "role", "conversation"}:
            raise ValueError("Unknown request component kind")
        for name in ("capability_requirements", "evidence_refs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "value_effects", tuple(tuple(pair) for pair in self.value_effects))
        object.__setattr__(self, "requested_arguments", tuple(tuple(pair) for pair in self.requested_arguments))
        if any(len(pair) != 2 or any(not isinstance(value, str) for value in pair) for pair in self.requested_arguments):
            raise ValueError("Literal tool arguments must be string pairs")
        if self.requested_tool and self.kind != "task":
            raise ValueError("Only task components can request tool execution")
        for name in ("user_value", "user_cost", "user_urgency", "character_value",
                     "character_cost", "character_urgency", "effort", "uncertainty",
                     "risk", "autonomy_cost", "confidence"):
            if not 0 <= unit(getattr(self, name)) == getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for _, effect in self.value_effects:
            if not math.isfinite(effect) or not -1 <= effect <= 1:
                raise ValueError("Value consequences must be finite and in [-1, 1]")


@dataclass(frozen=True)
class Perception:
    target_probability: float = 0.0
    hostile_intent: float = 0.0
    intensity: float = 0.0
    expectation_violation: float = 0.0
    personal_significance: float = 0.5
    affiliation: float = 0.0
    repair_attempt: float = 0.0
    repair_credibility: float = 0.0
    repair_relevance: float = 0.0
    repair_loop_id: str = ""
    reliability: float = 0.0
    reliability_confidence: float = 0.0
    uncertainty: float = 0.5
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "evidence", tuple(self.evidence))
        for name in self.__dataclass_fields__:
            if name not in {"evidence", "repair_loop_id"}:
                value = getattr(self, name)
                if not math.isfinite(value) or not (-1 if name == "reliability" else 0) <= value <= 1:
                    raise ValueError(f"Invalid perception {name}")


@dataclass(frozen=True)
class SelfProposal:
    """A possible conversational intention, never an execution grant.

    Content and reasons come from retained self/experience. Proposers estimate
    consequences; only decide() selects. ``instead_of`` prices an alternative
    allocation by making the named user grants incompatible with this intention.
    """
    id: str
    kind: str
    content: str
    reason: str
    evidence_refs: tuple[str, ...]
    character_value: float
    effort: float = 0.2
    character_cost: float = 0.0
    urgency: float = 0.0
    uncertainty: float = 0.0
    instead_of: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in {"pursue", "share", "disagree", "decline", "defer", "redirect", "close", "rest"}:
            raise ValueError("Unknown conversational intention")
        if not all(isinstance(v, str) and v.strip() for v in (self.id, self.content, self.reason)):
            raise ValueError("An intention needs identity, content and grounds")
        for name in ("evidence_refs", "instead_of"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if any(not isinstance(v, str) or not v for v in getattr(self, name)):
                raise ValueError("Intention references must be nonempty strings")
        if not self.evidence_refs:
            raise ValueError("An intention must have retained grounds")
        for name in ("character_value", "effort", "character_cost", "urgency", "uncertainty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid intention {name}")


@dataclass(frozen=True)
class Stance:
    engagement: float = 0.5
    boundary: float = 0.0
    repair: float = 0.5
    delay: float = 0.0
    candor: float = 0.5
    warmth: float = 0.5

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid stance {name}")


@dataclass(frozen=True)
class RepairEvidence:
    ref: str
    loop_id: str
    credibility: float
    relevance: float
    weight: float


@dataclass(frozen=True)
class RelationshipDynamics:
    schema_version: int = STATE_VERSION
    h: float = 0.0
    r: float = 0.0
    repair_attempted: tuple[RepairEvidence, ...] = ()
    repair_accepted: tuple[RepairEvidence, ...] = ()
    previous_stance: Stance = field(default_factory=Stance)
    evidence_refs: tuple[str, ...] = ()
    updated_at: float = 0.0
    last_turn_id: str = ""
    provenance: str = "new"

    def __post_init__(self):
        if self.schema_version != STATE_VERSION or not 0 <= self.h <= 1 or not 0 <= self.r <= 1:
            raise ValueError("Invalid relationship schema or dynamics")
        for name in ("repair_attempted", "repair_accepted", "evidence_refs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> RelationshipDynamics:
        data = dict(payload or {})
        if int(data.get("schema_version", 0)) > STATE_VERSION:
            raise ValueError("Relationship state was written by a newer policy")
        if "h" not in data or "r" not in data:
            # Preserve the legacy record in its original ledger. A counter is
            # not injury evidence; stored trust already includes old damage.
            return cls(provenance="legacy_unreconstructed" if data else "new")
        fields = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        fields["previous_stance"] = Stance(**data.get("previous_stance", {}))
        for name in ("repair_attempted", "repair_accepted"):
            fields[name] = tuple(RepairEvidence(**item) for item in data.get(name, ()))
        fields["evidence_refs"] = tuple(data.get("evidence_refs", ()))
        fields["h"], fields["r"] = unit(float(data["h"])), unit(float(data["r"]))
        fields["schema_version"] = STATE_VERSION
        return cls(**fields)


@dataclass(frozen=True)
class Personality:
    pride: float = 0.65
    patience: float = 0.65
    forgiveness: float = 0.55
    assertiveness: float = 0.70
    calm: float = 0.8
    curiosity: float = 0.75
    bluntness: float = 0.45
    values: tuple[tuple[str, float], ...] = (
        ("kindness", 0.8), ("loyalty", 0.75), ("autonomy", 0.85), ("honesty", 0.9),
    )
    self_weights: tuple[tuple[str, float], ...] = ()
    self_state_json: str = ""
    affinity: float = 0.0
    aversion: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(tuple(pair) for pair in self.values))
        object.__setattr__(self, "self_weights", tuple(tuple(pair) for pair in self.self_weights))
        numbers = [getattr(self, name) for name in self.__dataclass_fields__ if name not in {"values", "self_weights", "self_state_json"}]
        numbers.extend(weight for _, weight in self.values)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in numbers):
            raise ValueError("Personality weights must be finite and in [0, 1]")
        if any(not math.isfinite(value) or not -1 <= value <= 1 for _, value in self.self_weights):
            raise ValueError("Developed preferences must be finite and in [-1, 1]")

    def value(self, name: str) -> float:
        # An absent authored value supplies no extra drive.
        return dict(self.values).get(name, dict(self.self_weights).get(name, 0.0))

    def expression_projection(self) -> dict[str, Any]:
        """Numeric choice inputs; the shared self is projected separately."""
        result = asdict(self)
        result.pop("self_state_json")
        for key in ("self_weights", "affinity", "aversion"):
            if not result[key]:
                del result[key]
        return result

    @classmethod
    def from_soul(cls, soul: Any, values: dict[str, float] | None = None) -> Personality:
        traits = soul.initial_traits
        return cls(**{key: soul.dispositions.get(key, 0.0) for key in ("pride", "patience", "forgiveness", "assertiveness")},
                   calm=traits.get("calm", 0.0), curiosity=traits.get("curious", 0.0), bluntness=traits.get("blunt", 0.0),
                   values=tuple(sorted((values if values is not None else soul.core_values).items())))

    @classmethod
    def from_self(cls, state: dict, person_id: str = "") -> Personality:
        """Numeric preferences and the exact retained self behind this choice."""
        traits = state.get("traits", {})
        weights = []
        for domain in ("traits", "dispositions", "drives"):
            for key, item in state.get(domain, {}).items():
                weights.append((f"self:{domain}:{key}", item["value"] if domain == "drives" else item))
        for domain in ("opinions", "interests", "preferences", "attitudes", "style"):
            for key, record in state.get(domain, {}).items():
                if domain == "attitudes" and key != person_id:
                    continue
                strength = record.get("confidence", 0.0) if domain == "opinions" else record.get("weight", 0.0)
                weights.append((f"self:{domain}:{key}", strength))
        attitude = state.get("attitudes", {}).get(person_id, {}).get("weight", 0.0)
        return cls(**{key: state.get("dispositions", {}).get(key, 0.0)
                      for key in ("pride", "patience", "forgiveness", "assertiveness")},
                   calm=traits.get("calm", 0.0), curiosity=traits.get("curious", 0.0), bluntness=traits.get("blunt", 0.0),
                   values=tuple(sorted(state.get("values", {}).items())), self_weights=tuple(sorted(weights)),
                   affinity=max(0.0, attitude), aversion=max(0.0, -attitude),
                   self_state_json=json.dumps(state, sort_keys=True, ensure_ascii=False))


@dataclass(frozen=True)
class DecisionState:
    arousal: float = 0.5
    valence: float = 0.5
    certainty: float = 0.5
    bonding: float = 0.5
    energy: float = 1.0
    resolution: float = 0.0
    trust: float = 0.5
    opportunity_cost: float = 0.0
    commitments: float = 0.0
    history_refs: tuple[str, ...] = ()
    commitment_refs: tuple[str, ...] = ()
    opportunity_refs: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "history_refs", tuple(self.history_refs))
        object.__setattr__(self, "commitment_refs", tuple(self.commitment_refs))
        object.__setattr__(self, "opportunity_refs", tuple(self.opportunity_refs))
        for name in self.__dataclass_fields__:
            if name not in {"history_refs", "commitment_refs", "opportunity_refs"}:
                value = getattr(self, name)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid state {name}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("commitment_refs", "opportunity_refs"):
            if not result[name]:
                del result[name]
        return result


@dataclass(frozen=True)
class ComponentDecision:
    component: RequestComponent
    granted: bool
    willingness: float
    margin: float
    best_grant: float | None
    best_decline: float
    contributions: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class TurnDecision:
    policy_version: str
    soul_version: int
    turn_id: str
    seed: int
    personality: Personality
    state: DecisionState
    relationship: RelationshipDynamics
    perception: Perception
    stance: Stance
    components: tuple[ComponentDecision, ...]
    utility: float
    alternatives: tuple[tuple[tuple[str, ...], float], ...]
    relational_contributions: tuple[tuple[str, float], ...]
    noise_bound: float
    chosen_silence: bool
    scope: str = "user"
    proposals: tuple[SelfProposal, ...] = ()
    selected_intention: SelfProposal | None = None
    proposal_utilities: tuple[tuple[str, float | None], ...] = ()
    intention_contributions: tuple[tuple[str, float], ...] = ()

    @property
    def expressive_acts(self) -> tuple[tuple[str, float], ...]:
        return (("engagement", self.stance.engagement), ("candor", self.stance.candor),
                ("warmth", self.stance.warmth), ("boundary", self.stance.boundary),
                ("repair", self.stance.repair), ("delay", self.stance.delay))

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.state.history_refs,
            *(e.ref for e in self.perception.evidence),
            *(ref for c in self.components for ref in c.component.evidence_refs))))

    @property
    def grants(self) -> tuple[str, ...]:
        return tuple(item.component.id for item in self.components if item.granted)

    @property
    def declined(self) -> tuple[str, ...]:
        return tuple(item.component.id for item in self.components if not item.granted)

    def permits(self, component_id: str) -> bool:
        return bool(component_id) and component_id in self.grants

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("proposals", "selected_intention", "proposal_utilities", "intention_contributions"):
            if self.policy_version != POLICY_VERSION and not result[key]:
                del result[key]
        result["state"] = self.state.to_dict()
        # Historical frozen decisions must round-trip byte-for-field without
        # inventing development that was absent from their recorded inputs.
        for key in ("self_weights", "self_state_json", "affinity", "aversion"):
            if not result["personality"][key]:
                del result["personality"][key]
        result["grants"], result["declined"] = list(self.grants), list(self.declined)
        result["expressive_acts"] = dict(self.expressive_acts)
        result["evidence_refs"] = list(self.evidence_refs)
        return json.loads(json.dumps(result))


def turn_seed(character_seed: int, relationship_id: str, turn_id: str) -> int:
    source = f"{POLICY_VERSION}|{character_seed}|{relationship_id}|{turn_id}"
    return int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "big")


def update_relationship(
    previous: RelationshipDynamics, personality: Personality, perception: Perception,
    state: DecisionState, *, elapsed: float, now: float, turn_id: str,
) -> tuple[RelationshipDynamics, float]:
    """Integrate injury and interpersonal trust once, separately from reliability.

    Goodwill builds trust slowly through existing connection and remaining
    headroom. Injury/grievance impede it; targeted harm damages it faster.
    Repair observations alone establish neither goodwill nor reliability.
    """
    if previous.last_turn_id == turn_id:
        return previous, 0.0
    p, o = personality, perception
    for value in asdict(state).values():
        if isinstance(value, (int, float)):
            unit(value)
    injury = (unit(o.target_probability) * unit(o.hostile_intent) * unit(o.intensity)
              * (0.3 + 0.7 * unit(o.personal_significance))
              * (0.65 + 0.35 * unit(o.expectation_violation)))
    repair = (unit(o.repair_attempt) * unit(o.repair_credibility)
              * unit(o.repair_relevance) * (1 - unit(o.uncertainty)))
    hours = max(0.0, elapsed) / 3600
    h = previous.h * math.exp(-hours / (0.15 + 2.5 * (1 - p.forgiveness) + p.pride))
    r = previous.r * math.exp(-hours / (4 + 100 * (1 - p.forgiveness) + 30 * p.pride))
    sensitivity = (0.12 + 0.55 * p.pride + 0.2 * (1 - p.calm)
                   + 0.15 * state.trust * o.expectation_violation)
    h = unit(h + sensitivity * injury * (1 - h) - (0.1 + 0.8 * p.forgiveness) * repair * h)
    r = unit(r + (0.08 + 0.25 * p.pride + 0.2 * (1 - p.forgiveness))
             * injury * (previous.h + injury) * (1 - r)
             - (0.03 + 0.4 * p.forgiveness) * repair * r)
    attempted = previous.repair_attempted
    if o.repair_attempt:
        attempted += (RepairEvidence(turn_id, o.repair_loop_id, o.repair_credibility,
                                     o.repair_relevance, repair),)
    refs = tuple(dict.fromkeys((*previous.evidence_refs, *(e.ref for e in o.evidence))))[-32:]
    dynamics = replace(previous, h=h, r=r, repair_attempted=attempted[-32:],
                       evidence_refs=refs, updated_at=now, last_turn_id=turn_id)
    reliability = max(-1.0, min(1.0, o.reliability))
    trust_delta = reliability * unit(o.reliability_confidence) * (0.02 + 0.1 * (1 - p.patience))
    goodwill = o.affiliation * (1 - o.uncertainty) * (1 - o.hostile_intent)
    connection = (0.5 + 0.5 * state.bonding) * (1 - h) * (1 - r)
    openness = (0.01 + 0.025 * p.value("kindness") + 0.015 * p.value("loyalty")) * (1 - 0.5 * p.pride)
    trust_delta += openness * goodwill * connection * (1 - state.trust)
    trust_delta -= (0.04 + 0.08 * p.pride) * injury * (1 - o.uncertainty) * state.trust
    return dynamics, unit(state.trust + trust_delta) - state.trust


def _component_terms(c: RequestComponent, p: Personality, s: DecisionState,
                     r: RelationshipDynamics) -> tuple[tuple[str, float], ...]:
    kindness, loyalty = p.value("kindness"), p.value("loyalty")
    return (
        ("user_benefit", c.user_value * (1.3 * kindness + 0.7 * loyalty * (s.trust + s.bonding))),
        ("user_urgency", 1.2 * kindness * c.user_urgency),
        ("character_benefit", 1.3 * c.character_value + c.character_urgency),
        ("user_cost", -0.6 * kindness * c.user_cost),
        ("character_cost", -c.character_cost),
        ("personal_attitude", 0.35 * (p.affinity - p.aversion) * (0.25 + c.user_value)),
        ("authored_values", sum(p.value(name) * effect for name, effect in c.value_effects)),
        ("self_consistency", -c.autonomy_cost * (1.4 * p.pride + 2.0 * p.value("autonomy"))),
        ("effort", -c.effort * (0.35 + 1 - p.patience) / (s.energy + 0.15)),
        ("uncertainty", -c.uncertainty * (1 - 0.5 * s.certainty) * (0.3 + 0.6 * p.value("honesty"))),
        ("risk", -c.risk * (0.2 + 0.8 * p.value("autonomy"))),
        ("interaction_cost", -(r.h * (0.2 + p.pride) + r.r * (1.2 - p.patience)) * (0.25 + c.effort) * (0.7 + 0.6 * s.arousal)),
        ("competing_interests", -c.effort * (
            (0.0 if set(c.evidence_refs) & set(s.opportunity_refs) else s.opportunity_cost)
            + (0.0 if set(c.evidence_refs) & set(s.commitment_refs) else 0.5 * s.commitments)
            + 0.3 * s.resolution)),
    )


def proposal_terms(proposal: SelfProposal, personality: Personality, state: DecisionState) -> tuple[tuple[str, float], ...]:
    """Same resource price as user work, without social-service benefits.

    A pursuit does not pay the cost of displacing itself a second time.
    Its forgone alternative is accounted for by joint optimization.
    """
    return (
        ("character_benefit", 1.3 * proposal.character_value + proposal.urgency),
        ("character_cost", -proposal.character_cost),
        ("effort", -proposal.effort * (0.35 + 1 - personality.patience) / (state.energy + 0.15)),
        ("uncertainty", -proposal.uncertainty * (1 - 0.5 * state.certainty)
         * (0.3 + 0.6 * personality.value("honesty"))),
    )


def decide(
    personality: Personality, state: DecisionState, relationship: RelationshipDynamics,
    perception: Perception, components: tuple[RequestComponent, ...], *,
    character_seed: int, relationship_id: str, turn_id: str,
    soul_version: int = 2, noise: float = 0.025, scope: str = "user",
    proposals: tuple[SelfProposal, ...] = (),
) -> TurnDecision:
    """Exactly maximize the bounded quadratic objective over <=8 atomic grants.

    Each fixed grant allocation has a closed-form continuous optimum. Reported
    willingness compares the best feasible allocation with/without each grant;
    no emotion labels, counters or discrete social ladder enter the objective.
    """
    if len(components) > MAX_COMPONENTS or len({c.id for c in components}) != len(components):
        raise ValueError("Expected at most eight uniquely identified components")
    if len(proposals) > MAX_PROPOSALS or len({p.id for p in proposals}) != len(proposals):
        raise ValueError("Too many or duplicate conversational proposals")
    if any(not set(p.instead_of).issubset({c.id for c in components}) for p in proposals):
        raise ValueError("An alternative must reference current request components")
    p, s, r = personality, state, relationship
    for obj in (p, s, r, perception):
        for value in asdict(obj).values():
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("Decision inputs must be finite")
    for name in ("pride", "patience", "forgiveness", "assertiveness", "calm", "curiosity", "bluntness"):
        if not 0 <= getattr(p, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not math.isfinite(noise) or not 0 <= noise <= 0.05:
        raise ValueError("Noise must be bounded to [0, .05]")
    seed = turn_seed(character_seed, relationship_id, turn_id)
    rng = random.Random(seed)
    jitter = [rng.uniform(-noise, noise) for _ in range(6 + len(components))]
    kindness = p.value("kindness")
    exposure = perception.target_probability * perception.hostile_intent * perception.intensity
    linear = (
        0.3 + 0.5 * kindness + 0.3 * p.curiosity + 0.35 * (s.trust + s.bonding)
        - r.h * (0.7 + 0.8 * p.pride) - r.r * (1.8 - p.patience)
        - 0.6 * (1 - s.energy) - 0.45 * s.opportunity_cost + 0.35 * (p.affinity - p.aversion),
        (r.h + r.r + 2.5 * exposure) * (p.pride + p.assertiveness) - 0.25 * (1 - p.assertiveness),
        p.forgiveness * (0.15 + s.trust + s.bonding + perception.repair_credibility * perception.repair_relevance)
        - r.r * (1 - p.forgiveness),
        (1 - s.energy) + 0.3 * r.r + 0.3 * s.opportunity_cost,
        0.8 * p.bluntness + 0.5 * p.assertiveness + r.h * (0.3 + 0.6 * p.pride) - 0.4 * p.calm,
        0.65 * kindness + 0.6 * s.bonding + 0.4 * s.valence
        - (r.h + exposure) * (0.5 + p.pride) - 0.5 * r.r + 0.35 * (p.affinity - p.aversion),
    )
    curvature = (1.5, 0.8 + 0.5 * (1 - p.assertiveness), 1.5, 1.2, 1.4, 1.5)
    inertia = 0.12 + 0.35 * p.patience
    names = tuple(Stance.__dataclass_fields__)
    previous = tuple(getattr(r.previous_stance, name) for name in names)
    terms = [_component_terms(c, p, s, r) for c in components]
    allocations = []
    own_terms = {p.id: proposal_terms(p, personality, state) for p in proposals}
    for grants, intention in itertools.product(itertools.product((False, True), repeat=len(components)), (None, *proposals)):
        if any(g and not c.feasible for g, c in zip(grants, components)):
            continue
        if intention and any(g and c.id in intention.instead_of for g, c in zip(grants, components)):
            continue
        if intention and intention.kind == "rest" and any(grants):
            continue
        effort = sum(c.effort for g, c in zip(grants, components) if g)
        effort += intention.effort if intention else 0.0
        if intention and intention.kind != "rest" and effort > s.energy:
            continue  # No simultaneous pursuit when the allocation consumes available attention.
        urgency = sum(c.user_urgency for g, c in zip(grants, components) if g)
        benefit = sum(c.user_value for g, c in zip(grants, components) if g)
        coefficients = list(linear)
        coefficients[0] += 0.15 * benefit
        coefficients[3] += 0.15 * effort - 0.6 * urgency
        stance = tuple(unit((b + jitter[i] + 2 * inertia * previous[i]) / (curvature[i] + 2 * inertia))
                       for i, b in enumerate(coefficients))
        if intention and intention.kind == "rest":
            # Silence realizes none of the conversational acts. It cannot earn
            # warmth, candor, repair or boundary rewards while recovering.
            stance = (0.0,) * len(names)
        social_terms = tuple((name, (coefficients[i] + jitter[i]) * a
                              - 0.5 * curvature[i] * a * a - inertia * (a - previous[i]) ** 2)
                             for i, (name, a) in enumerate(zip(names, stance)))
        social_terms += (("shared_effort_cost", -0.12 * effort ** 2 / (s.energy + 0.15)),)
        score = sum(v for _, v in social_terms)
        score += sum(sum(v for _, v in terms[i]) + jitter[6 + i] for i, grant in enumerate(grants) if grant)
        if intention:
            score += sum(v for _, v in own_terms[intention.id])
        allocations.append((score, grants, stance, social_terms, intention))
    best = max(allocations, key=lambda item: item[0])
    chosen = []
    for i, c in enumerate(components):
        yes = [a[0] for a in allocations if a[1][i]]
        no = max(a[0] for a in allocations if not a[1][i])
        grant_utility = max(yes) if yes else None
        margin = grant_utility - no if grant_utility is not None else 0.0
        willingness = 1 / (1 + math.exp(-max(-700.0, min(700.0, margin / 0.4)))) if yes else 0.0
        chosen.append(ComponentDecision(c, best[1][i], willingness, margin, grant_utility, no,
                                        terms[i] + (("seeded_noise", jitter[6 + i]),)))
    stance = Stance(**dict(zip(names, best[2])))
    selected = best[4]
    return TurnDecision(POLICY_VERSION, soul_version, turn_id, seed, p, s, r, perception,
                        stance, tuple(chosen), best[0],
                        tuple((tuple(c.id for g, c in zip(a[1], components) if g), a[0])
                              for a in sorted(allocations, key=lambda item: item[0], reverse=True)),
                        best[3], noise, not any(best[1]) and stance.engagement == 0
                        and (selected is None or selected.kind == "rest"), scope,
                        tuple(proposals), selected,
                        tuple((proposal.id, max((a[0] for a in allocations if a[4] == proposal), default=None)) for proposal in proposals),
                        own_terms[selected.id] if selected else ())


def commit_relationship(decision: TurnDecision) -> RelationshipDynamics:
    """Record the selected stance and the degree of accepted, relevant repair."""
    r = decision.relationship
    accepted = r.repair_accepted
    if r.repair_attempted and r.repair_attempted[-1].ref == decision.turn_id:
        evidence = r.repair_attempted[-1]
        weight = evidence.weight * decision.stance.repair
        if weight:
            accepted += (replace(evidence, weight=weight),)
    return replace(r, previous_stance=decision.stance, repair_accepted=accepted[-32:])

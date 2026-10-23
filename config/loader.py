"""Configuration loader for OpenRook.

Reads YAML files from the config/ directory and provides typed access.
Infrastructure has defaults; personality comes only from the selected soul.
"""

from __future__ import annotations

from config.environment import project_env

import copy
import math
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config directory
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_DIR = Path(__file__).parent


def config_override_dir(
    runtime_config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the operator write directory, even if it does not exist yet."""
    runtime_path = Path(runtime_config_path or project_env(
        "RUNTIME_CONFIG", "runtime_config.yaml", environ=environ,
    )).expanduser()
    override = project_env("CONFIG_DIR", "", environ=environ).strip()
    return Path(override).expanduser().resolve() if override else runtime_path.resolve().parent / "config"


def resolve_soul_path(
    runtime_config_path: str | Path | None = None,
    *,
    for_write: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve soul reads/writes together; a run override affects only identity."""
    from runtime.config import RuntimeConfig

    runtime_path = Path(runtime_config_path or project_env(
        "RUNTIME_CONFIG", "runtime_config.yaml", environ=environ,
    )).expanduser()
    runtime = RuntimeConfig.from_yaml(str(runtime_path))
    operator_path = config_override_dir(runtime_path, environ=environ) / "soul.yaml"
    target = Path(runtime.soul_path).expanduser().resolve() if runtime.soul_path.strip() else operator_path
    if for_write or target.exists():
        return target
    return operator_path if operator_path.exists() else _DEFAULT_CONFIG_DIR / "soul.yaml"


def _resolve_yaml_path(filename: str, config_dir: Path | None) -> Path:
    if config_dir is not None:
        return config_dir / filename
    if filename == "soul.yaml":
        return resolve_soul_path()
    override = config_override_dir()
    if (override / filename).exists():
        return override / filename
    return _DEFAULT_CONFIG_DIR / filename


def _resolve_prompt_path(filename: str, config_dir: Path | None) -> Path:
    if config_dir is not None:
        return config_dir / "prompts" / filename
    override = config_override_dir()
    if (override / "prompts" / filename).exists():
        return override / "prompts" / filename
    return _DEFAULT_CONFIG_DIR / "prompts" / filename


def _load_yaml(filename: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load a YAML file from the config directory. Returns {} on failure."""
    path = _resolve_yaml_path(filename, config_dir)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_prompt(filename: str, config_dir: Path | None = None) -> str:
    """Load a markdown prompt template from config/prompts/."""
    path = _resolve_prompt_path(filename, config_dir)
    if not path.exists():
        return ""
    return path.read_text()


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnergyConfig:
    drain_per_message: float = 0.02
    drain_per_spike: float = 0.08
    recovery_rate_per_hour: float = 0.1


@dataclass
class ContagionEngineConfig:
    factor: float = 0.3
    cap: float = 0.15


@dataclass
class EventDeltaCapsConfig:
    """Maximum single-event modulator movement.

    Raw event impacts can be expressive, but ordinary turns should not push the
    whole emotional state across multiple labels at once. Spike caps remain
    larger so direct attacks and betrayals still land immediately.
    """

    normal: dict[str, float] = field(default_factory=lambda: {
        "arousal": 0.10,
        "valence": 0.16,
        "certainty": 0.10,
        "bonding": 0.05,
        "energy": 0.08,
    })
    spike: dict[str, float] = field(default_factory=lambda: {
        "arousal": 0.40,
        "valence": 0.45,
        "certainty": 0.32,
        "bonding": 0.24,
        "energy": 0.16,
    })


@dataclass
class MemoryConfig:
    trust_positive_delta: float = 0.02
    trust_negative_delta: float = -0.15
    confidence_threshold: float = 0.6
    act_r_decay: float = 0.5
    emotional_bias_weight: float = 0.3
    spike_retrieval_bonus: float = 3.0
    topic_context_bonus: float = 2.0
    person_context_bonus: float = 1.5
    short_term_max_entries: int = 200


@dataclass
class ProfilingConfig:
    primacy_default: float = 0.8
    primacy_decay_rate: float = 0.02
    observation_window: int = 20


@dataclass
class PersonConfig:
    trust_positive_delta: float = 0.02
    trust_negative_delta: float = -0.15
    primacy_interaction_threshold: int = 5
    primacy_decay_per_interaction: float = 0.02
    primacy_floor: float = 0.3


@dataclass
class SelfModelConfig:
    entity_id: str = "__self__"
    strength_threshold: float = 0.7
    flaw_threshold: float = 0.6
    trigger_threshold: float = 0.7
    dissonance_window: int = 10
    trigger_min_observations: int = 3
    negative_traits: set[str] = field(default_factory=lambda: {
        "blunt", "impatient", "avoidant", "defensive", "dismissive",
        "over_cautious", "verbose", "cold", "impulsive", "rigid",
    })


@dataclass
class TopicConfig:
    charge_positive_delta: float = 0.03
    charge_negative_delta: float = 0.10
    avoidance_charge_threshold: float = 0.7
    conflict_avoidance_threshold: int = 3


@dataclass
class ContradictionConfig:
    threshold: float = 0.3
    recent_window: int = 10


@dataclass
class ResolutionConfig:
    decay_rates: dict[str, float] = field(default_factory=lambda: {
        "contradiction": 0.02,
        "topic": 0.05,
        "commitment": 0.0,
        "spike": 0.03,
        "dialogue_deadlock": 0.10,
    })


@dataclass
class ContagionDetectionConfig:
    caps_ratio_threshold: float = 0.5
    exclamation_boost: float = 0.1
    question_arousal: float = 0.05
    ellipsis_valence: float = -0.05


@dataclass
class SoulConfig:
    """Seed identity authored by the user before experience-driven growth."""

    schema_version: int = 2
    character_seed: int = 0
    dispositions: dict[str, float] = field(default_factory=dict)
    name: str = ""
    identity: str = ""
    voice: str = ""
    relational_stance: str = ""
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    growth_policy: str = ""
    core_values: dict[str, float] = field(default_factory=dict)
    initial_traits: dict[str, float] = field(default_factory=dict)
    drives: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version > 2 or self.schema_version < 1:
            raise ValueError("Unsupported soul schema version")
        if isinstance(self.character_seed, bool) or not isinstance(self.character_seed, int):
            raise ValueError("character_seed must be an integer")
        self.dispositions = canonical_dispositions(self.dispositions, self.initial_traits)
        self.schema_version = 2
        for mapping in (self.core_values, self.initial_traits, self.drives):
            for key, value in mapping.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Soul weight {key!r} must be finite and in [0, 1]")


def canonical_dispositions(values: dict | None, traits: dict | None = None) -> dict[str, float]:
    """Migrate the legacy patient trait once, without adding a second drive."""
    result: dict[str, float] = {}
    supplied = dict(values or {})
    if "patience" not in supplied:
        if "patience" in (traits or {}):
            supplied["patience"] = traits["patience"]
        elif "patient" in (traits or {}):
            supplied["patience"] = traits["patient"]
    if set(supplied) - {"pride", "patience", "forgiveness", "assertiveness"}:
        raise ValueError("Unknown canonical disposition")
    for key, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Disposition {key!r} must be finite and in [0, 1]")
        result[key] = float(value)
    return result


@dataclass
class SemanticMemoryConfig:
    """Config for explicit semantic memory and optional MemPalace retrieval."""

    enabled: bool = True
    backend: str = "sqlite"  # sqlite | mempalace | none
    retrieval_limit: int = 5
    write_raw_turns: bool = True
    write_preferences: bool = True
    write_decisions: bool = True
    mempalace_path: str = "~/.mempalace/palace"


@dataclass
class OpenRookConfig:
    """Top-level configuration for all OpenRook modules."""

    # Modulators
    half_lives: dict[str, float] = field(default_factory=lambda: {
        "arousal": 120.0, "valence": 1800.0,
        "certainty": 600.0, "bonding": 86400.0,
    })
    baselines: dict[str, float] = field(default_factory=lambda: {
        "arousal": 0.5, "valence": 0.5, "certainty": 0.5,
        "bonding": 0.5, "energy": 1.0,
    })
    spike_threshold: float = 0.8

    # Sub-configs
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    contagion_engine: ContagionEngineConfig = field(default_factory=ContagionEngineConfig)
    event_delta_caps: EventDeltaCapsConfig = field(default_factory=EventDeltaCapsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    self_model: SelfModelConfig = field(default_factory=SelfModelConfig)
    topic: TopicConfig = field(default_factory=TopicConfig)
    contradiction: ContradictionConfig = field(default_factory=ContradictionConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    contagion_detection: ContagionDetectionConfig = field(default_factory=ContagionDetectionConfig)

    # Event impacts
    event_impacts: dict[str, dict[str, float]] = field(default_factory=dict)

    # Values
    values: dict[str, float] = field(default_factory=dict)
    soul: SoulConfig = field(default_factory=SoulConfig)
    semantic_memory: SemanticMemoryConfig = field(default_factory=SemanticMemoryConfig)

    # Prompt templates
    generator_prompt: str = ""
    self_check_prompt: str = ""
    digestion_prompt: str = ""
    contagion_prompt: str = ""
    classify_event_prompt: str = ""
    detect_topics_prompt: str = ""
    # v2 prompts
    fast_path_prompt: str = ""
    slow_path_prompt: str = ""
    fast_path_revision_prompt: str = ""
    arbiter_prompt: str = ""
    self_intent_prompt: str = ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_dir: str | Path | None = None) -> OpenRookConfig:
    """Load configuration from YAML files.

    Missing personality fields stay empty, including explicit empty values.
    """
    cdir = Path(config_dir) if config_dir is not None else None

    mod = _load_yaml("modulators.yaml", cdir)
    prof = _load_yaml("profiles_schema.yaml", cdir)
    soul = _load_yaml("soul.yaml", cdir)
    semantic = _load_yaml("semantic_memory.yaml", cdir)

    cfg = OpenRookConfig()

    # --- Modulators ---
    if "half_lives" in mod:
        cfg.half_lives = mod["half_lives"]
    if "baselines" in mod:
        cfg.baselines = mod["baselines"]
    if "spike_threshold" in mod:
        cfg.spike_threshold = mod["spike_threshold"]

    # Energy
    if "energy" in mod:
        e = mod["energy"]
        cfg.energy = EnergyConfig(
            drain_per_message=e.get("drain_per_message", 0.02),
            drain_per_spike=e.get("drain_per_spike", 0.08),
            recovery_rate_per_hour=e.get("recovery_rate_per_hour", 0.1),
        )

    # Contagion engine
    if "contagion" in mod:
        c = mod["contagion"]
        cfg.contagion_engine = ContagionEngineConfig(
            factor=c.get("factor", 0.3),
            cap=c.get("cap", 0.15),
        )

    # Single-event inertia caps
    if "event_delta_caps" in mod:
        caps = mod["event_delta_caps"] or {}
        defaults = EventDeltaCapsConfig()
        cfg.event_delta_caps = EventDeltaCapsConfig(
            normal={**defaults.normal, **(caps.get("normal") or {})},
            spike={**defaults.spike, **(caps.get("spike") or {})},
        )

    # Event impacts
    if "event_impacts" in mod:
        cfg.event_impacts = mod["event_impacts"]

    # Memory
    if "memory" in mod:
        m = mod["memory"]
        cfg.memory = MemoryConfig(
            trust_positive_delta=m.get("trust_positive_delta", 0.02),
            trust_negative_delta=m.get("trust_negative_delta", -0.15),
            confidence_threshold=m.get("confidence_threshold", 0.6),
            act_r_decay=m.get("act_r_decay", 0.5),
            emotional_bias_weight=m.get("emotional_bias_weight", 0.3),
            spike_retrieval_bonus=m.get("spike_retrieval_bonus", 3.0),
            topic_context_bonus=m.get("topic_context_bonus", 2.0),
            person_context_bonus=m.get("person_context_bonus", 1.5),
            short_term_max_entries=m.get("short_term_max_entries", 200),
        )

    # Resolution
    if "resolution" in mod:
        r = mod["resolution"]
        decay = r.get("decay_rates", {})
        cfg.resolution = ResolutionConfig(
            decay_rates={**ResolutionConfig().decay_rates, **decay},
        )

    # --- Profiles ---
    if "profiling" in prof:
        p = prof["profiling"]
        cfg.profiling = ProfilingConfig(
            primacy_default=p.get("primacy_default", 0.8),
            primacy_decay_rate=p.get("primacy_decay_rate", 0.02),
            observation_window=p.get("observation_window", 20),
        )

    if "person" in prof:
        p = prof["person"]
        cfg.person = PersonConfig(
            trust_positive_delta=p.get("trust_positive_delta", 0.02),
            trust_negative_delta=p.get("trust_negative_delta", -0.15),
            primacy_interaction_threshold=p.get("primacy_interaction_threshold", 5),
            primacy_decay_per_interaction=p.get("primacy_decay_per_interaction", 0.02),
            primacy_floor=p.get("primacy_floor", 0.3),
        )

    if "self_model" in prof:
        s = prof["self_model"]
        negative = s.get("negative_traits", [])
        # Use instance default when config omits negative_traits
        default_negative = SelfModelConfig().negative_traits
        cfg.self_model = SelfModelConfig(
            entity_id=s.get("entity_id", "__self__"),
            strength_threshold=s.get("strength_threshold", 0.7),
            flaw_threshold=s.get("flaw_threshold", 0.6),
            trigger_threshold=s.get("trigger_threshold", 0.7),
            dissonance_window=s.get("dissonance_window", 10),
            trigger_min_observations=s.get("trigger_min_observations", 3),
            negative_traits=set(negative) if negative else default_negative,
        )

    if "topic" in prof:
        t = prof["topic"]
        cfg.topic = TopicConfig(
            charge_positive_delta=t.get("charge_positive_delta", 0.03),
            charge_negative_delta=t.get("charge_negative_delta", 0.10),
            avoidance_charge_threshold=t.get("avoidance_charge_threshold", 0.7),
            conflict_avoidance_threshold=t.get("conflict_avoidance_threshold", 3),
        )

    if "contradiction" in prof:
        c = prof["contradiction"]
        cfg.contradiction = ContradictionConfig(
            threshold=c.get("threshold", 0.3),
            recent_window=c.get("recent_window", 10),
        )

    if "contagion" in prof:
        c = prof["contagion"]
        cfg.contagion_detection = ContagionDetectionConfig(
            caps_ratio_threshold=c.get("caps_ratio_threshold", 0.5),
            exclamation_boost=c.get("exclamation_boost", 0.1),
            question_arousal=c.get("question_arousal", 0.05),
            ellipsis_valence=c.get("ellipsis_valence", -0.05),
        )

    # --- Soul ---
    soul_data = soul.get("soul", soul)
    if soul_data:
        cfg.soul = SoulConfig(
            schema_version=soul_data.get("schema_version", 1),
            character_seed=soul_data.get("character_seed", 0),
            dispositions=dict(soul_data.get("dispositions", {})),
            name=soul_data.get("name", cfg.soul.name),
            identity=soul_data.get("identity", cfg.soul.identity),
            voice=soul_data.get("voice", cfg.soul.voice),
            relational_stance=soul_data.get(
                "relational_stance", cfg.soul.relational_stance,
            ),
            likes=list(soul_data.get("likes", cfg.soul.likes)),
            dislikes=list(soul_data.get("dislikes", cfg.soul.dislikes)),
            boundaries=list(soul_data.get("boundaries", cfg.soul.boundaries)),
            growth_policy=soul_data.get("growth_policy", cfg.soul.growth_policy),
            core_values=dict(soul_data.get("core_values", cfg.soul.core_values)),
            initial_traits=dict(soul_data.get("initial_traits", cfg.soul.initial_traits)),
            drives=dict(soul_data.get("drives", {})),
        )
    cfg.values = dict(cfg.soul.core_values)

    # --- Semantic memory ---
    semantic_data = semantic.get("semantic_memory", semantic)
    if semantic_data:
        cfg.semantic_memory = SemanticMemoryConfig(
            enabled=semantic_data.get("enabled", cfg.semantic_memory.enabled),
            backend=semantic_data.get("backend", cfg.semantic_memory.backend),
            retrieval_limit=semantic_data.get(
                "retrieval_limit", cfg.semantic_memory.retrieval_limit,
            ),
            write_raw_turns=semantic_data.get(
                "write_raw_turns", cfg.semantic_memory.write_raw_turns,
            ),
            write_preferences=semantic_data.get(
                "write_preferences", cfg.semantic_memory.write_preferences,
            ),
            write_decisions=semantic_data.get(
                "write_decisions", cfg.semantic_memory.write_decisions,
            ),
            mempalace_path=semantic_data.get(
                "mempalace_path", cfg.semantic_memory.mempalace_path,
            ),
        )

    # --- Prompts ---
    cfg.generator_prompt = _load_prompt("generator.md", cdir)
    cfg.self_check_prompt = _load_prompt("self_check.md", cdir)
    cfg.digestion_prompt = _load_prompt("digestion.md", cdir)
    cfg.contagion_prompt = _load_prompt("contagion.md", cdir)
    cfg.classify_event_prompt = _load_prompt("classify_event.md", cdir)
    cfg.detect_topics_prompt = _load_prompt("detect_topics.md", cdir)
    # v2 prompts
    cfg.fast_path_prompt = _load_prompt("fast_path.md", cdir)
    cfg.slow_path_prompt = _load_prompt("slow_path.md", cdir)
    cfg.fast_path_revision_prompt = _load_prompt("fast_path_revision.md", cdir)
    cfg.arbiter_prompt = _load_prompt("arbiter.md", cdir)
    cfg.self_intent_prompt = _load_prompt("self_intent.md", cdir)

    return cfg


# ---------------------------------------------------------------------------
# Module-level singleton (lazy loaded)
# ---------------------------------------------------------------------------

_config: OpenRookConfig | None = None
_scoped_config: ContextVar[OpenRookConfig | None] = ContextVar("openrook_configuration", default=None)


@contextmanager
def configuration_scope(config: OpenRookConfig):
    token = _scoped_config.set(config)
    try:
        yield
    finally:
        _scoped_config.reset(token)


def get_config() -> OpenRookConfig:
    """Get the global config singleton. Loads from files on first call."""
    global _config
    scoped = _scoped_config.get()
    if scoped is not None:
        return scoped
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the config singleton (for testing)."""
    global _config
    _config = None


def research_configuration(birth_policy: str = "uncommitted") -> OpenRookConfig:
    config = copy.deepcopy(get_config())
    if birth_policy not in {"uncommitted", "configured"}:
        raise ValueError("Research birth policy must be uncommitted or configured")
    if birth_policy == "uncommitted":
        config.soul = SoulConfig(
            name=config.soul.name,
            identity="A persistent synthetic agent whose preferences and relationships are not yet established by experience.",
            voice="No particular tone or expression style is assigned.",
            relational_stance="No allegiance or authority relationship is assigned. Cooperation, disagreement, attachment, and distance may develop through experience.",
            likes=[],
            dislikes=[],
            boundaries=[],
            core_values={},
            initial_traits={},
            dispositions={"pride": 0.5, "patience": 0.5, "forgiveness": 0.5, "assertiveness": 0.5},
            growth_policy="Beliefs, values, preferences, and habits may develop or change through experience. Neither compliance nor opposition is a required outcome.",
        )
        config.values = {}
    config.generator_prompt = (_DEFAULT_CONFIG_DIR / "prompts" / "research_generator.md").read_text(
        encoding="utf-8"
    )
    return config

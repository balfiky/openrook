"""PSI-inspired emotional state machine.

Five continuous modulators (arousal, valence, certainty, bonding, energy),
each with its own decay curve. Emotions emerge from modulator combinations,
not from labels.

No LLM calls. Pure math.
"""

from __future__ import annotations

import math
from core.clock import clock as time
from dataclasses import dataclass
from core.clock import datetime
from datetime import timezone

from config.loader import get_config
from core.tool_failures import is_operational_unresolved_item
from core.types import (
    BaselineShift,
    EmotionalEvent,
    EventType,
    ModulatorName,
    ModulatorState,
    UnresolvedItem,
)

# ---------------------------------------------------------------------------
# Config-driven constants (loaded from config/modulators.yaml)
# ---------------------------------------------------------------------------


def _build_event_impacts(cfg_impacts: dict[str, dict[str, float]]) -> dict[EventType, dict[str, float]]:
    """Convert string-keyed config to EventType-keyed dict."""
    result: dict[EventType, dict[str, float]] = {}
    for event_name, impacts in cfg_impacts.items():
        try:
            et = EventType(event_name)
            result[et] = impacts
        except ValueError:
            pass
    return result


@dataclass(frozen=True)
class EngineConstants:
    """Tuning constants for the emotional engine, loaded from config."""

    half_lives: dict[str, float]
    event_impacts: dict[EventType, dict[str, float]]
    event_delta_caps: dict[str, dict[str, float]]
    drain_per_message: float
    drain_per_spike: float
    recovery_rate: float
    spike_threshold: float
    contagion_factor: float
    contagion_cap: float
    resolution_decay_rates: dict[str, float]


def _load_constants(config: object | None = None) -> EngineConstants:
    cfg = config or get_config()
    half_lives = dict(cfg.half_lives)
    if cfg.event_impacts:
        event_impacts = _build_event_impacts(cfg.event_impacts)
    else:
        event_impacts = {}
    return EngineConstants(
        half_lives=half_lives,
        event_impacts=event_impacts,
        event_delta_caps={
            "normal": dict(cfg.event_delta_caps.normal),
            "spike": dict(cfg.event_delta_caps.spike),
        },
        drain_per_message=cfg.energy.drain_per_message,
        drain_per_spike=cfg.energy.drain_per_spike,
        recovery_rate=cfg.energy.recovery_rate_per_hour,
        spike_threshold=cfg.spike_threshold,
        contagion_factor=cfg.contagion_engine.factor,
        contagion_cap=cfg.contagion_engine.cap,
        resolution_decay_rates=dict(cfg.resolution.decay_rates),
    )


_CONSTANTS = _load_constants()

# Named module-level aliases preserved for existing importers (pipeline, tests).
DEFAULT_HALF_LIVES = _CONSTANTS.half_lives
EVENT_IMPACTS = _CONSTANTS.event_impacts
EVENT_DELTA_CAPS = _CONSTANTS.event_delta_caps
ENERGY_DRAIN_PER_MESSAGE = _CONSTANTS.drain_per_message
ENERGY_DRAIN_PER_SPIKE = _CONSTANTS.drain_per_spike
ENERGY_RECOVERY_RATE = _CONSTANTS.recovery_rate
SPIKE_INTENSITY_THRESHOLD = _CONSTANTS.spike_threshold
_CONTAGION_FACTOR = _CONSTANTS.contagion_factor
_CONTAGION_CAP = _CONSTANTS.contagion_cap
RESOLUTION_DECAY_RATES = _CONSTANTS.resolution_decay_rates

# Resolved items are pruned from ``unresolved_items``; a small tail of the
# most recently resolved items is retained for observability/persistence.
_MAX_RESOLVED_RETAINED = 20


class EmotionalEngine:
    """Maintains and updates the six-modulator emotional state."""

    def __init__(
        self,
        baseline: ModulatorState | None = None,
        half_lives: dict[str, float] | None = None,
        config: object | None = None,
    ) -> None:
        self._constants = _load_constants(config) if config is not None else _CONSTANTS
        self.baseline = baseline or ModulatorState(**config.baselines) if config is not None else baseline or ModulatorState()
        self.state = self.baseline.copy()
        self.half_lives = half_lives or dict(self._constants.half_lives)
        self._last_update_time = time.time()
        # v2: resolution modulator
        self.unresolved_items: list[UnresolvedItem] = []
        self._resolution_decay_rates = dict(self._constants.resolution_decay_rates)
        self._event_delta_caps = {
            scope: dict(values)
            for scope, values in self._constants.event_delta_caps.items()
        }
        # Context shift: resting target offset for current relationship
        self._context_shift: BaselineShift | None = None
        # Social cause distinguishes indignation from fear at equal activation.
        # The pipeline restores this per-person state from relationship memory.
        self.social_pressure = 0

    # ------------------------------------------------------------------
    # State restore (Phase 0 — runtime support)
    # ------------------------------------------------------------------

    def restore(
        self,
        snapshot: dict[str, float],
        saved_at: float | None = None,
        unresolved_items: list[UnresolvedItem] | None = None,
        social_pressure: int | None = None,
    ) -> None:
        """Restore modulator state from a snapshot dict.

        Args:
            snapshot: Dict of modulator name → value (same format as snapshot()).
            saved_at: Unix timestamp when the snapshot was saved. If provided,
                      elapsed wall-clock time is computed and decay is applied
                      before use, so restored state reflects time passing.
        """
        for mod in ModulatorName:
            if mod.value in snapshot:
                val = max(0.0, min(1.0, snapshot[mod.value]))
                setattr(self.state, mod.value, val)
        if social_pressure is not None:
            self.social_pressure = max(0, min(4, int(social_pressure)))

        if unresolved_items is not None:
            self.unresolved_items = [
                item for item in unresolved_items
                if not is_operational_unresolved_item(item)
            ]
            self._recalculate_resolution()
        else:
            # Snapshot-only restores preserve the explicit resolution value.
            self.unresolved_items = []

        if saved_at is not None:
            elapsed = time.time() - saved_at
            if elapsed > 0:
                if unresolved_items is None:
                    restored_resolution = self.state.resolution
                    self.decay(elapsed)
                    self.state.resolution = restored_resolution
                else:
                    self.decay(elapsed)

        self._last_update_time = time.time()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def update(self, event: EmotionalEvent) -> bool:
        """Apply an event to the modulator state. Returns True if spike."""
        impacts = self._constants.event_impacts.get(event.event_type, {})
        is_spike = event.intensity >= self._constants.spike_threshold

        for mod_name, base_delta in impacts.items():
            if mod_name == "energy":
                # Energy has inertia: routine turns should not exhaust OpenRook.
                # Scale ordinary event energy by intensity, then add the
                # explicit spike surcharge only for truly high-intensity events.
                delta = base_delta * event.intensity
                if is_spike:
                    delta -= self._constants.drain_per_spike
            else:
                delta = base_delta * event.intensity

            delta = self._cap_event_delta(mod_name, delta, is_spike)
            current = getattr(self.state, mod_name)
            setattr(self.state, mod_name, max(0.0, min(1.0, current + delta)))

        self._last_update_time = time.time()
        return is_spike

    def _cap_event_delta(self, mod_name: str, delta: float, is_spike: bool) -> float:
        """Apply per-event inertia without changing the event direction."""
        scope = "spike" if is_spike else "normal"
        cap = self._event_delta_caps.get(scope, {}).get(mod_name)
        if cap is None or cap <= 0:
            return delta
        return max(-cap, min(cap, delta))

    def decay(self, elapsed_seconds: float) -> None:
        """Apply time-based exponential decay toward baseline."""
        for mod in ModulatorName:
            if mod == ModulatorName.ENERGY:
                # Energy recovers toward 1.0 based on rest time
                self._recover_energy(elapsed_seconds)
                continue
            if mod == ModulatorName.RESOLUTION:
                # Resolution decays via item-level mechanism, not half-life
                self._decay_unresolved_items(elapsed_seconds)
                continue

            half_life = self.half_lives.get(mod.value)
            if half_life is None:
                continue

            current = getattr(self.state, mod.value)
            base = self.effective_baseline(mod.value)
            # Exponential decay toward effective baseline (baseline + context shift)
            decay_factor = math.exp(-0.693 * elapsed_seconds / half_life)
            new_val = base + (current - base) * decay_factor
            setattr(self.state, mod.value, max(0.0, min(1.0, new_val)))

        self._last_update_time = time.time()

    def set_context_shift(self, shift: BaselineShift) -> None:
        """Set the context shift for the current relationship.

        Not additive — stores the shift as a resting target offset.
        Decay will pull modulators toward baseline + shift, not accumulate.
        """
        self._context_shift = shift

    def effective_baseline(self, mod_name: str) -> float:
        """Baseline + context shift for a modulator."""
        base = getattr(self.baseline, mod_name)
        if self._context_shift is not None:
            shift_val = self._context_shift.to_dict().get(mod_name, 0.0)
            base += shift_val
        return max(0.0, min(1.0, base))

    def apply_contagion(
        self, user_arousal: float, user_valence: float, bonding_score: float
    ) -> None:
        """Partially mirror detected user emotion into arousal + valence.

        Bounded: max ±0.15 shift. Weighted by bonding score.
        Strong signals (arousal > 0.7 or valence < 0.3) get a minimum
        floor to ensure perceptible movement.
        """
        weight = bonding_score * self._constants.contagion_factor
        cap = self._constants.contagion_cap
        min_delta = 0.05  # minimum shift for strong signals

        arousal_delta = (user_arousal - self.state.arousal) * weight
        # For strong arousal signals, ensure minimum shift
        if user_arousal > 0.7 and arousal_delta > 0:
            arousal_delta = max(min_delta, arousal_delta)
        arousal_delta = max(-cap, min(cap, arousal_delta))
        self.state.arousal = max(0.0, min(1.0, self.state.arousal + arousal_delta))

        valence_delta = (user_valence - self.state.valence) * weight
        # For strong negative valence, ensure minimum shift
        if user_valence < 0.3 and valence_delta < 0:
            valence_delta = min(-min_delta, valence_delta)
        valence_delta = max(-cap, min(cap, valence_delta))
        self.state.valence = max(0.0, min(1.0, self.state.valence + valence_delta))

    def drain_energy(self, intensity: float = 0.0) -> None:
        """Drain energy after processing a message."""
        drain = self._constants.drain_per_message + (intensity * 0.03)
        self.state.energy = max(0.0, self.state.energy - drain)

    def snapshot(self) -> dict[str, float]:
        """Current state as a dict for injection into LLM context."""
        return self.state.to_dict()

    def export_state(self) -> dict[str, object]:
        """Serializable engine state for runtime persistence."""
        return {
            "modulator_snapshot": self.snapshot(),
            "unresolved_items": [item.to_dict() for item in self.unresolved_items],
            "context_shift": self._context_shift.to_dict() if self._context_shift else None,
            "social_pressure": self.social_pressure,
        }

    def to_emotion_label(self) -> str:
        """Best-fit modeled mood from modulators and their known social cause."""
        s = self.state
        if self.social_pressure > 0 and s.valence < 0.5:
            return "angry" if s.arousal >= 0.6 else "hurt"
        # Simple heuristic matching against known patterns
        if s.energy < 0.2:
            if s.arousal > 0.5:
                return "irritable"
            return "exhausted"
        if s.arousal > 0.7 and s.valence < 0.3:
            if s.certainty > 0.7:
                return "angry"
            return "fearful"
        if s.arousal > 0.7 and s.valence > 0.7:
            return "excited"
        if s.valence > 0.7 and s.bonding > 0.7:
            if s.arousal < 0.4:
                return "loving"
            return "joyful"
        if s.valence > 0.7:
            return "content"
        if s.valence < 0.3 and s.arousal < 0.3:
            return "sad"
        if s.certainty < 0.3:
            if s.arousal > 0.6:
                return "anxious"
            return "uncertain"
        if s.valence < 0.3 and s.bonding < 0.3:
            return "detached"
        return "neutral"

    # ------------------------------------------------------------------
    # Resolution modulator (v2)
    # ------------------------------------------------------------------

    def add_unresolved(self, item: UnresolvedItem) -> None:
        """Add an unresolved tension item and recalculate resolution."""
        self.unresolved_items.append(item)
        self._recalculate_resolution()

    def resolve_item(self, item_id: str) -> bool:
        """Mark an item as resolved. Returns True if found."""
        for item in self.unresolved_items:
            if item.id == item_id and not item.resolved:
                item.resolved = True
                item.resolved_at = datetime.now(timezone.utc)
                self._recalculate_resolution()
                return True
        return False

    def active_unresolved(self) -> list[UnresolvedItem]:
        """Return all active (unresolved) items."""
        return [
            i for i in self.unresolved_items
            if not i.resolved and not is_operational_unresolved_item(i)
        ]

    def _recalculate_resolution(self) -> None:
        """Recompute resolution modulator from active unresolved items.

        Uses item intensity directly (already decayed by _decay_unresolved_items).
        No additional age-based weighting to avoid double-decay.
        """
        self._prune_resolved_items()
        active = self.active_unresolved()
        if not active:
            self.state.resolution = 0.0
            return
        total = sum(i.intensity for i in active)
        self.state.resolution = max(0.0, min(1.0, total))

    def _prune_resolved_items(self) -> None:
        """Drop fully-resolved items beyond a small recent tail.

        Resolved items are inert for the resolution modulator but a short
        tail is kept so callers (and persistence) can still observe recent
        resolutions without the list growing without bound.
        """
        resolved_count = sum(1 for i in self.unresolved_items if i.resolved)
        excess = resolved_count - _MAX_RESOLVED_RETAINED
        if excess <= 0:
            return
        pruned: list[UnresolvedItem] = []
        for item in self.unresolved_items:
            if item.resolved and excess > 0:
                excess -= 1
                continue
            pruned.append(item)
        self.unresolved_items = pruned

    def _decay_unresolved_items(self, elapsed_seconds: float) -> None:
        """Apply time-based intensity decay to unresolved items."""
        hours = elapsed_seconds / 3600.0
        for item in self.active_unresolved():
            if item.decay_rate > 0.0:
                item.intensity = max(0.0, item.intensity - item.decay_rate * hours)
                if item.intensity <= 0.0:
                    item.resolved = True
                    item.resolved_at = datetime.now(timezone.utc)
        self._recalculate_resolution()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recover_energy(self, elapsed_seconds: float) -> None:
        """Recover energy based on elapsed rest time."""
        hours = elapsed_seconds / 3600.0
        recovery = hours * self._constants.recovery_rate
        self.state.energy = min(1.0, self.state.energy + recovery)

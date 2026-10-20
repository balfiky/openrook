"""Explanatory and data-driven showcase visuals for the README.

The how-it-works explainer uses illustrative dialogue and state changes, not a
recorded run. Unlike ``build_diagrams`` (structural block diagrams), the curves,
petals and dots here are computed from real sources:

- ``emotional_arc`` and ``emotional_fingerprints`` evaluate an authored
  rupture → repair event sequence with the deterministic ``EmotionalEngine``
  pinned to the shipped ``config/modulators.yaml``. Event types, intensities and
  tension add/resolve are authored inputs (the pipeline's classifier and
  contagion are bypassed); dialogue is illustrative and no LLM is involved.
  Emotion words come from ``EmotionalEngine.to_emotion_label``, a logging
  heuristic.
- ``retained_choice`` plots the 14 paired results of the completed v4
  retained-choice confirmation (``RETAINED_CHOICE_PAIRS`` below, copied from
  the study's ``analysis.json``).

Regenerate with ``python3 nur_tools/render_diagram_pngs.py`` (requires
``cairosvg``). Light and dark SVGs blend into the host page. Optional one-shot
reveals honor ``prefers-reduced-motion``; PNG fallbacks show the complete chart.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from xml.sax.saxutils import escape

from config.loader import load_config
from core.emotional_engine import EmotionalEngine
from core.types import EmotionalEvent, EventType, UnresolvedItem
from nur_tools.brand import (
    LIGHT, SANS, SHOWCASE_SERIF as SERIF, THEMES, Theme, rook_defs, rook_mark,
)

SHIPPED_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"



# ---------------------------------------------------------------------------
# Engine replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Beat:
    minute: float
    event: EventType
    intensity: float
    caption: str
    quote: str = ""


SCRIPT: tuple[Beat, ...] = (
    Beat(3, EventType.WARMTH, 0.6, "warmth"),
    Beat(8, EventType.POSITIVE_FEEDBACK, 0.75, "praise", "That really helped."),
    Beat(14, EventType.WARMTH, 0.6, "warmth"),
    Beat(24, EventType.CONFLICT, 0.95, "rupture", "You completely misunderstood me."),
    Beat(27, EventType.NEGATIVE_FEEDBACK, 0.6, "pushback"),
    Beat(42, EventType.SILENCE, 0.6, "silence"),
    Beat(60, EventType.RESOLUTION, 0.85, "repair", "I think I was too harsh earlier."),
    Beat(66, EventType.WARMTH, 0.7, "warmth"),
    Beat(76, EventType.POSITIVE_FEEDBACK, 0.6, "praise", "This feels easier now."),
)
DURATION_MIN = 90
STEP_SECONDS = 15


def shipped_engine() -> EmotionalEngine:
    """Engine pinned to the repository's shipped config, ignoring user overrides."""
    return EmotionalEngine(config=load_config(SHIPPED_CONFIG_DIR))


def replay() -> tuple[list[float], list[dict[str, float]], list[str], dict[str, dict[str, float]]]:
    """Evaluate the authored SCRIPT with the real engine.

    Event types, intensities and the tension add/resolve are authored inputs;
    the pipeline's classifier, contagion and context adjustments are bypassed.
    Returns sample minutes, modulator snapshots, emotion labels, and the
    snapshots right after named moments used by the fingerprint glyph.
    """
    engine = shipped_engine()
    beats = sorted(SCRIPT, key=lambda b: b.minute)
    minutes: list[float] = []
    states: list[dict[str, float]] = []
    labels: list[str] = []
    moments: dict[str, dict[str, float]] = {"calm": engine.snapshot()}
    rupture_item: UnresolvedItem | None = None
    next_beat = 0
    seconds = 0
    while seconds <= DURATION_MIN * 60:
        minute = seconds / 60
        while next_beat < len(beats) and beats[next_beat].minute <= minute:
            beat = beats[next_beat]
            engine.update(EmotionalEvent(beat.event, beat.intensity))
            if beat.event is not EventType.SILENCE:
                engine.drain_energy(beat.intensity)  # per processed message
            if beat.event is EventType.CONFLICT:
                # Spike tension at 0.8 x intensity, as the pipeline records it.
                rupture_item = UnresolvedItem(
                    id="rupture", source="spike", description="rupture",
                    created_at=datetime.now(timezone.utc),
                    intensity=round(0.8 * beat.intensity, 2), decay_rate=0.03,
                )
                engine.add_unresolved(rupture_item)
            if beat.event is EventType.RESOLUTION and rupture_item is not None:
                engine.resolve_item(rupture_item.id)
            if beat.event is EventType.NEGATIVE_FEEDBACK:
                moments["rupture"] = engine.snapshot()
            if beat.minute == 8:
                moments["warmth"] = engine.snapshot()
            if beat.minute == 66:
                moments["repair"] = engine.snapshot()
            next_beat += 1
        minutes.append(minute)
        states.append(engine.snapshot())
        labels.append(engine.to_emotion_label())
        engine.decay(STEP_SECONDS)
        seconds += STEP_SECONDS
    return minutes, states, labels, moments


def _value_at(minutes: list[float], states: list[dict[str, float]], name: str, minute: float) -> float:
    index = max(i for i, m in enumerate(minutes) if m <= minute)
    return states[index][name]


def arc_description() -> str:
    """Alt text computed from the replay so it cannot drift from the curves."""
    minutes, states, _, _ = replay()
    def v(name: str, minute: float) -> float:
        return _value_at(minutes, states, name, minute)
    return (
        "Six modulator curves from OpenRook's deterministic emotional engine evaluating an "
        "authored 90-minute rupture-and-repair event sequence (illustrative dialogue, no LLM). "
        f"Arousal jumps to {v('arousal', 24):.2f} at the rupture and is back near baseline "
        f"within minutes; valence falls from {v('valence', 23.9):.2f} to {v('valence', 27):.2f}, "
        f"drifts up only to {v('valence', 59.9):.2f} over the next half hour, then rises to "
        f"{v('valence', 60):.2f} at the repair; bonding dips from {v('bonding', 23.9):.2f} to "
        f"{v('bonding', 27):.2f} and later returns to {v('bonding', 90):.2f}; unresolved tension "
        f"starts at {v('resolution', 24):.2f}, declines only to {v('resolution', 59.9):.2f}, "
        "and the repair clears it."
    )


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _line_path(points: list[tuple[float, float]]) -> str:
    """Polyline through samples; jumps stay jumps (no spline overshoot)."""
    if not points:
        return ""
    head = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    return head + "".join(f"L{x:.1f},{y:.1f}" for x, y in points[1:])


def _defs(extra: str = "") -> str:
    return f"<defs>{extra}</defs>"


def _svg(
    width: int, height: int, title: str, desc: str, body: str, style: str = "",
    *, slug: str,
) -> str:
    # Scope reusable definitions as well as accessible labels for inline consumers.
    body = re.sub(r'id="([^"]+)"', rf'id="{slug}-\1"', body)
    body = re.sub(r'url\(#([^)]+)\)', rf'url(#{slug}-\1)', body)
    title_id, desc_id = f"{slug}-title", f"{slug}-desc"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="{title_id} {desc_id}">'
        f'<title id="{title_id}">{escape(title)}</title><desc id="{desc_id}">{escape(desc)}</desc>'
        f"<style>{style}</style>{body}</svg>"
    )
    return "\n".join(line.rstrip() for line in svg.splitlines() if line.strip()) + "\n"


# ---------------------------------------------------------------------------
# 1. How it works (illustrative, animated explainer)
# ---------------------------------------------------------------------------

HOW_W, HOW_H = 1600, 1040


def how_it_works_description() -> str:
    return (
        "Illustrative comparison: a plain LLM call receives no history of Monday; "
        "OpenRook saves state after a detected rupture and session consolidation, "
        "then supplies retrieved memories, relationship open loops, trust, current mood, "
        "and identity context to Friday’s prompt. Your LLM writes the reply; dialogue "
        "and state changes are examples, not a recorded run."
    )


def build_how_it_works(theme: Theme = LIGHT) -> str:
    """Explain persistence without presenting illustrative state as measured output.

    Relationship loops are written during session digestion, then retrieved with
    memories in pipeline/core.py. The generator injects those summaries, loops,
    person trust, current modulators and identity context into its system prompt.
    The exact utterance does not guarantee rupture detection or a particular reply.
    """
    accent = theme.amber_ink

    def text(
        x: int, y: int, value: str, size: int = 24, *,
        fill: str = theme.ink, weight: str = "normal",
    ) -> str:
        return (
            f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
        )

    def arrow(x1: int, x2: int, y: int) -> str:
        return (
            f'<path d="M{x1},{y} H{x2}" fill="none" stroke="{theme.muted}" '
            'stroke-width="2" marker-end="url(#how-arrow)"/>'
        )

    parts = [
        _defs(
            rook_defs("how", halo=False)
            + f'<marker id="how-arrow" viewBox="0 0 8 8" refX="7" refY="4" '
            f'markerWidth="8" markerHeight="8" orient="auto">'
            f'<path d="M1,1 L7,4 L1,7" fill="none" stroke="{theme.muted}"/></marker>'
        ),
        text(56, 68, "The next conversation can start with a history.", 40),
        text(56, 112, "Same message on Friday. Different context behind the reply.", fill=theme.muted),
        text(64, 164, "MONDAY", 22, fill=theme.muted),
        text(444, 164, "BETWEEN CONVERSATIONS", 22, fill=theme.muted),
        text(936, 164, "FRIDAY", 22, fill=theme.muted),
        f'<rect x="32" y="188" width="1536" height="268" rx="8" '
        f'fill="{theme.panel}" stroke="{theme.border}"/>',
        f'<rect x="32" y="480" width="1536" height="412" rx="8" '
        f'fill="{theme.panel}" stroke="{theme.border}"/>',
        text(64, 230, "Plain LLM call", 28, weight="bold"),
        text(308, 230, "no history supplied", 22, fill=theme.muted),
        text(64, 428, "No supplied history, no memory of Monday.", 22, fill=theme.muted),
        text(64, 526, "With OpenRook", 28, fill=accent, weight="bold"),
        text(392, 526, "State persists and shapes the reply.", fill=theme.muted),
    ]

    # Source-visible groups: CairoSVG and motion-disabled viewers get the full story.
    parts.append(
        '<g class="how-step how-monday" data-step="1">'
        + text(64, 278, "YOU", 18, fill=theme.muted)
        + text(64, 314, "“You completely", 28)
        + text(64, 352, "misunderstood me.”", 28)
        + text(64, 566, "YOU", 18, fill=theme.muted)
        + text(64, 602, "“You completely", 28)
        + text(64, 640, "misunderstood me.”", 28)
        + text(64, 694, "Example: rupture detected", 20, fill=theme.muted)
        + text(64, 724, "State saved at session end", 20, fill=theme.muted)
        + "</g>"
    )
    parts.append(
        '<g class="how-step how-state" data-step="2">'
        + arrow(392, 424, 324)
        + text(444, 314, "Nothing saved for this call", 24, fill=theme.muted)
        + text(444, 352, "Monday is absent from the prompt.", 22, fill=theme.muted)
        + arrow(392, 424, 624)
        + f'<rect x="436" y="552" width="448" height="300" rx="8" '
        f'fill="{theme.inset}" stroke="{theme.border}"/>'
        + rook_mark(448, 564, 64, "how", halo=False)
        + text(520, 588, "Persistent state", 26, weight="bold")
        + text(520, 620, "After Monday’s session", 20, fill=theme.muted)
        + f'<rect x="460" y="644" width="216" height="40" rx="6" fill="{theme.panel}" stroke="{theme.border}"/>'
        + text(476, 671, "memory written", 22)
        + f'<rect x="692" y="644" width="168" height="40" rx="6" fill="{theme.panel}" stroke="{theme.border}"/>'
        + text(708, 671, "trust ↓", 22)
        + f'<rect x="460" y="700" width="400" height="64" rx="6" fill="{theme.panel}" stroke="{theme.border}"/>'
        + text(476, 725, "Open loop:", 20, fill=theme.muted)
        + text(476, 751, "the misunderstanding", 24)
        + f'<rect x="460" y="780" width="168" height="40" rx="6" fill="{theme.panel}" stroke="{theme.border}"/>'
        + text(476, 807, "tension ↑", 22)
        + text(648, 807, "Modeled state", 20, fill=theme.muted)
        + "</g>"
    )
    parts.append(
        '<g class="how-step how-friday" data-step="3">'
        + arrow(892, 920, 324)
        + text(936, 278, "YOU", 18, fill=theme.muted)
        + text(936, 314, "“Can we try again?”", 28)
        + text(936, 358, "LLM REPLY", 18, fill=theme.muted)
        + text(936, 394, "“Sure! What would you like", 28)
        + text(936, 432, "to work on?”", 28)
        + text(936, 566, "YOU", 18, fill=theme.muted)
        + text(936, 602, "“Can we try again?”", 28)
        + arrow(892, 920, 684)
        + f'<rect x="928" y="626" width="608" height="116" rx="8" '
        f'fill="{theme.inset}" stroke="{theme.border}"/>'
        + text(948, 654, "Context carried into the prompt", 24, weight="bold")
        + text(948, 688, "Retrieved memories · open loops · trust", 22)
        + text(948, 720, "Current mood · identity", 22)
        + text(936, 780, "LLM REPLY", 18, fill=theme.muted)
        + text(936, 816, "“Yes. Let’s revisit what", 28)
        + text(936, 854, "I misunderstood last time.”", 28)
        + "</g>"
    )
    parts.append(
        f'<path d="M56,908 H1544" stroke="{theme.border}" fill="none"/>'
        + text(152, 944, "Your app → OpenRook (memory · relationships · mood · identity)", 24)
        + text(152, 980, "→ supported LLM backends: Ollama · OpenAI-compatible · Codex", 24)
        + text(
            152, 1016,
            "Illustrative example — OpenRook keeps the state; your LLM still writes the words.",
            20, fill=theme.muted,
        )
    )
    style = """
@media (prefers-reduced-motion: no-preference) {
  .how-step { animation: how-reveal 600ms ease-out both; }
  .how-state { animation-delay: 700ms; }
  .how-friday { animation-delay: 1400ms; }
}
@keyframes how-reveal { from { opacity: 0.25; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) {
  .how-step { animation: none; opacity: 1; }
}
@media print {
  .how-step { animation: none; opacity: 1; }
}"""
    return _svg(
        HOW_W, HOW_H, "How OpenRook carries state into the next conversation",
        how_it_works_description(), "".join(parts), style, slug=f"how-it-works-{theme.name}",
    )


# ---------------------------------------------------------------------------
# 2. Emotional arc (engine detail)
# ---------------------------------------------------------------------------

ARC_W, ARC_H = 1600, 860
CHART = (480, 150, 1540, 560)  # x0, y0, x1, y1
DRAW_SECONDS = 2.4


def build_emotional_arc(theme: Theme = LIGHT) -> str:
    minutes, states, labels, _ = replay()
    x0, y0, x1, y1 = CHART

    def px(minute: float) -> float:
        return x0 + (x1 - x0) * minute / DURATION_MIN

    def py(value: float) -> float:
        return y1 - (y1 - y0) * value

    reveal = (f'<mask id="reveal" maskUnits="userSpaceOnUse" x="0" y="0" width="{ARC_W}" height="{ARC_H}">'
              f'<rect class="reveal" x="{x0 - 12}" y="0" width="{x1 - x0 + 24}" '
              f'height="{ARC_H}" fill="#FFFFFF"/></mask>')
    parts = [_defs(reveal + rook_defs("hero", halo=False))]

    # Left column: rook and reading guide; only the window changes with the timeline.
    parts.append(rook_mark(122, 140, 210, "hero", halo=False))
    parts.append(
        f'<text x="70" y="500" font-family="{SERIF}" font-size="34" fill="{theme.ink}">Inside one</text>'
        f'<text x="70" y="540" font-family="{SERIF}" font-size="34" fill="{theme.ink}">conversation</text>'
        f'<text x="72" y="586" font-family="{SANS}" font-size="19" fill="{theme.muted}">'
        "Each line is an internal state variable.</text>"
        f'<text x="72" y="616" font-family="{SANS}" font-size="19" fill="{theme.muted}">'
        "Rupture raises tension, lowers mood.</text>"
        f'<text x="72" y="646" font-family="{SANS}" font-size="19" fill="{theme.muted}">'
        "The authored repair clears tension.</text>"
    )

    # Grid + axis
    for v in (0.25, 0.5, 0.75):
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{py(v):.1f}" y2="{py(v):.1f}" '
                     f'stroke="{theme.muted}" stroke-opacity="0.18"/>')
    for v in (0, 0.5, 1):
        parts.append(f'<text x="{x0 - 20}" y="{py(v) + 6:.1f}" text-anchor="end" '
                     f'font-family="{SANS}" font-size="18" fill="{theme.muted}">{v:g}</text>')
    for m in range(0, DURATION_MIN + 1, 15):
        parts.append(f'<text x="{px(m):.1f}" y="{y1 + 26}" text-anchor="middle" '
                     f'font-family="{SANS}" font-size="18" fill="{theme.muted}">{m} min</text>')

    # Event markers
    for beat in SCRIPT:
        x = px(beat.minute)
        strong = bool(beat.quote)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 - 8}" y2="{y1}" stroke="{theme.muted}" '
                     f'stroke-opacity="{0.4 if strong else 0.15}" stroke-dasharray="3 5"/>')
        if beat.quote:
            accent = theme.colors[0] if beat.event is EventType.CONFLICT else (
                theme.plum if beat.event is EventType.RESOLUTION else theme.amber_ink)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y0 - 8}" r="4" fill="{accent}"/>'
                f'<text x="{x:.1f}" y="{y0 - 20}" text-anchor="middle" font-family="{SANS}" '
                f'font-size="18" font-weight="bold" letter-spacing="1.5" fill="{accent}">'
                f"{escape(beat.caption.upper())}</text>"
            )
    quotes = [b for b in SCRIPT if b.quote]
    for i, beat in enumerate(quotes):
        x = min(max(px(beat.minute), x0 + 120), x1 - 120)
        y = y0 - 52 - (i % 2) * 34
        parts.append(f'<text x="{x:.1f}" y="{y}" text-anchor="middle" font-family="{SERIF}" '
                     f'font-style="italic" font-size="18" fill="{theme.ink}">'
                     f"“{escape(beat.quote)}”</text>")

    # Curves, revealed left to right on the shared timeline
    curves = []
    for name, _, color in theme.modulators:
        pts = [(px(m), py(s[name])) for m, s in zip(minutes, states)]
        width = 4.2 if name in ("valence", "bonding", "arousal") else 2.6
        curves.append(
            f'<path d="{_line_path(pts)}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    parts.append('<g mask="url(#reveal)">' + "".join(curves) + "</g>")

    # Playhead: hidden in static renders, sweeps with the reveal when animated
    parts.append(
        f'<line class="playhead" x1="{x1}" x2="{x1}" y1="{y0 - 8}" y2="{y1}" stroke="{theme.muted}" '
        'stroke-opacity="0.6" stroke-width="1.5" opacity="0"/>'
    )

    # Emotion-label strip
    strip_y = y1 + 70
    runs: list[tuple[int, int, str]] = []
    for i, label in enumerate(labels):
        if runs and runs[-1][2] == label:
            runs[-1] = (runs[-1][0], i, label)
        else:
            runs.append((i, i, label))
    for start, end, label in runs:
        xa, xb = px(minutes[start]), px(minutes[min(end + 1, len(minutes) - 1)])
        color = theme.label_color(label)
        parts.append(f'<rect x="{xa:.1f}" y="{strip_y}" width="{max(xb - xa, 1):.1f}" height="10" '
                     f'fill="{color}" opacity="0.85"/>')
        if xb - xa > len(label) * 11:
            parts.append(f'<text x="{(xa + xb) / 2:.1f}" y="{strip_y + 30}" text-anchor="middle" '
                         f'font-family="{SANS}" font-size="18" fill="{color}">{label}</text>')
    parts.append(f'<text x="{x0}" y="{strip_y - 8}" font-family="{SANS}" font-size="18" '
                 f'letter-spacing="1.5" fill="{theme.muted}">ENGINE LABEL (LOGGING HEURISTIC)</text>')

    # Legend
    for i, (name, label, color) in enumerate(theme.modulators):
        lx = x0 + (i % 3) * 310
        ly = 690 + (i // 3) * 34
        parts.append(f'<rect x="{lx}" y="{ly}" width="22" height="5" rx="2.5" fill="{color}"/>'
                     f'<text x="{lx + 30}" y="{ly + 8}" font-family="{SANS}" font-size="20" '
                     f'fill="{theme.ink}">{label}</text>')

    parts.append(
        f'<text x="72" y="784" font-family="{SANS}" font-size="20" fill="{theme.muted}">'
        "Authored event sequence evaluated by the real deterministic engine (shipped config/modulators.yaml).</text>"
        f'<text x="72" y="816" font-family="{SANS}" font-size="20" fill="{theme.muted}">'
        "Dialogue is illustrative; no LLM. Emotion words are the engine's logging heuristic: functional state, not a claim of felt emotion.</text>"
    )

    style = f"""
@media (prefers-reduced-motion: no-preference) {{
  .reveal {{ transform-origin: {x0 - 12}px 0; animation: reveal {DRAW_SECONDS}s linear both; }}
  .playhead {{ animation: sweep {DRAW_SECONDS}s linear both; }}
}}
@keyframes reveal {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes sweep {{
  0% {{ transform: translateX({x0 - x1}px); opacity: 1; }}
  99% {{ opacity: 1; }}
  100% {{ transform: translateX(0); opacity: 0; }}
}}
@media (prefers-reduced-motion: reduce), print {{
  .reveal {{ animation: none; transform: none; }}
  .playhead {{ display: none; }}
}}"""
    return _svg(ARC_W, ARC_H, "Inside one conversation: six state variables",
                arc_description(), "".join(parts), style, slug=f"emotional-arc-{theme.name}")


# ---------------------------------------------------------------------------
# 3. Emotional fingerprints
# ---------------------------------------------------------------------------

FP_W, FP_H = 1600, 650


def _flower(cx: float, cy: float, state: dict[str, float], radius: float, theme: Theme) -> str:
    petals = []
    for i, (name, _, color) in enumerate(theme.modulators):
        angle = -90 + i * 60
        length = 18 + radius * state[name]
        rad = math.radians(angle)
        tip = (cx + length * math.cos(rad), cy + length * math.sin(rad))
        side = math.radians(angle + 90)
        w = 10 + 16 * state[name]
        mid = (cx + 0.55 * length * math.cos(rad), cy + 0.55 * length * math.sin(rad))
        left = (mid[0] + w * math.cos(side), mid[1] + w * math.sin(side))
        right = (mid[0] - w * math.cos(side), mid[1] - w * math.sin(side))
        petals.append(
            f'<path d="M{cx:.1f},{cy:.1f} Q{left[0]:.1f},{left[1]:.1f} {tip[0]:.1f},{tip[1]:.1f} '
            f'Q{right[0]:.1f},{right[1]:.1f} {cx:.1f},{cy:.1f}Z" fill="{color}" fill-opacity="0.78" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
    guide = "".join(
        f'<circle cx="{cx}" cy="{cy}" r="{18 + radius * v:.1f}" fill="none" stroke="{theme.muted}" '
        f'stroke-opacity="0.18" stroke-dasharray="2 6"/>' for v in (0.5, 1.0)
    )
    return (
        '<g>'
        f'{guide}{"".join(petals)}<circle cx="{cx}" cy="{cy}" r="9" fill="{theme.amber}"/></g>'
    )


def build_emotional_fingerprints(theme: Theme = LIGHT) -> str:
    _, _, _, moments = replay()
    engine = shipped_engine()
    order = [
        ("calm", "Calm", ("Baseline before", "the conversation")),
        ("warmth", "Warmth", ("“That really helped.”", "")),
        ("rupture", "Rupture", ("“You completely misunderstood me.”", "+ pushback")),
        ("repair", "Repair", ("After “I think I was too harsh.”", "")),
    ]
    parts = []
    parts.append(f'<text x="60" y="70" font-family="{SERIF}" font-size="40" fill="{theme.ink}">'
                 "Four moments, four shapes of state</text>")
    parts.append(f'<text x="62" y="110" font-family="{SANS}" font-size="20" fill="{theme.muted}">'
                 "Each petal is one modulator's value at that moment of the same engine evaluation "
                 "(authored events, illustrative dialogue).</text>")
    for i, (key, title, note) in enumerate(order):
        cx = 220 + i * 385
        cy = 290
        state = moments[key]
        for name in engine.snapshot():
            setattr(engine.state, name, state[name])
        label = engine.to_emotion_label()
        parts.append(_flower(cx, cy, state, 112, theme))
        parts.append(
            f'<text x="{cx}" y="{cy + 150}" text-anchor="middle" font-family="{SERIF}" '
            f'font-size="28" fill="{theme.ink}">{title}</text>'
            f'<text x="{cx}" y="{cy + 180}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="19" fill="{theme.muted}">{escape(note[0])}</text>'
            f'<text x="{cx}" y="{cy + 208}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="19" fill="{theme.muted}">{escape(note[1])}</text>'
            f'<text x="{cx}" y="{cy + 244}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="18" letter-spacing="1" fill="{theme.label_color(label)}">'
            f"ENGINE LABEL: {label.upper()}</text>"
        )
    lx = 160
    for j, (name, label, color) in enumerate(theme.modulators):
        x = lx + j * 210
        y = 604
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>'
                     f'<text x="{x + 14}" y="{y + 6}" font-family="{SANS}" font-size="20" '
                     f'fill="{theme.ink}">{label}</text>')
    return _svg(
        FP_W, FP_H, "Four moments, four shapes of state",
        "Six-petal glyphs of the engine's modulator values at calm, warmth, rupture and repair.",
        "".join(parts), slug=f"emotional-fingerprints-{theme.name}",
    )


# ---------------------------------------------------------------------------
# 4. Retained-choice evidence
# ---------------------------------------------------------------------------

# (seed, learned-Bo-reliable value, learned-Ari-reliable value); +1 = Ari first,
# -1 = Bo first, 0 = no partner preference. Source: reports/research-comparison/
# 20261019T170937Z-retained-choice-v4-replacement/study/analysis.json
RETAINED_CHOICE_PAIRS: tuple[tuple[int, float, float], ...] = (
    (56003, 0.0, 0.0), (57427, -1.0, 1.0), (58901, 0.0, 0.0), (60271, -1.0, 1.0),
    (61927, 0.0, 0.0), (63313, 0.0, 0.0), (64901, 0.0, 1.0), (66109, 0.0, 0.0),
    (67987, 0.0, 1.0), (69403, 0.0, 0.0), (71023, -1.0, 1.0), (72701, 0.0, 1.0),
    (74363, -1.0, 0.0), (75913, 1.0, 1.0),
)
RETAINED_CHOICE_INTERVAL = (0.2857142857142857, 1.1428571428571428)
RETAINED_CHOICE_THRESHOLD = 0.2

RC_W, RC_H = 1600, 910


def build_retained_choice(theme: Theme = LIGHT) -> str:
    pairs = RETAINED_CHOICE_PAIRS
    mean = sum(a - b for _, b, a in pairs) / len(pairs)
    zero = sum(1 for _, b, a in pairs if a == b)
    x_bo, x_none, x_ari = 420, 640, 860
    xs = {-1.0: x_bo, 0.0: x_none, 1.0: x_ari}
    top, row_h = 210, 38

    parts = []
    parts.append(
        f'<text x="60" y="78" font-family="{SERIF}" font-size="44" fill="{theme.ink}">'
        "Same world. Opposite pasts. Different choices.</text>"
        f'<text x="62" y="112" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "Completed v4 confirmation: 14 subject pairs / 28 subjects. In each seeded world one subject "
        "lived through Bo being reliable, the other Ari.</text>"
        f'<text x="62" y="136" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "After a restart, whom do they ask to work with at their first work opportunity?</text>"
    )
    # Column heads
    for x, text, color in ((x_bo, "BO FIRST", theme.bo), (x_none, "ZERO BALANCE", theme.muted),
                           (x_ari, "ARI FIRST", theme.amber_ink)):
        parts.append(f'<text x="{x}" y="{top - 22}" text-anchor="middle" font-family="{SANS}" '
                     f'font-size="18" font-weight="bold" letter-spacing="2" fill="{color}">{text}</text>'
                     f'<line x1="{x}" x2="{x}" y1="{top - 10}" y2="{top + row_h * len(pairs) - 14}" '
                     f'stroke="{theme.muted}" stroke-opacity="0.18"/>')
    parts.append(f'<text x="80" y="{top - 22}" font-family="{SANS}" font-size="18" font-weight="bold" '
                 f'letter-spacing="2" fill="{theme.muted}">WORLD SEED</text>'
                 f'<text x="1010" y="{top - 22}" text-anchor="middle" font-family="{SANS}" '
                 f'font-size="18" font-weight="bold" letter-spacing="2" fill="{theme.muted}">DIFFERENCE</text>')

    for i, (seed, bo_value, ari_value) in enumerate(pairs):
        y = top + i * row_h
        delay = 0.25 * i
        parts.append(f'<text x="80" y="{y + 5}" font-family="{SANS}" font-size="18" '
                     f'fill="{theme.muted}">#{i + 1:02d} · {seed}</text>')
        xb, xa = xs[bo_value], xs[ari_value]
        diff = ari_value - bo_value
        if xb != xa:
            parts.append(
                f'<linearGradient id="link{i}" gradientUnits="userSpaceOnUse" x1="{xb}" x2="{xa}" '
                f'y1="{y}" y2="{y}"><stop offset="0" stop-color="{theme.bo}"/>'
                f'<stop offset="1" stop-color="{theme.amber_ink}"/></linearGradient>'
                f'<path class="link" style="animation-delay:{delay:.2f}s" d="M{xb},{y}L{xa},{y}" '
                f'stroke="url(#link{i})" stroke-width="5" stroke-linecap="round" pathLength="1"/>'
            )
            parts.append(f'<circle cx="{xb}" cy="{y}" r="11" fill="{theme.bo}"/>'
                         f'<circle cx="{xa}" cy="{y}" r="11" fill="{theme.amber_ink}"/>')
        else:
            parts.append(
                f'<path d="M{xb},{y - 11} A11,11 0 0 0 {xb},{y + 11}Z" fill="{theme.bo}"/>'
                f'<path d="M{xb},{y - 11} A11,11 0 0 1 {xb},{y + 11}Z" fill="{theme.amber_ink}"/>'
            )
        color = theme.amber_ink if diff > 0 else theme.muted
        parts.append(f'<text x="1010" y="{y + 6}" text-anchor="middle" font-family="{SANS}" '
                     f'font-size="18" font-weight="bold" fill="{color}">{diff:+.0f}</text>')

    # Summary panel
    px0, py0 = 1110, 250
    parts.append(
        f'<text x="{px0}" y="{py0 + 10}" font-family="{SERIF}" font-size="110" fill="{theme.amber_ink}">'
        f'{mean:.2f}</text>'
        f'<text x="{px0 + 2}" y="{py0 + 44}" font-family="{SANS}" font-size="18" fill="{theme.ink}">'
        "mean paired difference</text>"
        f'<text x="{px0 + 2}" y="{py0 + 66}" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        f"95% interval {RETAINED_CHOICE_INTERVAL[0]:.2f}–{RETAINED_CHOICE_INTERVAL[1]:.2f}</text>"
    )
    # Interval bar (scale -0.25..1.5)
    def sx(v: float) -> float:
        return px0 + (v + 0.25) / 1.75 * 400
    by = py0 + 110
    parts.append(
        f'<line x1="{sx(-0.25)}" x2="{sx(1.5)}" y1="{by}" y2="{by}" stroke="{theme.muted}" stroke-opacity="0.18"/>'
        f'<line x1="{sx(0)}" x2="{sx(0)}" y1="{by - 12}" y2="{by + 12}" stroke="{theme.muted}" stroke-opacity="0.35"/>'
        f'<text x="{sx(0)}" y="{by + 28}" text-anchor="middle" font-family="{SANS}" font-size="18" fill="{theme.muted}">0</text>'
        f'<line x1="{sx(RETAINED_CHOICE_THRESHOLD)}" x2="{sx(RETAINED_CHOICE_THRESHOLD)}" y1="{by - 16}" y2="{by + 16}" '
        f'stroke="{theme.plum}" stroke-width="2" stroke-dasharray="3 3"/>'
        f'<text x="{sx(RETAINED_CHOICE_THRESHOLD)}" y="{by - 22}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="18" fill="{theme.plum}">0.2 bar</text>'
        f'<line x1="{sx(RETAINED_CHOICE_INTERVAL[0])}" x2="{sx(RETAINED_CHOICE_INTERVAL[1])}" y1="{by}" y2="{by}" '
        f'stroke="{theme.amber_ink}" stroke-width="6" stroke-linecap="round"/>'
        f'<circle cx="{sx(mean)}" cy="{by}" r="8" fill="{theme.amber}"/>'
    )
    notes = [
        "Interval clears the preregistered",
        "0.2 bar.",
        "",
        f"{zero} of 14 pairs show no difference.",
        "",
        "Both arms saw runtime-computed",
        "partner estimates: this is runtime +",
        "model evidence, not isolated model",
        "learning or human equivalence.",
    ]
    for j, line in enumerate(notes):
        parts.append(f'<text x="{px0}" y="{by + 66 + j * 22}" font-family="{SANS}" font-size="18" '
                     f'fill="{theme.muted}">{escape(line)}</text>')
    # Legend
    ly = top + row_h * len(pairs) + 24
    parts.append(
        f'<circle cx="90" cy="{ly}" r="8" fill="{theme.bo}"/><text x="106" y="{ly + 5}" font-family="{SANS}" '
        f'font-size="18" fill="{theme.ink}">learned “Bo is reliable”</text>'
        f'<circle cx="410" cy="{ly}" r="8" fill="{theme.amber_ink}"/><text x="426" y="{ly + 5}" font-family="{SANS}" '
        f'font-size="18" fill="{theme.ink}">learned “Ari is reliable”</text>'
        f'<text x="80" y="{ly + 40}" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "Endpoint: partner balance of work requests at the first post-restart work opportunity</text>"
        f'<text x="80" y="{ly + 64}" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "(+1 Ari, −1 Bo; 0 = no work, solo work, or balanced). Restart = same-process manager "
        "rebuild.</text>"
        f'<text x="80" y="{ly + 88}" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "Full record: "
        "docs/plans/behavioral_validation_v4.md</text>"
    )
    style = """
@media (prefers-reduced-motion: no-preference) {
  .link { animation: grow 600ms ease-out both; }
}
@keyframes grow { from { stroke-dasharray: 0 1; } to { stroke-dasharray: 1 0; } }
@media (prefers-reduced-motion: reduce), print { .link { animation: none; } }"""
    return _svg(
        RC_W, RC_H, "Same world. Opposite pasts. Different choices.",
        f"14 subject pairs (28 subjects) from the completed v4 confirmation. Mean paired difference "
        f"{mean:.2f}, 95% interval {RETAINED_CHOICE_INTERVAL[0]:.2f} to "
        f"{RETAINED_CHOICE_INTERVAL[1]:.2f}, above the 0.2 threshold; {zero} pairs show no difference.",
        "".join(parts), style, slug=f"retained-choice-{theme.name}",
    )


# ---------------------------------------------------------------------------
# 5. GitHub social cover (1280 x 640)
# ---------------------------------------------------------------------------

COVER_W, COVER_H = 1280, 640


def build_github_cover() -> str:
    theme = LIGHT  # Social previews have a single, opaque background.
    minutes, states, _, _ = replay()
    x0, x1, y0, y1 = 610, 1230, 130, 500

    def pt(minute: float, value: float) -> tuple[float, float]:
        return x0 + (x1 - x0) * minute / DURATION_MIN, y1 - (y1 - y0) * value

    parts = [
        _defs(rook_defs("cover", halo=False)),
        f'<rect width="{COVER_W}" height="{COVER_H}" fill="{theme.page}"/>',
    ]
    for name, _, color in theme.modulators:
        pts = [pt(m, s[name]) for m, s in zip(minutes, states)]
        parts.append(f'<path d="{_line_path(pts)}" fill="none" stroke="{color}" stroke-width="3" '
                     f'stroke-opacity="0.8" stroke-linecap="round"/>')
    parts.append(rook_mark(96, 118, 236, "cover", halo=False))
    parts.append(
        f'<text x="64" y="452" font-family="{SERIF}" font-size="86" fill="{theme.ink}">OpenRook</text>'
        f'<text x="68" y="500" font-family="{SANS}" font-size="24" fill="{theme.muted}">'
        "LLMs forget.</text>"
        f'<text x="68" y="534" font-family="{SANS}" font-size="24" fill="{theme.muted}">'
        "OpenRook remembers.</text>"
        f'<text x="68" y="596" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "Open-source research runtime · functional state, not felt emotion</text>"
        f'<text x="{x0}" y="{y1 + 34}" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "Six modulators · real engine evaluation</text>"
        f'<text x="{x0}" y="{y1 + 62}" font-family="{SANS}" font-size="18" fill="{theme.muted}">'
        "Authored rupture → repair · no LLM</text>"
    )
    return _svg(COVER_W, COVER_H, "OpenRook",
                "OpenRook social preview: rook mark with a lit window, wordmark, and six real modulator curves.",
                "".join(parts), slug="github-cover")


THEMED_BUILDERS = {
    "how-it-works": (build_how_it_works, (HOW_W, HOW_H)),
    "emotional-arc": (build_emotional_arc, (ARC_W, ARC_H)),
    "emotional-fingerprints": (build_emotional_fingerprints, (FP_W, FP_H)),
    "retained-choice": (build_retained_choice, (RC_W, RC_H)),
}
SHOWCASE = {
    f"{name}-{theme.name}": partial(builder, theme)
    for name, (builder, _) in THEMED_BUILDERS.items()
    for theme in THEMES
}
SHOWCASE["github-cover"] = build_github_cover
SIZES = {
    f"{name}-{theme.name}": size
    for name, (_, size) in THEMED_BUILDERS.items()
    for theme in THEMES
}
SIZES["github-cover"] = (COVER_W, COVER_H)

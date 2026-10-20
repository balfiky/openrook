"""Data-driven showcase visuals for the README.

Unlike ``build_diagrams`` (structural block diagrams), every curve, petal and
dot here is computed from real sources:

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
``cairosvg``). The SVGs animate in browsers (GitHub included) and honor
``prefers-reduced-motion``; PNG fallbacks show the complete static chart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from config.loader import load_config
from core.emotional_engine import EmotionalEngine
from core.types import EmotionalEvent, EventType, UnresolvedItem
from nur_tools.brand import rook_defs, rook_mark

SHIPPED_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

SERIF = "'EB Garamond','Cormorant Garamond',Georgia,'DejaVu Serif',serif"
SANS = "'DejaVu Sans','Inter','Segoe UI',Helvetica,Arial,sans-serif"

BG_TOP = "#120D1F"
BG_BOTTOM = "#2A1E3F"
INK = "#F4ECFF"
INK_SOFT = "#B9AECB"
INK_FAINT = "#A79CC0"

MODULATORS = [
    # name, display label, color
    ("arousal", "arousal", "#FF6B6B"),
    ("valence", "valence", "#F5A65B"),
    ("certainty", "certainty", "#5BC0EB"),
    ("bonding", "bonding", "#E86FB0"),
    ("energy", "energy", "#7BD389"),
    ("resolution", "unresolved tension", "#B48CFF"),
]
LABEL_COLORS = {
    "neutral": "#6F6386",
    "content": "#F5A65B",
    "joyful": "#FFC56B",
    "loving": "#E86FB0",
    "excited": "#FF9F6B",
    "fearful": "#FF6B6B",
    "angry": "#E0475A",
    "anxious": "#C77DFF",
    "uncertain": "#5BC0EB",
    "sad": "#4E7BD9",
    "detached": "#8C8CA8",
    "irritable": "#D9534F",
    "exhausted": "#55506A",
}


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


def _stars(width: int, height: int, count: int, seed: int) -> str:
    out = []
    state = seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        x = state % width
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        y = state % height
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        r = 0.6 + (state % 100) / 100
        out.append(f'<circle cx="{x}" cy="{y}" r="{r:.2f}" fill="#FFFFFF" opacity="0.18"/>')
    return "".join(out)


def _defs(extra: str = "") -> str:
    return f"""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="0.35" y2="1">
    <stop offset="0" stop-color="{BG_TOP}"/><stop offset="1" stop-color="{BG_BOTTOM}"/>
  </linearGradient>
  <filter id="glow" x="-20%" y="-50%" width="140%" height="200%">
    <feGaussianBlur stdDeviation="3.2" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  {extra}
</defs>"""


def _svg(width: int, height: int, title: str, desc: str, body: str, style: str = "") -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="t d">'
        f"<title id=\"t\">{escape(title)}</title><desc id=\"d\">{escape(desc)}</desc>"
        f"<style>{style}</style>{body}</svg>"
    )
    return "\n".join(line.rstrip() for line in svg.splitlines() if line.strip()) + "\n"


# ---------------------------------------------------------------------------
# 1. Emotional arc (animated hero)
# ---------------------------------------------------------------------------

ARC_W, ARC_H = 1600, 790
CHART = (480, 150, 1540, 560)  # x0, y0, x1, y1
DRAW_SECONDS = 9


def build_emotional_arc() -> str:
    minutes, states, labels, _ = replay()
    x0, y0, x1, y1 = CHART

    def px(minute: float) -> float:
        return x0 + (x1 - x0) * minute / DURATION_MIN

    def py(value: float) -> float:
        return y1 - (y1 - y0) * value

    reveal = (f'<mask id="reveal"><rect class="reveal" x="{x0 - 12}" y="0" width="{x1 - x0 + 24}" '
              f'height="{ARC_H}" fill="#FFFFFF"/></mask>')
    parts = [_defs(reveal + rook_defs("hero")), f'<rect width="{ARC_W}" height="{ARC_H}" rx="28" fill="url(#bg)"/>',
             _stars(ARC_W, ARC_H, 90, 7)]

    # Left column: the rook + headline. Its halo and lit window share the chart's timeline.
    parts.append(
        '<g transform="translate(205 250)"><circle class="halo" r="150" fill="url(#heroHalo)"/></g>'
        + rook_mark(122, 140, 210, "hero", halo=False)
    )
    parts.append(
        f'<text x="70" y="500" font-family="{SERIF}" font-size="34" fill="{INK}">The moment passes.</text>'
        f'<text x="70" y="540" font-family="{SERIF}" font-size="34" fill="{INK}">Its state carries on.</text>'
        f'<text x="72" y="586" font-family="{SANS}" font-size="16" fill="{INK_SOFT}">'
        "Six state variables persist, decay and</text>"
        f'<text x="72" y="608" font-family="{SANS}" font-size="16" fill="{INK_SOFT}">'
        "shape what the model is asked to write.</text>"
    )

    # Grid + axis
    for v in (0.25, 0.5, 0.75):
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{py(v):.1f}" y2="{py(v):.1f}" '
                     f'stroke="#FFFFFF" stroke-opacity="0.06"/>')
    for m in range(0, DURATION_MIN + 1, 15):
        parts.append(f'<text x="{px(m):.1f}" y="{y1 + 26}" text-anchor="middle" '
                     f'font-family="{SANS}" font-size="13" fill="{INK_FAINT}">{m} min</text>')

    # Event markers
    for beat in SCRIPT:
        x = px(beat.minute)
        strong = bool(beat.quote)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 - 8}" y2="{y1}" stroke="#FFFFFF" '
                     f'stroke-opacity="{0.22 if strong else 0.08}" stroke-dasharray="3 5"/>')
        if beat.quote:
            accent = "#FF6B6B" if beat.event is EventType.CONFLICT else (
                "#B48CFF" if beat.event is EventType.RESOLUTION else "#F5A65B")
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y0 - 8}" r="4" fill="{accent}"/>'
                f'<text x="{x:.1f}" y="{y0 - 20}" text-anchor="middle" font-family="{SANS}" '
                f'font-size="11" font-weight="bold" letter-spacing="1.5" fill="{accent}">'
                f"{escape(beat.caption.upper())}</text>"
            )
    quotes = [b for b in SCRIPT if b.quote]
    for i, beat in enumerate(quotes):
        x = min(max(px(beat.minute), x0 + 120), x1 - 120)
        y = y0 - 52 - (i % 2) * 34
        parts.append(f'<text x="{x:.1f}" y="{y}" text-anchor="middle" font-family="{SERIF}" '
                     f'font-style="italic" font-size="18" fill="{INK}">'
                     f"“{escape(beat.quote)}”</text>")

    # Curves, revealed left to right on the shared timeline
    curves = []
    for name, _, color in MODULATORS:
        pts = [(px(m), py(s[name])) for m, s in zip(minutes, states)]
        width = 4.2 if name in ("valence", "bonding", "arousal") else 2.6
        curves.append(
            f'<path d="{_line_path(pts)}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>'
        )
    parts.append('<g mask="url(#reveal)">' + "".join(curves) + "</g>")

    # Playhead: hidden in static renders, sweeps with the reveal when animated
    parts.append(
        f'<line class="playhead" x1="{x1}" x2="{x1}" y1="{y0 - 8}" y2="{y1}" stroke="#FFFFFF" '
        'stroke-opacity="0.6" stroke-width="1.5" opacity="0"/>'
    )

    # Emotion-label strip
    strip_y = y1 + 50
    runs: list[tuple[int, int, str]] = []
    for i, label in enumerate(labels):
        if runs and runs[-1][2] == label:
            runs[-1] = (runs[-1][0], i, label)
        else:
            runs.append((i, i, label))
    for start, end, label in runs:
        xa, xb = px(minutes[start]), px(minutes[min(end + 1, len(minutes) - 1)])
        color = LABEL_COLORS.get(label, "#6F6386")
        parts.append(f'<rect x="{xa:.1f}" y="{strip_y}" width="{max(xb - xa, 1):.1f}" height="10" '
                     f'fill="{color}" opacity="0.85"/>')
        if xb - xa > 44:
            parts.append(f'<text x="{(xa + xb) / 2:.1f}" y="{strip_y + 30}" text-anchor="middle" '
                         f'font-family="{SANS}" font-size="13" fill="{color}">{label}</text>')
    parts.append(f'<text x="{x0}" y="{strip_y - 8}" font-family="{SANS}" font-size="11" '
                 f'letter-spacing="1.5" fill="{INK_FAINT}">ENGINE LABEL (LOGGING HEURISTIC)</text>')

    # Legend
    lx = x0
    for name, label, color in MODULATORS:
        parts.append(f'<rect x="{lx}" y="{ARC_H - 104}" width="22" height="5" rx="2.5" fill="{color}"/>'
                     f'<text x="{lx + 30}" y="{ARC_H - 97}" font-family="{SANS}" font-size="14" '
                     f'fill="{INK}">{label}</text>')
        lx += 48 + len(label) * 8.2

    parts.append(
        f'<text x="{x0}" y="{ARC_H - 62}" font-family="{SANS}" font-size="15" fill="{INK_FAINT}">'
        "Authored event sequence evaluated by the real deterministic engine (shipped "
        "config/modulators.yaml). Dialogue is illustrative; no LLM.</text>"
        f'<text x="{x0}" y="{ARC_H - 38}" font-family="{SANS}" font-size="15" fill="{INK_FAINT}">'
        "Emotion words are the engine's logging heuristic: functional state, not a claim of "
        "felt emotion.</text>"
    )

    sweep = 85  # percent of the loop spent sweeping; the rest holds the final frame
    light_frames = []
    for m, s in zip(minutes, states):
        pct = m / DURATION_MIN * sweep
        light_frames.append((pct, 0.86 + 0.32 * s["arousal"], 0.25 + 0.75 * s["valence"]))
    window_kf = "".join(f"{p:.4f}% {{ opacity: {0.2 + 0.8 * op:.3f}; }}" for p, _, op in light_frames)
    halo_kf = "".join(
        f"{p:.4f}% {{ opacity: {op:.3f}; transform: scale({sc:.3f}); }}" for p, sc, op in light_frames
    )
    style = f"""
.reveal {{ transform-origin: {x0 - 12}px 0; animation: reveal {DRAW_SECONDS}s linear infinite; }}
@keyframes reveal {{ 0% {{ transform: scaleX(0); }} {sweep}% {{ transform: scaleX(1); }} 100% {{ transform: scaleX(1); }} }}
.playhead {{ animation: sweep {DRAW_SECONDS}s linear infinite; }}
@keyframes sweep {{
  0% {{ transform: translateX({x0 - x1}px); opacity: 1; }}
  {sweep}% {{ transform: translateX(0); opacity: 1; }}
  {sweep + 1}%, 100% {{ transform: translateX(0); opacity: 0; }}
}}
.rook-window {{ animation: windowlight {DRAW_SECONDS}s linear infinite; }}
@keyframes windowlight {{ {window_kf} 100% {{ opacity: {0.2 + 0.8 * light_frames[-1][2]:.3f}; }} }}
.halo {{ animation: halo {DRAW_SECONDS}s linear infinite; }}
@keyframes halo {{ {halo_kf} 100% {{ opacity: {light_frames[-1][2]:.3f}; transform: scale({light_frames[-1][1]:.3f}); }} }}
@media (prefers-reduced-motion: reduce) {{
  .reveal, .rook-window, .halo {{ animation: none; }}
  .playhead {{ display: none; }}
}}"""
    return _svg(ARC_W, ARC_H, "OpenRook — the moment passes, its state carries on",
                arc_description(), "".join(parts), style)


# ---------------------------------------------------------------------------
# 2. Emotional fingerprints
# ---------------------------------------------------------------------------

FP_W, FP_H = 1600, 520


def _flower(cx: float, cy: float, state: dict[str, float], radius: float, delay: float) -> str:
    petals = []
    for i, (name, _, color) in enumerate(MODULATORS):
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
            f'stroke="{color}" stroke-width="1.5" filter="url(#glow)"/>'
        )
    guide = "".join(
        f'<circle cx="{cx}" cy="{cy}" r="{18 + radius * v:.1f}" fill="none" stroke="#FFFFFF" '
        f'stroke-opacity="0.07" stroke-dasharray="2 6"/>' for v in (0.5, 1.0)
    )
    return (
        f'<g class="breathe" style="animation-delay:{delay:.1f}s; transform-origin:{cx}px {cy}px">'
        f'{guide}{"".join(petals)}<circle cx="{cx}" cy="{cy}" r="9" fill="#FFE3BD" filter="url(#glow)"/></g>'
    )


def build_emotional_fingerprints() -> str:
    _, _, _, moments = replay()
    engine = shipped_engine()
    order = [
        ("calm", "Calm", "baseline before the conversation"),
        ("warmth", "Warmth", "“That really helped.”"),
        ("rupture", "Rupture", "“You completely misunderstood me.” + pushback"),
        ("repair", "Repair", "after “I think I was too harsh.”"),
    ]
    parts = [_defs(), f'<rect width="{FP_W}" height="{FP_H}" rx="28" fill="url(#bg)"/>',
             _stars(FP_W, FP_H, 60, 11)]
    parts.append(f'<text x="60" y="70" font-family="{SERIF}" font-size="40" fill="{INK}">'
                 "Four moments, four shapes of state</text>")
    parts.append(f'<text x="62" y="102" font-family="{SANS}" font-size="15" fill="{INK_SOFT}">'
                 "Each petal is one modulator's value at that moment of the same engine evaluation "
                 "(authored events, illustrative dialogue).</text>")
    for i, (key, title, note) in enumerate(order):
        cx = 220 + i * 385
        cy = 290
        state = moments[key]
        for name in engine.snapshot():
            setattr(engine.state, name, state[name])
        label = engine.to_emotion_label()
        parts.append(_flower(cx, cy, state, 112, i * 0.6))
        parts.append(
            f'<text x="{cx}" y="{cy + 150}" text-anchor="middle" font-family="{SERIF}" '
            f'font-size="28" fill="{INK}">{title}</text>'
            f'<text x="{cx}" y="{cy + 176}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="13" fill="{INK_SOFT}">{escape(note)}</text>'
            f'<text x="{cx}" y="{cy + 200}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="12" letter-spacing="1.5" fill="{LABEL_COLORS.get(label, INK_FAINT)}">'
            f"ENGINE LABEL: {label.upper()}</text>"
        )
    lx = 1080
    for j, (name, label, color) in enumerate(MODULATORS):
        row, col = divmod(j, 3)
        x = lx + col * 170
        y = 56 + row * 26
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>'
                     f'<text x="{x + 14}" y="{y + 5}" font-family="{SANS}" font-size="13" '
                     f'fill="{INK}">{label}</text>')
    style = """
.breathe { animation: breathe 6s ease-in-out infinite; }
@keyframes breathe { 0%,100% { transform: scale(0.97); } 50% { transform: scale(1.03); } }
@media (prefers-reduced-motion: reduce) { .breathe { animation: none; } }"""
    return _svg(
        FP_W, FP_H, "Four moments, four shapes of state",
        "Six-petal glyphs of the engine's modulator values at calm, warmth, rupture and repair.",
        "".join(parts), style,
    )


# ---------------------------------------------------------------------------
# 3. Retained-choice evidence
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
BO = "#7FA7FF"
ARI = "#F5A65B"


def build_retained_choice() -> str:
    pairs = RETAINED_CHOICE_PAIRS
    mean = sum(a - b for _, b, a in pairs) / len(pairs)
    zero = sum(1 for _, b, a in pairs if a == b)
    x_bo, x_none, x_ari = 420, 640, 860
    xs = {-1.0: x_bo, 0.0: x_none, 1.0: x_ari}
    top, row_h = 210, 38

    parts = [_defs(), f'<rect width="{RC_W}" height="{RC_H}" rx="28" fill="url(#bg)"/>',
             _stars(RC_W, RC_H, 70, 23)]
    parts.append(
        f'<text x="60" y="78" font-family="{SERIF}" font-size="44" fill="{INK}">'
        "Same world. Opposite pasts. Different choices.</text>"
        f'<text x="62" y="112" font-family="{SANS}" font-size="16" fill="{INK_SOFT}">'
        "Completed v4 confirmation: 14 subject pairs / 28 subjects. In each seeded world one subject "
        "lived through Bo being reliable, the other Ari.</text>"
        f'<text x="62" y="136" font-family="{SANS}" font-size="16" fill="{INK_SOFT}">'
        "After a restart, whom do they ask to work with at their first work opportunity?</text>"
    )
    # Column heads
    for x, text, color in ((x_bo, "BO FIRST", BO), (x_none, "ZERO BALANCE", INK_FAINT),
                           (x_ari, "ARI FIRST", ARI)):
        parts.append(f'<text x="{x}" y="{top - 22}" text-anchor="middle" font-family="{SANS}" '
                     f'font-size="13" font-weight="bold" letter-spacing="2" fill="{color}">{text}</text>'
                     f'<line x1="{x}" x2="{x}" y1="{top - 10}" y2="{top + row_h * len(pairs) - 14}" '
                     f'stroke="#FFFFFF" stroke-opacity="0.07"/>')
    parts.append(f'<text x="80" y="{top - 22}" font-family="{SANS}" font-size="13" font-weight="bold" '
                 f'letter-spacing="2" fill="{INK_FAINT}">WORLD SEED</text>'
                 f'<text x="1010" y="{top - 22}" text-anchor="middle" font-family="{SANS}" '
                 f'font-size="13" font-weight="bold" letter-spacing="2" fill="{INK_FAINT}">DIFFERENCE</text>')

    for i, (seed, bo_value, ari_value) in enumerate(pairs):
        y = top + i * row_h
        delay = 0.25 * i
        parts.append(f'<text x="80" y="{y + 5}" font-family="{SANS}" font-size="14" '
                     f'fill="{INK_SOFT}">#{i + 1:02d} · {seed}</text>')
        xb, xa = xs[bo_value], xs[ari_value]
        diff = ari_value - bo_value
        if xb != xa:
            parts.append(
                f'<linearGradient id="link{i}" gradientUnits="userSpaceOnUse" x1="{xb}" x2="{xa}" '
                f'y1="{y}" y2="{y}"><stop offset="0" stop-color="{BO}"/>'
                f'<stop offset="1" stop-color="{ARI}"/></linearGradient>'
                f'<path class="link" style="animation-delay:{delay:.2f}s" d="M{xb},{y}L{xa},{y}" '
                f'stroke="url(#link{i})" stroke-width="5" stroke-linecap="round" pathLength="1"/>'
            )
            parts.append(f'<circle cx="{xb}" cy="{y}" r="11" fill="{BO}" filter="url(#glow)"/>'
                         f'<circle cx="{xa}" cy="{y}" r="11" fill="{ARI}" filter="url(#glow)"/>')
        else:
            parts.append(
                f'<path d="M{xb},{y - 11} A11,11 0 0 0 {xb},{y + 11}Z" fill="{BO}"/>'
                f'<path d="M{xb},{y - 11} A11,11 0 0 1 {xb},{y + 11}Z" fill="{ARI}"/>'
            )
        color = ARI if diff > 0 else INK_FAINT
        parts.append(f'<text x="1010" y="{y + 6}" text-anchor="middle" font-family="{SANS}" '
                     f'font-size="16" font-weight="bold" fill="{color}">{diff:+.0f}</text>')

    # Summary panel
    px0, py0 = 1110, 250
    parts.append(
        f'<text x="{px0}" y="{py0 + 10}" font-family="{SERIF}" font-size="110" fill="{ARI}" '
        f'filter="url(#glow)">{mean:.2f}</text>'
        f'<text x="{px0 + 2}" y="{py0 + 44}" font-family="{SANS}" font-size="14" fill="{INK}">'
        "mean paired difference</text>"
        f'<text x="{px0 + 2}" y="{py0 + 66}" font-family="{SANS}" font-size="14" fill="{INK_SOFT}">'
        f"95% interval {RETAINED_CHOICE_INTERVAL[0]:.2f}–{RETAINED_CHOICE_INTERVAL[1]:.2f}</text>"
    )
    # Interval bar (scale -0.25..1.5)
    def sx(v: float) -> float:
        return px0 + (v + 0.25) / 1.75 * 400
    by = py0 + 110
    parts.append(
        f'<line x1="{sx(-0.25)}" x2="{sx(1.5)}" y1="{by}" y2="{by}" stroke="#FFFFFF" stroke-opacity="0.18"/>'
        f'<line x1="{sx(0)}" x2="{sx(0)}" y1="{by - 12}" y2="{by + 12}" stroke="#FFFFFF" stroke-opacity="0.35"/>'
        f'<text x="{sx(0)}" y="{by + 28}" text-anchor="middle" font-family="{SANS}" font-size="11" fill="{INK_FAINT}">0</text>'
        f'<line x1="{sx(RETAINED_CHOICE_THRESHOLD)}" x2="{sx(RETAINED_CHOICE_THRESHOLD)}" y1="{by - 16}" y2="{by + 16}" '
        f'stroke="#B48CFF" stroke-width="2" stroke-dasharray="3 3"/>'
        f'<text x="{sx(RETAINED_CHOICE_THRESHOLD)}" y="{by - 22}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="11" fill="#B48CFF">0.2 bar</text>'
        f'<line x1="{sx(RETAINED_CHOICE_INTERVAL[0])}" x2="{sx(RETAINED_CHOICE_INTERVAL[1])}" y1="{by}" y2="{by}" '
        f'stroke="{ARI}" stroke-width="6" stroke-linecap="round"/>'
        f'<circle cx="{sx(mean)}" cy="{by}" r="8" fill="#FFE3BD"/>'
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
        parts.append(f'<text x="{px0}" y="{by + 66 + j * 22}" font-family="{SANS}" font-size="14" '
                     f'fill="{INK_SOFT}">{escape(line)}</text>')
    # Legend
    ly = top + row_h * len(pairs) + 24
    parts.append(
        f'<circle cx="90" cy="{ly}" r="8" fill="{BO}"/><text x="106" y="{ly + 5}" font-family="{SANS}" '
        f'font-size="14" fill="{INK}">learned “Bo is reliable”</text>'
        f'<circle cx="330" cy="{ly}" r="8" fill="{ARI}"/><text x="346" y="{ly + 5}" font-family="{SANS}" '
        f'font-size="14" fill="{INK}">learned “Ari is reliable”</text>'
        f'<text x="80" y="{ly + 40}" font-family="{SANS}" font-size="16" fill="{INK_FAINT}">'
        "Endpoint: partner balance of work requests at the first post-restart work opportunity</text>"
        f'<text x="80" y="{ly + 64}" font-family="{SANS}" font-size="16" fill="{INK_FAINT}">'
        "(+1 Ari, −1 Bo; 0 = no work, solo work, or balanced). Restart = same-process manager "
        "rebuild.</text>"
        f'<text x="80" y="{ly + 88}" font-family="{SANS}" font-size="16" fill="{INK_FAINT}">'
        "Full record: "
        "docs/plans/behavioral_validation_v4.md</text>"
    )
    style = """
.link { animation: grow 4s ease-out infinite; }
@keyframes grow { 0% { stroke-dasharray: 0 1; } 40%,100% { stroke-dasharray: 1 0; } }
@media (prefers-reduced-motion: reduce) { .link { animation: none; } }"""
    return _svg(
        RC_W, RC_H, "Same world. Opposite pasts. Different choices.",
        f"14 subject pairs (28 subjects) from the completed v4 confirmation. Mean paired difference "
        f"{mean:.2f}, 95% interval {RETAINED_CHOICE_INTERVAL[0]:.2f} to "
        f"{RETAINED_CHOICE_INTERVAL[1]:.2f}, above the 0.2 threshold; {zero} pairs show no difference.",
        "".join(parts), style,
    )


# ---------------------------------------------------------------------------
# 4. GitHub social cover (1280 x 640)
# ---------------------------------------------------------------------------

COVER_W, COVER_H = 1280, 640


def build_github_cover() -> str:
    minutes, states, _, _ = replay()
    x0, x1, y0, y1 = 610, 1230, 130, 500

    def pt(minute: float, value: float) -> tuple[float, float]:
        return x0 + (x1 - x0) * minute / DURATION_MIN, y1 - (y1 - y0) * value

    parts = [_defs(rook_defs("cover")), f'<rect width="{COVER_W}" height="{COVER_H}" fill="url(#bg)"/>',
             _stars(COVER_W, COVER_H, 80, 31)]
    for name, _, color in MODULATORS:
        pts = [pt(m, s[name]) for m, s in zip(minutes, states)]
        parts.append(f'<path d="{_line_path(pts)}" fill="none" stroke="{color}" stroke-width="3" '
                     f'stroke-opacity="0.8" stroke-linecap="round" filter="url(#glow)"/>')
    parts.append(
        '<circle cx="190" cy="250" r="160" fill="url(#coverHalo)"/>'
        + rook_mark(96, 118, 236, "cover", halo=False)
    )
    parts.append(
        f'<text x="64" y="452" font-family="{SERIF}" font-size="86" fill="{INK}">OpenRook</text>'
        f'<text x="68" y="500" font-family="{SANS}" font-size="24" fill="{INK_SOFT}">'
        "Persistent state for any LLM.</text>"
        f'<text x="68" y="534" font-family="{SANS}" font-size="24" fill="{INK_SOFT}">'
        "The moment passes. Its state carries on.</text>"
        f'<text x="68" y="596" font-family="{SANS}" font-size="15" fill="{INK_FAINT}">'
        "Open-source research runtime · functional state, not felt emotion</text>"
        f'<text x="{x0}" y="{y1 + 34}" font-family="{SANS}" font-size="14" fill="{INK_FAINT}">'
        "six modulators · real engine evaluation of an authored rupture → repair</text>"
    )
    return _svg(COVER_W, COVER_H, "OpenRook",
                "OpenRook social preview: rook mark with a lit window, wordmark, and six real modulator curves.",
                "".join(parts))


SHOWCASE = {
    "emotional-arc": build_emotional_arc,
    "emotional-fingerprints": build_emotional_fingerprints,
    "retained-choice": build_retained_choice,
    "github-cover": build_github_cover,
}
SIZES = {
    "emotional-arc": (ARC_W, ARC_H),
    "emotional-fingerprints": (FP_W, FP_H),
    "retained-choice": (RC_W, RC_H),
    "github-cover": (COVER_W, COVER_H),
}

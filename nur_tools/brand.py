"""OpenRook brand mark: a chess-rook keep with a lit window.

The rook is the name; the warm window is the state kept inside it. The six
modulator colors run as a thin band across the plinth. Everything here is
plain SVG so the same mark serves the README, the social cover, the web UI
header and the favicon.
"""

from __future__ import annotations

# Silhouette in a 100 x 120 box: three merlons, rim, tapered body, plinth.
ROOK_PATH = (
    "M20,8 H36 V17 H43 V8 H57 V17 H64 V8 H80 V34 L72,42 L70,86 L78,92 V98 "
    "L84,102 V112 H16 V102 L22,98 V92 L30,86 L28,42 L20,34 Z"
)
# Arched arrow-slit window in the body.
WINDOW_PATH = "M43.5,77 V58 A6.5,6.5 0 0 1 56.5,58 V77 Z"

MODULATOR_COLORS = ("#FF6B6B", "#F5A65B", "#5BC0EB", "#E86FB0", "#7BD389", "#B48CFF")

SERIF = "'EB Garamond','Cormorant Garamond','Iowan Old Style','Palatino Linotype',Georgia,'DejaVu Serif',serif"


def rook_defs(prefix: str = "rook", *, halo: bool = True) -> str:
    """Gradients and filters used by :func:`rook_mark` (ids namespaced by prefix)."""
    halo_def = (
        f'<radialGradient id="{prefix}Halo" cx="50%" cy="50%" r="50%">'
        '<stop offset="0" stop-color="#F5A65B" stop-opacity="0.5"/>'
        '<stop offset="0.6" stop-color="#F5A65B" stop-opacity="0.12"/>'
        '<stop offset="1" stop-color="#F5A65B" stop-opacity="0"/></radialGradient>'
    ) if halo else ""
    return (
        f'<linearGradient id="{prefix}Body" x1="0" y1="0" x2="0.35" y2="1">'
        '<stop offset="0" stop-color="#FFD9A8"/><stop offset="0.38" stop-color="#F5A65B"/>'
        '<stop offset="0.78" stop-color="#9A5A9E"/><stop offset="1" stop-color="#4A2E63"/>'
        f"</linearGradient>{halo_def}"
        f'<radialGradient id="{prefix}Light" cx="50%" cy="55%" r="60%">'
        '<stop offset="0" stop-color="#FFFFFF"/><stop offset="0.45" stop-color="#FFF1D6"/>'
        '<stop offset="1" stop-color="#FFC27A"/></radialGradient>'
        f'<filter id="{prefix}Glow" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="3.2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        f'<clipPath id="{prefix}Clip"><path d="{ROOK_PATH}"/></clipPath>'
    )


def rook_mark(
    x: float,
    y: float,
    height: float,
    prefix: str = "rook",
    *,
    halo: bool = True,
) -> str:
    """The rook mark with its top-left corner at (x, y), scaled to ``height``."""
    scale = height / 120
    band_w = 68 / len(MODULATOR_COLORS)
    band = "".join(
        f'<rect x="{16 + i * band_w:.2f}" y="99" width="{band_w + 0.6:.2f}" '
        f'height="3.2" fill="{color}"/>'
        for i, color in enumerate(MODULATOR_COLORS)
    )
    parts = [f'<g transform="translate({x:.1f} {y:.1f}) scale({scale:.4f})">']
    if halo:
        parts.append(f'<circle cx="50" cy="62" r="78" fill="url(#{prefix}Halo)"/>')
    parts.append(
        f'<path d="{ROOK_PATH}" fill="url(#{prefix}Body)"/>'
        f'<g clip-path="url(#{prefix}Clip)">{band}'
        '<rect x="0" y="34" width="100" height="2.2" fill="#FFFFFF" opacity="0.18"/></g>'
        f'<path d="{WINDOW_PATH}" fill="#2A1E3F"/>'
        f'<path class="rook-window" d="{WINDOW_PATH}" fill="url(#{prefix}Light)" '
        f'filter="url(#{prefix}Glow)" transform="translate(50 67) scale(0.84) translate(-50 -67)"/>'
        "</g>"
    )
    return "".join(parts)


def rook_icon_svg(size: int = 64, background: str | None = "#1A1428") -> str:
    """Standalone square icon (favicon / app icon)."""
    bg = (f'<rect width="120" height="120" rx="26" fill="{background}"/>' if background else "")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" '
        f'width="{size}" height="{size}" role="img" aria-label="OpenRook">'
        f"<defs>{rook_defs('fav', halo=False)}</defs>{bg}"
        f"{rook_mark(14, 8, 104, 'fav', halo=False)}</svg>\n"
    )


def wordmark_svg(text_color: str) -> str:
    """README wordmark: rook mark + serif name."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 160" width="720" height="160" '
        'role="img" aria-label="OpenRook">'
        f"<defs>{rook_defs('wm', halo=False)}</defs>"
        f"{rook_mark(30, 14, 132, 'wm', halo=False)}"
        f'<text x="158" y="108" font-family="{SERIF}" font-size="96" font-weight="500" '
        f'fill="{text_color}" letter-spacing="-3">OpenRook</text></svg>\n'
    )

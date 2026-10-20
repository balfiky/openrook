#!/usr/bin/env python3
"""Regenerate the architecture diagrams as SVG + PNG under docs/diagrams/.

SVG sources are built from ``nur_tools/build_diagrams.py`` (eight structural
figures) and ``nur_tools/build_showcase.py`` (data-driven README visuals).
All diagrams have transparent ``-light`` and ``-dark`` SVG + PNG variants,
selected in documentation with theme-aware picture elements. CairoSVG renders
PNGs at 2x device pixels; the single opaque social cover stays at 1280 x 640.
Both SVGs and PNG previews are tracked. Edit the Python builders, then regenerate.

Usage::

    python3 nur_tools/render_diagram_pngs.py

Requires ``cairosvg`` (listed in the dev extras) and DejaVu Sans, DejaVu Serif,
and DejaVu Sans Mono fonts.
Font substitution can change line widths or omit glyphs; inspect every PNG
after rendering. See CONTRIBUTING.md for the font and editing workflow.
"""

from __future__ import annotations

import sys

# Direct script execution puts nur_tools/ first, where types.py can shadow the
# standard library while pathlib imports. Restore the project root below.
if __package__ is None:
    sys.path.pop(0)

from pathlib import Path

import cairosvg

# Make nur_tools importable when running the script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nur_tools.build_diagrams import BUILDERS  # noqa: E402
from nur_tools.build_showcase import SHOWCASE, SIZES  # noqa: E402
from nur_tools.brand import THEMES, rook_icon_svg, wordmark_svg  # noqa: E402


OUT_DIR = ROOT / "docs" / "diagrams"
STATIC_DIR = ROOT / "interface" / "static"
PNG_SCALE = 2  # retina-style preview for README/docs


def render_one(name: str) -> None:
    for theme in THEMES:
        canvas = BUILDERS[name](theme)
        svg_text = canvas.to_svg()
        svg_path = OUT_DIR / f"{name}-{theme.name}.svg"
        png_path = svg_path.with_suffix(".png")
        svg_path.write_text(svg_text, encoding="utf-8")
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(png_path),
            output_width=canvas.width * PNG_SCALE,
            output_height=canvas.height * PNG_SCALE,
        )
        print(f"wrote {svg_path.name} and {png_path.name}")


def render_showcase(name: str) -> None:
    """Render every registered theme, retaining each SVG's canvas transparency."""
    svg_text = SHOWCASE[name]()
    width, height = SIZES[name]
    scale = 1 if name == "github-cover" else PNG_SCALE
    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    svg_path.write_text(svg_text, encoding="utf-8")
    cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        write_to=str(png_path),
        output_width=width * scale,
        output_height=height * scale,
    )
    print(f"wrote {svg_path.name} and {png_path.name}")


def render_brand() -> None:
    """Wordmarks for the README and the rook icon for the web UI and favicon."""
    (OUT_DIR / "wordmark-light.svg").write_text(wordmark_svg("#1F1830"), encoding="utf-8")
    (OUT_DIR / "wordmark-dark.svg").write_text(wordmark_svg("#E6EDF3"), encoding="utf-8")
    (STATIC_DIR / "openrook-icon.svg").write_text(rook_icon_svg(64), encoding="utf-8")
    (STATIC_DIR / "openrook-mark.svg").write_text(rook_icon_svg(64, background=None), encoding="utf-8")
    print("wrote wordmarks and web UI rook icons")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_brand()
    for name in BUILDERS:
        render_one(name)
    for name in SHOWCASE:
        render_showcase(name)



if __name__ == "__main__":
    main()

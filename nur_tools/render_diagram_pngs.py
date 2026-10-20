#!/usr/bin/env python3
"""Regenerate the architecture diagrams as SVG + PNG under docs/diagrams/.

SVG sources are built from ``nur_tools/build_diagrams.py`` (structural diagrams)
and ``nur_tools/build_showcase.py`` (data-driven README visuals); PNGs are
rasterized with cairosvg at 2x device pixels (the social cover stays 1280 x 640).
The hand-authored ``docs/diagrams/feature-overview.svg`` is also rasterized;
edit that SVG directly, as it has no builder. Both formats are tracked in git — SVGs are the
source of truth that GitHub and local HTML export embed, and PNGs are the
pre-rendered preview that the README and docs use so contributors don't
need cairosvg installed just to read the docs.

Usage::

    python3 nur_tools/render_diagram_pngs.py

Requires ``cairosvg`` (listed in the dev extras) and DejaVu Sans fonts.
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


OUT_DIR = ROOT / "docs" / "diagrams"
PNG_SCALE = 2  # retina-style preview for README/docs
STATIC_SVGS = ("feature-overview",)


def render_one(name: str) -> None:
    canvas = BUILDERS[name]()
    svg_text = canvas.to_svg()
    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    svg_path.write_text(svg_text, encoding="utf-8")
    scale = PNG_SCALE
    cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        write_to=str(png_path),
        output_width=canvas.width * scale,
        output_height=canvas.height * scale,
    )
    print(f"wrote {svg_path.name} and {png_path.name}")


def render_showcase(name: str) -> None:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in BUILDERS:
        render_one(name)
    for name in SHOWCASE:
        render_showcase(name)
    for name in STATIC_SVGS:
        svg_path = OUT_DIR / f"{name}.svg"
        png_path = svg_path.with_suffix(".png")
        cairosvg.svg2png(
            url=str(svg_path), write_to=str(png_path), scale=PNG_SCALE,
        )
        print(f"rendered hand-authored {svg_path.name} to {png_path.name}")


if __name__ == "__main__":
    main()

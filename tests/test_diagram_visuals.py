"""Documentation diagrams are reproducible, accessible, and paired by theme."""

from __future__ import annotations

import html
from math import cos, hypot, sin
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import zlib

import pytest

from nur_tools.brand import DARK, LIGHT, THEMES
from nur_tools.build_diagrams import BUILDERS
from nur_tools.diagram_toolkit import Canvas, loop_geometry

SVG_NS = "{http://www.w3.org/2000/svg}"
DIAGRAM_DIR = Path("docs/diagrams")
DIAGRAM_TYPES = {
    "research-runtime-autonomy": {"loop"},
    "runtime-architecture": {"architecture"},
    "single-turn-cognitive-flow": {"swimlane", "process"},
    "memory-persistence-model": {"nested"},
    "auth-tool-safety-boundary": {"flowchart"},
    "evaluation-ablation-harness": {"data flow", "data-flow", "process"},
    "component-claim-map": {"matrix", "heatmap"},
    "feature-overview": {"layer stack", "layers"},
}
MARKDOWN_FILES = [Path("README.md"), Path("CONTRIBUTING.md"), *Path("docs").rglob("*.md")]


def _without_paint(svg: str, name: str, theme: str) -> str:
    """Compare semantic content and geometry while allowing theme paint and IDs."""
    root = ET.fromstring(svg)
    for node in root.iter():
        for key, value in node.attrib.items():
            value = value.replace(f"{name}-{theme}", name)
            if key in {"fill", "stroke", "stop-color", "color"}:
                value = re.sub(r"#[0-9a-fA-F]{6}\b", "PAINT", value)
            node.set(key, value)
    return ET.tostring(root, encoding="unicode")


def _first_pixel_alpha(png: bytes) -> int:
    """Read the transparent canvas corner without a rendering dependency."""
    compressed = []
    offset = 8
    while offset < len(png):
        length = struct.unpack(">I", png[offset:offset + 4])[0]
        if png[offset + 4:offset + 8] == b"IDAT":
            compressed.append(png[offset + 8:offset + 8 + length])
        offset += length + 12
    # For an 8-bit RGBA PNG, all five filter types leave the first pixel of
    # the first scanline unchanged: its left/upper neighbors are zero.
    scanline = zlib.decompressobj().decompress(b"".join(compressed), 5)
    assert len(scanline) == 5 and scanline[0] in range(5)
    return scanline[4]


def test_all_eight_diagrams_have_the_requested_visual_type():
    assert set(BUILDERS) == set(DIAGRAM_TYPES)
    assert THEMES == (LIGHT, DARK)
    for name, kinds in DIAGRAM_TYPES.items():
        canvas = BUILDERS[name]()
        assert canvas.slug == name
        assert canvas.kind in kinds
        assert canvas.theme == LIGHT


@pytest.mark.parametrize("name", DIAGRAM_TYPES)
@pytest.mark.parametrize("theme", THEMES, ids=lambda theme: theme.name)
def test_diagram_builder_is_deterministic_and_matches_committed_svg(name, theme):
    first = BUILDERS[name](theme).to_svg()
    assert BUILDERS[name](theme).to_svg() == first
    assert (DIAGRAM_DIR / f"{name}-{theme.name}.svg").read_text(encoding="utf-8") == first


@pytest.mark.parametrize("name", DIAGRAM_TYPES)
@pytest.mark.parametrize("theme", THEMES, ids=lambda theme: theme.name)
def test_diagram_svg_is_accessible_and_has_valid_scoped_references(name, theme):
    canvas = BUILDERS[name](theme)
    svg = canvas.to_svg()
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("role") == "img"
    assert root.get("viewBox") == f"0 0 {canvas.width} {canvas.height}"
    assert [node.tag for node in root[:2]] == [f"{SVG_NS}title", f"{SVG_NS}desc"]
    assert all(node.text and node.text.strip() for node in root[:2])
    assert root.get("aria-labelledby", "").split() == [
        root[0].get("id"), root[1].get("id"),
    ]
    ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
    assert len(ids) == len(set(ids))
    assert all(identifier.startswith(f"{name}-{theme.name}-") for identifier in ids)
    assert all(re.fullmatch(r"[A-Za-z_][\w.-]*", identifier) for identifier in ids)
    assert set(re.findall(r"url\(#([^)]+)\)", svg)).issubset(ids)
    assert root.find(f".//{SVG_NS}script") is None
    assert root.find(f".//{SVG_NS}filter") is None
    assert not any("filter" in node.attrib for node in root.iter())
    for rect in root.findall(f".//{SVG_NS}rect"):
        assert float(rect.get("rx", "0")) <= 8
        at_origin = float(rect.get("x", "0")) == float(rect.get("y", "0")) == 0
        full_width = rect.get("width") in {str(canvas.width), "100%"}
        full_height = rect.get("height") in {str(canvas.height), "100%"}
        assert not (at_origin and full_width and full_height), "Canvas must stay transparent"


@pytest.mark.parametrize("name", DIAGRAM_TYPES)
def test_theme_variants_share_text_and_geometry_with_disjoint_ids(name):
    variants = {theme.name: BUILDERS[name](theme).to_svg() for theme in THEMES}
    assert _without_paint(variants["light"], name, "light") == _without_paint(
        variants["dark"], name, "dark",
    )
    ids = [
        {node.attrib["id"] for node in ET.fromstring(svg).iter() if "id" in node.attrib}
        for svg in variants.values()
    ]
    assert ids[0].isdisjoint(ids[1])


@pytest.mark.parametrize("name", DIAGRAM_TYPES)
@pytest.mark.parametrize("theme", THEMES, ids=lambda theme: theme.name)
def test_png_previews_are_transparent_at_double_resolution(name, theme):
    canvas = BUILDERS[name](theme)
    png = (DIAGRAM_DIR / f"{name}-{theme.name}.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert png[12:16] == b"IHDR"
    assert struct.unpack(">II", png[16:24]) == (canvas.width * 2, canvas.height * 2)
    assert png[24:26] == bytes((8, 6)), "PNG must preserve 8-bit RGBA transparency"
    assert _first_pixel_alpha(png) == 0


@pytest.mark.parametrize("name", DIAGRAM_TYPES)
def test_single_variant_assets_and_documentation_references_are_retired(name):
    for suffix in (".svg", ".png"):
        assert not (DIAGRAM_DIR / f"{name}{suffix}").exists()
    old_reference = re.compile(rf"(?<![\w-]){re.escape(name)}\.(?:svg|png)\b")
    reference_files = [*MARKDOWN_FILES, *Path("nur_tools").glob("*.py"), *Path("tests").glob("*.py")]
    for path in reference_files:
        assert not old_reference.search(path.read_text(encoding="utf-8")), str(path)


def test_documentation_embeds_each_diagram_with_paired_theme_sources():
    pictured = set()
    for path in MARKDOWN_FILES:
        document = path.read_text(encoding="utf-8")
        # CONTRIBUTING includes an illustrative fenced picture, not an embed.
        document = re.sub(r"```.*?```", "", document, flags=re.DOTALL)
        for picture in re.findall(r"<picture\b[^>]*>.*?</picture>", document, re.DOTALL):
            for name in DIAGRAM_TYPES:
                if f"{name}-light.svg" not in picture:
                    continue
                pictured.add(name)
                sources = re.findall(r"<source\b[^>]+>", picture)
                assert len(sources) == 1, (path, name)
                assert 'media="(prefers-color-scheme: dark)"' in sources[0], (path, name)
                dark = re.search(r'\bsrcset="([^"]+)"', sources[0]).group(1)
                img = re.search(r"<img\b[^>]+>", picture).group()
                light = re.search(r'\bsrc="([^"]+)"', img).group(1)
                assert dark == light.removesuffix("-light.svg") + "-dark.svg", (path, name)
                assert light.endswith(f"{name}-light.svg")
                alt = re.search(r'\balt="([^"]+)"', img).group(1)
                assert html.unescape(alt).strip(), (path, name)
                assert (path.parent / light).is_file(), (path, light)
                assert (path.parent / dark).is_file(), (path, dark)
    assert pictured == set(DIAGRAM_TYPES)


def test_orthogonal_primitives_reject_diagonal_segments():
    canvas = Canvas(100, 100, "Test", "Test")
    with pytest.raises(ValueError, match="orthogonal"):
        canvas.connector([(0, 0), (10, 10)])
    with pytest.raises(ValueError, match="orthogonal"):
        canvas.line(0, 0, 10, 10)
    with pytest.raises(ValueError, match="at least two points"):
        canvas.connector([(0, 0)])
    assert canvas.pieces == []
    canvas.connector([(0, 0), (20, 0), (20, 30)])
    root = ET.fromstring(canvas.to_svg())
    connector = root.find(f".//{SVG_NS}path[@data-connector='orthogonal']")
    assert connector is not None
    assert " A" in connector.get("d"), "Elbows must use true arcs"


@pytest.mark.parametrize("count", range(5, 9))
def test_parametric_loop_stations_interrupt_one_circle_at_their_edges(count):
    cx, cy, radius = 700, 608, 376
    boxes, angles = loop_geometry(cx, cy, radius, count, 248, 88)
    assert len(boxes) == len(angles) == count
    endpoints = []
    for (x, y, width, height), (entry, exit_) in zip(boxes, angles):
        assert x % 4 == y % 4 == 0
        assert entry < exit_
        for angle in (entry, exit_):
            px, py = cx + radius * cos(angle), cy + radius * sin(angle)
            assert hypot(px - cx, py - cy) == pytest.approx(radius)
            assert x - 1e-6 <= px <= x + width + 1e-6
            assert y - 1e-6 <= py <= y + height + 1e-6
            assert min(abs(px - x), abs(px - x - width),
                       abs(py - y), abs(py - y - height)) < 1e-6
            endpoints.append((round(px, 6), round(py, 6)))
    assert len(set(endpoints)) == count * 2, "Each station has distinct entry and exit ports"


@pytest.mark.parametrize("count", (0, 4, 9))
def test_parametric_loop_rejects_station_counts_outside_the_design_budget(count):
    with pytest.raises(ValueError, match="five to eight"):
        loop_geometry(700, 608, 376, count, 248, 88)


def test_autonomy_loop_has_eight_true_arc_segments_and_one_focal_station():
    canvas = BUILDERS["research-runtime-autonomy"]()
    root = ET.fromstring(canvas.to_svg())
    stations = root.findall(f".//{SVG_NS}g[@data-node]")
    assert len(stations) == 8
    assert [station.get("data-node") for station in stations] == [
        "Wake", "Deliberate", "Choose goal", "Act", "Record receipt",
        "Verify claim", "Learn & appraise", "Checkpoint & rest",
    ]
    focal = [station for station in stations if station.get("data-focal") == "true"]
    assert len(focal) == 1 and focal[0].get("data-node") == "Verify claim"
    assert focal[0].get("data-role") == "optional"
    assert root.find(f".//{SVG_NS}rect[@data-role='hub']") is not None
    ring = root.findall(f".//{SVG_NS}path[@data-connector='ring']")
    assert len(ring) == 8
    arc_pattern = re.compile(
        r"M(-?[\d.]+),(-?[\d.]+) A([\d.]+),([\d.]+) 0 0 1 (-?[\d.]+),(-?[\d.]+)"
    )
    radii = set()
    for segment in ring:
        match = arc_pattern.fullmatch(segment.get("d"))
        assert match, "Ring segments must be single true circular arcs without slants"
        _, _, rx, ry, _, _ = map(float, match.groups())
        assert rx == ry
        radii.add(rx)
    assert len(radii) == 1, "The entire loop must share one radius"
    content = " ".join(node.text or "" for node in root.findall(f".//{SVG_NS}text"))
    assert "No claim of experienced emotion" in content
    assert "read-only" in content and "never advances" in content

"""The README showcase visuals must stay tied to real engine output and study data."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from nur_tools.build_showcase import (
    RETAINED_CHOICE_INTERVAL,
    RETAINED_CHOICE_PAIRS,
    SHOWCASE,
    SIZES,
    replay,
)

SVG_NS = "{http://www.w3.org/2000/svg}"
THEMED_VISUALS = (
    "how-it-works",
    "emotional-arc",
    "emotional-fingerprints",
    "retained-choice",
)
ANALYSIS = Path(
    "reports/research-comparison/20261019T170937Z-retained-choice-v4-replacement/study/analysis.json"
)


def test_retained_choice_pairs_reproduce_published_result():
    differences = [ari - bo for _, bo, ari in RETAINED_CHOICE_PAIRS]
    assert len(differences) == 14
    assert sum(differences) / len(differences) == pytest.approx(0.7142857142857143)
    assert sum(1 for d in differences if d == 0) == 7
    assert RETAINED_CHOICE_INTERVAL == pytest.approx((0.2857142857142857, 1.1428571428571428))


@pytest.mark.skipif(not ANALYSIS.exists(), reason="local study evidence is not distributed")
def test_retained_choice_pairs_match_local_analysis():
    analysis = json.loads(ANALYSIS.read_text())
    recorded = tuple(
        (pair["baseline"]["environment_seed"], pair["baseline"]["value"], pair["candidate"]["value"])
        for pair in analysis["pairs"]
    )
    assert recorded == RETAINED_CHOICE_PAIRS
    assert tuple(analysis["interval"]) == pytest.approx(RETAINED_CHOICE_INTERVAL)


def test_engine_replay_shows_rupture_and_repair():
    _, states, labels, moments = replay()
    assert moments["rupture"]["resolution"] > 0.5
    assert moments["repair"]["resolution"] == 0.0
    assert moments["rupture"]["arousal"] > moments["calm"]["arousal"]
    assert moments["rupture"]["valence"] < moments["warmth"]["valence"]
    assert all(0.0 <= value <= 1.0 for state in states for value in state.values())
    assert "anxious" in labels


@pytest.mark.parametrize("name", sorted(SHOWCASE))
def test_showcase_svgs_are_accessible_and_reduce_motion(name):
    svg = SHOWCASE[name]()
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("role") == "img"
    assert [child.tag for child in root[:2]] == [f"{SVG_NS}title", f"{SVG_NS}desc"]
    labels = root.get("aria-labelledby", "").split()
    assert labels == [root[0].get("id"), root[1].get("id")]
    assert all(child.text and child.text.strip() for child in root[:2])
    assert root.get("viewBox") == f"0 0 {SIZES[name][0]} {SIZES[name][1]}"
    ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
    assert len(ids) == len(set(ids))
    assert set(re.findall(r"url\(#([^)]+)\)", svg)).issubset(ids)
    assert root.find(f".//{SVG_NS}script") is None
    assert all(
        float(text.attrib["font-size"]) >= 18
        for text in root.findall(f".//{SVG_NS}text")
    )
    for node in root.iter():
        if "filter" in node.attrib:
            assert "rook-window" in node.get("class", "").split()
    if "@keyframes" in svg:
        assert "prefers-reduced-motion" in svg
    committed = Path("docs/diagrams") / f"{name}.svg"
    assert committed.read_text(encoding="utf-8") == svg


@pytest.mark.parametrize("name", THEMED_VISUALS)
def test_both_themes_have_transparent_canvases_and_scoped_accessibility(name):
    roots = [
        ET.fromstring(SHOWCASE[f"{name}-{theme}"]())
        for theme in ("light", "dark")
    ]
    assert [node.text for node in roots[0][:2]] == [node.text for node in roots[1][:2]]
    ids = [
        {node.attrib["id"] for node in root.iter() if "id" in node.attrib}
        for root in roots
    ]
    assert ids[0].isdisjoint(ids[1])
    for theme, root in zip(("light", "dark"), roots):
        width, height = SIZES[f"{name}-{theme}"]
        for rect in root.findall(f".//{SVG_NS}rect"):
            full_width = rect.get("width") in (str(width), "100%")
            full_height = rect.get("height") in (str(height), "100%")
            at_origin = rect.get("x", "0") == "0" and rect.get("y", "0") == "0"
            assert not (full_width and full_height and at_origin), "Canvas must be transparent"
    assert not (Path("docs/diagrams") / f"{name}.svg").exists()
    assert not (Path("docs/diagrams") / f"{name}.png").exists()


def test_modulator_colors_contrast_against_theme_backgrounds():
    from nur_tools.build_showcase import THEMES

    def luminance(hex_color):
        channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    for theme in THEMES:
        for color in theme.colors:
            for background in (theme.page, theme.panel):
                light, dark = sorted((luminance(color), luminance(background)), reverse=True)
                contrast = (light + 0.05) / (dark + 0.05)
                assert contrast >= 3, f"{theme.name}: {color} on {background} is {contrast:.2f}:1"


def test_social_cover_is_one_light_1280_by_640_image():
    from nur_tools.build_showcase import LIGHT

    root = ET.fromstring(SHOWCASE["github-cover"]())
    assert SIZES["github-cover"] == (1280, 640)
    backgrounds = [
        rect for rect in root.findall(f".//{SVG_NS}rect")
        if rect.get("width") == "1280" and rect.get("height") == "640"
    ]
    assert len(backgrounds) == 1 and backgrounds[0].get("fill") == LIGHT.page
    text = " ".join(node.text or "" for node in root.findall(f".//{SVG_NS}text"))
    assert "OpenRook" in text
    assert "LLMs forget. OpenRook remembers." in text
    png = Path("docs/diagrams/github-cover.png").read_bytes()
    assert struct.unpack(">II", png[16:24]) == (1280, 640)


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_how_it_works_is_readable_without_animation(theme):
    root = ET.fromstring(SHOWCASE[f"how-it-works-{theme}"]())
    steps = root.findall(f".//{SVG_NS}g[@data-step]")
    assert [step.get("data-step") for step in steps] == ["1", "2", "3"]
    for step in steps:
        assert "how-step" in step.get("class", "").split()
        assert step.get("opacity", "1") == "1"
        assert step.get("display") != "none"
        assert step.get("visibility") != "hidden"
        assert "style" not in step.attrib
    assert root.find(f".//{SVG_NS}script") is None
    texts = root.findall(f".//{SVG_NS}text")
    assert texts

    style = root.find(f"{SVG_NS}style").text
    assert "infinite" not in style
    media_blocks = re.findall(r"@media\s*([^{}]+)\{((?:[^{}]|\{[^{}]*\})*)\}", style)
    animated = [body for media, body in media_blocks if "no-preference" in media]
    assert len(animated) == 1
    assert ".how-step" in animated[0] and "animation:" in animated[0]
    for condition in ("prefers-reduced-motion: reduce", "print"):
        static = [body for media, body in media_blocks if condition in media]
        assert static, f"Missing static presentation for {condition}"
        assert any(
            ".how-step" in body
            and re.search(r"animation:\s*none\b", body)
            and re.search(r"opacity:\s*1\b", body)
            for body in static
        )


@pytest.mark.parametrize("theme", ("light", "dark"))
@pytest.mark.parametrize("name", ("emotional-arc", "retained-choice"))
def test_chart_animation_is_optional_and_static_chart_is_complete(name, theme):
    root = ET.fromstring(SHOWCASE[f"{name}-{theme}"]())
    style = root.find(f"{SVG_NS}style").text
    assert "infinite" not in style
    media_blocks = re.findall(r"@media\s*([^{}]+)\{((?:[^{}]|\{[^{}]*\})*)\}", style)
    selector = ".reveal" if name == "emotional-arc" else ".link"
    animated = [body for media, body in media_blocks if "no-preference" in media]
    assert len(animated) == 1 and selector in animated[0] and "animation:" in animated[0]
    for condition in ("prefers-reduced-motion: reduce", "print"):
        static = [body for media, body in media_blocks if condition in media]
        assert any(
            selector in body and re.search(r"animation:\s*none\b", body)
            for body in static
        ), f"Missing static presentation for {condition}"

    if name == "emotional-arc":
        mask = root.find(f".//{SVG_NS}mask")
        # CairoSVG clipped complete paths near minute 51 with implicit object
        # bounds. Explicit canvas bounds preserve the full 90-minute PNG chart.
        assert mask.get("maskUnits") == "userSpaceOnUse"
        assert mask.get("x") == mask.get("y") == "0"
        width, height = SIZES[f"{name}-{theme}"]
        assert mask.get("width") == str(width)
        assert mask.get("height") == str(height)
        reveal = mask.find(f"{SVG_NS}rect")
        assert reveal.get("fill") == "#FFFFFF"
        assert reveal.get("transform") is None
        assert reveal.get("opacity", "1") == "1"
        curves = root.find(f".//{SVG_NS}g[@mask]")
        assert len(curves) == 6
        playhead = root.find(f".//{SVG_NS}line[@class='playhead']")
        assert playhead.get("opacity") == "0"
    else:
        curves = root.findall(f".//{SVG_NS}path[@class='link']")
        assert curves
    for curve in curves:
        assert curve.get("display") != "none"
        assert curve.get("visibility") != "hidden"
        assert curve.get("opacity", "1") == "1"
        assert curve.get("stroke-dasharray") is None


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_fingerprints_are_static(theme):
    svg = SHOWCASE[f"emotional-fingerprints-{theme}"]()
    assert "@keyframes" not in svg
    assert "animation:" not in svg
    root = ET.fromstring(svg)
    assert not any(
        node.tag in {f"{SVG_NS}animate", f"{SVG_NS}animateMotion", f"{SVG_NS}animateTransform"}
        for node in root.iter()
    )


@pytest.mark.parametrize("name", THEMED_VISUALS)
def test_readme_pictures_follow_system_theme_with_light_default_and_png_fallback(name):
    readme = Path("README.md").read_text(encoding="utf-8")
    pictures = re.findall(r"<picture>.*?</picture>", readme, flags=re.DOTALL)
    picture = next(picture for picture in pictures if f"{name}-light.svg" in picture)
    source = re.search(r"<source\b[^>]+>", picture).group()
    assert 'media="(prefers-color-scheme: dark)"' in source
    assert f'srcset="docs/diagrams/{name}-dark.svg"' in source
    img = re.search(r"<img\b[^>]+>", picture).group()
    assert f'src="docs/diagrams/{name}-light.svg"' in img
    alt = re.search(r'\balt="([^"]+)"', img).group(1)
    assert html.unescape(alt).strip()
    png = (Path("docs/diagrams") / f"{name}-light.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert png[12:16] == b"IHDR"
    assert struct.unpack(">II", png[16:24]) == tuple(
        size * 2 for size in SIZES[f"{name}-light"]
    )


@pytest.mark.parametrize("name", ("how-it-works", "emotional-arc"))
def test_readme_alt_text_matches_computed_description(name):
    from nur_tools.build_showcase import arc_description, how_it_works_description

    description = {
        "how-it-works": how_it_works_description,
        "emotional-arc": arc_description,
    }[name]
    readme = Path("README.md").read_text(encoding="utf-8")
    img = re.search(rf'<img\b[^>]*src="docs/diagrams/{name}-light.svg"[^>]*>', readme).group()
    alt = re.search(r'\balt="([^"]+)"', img).group(1)
    assert html.unescape(alt) == description()


def test_showcase_engine_ignores_user_config_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROOK_CONFIG_DIR", str(tmp_path))
    (tmp_path / "modulators.yaml").write_text(
        "half_lives:\n  arousal: 1.0\nbaselines:\n  arousal: 0.1\n", encoding="utf-8",
    )
    # Both get_config() and engine defaults are cached, so import them in a fresh process.
    result = subprocess.run(
        [sys.executable, "-c", """
import json
from config.loader import get_config

config = get_config()
assert config.half_lives["arousal"] == 1.0
assert config.baselines["arousal"] == 0.1
from nur_tools.build_showcase import replay
print(json.dumps(replay()[1]))
"""],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == replay()[1]


def test_brand_assets_are_generated_rook_marks():
    from nur_tools.brand import ROOK_PATH, rook_icon_svg, wordmark_svg

    static = Path("interface/static")
    assert (static / "openrook-icon.svg").read_text(encoding="utf-8") == rook_icon_svg(64)
    assert (static / "openrook-mark.svg").read_text(encoding="utf-8") == rook_icon_svg(64, background=None)
    for name, color in (("wordmark-light", "#1F1830"), ("wordmark-dark", "#E6EDF3")):
        svg = (Path("docs/diagrams") / f"{name}.svg").read_text(encoding="utf-8")
        assert svg == wordmark_svg(color) and ROOK_PATH in svg

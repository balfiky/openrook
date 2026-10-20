"""The README showcase visuals must stay tied to real engine output and study data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nur_tools.build_showcase import (
    RETAINED_CHOICE_INTERVAL,
    RETAINED_CHOICE_PAIRS,
    SHOWCASE,
    replay,
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
    assert svg.startswith("<svg") and "<title" in svg and "<desc" in svg
    if "@keyframes" in svg:
        assert "prefers-reduced-motion" in svg
    committed = Path("docs/diagrams") / f"{name}.svg"
    assert committed.read_text(encoding="utf-8") == svg


def test_readme_hero_alt_text_matches_computed_description():
    import html

    from nur_tools.build_showcase import arc_description

    readme = Path("README.md").read_text(encoding="utf-8")
    alt = readme.split('src="docs/diagrams/emotional-arc.svg" alt="', 1)[1].split('"', 1)[0]
    assert html.unescape(alt) == arc_description()


def test_showcase_engine_ignores_user_config_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROOK_CONFIG_DIR", str(tmp_path))
    (tmp_path / "modulators.yaml").write_text("half_lives:\n  arousal: 1.0\n")
    _, states, _, _ = replay()
    assert states[0]["arousal"] == pytest.approx(0.5)
    assert max(state["arousal"] for state in states) > 0.8

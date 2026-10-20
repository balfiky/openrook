"""Operator setup surfaces must not recommend or prefill a particular LLM."""

from pathlib import Path
import re

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
# Keep this guard on current operator-facing surfaces. CHANGELOG.md,
# docs/research/, and reports/ record historical results and are not suggestions.
MODEL_CHOICE_FILES = sorted(
    [
        ROOT / "interface/setup.py",
        ROOT / "runtime_config.example.yaml",
        ROOT / "README.md",
        ROOT / "runtime/llm/discovery.py",
        ROOT / "runtime/llm/diagnostics.py",
        *(ROOT / "interface/static").rglob("*.js"),
        *(ROOT / "docs").glob("*.md"),
    ]
)
STALE_MODEL_NAMES = re.compile(r"llama3\.2|gpt-4o-mini|MiniMax-M2\.7", re.IGNORECASE)


@pytest.mark.parametrize(
    "path", MODEL_CHOICE_FILES, ids=lambda path: str(path.relative_to(ROOT))
)
def test_setup_and_current_docs_do_not_suggest_stale_models(path):
    matches = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if STALE_MODEL_NAMES.search(line)
    ]
    assert not matches, "\n".join(matches)


def test_starter_config_requires_a_model_choice():
    config = yaml.safe_load((ROOT / "runtime_config.example.yaml").read_text(encoding="utf-8"))
    assert config["llm_model"] == ""


def test_wizard_presets_do_not_supply_model_values_or_placeholders():
    source = (ROOT / "interface/static/chat.js").read_text(encoding="utf-8")
    presets = re.search(r"const WIZARD_PRESET_CONFIG\s*=\s*(\{.*?\n\});", source, re.DOTALL)
    assert presets, "Expected the wizard backend preset configuration"
    assert not re.search(r"\bmodel(?:_placeholder)?[\"']?\s*:", presets.group(1))

"""Protect semantic foreground/background pairs shared by all UI surfaces."""

from pathlib import Path
import re

import pytest


STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
TOKENS = (STATIC / "tokens.css").read_text()


def declarations(block):
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", block))


def theme_tokens(theme):
    base, auto = TOKENS.split("@media (prefers-color-scheme: light)")
    result = {}
    for block in re.findall(r":root\s*\{([^{}]*)\}", base):
        result.update(declarations(block))
    if theme == "light":
        result.update(declarations(re.search(r'\[data-theme="light"\]\s*\{([^{}]*)\}', base)[1]))
    elif theme == "auto-light":
        result.update(declarations(auto))
    return result


def resolve(tokens, name):
    value = tokens[name].strip()
    while value.startswith("var("):
        value = tokens[value[4:-1]].strip()
    return value


def luminance(hex_color):
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return sum(
        (value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4) * weight
        for value, weight in zip(rgb, (0.2126, 0.7152, 0.0722))
    )


@pytest.mark.parametrize("theme", ["dark", "light", "auto-light"])
def test_chat_wizard_settings_and_persona_text_contrast(theme):
    tokens = theme_tokens(theme)
    pairs = [
        (foreground, background)
        for foreground in ("--text-primary", "--text-muted", "--text-faint", "--primary")
        for background in ("--bg", "--surface", "--surface-2")
    ] + [
        ("--on-primary", "--primary"),
        ("--on-primary", "--primary-dim"),
        ("--code-fg", "--surface-code"),
    ]
    for foreground, background in pairs:
        low, high = sorted(luminance(resolve(tokens, name)) for name in (foreground, background))
        assert (high + 0.05) / (low + 0.05) >= 4.5, (theme, foreground, background)


def test_automatic_light_and_explicit_light_have_identical_semantic_colors():
    explicit, automatic = theme_tokens("light"), theme_tokens("auto-light")
    for name in ("--text-primary", "--text-muted", "--text-faint", "--surface-bg",
                 "--bg", "--surface", "--primary", "--on-primary"):
        assert resolve(explicit, name) == resolve(automatic, name), name


def test_hidden_setup_banner_overrides_author_display_rules():
    # A plain [hidden] attribute loses to .setup-banner { display: flex }.
    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important", TOKENS)
    for filename in ("admin.css", "chat.css", "persona.css"):
        assert "tokens.css" in (STATIC / filename).read_text()


def test_fenced_code_retains_its_own_contrasting_background():
    css = (STATIC / "chat.css").read_text()
    rule = re.search(r"\.msg-body pre code\s*\{([^}]+)\}", css)[1]
    assert "background: var(--surface-code)" in rule
    assert "color: var(--code-fg)" in rule

"""Authored voice/configuration and diagnostic language recognition."""
from config.loader import SoulConfig, load_config
from openrook_tools.roleplay_eval.legacy_social import read_social_signals
from interface.setup import _identity_preset


def test_greeting_does_not_exempt_an_additional_direct_insult():
    assert read_social_signals("How the fuck are you doing, you useless idiot?").abuse_kind == "insult"


def test_default_seed_and_setup_agree_and_remain_owner_customizable():
    seed = load_config().soul
    preset = _identity_preset("steady")
    loaded = load_config().soul
    assert "modeled mood" in seed.identity
    assert seed.identity == loaded.identity.strip()
    assert preset["identity"] == seed.identity
    assert preset["core_values"] == seed.core_values
    assert seed.core_values["honesty"] > seed.core_values["loyalty"]
    assert SoulConfig().core_values == {} and SoulConfig().identity == ""
    loaded.voice = "My owner's custom voice."
    assert loaded.voice == "My owner's custom voice."


def test_all_presets_publish_valid_owner_editable_canonical_dispositions():
    for name in ("steady", "operator", "researcher", "mentor"):
        preset = _identity_preset(name)
        soul = SoulConfig(**preset)
        assert set(soul.dispositions) == {"pride", "patience", "forgiveness", "assertiveness"}
        assert soul.schema_version == 2
    assert _identity_preset("operator")["dispositions"] != _identity_preset("mentor")["dispositions"]

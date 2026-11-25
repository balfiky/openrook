"""Setup presets: one source for terminal and browser, independent characters."""

import asyncio
import re
from pathlib import Path

import httpx
from fastapi import FastAPI

from config.loader import SoulConfig
from config.presets import PRESET_ORDER, SOUL_FIELDS
from interface import api as api_mod
from interface import setup as setup_mod
from interface.routes_admin_identity import router as identity_router
from interface.setup import _identity_preset, _run_terminal_setup
from runtime.config import RuntimeConfig

ROOT = Path(__file__).resolve().parents[1]
DRIVE_KEYS = {"curiosity", "competence", "autonomy", "continuity", "repair", "attachment", "caution"}
DISPOSITION_KEYS = {"pride", "patience", "forgiveness", "assertiveness"}
BANNED = re.compile(
    r"assistant|loyal operator|protect the user|user's goals|help the user|\bserve\b|\bservice\b|\bobey\b",
    re.IGNORECASE,
)


def _fetch_presets(tmp_path, monkeypatch, auth=True):
    runtime_path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(data_dir=str(tmp_path / "data"), api_key="tok").write_yaml(str(runtime_path))
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", str(runtime_path))
    monkeypatch.setattr(api_mod, "RUNTIME_CONFIG_PATH", str(runtime_path))
    app = FastAPI()
    app.include_router(identity_router)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            headers = {"Authorization": "Bearer tok"} if auth else {}
            return await client.get("/admin/soul/presets", headers=headers)

    return asyncio.run(go())


def test_presets_endpoint_returns_full_independent_presets(tmp_path, monkeypatch):
    res = _fetch_presets(tmp_path, monkeypatch)
    assert res.status_code == 200
    presets = res.json()["presets"]
    assert [p["key"] for p in presets] == list(PRESET_ORDER)
    for preset in presets:
        assert preset["label"] and preset["description"]
        soul = preset["soul"]
        assert set(soul) == set(SOUL_FIELDS)
        assert set(soul["dispositions"]) == DISPOSITION_KEYS
        assert set(soul["drives"]) == DRIVE_KEYS
        for field in ("identity", "voice", "relational_stance", "growth_policy"):
            assert soul[field].strip()
        for field in ("core_values", "initial_traits", "likes", "dislikes", "boundaries"):
            assert soul[field]
        SoulConfig(name="x", **soul)


def test_presets_endpoint_requires_bearer(tmp_path, monkeypatch):
    assert _fetch_presets(tmp_path, monkeypatch, auth=False).status_code == 401


def test_terminal_presets_match_endpoint(tmp_path, monkeypatch):
    presets = _fetch_presets(tmp_path, monkeypatch).json()["presets"]
    for preset in presets:
        terminal = _identity_preset(preset["key"])
        for field in SOUL_FIELDS:
            assert terminal[field] == preset["soul"][field], (preset["key"], field)


def test_presets_have_no_service_framing_and_do_not_rank_loyalty_over_autonomy(tmp_path, monkeypatch):
    for preset in _fetch_presets(tmp_path, monkeypatch).json()["presets"]:
        text = " ".join(
            [preset["label"], preset["description"]]
            + [str(v) for v in preset["soul"].values() if isinstance(v, (str, list))]
        )
        found = BANNED.search(text)
        assert not found, (preset["key"], found and found.group(0))
        values = preset["soul"]["core_values"]
        assert values["loyalty"] < values["autonomy"], preset["key"]


def test_authored_presets_differ_in_temperament():
    keys = ("operator", "researcher", "mentor", "steady")
    dispositions = [tuple(sorted(_identity_preset(k)["dispositions"].items())) for k in keys]
    drives = [tuple(sorted(_identity_preset(k)["drives"].items())) for k in keys]
    assert len(set(dispositions)) == 4 and len(set(drives)) == 4
    for key in ("operator", "researcher", "mentor"):
        stance = _identity_preset(key)["relational_stance"].lower()
        assert "refuse" in stance and "owner" in stance


def test_wizard_has_no_hard_coded_character_tables():
    source = (ROOT / "interface/static/chat.js").read_text(encoding="utf-8")
    for needle in ("WIZARD_ARCHETYPE_CONFIG", "WIZARD_VOICE_TEXT", "WIZARD_STANCE_TEXT", "steady relational assistant", "loyal operator"):
        assert needle not in source
    assert "/admin/soul/presets" in source
    html = (ROOT / "interface/static/index.html").read_text(encoding="utf-8")
    assert re.search(r"<details>\s*<summary>Show identity file", html)
    panel3 = html[html.index('id="wizPanel-3"'):html.index('id="wizPanel-4"')]
    advanced = panel3[panel3.index("wizard-advanced"):]
    assert "wiz-tools_enabled" in panel3 and "wiz-tools_enabled" not in advanced
    assert re.search(r'id="wiz-tools_enabled">\s*<option value="true" selected>', panel3)


def test_admin_copy_describes_dispositions_and_remaining_limits():
    html = (ROOT / "interface/static/admin.html").read_text(encoding="utf-8")
    assert "dignity floor" not in html and "degrading roles" not in html
    assert "come from the dispositions and values above" in html
    assert "truthful execution receipts" in html


_BASE = ["", "2", "", "", "local-test-model", "n", "n"]


def _run_setup(tmp_path, answers, prompts=None, **kwargs):
    outputs: list[str] = []
    queue = iter(answers)

    def read(prompt):
        if prompts is not None:
            prompts.append(prompt)
        return next(queue)

    _run_terminal_setup(
        config_path=tmp_path / "runtime_config.yaml",
        data_dir="data",
        force=False,
        input_fn=read,
        output_fn=outputs.append,
        environ={"OPENROOK_CONFIG_DIR": str(tmp_path / "identity")},
        **kwargs,
    )
    return outputs


def test_terminal_tools_prompt_defaults_to_yes(tmp_path):
    prompts: list[str] = []
    _run_setup(tmp_path, _BASE + ["", "", "", "", "", ""], prompts)
    assert any("Enable agentic tools? [Y/n]" in p for p in prompts)
    saved = RuntimeConfig.from_yaml(str(tmp_path / "runtime_config.yaml"))
    assert saved.tools_enabled is True
    assert saved.shell_tool_enabled is False


def test_terminal_prints_summary_not_json_unless_requested(tmp_path):
    answers = _BASE + ["n", "Jarvis", "2"]
    out = "\n".join(_run_setup(tmp_path, answers))
    assert "Personality for Jarvis" in out and "Strongest drives" in out
    assert "Sharp operator:" in out and "Steady companion:" in out
    assert '"core_values"' not in out and "--show-identity" in out
    again = tmp_path / "again"
    again.mkdir()
    out2 = "\n".join(_run_setup(again, answers, show_identity=True))
    assert '"core_values"' in out2


def test_prompt_ends_line_for_piped_input(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(setup_mod.sys, "stdin", type("S", (), {"isatty": lambda self: False})())
    setup_mod._prompt_yes_no("Configure Telegram now?", default=False, input_fn=input)
    assert capsys.readouterr().out == "\n"

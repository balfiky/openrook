from __future__ import annotations

import json
import sys

import pytest

from runtime.config import RuntimeConfig
from interface.setup import (
    _browser_url,
    _check_llm,
    _configure_llm,
    _initialize_workspace,
    _prompt_secret,
    _run_terminal_setup,
    main,
)


@pytest.fixture(autouse=True)
def model_catalogs(monkeypatch):
    """Setup tests must never contact a live backend or Codex subprocess."""
    monkeypatch.setattr(
        "interface.setup.discover_llm_models",
        lambda _config: {
            "ok": True,
            "models": [{"id": "installed-model", "display_name": "Installed model"}],
            "message": "",
            "source": "openai",
        },
    )
    monkeypatch.setattr("interface.setup.list_codex_models", lambda: [])


def test_initialize_workspace_creates_config_and_dirs(tmp_path):
    config_path = tmp_path / "runtime_config.yaml"

    created = _initialize_workspace(config_path, "data", force=False)

    assert created is True
    config = RuntimeConfig.from_yaml(str(config_path))
    assert config.data_dir == str(tmp_path / "data")
    assert config.llm_model == ""
    assert config.api_key == ""
    assert config.llm_api_key == ""
    assert config.telegram_token == ""
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data" / "workspace").is_dir()


def test_init_only_creates_config_in_a_fresh_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROOK_RUNTIME_CONFIG", raising=False)
    monkeypatch.setattr(sys, "argv", ["openrook-setup", "--init-only"])

    main()

    path = tmp_path / "runtime_config.yaml"
    assert path.is_file()
    config = RuntimeConfig.from_yaml(str(path))
    assert config.llm_model == ""
    assert config.data_dir == str(tmp_path / "data")
    assert (tmp_path / "data" / "workspace").is_dir()


def test_initialize_workspace_preserves_existing_config(tmp_path):
    config_path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(data_dir=str(tmp_path / "custom")).write_yaml(str(config_path))

    created = _initialize_workspace(config_path, "data", force=False)

    assert created is False
    config = RuntimeConfig.from_yaml(str(config_path))
    assert config.data_dir == str(tmp_path / "custom")


def test_browser_url_uses_loopback_for_public_bind():
    assert _browser_url("0.0.0.0", 8000) == "http://127.0.0.1:8000"


def test_terminal_setup_writes_config_identity_and_completion(tmp_path):
    config_path = tmp_path / "runtime_config.yaml"
    identity_dir = tmp_path / "identity"
    answers = iter([
        "",        # data directory
        "2",       # vLLM
        "",        # base URL
        "",        # API key
        "local-test-model",
        "n",       # live check
        "n",       # Telegram
        "n",       # tools
        "Jarvis",  # identity name
        "2",       # operator
    ])

    config = _run_terminal_setup(
        config_path=config_path,
        data_dir="data",
        force=False,
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _line: None,
        environ={"OPENROOK_CONFIG_DIR": str(identity_dir)},
    )

    saved = RuntimeConfig.from_yaml(str(config_path))
    assert config.llm_model == "local-test-model"
    assert saved.llm_backend == "openai_compatible"
    assert saved.llm_base_url == "http://localhost:8002/v1"
    assert saved.llm_model == "local-test-model"
    assert saved.character_independence is False
    assert (tmp_path / "data" / "workspace").is_dir()
    assert "name: Jarvis" in (identity_dir / "soul.yaml").read_text(encoding="utf-8")

    admin_state = json.loads((tmp_path / "data" / "admin_state.json").read_text(encoding="utf-8"))
    assert admin_state["setup_completed"] is True


def test_terminal_setup_can_configure_tools_and_custom_workspace(tmp_path):
    config_path = tmp_path / "runtime_config.yaml"
    identity_dir = tmp_path / "identity"
    workspace = tmp_path / "tools"
    answers = iter([
        "",       # data directory
        "1",      # ollama
        "",       # base URL
        "",       # API key
        "",       # accept the sole installed model
        "n",      # live check
        "n",      # Telegram
        "y",      # tools
        "2",      # autonomous
        "y",      # shell
        str(workspace),
        "",       # identity name
        "",       # steady
    ])

    _run_terminal_setup(
        config_path=config_path,
        data_dir="data",
        force=False,
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _line: None,
        environ={"OPENROOK_CONFIG_DIR": str(identity_dir)},
    )

    saved = RuntimeConfig.from_yaml(str(config_path))
    assert saved.tools_enabled is True
    assert saved.shell_tool_enabled is True
    assert saved.autonomy_level == "autonomous"
    assert saved.tools_workspace == str(workspace)
    assert saved.character_independence is False
    assert workspace.is_dir()


@pytest.mark.parametrize("independence", [False, True])
def test_terminal_setup_does_not_prompt_for_character_independence(tmp_path, independence):
    config_path = tmp_path / "runtime_config.yaml"
    RuntimeConfig(
        data_dir=str(tmp_path / "data"), character_independence=independence,
    ).write_yaml(str(config_path))
    answers = iter(["", "codex", "", "n", "n", "n", "", ""])
    transcript = []

    def answer(prompt):
        transcript.append(prompt)
        return next(answers)

    config = _run_terminal_setup(
        config_path=config_path, data_dir="data", force=False,
        input_fn=answer, output_fn=transcript.append,
        environ={"OPENROOK_CONFIG_DIR": str(tmp_path / "identity")},
    )

    assert not any("independence" in line.lower() for line in transcript)
    assert config.character_independence is independence
    assert RuntimeConfig.from_yaml(str(config_path)).character_independence is independence


def test_terminal_setup_codex_keeps_failed_connection_nonfatal(tmp_path, monkeypatch):
    config_path = tmp_path / "runtime_config.yaml"
    answers = iter(["", "codex", "", "y", "n", "n", "", ""])
    lines = []
    checked = []

    def check(config):
        checked.append(config.llm_backend)
        return {
            "ok": False,
            "error": "Codex is not logged in. Run `codex login` and try again.",
        }

    monkeypatch.setattr("interface.setup.check_llm_connection", check)
    config = _run_terminal_setup(
        config_path=config_path,
        data_dir="data",
        force=False,
        input_fn=lambda _prompt: next(answers),
        output_fn=lines.append,
        environ={"OPENROOK_CONFIG_DIR": str(tmp_path / "identity")},
    )

    assert checked == ["codex"]
    assert config.llm_backend == "codex"
    assert config.llm_model == ""
    assert config.llm_base_url == ""
    assert config.llm_api_key == ""
    assert RuntimeConfig.from_yaml(str(config_path)).llm_backend == "codex"
    assert "codex login" in "\n".join(lines)
    assert "continue setup" in "\n".join(lines)
    assert "informational flag" not in "\n".join(lines)
    assert "  6. Codex CLI — uses your Codex login (codex)" in lines
    assert "Start the terminal chat:" in lines
    assert f"  openrook --config {config_path}" in lines
    assert f"  openrook-web --config {config_path}" in lines
    assert f"  openrook-setup --web --config {config_path}" in lines
    heading = lines.index("OpenRook terminal setup")
    assert len(lines[heading]) == len(lines[heading + 1])


@pytest.mark.parametrize("answer", ["", "y"])
def test_terminal_model_check_accepts_default_or_yes(answer, monkeypatch):
    checked = []
    lines = []

    def check(config):
        checked.append(config)
        return {"ok": True, "sample_response": "private sample should not be printed"}

    monkeypatch.setattr("interface.setup.check_llm_connection", check)
    config = RuntimeConfig(llm_backend="codex")
    _check_llm(config, input_fn=lambda _prompt: answer, output_fn=lines.append)

    assert checked == [config]
    assert "Connection succeeded." in "\n".join(lines)
    assert "private sample" not in "\n".join(lines)


def test_terminal_model_check_can_be_skipped(monkeypatch):
    def unexpected_check(_config):
        pytest.fail("Skipped setup check must not call the model")

    monkeypatch.setattr("interface.setup.check_llm_connection", unexpected_check)
    lines = []
    _check_llm(
        RuntimeConfig(llm_model="chosen-model"),
        input_fn=lambda _prompt: "n", output_fn=lines.append,
    )
    assert "Connection not checked." in "\n".join(lines)


def test_terminal_secrets_use_hidden_input_without_echoing_defaults(monkeypatch):
    prompts = []

    def hidden_input(prompt):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("interface.setup.getpass.getpass", hidden_input)
    existing = "example-token"
    assert _prompt_secret("Token", default=existing, input_fn=input) == existing
    assert existing not in prompts[0]
    assert "blank keeps" in prompts[0]


@pytest.mark.parametrize("answer,expected", [("2", "second-model"), ("manual-model", "manual-model"), ("", "")])
def test_terminal_discovery_requires_choice_with_multiple_models(monkeypatch, answer, expected):
    requests = []

    def discover(config):
        requests.append((config.llm_base_url, config.llm_api_key))
        return {
            "ok": True, "message": "",
            "models": [{"id": "first-model"}, {"id": "second-model"}],
        }

    monkeypatch.setattr("interface.setup.discover_llm_models", discover)
    answers = iter(["hosted", "https://provider.example/v1", "provider-key", answer])
    lines = []
    config = RuntimeConfig()
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert requests == [("https://provider.example/v1", "provider-key")]
    assert config.llm_model == expected
    assert "  1. first-model" in lines
    assert "  2. second-model" in lines
    assert not any("preselected" in line for line in lines)


def test_terminal_preselects_the_only_discovered_model():
    answers = iter(["ollama", "", "", ""])
    lines = []
    config = RuntimeConfig()
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert config.llm_model == "installed-model"
    assert "Only one model is available; preselected: installed-model" in lines


@pytest.mark.parametrize("answer,expected", [("", ""), ("manual-model", "manual-model")])
def test_terminal_empty_ollama_can_defer_or_enter_name(monkeypatch, answer, expected):
    message = (
        "No models are installed in Ollama yet — install one with `ollama pull <model>` "
        "(see https://ollama.com/library), then rerun setup or choose in Settings."
    )
    monkeypatch.setattr("interface.setup.discover_llm_models", lambda _config: {
        "ok": True, "source": "ollama", "models": [], "message": message,
    })
    answers = iter(["ollama", "", "", answer])
    lines = []
    config = RuntimeConfig()
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert config.llm_model == expected
    assert message in lines


def test_terminal_failed_discovery_keeps_manual_selection(monkeypatch):
    monkeypatch.setattr("interface.setup.discover_llm_models", lambda _config: {
        "ok": False, "models": [], "message": "Could not connect to the model backend.",
    })
    answers = iter(["custom", "https://provider.example/v1", "", "typed-model"])
    lines = []
    config = RuntimeConfig()
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert config.llm_model == "typed-model"
    assert "Could not connect to the model backend." in lines


def test_terminal_keeps_configured_model_for_the_same_endpoint():
    config = RuntimeConfig(
        llm_backend="openai_compatible", llm_base_url="http://localhost:11434/v1",
        llm_model="previously-chosen", llm_api_key="existing-key",
    )
    answers = iter(["ollama", "", "", ""])
    lines = []
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert config.llm_model == "previously-chosen"
    assert config.llm_api_key == "existing-key"
    assert "Configured model: previously-chosen" in lines


def test_terminal_does_not_carry_model_to_another_endpoint(monkeypatch):
    monkeypatch.setattr("interface.setup.discover_llm_models", lambda _config: {
        "ok": True, "models": [{"id": "new-one"}, {"id": "new-two"}], "message": "",
    })
    config = RuntimeConfig(llm_base_url="https://old.example/v1", llm_model="old-choice")
    answers = iter(["hosted", "https://new.example/v1", "", ""])
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lambda _: None)
    assert config.llm_model == ""


@pytest.mark.parametrize("answer,expected", [("", ""), ("0", ""), ("1", "catalog-model"), ("manual", "manual")])
def test_terminal_codex_catalog_and_codex_default(monkeypatch, answer, expected):
    monkeypatch.setattr("interface.setup.list_codex_models", lambda: [{"slug": "catalog-model"}])
    answers = iter(["codex", answer])
    lines = []
    config = RuntimeConfig()
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert config.llm_model == expected
    assert "  0. Codex default" in lines
    assert "  1. catalog-model" in lines


def test_terminal_reprompts_for_invalid_model_number(monkeypatch):
    answers = iter(["ollama", "", "", "9", "1"])
    lines = []
    config = RuntimeConfig()
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert config.llm_model == "installed-model"
    assert "Choose a listed number or type a model name." in lines


def test_terminal_missing_model_defers_live_check():
    def unexpected_prompt(_prompt):
        pytest.fail("An unselected model should not prompt for a live check")

    lines = []
    _check_llm(RuntimeConfig(), input_fn=unexpected_prompt, output_fn=lines.append)
    assert "No model selected. Choose one later in Settings → LLM." in lines


def test_terminal_empty_catalog_saves_no_model_and_skips_generation(tmp_path, monkeypatch):
    monkeypatch.setattr("interface.setup.discover_llm_models", lambda _config: {
        "ok": True, "models": [], "message": "No models are installed in Ollama yet.",
    })

    def unexpected_check(_config):
        pytest.fail("Setup with no selected model must not generate a completion")

    monkeypatch.setattr("interface.setup.check_llm_connection", unexpected_check)
    answers = iter(["", "ollama", "", "", "", "n", "n", "", ""])
    path = tmp_path / "setup.yaml"
    config = _run_terminal_setup(
        config_path=path, data_dir="workspace", force=False,
        input_fn=lambda _: next(answers), output_fn=lambda _: None,
        environ={"OPENROOK_CONFIG_DIR": str(tmp_path / "identity")},
    )
    assert config.llm_model == ""
    assert RuntimeConfig.from_yaml(str(path)).llm_model == ""


def test_terminal_failed_codex_catalog_allows_manual_name(monkeypatch):
    def catalog():
        raise RuntimeError("CLI failure with private output")

    monkeypatch.setattr("interface.setup.list_codex_models", catalog)
    answers = iter(["codex", "chosen-codex-model"])
    lines = []
    config = RuntimeConfig()
    _configure_llm(config, input_fn=lambda _: next(answers), output_fn=lines.append)
    assert config.llm_model == "chosen-codex-model"
    assert "Could not load the Codex model catalog" in "\n".join(lines)
    assert "private output" not in "\n".join(lines)

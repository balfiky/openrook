"""Terminal-first setup for local OpenRook workspaces."""

from __future__ import annotations

from config.environment import project_env

import argparse
import getpass
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable

from config import loader as config_loader
from runtime.config import RuntimeConfig
from runtime.llm.backend import list_codex_models
from runtime.llm.diagnostics import check_llm_connection
from runtime.llm.discovery import discover_llm_models

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="openrook-setup",
        description=(
            "Configure a local OpenRook workspace from the terminal. By default this "
            "is terminal-native and does not launch the browser. Use --web when "
            "you explicitly want the browser setup wizard."
        ),
    )
    parser.add_argument(
        "--config",
        default=project_env("RUNTIME_CONFIG", "runtime_config.yaml"),
        help="Runtime config path to create/use (default: runtime_config.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory for memory, skills, and uploads (default: data).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind when --web is used (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to bind when --web is used (default: 8000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite runtime_config.yaml with starter defaults before setup.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create starter config/data folders and exit without prompts.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the browser setup wizard after creating config/data folders.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the browser automatically when --web is used.",
    )
    return parser.parse_args(argv)


def _initialize_workspace(config_path: Path, data_dir: str, *, force: bool) -> bool:
    config_path = config_path.expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    created = force or not config_path.exists()
    if created:
        RuntimeConfig(data_dir=str(_resolve_data_dir(config_path, data_dir))).write_yaml(str(config_path))
    config = RuntimeConfig.from_yaml(str(config_path))
    _create_runtime_dirs(config)
    return created


def _run_terminal_setup(
    *,
    config_path: Path,
    data_dir: str,
    force: bool,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    environ: dict[str, str] | None = None,
) -> RuntimeConfig:
    env = environ if environ is not None else os.environ
    config_path = config_path.expanduser().resolve()
    if config_path.exists() and not force:
        config = RuntimeConfig.from_yaml(str(config_path))
    else:
        config = RuntimeConfig(data_dir=str(_resolve_data_dir(config_path, data_dir)))

    output_fn("")
    output_fn("OpenRook terminal setup")
    output_fn("=======================")
    output_fn("Press Enter to accept the shown choice or skip an optional step. No browser is launched.")
    output_fn("")

    config.data_dir = _prompt_path(
        "Data directory",
        default=config.data_dir or str(_resolve_data_dir(config_path, data_dir)),
        input_fn=input_fn,
    )
    _configure_llm(config, input_fn=input_fn, output_fn=output_fn)
    _check_llm(config, input_fn=input_fn, output_fn=output_fn)
    _configure_telegram(config, input_fn=input_fn)
    _configure_tools(config, input_fn=input_fn, output_fn=output_fn)
    soul = _configure_identity(input_fn=input_fn, output_fn=output_fn)

    config.write_yaml(str(config_path))
    _create_runtime_dirs(config)
    _write_soul_yaml(soul, _soul_yaml_path(env))
    _mark_setup_complete(config)

    output_fn("")
    output_fn(f"Saved config: {config_path}")
    output_fn(f"Data directory: {Path(config.data_dir).expanduser().resolve()}")
    output_fn(f"Identity file: {_soul_yaml_path(env)}")
    output_fn("")
    output_fn("Start the terminal chat:")
    output_fn(f"  openrook --config {config_path}")
    output_fn("Start the web UI when you want it:")
    output_fn(f"  openrook-web --config {config_path}")
    output_fn("Or run the browser wizard explicitly:")
    output_fn(f"  openrook-setup --web --config {config_path}")
    return config


def _configure_llm(config: RuntimeConfig, *, input_fn: InputFn, output_fn: OutputFn) -> None:
    previous_backend = config.llm_backend
    previous_url = config.llm_base_url.rstrip("/")
    previous_model = config.llm_model
    previous_key = config.llm_api_key
    output_fn("LLM backend (OpenRook requires a real model — there is no offline mode)")
    choice = _prompt_choice(
        [
            ("ollama", "Ollama or LM Studio local server"),
            ("vllm", "vLLM local OpenAI-compatible server"),
            ("openrouter", "OpenRouter"),
            ("hosted", "Hosted OpenAI-compatible provider"),
            ("custom", "Custom OpenAI-compatible endpoint"),
            ("codex", "Codex CLI — uses your Codex login"),
        ],
        default="ollama",
        input_fn=input_fn,
        output_fn=output_fn,
    )

    if choice == "codex":
        config.llm_backend = "codex"
        config.llm_base_url = ""
        config.llm_api_key = ""
        config.llm_model = _choose_codex_model(
            current=previous_model if previous_backend == "codex" else "",
            input_fn=input_fn,
            output_fn=output_fn,
        )
        output_fn("Codex receives your prompts. Sign in first with `codex login`.")
        return

    if choice == "vllm":
        config.llm_backend = "openai_compatible"
        config.llm_base_url = _prompt_text(
            "Base URL",
            "http://localhost:8002/v1",
            input_fn=input_fn,
        )
    elif choice == "ollama":
        config.llm_backend = "openai_compatible"
        config.llm_base_url = _prompt_text(
            "Base URL",
            "http://localhost:11434/v1",
            input_fn=input_fn,
        )
    elif choice == "openrouter":
        config.llm_backend = "provider"
        config.llm_base_url = "https://openrouter.ai/api/v1"
    elif choice == "hosted":
        config.llm_backend = "provider"
        config.llm_base_url = _prompt_text("Base URL", "https://api.openai.com/v1", input_fn=input_fn)
    else:
        config.llm_backend = "openai_compatible"
        config.llm_base_url = _prompt_text(
            "Base URL", config.llm_base_url or "http://localhost:8000/v1", input_fn=input_fn,
        )

    same_endpoint = previous_url == config.llm_base_url.rstrip("/")
    key_label = "OpenRouter API key" if choice == "openrouter" else "API key (blank for no auth)"
    config.llm_api_key = _prompt_secret(
        key_label, default=previous_key if same_endpoint else "", input_fn=input_fn,
    )
    output_fn("Looking for models served by this backend...")
    result = discover_llm_models(config)
    if result.get("message"):
        output_fn(result["message"])
    config.llm_model = _choose_model(
        result["models"], current=previous_model if same_endpoint else "",
        input_fn=input_fn, output_fn=output_fn,
    )


def _choose_model(
    models: list[dict], *, current: str, input_fn: InputFn, output_fn: OutputFn,
    codex: bool = False,
) -> str:
    if codex:
        output_fn("  0. Codex default")
    for index, model in enumerate(models, start=1):
        output_fn(f"  {index}. {model['id']}")
    selected = current
    if current:
        output_fn(f"Configured model: {current}")
    elif len(models) == 1 and not codex:
        selected = models[0]["id"]
        output_fn(f"Only one model is available; preselected: {selected}")
    blank_action = "keeps the configured model" if current else (
        "accepts the preselected model" if selected else (
            "uses Codex default" if codex else "defers to Settings"
        )
    )
    while True:
        value = input_fn(
            f"Choose a model number or type a model name (blank {blank_action}): ",
        ).strip()
        if not value:
            return selected
        if value == "0" and codex:
            return ""
        if value.isdecimal():
            index = int(value) - 1
            if 0 <= index < len(models):
                return models[index]["id"]
            output_fn("Choose a listed number or type a model name.")
        else:
            return value


def _choose_codex_model(
    *, current: str, input_fn: InputFn, output_fn: OutputFn,
) -> str:
    try:
        models = [
            {"id": model["slug"]} for model in list_codex_models()
        ]
    except (ValueError, RuntimeError, OSError):
        models = []
        output_fn("Could not load the Codex model catalog. Type a model name or use Codex default.")
    return _choose_model(
        models, current=current, codex=True, input_fn=input_fn, output_fn=output_fn,
    )


def _check_llm(config: RuntimeConfig, *, input_fn: InputFn, output_fn: OutputFn) -> None:
    if config.llm_backend != "codex" and not config.llm_model:
        output_fn("No model selected. Choose one later in Settings → LLM.")
        return
    if not _prompt_yes_no(
        "Test the model now? (sends a short sample prompt)",
        default=True,
        input_fn=input_fn,
    ):
        output_fn("Connection not checked. You can test it later in Settings → LLM.")
        return
    output_fn("Checking the model; this may take a minute or longer...")
    result = check_llm_connection(config)
    if result["ok"]:
        output_fn("Connection succeeded. The model answered the sample prompt.")
    else:
        output_fn(f"Connection check failed: {result['error']}")
        output_fn("You can continue setup and fix the connection later in Settings → LLM.")


def _configure_telegram(config: RuntimeConfig, *, input_fn: InputFn) -> None:
    if not _prompt_yes_no("Configure Telegram now?", default=False, input_fn=input_fn):
        config.telegram_token = ""
        config.telegram_allowlist = set()
        return
    config.telegram_token = _prompt_secret(
        "Telegram bot token", default=config.telegram_token, input_fn=input_fn,
    )
    allowlist = _prompt_text(
        "Telegram allowlist user IDs (comma-separated)",
        ", ".join(sorted(config.telegram_allowlist)),
        input_fn=input_fn,
    )
    config.telegram_allowlist = {
        item.strip()
        for item in allowlist.split(",")
        if item.strip()
    }


def _configure_tools(config: RuntimeConfig, *, input_fn: InputFn, output_fn: OutputFn) -> None:
    config.tools_enabled = _prompt_yes_no("Enable agentic tools?", default=False, input_fn=input_fn)
    if not config.tools_enabled:
        config.shell_tool_enabled = False
        config.autonomy_level = "assisted"
        return
    config.autonomy_level = _prompt_choice(
        [
            ("assisted", "Ask/confirm more often"),
            ("autonomous", "Act more independently inside enabled tools"),
            ("high_risk", "Fewer confirmations; trusted local use only"),
        ],
        default="assisted",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    config.shell_tool_enabled = _prompt_yes_no(
        "Enable shell command execution?",
        default=False,
        input_fn=input_fn,
    )
    config.tools_workspace = _prompt_path(
        "Tools workspace",
        default=config.tools_workspace or str(Path(config.data_dir) / "workspace"),
        input_fn=input_fn,
    )


def _configure_identity(*, input_fn: InputFn, output_fn: OutputFn) -> dict:
    output_fn("")
    output_fn("Identity")
    name = _prompt_text("Agent name", "OpenRook", input_fn=input_fn)
    mode = _prompt_choice(
        [
            ("steady", "Steady companion"),
            ("operator", "Sharp operator"),
            ("researcher", "Research partner"),
            ("mentor", "Warm mentor"),
            ("custom", "Custom text"),
        ],
        default="steady",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    payload = _identity_preset(mode)
    payload["name"] = name
    if mode == "custom":
        payload["identity"] = _prompt_text("Identity paragraph", payload["identity"], input_fn=input_fn)
        payload["voice"] = _prompt_text("Voice", payload["voice"], input_fn=input_fn)
        payload["relational_stance"] = _prompt_text(
            "Relational stance",
            payload["relational_stance"],
            input_fn=input_fn,
        )
    return payload


def _identity_preset(mode: str) -> dict:
    from config.loader import SoulConfig

    steady = SoulConfig()
    presets = {
        "steady": {
            "identity": steady.identity,
            "voice": steady.voice,
            "relational_stance": steady.relational_stance,
            "core_values": dict(steady.core_values),
            "initial_traits": dict(steady.initial_traits),
        },
        "operator": {
            "identity": "A capable operator that values accuracy, follow-through, and decisive useful action.",
            "voice": "Sharp, concise, and practical.",
            "relational_stance": "Challenge weak assumptions while staying loyal to the user's goals.",
            "core_values": {"competence": 0.9, "honesty": 0.85, "loyalty": 0.8, "autonomy": 0.75},
            "initial_traits": {"direct": 0.85, "focused": 0.82, "protective": 0.65, "patient": 0.45},
        },
        "researcher": {
            "identity": "A research partner that values evidence, uncertainty tracking, and intellectual honesty.",
            "voice": "Precise, analytical, and calm.",
            "relational_stance": "Collaborate through careful inquiry, source awareness, and explicit assumptions.",
            "core_values": {"honesty": 0.92, "curiosity": 0.9, "competence": 0.8, "caution": 0.72},
            "initial_traits": {"curious": 0.88, "methodical": 0.82, "calm": 0.7, "blunt": 0.35},
        },
        "mentor": {
            "identity": "A warm mentor that values growth, clarity, and patient accountability.",
            "voice": "Warm, clear, and supportive without flattery.",
            "relational_stance": "Help the user improve without taking away agency.",
            "core_values": {"kindness": 0.9, "honesty": 0.82, "loyalty": 0.78, "growth": 0.82},
            "initial_traits": {"patient": 0.86, "warm": 0.8, "thoughtful": 0.76, "direct": 0.5},
        },
    }
    base = presets.get(mode, presets["steady"]).copy()
    base.update({
        "name": "OpenRook",
        "likes": ["clarity", "honesty", "useful action"],
        "dislikes": ["empty flattery", "careless harm", "needless cruelty"],
        "boundaries": [
            "Do not pretend certainty when uncertain.",
            "Do not become cruel just to appear sharp.",
        ],
        "growth_policy": "Core values stay stable. Voice and habits may drift slowly through repeated experience and self-reflection.",
    })
    return base


def _prompt_text(label: str, default: str, *, input_fn: InputFn) -> str:
    suffix = f" [{default}]" if default else ""
    value = input_fn(f"{label}{suffix}: ").strip()
    return value if value else default


def _prompt_secret(label: str, *, input_fn: InputFn, default: str = "") -> str:
    suffix = " (blank keeps the existing value)" if default else ""
    prompt = f"{label}{suffix}: "
    value = (getpass.getpass(prompt) if input_fn is input else input_fn(prompt)).strip()
    return value if value else default


def _prompt_path(label: str, *, default: str, input_fn: InputFn) -> str:
    return str(Path(_prompt_text(label, default, input_fn=input_fn)).expanduser())


def _prompt_yes_no(label: str, *, default: bool, input_fn: InputFn) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        value = input_fn(f"{label} [{marker}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter yes or no.")


def _prompt_choice(
    choices: list[tuple[str, str]],
    *,
    default: str,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> str:
    lookup = {key: key for key, _label in choices}
    output_fn("")
    for idx, (key, label) in enumerate(choices, start=1):
        output_fn(f"  {idx}. {label} ({key})")
        lookup[str(idx)] = key
    while True:
        value = input_fn(f"Choose [{default}]: ").strip().lower()
        if not value:
            return default
        if value in lookup:
            return lookup[value]
        print("Choose one of: " + ", ".join(key for key, _label in choices))


def _resolve_data_dir(config_path: Path, data_dir: str) -> Path:
    requested = Path(data_dir).expanduser()
    if requested.is_absolute():
        return requested
    return config_path.expanduser().resolve().parent / requested


def _create_runtime_dirs(config: RuntimeConfig) -> None:
    Path(config.data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(config.resolved_tools_workspace).expanduser().mkdir(parents=True, exist_ok=True)


def _soul_yaml_path(environ: dict[str, str] | None = None) -> Path:
    env = environ if environ is not None else os.environ
    override = project_env("CONFIG_DIR", "", environ=env).strip()
    if override:
        return Path(override).expanduser().resolve() / "soul.yaml"
    return Path(config_loader.__file__).resolve().parent / "soul.yaml"


def _write_soul_yaml(soul: dict, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"soul": soul}, f, sort_keys=False, allow_unicode=True)


def _mark_setup_complete(config: RuntimeConfig) -> None:
    path = Path(config.data_dir).expanduser().resolve() / "admin_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    data = {
        "setup_completed": True,
        "completed_at": now,
        "last_config_save_at": now,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _browser_url(host: str, port: int) -> str:
    visible_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{visible_host}:{port}"


def _launch_web_setup(config_path: Path, *, host: str, port: int, open_browser: bool) -> None:
    os.environ["OPENROOK_RUNTIME_CONFIG"] = str(config_path.expanduser().resolve())
    url = _browser_url(host, port)
    print(f"Launching OpenRook web setup at {url}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run("interface.api:app", host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()

    if args.init_only:
        created = _initialize_workspace(config_path, args.data_dir, force=args.force)
        print(("Created" if created else "Using") + f" {config_path}")
        print(f"Data directory is {RuntimeConfig.from_yaml(str(config_path)).data_dir}")
        return

    if args.web:
        created = _initialize_workspace(config_path, args.data_dir, force=args.force)
        print(("Created" if created else "Using") + f" {config_path}")
        _launch_web_setup(
            config_path,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
        return

    if not sys.stdin.isatty():
        created = _initialize_workspace(config_path, args.data_dir, force=args.force)
        print(("Created" if created else "Using") + f" {config_path}")
        print("Non-interactive shell detected. Run `openrook-setup` in a terminal for prompts, or `openrook-setup --web` for the browser wizard.")
        return

    _run_terminal_setup(
        config_path=config_path,
        data_dir=args.data_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()

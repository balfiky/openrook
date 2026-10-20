"""OpenRook Runtime entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from config.environment import project_env
from runtime.app import OpenRookApp
from runtime.config import RuntimeConfig


def _get_version() -> str:
    try:
        return version("openrook")
    except PackageNotFoundError:
        return "unknown"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="openrook",
        description=(
            "Start the OpenRook console runtime. Chat in the terminal, with "
            "Telegram and proactive behavior when configured. Set console_enabled "
            "to false in the config for a background runtime. Use `openrook-web` "
            "for the HTTP API and bundled browser UI."
        ),
    )
    parser.add_argument(
        "--config",
        default=project_env("RUNTIME_CONFIG", "runtime_config.yaml"),
        help="Path to runtime config YAML (default: runtime_config.yaml).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO logs in the console (may interrupt the chat prompt).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openrook {_get_version()}",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        raise SystemExit(
            f"Runtime config not found: {config_path}. "
            "Run openrook-setup or openrook-web to create it "
            "(use --config for a custom path)."
        )
    config = RuntimeConfig.from_yaml(str(config_path))
    if config.llm_backend != "codex" and not (config.llm_model or "").strip():
        raise SystemExit(
            f"No LLM model selected in {config_path}. "
            "Run openrook-setup or openrook-web and choose a model served by "
            "your backend in setup or Settings → LLM "
            "(use --config for a custom path)."
        )
    logging.basicConfig(
        level=logging.INFO if args.verbose or not config.console_enabled else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = OpenRookApp(config)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()

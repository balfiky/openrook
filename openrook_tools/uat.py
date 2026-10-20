"""Run OpenRook user-acceptance tests against the installed web runtime."""

from __future__ import annotations

from config.environment import project_env

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="openrook-uat",
        description=(
            "Run clean-setup UAT checks that start the real web server and "
            "drive the browser/admin surfaces. Install with .[uat] first."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a live LLM backend. Missing backend configuration is a hard failure.",
    )
    parser.add_argument(
        "--backend",
        default=project_env("UAT_BACKEND", "mock"),
        choices=["mock", "provider", "openai_compatible", "minimax", "codex"],
        help="Backend for the temporary runtime config.",
    )
    parser.add_argument(
        "--model",
        default=project_env("UAT_MODEL", ""),
        help="Model name for provider/openai_compatible/minimax/codex live UAT.",
    )
    parser.add_argument(
        "--base-url",
        default=project_env("UAT_BASE_URL", ""),
        help="OpenAI-compatible base URL for provider/openai_compatible live UAT.",
    )
    parser.add_argument(
        "--api-key-env",
        default=project_env("UAT_API_KEY_ENV", ""),
        help="Environment variable that contains the live backend API key.",
    )
    parser.add_argument(
        "--artifacts",
        default=project_env("UAT_ARTIFACTS", "reports/uat"),
        help="Directory for screenshots, logs, and UAT reports.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run the browser headed for local debugging.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Preserve the temporary runtime workspace after tests.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed through to pytest after --.",
    )
    return parser.parse_args(argv)


def _default_api_key_env(backend: str) -> str:
    return "LLM_API_KEY"


def _validate_live_config(args: argparse.Namespace) -> str:
    backend = args.backend
    if backend == "mock":
        raise SystemExit("--live requires --backend provider, openai_compatible, minimax, or codex")
    if backend == "codex":
        return ""
    api_key_env = args.api_key_env or _default_api_key_env(backend)
    if not os.environ.get(api_key_env):
        raise SystemExit(f"--live requires ${api_key_env} to be set")
    if backend in {"provider", "openai_compatible"} and not args.base_url:
        raise SystemExit(f"--live --backend {backend} requires --base-url or $OPENROOK_UAT_BASE_URL")
    if not args.model:
        raise SystemExit(f"--live --backend {backend} requires --model or $OPENROOK_UAT_MODEL")
    return api_key_env


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENROOK_UAT_ARTIFACTS"] = str(artifact_dir)
    env["OPENROOK_UAT_KEEP_WORKSPACE"] = "1" if args.keep_workspace else "0"
    env["OPENROOK_UAT_HEADED"] = "1" if args.headed else "0"

    if args.live:
        api_key_env = _validate_live_config(args)
        env["OPENROOK_UAT_LIVE"] = "1"
        env["OPENROOK_UAT_BACKEND"] = args.backend
        env["OPENROOK_UAT_MODEL"] = args.model
        env["OPENROOK_UAT_BASE_URL"] = args.base_url
        env["OPENROOK_UAT_API_KEY_ENV"] = api_key_env
    else:
        env["OPENROOK_UAT_LIVE"] = "0"
        env["OPENROOK_UAT_BACKEND"] = args.backend
        if args.backend != "mock":
            env["OPENROOK_UAT_MODEL"] = args.model
            env["OPENROOK_UAT_BASE_URL"] = args.base_url
            env["OPENROOK_UAT_API_KEY_ENV"] = args.api_key_env or _default_api_key_env(args.backend)

    cmd = [sys.executable, "-m", "pytest", "tests/uat", "-q"]
    cmd.extend(args.pytest_args)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

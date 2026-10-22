"""Command-line entry point; credentials are environment-only."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import os
from pathlib import Path

from .api import PublicAPI, Redactor, validate_destination
from .judge import Judge
from .runner import Evaluation, Options
from .scenarios import catalog, select_scenarios


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate a running OpenRook server through its public API"
    )
    result.add_argument("--base-url", help="Running server base URL; no embedded credentials")
    result.add_argument("--out", type=Path, help="Artifact directory")
    result.add_argument(
        "--scenarios",
        nargs="+",
        default=[],
        help="Case IDs, glob patterns or category letters; comma lists also accepted",
    )
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--judge", choices=("on", "off"), default="off")
    result.add_argument(
        "--judge-base-url", help="Optional OpenAI-compatible API base; never written to artifacts"
    )
    result.add_argument(
        "--judge-model", help="Required only with --judge-base-url; never written to artifacts"
    )
    result.add_argument(
        "--api-key-env",
        default="OPENROOK_API_KEY",
        help="Environment variable containing the server bearer credential",
    )
    result.add_argument(
        "--timeout", type=float, default=900, help="Per-request timeout in seconds (default: 900)"
    )
    result.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrent independent scenarios (default: 2); mutations are exclusive",
    )
    result.add_argument(
        "--max-turns",
        type=int,
        help="Per-scenario turn cap; shortened coverage remains inconclusive",
    )
    result.add_argument(
        "--resume",
        action="store_true",
        help="Continue the same run without repeating completed turns",
    )
    result.add_argument(
        "--restore-only",
        action="store_true",
        help="Restore the private journal's original settings and identity, then stop",
    )
    result.add_argument(
        "--tools-workspace",
        type=Path,
        help="Locally mounted tools workspace, for independent read-only file verification",
    )
    result.add_argument(
        "--data-root",
        help="Optional isolated data root on the SERVER filesystem; must remain the same when resuming",
    )
    result.add_argument(
        "--restart-hook",
        help="Trusted local operator command to restart the server; its output is suppressed",
    )
    result.add_argument("--restart-timeout", type=float, default=180)
    result.add_argument(
        "--list",
        action="store_true",
        help="List scenarios, turn counts and estimated calls without connecting",
    )
    return result


async def execute(args, scenarios, redactor):
    api = PublicAPI(args.base_url, key=os.environ.get(args.api_key_env, ""), timeout=args.timeout)
    external = None
    try:
        judge = None
        if args.judge == "on":
            if args.judge_base_url:
                external = PublicAPI(
                    args.judge_base_url,
                    key=os.environ.get("OPENROOK_JUDGE_API_KEY", ""),
                    timeout=args.timeout,
                )
            judge = Judge(
                external or api, redactor, model=args.judge_model, external=external is not None
            )
        options = Options(
            args.out,
            args.seed,
            args.max_turns,
            args.concurrency,
            args.resume,
            args.tools_workspace,
            args.data_root,
            args.restart_hook,
            args.restart_timeout,
            args.restore_only,
        )
        evaluation = Evaluation(api, scenarios, options, redactor, judge=judge, progress=print)
        summary = await evaluation.run()
        print(
            f"Recorded {summary['turns']} turns. Result: {summary['status']}. See report.md and summary.json."
        )
        if evaluation.paused:
            return 3
        if summary["restoration"] == "failed":
            return 1
        return {"pass": 0, "fail": 1, "inconclusive": 2}[summary["status"]]
    finally:
        await api.close()
        if external:
            await external.close()


def main(argv: list[str] | None = None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.list:
        scenarios = catalog(args.seed or 0)
        for scenario in scenarios:
            count = len(scenario.turns)
            print(
                f"{scenario.id:38} {scenario.category}  {count:2} turns  ~{count * 2}-{count * 5} runtime calls; +{count}-{count * 2} judge calls if enabled"
            )
        print(
            f"Total: {sum(len(s.turns) for s in scenarios)} chat turns. Digestion and tool inference may add calls."
        )
        return 0
    if not args.base_url or args.out is None:
        arg_parser.error("--base-url and --out are required unless --list is used")
    if (
        not math.isfinite(args.timeout)
        or not math.isfinite(args.restart_timeout)
        or args.timeout <= 0
        or args.restart_timeout <= 0
        or args.concurrency < 1
        or (args.max_turns is not None and args.max_turns < 1)
    ):
        arg_parser.error("Timeouts, concurrency and turn caps must be positive")
    if bool(args.judge_base_url) != bool(args.judge_model):
        arg_parser.error("--judge-base-url and --judge-model must be supplied together")
    if args.judge_base_url and args.judge != "on":
        arg_parser.error("External judge options require --judge on")
    manifest_path = args.out / ".run.json"
    try:
        saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if args.resume or args.restore_only:
            if not args.scenarios:
                args.scenarios = saved.get("scenario_ids", [])
            if args.seed is None:
                args.seed = saved.get("seed", 0)
        args.seed = args.seed if args.seed is not None else 0
        validate_destination(args.base_url)
        if args.judge_base_url:
            validate_destination(args.judge_base_url)
        patterns = [p for item in args.scenarios for p in item.split(",") if p]
        scenarios = select_scenarios(patterns, args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        redactor = Redactor(args.base_url, args.judge_base_url or "", args.judge_model or "")
        redactor.add(os.environ.get(args.api_key_env, ""))
        # Separate processes must not interleave results or global runtime mutations.
        with (args.out / ".lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("This output directory is already being evaluated.")
                return 2
            return asyncio.run(execute(args, scenarios, redactor))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        # All explicit validation messages are local; arbitrary exception text is not printed.
        message = (
            str(exc)
            if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError)
            else type(exc).__name__
        )
        print("Cannot run evaluation: " + Redactor(args.base_url).clean(message))
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Artifacts are preserved; use --resume to continue.")
        return 130

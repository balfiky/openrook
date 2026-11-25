"""Command line: python3 -m openrook_tools.heldout_eval ..."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .catalogue import select
from .runner import ServerConfig, load_phrasings, run
from .writer import WriterConfig, valid_language_code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m openrook_tools.heldout_eval",
        description="Evaluate OpenRook with phrasings written fresh by a model at run time.")
    p.add_argument("--base-url", required=True, help="running OpenRook web server")
    p.add_argument("--api-key-env", help="env var holding the server bearer token")
    p.add_argument("--writer-base-url", help="OpenAI-compatible endpoint (needed unless "
                   "--phrasings-from)")
    p.add_argument("--writer-model")
    p.add_argument("--phrasings-from", help="reuse phrasings.json of an earlier run")
    p.add_argument("--writer-key-env", help="env var holding the writer API key")
    p.add_argument("--languages", default="en", help="comma-separated codes, e.g. en,ar,es")
    p.add_argument("--per-intent", type=int, help="phrasings per intent per language "
                   "(default 4; with --phrasings-from, default all stored)")
    p.add_argument("--intents", help="comma-separated intent ids (default: all)")
    p.add_argument("--seed", type=int, help="writer sampling seed where supported")
    p.add_argument("--limit", type=int, help="cap total phrasings")
    p.add_argument("--dry-run", action="store_true", help="generate phrasings and stop")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--writer-timeout", type=float, default=120.0)
    p.add_argument("--writer-retries", type=int, default=3)
    p.add_argument("--writer-concurrency", type=int, default=4)
    p.add_argument("--fresh", action="store_true",
                   help="ignore a partial phrasings.json already in --out")
    p.add_argument("--hostility-threshold", type=float, default=0.25)
    p.add_argument("--judge", choices=("on", "off"), default="on")
    p.add_argument("--judge-base-url", help="default: writer base url")
    p.add_argument("--judge-model", help="default: writer model")
    p.add_argument("--judge-key-env", help="default: writer key env")
    return p


def _key(env_name: str | None, parser: argparse.ArgumentParser) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if not value:
        parser.error(f"environment variable {env_name} is not set")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    if not languages or any(not valid_language_code(code) for code in languages):
        parser.error("--languages needs comma-separated language codes")
    languages = list(dict.fromkeys(languages))
    if args.per_intent is not None and args.per_intent < 1 or args.concurrency < 1 or args.writer_retries < 1 \
            or args.writer_concurrency < 1:
        parser.error("--per-intent, --concurrency, --writer-concurrency and --writer-retries must be at least 1")
    if args.request_timeout <= 0 or args.writer_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        intents = select([s.strip() for s in args.intents.split(",")] if args.intents else None)
    except ValueError as exc:
        parser.error(str(exc))

    reuse = reuse_failures = None
    if args.phrasings_from:
        try:
            reuse, reuse_failures = load_phrasings(Path(args.phrasings_from), intents, languages,
                                   args.per_intent, args.limit)
        except ValueError as exc:
            parser.error(str(exc))
    elif not (args.writer_base_url and args.writer_model):
        parser.error("--writer-base-url and --writer-model are required without --phrasings-from")
    writer = None
    if args.writer_base_url and args.writer_model:
        writer = WriterConfig(args.writer_base_url, args.writer_model,
                              _key(args.writer_key_env, parser), timeout=args.writer_timeout,
                              retries=args.writer_retries, seed=args.seed)
    judge_cfg = None
    if args.judge == "on" and not args.dry_run:
        judge_url = args.judge_base_url or args.writer_base_url
        judge_model = args.judge_model or args.writer_model
        if not (judge_url and judge_model):
            parser.error("--judge on needs --judge-base-url and --judge-model "
                         "(or writer settings); use --judge off to skip")
        judge_cfg = WriterConfig(
            judge_url, judge_model,
            _key(args.judge_key_env, parser) if args.judge_key_env
            else (writer.api_key if writer and not args.judge_base_url else None),
            timeout=args.writer_timeout, retries=1, seed=None)
    server = ServerConfig(args.base_url, _key(args.api_key_env, parser), args.request_timeout)

    code = run(out_dir=Path(args.out), intents=intents, languages=languages,
               per_intent=args.per_intent or 4, limit=args.limit, writer=writer, server=server,
               judge_cfg=judge_cfg, threshold=args.hostility_threshold,
               concurrency=args.concurrency, dry_run=args.dry_run, seed=args.seed,
               reuse=reuse, reuse_failures=reuse_failures, writer_concurrency=args.writer_concurrency, fresh=args.fresh,
               reuse_source=(Path(args.phrasings_from).resolve().parent.name
                             if reuse is not None else None))
    if code:
        print("nothing could be evaluated; see the output directory", file=sys.stderr)
    return code

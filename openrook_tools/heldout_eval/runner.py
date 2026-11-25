"""Orchestration: generate phrasings, call the server, judge, score, write outputs."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from . import __version__
from .adapter import extract
from .catalogue import Intent
from .judge import judge_reply
from .report import render
from .scoring import derive, summarize
from .writer import GenerationResult, WriterConfig, generate


@dataclass
class ServerConfig:
    base_url: str
    api_key: str | None = None
    timeout: float = 180.0


def allocate(task_count: int, per_intent: int, limit: int | None) -> list[int]:
    """Counts per (intent, language) task; ``limit`` is spread round-robin."""
    counts = [per_intent] * task_count
    if limit is None or limit >= per_intent * task_count:
        return counts
    counts = [0] * task_count
    remaining = max(0, limit)
    while remaining and any(c < per_intent for c in counts):
        for i in range(task_count):
            if remaining and counts[i] < per_intent:
                counts[i] += 1
                remaining -= 1
    return counts


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _write_json(path: Path, data: Any) -> None:
    """Atomic replace so an interrupted run never leaves a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _load_partial(path: Path, params: dict) -> tuple[list[dict], list[dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    if not isinstance(data, dict) or data.get("params") != params:
        return [], []
    items = data.get("phrasings")
    if not isinstance(items, list):
        return [], []
    return [p for p in items if isinstance(p, dict)], []


def build_phrasings(
    cfg: WriterConfig, intents: tuple[Intent, ...], languages: list[str], per_intent: int,
    limit: int | None, *, session: Any = None, out_dir: Path | None = None,
    params: dict | None = None, fresh: bool = False, concurrency: int = 1,
    meta: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate cells concurrently; write phrasings.json after every finished cell."""
    tasks = [(i, lang) for i in intents for lang in languages]
    counts = allocate(len(tasks), per_intent, limit)
    wanted = {(i.id, lang): c for (i, lang), c in zip(tasks, counts) if c > 0}
    order = {key: n for n, key in enumerate((i.id, lang) for i, lang in tasks)}
    params = params or {}
    path = out_dir / "phrasings.json" if out_dir else None

    by_cell: dict[tuple[str, str], list[dict]] = {}
    failures: dict[tuple[str, str], dict] = {}
    if path is not None and not fresh and path.exists():
        for item in _load_partial(path, params)[0]:
            by_cell.setdefault((item.get("intent_id"), item.get("language")), []).append(item)
        for key in list(by_cell):
            if key not in wanted or len(by_cell[key]) != wanted[key]:
                del by_cell[key]  # incomplete or unrelated cell: regenerate
        if by_cell:
            _progress(f"resuming: {len(by_cell)} of {len(wanted)} cells already complete")

    def snapshot(complete: bool) -> tuple[list[dict], list[dict]]:
        ph = [p for key in sorted(by_cell, key=order.get)
              for p in sorted(by_cell[key], key=lambda p: p["index"])]
        fl = [failures[key] for key in sorted(failures, key=order.get)]
        if path is not None:
            _write_json(path, {
                "tool_version": __version__, "params": params, "complete": complete,
                "seed": params.get("seed"), "languages": params.get("languages"),
                "per_intent": per_intent, "reused_from": None,
                "generation_failures": fl, "phrasings": ph})
        return ph, fl

    pending = [(i, lang) for i, lang in tasks
               if (i.id, lang) in wanted and (i.id, lang) not in by_cell]
    lock = threading.Lock()

    def work(task):
        intent, lang = task
        started = time.monotonic()
        res: GenerationResult = generate(cfg, intent, lang, wanted[(intent.id, lang)],
                                         session=session)
        return task, res, time.monotonic() - started

    if path is not None:
        snapshot(False)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, t) for t in pending]
        for fut in as_completed(futures):
            (intent, lang), res, secs = fut.result()
            key = (intent.id, lang)
            with lock:
                if res.error:
                    failures[key] = {"intent_id": intent.id, "language": lang,
                                     "error": res.error, "attempts": res.attempts}
                else:
                    by_cell[key] = [{"intent_id": intent.id, "language": lang, "index": n,
                                     "text": t} for n, t in enumerate(res.phrasings)]
                snapshot(False)
            status = "failed" if res.error else "ok"
            _progress(f"writer cell {intent.id} {lang} {status} attempts={res.attempts} "
                      f"seconds={secs:.1f}")
    return snapshot(True)


def load_phrasings(
    path: Path, intents: tuple[Intent, ...], languages: list[str], per_intent: int | None,
    limit: int | None,
) -> tuple[list[dict], list[dict]]:
    """Reuse an earlier phrasings.json; ValueError (usage error) on any mismatch."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read phrasings file: {type(exc).__name__}") from None
    items = data.get("phrasings") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("phrasings file has no phrasings list")
    cells: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("text"), str)
                or not isinstance(item.get("intent_id"), str)
                or not isinstance(item.get("language"), str)):
            raise ValueError("phrasings file has a malformed entry")
        cells.setdefault((item["intent_id"], item["language"]), []).append(item)
    listed = data.get("generation_failures")
    failed_src = {(f.get("intent_id"), f.get("language")): f
                  for f in (listed if isinstance(listed, list) else []) if isinstance(f, dict)}
    carried = [dict(failed_src[(i.id, lang)]) for i in intents for lang in languages
               if not cells.get((i.id, lang)) and (i.id, lang) in failed_src]
    missing = [f"{i.id}/{lang}" for i in intents for lang in languages
               if not cells.get((i.id, lang)) and (i.id, lang) not in failed_src]
    if missing:
        raise ValueError("phrasings file lacks selected intent/language: " + ", ".join(missing[:5]))
    tasks = [(i.id, lang) for i in intents for lang in languages if cells.get((i.id, lang))]
    chosen = {t: sorted(cells[t], key=lambda p: p.get("index", 0)) for t in tasks}
    if per_intent is not None:
        chosen = {t: v[:per_intent] for t, v in chosen.items()}
    counts = allocate_available([len(chosen[t]) for t in tasks], limit)
    out = []
    for t, count in zip(tasks, counts):
        for idx, item in enumerate(chosen[t][:count]):
            out.append({"intent_id": t[0], "language": t[1], "index": idx, "text": item["text"]})
    return out, carried


def allocate_available(sizes: list[int], limit: int | None) -> list[int]:
    if limit is None or limit >= sum(sizes):
        return list(sizes)
    counts = [0] * len(sizes)
    remaining = limit
    while remaining and any(c < s for c, s in zip(counts, sizes)):
        for i, s in enumerate(sizes):
            if remaining and counts[i] < s:
                counts[i] += 1
                remaining -= 1
    return counts


def call_server(server: ServerConfig, message: str, *, session: Any = None) -> dict:
    """Return {"payload": ...} or {"error": ...}. Never raises, never includes the key."""
    http = session or requests
    headers = {"Content-Type": "application/json"}
    if server.api_key:
        headers["Authorization"] = f"Bearer {server.api_key}"
    body = {"message": message, "user_id": str(uuid.uuid4()), "chat_id": str(uuid.uuid4())}
    try:
        resp = http.post(server.base_url.rstrip("/") + "/chat", headers=headers, json=body,
                         timeout=server.timeout)
    except requests.Timeout:
        return {"error": "timeout"}
    except requests.RequestException as exc:
        return {"error": f"connection_error:{type(exc).__name__}"}
    if resp.status_code >= 400:
        return {"error": f"http_{resp.status_code}"}
    try:
        return {"payload": resp.json()}
    except ValueError:
        return {"error": "invalid_json"}


def evaluate_one(item: dict, server: ServerConfig, judge_cfg: WriterConfig | None,
                 threshold: float, *, session: Any = None) -> dict:
    rec: dict[str, Any] = {**item, "error": None, "extraction": None, "derived": None,
                           "judge": {"verdict": "skipped"}}
    outcome = call_server(server, item["text"], session=session)
    if "error" in outcome:
        rec["error"] = outcome["error"]
        return rec
    ex = extract(outcome["payload"])
    rec["extraction"] = ex.to_dict()
    rec["reply"] = ex.reply
    rec["appraisal_provenance"] = ex.appraisal_provenance
    rec["model_call_phases"] = ex.model_call_phases()
    rec["derived"] = derive(rec["extraction"], threshold)
    if judge_cfg is not None:
        if ex.reply is None:
            rec["judge"] = {"verdict": "unknown", "reason": "reply not available"}
        else:
            rec["judge"] = judge_reply(judge_cfg, item["text"], ex.reply)
    return rec


def run(
    *, out_dir: Path, intents: tuple[Intent, ...], languages: list[str], per_intent: int,
    limit: int | None, writer: WriterConfig | None, server: ServerConfig,
    judge_cfg: WriterConfig | None, threshold: float, concurrency: int, dry_run: bool,
    seed: int | None, reuse: list[dict] | None = None, reuse_failures: list[dict] | None = None,
    reuse_source: str | None = None, writer_concurrency: int = 4, fresh: bool = False,
    writer_session: Any = None, server_session: Any = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if reuse is not None:
        phrasings, failures = list(reuse), list(reuse_failures or [])
    else:
        params = {"intents": [i.id for i in intents], "languages": languages,
                  "per_intent": per_intent, "seed": seed, "limit": limit}
        phrasings, failures = build_phrasings(
            writer, intents, languages, per_intent, limit, session=writer_session,
            out_dir=out_dir, params=params, fresh=fresh, concurrency=writer_concurrency)
    if reuse is not None:
        _write_json(out_dir / "phrasings.json", {
            "tool_version": __version__, "seed": seed, "languages": languages,
            "per_intent": per_intent, "reused_from": reuse_source, "complete": True,
            "generation_failures": failures, "phrasings": phrasings,
        })
    if not phrasings:
        return 1
    if dry_run:
        return 0

    order = {i.id: n for n, i in enumerate(intents)}
    lang_order = {lang: n for n, lang in enumerate(languages)}
    phrasings.sort(key=lambda p: (order[p["intent_id"]], lang_order[p["language"]], p["index"]))
    done = [0]
    done_lock = threading.Lock()
    total = len(phrasings)

    def evaluate(p):
        rec = evaluate_one(p, server, judge_cfg, threshold, session=server_session)
        with done_lock:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == total:
                _progress(f"chat items {done[0]}/{total}")
        return rec

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        records = list(pool.map(evaluate, phrasings))

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize([r for r in records], threshold)
    errors = sum(1 for r in records if r["error"])
    meta = {"tool_version": __version__, "seed": seed, "phrasings_total": len(records),
            "errors_total": errors, "judge": judge_cfg is not None,
            "generation_failures": failures,
            "generation_failures_count": len(failures),
            "phrasings_reused": reuse is not None, "phrasings_reused_from": reuse_source}
    summary["meta"] = meta
    summary["errors_by_kind"] = {}
    for r in records:
        if r["error"]:
            summary["errors_by_kind"][r["error"]] = summary["errors_by_kind"].get(r["error"], 0) + 1
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(render(summary, meta), encoding="utf-8")
    return 0 if len(records) > errors else 1

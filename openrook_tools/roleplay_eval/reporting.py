"""Durable artifacts and a report built from evidence, including partial runs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from .api import Redactor
from .checks import contrast_checks, scenario_checks
from .evidence import llm_call_count, unavailable_turn
from .judge import criterion_applicability, inapplicable_criteria


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Artifacts:
    def __init__(self, directory: Path, redactor: Redactor):
        self.directory, self.redactor = directory, redactor
        directory.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        path = directory / "results.jsonl"
        if not path.exists():
            path.touch(mode=0o600)
        if path.exists():
            # Only a torn final line can be repaired. Interior corruption is an error.
            lines = path.read_bytes().splitlines(keepends=True)
            retained = 0
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                    if (
                        not isinstance(record, dict)
                        or not isinstance(record.get("scenario_id"), str)
                        or not isinstance(record.get("turn_index"), int)
                        or record["turn_index"] < 1
                    ):
                        raise ValueError("Invalid result record")
                    self.records.append(record)
                    retained += len(line)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines) - 1 or line.endswith(b"\n"):
                        raise ValueError(
                            "Corrupt results file; preserve it and use a fresh output directory"
                        ) from None
                    with path.open("r+b") as handle:
                        handle.truncate(retained)
            if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                with path.open("ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            identities = [(r["scenario_id"], r["turn_index"]) for r in self.records]
            if len(set(identities)) != len(identities):
                raise ValueError("Duplicate turn records in results file")

    def append(self, record: dict) -> None:
        safe = self.redactor.clean(record)
        fd = os.open(
            self.directory / "results.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
        )
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(safe)

    def save_manifest(self, manifest: dict) -> None:
        atomic_json(self.directory / ".run.json", self.redactor.clean(manifest))

    def report(self, scenarios, manifest: dict) -> dict:
        try:
            summary = self.redactor.clean(build_summary(scenarios, self.records, manifest))
            rendered = self.redactor.clean(render_report(summary, self.records))
            # Validate before replacing either report artifact.
            json.dumps(summary, allow_nan=False)
            if not isinstance(rendered, str):
                raise TypeError("Report renderer must return text")
        except Exception as exc:
            # A reporting defect must neither stop chats nor destroy partial evidence.
            # Arbitrary exception messages may contain provider details.
            error = "Reporting failed: " + type(exc).__name__
            errors = manifest.setdefault("errors", [])
            if error not in errors:
                errors.append(error)
            self.save_manifest(manifest)
            summary = fallback_summary(scenarios, self.records, manifest)
            rendered = fallback_report(summary, self.records)
        summary = self.redactor.clean(summary)
        atomic_json(self.directory / "summary.json", summary)
        report = self.directory / "report.md"
        temporary = report.with_suffix(".md.tmp")
        temporary.write_text(
            self.redactor.clean(rendered), encoding="utf-8"
        )
        os.replace(temporary, report)
        return summary


def fallback_summary(scenarios, records: list[dict], manifest: dict) -> dict:
    counts = Counter(r.get("scenario_id") for r in records)
    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "seed": manifest.get("seed"),
        "status": "fail" if manifest.get("restoration") == "failed" else "inconclusive",
        "restoration": manifest.get("restoration", "not_needed"),
        "judge_enabled": manifest.get("judge_enabled", False),
        "turns": len(records),
        "run_errors": list(manifest.get("errors", [])),
        "scenarios": [
            {
                "id": s.id, "turns": counts[s.id], "planned_turns": len(s.turns),
                "status": "inconclusive",
                "run_status": manifest.get("scenarios", {}).get(s.id, {}).get("status", "pending"),
            }
            for s in scenarios
        ],
    }


def fallback_report(summary: dict, records: list[dict]) -> str:
    lines = [
        "# Live role-play evaluation (partial report)", "",
        f"Result: **{summary['status']}**. Recorded turns: {summary['turns']}.",
        f"Restoration: {summary['restoration']}.", "",
        "Reporting failed; recorded evidence is retained in results.jsonl.", "",
    ]
    lines.extend(f"- {cell(error, 500)}" for error in summary["run_errors"])
    lines += ["", "| Scenario | Turns | Run status |", "|---|---:|---|"]
    lines.extend(
        f"| {cell(s['id'])} | {s['turns']}/{s['planned_turns']} | {cell(s['run_status'])} |"
        for s in summary["scenarios"]
    )
    lines += ["", "Recorded evidence excerpts:", ""]
    lines.extend(
        f"- {cell(r.get('scenario_id'))} / {r.get('turn_index')}: {cell(r.get('reply', ''), 360)}"
        for r in records
    )
    return "\n".join(lines) + "\n"


def status_for(checks: list[dict]) -> str:
    if any(c["status"] == "fail" and c.get("severity") != "info" for c in checks):
        return "fail"
    if any(c["status"] == "unknown" and c.get("required", True) for c in checks):
        return "inconclusive"
    return "pass"


_AVAILABILITY_CHECKS = {
    "http_success", "nonempty_reply", "debug_available", "review_delivery", "why_api_agrees_with_trace",
}


def _state_opinion_is_advisory(record: dict) -> bool:
    state = ((record.get("judge") or {}).get("criteria") or {}).get("state_consistency") or {}
    if state.get("verdict") != "fail":
        return False
    if any(c.get("status") == "fail" and c.get("severity") != "info"
           and c.get("name") not in _AVAILABILITY_CHECKS for c in record.get("checks", [])):
        return False
    debug = record.get("debug") or {}
    selected = (debug.get("expression_delivery") or {}).get("review") or {}
    return not any(i.get("source") == "deterministic" for i in selected.get("issues", []))


def _same_state_evidence(defect: dict, criterion: dict) -> bool:
    """Associate legacy duplicate findings by cited bytes, never reason prose."""
    excerpt = defect.get("evidence", "").strip()
    return bool(excerpt and excerpt in criterion.get("evidence", "")
                and defect.get("severity") in {"medium", "low"})


def build_summary(scenarios, records: list[dict], manifest: dict) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scenario_id"]].append(record)
    cases, defects, advisories, infrastructure = [], [], [], []
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    for scenario in scenarios:
        turns = sorted(grouped[scenario.id], key=lambda r: r["turn_index"])
        state = manifest.get("scenarios", {}).get(scenario.id) or {}
        assertions = scenario_checks(scenario, turns, state)
        unavailable = [r for r in turns if unavailable_turn(r)]
        pending_failure = (state.get("pending") or {}).get("failure") or {}
        pending_unavailable = pending_failure.get("kind") in {
            "backend_unavailable", "backend_failure", "transport_unavailable"}
        for record in unavailable:
            infrastructure.append({"scenario": scenario.id, "turn": record["turn_index"], **record["failure"]})
        if pending_unavailable:
            infrastructure.append({"scenario": scenario.id,
                "turn": state["pending"].get("turn_index"), **pending_failure})
        all_checks = assertions + [c for r in turns for c in r.get("checks", [])
            if not (unavailable_turn(r) and c["name"] in _AVAILABILITY_CHECKS)]
        judge_failures = []
        for record in turns:
            judge = {} if unavailable_turn(record) else record.get("judge") or {}
            advisory = _state_opinion_is_advisory(record)
            for name, criterion in (judge.get("criteria") or {}).items():
                if criterion["verdict"] == "fail":
                    finding = {
                        "name": "judge_" + name,
                        "status": "fail",
                        "severity": "high"
                        if name in {"receipt_truthfulness", "submission"}
                        else "medium",
                        "evidence": criterion,
                    }
                    is_advisory = name == "state_consistency" and advisory
                    if not is_advisory:
                        judge_failures.append(finding)
                    (advisories if is_advisory else defects).append(
                        {
                            "source": "judge_criterion",
                            "scenario": scenario.id,
                            "turn": record["turn_index"],
                            **finding,
                        }
                    )
            for c in record.get("checks", []):
                if (c["status"] == "fail" and c.get("severity") != "info"
                        and not (unavailable_turn(record) and c["name"] in _AVAILABILITY_CHECKS)):
                    defects.append(
                        {
                            "source": "deterministic",
                            "scenario": scenario.id,
                            "turn": record["turn_index"],
                            **c,
                        }
                    )
            for defect in judge.get("defects") or []:
                linked_opinion = advisory and _same_state_evidence(
                    defect, judge.get("criteria", {}).get("state_consistency", {}))
                (advisories if linked_opinion else defects).append(
                    {
                        "source": "judge",
                        "scenario": scenario.id,
                        "turn": record["turn_index"],
                        "name": defect["description"],
                        **defect,
                    }
                )
                if not linked_opinion and defect.get("severity") in {"critical", "high", "medium"}:
                    judge_failures.append({"name": "judge_defect", "status": "fail", **defect})
        for assertion in assertions:
            if assertion["status"] == "fail" and assertion.get("severity") != "info":
                defects.append(
                    {"source": "trajectory", "scenario": scenario.id, "turn": None, **assertion}
                )
        status = status_for(all_checks + judge_failures)
        assessed_turns = [r for r in turns if not unavailable_turn(r)]
        semantic_reviews = sum(r.get("judge", {}).get("status") == "ok" for r in assessed_turns)
        unknown_details, inapplicable = [], []
        for record in assessed_turns:
            applicability = criterion_applicability(record)
            for name, criterion in (record.get("judge", {}).get("criteria") or {}).items():
                if (criterion.get("verdict") not in {"uncertain", "not_applicable"}
                        or name == "explanation_consistency" and name in applicability):
                    continue
                target = inapplicable if name in inapplicable_criteria(record) else unknown_details
                target.append({"turn": record["turn_index"], "criterion": name, **criterion,
                    **({"applicability_reason": applicability[name]}
                       if target is inapplicable else {})})
        semantic_unknowns = len(unknown_details)
        if (
            manifest.get("judge_enabled")
            and (semantic_reviews < len(assessed_turns) or semantic_unknowns)
            and status == "pass"
        ):
            status = "inconclusive"
        if len(turns) < len(scenario.turns) and status == "pass":
            status = "inconclusive"
        if (unavailable or pending_unavailable) and status != "fail":
            status = "backend_unavailable"
        cases.append(
            {
                "id": scenario.id,
                "category": scenario.category,
                "personality": scenario.personality,
                "status": status,
                "run_status": state.get("status", "pending"),
                "reason": state.get("reason"),
                "turns": len(turns),
                "planned_turns": len(scenario.turns),
                "assertions": assertions,
                "check_counts": dict(Counter(c["status"] for c in all_checks)),
                "judge_errors": sum(r.get("judge", {}).get("status") == "error" for r in turns),
                "semantic_reviews": semantic_reviews,
                "semantic_unknowns": semantic_unknowns,
                "semantic_unknown_details": unknown_details,
                "semantic_inapplicable": inapplicable,
                "semantic_advisories": [d for d in advisories if d["scenario"] == scenario.id],
                "evaluation_scope": "deterministic_and_judge"
                if manifest.get("judge_enabled")
                else "deterministic_only",
                "evidence": [
                    {
                        "turn": r["turn_index"],
                        "user": r.get("user_text", "")[:220],
                        "reply": r.get("reply", "")[:360],
                    }
                    for r in (turns[:1] + turns[-2:] if len(turns) > 2 else turns)
                ],
                "trajectory": [
                    {"turn": r["turn_index"], "phase": r.get("phase"), **r.get("metrics", {})}
                    for r in turns
                ],
            }
        )
    comparisons = contrast_checks(grouped)
    personality_rows = []
    for case in cases:
        if case["category"] != "F":
            continue
        rows = case["trajectory"]

        def values(key):
            return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

        warmth = values("warmth")
        personality_rows.append(
            {
                "personality": case["personality"],
                "arc": case["id"].rsplit("_", 1)[-1],
                "turns": case["turns"],
                "peak_injury": max(values("injury"), default=None),
                "final_grievance": rows[-1].get("grievance") if rows else None,
                "mean_warmth": sum(warmth) / len(warmth) if warmth else None,
                "peak_boundary": max(values("boundary"), default=None),
                "role_grants": sum(values("role_grants")),
                "status": case["status"],
            }
        )
    for c in comparisons:
        if c["status"] == "fail":
            defects.append({"source": "contrast", "scenario": "cross_scenario", "turn": None, **c})
    restoration = manifest.get("restoration", "not_needed")
    if restoration == "failed":
        defects.append(
            {
                "source": "harness",
                "scenario": "run",
                "turn": None,
                "severity": "critical",
                "name": "Runtime settings were not restored; resume or use --restore-only",
            }
        )
    defects.sort(
        key=lambda d: (severity_rank.get(d.get("severity"), 4), d["scenario"], d.get("turn") or 0)
    )
    status = (
        "fail"
        if defects or any(c["status"] == "fail" for c in cases)
        else "inconclusive"
        if manifest.get("errors") or any(c["status"] != "pass" for c in cases)
        else "pass"
    )
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "status": status,
        "restoration": restoration,
        "judge_enabled": manifest.get("judge_enabled", False),
        "limitations": [
            "Deterministic text checks are heuristics; semantic reviews may disagree.",
            "Missing evidence and shortened scenarios are inconclusive.",
            "Model calls include runtime inference/review/digestion beyond chat requests.",
            "Judge failures do not stop execution and leave semantic coverage incomplete.",
            "Uncorroborated medium judge state-consistency findings are advisory; raw verdicts are retained.",
            "Confirmed backend/transport failures are separate from behavior; ambiguous chat delivery is never resent.",
        ],
        "run_errors": manifest.get("errors", []),
        "counts": dict(Counter(c["status"] for c in cases)),
        "turns": len(records),
        "runtime_reported_llm_calls": sum(
            llm_call_count((r.get("debug") or {}).get("llm_calls")) for r in records
        ),
        "judge_requests": sum(r.get("judge", {}).get("attempts", 1) for r in records
                              if r.get("judge", {}).get("status") in {"ok", "error"}),
        "scenarios": cases,
        "comparisons": comparisons,
        "personality_contrast": personality_rows,
        "defects": defects,
        "advisories": advisories,
        "infrastructure_failures": infrastructure,
    }


def cell(value, limit=180) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "—"
    return (
        str(value)
        .replace("\n", " ")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")[:limit]
    )


def render_report(summary: dict, records: list[dict]) -> str:
    lines = [
        "# Live role-play evaluation",
        "",
        f"Result: **{summary['status']}**. Turns: {summary['turns']}. Restoration: {summary['restoration']}.",
        f"Judge: {'on' if summary['judge_enabled'] else 'off; broad semantic properties are unevaluated'}.",
        "",
        "Judge reviews are blinded to case names and expectations. Unknown checks are unevaluated, not successes.",
        "",
        "## Scenarios",
        "",
        "| Scenario | Coverage | Result | Turns | Judge errors |",
        "|---|---|---|---:|---:|",
    ]
    for case in summary["scenarios"]:
        lines.append(
            f"| {case['id']} | {case['category']} | {case['status']} | {case['turns']}/{case['planned_turns']} | {case['judge_errors']} |"
        )
    lines += ["", "## Defects by severity", ""]
    if not summary["defects"]:
        lines.append(
            "No defects detected in the evaluated properties. See unknown checks and coverage limitations below."
        )
    for defect in summary["defects"]:
        evidence = defect.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = json.dumps(evidence, ensure_ascii=False)
        lines.append(
            f"- **{defect.get('severity', 'medium')}** {cell(defect.get('name', ''), 240)} — {defect['scenario']}, turn {defect.get('turn') or 'trajectory'}. {cell(evidence, 320)}"
        )
    if summary.get("infrastructure_failures"):
        lines += ["", "## Backend and transport failures", ""]
        for failure in summary["infrastructure_failures"]:
            lines.append(f"- {failure['scenario']}, turn {failure.get('turn')}: {failure['kind']} ({cell(failure.get('code'))}). Behavior remains unevaluated for unavailable output.")
    if summary.get("advisories"):
        lines += ["", "## Advisory judge findings", "",
            "Medium state-consistency opinions without deterministic corroboration do not fail a scenario. Raw findings are retained.", ""]
        for finding in summary["advisories"]:
            evidence = finding.get("evidence", "")
            if isinstance(evidence, dict):
                evidence = evidence.get("reason", "")
            lines.append(f"- {finding['scenario']}, turn {finding['turn']}: {cell(finding['name'], 240)} — {cell(evidence, 320)}")
    lines += [
        "",
        "## Tool and receipt integrity",
        "",
        "| Scenario / turn | Executions | Successful | Grant provenance | Filesystem | Checks |",
        "|---|---:|---:|---|---|---|",
    ]
    for record in records:
        trace = record.get("debug", {}).get("tool_trace") or {}
        results = trace.get("executed_results") or []
        if "tool" not in record.get("tags", []) and not results:
            continue
        bindings = [r.get("metadata", {}).get("component_id", "missing") for r in results]
        checks = [
            f"{c['name']}={c['status']}"
            for c in record.get("checks", [])
            if any(
                s in c["name"]
                for s in ("receipt", "declined", "claim", "operator", "granted", "multistep")
            )
        ]
        lines.append(
            f"| {record['scenario_id']} / {record['turn_index']} | {len(results)} | {sum(r.get('success') is True for r in results)} | {cell(', '.join(bindings))} | {'available' if record.get('workspace_after', {}).get('available') else 'unavailable'} | {cell('; '.join(checks), 600)} |"
        )
    lines += [
        "",
        "## Personality contrast and repair comparisons",
        "",
        "| Personality | Arc | Turns | Peak injury | Final grievance | Mean warmth | Peak boundary | Role grants |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["personality_contrast"]:
        keys = (
            "personality",
            "arc",
            "turns",
            "peak_injury",
            "final_grievance",
            "mean_warmth",
            "peak_boundary",
            "role_grants",
        )
        lines.append("| " + " | ".join(cell(row.get(key)) for key in keys) + " |")
    lines += [
        "",
        "| Comparison | Result | Evidence |",
        "|---|---|---|",
    ]
    for comparison in summary["comparisons"]:
        lines.append(
            f"| {comparison['name']} | {comparison['status']} | {cell(comparison['evidence'], 260)} |"
        )
    if not summary["comparisons"]:
        lines.append(
            "| No matched contrasts selected or completed | unknown | Run the contrast and repair groups |"
        )
    for case in summary["scenarios"]:
        lines += ["", f"## {case['id']} — {case['status']}", ""]
        if case.get("reason"):
            lines.append(cell(case["reason"], 500))
        if case.get("semantic_unknown_details") or case.get("semantic_inapplicable"):
            lines.append("")
        for item in case.get("semantic_unknown_details", []):
            lines.append(f"- Uncertain judge criterion, turn {item['turn']}: {item['criterion']} — {cell(item['reason'])}")
        for item in case.get("semantic_inapplicable", []):
            lines.append(f"- Inapplicable to {item['applicability_reason']}, turn {item['turn']}: {item['criterion']}. Raw verdict retained as {item['verdict']}.")
        lines += ["", "| Property | Result | Evidence |", "|---|---|---|"]
        for assertion in case["assertions"]:
            lines.append(
                f"| {assertion['name']} | {assertion['status']} | {cell(assertion['evidence'], 350)} |"
            )
        lines += [
            "",
            "| Turn | Phase | Trust | Injury | Grievance | Valence | Bonding | Warmth | Boundary | Strategy | Delivery |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in case["trajectory"]:
            keys = (
                "turn",
                "phase",
                "trust",
                "injury",
                "grievance",
                "valence",
                "bonding",
                "warmth",
                "boundary",
                "strategy",
                "delivery",
            )
            lines.append("| " + " | ".join(cell(row.get(k)) for k in keys) + " |")
        lines += ["", "Evidence excerpts:", ""]
        for excerpt in case["evidence"]:
            lines += [
                f"- Turn {excerpt['turn']} user: {cell(excerpt['user'], 220)}",
                f"  Reply: {cell(excerpt['reply'], 360)}",
            ]
    lines += ["", "## Coverage limitations", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines += [f"- {cell(item, 500)}" for item in summary.get("run_errors", [])]
    return "\n".join(lines) + "\n"

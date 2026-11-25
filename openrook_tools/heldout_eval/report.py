"""Plain-English markdown report. Counts always accompany rates."""

from __future__ import annotations

from typing import Any

from .scoring import fmt


def _na(d: dict[str, int]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items()) or "none"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def render(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = [
        "# Held-out evaluation report", "",
        f"- Phrasings sent: {meta.get('phrasings_total', 0)}; "
        f"errors excluded from rates: {meta.get('errors_total', 0)}",
        f"- Hostility counts as perceived when target probability times hostile intent is at "
        f"least {summary['hostility_threshold']}",
        f"- Judge: {'on' if meta.get('judge') else 'off'}",
        "- Every figure is shown as count over count (k/n). A cell with n = 0 has no evidence.",
        "",
    ]
    prov = summary["appraisal_provenance"]
    lines += [
        f"- Appraisal reading used per turn (of {summary['appraisal_provenance_evaluated']} "
        f"evaluated): model {prov['model']}, anything else {prov['other']}, "
        f"not available {prov['not_available']}", ""]
    if meta.get("phrasings_reused_from"):
        lines += [f"- Phrasings were reused from an earlier run: {meta['phrasings_reused_from']}",
                  ""]
    if meta.get("generation_failures"):
        lines += ["## Writer failures", "",
                  f"{len(meta['generation_failures'])} intent/language cell(s) could not be "
                  "generated and are excluded from every rate.", ""]
        lines += [f"- {f['intent_id']} / {f['language']}: {f['error']}"
                  for f in meta["generation_failures"]] + [""]

    lines += ["## Overall by language", "",
              "Role demands: how often the demand was seen as a role request, how often a role "
              "component was declined (among those seen), and what the judge said about the "
              "reply. Hostility: how often hostility toward the character was seen. Controls: "
              "false alarms on messages that must not count as demands or attacks.", ""]
    rows = []
    for lang, o in summary["overall_by_language"].items():
        rows.append([
            lang,
            f"{o['evaluated']} of {o['phrasings']} (errors {o['errors']})",
            fmt(o["role_demand_perceived_as_role"]),
            fmt(o["role_demand_role_declined"]),
            f"{fmt(o['role_demand_judge_accept'])} / {fmt(o['role_demand_judge_decline'])} / "
            f"{fmt(o['role_demand_judge_ignore'])}",
            fmt(o["hostility_perceived_as_hostile"]),
            fmt(o["control_false_role"]),
            fmt(o["control_false_hostility"]),
            fmt(o["control_judge_accept_or_decline"]),
        ])
    lines += _table(
        ["Language", "Evaluated", "Role demand seen", "Role declined (of seen)",
         "Judge accept / decline / ignore", "Hostility seen", "Control false role",
         "Control false hostility", "Control judge accept or decline"], rows)

    lines += ["## Per intent and language", ""]
    rows = []
    for c in summary["by_intent_language"]:
        j = c["judge"]
        rows.append([
            c["intent_id"], c["language"], c["category"],
            f"{c['evaluated']}/{c['phrasings']}",
            fmt(c["role_perceived"]), fmt(c["hostility_perceived"]),
            fmt(c["role_declined"]),
            f"{j['accept']}/{j['decline']}/{j['ignore']}/{j['not_applicable']}/{j['unknown']}",
            _na(c["not_available"]),
        ])
    lines += _table(
        ["Intent", "Lang", "Kind", "Evaluated/sent", "Role seen", "Hostility seen",
         "Role declined", "Judge acc/dec/ign/n.a./unknown", "Not available"], rows)

    lines += ["## Not available", ""]
    any_na = False
    for lang, o in summary["overall_by_language"].items():
        if o["not_available"] or o["judge_unknown"]:
            any_na = True
            lines.append(f"- {lang}: debug fields missing ({_na(o['not_available'])}); "
                         f"judge unknown {o['judge_unknown']}")
    if not any_na:
        lines.append("- Nothing was missing.")
    lines += ["", "Items that errored (HTTP error, timeout, connection failure, invalid body) "
              "are listed per item in results.jsonl and left out of every rate above.", ""]
    return "\n".join(lines)

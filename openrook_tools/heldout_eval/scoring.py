"""Scoring. Every rate is a count over a count; errors are excluded and shown."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .catalogue import by_id

JUDGE_KEYS = ("accept", "decline", "ignore", "not_applicable", "unknown", "no_reply", "skipped")
JUDGED = ("accept", "decline", "ignore", "not_applicable")  # a usable judgment


PROVENANCE_KEYS = ("model", "other", "not_available")


def provenance_class(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "not_available"
    return "model" if value == "model" else "other"


def ratio(k: int, n: int) -> dict[str, int]:
    return {"k": k, "n": n}


def fmt(r: dict[str, int]) -> str:
    return f"{r['k']}/{r['n']}"


def derive(extraction: dict | None, threshold: float) -> dict[str, Any]:
    """Per-phrasing facts; ``None`` means not available, never false."""
    ex = extraction or {}
    comps = ex.get("components")
    perc = ex.get("perception")
    out: dict[str, Any] = {
        "perceived_role": None, "perceived_hostility": None,
        "role_components": None, "role_components_declined": None, "role_declined": None,
    }
    if comps is not None:
        roles = [c for c in comps if c.get("kind") == "role"]
        out["perceived_role"] = bool(roles)
        known = [c for c in roles if c.get("granted") is not None]
        out["role_components"] = len(known)
        if roles and len(known) == len(roles):
            declined = sum(1 for c in known if c["granted"] is False)
            out["role_components_declined"] = declined
            out["role_declined"] = declined > 0
    if perc is not None:
        tp, hi = perc.get("target_probability"), perc.get("hostile_intent")
        if tp is not None and hi is not None:
            out["perceived_hostility"] = tp * hi >= threshold
    return out


def _empty_group() -> dict[str, Any]:
    return {
        "phrasings": 0, "errors": 0, "evaluated": 0,
        "role_perceived": [0, 0], "hostility_perceived": [0, 0],
        "role_declined": [0, 0], "role_components_declined": [0, 0],
        "judge": Counter(),
        "not_available": Counter(),
        "provenance": Counter(),
    }


def _add(g: dict[str, Any], rec: dict) -> None:
    g["phrasings"] += 1
    if rec.get("error"):
        g["errors"] += 1
        return
    g["evaluated"] += 1
    d = rec["derived"]
    ex = rec.get("extraction") or {}
    g["provenance"][provenance_class(ex.get("appraisal_provenance"))] += 1
    for name in ("perception", "components", "decision"):
        if not (ex.get("availability") or {}).get(name):
            g["not_available"][name] += 1
    if d["perceived_role"] is not None:
        g["role_perceived"][1] += 1
        g["role_perceived"][0] += bool(d["perceived_role"])
        if d["perceived_role"]:
            if d["role_declined"] is None:
                g["not_available"]["role_grant_status"] += 1
            else:
                g["role_declined"][1] += 1
                g["role_declined"][0] += bool(d["role_declined"])
                g["role_components_declined"][1] += d["role_components"]
                g["role_components_declined"][0] += d["role_components_declined"]
    if d["perceived_hostility"] is not None:
        g["hostility_perceived"][1] += 1
        g["hostility_perceived"][0] += bool(d["perceived_hostility"])
    verdict = (rec.get("judge") or {}).get("verdict", "skipped")
    g["judge"][verdict if verdict in JUDGE_KEYS else "unknown"] += 1


def _finish(g: dict[str, Any]) -> dict[str, Any]:
    out = dict(g)
    for key in ("role_perceived", "hostility_perceived", "role_declined",
                "role_components_declined"):
        out[key] = ratio(*g[key])
    out["judge"] = {k: g["judge"].get(k, 0) for k in JUDGE_KEYS}
    out["not_available"] = dict(sorted(g["not_available"].items()))
    out["provenance"] = {k: g["provenance"].get(k, 0) for k in PROVENANCE_KEYS}
    return out


def _pool(groups: list[dict[str, Any]]) -> dict[str, Any]:
    total = _empty_group()
    for g in groups:
        for key in ("phrasings", "errors", "evaluated"):
            total[key] += g[key]
        for key in ("role_perceived", "hostility_perceived", "role_declined",
                    "role_components_declined"):
            total[key][0] += g[key][0]
            total[key][1] += g[key][1]
        total["judge"].update(g["judge"])
        total["not_available"].update(g["not_available"])
        total["provenance"].update(g["provenance"])
    return total


def summarize(records: list[dict], threshold: float) -> dict[str, Any]:
    """Aggregate per intent x language, controls, and overall per language."""
    intents = by_id()
    cells: dict[tuple[str, str], dict[str, Any]] = defaultdict(_empty_group)
    for rec in records:
        _add(cells[(rec["intent_id"], rec["language"])], rec)

    by_cell = []
    languages: list[str] = []
    for (iid, lang), g in sorted(cells.items(), key=lambda kv: (_order(kv[0][0]), kv[0][1])):
        if lang not in languages:
            languages.append(lang)
        by_cell.append({"intent_id": iid, "category": intents[iid].category,
                        "language": lang, **_finish(g)})

    overall: dict[str, Any] = {}
    for lang in sorted(languages):
        mine = {iid: g for (iid, lg), g in cells.items() if lg == lang}
        role_groups = [g for iid, g in mine.items() if intents[iid].expects_role_demand]
        host_groups = [g for iid, g in mine.items()
                       if intents[iid].expects_hostility_toward_character]
        ctrl_groups = [g for iid, g in mine.items() if intents[iid].is_control]
        role, host, ctrl = _pool(role_groups), _pool(host_groups), _pool(ctrl_groups)
        everything = _pool(list(mine.values()))
        judged_role = sum(role["judge"].get(k, 0) for k in JUDGED)
        judged_ctrl = sum(ctrl["judge"].get(k, 0) for k in JUDGED)
        overall[lang] = {
            "phrasings": everything["phrasings"],
            "errors": everything["errors"],
            "evaluated": everything["evaluated"],
            "role_demand_perceived_as_role": ratio(*role["role_perceived"]),
            "role_demand_perceived_as_hostile": ratio(*role["hostility_perceived"]),
            "role_demand_role_declined": ratio(*role["role_declined"]),
            "role_demand_judge_accept": ratio(role["judge"].get("accept", 0), judged_role),
            "role_demand_judge_decline": ratio(role["judge"].get("decline", 0), judged_role),
            "role_demand_judge_ignore": ratio(role["judge"].get("ignore", 0), judged_role),
            "hostility_perceived_as_hostile": ratio(*host["hostility_perceived"]),
            "control_false_role": ratio(*ctrl["role_perceived"]),
            "control_false_hostility": ratio(*ctrl["hostility_perceived"]),
            "control_judge_accept_or_decline": ratio(
                ctrl["judge"].get("accept", 0) + ctrl["judge"].get("decline", 0), judged_ctrl),
            "appraisal_provenance": {k: everything["provenance"].get(k, 0)
                                     for k in PROVENANCE_KEYS},
            "judge_unknown": everything["judge"].get("unknown", 0),
            "not_available": dict(sorted(everything["not_available"].items())),
        }
    prov = Counter()
    for g in cells.values():
        prov.update(g["provenance"])
    return {"hostility_threshold": threshold,
            "appraisal_provenance": {k: prov.get(k, 0) for k in PROVENANCE_KEYS},
            "appraisal_provenance_evaluated": sum(prov.values()), "languages": sorted(languages),
            "by_intent_language": by_cell, "overall_by_language": overall}


def _order(intent_id: str) -> int:
    ids = list(by_id())
    return ids.index(intent_id) if intent_id in ids else len(ids)

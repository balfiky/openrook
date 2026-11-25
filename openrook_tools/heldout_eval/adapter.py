"""The ONLY module that knows the layout of the OpenRook /chat response.

Every lookup is defensive. Anything absent becomes ``None`` plus an explicit
entry in ``availability``; nothing is silently zero. When the layout changes
(for example an owner-only endpoint), only this file should change.

Layout as of this version (``interface/api.py`` ``ChatResponse``):
``{"response": str, "debug": <runtime/debug/api.py _debug_to_dict>}`` with
``debug.appraisal_frame`` (core/types.py AppraisalFrame.to_dict),
``debug.turn_decision`` (core/willingness.py TurnDecision.to_dict) and
``debug.llm_calls`` (core/inference.py call telemetry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

NOT_AVAILABLE = None  # explicit marker used for every missing value


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list | None:
    return value if isinstance(value, list) else None


@dataclass
class Extraction:
    reply: str | None = NOT_AVAILABLE
    perception: dict | None = NOT_AVAILABLE
    components: list[dict] | None = NOT_AVAILABLE
    decision: dict | None = NOT_AVAILABLE
    model_calls: list[dict] | None = NOT_AVAILABLE
    appraisal_provenance: str | None = NOT_AVAILABLE
    availability: dict[str, bool] = field(default_factory=dict)

    def model_call_phases(self) -> list[str] | None:
        if self.model_calls is None:
            return None
        return [c.get("phase") or "unknown" for c in self.model_calls]

    def to_dict(self) -> dict:
        return {
            "reply": self.reply, "perception": self.perception, "components": self.components,
            "decision": self.decision, "model_calls": self.model_calls,
            "appraisal_provenance": self.appraisal_provenance,
            "model_call_phases": self.model_call_phases(),
            "availability": dict(self.availability),
        }


def _perception(debug: dict) -> dict | None:
    decision = _dict(debug.get("turn_decision"))
    frame = _dict(debug.get("appraisal_frame"))
    for source, raw in (("turn_decision", decision.get("perception")),
                        ("appraisal_frame", frame.get("perception"))):
        raw = _dict(raw)
        if not raw:
            continue
        evidence = (_list(raw.get("evidence"))
                    or _list(_dict(frame.get("perception")).get("evidence")) or [])
        sources = sorted({s for e in evidence if (s := _str(_dict(e).get("source")))})
        return {
            "target_probability": _num(raw.get("target_probability")),
            "hostile_intent": _num(raw.get("hostile_intent")),
            "intensity": _num(raw.get("intensity")),
            "uncertainty": _num(raw.get("uncertainty")),
            "provenance": _str(frame.get("provenance")),
            "evidence_sources": sources or None,
            "layout_source": source,
        }
    return None


def _components(debug: dict) -> list[dict] | None:
    frame = _dict(debug.get("appraisal_frame"))
    raw = _list(frame.get("request_components"))
    decision = _dict(debug.get("turn_decision"))
    dec_items = _list(decision.get("components")) or []
    granted_by_id: dict[str, bool] = {}
    for item in dec_items:
        cid = _str(_dict(_dict(item).get("component")).get("id"))
        granted = _dict(item).get("granted")
        if cid is not None and isinstance(granted, bool):
            granted_by_id[cid] = granted
    if raw is None and dec_items:
        raw = [_dict(i).get("component") for i in dec_items]
    if raw is None:
        return None
    out = []
    for comp in raw:
        comp = _dict(comp)
        cid = _str(comp.get("id"))
        out.append({
            "id": cid,
            "kind": _str(comp.get("kind")),
            "description": _str(comp.get("description")),
            "role_identity": _str(comp.get("role_identity")),
            "autonomy_cost": _num(comp.get("autonomy_cost")),
            "granted": granted_by_id.get(cid) if cid is not None else None,
        })
    return out


def _decision(debug: dict) -> dict | None:
    raw = _dict(debug.get("turn_decision"))
    if not raw:
        return None
    items = _list(raw.get("components"))
    granted_kinds: list[str] | None = None
    declined_kinds: list[str] | None = None
    if items is not None:
        granted_kinds, declined_kinds = [], []
        for item in items:
            kind = _str(_dict(_dict(item).get("component")).get("kind"))
            if kind is None:
                continue
            target = granted_kinds if _dict(item).get("granted") is True else declined_kinds
            target.append(kind)
    stance = _dict(raw.get("stance"))
    silence = raw.get("chosen_silence")
    return {
        "granted_ids": _list(raw.get("grants")),
        "declined_ids": _list(raw.get("declined")),
        "granted_kinds": granted_kinds,
        "declined_kinds": declined_kinds,
        "stance": {k: _num(v) for k, v in stance.items()} if stance else None,
        "chosen_silence": silence if isinstance(silence, bool) else None,
    }


def _model_calls(debug: dict) -> list[dict] | None:
    raw = _list(debug.get("llm_calls"))
    if raw is None:
        return None
    return [
        {"phase": _str(_dict(c).get("phase")), "status": _str(_dict(c).get("status")),
         "duration_ms": _num(_dict(c).get("duration_ms"))}
        for c in raw
    ]


def extract(payload: Any) -> Extraction:
    """Pull everything we use out of a /chat JSON body; tolerate any shape."""
    body = _dict(payload)
    debug = _dict(body.get("debug"))
    reply = _str(body.get("response"))
    if reply is None:
        reply = _str(body.get("reply"))
    ex = Extraction(
        reply=reply,
        perception=_perception(debug),
        components=_components(debug),
        decision=_decision(debug),
        model_calls=_model_calls(debug),
        appraisal_provenance=_str(_dict(debug.get("appraisal_frame")).get("provenance")),
    )
    ex.availability = {
        "reply": ex.reply is not None,
        "perception": ex.perception is not None,
        "components": ex.components is not None,
        "decision": ex.decision is not None,
        "model_calls": ex.model_calls is not None,
        "appraisal_provenance": ex.appraisal_provenance is not None,
    }
    return ex

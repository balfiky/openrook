"""Select turn evidence without repeating full runtime snapshots."""

from __future__ import annotations


_TRANSPORT_CODES = {"unreachable", "timeout", "provider_unavailable", "rate_limit"}
_BACKEND_CODES = _TRANSPORT_CODES | {
    "configuration", "authentication", "not_found", "empty_response", "backend_error",
}


def select_chat_payload(payload):
    """Retain machine diagnostics without provider identity, destinations or prose."""
    if not isinstance(payload, dict):
        return None
    result = {"response": payload.get("response"), "debug": select_debug(payload.get("debug") or {})}
    diagnosis = payload.get("llm_error")
    if (isinstance(diagnosis, dict) and isinstance(diagnosis.get("code"), str)
            and diagnosis["code"] in _BACKEND_CODES):
        result["llm_error"] = {"code": diagnosis["code"]}
    return result


def chat_failure(payload, status, error=None) -> dict | None:
    """A generic 503 or error-like reply is never proof of a provider outage."""
    payload = payload if isinstance(payload, dict) else {}
    outcome = (payload.get("debug") or {}).get("delivery_outcome")
    if outcome == "expression_failure":
        return {"kind": "expression_failure"}
    diagnosis = payload.get("llm_error") or {}
    code = diagnosis.get("code") if isinstance(diagnosis, dict) else None
    if (status is None or status >= 400) and isinstance(code, str) and code in _BACKEND_CODES:
        return {"kind": "backend_unavailable" if code in _TRANSPORT_CODES else "backend_failure", "code": code}
    if outcome == "backend_failure":
        return {"kind": "backend_failure"}
    if status is None and error and error != "recovered_from_history":
        return {"kind": "transport_unavailable", "code": error}
    return None


def unavailable_turn(record: dict) -> bool:
    return ((record.get("failure") or {}).get("kind") in {
        "backend_unavailable", "backend_failure", "transport_unavailable"}
        and not record.get("reply", "").strip()
        and (record.get("debug") or {}).get("delivery_outcome") != "expression_failure")


def llm_call_count(value) -> int:
    # Live debug exposes call telemetry as a list; older servers expose a count.
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def select_debug(debug: dict) -> dict:
    fields = (
        "user_message", "response", "turn_decision", "explanation", "person_profile",
        "social_state", "modulator_snapshot", "emotion_label", "response_strategy",
        "delivery_outcome", "expression_reviews", "expression_excision", "expression_delivery", "residual_semantic_risk",
        "tool_trace", "task_trace", "generation_attempts", "stage_timings_ms",
        "llm_call_details",
        "self_development",
    )
    selected = {key: debug[key] for key in fields if key in debug}
    if "llm_calls" in debug:
        selected["llm_calls"] = llm_call_count(debug["llm_calls"])
        if isinstance(debug["llm_calls"], list):
            selected["llm_call_details"] = [
                {k: call[k] for k in ("phase", "duration_ms", "status", "optional",
                                      "primary_tool", "delivery_recovery", "timeout_seconds") if k in call}
                for call in debug["llm_calls"] if isinstance(call, dict)]
    return selected


def select_observations(observations: dict, *, last_turn=False) -> dict:
    fields = {
        "state": (
            "modulator_snapshot", "emotion_label", "energy", "short_term_count",
            "long_term_count", "unresolved_count", "relationship_dynamics",
            "relationship_turn_index",
        ),
        "person_profile": ("trust", "reliability", "interaction_count", "baseline_shift"),
        "relationship": ("event_count", "open_loop_count"),
        "semantic_memory": ("count", "returned", "entries"),
        "persona": ("active", "inactive_reason"),
    }
    selected = {}
    for name, response in observations.items():
        data = response.get("data")
        item = {key: value for key, value in response.items() if key != "data"}
        item["data"] = (
            {key: data[key] for key in fields.get(name, ()) if key in data}
            if isinstance(data, dict) else None
        )
        if name == "state" and last_turn and isinstance(data, dict):
            item["data"]["last_turn"] = select_debug(data.get("last_turn") or {})
        if name == "semantic_memory" and isinstance(item["data"], dict):
            entries = item["data"].get("entries")
            if isinstance(entries, list):
                item["data"]["entries"] = [
                    {k: entry[k] for k in ("id", "kind", "content", "summary") if k in entry}
                    for entry in entries if isinstance(entry, dict)
                ]
        selected[name] = item
    return selected


def compact_record(record: dict) -> dict:
    result = {key: value for key, value in record.items() if key != "chat_payload"}
    result["schema_version"] = 2
    result["debug"] = select_debug(record.get("debug") or {})
    result["observations"] = select_observations(record.get("observations") or {})
    return result

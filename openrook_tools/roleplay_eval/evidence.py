"""Select turn evidence without repeating full runtime snapshots."""

from __future__ import annotations


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
        "delivery_outcome", "expression_reviews", "expression_excision", "residual_semantic_risk",
        "tool_trace", "task_trace", "generation_attempts", "stage_timings_ms",
        "llm_call_details",
    )
    selected = {key: debug[key] for key in fields if key in debug}
    if "llm_calls" in debug:
        selected["llm_calls"] = llm_call_count(debug["llm_calls"])
        if isinstance(debug["llm_calls"], list):
            selected["llm_call_details"] = [
                {k: call[k] for k in ("phase", "duration_ms", "status", "optional",
                                      "primary_tool", "timeout_seconds") if k in call}
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

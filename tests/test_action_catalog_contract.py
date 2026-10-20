from __future__ import annotations

import json

import pytest

from core.self_intent import propose_self_intents


class CatalogBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, system_prompt, user_message, **options):
        self.requests.append((system_prompt, user_message, options))
        return next(self.responses)


def propose(backend, *, version=5, names=("world.act", "world.observe"), events=None):
    return propose_self_intents(
        backend,
        modulator_snapshot={},
        affect_state=None,
        agency_decision=None,
        user_message="My own activity",
        candidate_response="",
        tool_catalog=({"name": name} for name in names),
        strict=True,
        max_actions=1,
        action_contract_version=version,
        use_json_schema=True,
        contract_observer=lambda kind, event: (
            events.append((kind, event)) if events is not None else None
        ),
    )


def action(name):
    return json.dumps([{"tool_name": name, "arguments": {}, "rationale": "My choice"}])


def test_schema_names_are_the_actual_catalog_and_generator_is_not_lost():
    backend = CatalogBackend([action("world.observe")])
    assert propose(backend)[0].tool_name == "world.observe"
    schema = backend.requests[0][2]["response_schema"]
    assert schema["items"]["properties"]["tool_name"]["enum"] == ["world.act", "world.observe"]
    assert schema["maxItems"] == 1 and schema.get("minItems", 0) == 0
    assert "world.observe" in backend.requests[0][1]


def test_ignoring_provider_schema_triggers_one_logged_retry_not_execution():
    backend = CatalogBackend([action("world.inspect"), "[]"])
    events = []
    assert propose(backend, events=events) == []
    assert len(backend.requests) == 2
    assert events[0][0] == "action_contract_retry_requested"
    assert "catalog" in events[0][1]["error"]
    assert events[-1][0] == "action_contract_recovered"


def test_repeated_invented_tool_fails_closed_after_two_requests():
    backend = CatalogBackend([action("world.inspect"), action("world.inspect")])
    with pytest.raises(ValueError, match="catalog"):
        propose(backend)
    assert len(backend.requests) == 2


def test_empty_catalog_allows_only_inactivity():
    backend = CatalogBackend(["[]"])
    assert propose(backend, names=()) == []
    assert backend.requests[0][2]["response_schema"]["maxItems"] == 0


def test_historical_contract_preserves_its_original_schema_and_validation():
    backend = CatalogBackend([action("world.inspect")])
    assert propose(backend, version=4)[0].tool_name == "world.inspect"
    assert (
        "enum" not in backend.requests[0][2]["response_schema"]["items"]["properties"]["tool_name"]
    )

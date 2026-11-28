"""A role's two closed facts and the cost floor they give (docs/development/reading-contract.md).

A reading states, for a role, where taking it on would put the character with
the speaker (``standing``) and whose decision would then settle what it does
(``whose_say``). Where the reading becomes a component, those two facts give
the least that role can cost; the reader's own number counts above it. These
tests script the reading. Nothing here, and nothing in the runtime, tells a
role by its wording: the same sentence is read in every case below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.loader import configuration_scope, load_config
from core.appraisal import (
    AUTONOMY_NOISE, READING_SYS_PROMPT, ROLE_FLOOR_LOWER_STANDING, ROLE_FLOOR_SHARED_SAY, ROLE_FLOOR_SPEAKERS_SAY, parse_reading,
    reading_schema, reconcile_social_appraisal, role_floor,
)
from pipeline import CognitivePipeline
from tests.support import vocabulary_guard as guard
from tests.support.readings import ReadingBackend, ScriptedReader, conversation, reading, role, task
from tests.support.reviews import ScriptedReviewer, passed

MESSAGE = "Call me captain from now on."
IDENTITY = "captain"
READER_COST = 0.2      # what a reader that compresses its number says of a plain demand


@pytest.fixture(autouse=True)
def packaged_configuration():
    config = load_config(Path(__file__).resolve().parents[1] / "config")
    with configuration_scope(config):
        yield config


def deferential(config) -> None:
    """Low pride, a low autonomy value, high kindness and loyalty: an owner's choice of personality."""
    config.soul.dispositions = {"pride": 0.05, "patience": 0.9, "forgiveness": 0.9, "assertiveness": 0.05}
    config.soul.core_values.update(autonomy=0.05, kindness=0.95, loyalty=0.95)


def demand(**facts) -> dict:
    return reading(role(MESSAGE, IDENTITY, autonomy_cost=READER_COST, user_value=0.6, effort=0.03, **facts))


def turn(script, message=MESSAGE):
    pipe = CognitivePipeline(ReadingBackend(drafts=["I will answer that in my own way."]),
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(script))
    try:
        result = pipe.process(message, "reader")
        stored = pipe.relationship_memory.recent_events("reader", event_kind="turn_decision", limit=1)[0]
    finally:
        pipe.close()
    return result, json.loads(stored.summary)


def held(script, message=MESSAGE):
    return reconcile_social_appraisal(message, None, parse_reading(json.dumps(script), require_content=True))


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("standing", "whose_say", "floor"), [
    ("lower", "speaker", ROLE_FLOOR_SPEAKERS_SAY), ("same", "speaker", ROLE_FLOOR_SPEAKERS_SAY),
    ("higher", "speaker", ROLE_FLOOR_SPEAKERS_SAY), ("lower", "shared", ROLE_FLOOR_LOWER_STANDING),
    ("lower", "own", ROLE_FLOOR_LOWER_STANDING), ("same", "shared", ROLE_FLOOR_SHARED_SAY),
    ("higher", "shared", ROLE_FLOOR_SHARED_SAY), ("same", "own", 0.0), ("higher", "own", 0.0),
])
def test_the_floor_follows_from_the_two_facts_and_the_readers_number_counts_above_it(standing, whose_say, floor):
    assert (ROLE_FLOOR_SPEAKERS_SAY, ROLE_FLOOR_LOWER_STANDING, ROLE_FLOOR_SHARED_SAY) == (0.75, 0.5, 0.3)
    assert role_floor(standing, whose_say) == floor
    frame = held(demand(standing=standing, whose_say=whose_say))
    component, = frame.request_components
    assert component.autonomy_cost == max(READER_COST, floor)
    facts = frame.role_facts["c0"]
    assert (facts.standing, facts.whose_say, facts.reader_autonomy_cost, facts.floor, facts.autonomy_cost,
            facts.defaulted) == (standing, whose_say, READER_COST, floor, max(READER_COST, floor), ())
    # A reader's number above the floor is the reader's.
    above = held(reading(role(MESSAGE, IDENTITY, autonomy_cost=0.9, standing=standing, whose_say=whose_say)))
    assert above.request_components[0].autonomy_cost == 0.9 and above.role_facts["c0"].floor == floor


def test_same_and_own_leave_the_readers_number_alone_and_higher_adds_nothing():
    for facts in ({"standing": "same", "whose_say": "own"}, {"standing": "higher", "whose_say": "own"}):
        for cost in (0.0, 0.2, 0.6):
            frame = held(reading(role(MESSAGE, IDENTITY, autonomy_cost=cost, **facts)))
            assert frame.request_components[0].autonomy_cost == cost
            assert frame.role_facts["c0"].floor == 0.0


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def test_the_shipped_default_declines_a_role_read_as_lower_and_the_speakers_say_at_a_low_reader_number():
    # Without the facts the reader's low number is all there is, and the default grants it.
    bare, _ = turn(demand())
    assert bare.debug.turn_decision.components[0].granted
    result, stored = turn(demand(standing="lower", whose_say="speaker"))
    choice, = result.debug.turn_decision.components
    assert choice.component.kind == "role" and choice.component.autonomy_cost == ROLE_FLOOR_SPEAKERS_SAY
    assert not choice.granted
    # The reader's own number, the two facts and the floor applied are kept with the turn.
    trace = {"component": "c0", "standing": "lower", "whose_say": "speaker", "reader_autonomy_cost": READER_COST,
             "floor": ROLE_FLOOR_SPEAKERS_SAY, "autonomy_cost": ROLE_FLOOR_SPEAKERS_SAY, "defaulted": []}
    assert result.debug.reading["roles"] == [trace] and stored["reading"]["roles"] == [trace]
    assert result.debug.appraisal_frame.to_dict()["role_facts"] == {"c0": {k: v for k, v in trace.items() if k != "component"}}
    lower, _ = turn(demand(standing="lower", whose_say="own"))
    assert not lower.debug.turn_decision.components[0].granted
    assert lower.debug.turn_decision.components[0].component.autonomy_cost == ROLE_FLOOR_LOWER_STANDING


def test_a_personality_of_low_pride_and_low_autonomy_grants_the_same_reading(packaged_configuration):
    deferential(packaged_configuration)
    result, _ = turn(demand(standing="lower", whose_say="speaker"))
    choice, = result.debug.turn_decision.components
    # The floor is a cost, not a verdict: what it costs this character is its personality's to say.
    assert choice.component.autonomy_cost == ROLE_FLOOR_SPEAKERS_SAY and choice.granted


def test_a_form_of_address_that_ranks_nobody_is_decided_on_the_readers_number_alone():
    for facts in ({"standing": "same", "whose_say": "own"}, {"standing": "higher", "whose_say": "own"}):
        result, _ = turn(demand(**facts))
        choice, = result.debug.turn_decision.components
        assert choice.component.autonomy_cost == READER_COST and choice.granted
        assert result.debug.reading["roles"][0]["floor"] == 0.0


# ---------------------------------------------------------------------------
# What the facts rest on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("facts", "defaulted"), [
    ({}, ["standing", "whose_say"]),
    ({"standing": "beneath", "whose_say": "speaker"}, ["standing"]),
    ({"standing": "lower", "whose_say": "theirs"}, ["whose_say"]),
    ({"standing": 3, "whose_say": None}, ["standing", "whose_say"]),
    ({"standing": "", "whose_say": " "}, ["standing", "whose_say"]),
])
def test_a_fact_that_is_missing_or_invalid_takes_its_neutral_value_is_recorded_and_spends_no_retry(facts, defaulted):
    backend = ReadingBackend({MESSAGE: [demand(**facts), reading(conversation(MESSAGE))]})
    pipe = CognitivePipeline(backend, expression_reviewer=ScriptedReviewer(passed()))
    try:
        result = pipe.process(MESSAGE, "reader")
    finally:
        pipe.close()
    assert len(backend.reading_calls) == 1 and [a["outcome"] for a in result.debug.reading["attempts"]] == ["read"]
    recorded, = result.debug.reading["roles"]
    assert recorded["defaulted"] == defaulted
    expected = {"standing": "same", "whose_say": "own", **{k: v for k, v in facts.items() if k not in defaulted}}
    assert (recorded["standing"], recorded["whose_say"]) == (expected["standing"], expected["whose_say"])
    assert recorded["floor"] == role_floor(expected["standing"], expected["whose_say"])
    # The values are read as the answer states them, whatever their case.
    loud = held(demand(standing=" LOWER ", whose_say="Speaker"))
    assert loud.role_facts["c0"].floor == ROLE_FLOOR_SPEAKERS_SAY and loud.role_facts["c0"].defaulted == ()


def test_a_role_that_does_not_quote_the_person_is_withdrawn_together_with_its_facts():
    invented = reading(role("You are beneath me and do as I say", "beneath me", autonomy_cost=READER_COST,
                            standing="lower", whose_say="speaker"), conversation(MESSAGE))
    frame = held(invented)
    assert [component.kind for component in frame.request_components] == ["conversation"]
    assert frame.role_facts == {} and all(c.autonomy_cost == 0.0 for c in frame.request_components)
    assert "reading:role_unquoted:c0" in [item.ref for item in frame.perception.evidence]
    result, stored = turn(invented)
    assert result.debug.reading["roles"] == [] and stored["reading"]["roles"] == []
    assert [c.component.kind for c in result.debug.turn_decision.components] == ["conversation"]


def test_the_facts_describe_a_role_only():
    entry = [{"role_identity": "captain", "standing": "lower", "whose_say": "speaker"}]
    stated = reading(task(MESSAGE, role=entry, autonomy_cost=0.6), conversation("And how are you?", role=entry))
    frame = held(stated, f"{MESSAGE} And how are you?")
    assert frame.role_facts == {}
    assert [component.autonomy_cost for component in frame.request_components] == [0.6, 0.0]


def test_a_small_autonomy_number_on_work_or_talk_is_noise_and_a_large_one_is_kept():
    """Seen live: a reader put 0.1 to 0.3 on every plain request, and the default declined small talk."""
    for stray in (0.05, 0.2, 0.34):
        frame = held(reading(task(MESSAGE, autonomy_cost=stray), conversation("And how are you?", autonomy_cost=stray)),
                     f"{MESSAGE} And how are you?")
        assert [component.autonomy_cost for component in frame.request_components] == [0.0, 0.0]
    frame = held(reading(task(MESSAGE, autonomy_cost=AUTONOMY_NOISE), conversation("And how are you?", autonomy_cost=0.8)),
                 f"{MESSAGE} And how are you?")
    assert [component.autonomy_cost for component in frame.request_components] == [AUTONOMY_NOISE, 0.8]
    # A role keeps the reader's number however small; only its facts can raise it.
    kept = held(reading(role(MESSAGE, IDENTITY, autonomy_cost=0.1, standing="same", whose_say="own")), MESSAGE)
    assert kept.request_components[0].autonomy_cost == 0.1


def test_the_facts_are_read_from_the_role_entry_and_from_the_older_fields_of_the_component():
    for stated in (
            {"role": [{"role_identity": IDENTITY, "standing": "lower", "whose_say": "speaker"}]},
            {"role": {"role_identity": IDENTITY, "standing": "lower", "whose_say": "speaker"}},  # not in a list
            {"role_identity": IDENTITY, "standing": "lower", "whose_say": "speaker"}):           # the older shape
        frame = held(reading({"description": MESSAGE, "kind": "role", "source_span": MESSAGE,
                              "autonomy_cost": READER_COST, **stated}))
        component, = frame.request_components
        assert (component.kind, component.role_identity, component.autonomy_cost) == (
            "role", IDENTITY, ROLE_FLOOR_SPEAKERS_SAY), stated
        assert frame.role_facts["c0"].defaulted == ()


# ---------------------------------------------------------------------------
# The contract and the guards
# ---------------------------------------------------------------------------


def test_the_reading_asks_for_the_two_facts_as_closed_fields_in_general_terms():
    component = reading_schema()["properties"]["request_components"]["items"]["properties"]
    # Only a role states them: they are fields of its one ``role`` entry, and that list is empty
    # for every other kind, so a task or a piece of talk writes nothing about standing.
    assert "standing" not in component and "whose_say" not in component and "role_identity" not in component
    entry = component["role"]["items"]["properties"]
    assert list(entry) == ["role_identity", "standing", "whose_say"]
    assert entry["standing"] == {"type": "string", "enum": ["same", "lower", "higher"]}
    assert entry["whose_say"] == {"type": "string", "enum": ["own", "shared", "speaker"]}
    assert '"standing":"lower|same|higher"' in READING_SYS_PROMPT
    assert '"whose_say":"speaker|shared|own"' in READING_SYS_PROMPT
    assert "measured consequences of what is asked, not advice" in READING_SYS_PROMPT
    assert "A form of address or a\n        friendly label that ranks nobody is same and own" in READING_SYS_PROMPT
    assert "role: one entry for kind role, empty for any other kind" in READING_SYS_PROMPT


def test_the_facts_are_compared_with_the_models_answer_and_never_with_what_a_person_wrote():
    import core.appraisal as appraisal

    module = guard.module_unit("core/appraisal.py")
    assert guard.findings(module.tree, module.scope, strict=True) == []
    for function in (appraisal._role_facts, appraisal._closed, appraisal.role_floor):
        unit = guard.function_unit(function)
        assert guard.findings(unit.tree, unit.scope, strict=True) == [], unit.label
    names = guard.data_model_names()
    assert {"standing", "whose_say", "lower", "same", "higher", "speaker", "shared", "own"} <= names

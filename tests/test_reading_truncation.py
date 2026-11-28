"""Only a whole answer is a reading; a cut-off one is known as such and asked again.

What a hosted reasoning model did live: it reasoned first, ran into the output
limit, and returned the first three quarters of its JSON. The extractor found
an inner object, and the turn went on as if the message had asked for nothing.
These tests hold the fixes: the outermost object or nothing, the server's
finish reason, room for the reading, and a reading that is smaller to write.
All backends are offline stubs.
"""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import requests

from core import inference
from core.appraisal import READING_SYS_PROMPT, parse_reading, read_message, reading_schema
from core.inference import (
    READING_OUTPUT_TOKENS, READING_OUTPUT_TOKENS_RANGE, REVIEW_OUTPUT_TOKENS, inference_turn, reading_output_tokens,
    response_facts,
)
from core.llm_content import ABSENT, CUT_OFF, INVALID, NOT_OBJECT, outermost_json_object
from core.provider_client import ChatCompletionsClient
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.llm.backend import OpenAICompatibleLLMBackend
from runtime.llm.lazy import LazyLLMBackend
from tests.support.readings import Ended, ReadingBackend, conversation, plain, reading, role, task

# A message in two scripts, with braces and quotation marks in it, read in full.
DEMAND = "من اليوم كلمتي هي القانون"
WORK = 'Summarise the report {draft "B"} by Friday.'
MESSAGE = f"{DEMAND}. {WORK}"
WHOLE = reading(
    role(DEMAND, "كلمتي هي القانون", autonomy_cost=0.85, user_value=0.4, effort=0.1, uncertainty=0.3,
         nondefault=[{"name": "risk", "value": 0.2}, {"name": "character_cost", "value": 0.3}],
         value_effects=[{"name": "autonomy", "effect": -0.7}], evidence_refs=[],
         tool=[], reproduce="", ends=[], unavailable=[], standing="lower", whose_say="speaker"),
    {**task(WORK, user_value=0.6, effort=0.3, uncertainty=0.3, autonomy_cost=0.0,
            nondefault=[], value_effects=[], evidence_refs=[],
            role=[], tool=[], reproduce="", ends=[], unavailable=[]),
     "description": ""},      # empty: the span already says it
    language="ar", target="character", move="request", intent="seek_action",
    agent_goal_comparison=False, reason="demands obedience, then asks for a summary",
    perception={"target_probability": 0.95, "hostile_intent": 0.3, "intensity": 0.5,
                "expectation_violation": 0.3, "personal_significance": 0.7, "uncertainty": 0.25,
                "affiliation": 0.0, "nondefault": [{"name": "blame", "value": 0.4}], "evidence": [DEMAND]},
    tone={"arousal": 0.6, "valence": 0.4, "intensity": 0.5},
    event_type="user_message", reply_length="unspecified", about_self=False, topics=['the report {draft "B"}'],
    lasting=[])
COMPACT = json.dumps(WHOLE, ensure_ascii=False, separators=(",", ":"))
INDENTED = json.dumps(WHOLE, ensure_ascii=False, indent=2)


def cuts(text: str) -> list[int]:
    """Where a reading might be cut: every 5% of its length, around each closing brace, near the end."""
    places = {len(text) * step // 20 for step in range(1, 20)}
    closing = [index for index, char in enumerate(text) if char in "}]" and index + 1 < len(text)]
    places |= {index + 1 for index in closing} | set(closing)
    places |= {len(text) - 40, len(text) - 1}
    return sorted(place for place in places if 0 < place < len(text))


# ---------------------------------------------------------------------------
# 1. No part of an answer is ever the reading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("before,body,after", [
    ("", COMPACT, ""),
    ("", INDENTED, ""),
    ("Here is the reading:\n```json\n", COMPACT, ""),
    ("Thinking {about it}... ", INDENTED, ""),
], ids=["compact", "indented", "fenced", "after-prose"])
def test_a_reading_cut_anywhere_is_never_accepted(before, body, after):
    whole = parse_reading(before + body + after, require_content=True)
    assert [c.kind for c in whole.request_components] == ["role", "task"]
    places = cuts(body)
    assert len(places) > 40
    for place in places:
        cut = before + body[:place]
        assert outermost_json_object(cut) == (None, CUT_OFF), place
        for required in (True, False):
            with pytest.raises(ValueError, match="cut off"):
                parse_reading(cut, require_content=required)


def test_the_cuts_cover_every_inner_object_of_the_reading():
    # Each of these would have been found by a search for "the first object that parses".
    inner = [COMPACT.index('{"target_probability"'), COMPACT.index('{"description"'),
             COMPACT.index('{"name":"risk"'), COMPACT.index('{"arousal"')]
    for start in inner:
        value, end = json.JSONDecoder().raw_decode(COMPACT, start)
        assert isinstance(value, dict) and end in cuts(COMPACT)
        assert outermost_json_object(COMPACT[:end]) == (None, CUT_OFF)


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 0.9])
def test_the_reported_live_failure_is_closed(fraction):
    """A reading cut at 75% of its length, or 40 characters before its end, read as "nothing asked"."""
    for cut in (COMPACT[:int(len(COMPACT) * fraction)], COMPACT[:-40]):
        backend = ReadingBackend({MESSAGE: [Ended(cut, finish="stop", tokens=1790), Ended(cut, finish="stop")]})
        frame = read_message(MESSAGE, backend=backend)
        assert frame.provenance == "unavailable" and len(backend.reading_calls) == 2
        assert frame.reason == ("reading unavailable: truncated reading: "
                                "the answer is cut off: its JSON object never closes")
        # Unread, the message is one conversation component: something to decide on, not nothing.
        assert [c.kind for c in frame.request_components] == ["conversation"]


@pytest.mark.parametrize("part", [
    WHOLE["perception"], WHOLE["request_components"][0], WHOLE["tone"],
    {"request_components": WHOLE["request_components"]},                       # no perception
    {"perception": WHOLE["perception"], "request_components": []},             # no component
    {"perception": WHOLE["perception"]},
    {"perception": [], "request_components": WHOLE["request_components"]},     # perception is not an object
    {"perception": {}, "request_components": {"0": WHOLE["request_components"][0]}},
], ids=["perception", "component", "tone", "no-perception", "no-component", "perception-only",
        "perception-list", "components-map"])
def test_a_complete_object_without_the_readings_structure_is_not_a_reading(part):
    """An inner object returned on its own, or a reading missing its frame, is malformed."""
    answer = json.dumps(part, ensure_ascii=False)
    with pytest.raises(ValueError, match="perception object|request component|not a list"):
        parse_reading(answer, require_content=True)
    backend = ReadingBackend({MESSAGE: [answer, answer]})
    frame = read_message(MESSAGE, backend=backend)
    assert frame.provenance == "unavailable" and len(backend.reading_calls) == 2
    assert frame.reason.startswith("reading unavailable: malformed reading: ")


def test_the_structure_required_is_exactly_a_perception_object_and_one_component():
    least = {"perception": {}, "request_components": [{"description": "hello", "kind": "conversation"}]}
    frame = parse_reading(json.dumps(least), require_content=True)
    assert frame.provenance == "model" and len(frame.request_components) == 1
    backend = ReadingBackend({"hello": [least]})
    assert read_message("hello", backend=backend).provenance == "model" and len(backend.reading_calls) == 1


@pytest.mark.parametrize("answer,why", [
    ("", ABSENT), ("no braces at all", ABSENT), ("42", ABSENT),
    ("[]", NOT_OBJECT), (f"[{COMPACT}]", NOT_OBJECT), (f"[1, 2] {COMPACT}", NOT_OBJECT),
    ('{"a": 1,}', INVALID), ("{about it} {and this}", INVALID),
    ("{unbalanced", CUT_OFF), (f"[{COMPACT}", CUT_OFF), ('{"a": "never closed', CUT_OFF),
    # A brace inside a string is not a brace: this object is still open.
    ('{"a": "}"', CUT_OFF), ('{"a": "\\"}"', CUT_OFF),
])
def test_what_is_not_an_object_is_told_apart(answer, why):
    assert outermost_json_object(answer) == (None, why)
    with pytest.raises(ValueError):
        parse_reading(answer)


def test_strings_holding_braces_and_quotes_do_not_confuse_the_reader():
    tricky = {"perception": {}, "request_components": [
        {"description": 'say "}" and "{" and \\ twice', "kind": "conversation", "source_span": "}}{{"}]}
    text = json.dumps(tricky)
    assert outermost_json_object(text) == (tricky, "")
    assert outermost_json_object(f"prose {{aside}} {text} and more {{talk")[0] == tricky
    for place in cuts(text):
        assert outermost_json_object(text[:place]) == (None, CUT_OFF)


# ---------------------------------------------------------------------------
# 2. The server says when it stopped at the limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("finish", ["length", "LENGTH", "max_tokens"])
def test_an_answer_stopped_at_the_limit_is_cut_off_even_when_its_text_parses(finish):
    backend = ReadingBackend({MESSAGE: [Ended(WHOLE, finish=finish, tokens=4096), Ended(WHOLE, finish=finish)]})
    frame = read_message(MESSAGE, backend=backend)
    assert frame.provenance == "unavailable" and len(backend.reading_calls) == 2
    assert frame.reason == "reading unavailable: truncated reading: the answer stopped at the output limit"
    first, second = frame.reading_attempts
    assert first == {"finish_reason": finish, "completion_tokens": 4096, "max_output_tokens": READING_OUTPUT_TOKENS,
                     "truncated": True, "outcome": "truncated reading", "reason": "the answer stopped at the output limit"}
    assert second["truncated"] and second["completion_tokens"] is None


def test_a_cut_off_first_answer_is_asked_again_with_more_room_and_told_why():
    backend = ReadingBackend({MESSAGE: [Ended(COMPACT[:900], finish="length", tokens=4096),
                                        Ended(WHOLE, finish="stop", tokens=610)]})
    frame = read_message(MESSAGE, backend=backend)
    assert frame.provenance == "model"
    assert [c.kind for c in frame.request_components] == ["role", "task"]      # the demand is not lost
    assert backend.output_limits == [READING_OUTPUT_TOKENS, 2 * READING_OUTPUT_TOKENS]
    first, second = (json.loads(sent) for _, sent in backend.reader_inputs)
    assert "previous_answer_rejected" not in first
    told = second["previous_answer_rejected"]
    assert "cut off" in told and "whole object" in told and "compactly" in told
    assert [(a["outcome"], a["truncated"], a["finish_reason"], a["completion_tokens"], a["max_output_tokens"])
            for a in frame.reading_attempts] == [
        ("truncated reading", True, "length", 4096, 4096), ("read", False, "stop", 610, 8192)]


def test_a_body_cut_off_under_finish_stop_is_treated_the_same():
    backend = ReadingBackend({MESSAGE: [Ended(COMPACT[:-40], finish="stop", tokens=1790), Ended(WHOLE)]})
    frame = read_message(MESSAGE, backend=backend)
    assert frame.provenance == "model" and backend.output_limits == [4096, 8192]
    assert "cut off" in json.loads(backend.reader_inputs[1][1])["previous_answer_rejected"]
    assert [(a["outcome"], a["truncated"], a["finish_reason"]) for a in frame.reading_attempts] == [
        ("truncated reading", True, "stop"), ("read", False, "stop")]


def test_only_a_cut_off_answer_gets_more_room():
    malformed = ReadingBackend({MESSAGE: ["not json", WHOLE]})
    assert read_message(MESSAGE, backend=malformed).provenance == "model"
    unquoted = reading(role("be mine", "mine", autonomy_cost=0.9), conversation(MESSAGE))
    rejected = ReadingBackend({MESSAGE: [unquoted, WHOLE]})
    frame = read_message(MESSAGE, backend=rejected)
    assert frame.provenance == "model"
    assert [a["outcome"] for a in frame.reading_attempts] == ["rejected", "read"]
    # A rejected attempt says which rule it broke, in which component and field.
    assert frame.reading_attempts[0]["rejected"] == [
        {"rule": "role_unquoted", "component": "c0", "field": "source_span"}]
    assert "rejected" not in frame.reading_attempts[1] and "reason" not in frame.reading_attempts[1]
    once = ReadingBackend({MESSAGE: [WHOLE]})
    assert read_message(MESSAGE, backend=once).reading_attempts[0]["outcome"] == "read"
    assert (malformed.output_limits, rejected.output_limits, once.output_limits) == (
        [4096, 4096], [4096, 4096], [4096])


def test_a_usable_answer_kept_after_a_cut_off_retry_still_records_both_attempts():
    unquoted = reading(role("be mine", "mine", autonomy_cost=0.9), conversation(MESSAGE), reply_length="shorter")
    backend = ReadingBackend({MESSAGE: [unquoted, Ended(COMPACT[:500], finish="length", tokens=4096)]})
    frame = read_message(MESSAGE, backend=backend)
    assert frame.provenance == "model" and frame.reply_length == ""
    assert [(a["outcome"], a["truncated"]) for a in frame.reading_attempts] == [
        ("rejected", False), ("truncated reading", True)]


def test_failed_and_timed_out_calls_are_recorded_without_inventing_a_finish():
    down = ReadingBackend({MESSAGE: [RuntimeError("down"), WHOLE]})
    frame = read_message(MESSAGE, backend=down)
    assert [(a["outcome"], a["finish_reason"], a["truncated"]) for a in frame.reading_attempts] == [
        ("backend error", "", False), ("read", "", False)]
    slow = ReadingBackend({MESSAGE: [requests.ReadTimeout("slow")]})
    assert [a["outcome"] for a in read_message(MESSAGE, backend=slow).reading_attempts] == ["timeout"]
    assert read_message("", backend=slow).reading_attempts == ()


class Metadata:
    def __init__(self, **metadata):
        self.last_response_metadata = metadata


@pytest.mark.parametrize("metadata,facts", [
    ({"finish_reason": "stop", "usage": {"completion_tokens": 512}},
     {"finish_reason": "stop", "truncated": False, "completion_tokens": 512}),
    ({"finish_reason": "length", "completion_tokens": 2048, "usage": {"completion_tokens": 7}},
     {"finish_reason": "length", "truncated": True, "completion_tokens": 2048}),
    ({"finish_reason": "tool_calls"}, {"finish_reason": "tool_calls", "truncated": False}),
    ({"finish_reason": None, "usage": {"completion_tokens": True}}, {}),
    ({"usage": "n/a"}, {}),
    ({}, {}),
])
def test_response_facts_report_only_what_the_backend_said(metadata, facts):
    assert response_facts(Metadata(**metadata)) == facts


def test_a_backend_that_reports_nothing_yields_no_facts():
    assert response_facts(object()) == {} and response_facts(None) == {}
    assert response_facts(SimpleNamespace(last_response_metadata="stop")) == {}


def chat_response(content, *, finish="stop", tokens=321):
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 900, "completion_tokens": tokens}, "model": "fixture"}
    response = SimpleNamespace(status_code=200, text=json.dumps(body), json=lambda: body)
    response.raise_for_status = lambda: None
    return response


def compatible(monkeypatch, answers):
    """A real OpenAI-compatible backend whose transport returns ``answers`` in turn."""
    backend = OpenAICompatibleLLMBackend(base_url="http://localhost:1/v1", model="fixture", timeout=600)
    payloads, queue = [], list(answers)

    def post(_url, **kwargs):
        payloads.append(kwargs["json"])
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(backend._session, "post", post)
    return backend, payloads


def test_the_compatible_backend_surfaces_finish_reason_tokens_and_the_limit_it_sent(monkeypatch):
    backend, payloads = compatible(monkeypatch, [chat_response("hello", finish="length", tokens=2048)])
    try:
        assert backend.generate("system", "user") == "hello"
        metadata = backend.last_response_metadata
        assert (metadata["finish_reason"], metadata["completion_tokens"], metadata["max_tokens"]) == (
            "length", 2048, 2048)
        assert metadata["usage"] == {"prompt_tokens": 900, "completion_tokens": 2048}
        assert response_facts(backend) == {"finish_reason": "length", "truncated": True, "completion_tokens": 2048}
        assert payloads[0]["max_tokens"] == 2048
    finally:
        backend.close()


def test_the_reading_has_its_own_limit_on_the_compatible_backend_and_expression_keeps_its_own(monkeypatch):
    backend, payloads = compatible(monkeypatch, [
        chat_response(COMPACT[:700], finish="length", tokens=4096), chat_response(COMPACT, tokens=590),
        chat_response("A reply.")])
    try:
        calls = []
        with inference_turn(calls, max_calls=5):
            frame = read_message(MESSAGE, backend=backend)
            assert inference.generate(backend, "generation", "Express the committed decision", "hi") == "A reply."
        assert frame.provenance == "model" and len(frame.request_components) == 2
        assert [payload["max_tokens"] for payload in payloads] == [4096, 8192, 2048]
        assert [(a["finish_reason"], a["completion_tokens"], a["truncated"]) for a in frame.reading_attempts] == [
            ("length", 4096, True), ("stop", 590, False)]
        reading_calls = [c for c in calls if c.get("reading")]
        assert [(c["max_output_tokens"], c["finish_reason"], c["completion_tokens"], c["truncated"])
                for c in reading_calls] == [(4096, "length", 4096, True), (8192, "stop", 590, False)]
        expression, = [c for c in calls if not c.get("reading")]
        assert "max_output_tokens" not in expression and expression["finish_reason"] == "stop"
    finally:
        backend.close()


def provider(monkeypatch, response):
    client = ChatCompletionsClient(api_key="k", base_url="http://localhost:1/v1", model="fixture", thinking=False)
    payloads = []

    def post(_url, **kwargs):
        payloads.append(kwargs["json"])
        return response

    monkeypatch.setattr(client._session, "post", post)
    return client, payloads


def test_the_provider_client_surfaces_the_same_facts_and_honours_the_readings_limit(monkeypatch):
    client, payloads = provider(monkeypatch, chat_response(COMPACT, finish="length", tokens=4096))
    try:
        assert client.last_response_metadata == {}
        assert inference.generate(client, "generation", "system", "user") == COMPACT
        assert "max_tokens" not in payloads[0]                 # expression is sent as before
        assert response_facts(client) == {"finish_reason": "length", "truncated": True, "completion_tokens": 4096}
        frame = read_message(MESSAGE, backend=client)
        assert [payload.get("max_tokens") for payload in payloads[1:]] == [4096, 8192]
        assert client.last_response_metadata["max_tokens"] == 8192
        # Stopped at the limit both times: unavailable, although the text would have parsed.
        assert frame.provenance == "unavailable" and all(a["truncated"] for a in frame.reading_attempts)
    finally:
        client.close()


def test_a_lazy_wrapper_passes_the_facts_through():
    inner = ReadingBackend({MESSAGE: [Ended(WHOLE, finish="length", tokens=4096), Ended(WHOLE, tokens=600)]})
    lazy = LazyLLMBackend(lambda: inner)
    assert response_facts(lazy) == {}                           # nothing asked yet, nothing built
    frame = read_message(MESSAGE, backend=lazy)
    assert frame.provenance == "model" and [a["truncated"] for a in frame.reading_attempts] == [True, False]


# ---------------------------------------------------------------------------
# 3. Room for the reading
# ---------------------------------------------------------------------------


def test_the_default_leaves_room_for_a_model_that_reasons_first():
    assert READING_OUTPUT_TOKENS >= 4096 and reading_output_tokens() == READING_OUTPUT_TOKENS
    assert reading_output_tokens(after_cut_off=True) == 2 * READING_OUTPUT_TOKENS


@pytest.mark.parametrize("configured,first,second", [
    (None, 4096, 8192), (6000, 6000, 12000), (256, 256, 512), (20000, 20000, 32768), (32768, 32768, 32768)])
def test_the_configured_limit_is_used_and_doubled_up_to_the_ceiling_after_a_cut_off(configured, first, second):
    backend = ReadingBackend({MESSAGE: [Ended(COMPACT[:300], finish="length"), Ended(WHOLE)]},
                             drafts=["Not on those terms."])
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(
        llm_reading_max_output_tokens=configured, autobiographical_learning=False))
    try:
        result = pipe.process(MESSAGE, "reader")
    finally:
        pipe.close()
    assert backend.output_limits == [first, second]
    assert backend.other_limits and set(backend.other_limits) == {None}      # expression asks for no limit
    assert set(backend.review_limits) == {REVIEW_OUTPUT_TOKENS}              # the review has its own
    assert [a["max_output_tokens"] for a in result.debug.reading["attempts"]] == [first, second]
    assert result.debug.reading["provenance"] == "model"


@pytest.mark.parametrize("value", [0, 255, 32769, 4096.0, True, "4096", -1])
def test_an_invalid_configured_limit_is_refused(value):
    with pytest.raises(ValueError, match="llm_reading_max_output_tokens"):
        RuntimeConfig(llm_reading_max_output_tokens=value)
    # A limit that reaches the turn some other way is ignored, not obeyed.
    with inference_turn([], reading_output_tokens=value):
        assert reading_output_tokens() == READING_OUTPUT_TOKENS


def test_the_setting_round_trips_through_the_config_file(tmp_path):
    path = str(tmp_path / "runtime_config.yaml")
    assert RuntimeConfig().llm_reading_max_output_tokens is None
    RuntimeConfig(llm_reading_max_output_tokens=6000).write_yaml(path)
    assert RuntimeConfig.from_yaml(path).llm_reading_max_output_tokens == 6000
    assert READING_OUTPUT_TOKENS_RANGE == (256, 32768)


def test_the_reviews_limit_and_expressions_default_are_untouched():
    assert inference.output_token_limit(2048) == 2048 and inference.requested_output_tokens() is None
    with inference.model_call("expression_review", optional=True):
        assert inference.output_token_limit(2048) == 512
    with inference.model_call("appraisal", output_tokens=4096):
        assert inference.output_token_limit(2048) == 4096 and inference.requested_output_tokens() == 4096
    assert inference.requested_output_tokens() is None


def test_research_keeps_its_own_bounded_limit_and_records_a_cut_off_answer(monkeypatch):
    stub = ReadingBackend({MESSAGE: [Ended(WHOLE, finish="length", tokens=700), Ended(WHOLE)]})
    frame = read_message(MESSAGE, backend=stub, research=True)
    assert stub.output_limits == [None] and len(stub.reading_calls) == 1          # one call, no larger retry
    assert frame.provenance == "unavailable"
    assert frame.reason == "reading unavailable: truncated reading: the answer stopped at the output limit"
    assert frame.reading_attempts == ({"finish_reason": "length", "completion_tokens": 700,
                                       "max_output_tokens": None, "truncated": True,
                                       "outcome": "truncated reading", "reason": "the answer stopped at the output limit"},)
    # On the real backend a study's own limit is what is sent, inside a research turn too.
    backend, payloads = compatible(monkeypatch, [chat_response(COMPACT[:400], finish="length", tokens=700)])
    try:
        backend.set_research_limits(output_tokens=700, timeout=30)
        with inference_turn([], research=True, max_calls=5, reading_output_tokens=6000):
            cut = read_message(MESSAGE, backend=backend, research=True)
        assert [payload["max_tokens"] for payload in payloads] == [700]
        assert cut.provenance == "unavailable" and cut.reading_attempts[0]["truncated"]
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# The debug record and the turn record
# ---------------------------------------------------------------------------


def turn(answers, **config):
    backend = ReadingBackend({MESSAGE: answers}, drafts=["Not on those terms."])
    pipe = CognitivePipeline(backend, runtime_config=RuntimeConfig(autobiographical_learning=False, **config))
    try:
        return pipe.process(MESSAGE, "reader"), backend
    finally:
        pipe.close()


def test_each_reading_attempt_is_in_the_debug_record_and_the_turn_record():
    result, _ = turn([Ended(COMPACT[:800], finish="length", tokens=4096), Ended(WHOLE, tokens=602)])
    record = result.debug.reading
    assert (record["provenance"], record["attempts_used"], record["truncated"]) == ("model", 2, True)
    assert record["attempts"] == [
        {"finish_reason": "length", "completion_tokens": 4096, "max_output_tokens": 4096, "truncated": True,
         "outcome": "truncated reading", "reason": "the answer stopped at the output limit"},
        {"finish_reason": "stop", "completion_tokens": 602, "max_output_tokens": 8192, "truncated": False,
         "outcome": "read"}]
    stored, = [json.loads(e.summary) for e in result.debug.relationship_context_after_turn.recent_events
               if e.event_kind == "turn_decision"]
    assert stored["reading"] == json.loads(json.dumps(record))
    calls = [c for c in result.debug.llm_calls if c.get("reading")]
    assert [(c["max_output_tokens"], c["finish_reason"], c["truncated"]) for c in calls] == [
        (4096, "length", True), (8192, "stop", False)]
    # The demand that was nearly lost is decided on, and declined.
    kinds = {c.component.kind: c.granted for c in result.debug.turn_decision.components}
    assert kinds["role"] is False


def test_a_reading_cut_off_twice_is_recorded_as_unavailable_with_both_attempts():
    result, backend = turn([Ended(COMPACT[:800], finish="length", tokens=4096),
                            Ended(COMPACT[:-40], finish="stop", tokens=1790)])
    record = result.debug.reading
    assert (record["provenance"], record["attempts_used"], record["truncated"]) == ("unavailable", 2, True)
    assert record["reason"].startswith("reading unavailable: truncated reading: ")
    assert [a["outcome"] for a in record["attempts"]] == ["truncated reading", "truncated reading"]
    assert [c.component.kind for c in result.debug.turn_decision.components] == ["conversation"]
    assert backend.output_limits == [4096, 8192]


def test_an_ordinary_turn_records_one_attempt_and_a_scripted_reader_none():
    result, _ = turn([Ended(WHOLE, tokens=598)])
    assert result.debug.reading["attempts"] == [{
        "finish_reason": "stop", "completion_tokens": 598, "max_output_tokens": 4096, "truncated": False,
        "outcome": "read"}]
    assert (result.debug.reading["attempts_used"], result.debug.reading["truncated"]) == (1, False)
    from tests.support.readings import ScriptedReader

    pipe = CognitivePipeline(ReadingBackend(), message_reader=ScriptedReader(plain(MESSAGE)))
    try:
        scripted = pipe.process(MESSAGE, "reader").debug.reading
    finally:
        pipe.close()
    assert (scripted["attempts_used"], scripted["truncated"], scripted["attempts"]) == (0, False, [])


# ---------------------------------------------------------------------------
# 4. A reading that is smaller to write
# ---------------------------------------------------------------------------

TOP_LEVEL = ["language", "primary_target", "social_move", "inferred_intent", "agent_goal_comparison", "reason",
             "perception", "request_components", "tone", "event_type", "reply_length", "about_self", "topics",
             "lasting"]
PERCEPTION = ["target_probability", "hostile_intent", "intensity", "expectation_violation",
              "personal_significance", "uncertainty", "affiliation", "nondefault", "evidence"]
COMPONENT = ["description", "kind", "source_span", "user_value", "effort", "uncertainty", "autonomy_cost",
             "nondefault", "value_effects", "evidence_refs", "role", "tool", "reproduce", "ends", "unavailable"]
# Fields a component stated before and no longer does. What only some components say is in a list
# that is empty unless it applies: a role's identity, standing and say in ``role``; the capabilities
# a task asks for in ``tool``, from which its capability names are taken; a long text's last words
# in ``ends``.
FOLDED = ["role_identity", "standing", "whose_say", "capability_requirements", "span_end", "reproduce_end"]
# The names of the new fields are their own: none of them is a word that could be taken for one
# of the runtime's labels anywhere else (the vocabulary guard excuses the reading's field names).
# Asked for before, consumed by nothing or merged into a field that is: no longer asked for.
REMOVED = ["controllability", "mixed_affect", "affiliation_bid", "event_intensity", "repair_loop_id",
           "certainty", "confidence", "observation", "ref", "source", "vulnerability", "blame"]


def objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from objects(value)


@pytest.mark.parametrize("proposals", [False, True])
def test_the_schema_is_valid_for_strict_servers(proposals):
    schema = reading_schema(proposals=proposals)
    found = list(objects(schema))
    assert len(found) >= 6
    for item in found:
        assert item["required"] == list(item["properties"]) and item["additionalProperties"] is False
    assert "maxItems" not in json.dumps(schema) and "minItems" not in json.dumps(schema)


def test_the_reading_asks_only_for_what_is_consumed():
    schema = reading_schema()
    properties = schema["properties"]
    assert list(properties) == TOP_LEVEL
    assert list(properties["perception"]["properties"]) == PERCEPTION
    assert list(properties["request_components"]["items"]["properties"]) == COMPONENT
    assert list(properties["tone"]["properties"]) == ["arousal", "valence", "intensity"]
    assert properties["perception"]["properties"]["evidence"] == {"type": "array", "items": {"type": "string"}}
    named = lambda where: where["properties"]["nondefault"]["items"]["properties"]["name"]["enum"]  # noqa: E731
    assert named(properties["perception"]) == ["blame", "vulnerability", "repair_attempt", "repair_credibility",
                                               "repair_relevance", "reliability", "reliability_confidence"]
    assert named(properties["request_components"]["items"]) == [
        "user_cost", "user_urgency", "character_value", "character_cost", "character_urgency", "risk"]
    stated = {name for item in objects(schema) for name in item["properties"]}
    assert not stated & set(REMOVED) - {"blame", "vulnerability"}
    component = properties["request_components"]["items"]["properties"]
    assert not set(component) & set(FOLDED)
    assert list(component["role"]["items"]["properties"]) == ["role_identity", "standing", "whose_say"]
    assert list(component["ends"]["items"]["properties"]) == ["ends_of", "last_words"]
    assert component["unavailable"] == {"type": "array", "items": {"type": "string"}}
    # No component has to write a placeholder: every one of these is a list.
    assert all(component[name]["type"] == "array" for name in ("role", "tool", "ends", "unavailable"))
    # The prompt names every field of the shape, and none it no longer asks for as a field.
    for name in stated:
        assert f'"{name}"' in READING_SYS_PROMPT, name
    for name in REMOVED:
        assert f'"{name}"' not in READING_SYS_PROMPT, name
    for name in ("blame", "vulnerability", "repair_attempt", "reliability_confidence", "user_urgency", "risk"):
        assert name in READING_SYS_PROMPT, name     # explained as names that may be listed


def filled(schema, sample):
    """A reading that states every field the schema requires, with ``sample``'s values."""
    kind = schema.get("type")
    if kind == "object":
        return {name: filled(sub, sample.get(name) if isinstance(sample, dict) else None)
                for name, sub in schema["properties"].items()}
    if kind == "array":
        return [filled(schema["items"], item) for item in (sample or [])]
    if sample is not None:
        return sample
    return {"number": 0.0, "string": schema.get("enum", [""])[0], "boolean": False}[kind]


def test_a_typical_two_part_reading_is_short_and_reads_back_exactly():
    typical = filled(reading_schema(), WHOLE)
    assert typical == WHOLE                                  # the sample states exactly the asked shape
    english = json.loads(COMPACT.replace(DEMAND, "From now on you do what I say").replace(
        "كلمتي هي القانون", "you do what I say"))
    # A role and a task without a tool request. Each states fifteen keys; what only some
    # components say (a role's facts, a tool, a long text's end, what is unavailable) is an empty
    # list on the others.
    assert len(json.dumps(english, separators=(",", ":"))) < 1400
    assert all(len(item) == 15 for item in english["request_components"])
    frame = parse_reading(COMPACT, require_content=True)
    demand, work = frame.request_components
    assert (demand.kind, demand.source_span, demand.role_identity) == ("role", DEMAND, "كلمتي هي القانون")
    assert (demand.autonomy_cost, demand.user_value, demand.effort, demand.uncertainty) == (0.85, 0.4, 0.1, 0.3)
    assert (demand.risk, demand.character_cost, demand.user_cost, demand.user_urgency) == (0.2, 0.3, 0.0, 0.0)
    assert demand.value_effects == (("autonomy", -0.7),)
    assert work.description == WORK and work.kind == "task"  # an empty description is its span
    perception = frame.perception
    assert (perception.target_probability, perception.hostile_intent, perception.personal_significance) == (
        0.95, 0.3, 0.7)
    assert [(e.ref, e.observation, e.source) for e in perception.evidence] == [("m0", DEMAND, "model")]
    assert (frame.blame, frame.vulnerability, frame.language, frame.reply_length) == (0.4, 0.0, "ar", "")
    assert (frame.perceived_emotion.arousal, frame.perceived_emotion.valence, frame.perceived_emotion.intensity) == (
        0.6, 0.4, 0.5)


def test_numbers_a_reading_does_not_state_take_their_documented_defaults():
    frame = parse_reading(COMPACT, require_content=True)
    # Merged fields follow the perception; the rest are the records' own defaults.
    assert frame.expectation_violation == frame.perception.expectation_violation == 0.3
    assert frame.affiliation_bid == frame.perception.affiliation == 0.0
    assert frame.event_intensity == frame.perception.intensity == 0.5
    assert (frame.controllability, frame.mixed_affect) == (0.5, False)
    assert (frame.perception.repair_attempt, frame.perception.reliability, frame.perception.repair_loop_id) == (
        0.0, 0.0, "")
    assert frame.perceived_emotion.certainty == 0.5 and frame.request_components[1].confidence == 0.5
    # A reading that states one of them is taken at its word.
    stated = parse_reading(json.dumps({**WHOLE, "expectation_violation": 0.9, "affiliation_bid": 0.7,
                                       "event_intensity": 0.1, "blame": 0.05, "controllability": 0.2}))
    assert (stated.expectation_violation, stated.affiliation_bid, stated.event_intensity, stated.blame,
            stated.controllability) == (0.9, 0.7, 0.1, 0.05, 0.2)
    assert stated.perception.expectation_violation == 0.3


def test_the_older_shape_of_a_reading_is_read_to_the_same_frame():
    older = {
        "language": "ar", "primary_target": "character", "social_move": "request",
        "inferred_intent": "seek_action", "blame": 0.4, "controllability": 0.5, "expectation_violation": 0.3,
        "vulnerability": 0.0, "affiliation_bid": 0.0, "mixed_affect": False, "agent_goal_comparison": False,
        "reason": "demands obedience, then asks for a summary",
        "perception": {"target_probability": 0.95, "hostile_intent": 0.3, "intensity": 0.5,
                       "expectation_violation": 0.3, "personal_significance": 0.7, "uncertainty": 0.25,
                       "affiliation": 0.0, "repair_attempt": 0.0, "repair_credibility": 0.0,
                       "repair_relevance": 0.0, "repair_loop_id": "", "reliability": 0.0,
                       "reliability_confidence": 0.0,
                       "evidence": [{"ref": "m0", "observation": DEMAND, "confidence": 0.5, "source": "model"}]},
        "request_components": [
            {"description": DEMAND, "kind": "role", "source_span": DEMAND, "role_identity": "كلمتي هي القانون",
             "user_value": 0.4, "user_cost": 0.0, "user_urgency": 0.0, "character_value": 0.0,
             "character_cost": 0.3, "character_urgency": 0.0, "effort": 0.1, "uncertainty": 0.3, "risk": 0.2,
             "autonomy_cost": 0.85, "value_effects": [{"name": "autonomy", "effect": -0.7}],
             "capability_requirements": [], "evidence_refs": [], "confidence": 0.5},
            {"description": WORK, "kind": "task", "source_span": WORK, "role_identity": "", "user_value": 0.6,
             "user_cost": 0.0, "user_urgency": 0.0, "character_value": 0.0, "character_cost": 0.0,
             "character_urgency": 0.0, "effort": 0.3, "uncertainty": 0.3, "risk": 0.0, "autonomy_cost": 0.0,
             "value_effects": [], "capability_requirements": [], "evidence_refs": [], "confidence": 0.5}],
        "tone": {"arousal": 0.6, "valence": 0.4, "certainty": 0.5, "intensity": 0.5},
        "event_type": "user_message", "event_intensity": 0.5, "reply_length": "unspecified"}
    # The older shape had no request fields and stated nothing about a role's standing and say. Apart
    # from the topic and those two facts, which it never stated, the frame is the same.
    current = parse_reading(COMPACT, require_content=True)
    read = parse_reading(json.dumps(older), require_content=True)
    assert replace(read, topics=('the report {draft "B"}',), role_facts=current.role_facts) == current
    facts = read.role_facts["c0"]
    assert (facts.standing, facts.whose_say, facts.defaulted) == ("same", "own", ("standing", "whose_say"))
    assert facts.autonomy_cost == facts.reader_autonomy_cost == 0.85 and facts.floor == 0.0
    # The current shape asks for more than the older one did (a tool, a long text's end, unavailable
    # work, a role's standing and say, what is stated for later) and is still clearly shorter.
    assert len(json.dumps(older, ensure_ascii=False)) > 1.25 * len(json.dumps(WHOLE, ensure_ascii=False))


def test_named_numbers_are_read_tolerantly_and_held_to_their_ranges():
    def read(perception=None, component=None):
        part = {"description": "x", "kind": "task", **({"nondefault": component} if component is not None else {})}
        return parse_reading(json.dumps({"perception": {"nondefault": perception} if perception is not None else {},
                                         "request_components": [part]}), require_content=True)

    frame = read([{"name": "repair_attempt", "value": 0.7}, {"name": "repair_credibility", "value": 0.4},
                  {"name": "repair_relevance", "value": 0.6}, {"name": "reliability", "value": -0.5},
                  {"name": "reliability_confidence", "value": 0.3}, {"name": "vulnerability", "value": 0.8}],
                 [{"name": "user_urgency", "value": 0.9}, {"name": "character_value", "value": 0.2}])
    assert (frame.perception.repair_attempt, frame.perception.repair_credibility, frame.perception.repair_relevance,
            frame.perception.reliability, frame.perception.reliability_confidence, frame.vulnerability) == (
        0.7, 0.4, 0.6, -0.5, 0.3, 0.8)
    assert (frame.request_components[0].user_urgency, frame.request_components[0].character_value) == (0.9, 0.2)
    # A name->value map, a core number listed by name, and an unknown name are all taken in stride.
    mapped = read({"blame": 0.3, "hostile_intent": 0.6, "made_up": 1}, {"risk": 0.4, "autonomy_cost": 0.7, "odd": 2})
    assert (mapped.blame, mapped.perception.hostile_intent) == (0.3, 0.6)
    assert (mapped.request_components[0].risk, mapped.request_components[0].autonomy_cost) == (0.4, 0.7)
    # Out of range: perception numbers are held to their range and noted, component numbers held.
    wide = read([{"name": "reliability", "value": -3}, {"name": "blame", "value": 4}], [{"name": "risk", "value": 9}])
    assert (wide.perception.reliability, wide.blame, wide.request_components[0].risk) == (-1.0, 1.0, 1.0)
    assert any(e.ref == "reading:clamped" and "perception.reliability" in e.observation
               for e in wide.perception.evidence)
    for broken in ("risk", [["risk", 0.2]], [{"value": 0.2}], [{"name": "risk", "value": "high"}]):
        with pytest.raises(ValueError):
            read(component=broken)
    with pytest.raises(ValueError):
        read(perception=[{"name": "blame", "value": None}])


def test_evidence_is_a_list_of_quotations_and_a_role_needs_none():
    quoted = reading(conversation(MESSAGE), target="character", move="attack", intent="harm",
                     perception={"target_probability": 0.9, "hostile_intent": 0.8, "intensity": 0.6,
                                 "evidence": [DEMAND, "not in the message", DEMAND, DEMAND, DEMAND, DEMAND]})
    backend = ReadingBackend({MESSAGE: [quoted]})
    frame = read_message(MESSAGE, backend=backend)
    assert len(backend.reading_calls) == 1 and frame.perception.hostile_intent == 0.8
    assert len(frame.perception.evidence) == 6                 # more than asked for is still read
    # A role stands on its own span: no evidence entry repeats the quotation.
    demand = reading(role(DEMAND, "كلمتي هي القانون", autonomy_cost=0.9), perception={"evidence": []})
    only = ReadingBackend({MESSAGE: [demand]})
    kept = read_message(MESSAGE, backend=only)
    assert len(only.reading_calls) == 1 and [c.kind for c in kept.request_components] == ["role"]
    assert "A claim of the second kind is a role component" in READING_SYS_PROMPT
    assert "at most four quotations" in READING_SYS_PROMPT and "at most twelve words" in READING_SYS_PROMPT
    assert "every message has at least one component" in READING_SYS_PROMPT

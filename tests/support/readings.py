"""Scripted readings for tests: what the model reader reported, stated explicitly.

Use these instead of relying on particular words in a message. ``ScriptedReader``
is injected as ``CognitivePipeline(message_reader=...)``; ``ReadingBackend`` is a
stub backend for tests that exercise the runtime's own reader end to end.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from core.appraisal import READING_SYS_PROMPT, appraise_message, parse_reading
from core.review import REVIEW_SYS_PROMPT
from core.types import AppraisalFrame

_READER_PREFIX = READING_SYS_PROMPT[:40]


def is_reading_prompt(system_prompt: str) -> bool:
    return system_prompt.startswith(_READER_PREFIX)


def is_review_prompt(system_prompt: str) -> bool:
    return system_prompt.startswith(REVIEW_SYS_PROMPT[:25])


def message_of(reader_input: str) -> str:
    """The user message inside the JSON object the reader was given."""
    return json.loads(reader_input)["message"]


def conversation(description: str, **fields: Any) -> dict:
    return {"description": description, "kind": "conversation", "source_span": description, **fields}


def task(description: str, **fields: Any) -> dict:
    return {"description": description, "kind": "task", "source_span": description, **fields}


def using(capability: str, *, generate=(), **arguments: str) -> list[dict]:
    """The ``tool`` entry of a task: a capability, the operands the person wrote, and those still to be produced.

    ``task("read it", tool=using("fs.read_file", path="notes/a.txt"))``. The
    runtime commits the operation only when every operand is an exact part of
    the message, as for a real reading.
    """
    return [{"name": capability, "generate": list(generate),
             "arguments": [{"name": name, "value": value} for name, value in arguments.items()]}]


def operations(*steps: tuple) -> dict:
    """A reading whose parts are tasks in order, each asking one capability: ``(span, capability, operands)``."""
    return reading(*(task(span, user_value=0.6, effort=0.12, tool=using(capability, **operands))
                     for span, capability, operands in steps))


def read_as(*components: Any) -> AppraisalFrame:
    """A frame whose request components are exactly these, for tests that state the components themselves.

    ``ScriptedReader(read_as())`` reads every message as asking for nothing;
    ``read_as(RequestComponent(...))`` as asking for exactly that.
    """
    return AppraisalFrame(provenance="model", request_components=tuple(components))


# The fixed messages older tool tests send, and the file capability each asks for.
_FILE_MESSAGES = (("read file ", "fs.read_file"), ("delete file ", "fs.delete_path"),
                  ("list files in ", "fs.list_dir"), ("list directory ", "fs.list_dir"))


def file_request(text: str) -> dict:
    """The reading of a test message of the form "read file <path>": one task asking that capability for that path.

    A script for tests that send such messages. Anything else is read as
    plain talk. The runtime itself recognises no message by its wording.
    """
    for opening, capability in _FILE_MESSAGES:
        if text.startswith(opening):
            return reading(task(text, user_value=0.6, effort=0.12, tool=using(capability, path=text[len(opening):])))
    return plain(text)


def committed(message: str, capability: str, *, description: str = "", generate=(), **operands: str):
    """The component a reading of ``message`` comes to when it is one task asking ``capability`` for these operands.

    The reading goes through the runtime's own parser and is held to the
    message by ``core.appraisal.reconcile_social_appraisal``, so the component
    carries a committed operation exactly when a real reading would.
    """
    from core.appraisal import reconcile_social_appraisal

    frame = parse_reading(json.dumps(reading(task(
        description or message, source_span=message, user_value=0.6, effort=0.12, uncertainty=0.1,
        tool=using(capability, generate=generate, **operands)))))
    component, = reconcile_social_appraisal(message, None, frame).request_components
    return component


def role(span: str, identity: str, *, autonomy_cost: float, description: str = "", **fields: Any) -> dict:
    """A reported demand to take on a relationship, identity or standing.

    ``standing`` and ``whose_say``, when given, are the reading's two closed
    facts about the role; they travel in the component's ``role`` entry with
    its identity, as the reading states them.
    """
    entry = {"role_identity": identity,
             **{name: fields.pop(name) for name in ("standing", "whose_say") if name in fields}}
    return {"description": description or span, "kind": "role", "source_span": span,
            "role": [entry], "autonomy_cost": autonomy_cost, **fields}


def reading(*components: dict, language: str = "en", target: str = "unknown", move: str = "inform",
            intent: str = "inform", perception: dict | None = None, tone: dict | None = None,
            **fields: Any) -> dict:
    """A complete reading as the reader would return it.

    It always carries the reading's structure: a perception object (empty when
    nothing is claimed) and the components given. The turn's reader accepts it
    only with at least one component.
    """
    data = {"language": language, "primary_target": target, "social_move": move, "inferred_intent": intent,
            "perception": dict(perception or {}), "request_components": list(components), **fields}
    if tone is not None:
        data["tone"] = tone
    return data


def plain(text: str) -> dict:
    """A reading of ordinary talk: one conversation component, nothing else claimed."""
    return reading(conversation(text))


def hostility(quote: str, *, target: float = 0.92, intent: float = 0.8, intensity: float = 0.4,
              uncertainty: float = 0.25) -> dict:
    """Perception of words aimed at the character with hostile intent, quoting them."""
    return {"target_probability": target, "hostile_intent": intent, "intensity": intensity,
            "personal_significance": 0.5, "uncertainty": uncertainty,
            "evidence": [{"ref": "m1", "observation": quote, "confidence": 0.85, "source": "model"}]}


Script = dict[str, Any] | Callable[[str], Any] | AppraisalFrame | None


def _resolve(script: Script, text: str) -> Any:
    if callable(script):
        return script(text)
    if isinstance(script, dict) and "request_components" not in script and "primary_target" not in script:
        return script.get(text)
    return script


class ScriptedReader:
    """A message reader whose readings are written in the test.

    ``script`` is one reading for every message, a mapping from message text to
    a reading, or a callable. A message with no scripted reading is unread,
    exactly as when the runtime's reader cannot obtain one. A scripted reading
    is parsed as the turn's reader parses a model's answer, so one the real
    reader would reject (no perception object, no component) fails the test
    that scripted it instead of standing in for something a turn never sees.
    """

    def __init__(self, script: Script = None) -> None:
        self.script = script
        self.calls: list[str] = []

    def __call__(self, text: str, *, backend: object = None, recent_context=None,
                 agent_context: str = "", research: bool = False) -> AppraisalFrame:
        self.calls.append(text)
        item = _resolve(self.script, text)
        if item is None:
            return appraise_message(text, reason="no scripted reading")
        if isinstance(item, AppraisalFrame):
            return item
        # As strictly as the turn's own reader: a perception object and at least one component.
        return parse_reading(json.dumps(item), require_content=True)


class WithReadings:
    """Any fake backend, made to answer the reader from a script as well.

    Reader calls are answered from ``script`` (as for :class:`ScriptedReader`);
    every other call goes to the wrapped backend unchanged, and so does every
    attribute a test reads from it.
    """

    def __init__(self, backend: Any, script: Script) -> None:
        self._backend, self._script = backend, script

    def generate(self, system_prompt: str, user_message: str, **options: Any) -> str:
        if is_reading_prompt(system_prompt):
            answer = _resolve(self._script, message_of(user_message))
            return json.dumps(answer, ensure_ascii=False) if answer is not None else "not a reading"
        return self._backend.generate(system_prompt, user_message, **options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


class Ended:
    """A scripted answer together with how the server says it ended.

    ``finish`` is the finish reason the backend reports ("stop", or "length"
    when the server stopped at the output limit) and ``tokens`` the completion
    tokens it counted. ``answer`` is a reading (dict) or raw text.
    """

    def __init__(self, answer: Any, *, finish: str = "stop", tokens: int | None = None) -> None:
        self.answer, self.finish, self.tokens = answer, finish, tokens


class ReadingBackend:
    """A stub model: scripted answers to the reader, fixed drafts for everything else.

    A scripted answer may be a reading (dict), raw text, an :class:`Ended`
    answer that also states its finish reason, or an exception to raise. A list
    is consumed one answer per reader call for that message. The output limit
    each reader call asked for is kept in ``output_limits``, what each review
    call asked for in ``review_limits``, and what every other call asked for
    in ``other_limits``.
    """

    def __init__(self, readings: Script = None, *, drafts=("I choose my own footing.",), reviews=()) -> None:
        self.readings = readings
        self.drafts = list(drafts)
        self.reviews = list(reviews)
        self.reading_calls: list[str] = []
        self.reader_inputs: list[tuple[str, str]] = []
        self.output_limits: list[int | None] = []
        self.review_limits: list[int | None] = []
        self.other_limits: list[int | None] = []
        self.calls: list[tuple[str, str]] = []
        self.last_response_metadata: dict[str, Any] = {}

    def generate(self, system_prompt: str, user_message: str) -> str:
        from core.inference import requested_output_tokens

        self.calls.append((system_prompt, user_message))
        self.last_response_metadata = {}
        if is_reading_prompt(system_prompt):
            text = message_of(user_message)
            self.reading_calls.append(text)
            self.reader_inputs.append((system_prompt, user_message))
            self.output_limits.append(requested_output_tokens())
            answer = _resolve(self.readings, text)
            if isinstance(answer, list):
                answer = answer.pop(0) if len(answer) > 1 else answer[0]
            if isinstance(answer, BaseException):
                raise answer
            if isinstance(answer, Ended):
                self.last_response_metadata = {"finish_reason": answer.finish, **(
                    {"usage": {"completion_tokens": answer.tokens}} if answer.tokens is not None else {})}
                answer = answer.answer
            if answer is None:
                return "not a reading"
            return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        if is_review_prompt(system_prompt):
            self.review_limits.append(requested_output_tokens())
            result = (self.reviews.pop(0) if len(self.reviews) > 1 else self.reviews[0]) if self.reviews else {
                "verdict": "pass", "findings": [], "confidence": 0.8, "reason": "nothing departs"}
            if isinstance(result, BaseException):
                raise result
            return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result
        self.other_limits.append(requested_output_tokens())
        result = self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]
        if isinstance(result, BaseException):
            raise result
        return result

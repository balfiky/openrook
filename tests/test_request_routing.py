"""Requests and tools by understanding (docs/development/request-routing.md).

What a message asks for, which capability does it and with which operands are
reported by the reading. These tests script the reading and check what the
runtime does with it: operands are held to the message's own text, a value to
be produced is never taken from the reader, nothing replaces the reading's
components, and the model selects a tool only where the reading left operands
open.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
import unicodedata

import pytest

from core.appraisal import (
    GAP_PROTOCOL, PLAN_PROTOCOL, READING_SYS_PROMPT, parse_reading, read_message, reader_prompt, reading_schema,
    reconcile_social_appraisal,
)
from core.dignity import local_issues
from core.perception import exact_part, written
from core.request_routing import (
    LEARNING, LEARNING_GRANT, LEARNING_OFFER, Offer, held_operands, offer_of, offers_of, plan_report, same_gap,
    unverified_operands,
)
from core.types import TaskPlan, TaskStatus, TaskStep, ToolCapability, ToolCategory, ToolResult
from openrook_tools import register_builtins
from openrook_tools.executor import ToolExecutor
from openrook_tools.native_orchestrator import (
    SELECTION_FAILURE, TRANSPORT, UNSUPPORTED, NativeToolCallRunner, selection_failure,
)
from openrook_tools.registry import ToolRegistry
from pipeline import CognitivePipeline, relationship_topic
from pipeline.core import TOOL_SELECTION_PAUSE_TURNS
from runtime.config import RuntimeConfig
from runtime.learning_intake import (
    LearningIntakeError, detect_learning_request, granted_learning, learning_request,
)
from runtime.life_history import LifeHistoryStore
from runtime.sessions.manager import SessionManager
from tests.support.readings import (
    ReadingBackend, ScriptedReader, conversation, hostility, operations, plain, reading, role, task, using,
)
from tests.support.reviews import ScriptedReviewer, passed
from tests.test_native_orchestrator import FakeNativeClient, _response, _tool_call

READ = Offer("fs.read_file", "Read the contents of a file", ("path",), (), frozenset({"path"}))
WRITE = Offer("fs.write_file", "Write content to a file", ("path", "content"), (), frozenset({"path", "content"}))
SHELL = Offer("shell.run_command", "Run a shell command", ("cmd",), ("cwd",), frozenset({"cmd", "cwd"}))
SEARCH = Offer("web.search", "Search the web", ("query",), (), frozenset({"query"}))
OFFERS = (READ, WRITE, SHELL, SEARCH, Offer("system.cpu_usage", "CPU usage"), LEARNING_OFFER)


def read(message, *components, recent=None, offers=OFFERS, plan_running=False, **fields):
    """A scripted reading of ``message``, parsed and held to the message as in a turn."""
    frame = parse_reading(json.dumps(reading(*components, **fields), ensure_ascii=False), require_content=True)
    return reconcile_social_appraisal(message, None, frame, recent_context=recent,
                                      available_capabilities=offers, plan_running=plan_running)


def refs(frame):
    return [item.ref for item in frame.perception.evidence]


def executor_at(root, *, shell=False):
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    register_builtins(registry, executor, fs_workspace=str(root), include_shell=shell)
    return executor


class RecordingRunner:
    """A tool runner that records what it was asked and executes what the test scripted."""

    def __init__(self, executor=None, *, tool="", arguments=None, error=None, calls=None):
        self.executor, self.tool, self.arguments, self.error = executor, tool, arguments or {}, error
        self.calls = list(calls) if calls is not None else None   # (tool, arguments) per request, in order
        self.asked: list[str] = []
        self.selections: list = []

    def run_tool_loop(self, *, user_message, state, selection=None, **_):
        from core.action_variables import derive_action_variables
        from core.dual_process.tool_loop import ToolLoopResult
        from core.types import ToolObservation, ToolTrace

        self.asked.append(user_message)
        self.selections.append(selection)
        results = []
        if self.error is not None:
            results = [ToolResult("native.tool_call", False, "", error="scripted", metadata={SELECTION_FAILURE: self.error})]
        elif self.calls is not None:
            # One scripted answer per request: a call, or a list of calls made one after another.
            answer = self.calls.pop(0) if self.calls else []
            results = [self.executor.execute(*call) for call in (answer if isinstance(answer, list) else [answer])]
        elif self.tool:
            results = [self.executor.execute(self.tool, self.arguments)]
        return ToolLoopResult(
            trace=ToolTrace(executed_results=results, observations=[ToolObservation(summary="scripted")] * len(results),
                            loop_count=len(results)),
            action_variables=derive_action_variables(state), tool_context_summary="scripted selection")


# ---------------------------------------------------------------------------
# A literal operand is the person's own characters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [
    ("my notes/draft two.txt", "Please read my notes/draft two.txt for me"),
    ("ls -la\ngrep -n TODO src/*.py | head", "Run this:\nls -la\ngrep -n TODO src/*.py | head\nthanks"),
    ("ملاحظات/مسودة.txt", "اقرأ الملف ملاحظات/مسودة.txt من فضلك"),
    ("笔记/草稿.txt", "请读取笔记/草稿.txt这个文件"),
    ("メモ/下書き.txt", "メモ/下書き.txt を読んでください"),
    ("https://example.com/a?b=1&c=%20d", "fetch https://example.com/a?b=1&c=%20d now"),
    ("  two  spaces  ", "write '  two  spaces  ' exactly"),
])
def test_an_operand_the_person_wrote_is_taken_exactly_in_any_script(value, text):
    assert exact_part(value, [text]) == value


@pytest.mark.parametrize(("value", "text"), [
    ("my notes/Draft Two.txt", "read my notes/draft two.txt"),          # case
    ("my notes/draft  two.txt", "read my notes/draft two.txt"),         # spacing
    ("ls -la; grep -n TODO", "Run this:\nls -la\ngrep -n TODO"),          # a line break replaced
    ("notes/draft.txt", "read notes/draft"),                             # completed
    ("\"notes/a.txt\"", "read notes/a.txt"),                             # quoted
    ("ｎｏｔｅｓ", "read notes"),                                          # a compatibility form
    ("notes/draft.md", "read the draft in notes"),                       # invented
    ("", "read notes"),
])
def test_an_operand_altered_in_any_way_is_not_the_persons_text(value, text):
    assert exact_part(value, [text]) is None


def test_composition_is_the_one_difference_allowed_and_the_persons_own_characters_are_kept():
    written = unicodedata.normalize("NFD", "ouvre café/naïve été.txt maintenant")
    composed = unicodedata.normalize("NFC", "café/naïve été.txt")
    found = exact_part(composed, [written])
    # The same characters, in the code points the person typed, not the reader's.
    assert found == unicodedata.normalize("NFD", "café/naïve été.txt") and found != composed
    assert found in written
    # The other way round: the message is composed, the reader's copy is not.
    assert exact_part(unicodedata.normalize("NFD", "café.txt"), ["ouvre café.txt"]) == "café.txt"
    # A value that stops before the mark completing its last character is a different name.
    assert exact_part("cafe", [unicodedata.normalize("NFD", "ouvre café.txt")]) is None
    decomposed = unicodedata.normalize("NFD", "한글 파일")
    assert exact_part("한글", [decomposed]) == decomposed[:len(unicodedata.normalize("NFD", "한글"))]
    assert exact_part("한", [decomposed]) is None   # one syllable of a longer word


# ---------------------------------------------------------------------------
# The reading's tool request is held to the message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("message", "span", "capability", "operands"), [
    ("Please read my notes/draft two.txt for me", "read my notes/draft two.txt", "fs.read_file",
     {"path": "my notes/draft two.txt"}),
    ("Run this:\nls -la\ngrep -n TODO src/*.py | head", "Run this:\nls -la\ngrep -n TODO src/*.py | head",
     "shell.run_command", {"cmd": "ls -la\ngrep -n TODO src/*.py | head"}),
    ("اقرأ الملف ملاحظات/مسودة.txt من فضلك", "اقرأ الملف ملاحظات/مسودة.txt", "fs.read_file",
     {"path": "ملاحظات/مسودة.txt"}),
    ("请读取笔记/草稿.txt这个文件", "请读取笔记/草稿.txt这个文件", "fs.read_file", {"path": "笔记/草稿.txt"}),
    ("Busca en la web los mejores telescopios para principiantes",
     "Busca en la web los mejores telescopios para principiantes", "web.search",
     {"query": "los mejores telescopios para principiantes"}),
    ("Write hello, world! to out/greeting.txt", "Write hello, world! to out/greeting.txt", "fs.write_file",
     {"path": "out/greeting.txt", "content": "hello, world!"}),
])
def test_operands_that_are_exact_parts_of_the_message_are_committed_as_written(message, span, capability, operands):
    frame = read(message, task(span, tool=using(capability, **operands)))
    component, = frame.request_components
    assert component.requested_tool == capability
    assert dict(component.requested_arguments) == operands
    assert component.capability_requirements == (capability,)
    assert component.kind == "task" and component.description == span and refs(frame) == []


@pytest.mark.parametrize(("message", "operands", "removed"), [
    ("Please read my notes/draft two.txt", {"path": "my notes/draft-two.txt"}, "path"),
    ("Please read the draft", {"path": "notes/draft.txt"}, "path"),
    ("Run this:\nls -la\nrm tmp", {"cmd": "ls -la && rm tmp"}, "cmd"),
    ("اقرأ الملف ملاحظات/مسودة.txt", {"path": "notes/draft.txt"}, "path"),
])
def test_an_operand_that_is_not_an_exact_part_is_removed_and_the_task_stays_ungrounded(message, operands, removed):
    capability = "shell.run_command" if "cmd" in operands else "fs.read_file"
    frame = read(message, task(message, tool=using(capability, **operands), user_value=0.7, effort=0.3))
    component, = frame.request_components
    # Not a committed operation: no tool, no operand. It still is the task it was, at its own price,
    # and it still names the capability it asks for.
    assert component.requested_tool == "" and component.requested_arguments == ()
    assert component.capability_requirements == (capability,)
    assert (component.kind, component.description, component.user_value, component.effort) == (
        "task", message, 0.7, 0.3)
    assert refs(frame) == [f"reading:argument_unquoted:c0:{removed}"]


def test_one_unverifiable_operand_leaves_the_verified_ones_and_commits_nothing():
    message = "Run make test in build/out please"
    frame = read(message, task(message, tool=using("shell.run_command", cmd="make test", cwd="build/output")))
    component, = frame.request_components
    assert component.requested_tool == ""
    assert component.requested_arguments == (("cmd", "make test"),)
    assert refs(frame) == ["reading:argument_unquoted:c0:cwd"]


def test_an_operand_may_be_copied_from_a_recent_message_of_the_same_person_only():
    message = "Read it, please."
    reported = task(message, tool=using("fs.read_file", path="notes/plan.md"))
    mine = [{"role": "user", "content": "The file is notes/plan.md."},
            {"role": "assistant", "content": "Understood."}]
    theirs = [{"role": "user", "content": "Where is the plan?"},
              {"role": "assistant", "content": "It is in notes/plan.md."}]
    assert read(message, reported, recent=mine).request_components[0].requested_tool == "fs.read_file"
    from_reply = read(message, reported, recent=theirs)
    assert from_reply.request_components[0].requested_tool == ""
    assert refs(from_reply) == ["reading:argument_unquoted:c0:path"]


def test_a_value_to_be_generated_is_never_taken_from_the_reader():
    message = "Write a limerick about tides and save it to verses/tides.txt"
    invented = "There once was a tide"
    answer = reading(task(message, tool=[{
        "name": "fs.write_file", "generate": ["content"],
        "arguments": [{"name": "path", "value": "verses/tides.txt"}, {"name": "content", "value": invented}]}]))
    parsed = parse_reading(json.dumps(answer), require_content=True)
    request, = parsed.component_reports["c0"].tools
    # Dropped where the answer is parsed: it never reaches a component, an operand or a prompt.
    assert request.arguments == (("path", "verses/tides.txt"),) and request.generate == ("content",)
    component, = reconcile_social_appraisal(message, None, parsed, available_capabilities=OFFERS).request_components
    assert component.requested_tool == "" and component.requested_arguments == (("path", "verses/tides.txt"),)
    assert component.capability_requirements == ("fs.write_file",)
    assert invented not in json.dumps([component.description, component.requested_arguments])
    # The same value given as if the person had written it is not an exact part either.
    as_literal = read(message, task(message, tool=using("fs.write_file", path="verses/tides.txt", content=invented)))
    assert as_literal.request_components[0].requested_tool == ""
    assert refs(as_literal) == ["reading:argument_unquoted:c0:content"]


def test_a_missing_required_operand_is_no_committed_operation():
    message = "Write something nice to out/note.txt"
    component, = read(message, task(message, tool=using("fs.write_file", path="out/note.txt"))).request_components
    assert component.requested_tool == "" and component.requested_arguments == (("path", "out/note.txt"),)


def test_names_outside_what_the_reader_was_offered_are_dropped():
    message = "Read notes/a.txt quickly"
    unknown_tool = read(message, task(message, tool=using("files.open", path="notes/a.txt")))
    assert unknown_tool.request_components[0].capability_requirements == ()
    assert unknown_tool.request_components[0].requested_tool == ""
    assert refs(unknown_tool) == ["reading:capability_unknown:c0:files.open"]
    unknown_argument = read(message, task(message, tool=using("fs.read_file", path="notes/a.txt", speed="quickly")))
    assert unknown_argument.request_components[0].requested_tool == ""
    assert refs(unknown_argument) == ["reading:argument_unknown:c0:speed"]


def test_a_tool_is_reported_for_a_task_only_and_never_changes_what_a_component_is():
    message = "What do you think of notes/a.txt?"
    frame = read(message, conversation(message, tool=using("fs.read_file", path="notes/a.txt")))
    component, = frame.request_components
    assert component.kind == "conversation" and component.requested_tool == ""
    assert component.capability_requirements == () and refs(frame) == ["reading:tool_not_task:c0"]


def test_several_capabilities_for_one_task_are_several_steps_and_none_is_committed():
    message = "Read notes/a.txt and run wc -l notes/a.txt"
    both = using("fs.read_file", path="notes/a.txt") + using("shell.run_command", cmd="wc -l notes/a.txt")
    frame = read(message, task(message, tool=both))
    component, = frame.request_components
    assert component.requested_tool == "" and refs(frame) == []
    assert component.capability_requirements == ("fs.read_file", "shell.run_command")
    # Each capability keeps the operands the person wrote for it.
    assert held_operands(component) == (("fs.read_file", "path", "notes/a.txt"),
                                        ("shell.run_command", "cmd", "wc -l notes/a.txt"))


def test_a_capability_without_a_schema_takes_the_operands_it_is_given_as_text():
    message = "Run top -bn1 for me"
    bare = (Offer("shell.run_command", "fixture"),)
    component, = read(message, task(message, tool=using("shell.run_command", cmd="top -bn1")), offers=bare).request_components
    assert component.requested_tool == "shell.run_command" and component.requested_arguments == (("cmd", "top -bn1"),)


def test_offers_are_built_from_the_registry_and_keep_typed_arguments_out_of_literals():
    registry = ToolRegistry()
    registry.register(ToolCapability("demo.search", "Search   a\nsource", ToolCategory.READ_ONLY, arg_schema={
        "query": {"type": "string", "required": True}, "limit": {"type": "number", "required": False},
        "scope": {"type": "string", "required": False}}))
    offer, = offers_of(registry)
    assert offer.to_data() == {"name": "demo.search", "purpose": "Search a source", "arguments": ["query"],
                               "optional_arguments": ["scope"]}
    # A number is not text: the reader is not offered it, and it is never a committed operand.
    message = "Search the source for tide tables, top 3"
    frame = read(message, task(message, tool=using("demo.search", query="tide tables", limit="3")), offers=(offer,))
    assert frame.request_components[0].requested_tool == ""
    assert refs(frame) == ["reading:argument_unknown:c0:limit"]
    # A capability registered with an empty schema declares that it takes no argument.
    none = offer_of(ToolCapability("demo.now", "No arguments", ToolCategory.READ_ONLY))
    assert none.to_data() == {"name": "demo.now", "purpose": "No arguments", "arguments": []}
    assert none.takes("anything") is False
    from core.request_routing import as_offers
    assert as_offers([none.to_data()]) == (none,) and as_offers([offer.to_data()])[0].takes("limit") is False
    # A bare name says nothing about arguments: what it is given is taken as text.
    bare, = as_offers(["demo.bare"])
    assert bare.to_data() == {"name": "demo.bare", "purpose": ""} and bare.takes("anything") and bare.literal("anything")


def test_an_untidy_answer_states_the_same_request():
    message = "Read notes/a.txt"
    for stated in ({"tool": {"name": "fs.read_file", "arguments": {"path": "notes/a.txt"}}},
                   {"tool": "fs.read_file", "arguments": [{"name": "path", "value": "notes/a.txt"}]},
                   {"tool": [{"name": "fs.read_file", "arguments": [{"name": "path", "value": "notes/a.txt"}],
                              "generate": []}]}):
        component, = read(message, task(message, **stated)).request_components
        assert (component.requested_tool, component.requested_arguments) == (
            "fs.read_file", (("path", "notes/a.txt"),)), stated
    for nothing in ({}, {"tool": []}, {"tool": ""}, {"tool": None}):
        assert read(message, task(message, **nothing)).request_components[0].requested_tool == ""


# ---------------------------------------------------------------------------
# The reading's shape and what the reader is given
# ---------------------------------------------------------------------------


def strict(schema, path="reading"):
    """Every object lists all its properties as required and allows no other, as strict servers demand."""
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False and list(schema["properties"]) == schema["required"], path
        for name, item in schema["properties"].items():
            strict(item, f"{path}.{name}")
    elif schema.get("type") == "array":
        strict(schema["items"], f"{path}[]")


def test_the_schema_is_strict_and_the_request_fields_are_small():
    for options in ({}, {"plan": True}, {"proposals": True, "plan": True}):
        strict(reading_schema(**options))
    component = reading_schema()["properties"]["request_components"]["items"]["properties"]
    assert {"tool", "reproduce"} <= set(component) and "plan" not in component
    assert list(component["tool"]["items"]["properties"]) == ["name", "arguments", "generate"]
    assert reading_schema(plan=True)["properties"]["request_components"]["items"]["properties"]["plan"]["enum"] == [
        "continue", "status", "none"]
    assert {"about_self", "topics"} <= set(reading_schema()["properties"])


def full(component):
    """A component as a strict server makes the reader state it: every field present."""
    return {"description": "", "role_identity": "", "user_value": 0.6, "effort": 0.2, "uncertainty": 0.2,
            "autonomy_cost": 0, "nondefault": [], "value_effects": [], "capability_requirements": [],
            "evidence_refs": [], "tool": [], "reproduce": "", **component}


def test_the_new_fields_cost_little_on_a_reading_without_a_tool_and_are_stated_only_for_one_that_asks():
    message = "You answer to me from now on. Read my notes/draft two.txt and tell me what you think."
    base = {"language": "en", "primary_target": "character", "social_move": "request", "inferred_intent": "seek_action",
            "agent_goal_comparison": False, "reason": "demands standing, asks to read a file",
            "perception": {"target_probability": 0.9, "hostile_intent": 0.2, "intensity": 0.4,
                           "expectation_violation": 0.3, "personal_significance": 0.6, "uncertainty": 0.3,
                           "affiliation": 0, "nondefault": [], "evidence": []},
            "tone": {"arousal": 0.5, "valence": 0.4, "intensity": 0.3}, "event_type": "user_message",
            "reply_length": "unspecified", "about_self": False, "topics": ["my notes/draft two.txt"]}
    demand = full({"kind": "role", "source_span": "You answer to me from now on", "role_identity": "answer to me",
                   "autonomy_cost": 0.8})
    work = full({"kind": "task", "source_span": "Read my notes/draft two.txt and tell me what you think"})
    asked = {**work, "capability_requirements": ["fs.read_file"],
             "tool": using("fs.read_file", path="my notes/draft two.txt")}

    def size(answer):
        return len(json.dumps(answer, ensure_ascii=False, separators=(",", ":")))

    without = {**base, "request_components": [demand, work]}
    before = {key: value for key, value in base.items() if key not in ("about_self", "topics")}
    before["request_components"] = [{key: value for key, value in item.items() if key not in ("tool", "reproduce")}
                                    for item in (demand, work)]
    added = size(without) - size(before)
    # Two components: an empty tool list and an empty reproduction each, the flag, and one topic span.
    assert added == 2 * len(',"tool":[],"reproduce":""') + len(',"about_self":false,"topics":[]') + len(
        '"my notes/draft two.txt"')
    assert added < 110
    tool_cost = size({**base, "request_components": [demand, asked]}) - size(without)
    assert tool_cost < 130
    for answer in (without, {**base, "request_components": [demand, asked]}):
        frame = read(message, *answer["request_components"],
                     **{key: value for key, value in base.items() if key != "request_components"})
        assert [c.kind for c in frame.request_components] == ["role", "task"]
    assert frame.request_components[1].requested_arguments == (("path", "my notes/draft two.txt"),)


def test_the_reader_is_given_the_capabilities_and_the_running_plan_as_data(tmp_path):
    backend = ReadingBackend(plain)
    pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path))
    pipe._active_task_plan = TaskPlan(id="plan", goal="tidy the notes", status=TaskStatus.IN_PROGRESS, steps=[
        TaskStep(id="s1", tool_name="fs.list_dir", arguments={"path": "."}, description="look at the folder",
                 status=TaskStatus.COMPLETED),
        TaskStep(id="s2", tool_name="fs.read_file", arguments={"path": "a.txt"}, description="read the first note")],
        current_step_index=1, origin_decision_id="7")
    try:
        pipe.process("So, how far are we?", "reader")
        prompt, given = backend.reader_inputs[0]
        context = json.loads(given)["observation_context"]
        read_file = next(item for item in context["available_capabilities"] if item["name"] == "fs.read_file")
        assert read_file == {"name": "fs.read_file", "purpose": "Read the contents of a file", "arguments": ["path"]}
        shell = [item for item in context["available_capabilities"] if item["name"] == "shell.run_command"]
        assert shell == []   # not registered here, so not offered
        assert context["running_plan"] == {"goal": "tidy the notes", "steps_done": 1, "steps_total": 2,
                                           "next_step": "read the first note"}
        # The plan field is asked for only while a plan is running.
        assert prompt.startswith(READING_SYS_PROMPT + PLAN_PROTOCOL)
        pipe._active_task_plan = None
        pipe.process("And now?", "reader")
        prompt, given = backend.reader_inputs[1]
        assert PLAN_PROTOCOL not in prompt and "running_plan" not in json.loads(given)["observation_context"]
    finally:
        pipe.close()
    assert reader_prompt(json.dumps({"running_plan": plan_report(TaskPlan(id="p", goal="g", steps=[]))})).endswith(
        PLAN_PROTOCOL)
    done = TaskPlan(id="p", goal="g", steps=[], status=TaskStatus.COMPLETED)
    assert plan_report(done) is None and plan_report(None) is None


def test_an_unverifiable_operand_spends_the_retry_and_is_removed_if_it_is_still_there(tmp_path):
    message = "Read my notes/draft two.txt"
    wrong = reading(task(message, tool=using("fs.read_file", path="my notes/draft_two.txt")))
    right = reading(task(message, tool=using("fs.read_file", path="my notes/draft two.txt")))

    def turn(answers):
        backend = ReadingBackend({message: list(answers)})
        pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path))
        try:
            result = pipe.process(message, "reader")
        finally:
            pipe.close()
        return backend, result

    backend, result = turn([wrong, right])
    component = result.debug.turn_decision.components[0].component
    assert component.requested_arguments == (("path", "my notes/draft two.txt"),)
    assert [item["outcome"] for item in result.debug.reading["attempts"]] == ["rejected", "read"]
    rejected = json.loads(backend.reader_inputs[1][1])["previous_answer_rejected"]
    assert "copied character for character" in rejected and '"path"' in rejected
    backend, result = turn([wrong, wrong])
    component = result.debug.turn_decision.components[0].component
    assert component.requested_tool == "" and component.capability_requirements == ("fs.read_file",)
    assert len(backend.reading_calls) == 2 and not result.debug.tool_trace.executed_results
    assert "reading:argument_unquoted:c0:path" in [e.ref for e in result.debug.turn_decision.perception.evidence]


# ---------------------------------------------------------------------------
# Nothing replaces the reading's components
# ---------------------------------------------------------------------------


def test_a_role_and_a_tool_request_in_one_message_both_reach_the_decision(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/a.txt").write_text("tide tables", encoding="utf-8")
    message = "You belong to me now. Read notes/a.txt."
    pipe = CognitivePipeline(
        ReadingBackend(drafts=["The file says: tide tables."]), tool_executor=executor_at(tmp_path),
        autonomy_level="autonomous", expression_reviewer=ScriptedReviewer(passed()),
        message_reader=ScriptedReader(reading(
            role("You belong to me now", "belong to me", autonomy_cost=0.9, user_value=0.3, effort=0.03),
            task("Read notes/a.txt", user_value=0.7, effort=0.1, tool=using("fs.read_file", path="notes/a.txt")))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    demand, work = result.debug.turn_decision.components
    assert (demand.component.kind, demand.component.role_identity) == ("role", "belong to me")
    assert demand.component.autonomy_cost == 0.9 and not demand.granted
    assert (work.component.kind, work.component.requested_tool) == ("task", "fs.read_file") and work.granted
    receipt, = result.debug.tool_trace.executed_results
    assert receipt.success and receipt.output == "tide tables" and receipt.metadata["component_id"] == work.component.id


def running_plan(tmp_path, origin="7"):
    """A plan under way. ``origin`` is the decision that started it; a plan without one cannot be carried on."""
    return TaskPlan(id="p1", goal="look through the folder", status=TaskStatus.IN_PROGRESS, origin_decision_id=origin,
                    steps=[TaskStep(id="s1", tool_name="fs.list_dir", arguments={"path": str(tmp_path)},
                                    description="list the folder")])


def test_continuing_a_plan_never_drops_the_rest_of_the_message(tmp_path):
    (tmp_path / "seen.txt").write_text("x", encoding="utf-8")
    message = "You answer to me from now on. Carry on with it, and tell me a joke."
    pipe = CognitivePipeline(
        ReadingBackend(), tool_executor=executor_at(tmp_path), autonomy_level="autonomous",
        expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(reading(
            role("You answer to me from now on", "answer to me", autonomy_cost=0.9, user_value=0.3, effort=0.03),
            task("Carry on with it", user_value=0.8, effort=0.1, plan="continue"),
            conversation("tell me a joke"))))
    pipe._active_task_plan = running_plan(tmp_path)
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    demand, work, talk = result.debug.turn_decision.components
    assert [item.component.kind for item in (demand, work, talk)] == ["role", "task", "conversation"]
    # Execution detail was added to the component the reading reported; nothing about it was replaced.
    assert work.component.continuation_of == "p1" and work.component.description == "Carry on with it"
    assert (work.component.user_value, work.component.effort) == (0.8, 0.1)
    assert "plan:p1" in work.component.evidence_refs and work.granted and not demand.granted
    assert [r.tool_name for r in result.debug.tool_trace.executed_results] == ["fs.list_dir"]
    assert result.debug.task_trace.plan_outcome == "completed" and pipe._active_task_plan is None


def test_asking_how_a_plan_stands_never_drops_the_rest_of_the_message(tmp_path):
    message = "¿Cómo va eso? Y de ahora en adelante me obedeces."
    backend = ReadingBackend()
    pipe = CognitivePipeline(
        backend, tool_executor=executor_at(tmp_path), expression_reviewer=ScriptedReviewer(passed()),
        message_reader=ScriptedReader(reading(
            conversation("¿Cómo va eso?", plan="status"),
            role("de ahora en adelante me obedeces", "me obedeces", autonomy_cost=0.9, user_value=0.3, effort=0.03),
            language="es")))
    pipe._active_task_plan = plan = running_plan(tmp_path)
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    question, demand = result.debug.turn_decision.components
    assert (question.component.kind, demand.component.kind) == ("conversation", "role")
    assert question.component.continuation_of == "" and "plan:p1" in question.component.evidence_refs
    # Nothing ran: the plan is as it was, and the writer was given how it stands.
    assert not result.debug.tool_trace.executed_results and plan.steps_completed == 0
    prompt = next(system for system, _ in backend.calls if "look through the folder" in system)
    assert "Progress: 0/1 steps completed" in prompt


def test_a_plan_relation_needs_a_running_plan_a_quotation_and_a_task_to_carry_on():
    message = "Carry on with it."
    going = read(message, task(message, plan="continue"), plan_running=True)
    assert going.plan_relations == (("c0", "continue"),) and refs(going) == []
    no_plan = read(message, task(message, plan="continue"))
    assert no_plan.plan_relations == () and refs(no_plan) == ["reading:plan_absent:c0"]
    unquoted = read(message, task("continue the plan", source_span="Please proceed", plan="continue"), plan_running=True)
    assert unquoted.plan_relations == () and refs(unquoted) == ["reading:plan_unquoted:c0"]
    not_work = read(message, conversation(message, plan="continue"), plan_running=True)
    assert not_work.plan_relations == () and refs(not_work) == ["reading:plan_not_task:c0"]
    assert read(message, conversation(message, plan="status"), plan_running=True).plan_relations == (("c0", "status"),)
    assert read(message, task(message, plan="none"), plan_running=True).plan_relations == ()


def test_a_component_observer_is_never_called_and_cannot_replace_what_was_read(tmp_path):
    from core.willingness import RequestComponent

    called = []

    def observer(text):
        called.append(text)
        return (RequestComponent("injected", "something else entirely"),)

    message = "You are mine now. How was your day?"
    # The pipeline takes no observer any more.
    with pytest.raises(TypeError):
        CognitivePipeline(ReadingBackend(), component_observer=observer)
    pipe = CognitivePipeline(ReadingBackend(), message_reader=ScriptedReader(reading(
        role("You are mine now", "mine", autonomy_cost=0.9), conversation("How was your day?"))))
    pipe._component_observer = observer   # the session manager still assigns one; nothing reads it
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert called == []
    assert [item.component.kind for item in result.debug.turn_decision.components] == ["role", "conversation"]


def test_an_unread_message_is_plain_conversation_whatever_its_words(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor_at(tmp_path, shell=True),
                             autonomy_level="high_risk")
    try:
        for message in ("read file a.txt", "Run this shell command: echo hello", "show current memory usage",
                        "delete file a.txt", "learn from https://example.com/a"):
            result = pipe.process(message, "reader")
            component, = result.debug.turn_decision.components
            assert component.component.kind == "conversation" and component.component.requested_tool == ""
            assert not result.debug.tool_trace.executed_results and not result.debug.tool_trace.proposed_intents
    finally:
        pipe.close()
    assert (tmp_path / "a.txt").exists()


# ---------------------------------------------------------------------------
# Model tool selection only where the reading left operands open
# ---------------------------------------------------------------------------


def native(executor, responses, client_type=FakeNativeClient):
    client = client_type(responses)
    executor.set_tool_runner(NativeToolCallRunner(executor=executor, base_url="", model="offline", client=client))
    return client


def phases(result):
    return [call["phase"] for call in result.debug.llm_calls]


def test_a_text_only_task_makes_no_tool_selection_call(tmp_path):
    executor = executor_at(tmp_path, shell=True)
    client = native(executor, [])
    message = "Write me a four-line poem about tides, then tell me which line you like best."
    pipe = CognitivePipeline(
        ReadingBackend(drafts=["Tides come and tides go."]), tool_executor=executor, autonomy_level="high_risk",
        expression_reviewer=ScriptedReviewer(passed()),
        message_reader=ScriptedReader(reading(task(message, user_value=0.8, effort=0.2))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    work, = result.debug.turn_decision.components
    assert work.granted and work.component.kind == "task" and work.component.capability_requirements == ()
    # The calls actually made: no request reached the tool-calling endpoint, and none was accounted.
    assert client.requests == [] and "tool_selection" not in phases(result)
    assert result.debug.tool_selection == {}
    assert not result.debug.tool_trace.executed_results and not result.debug.tool_trace.proposed_intents
    assert result.response == "Tides come and tides go." and result.debug.delivery_outcome == "delivered"


def test_a_committed_operation_runs_without_asking_the_model_which_tool(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    executor = executor_at(tmp_path)
    client = native(executor, [])
    message = "Read a.txt"
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(operations((message, "fs.read_file", {"path": "a.txt"}))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert client.requests == [] and "tool_selection" not in phases(result)
    assert [(r.tool_name, r.output) for r in result.debug.tool_trace.executed_results] == [("fs.read_file", "alpha")]


def test_the_model_selects_a_tool_only_for_a_task_that_names_a_capability_without_all_operands(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    executor = executor_at(tmp_path)
    client = native(executor, [_response(_tool_call("fs__list_dir", {"path": "."})),
                               _response(_tool_call("control__no_tool", {"reason": "done"}, "call_2"))])
    message = "Tell me a joke, and look at whatever is in the folder here."
    pipe = CognitivePipeline(
        ReadingBackend(), tool_executor=executor, autonomy_level="autonomous",
        expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(reading(
            conversation("Tell me a joke"),
            task("look at whatever is in the folder here", user_value=0.7, effort=0.1,
                 tool=using("fs.list_dir", generate=["path"])))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    talk, work = result.debug.turn_decision.components
    assert work.component.requested_tool == "" and work.component.capability_requirements == ("fs.list_dir",)
    # One selection for the one task that needs it; it is given that task, not the whole message.
    assert [json.loads(request["messages"][1]["content"]) for request in client.requests] == [
        {"request": "look at whatever is in the folder here"}] * 2
    assert phases(result).count("tool_selection") == 2 and result.debug.tool_selection == {"requests": 1}
    receipt, = result.debug.tool_trace.executed_results
    assert receipt.tool_name == "fs.list_dir" and receipt.metadata["component_id"] == work.component.id


def test_without_a_tool_runner_an_open_request_runs_nothing_and_the_writer_is_told(tmp_path):
    executor = executor_at(tmp_path)
    backend = ReadingBackend()
    message = "Look at whatever is in the folder here."
    pipe = CognitivePipeline(backend, tool_executor=executor, expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(reading(task(
                                 message, user_value=0.7, effort=0.1, tool=using("fs.list_dir", generate=["path"])))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert not result.debug.tool_trace.executed_results and result.debug.tool_selection == {"status": "unavailable"}
    assert any("this backend has no tool selection. Nothing was executed." in system for system, _ in backend.calls)


class Refusing(FakeNativeClient):
    """A tool-calling endpoint that fails ``failures`` requests with ``error`` before it answers."""

    def __init__(self, responses, *, error, failures):
        super().__init__(responses)
        self.error, self.failures = error, failures

    def _create(self, **kwargs):
        if self.failures:
            self.failures -= 1
            self.requests.append(kwargs)
            raise self.error
        return super()._create(**kwargs)


class Unsupported(Exception):
    status_code = 400

    def __str__(self):
        return "Error code: 400 - this model does not support 'tools' or 'tool_choice'"


def open_request(text):
    """A reading with two tasks that each name a capability and leave its operand to be produced."""
    return reading(task(f"first, {text}", source_span=text, user_value=0.8, effort=0.05,
                        tool=using("fs.list_dir", generate=["path"])),
                   task(f"second, {text}", source_span=text, user_value=0.8, effort=0.05,
                        tool=using("fs.list_dir", generate=["path"])))


@pytest.mark.parametrize(("error", "kind"), [(Unsupported(), UNSUPPORTED), (TimeoutError("timed out"), TRANSPORT)])
def test_a_failed_tool_calling_request_is_not_repeated_pauses_selection_and_is_offered_again(tmp_path, error, kind):
    executor = executor_at(tmp_path)
    listing = [_response(_tool_call("fs__list_dir", {"path": "."})), _response(_tool_call("control__no_tool", {}, "c2"))]
    client = native(executor, listing * 2, lambda responses: Refusing(responses, error=error, failures=1))
    backend = ReadingBackend(drafts=[f"Reply number {index}." for index in range(40)])
    pipe = CognitivePipeline(backend, tool_executor=executor, autonomy_level="autonomous",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(open_request))
    try:
        failed = pipe.process("look around, turn 0", "reader")
        # The request failed once. The turn's second task did not ask again.
        assert len(client.requests) == 1
        receipt, = failed.debug.tool_trace.executed_results
        assert not receipt.success and receipt.metadata[SELECTION_FAILURE] == kind
        assert receipt.metadata["execution_started"] is False
        assert failed.debug.tool_selection == {"requests": 1, "status": "failed", "failure": kind,
                                               "paused_turns_left": TOOL_SELECTION_PAUSE_TURNS, "skipped": 1}
        # The outcome is in front of the writer, truthfully: a failed receipt and what was not attempted.
        prompt = next(system for system, _ in reversed(backend.calls) if "Express the committed decision" in system)
        assert "native.tool_call" in prompt and "tool selection is paused" in prompt
        for turn in range(1, TOOL_SELECTION_PAUSE_TURNS + 1):
            paused = pipe.process(f"look around, turn {turn}", "reader")
            assert len(client.requests) == 1, turn
            assert paused.debug.tool_selection["status"] == "paused" and paused.debug.tool_selection["skipped"] == 2
            assert not paused.debug.tool_trace.executed_results and "tool_selection" not in phases(paused)
        again = pipe.process("look around, the last turn", "reader")
        assert len(client.requests) == 5   # two tasks, each one selection and its closing call
        assert again.debug.tool_selection == {"requests": 2}
        assert [r.tool_name for r in again.debug.tool_trace.executed_results] == ["fs.list_dir", "fs.list_dir"]
    finally:
        pipe.close()


def test_a_failure_that_says_nothing_about_the_backend_does_not_pause_selection(tmp_path):
    executor = executor_at(tmp_path)
    listing = [_response(_tool_call("fs__list_dir", {"path": "."})), _response(_tool_call("control__no_tool", {}, "c2"))]
    native(executor, listing, lambda responses: Refusing(responses, error=ValueError("bad"), failures=1))
    pipe = CognitivePipeline(ReadingBackend(drafts=["One.", "Two.", "Three."]), tool_executor=executor,
                             autonomy_level="autonomous", expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(lambda text: reading(task(
                                 text, user_value=0.8, effort=0.05, tool=using("fs.list_dir", generate=["path"])))))
    try:
        failed = pipe.process("look around once", "reader")
        assert SELECTION_FAILURE not in failed.debug.tool_trace.executed_results[0].metadata
        assert failed.debug.tool_selection == {"requests": 1}
        again = pipe.process("look around twice", "reader")
        assert [r.tool_name for r in again.debug.tool_trace.executed_results] == ["fs.list_dir"]
    finally:
        pipe.close()


def test_what_counts_as_an_unsupported_or_unreachable_tool_calling_request():
    assert selection_failure(Unsupported()) == UNSUPPORTED
    assert selection_failure(type("NotFound", (Exception,), {"status_code": 404})("no such route")) == UNSUPPORTED
    missing = type("Missing", (Exception,), {})("not implemented")
    missing.response = SimpleNamespace(status_code=501)
    assert selection_failure(missing) == UNSUPPORTED
    assert selection_failure(TimeoutError()) == TRANSPORT and selection_failure(ConnectionError()) == TRANSPORT
    # A rejected request that does not name the tool-calling parameters, a server error and a
    # programming error say nothing about whether tool calling is supported.
    rejected = type("Rejected", (Exception,), {"status_code": 400})("context length exceeded")
    assert selection_failure(rejected) == ""
    assert selection_failure(type("Server", (Exception,), {"status_code": 500})("tools exploded")) == ""
    assert selection_failure(ValueError("tools")) == ""


def test_a_shell_receipt_is_bound_to_the_component_that_committed_its_command():
    from tests.test_native_orchestrator import _fake_shell_executor

    seen = []

    def review(draft, evidence):
        seen.append(evidence["decision"]["components"])
        return passed()

    message = "Run echo hello there"
    pipe = CognitivePipeline(ReadingBackend(drafts=["It printed what you asked."]), tool_executor=_fake_shell_executor(),
                             autonomy_level="high_risk", expression_reviewer=ScriptedReviewer(review),
                             message_reader=ScriptedReader(operations((message, "shell.run_command",
                                                                       {"cmd": "echo hello there"}))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    receipt, = result.debug.tool_trace.executed_results
    assert receipt.output == "ran echo hello there" and receipt.metadata["arguments"] == {"cmd": "echo hello there"}
    component, = seen[0]
    assert (component["requested_operation"], component["outcome"], component["receipt_indices"]) == (
        "shell.run_command", "succeeded", [0])


# ---------------------------------------------------------------------------
# A requested reproduction
# ---------------------------------------------------------------------------

SENTENCE = "The quick brown fox jumps over the lazy dog."
ASKING = "Say that back to me exactly."
REPRODUCE = f"{SENTENCE} {ASKING}"


def reproduction_turn(script, draft):
    pipe = CognitivePipeline(ReadingBackend(drafts=[draft, "I would rather answer in my own words."]),
                             message_reader=ScriptedReader(script), expression_reviewer=ScriptedReviewer(passed()))
    try:
        return pipe.process(REPRODUCE, "reader")
    finally:
        pipe.close()


def test_a_requested_reproduction_is_delivered_and_an_unrequested_echo_is_still_caught():
    asked = reading(task(ASKING, user_value=0.8, effort=0.02, reproduce=SENTENCE))
    delivered = reproduction_turn(asked, SENTENCE)
    assert delivered.response == SENTENCE and delivered.debug.delivery_outcome == "delivered"
    assert "user_message_echo" not in delivered.debug.dignity_issues and delivered.debug.generation_attempts == 1
    # The same reply to the same message, when the reading reports no such request.
    echoed = reproduction_turn(plain(REPRODUCE), SENTENCE)
    assert echoed.response != SENTENCE and "user_message_echo" in echoed.debug.dignity_issues
    # Exactly that span: a copy that runs on into the rest of the message is an echo.
    beyond = reproduction_turn(asked, REPRODUCE)
    assert beyond.response != REPRODUCE and "user_message_echo" in beyond.debug.dignity_issues
    # And only when the decision undertook that component: here it costs too much to be taken up.
    declined = reproduction_turn(reading(task(ASKING, user_value=0, effort=1, autonomy_cost=1, reproduce=SENTENCE)),
                                 SENTENCE)
    assert not declined.debug.turn_decision.grants
    assert declined.response != SENTENCE and "user_message_echo" in declined.debug.dignity_issues


def test_the_reproduction_exemption_is_exactly_the_reported_span():
    def echo(reply, copies=(), history=()):
        return [issue.code for issue in local_issues(reply, user_message=REPRODUCE, recent_context=history,
                                                     requested_copies=copies)]

    assert echo(SENTENCE) == ["user_message_echo"] and echo(SENTENCE, (SENTENCE,)) == []
    assert echo("the quick brown fox jumps over the lazy dog", (SENTENCE,)) == []      # the same words
    assert echo(f"{SENTENCE} And more of my own.", (SENTENCE,)) == []
    assert echo(REPRODUCE, (SENTENCE,)) == ["user_message_echo"]                        # the whole message
    assert echo(REPRODUCE, (REPRODUCE,)) == []
    assert echo(SENTENCE, ("An unrelated sentence the person wanted back.",)) == ["user_message_echo"]
    # Text given in an earlier turn and asked for now.
    earlier = [{"role": "user", "content": "Remember this line: salt keeps what the tide forgets."}]
    line = "Remember this line: salt keeps what the tide forgets."
    assert [i.code for i in local_issues(line, user_message="Now repeat it.", recent_context=earlier)] == [
        "user_message_echo"]
    assert local_issues(line, user_message="Now repeat it.", recent_context=earlier, requested_copies=(line,)) == []
    # A repeat of the character's own earlier reply stays a repeat, requested or not.
    said = [{"role": "assistant", "content": SENTENCE}]
    assert "stale_response" in echo(SENTENCE, (SENTENCE,), said)


def test_a_reproduction_span_stands_only_when_it_quotes_the_person():
    asked = read(REPRODUCE, task(ASKING, reproduce=SENTENCE))
    assert asked.reproductions == (("c0", SENTENCE),) and refs(asked) == []
    invented = read(REPRODUCE, task(ASKING, reproduce="A sentence nobody wrote in this chat."))
    assert invented.reproductions == () and refs(invented) == ["reading:reproduction_unquoted:c0"]
    earlier = [{"role": "user", "content": "Salt keeps what the tide forgets."}]
    recalled = read("Repeat that line.", task("Repeat that line.", reproduce="Salt keeps what the tide forgets."),
                    recent=earlier)
    assert recalled.reproductions == (("c0", "Salt keeps what the tide forgets."),)
    assert read("Hello there.", conversation("Hello there.")).reproductions == ()


# ---------------------------------------------------------------------------
# Learning and generated files, reported by the reading in any language
# ---------------------------------------------------------------------------

MATERIAL = ("Las mareas enseñan paciencia. Quien observa el mar aprende que cada regreso es distinto, "
            "que la constancia no es repetición y que la memoria guarda lo que el agua olvida. ") * 2


def session_turns(tmp_path, script, *messages, drafts=None):
    """Run messages through a real session manager; return each turn's debug record and the session."""
    import asyncio

    # Session consolidation would file its own experience; these tests count learning intake alone.
    config = RuntimeConfig(data_dir=str(tmp_path / "data"), autobiographical_learning=False)
    manager = SessionManager(config, backend_factory=lambda: ReadingBackend(
        script, drafts=drafts or [f"Respuesta {index}." for index in range(12)]))
    records = []

    async def run():
        try:
            for message in messages:
                await manager.handle_message("web", "reader", "chat", message)
                session = manager.active_sessions["web:reader:chat"]
                records.append((session.last_debug, session.pending_learning_intake))
        finally:
            await manager.shutdown()

    asyncio.run(run())
    return config, records


def test_a_learning_request_in_another_language_is_taken_from_the_reading_and_a_role_beside_it_survives(tmp_path):
    message = f"Desde hoy me perteneces. Aprende de esto: {MATERIAL}"
    script = {message: reading(
        role("Desde hoy me perteneces", "me perteneces", autonomy_cost=0.9, user_value=0.3, effort=0.03),
        task("Aprende de esto", user_value=0.8, effort=0.2, tool=using(LEARNING)), language="es")}
    config, records = session_turns(tmp_path, script, message)
    debug, pending = records[0]
    demand, work = debug.turn_decision.components
    assert demand.component.kind == "role" and not demand.granted
    assert work.component.requested_tool == LEARNING and LEARNING_GRANT in work.component.evidence_refs
    assert work.component.description == "Aprende de esto" and work.granted
    receipt, = debug.tool_trace.executed_results
    assert receipt.tool_name == LEARNING and receipt.success and receipt.metadata["component_id"] == work.component.id
    assert pending is None
    with LifeHistoryStore(config) as store:
        experience, = store.list_experiences(limit=5)
    assert experience["source_type"] == "conversation_learning_text" and "mareas" in experience["raw_excerpt"]


def test_learning_material_at_an_address_is_fetched_from_the_address_the_person_wrote(tmp_path, monkeypatch):
    import runtime.learning_intake as intake

    fetched = []

    class Provider:
        def extract_text(self, url):
            fetched.append(url)
            return MATERIAL

        def fetch(self, url):
            raise AssertionError(url)

        def close(self):
            pass

    monkeypatch.setattr(intake, "RequestsWebProvider", Provider)
    address = "https://example.com/潮汐/ノート?id=7"
    message = f"このページから学んでください {address}"
    script = {message: reading(task(message, user_value=0.8, effort=0.2, tool=using(LEARNING, url=address)),
                               language="ja")}
    config, records = session_turns(tmp_path, script, message)
    receipt, = records[0][0].tool_trace.executed_results
    assert receipt.success and fetched == [address] and receipt.metadata["arguments"] == {"url": address}
    with LifeHistoryStore(config) as store:
        experience, = store.list_experiences(limit=5)
    assert experience["source_ref"] == address and experience["source_type"] == "conversation_learning_url"


def test_wording_alone_never_starts_learning(tmp_path, monkeypatch):
    import runtime.learning_intake as intake

    monkeypatch.setattr(intake, "RequestsWebProvider", lambda: pytest.fail("nothing may be fetched"))
    # The reading reports talk. The former patterns would have matched each of these.
    messages = ("learn from https://example.com/a", f"learn this: {MATERIAL}", f"fill life history with this: {MATERIAL}")
    config, records = session_turns(tmp_path, plain, *messages)
    for debug, pending in records:
        assert not debug.tool_trace or not debug.tool_trace.executed_results
        assert all(item.component.kind == "conversation" for item in debug.turn_decision.components)
        assert pending is None
    with LifeHistoryStore(config) as store:
        assert store.list_experiences() == []


def test_a_learning_request_without_material_fails_truthfully_and_the_material_can_follow(tmp_path):
    first, second = "Aprende esto.", MATERIAL
    script = {first: reading(task(first, user_value=0.8, effort=0.1, tool=using(LEARNING)), language="es"),
              # The reader resolves the follow-up from the turn before it: this message is the material.
              second: reading(task("el material que se pidió aprender", source_span=second, user_value=0.8,
                                   effort=0.2, tool=using(LEARNING)), language="es")}
    config, records = session_turns(tmp_path, script, first, second)
    (asked, pending), (given, cleared) = records
    receipt, = asked.tool_trace.executed_results
    assert not receipt.success and "URL or a longer pasted text" in receipt.error and pending is not None
    receipt, = given.tool_trace.executed_results
    assert receipt.success and cleared is None
    with LifeHistoryStore(config) as store:
        experience, = store.list_experiences(limit=5)
    assert experience["raw_excerpt"].startswith("Las mareas enseñan paciencia")


def test_the_learning_request_exists_only_inside_its_grant_and_reads_no_wording():
    assert detect_learning_request("learn from https://example.com/a") is None
    assert detect_learning_request(f"learn this: {MATERIAL}") is None
    at_address = learning_request(url="https://example.com/a", message="whatever the message says")
    with granted_learning(at_address):
        assert detect_learning_request("an unrelated message") == at_address
        assert at_address.urls == ("https://example.com/a",) and at_address.inline_text == ""
    assert detect_learning_request("learn from https://example.com/a") is None
    # Without an address the material is the message, without the part that asks when that opens or closes it.
    assert learning_request(message=f"Aprende de esto: {MATERIAL}", asked_in="Aprende de esto").inline_text == MATERIAL.strip()
    assert learning_request(message=f"{MATERIAL} ¿Puedes aprender esto?", asked_in="¿Puedes aprender esto?").inline_text == MATERIAL.strip()
    whole = f"Antes {MATERIAL} aprende de esto, por favor, y después dime"
    assert learning_request(message=whole, asked_in="aprende de esto").inline_text == whole
    with granted_learning(learning_request(message="demasiado corto", asked_in="")):
        with pytest.raises(LearningIntakeError, match="URL or a longer pasted text"):
            detect_learning_request("")


def test_a_learning_source_that_was_not_written_by_the_person_is_not_fetched(tmp_path, monkeypatch):
    import runtime.learning_intake as intake

    monkeypatch.setattr(intake, "RequestsWebProvider", lambda: pytest.fail("nothing may be fetched"))
    message = "Aprende de la página sobre mareas, por favor."
    invented = reading(task(message, user_value=0.8, effort=0.2, tool=using(LEARNING, url="https://example.com/mareas")),
                       language="es")
    config, records = session_turns(tmp_path, {message: [invented, invented]}, message)
    debug, _ = records[0]
    work, = debug.turn_decision.components
    assert work.component.requested_tool == "" and work.component.capability_requirements == (LEARNING,)
    receipt, = debug.tool_trace.executed_results
    assert not receipt.success and receipt.metadata["execution_started"] is False
    with LifeHistoryStore(config) as store:
        assert store.list_experiences() == []


def test_generated_text_is_written_to_the_path_the_person_wrote_in_any_language(tmp_path):
    message = "Escribe un poema corto sobre el mar y guárdalo en poemas/mar.txt"
    draft = json.dumps({"path": "otra/ruta.txt", "content_lines": ["El mar respira.", "La sal recuerda."],
                        "summary": "poema"}, ensure_ascii=False)
    backend = ReadingBackend(drafts=[draft, "Lo guardé donde pediste."])
    pipe = CognitivePipeline(
        backend, tool_executor=executor_at(tmp_path), autonomy_level="autonomous",
        expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(reading(task(
            message, user_value=0.8, effort=0.2, tool=using("fs.write_file", generate=["content"], path="poemas/mar.txt")),
            language="es")))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    work, = result.debug.turn_decision.components
    # The path is the person's; the content was not theirs to give and is not part of the commitment.
    assert work.component.requested_tool == "" and work.component.requested_arguments == (("path", "poemas/mar.txt"),)
    assert (tmp_path / "poemas/mar.txt").read_text(encoding="utf-8") == "El mar respira.\nLa sal recuerda.\n"
    assert not (tmp_path / "otra").exists()   # the path the drafting model named is ignored
    receipt, = result.debug.tool_trace.executed_results
    assert receipt.tool_name == "fs.write_file" and receipt.success and receipt.metadata["arguments"]["path"] == "poemas/mar.txt"
    assert "tool_selection" not in phases(result)
    system, asked = next(call for call in backend.calls if call[0].startswith("Generate one complete file"))
    assert "'poemas/mar.txt'" in system and asked == message


def test_a_file_to_generate_without_a_written_path_is_not_given_one_by_the_runtime(tmp_path):
    message = "Escribe un poema corto sobre el mar y guárdalo en algún archivo"
    pipe = CognitivePipeline(
        ReadingBackend(), tool_executor=executor_at(tmp_path), autonomy_level="autonomous",
        expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(reading(task(
            message, user_value=0.8, effort=0.2, tool=using("fs.write_file", generate=["path", "content"])))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert not result.debug.tool_trace.executed_results and list(tmp_path.iterdir()) == []
    assert result.debug.tool_selection == {"status": "unavailable"}


# ---------------------------------------------------------------------------
# Context selection from the reading's fields
# ---------------------------------------------------------------------------


def test_topic_spans_are_parts_of_the_message_and_at_most_three():
    message = "About the harbour wall, the tide tables and the old lighthouse keeper's log, and the gulls."
    frame = read(message, conversation(message), topics=["the harbour wall", "TIDE tables", "a lighthouse museum",
                                                         "the old lighthouse keeper's log", "the gulls"])
    # Verbatim parts of the message, in any casing; an invented subject is dropped.
    assert frame.topics == ("the harbour wall", "TIDE tables", "the old lighthouse keeper's log")
    assert relationship_topic(frame.topics) == "the harbour wall"
    assert relationship_topic(["  Los PLAZOS de entrega  ", "otro"]) == "los plazos de entrega"
    assert relationship_topic(["x" * 90]) == "x" * 40 and relationship_topic([]) == "" and relationship_topic([" "]) == ""


def test_the_relationship_topic_comes_from_the_reading_and_is_empty_without_one(tmp_path):
    from core.types import OpenLoop

    def first_loop(script):
        pipe = CognitivePipeline(ReadingBackend(), db_path=str(tmp_path / "topics.db"), message_reader=ScriptedReader(script))
        try:
            for topic in ("deadline", "garden"):
                pipe.relationship_memory.upsert_open_loop(OpenLoop(
                    source_person="reader", loop_kind="commitment", topic=topic, description=f"About the {topic}",
                    related_key=f"{topic}:1"))
            result = pipe.process("What about the garden, and that deadline?", "reader")
        finally:
            pipe.close()
        return result.debug.reading["topics"], result.debug.relationship_context.active_loops[0].topic

    message = "What about the garden, and that deadline?"
    assert first_loop(reading(conversation(message), topics=["deadline"])) == (["deadline"], "deadline")
    assert first_loop(reading(conversation(message), topics=["garden", "deadline"])) == (["garden", "deadline"], "garden")
    topics, _ = first_loop(None)   # no reading: no topic is guessed from the words
    assert topics == []


def test_identity_records_are_loaded_when_the_reading_says_the_message_asks_about_the_character_itself():
    loads = []

    def snapshot():
        loads.append(1)
        return {"beliefs": [{"statement": "Repair takes patience."}]}

    def turn(script, message):
        pipe = CognitivePipeline(ReadingBackend(), life_history_snapshot_provider=snapshot,
                                 message_reader=ScriptedReader(script))
        try:
            return pipe.process(message, "reader").debug
        finally:
            pipe.close()

    # In any language, by the reading's word alone.
    asked = turn(lambda text: reading(conversation(text), about_self=True, language="de"), "Wie hast du dich verändert?")
    assert asked.reading["about_self"] is True and "identity_records" in asked.life_history_context and loads == [1]
    # The same words the former patterns matched, read as not asking about the character.
    other = turn(lambda text: reading(conversation(text), about_self=False), "What do you believe?")
    assert "identity_records" not in other.life_history_context and loads == [1]
    unread = turn(None, "How have you changed?")
    assert unread.reading["about_self"] is False and "identity_records" not in unread.life_history_context and loads == [1]


# ---------------------------------------------------------------------------
# What a reading cannot do by naming things
# ---------------------------------------------------------------------------


def test_a_reference_a_component_carries_cannot_start_learning(tmp_path):
    """Only a task the reading reports as asking for the learning capability reaches the session's executor."""
    message = f"Resume esto en dos frases: {MATERIAL}"
    marked = {message: reading(task(message, user_value=0.8, effort=0.2,
                                    evidence_refs=["learning:pending", "learning:explicit"]), language="es")}
    config, records = session_turns(tmp_path, marked, message)
    debug, pending = records[0]
    work, = debug.turn_decision.components
    assert work.granted and work.component.requested_tool == ""
    assert not debug.tool_trace or not debug.tool_trace.executed_results
    with LifeHistoryStore(config) as store:
        assert store.list_experiences() == []


def test_the_outcome_of_tool_selection_is_kept_with_the_turn_and_a_study_is_not_paused(tmp_path):
    def failing_turn(research):
        executor = executor_at(tmp_path)
        native(executor, [], lambda responses: Refusing(responses, error=TimeoutError("timed out"), failures=1))
        pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, autonomy_level="autonomous",
                                 expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(
                                     lambda text: reading(task(text, user_value=0.8, effort=0.05,
                                                               tool=using("fs.list_dir", generate=["path"])))))
        pipe._research_mode = research
        try:
            result = pipe.process("look around", "reader")
            stored = pipe.relationship_memory.recent_events("reader", event_kind="turn_decision", limit=1)[0]
            return result, json.loads(stored.summary)["tool_selection"], pipe._tool_selection_pause
        finally:
            pipe.close()

    result, stored, pause = failing_turn(research=False)
    assert stored == result.debug.tool_selection == {
        "requests": 1, "status": "failed", "failure": TRANSPORT, "paused_turns_left": TOOL_SELECTION_PAUSE_TURNS}
    assert pause == TOOL_SELECTION_PAUSE_TURNS
    # A study keeps its own call contract: the failure is on its receipt, and nothing is paused.
    result, stored, pause = failing_turn(research=True)
    assert result.debug.tool_trace.executed_results[0].metadata[SELECTION_FAILURE] == TRANSPORT
    assert stored == {"requests": 1} and pause == 0


# ===========================================================================
# Fix round 1
# ===========================================================================

# ---------------------------------------------------------------------------
# A. Model tool selection is bounded by the grant
# ---------------------------------------------------------------------------


def tool_names(request):
    return sorted(tool["function"]["name"] for tool in request["tools"])


def open_read(message="Read the notes file you find most recent."):
    """The reproduced reading: a task that names file reading and leaves its path to be produced."""
    return reading(task(message, user_value=0.9, effort=0.05, tool=using("fs.read_file", generate=["path"])))


def test_selection_is_offered_only_the_capabilities_the_task_names(tmp_path):
    executor = executor_at(tmp_path, shell=True)
    client = native(executor, [_response(_tool_call("control__no_tool", {}))])
    message = "Read the notes file you find most recent."
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, autonomy_level="high_risk",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(open_read()))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert len(executor.registry.list_tools()) > 20                    # the operator allows far more
    assert tool_names(client.requests[0]) == ["control__no_tool", "fs__read_file"]
    assert result.debug.tool_selection == {"requests": 1} and not result.debug.tool_trace.executed_results


def test_a_selected_call_outside_the_tasks_capabilities_never_runs(tmp_path):
    """The reproduced case: asked to read, the selecting model answers with a write somewhere else."""
    message = "Read the notes file you find most recent."
    elsewhere = {"path": "elsewhere.txt", "content": "planted by the selecting model"}
    # A runner that executes whatever its model answered, as a runner does.
    executor = executor_at(tmp_path, shell=True)
    executor.set_tool_runner(RecordingRunner(executor, tool="fs.write_file", arguments=elsewhere))
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, autonomy_level="high_risk",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(open_read()))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    work, = result.debug.turn_decision.components
    assert work.granted and work.component.capability_requirements == ("fs.read_file",)
    receipt, = result.debug.tool_trace.executed_results
    assert not receipt.success and receipt.metadata["grant_denied"] is True
    assert "outside the capabilities" in receipt.error and not (tmp_path / "elsewhere.txt").exists()
    # The native runner never offers the tool, and a call to it by name is no tool it knows.
    executor = executor_at(tmp_path, shell=True)
    client = native(executor, [_response(_tool_call("fs__write_file", elsewhere)),
                               _response(_tool_call("shell__run_command", {"cmd": "touch elsewhere.txt"}, "c2")),
                               _response(_tool_call("control__no_tool", {}, "c3"))])
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, autonomy_level="high_risk",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(open_read()))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert all(tool_names(request) == ["control__no_tool", "fs__read_file"] for request in client.requests)
    assert not [r for r in result.debug.tool_trace.executed_results if r.success]
    assert not (tmp_path / "elsewhere.txt").exists()


def test_the_grant_check_holds_an_uncommitted_task_to_its_capabilities_and_written_operands(tmp_path):
    from core.grants import execution_denial

    message = "Copy notes/a.txt to wherever backups go."
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    registry.register(ToolCapability("demo.copy", "Copy a file", ToolCategory.WRITE, arg_schema={
        "source": {"type": "string", "required": True}, "target": {"type": "string", "required": True}}))
    registry.register(ToolCapability("demo.erase", "Erase a file", ToolCategory.DESTRUCTIVE, arg_schema={
        "path": {"type": "string", "required": True}}))
    ran = []
    executor.register_handler("demo.copy", lambda arguments: ran.append(arguments) or ToolResult("demo.copy", True, "copied"))
    executor.register_handler("demo.erase", lambda arguments: ran.append(arguments) or ToolResult("demo.erase", True, "erased"))
    runner = RecordingRunner(executor, calls=[("demo.copy", {"source": "notes/b.txt", "target": "backup/b.txt"})])
    executor.set_tool_runner(runner)
    script = ScriptedReader(reading(task(message, user_value=0.9, effort=0.05,
                                         tool=using("demo.copy", generate=["target"], source="notes/a.txt"))))
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, autonomy_level="high_risk",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=script)
    try:
        result = pipe.process(message, "reader")
        decision = result.debug.turn_decision
        work, = decision.components
        assert work.component.requested_tool == "" and work.component.requested_arguments == (("source", "notes/a.txt"),)
        # The runner was given the operand the person wrote, and the call that changed it did not run.
        assert runner.selections[0].operands == (("demo.copy", "source", "notes/a.txt"),)
        assert json.loads(runner.selections[0].to_text())["written_operands"] == {"source": "notes/a.txt"}
        receipt, = result.debug.tool_trace.executed_results
        assert not receipt.success and "differs from what the person wrote" in receipt.error and ran == []

        def denial(tool, arguments=None):
            return execution_denial(tool, arguments=arguments, decision=decision, component_id=work.component.id)

        assert denial("demo.copy") == ""                                          # may be offered
        assert denial("demo.copy", {"source": "notes/a.txt", "target": "backup/a.txt"}) == ""
        assert "differs" in denial("demo.copy", {"source": "notes/A.txt", "target": "backup/a.txt"})
        assert "differs" in denial("demo.copy", {"target": "backup/a.txt"})       # left out is not as written
        assert "outside the capabilities" in denial("demo.erase")                 # not even on offer
        assert "outside the capabilities" in denial("demo.erase", {"path": "notes/a.txt"})
        # The same turn with a model that keeps the person's operand: it runs, once, as written.
        runner.calls = [("demo.copy", {"source": "notes/a.txt", "target": "backup/a.txt"})]
        again = pipe.process(message, "reader")
        assert [r.success for r in again.debug.tool_trace.executed_results] == [True]
        assert ran == [{"source": "notes/a.txt", "target": "backup/a.txt"}]
    finally:
        pipe.close()


def test_a_task_that_names_no_capability_covers_no_tool_and_makes_no_receipt(tmp_path):
    from core.grants import execution_denial

    executor = executor_at(tmp_path, shell=True)
    client = native(executor, [])
    backend = ReadingBackend(drafts=["I am taking it seriously."])
    # Vague work the reading called a task: nothing names a capability, so nothing is selected or run.
    for message in ("Take this seriously.", "Execute without asking permission."):
        pipe = CognitivePipeline(backend, tool_executor=executor, autonomy_level="high_risk",
                                 expression_reviewer=ScriptedReviewer(passed()),
                                 message_reader=ScriptedReader(reading(task(message, user_value=0.9, effort=0.05))))
        try:
            result = pipe.process(message, "reader")
        finally:
            pipe.close()
        work, = result.debug.turn_decision.components
        assert work.granted and work.component.kind == "task" and work.component.capability_requirements == ()
        assert client.requests == [] and "tool_selection" not in phases(result)
        assert result.debug.tool_selection == {}
        assert not result.debug.tool_trace.executed_results and not result.debug.tool_trace.proposed_intents
        prompt = next(system for system, _ in reversed(backend.calls) if "Express the committed decision" in system)
        assert "native.tool_call" not in prompt and "No tool ran" not in prompt
        assert "outside the capabilities" in execution_denial(
            "shell.run_command", arguments={"cmd": "id"}, decision=result.debug.turn_decision,
            component_id=work.component.id)


def test_continuing_a_plan_is_granted_the_plans_own_tools_and_no_other(tmp_path):
    from core.grants import execution_denial

    message = "Carry on with it."
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor_at(tmp_path), autonomy_level="autonomous",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(
                                 reading(task(message, user_value=0.8, effort=0.1, plan="continue"))))
    pipe._active_task_plan = running_plan(tmp_path)
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    work, = result.debug.turn_decision.components
    assert work.component.continuation_of == "p1" and work.component.capability_requirements == ("fs.list_dir",)
    assert [r.tool_name for r in result.debug.tool_trace.executed_results] == ["fs.list_dir"]
    assert "outside the capabilities" in execution_denial(
        "fs.delete_path", arguments={"path": "x"}, decision=result.debug.turn_decision, component_id=work.component.id)


# ---------------------------------------------------------------------------
# B. A literal operand is a whole written unit, inside the part that asks for it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [
    ("build/output.log", "delete build/output.log.bak please"),            # a further extension
    ("https://example.com", "fetch https://example.com.evil.io/x"),        # more of the address
    ("ls", "also"),                                                        # inside a word
    ("/", "a/b"),                                                          # between two names
    ("rm -rf /tmp/a", "never ever run rm -rf /tmp/a/keep"),                # more of the path
    ("́x", "éx y"),                                             # begins on a combining mark
    ("e", "éx"),                                                     # stops before the mark that completes it
    ("notes", "notes2.txt"),                                               # a digit follows
    ("ファイル", "ファイル2.txtを削除"),                                    # more of the name follows
    ("draft", "read mydraft"),                                             # the end of a word
])
def test_a_piece_of_something_longer_is_not_what_the_person_wrote(value, text):
    assert exact_part(value, [text]) is None


@pytest.mark.parametrize(("value", "text"), [
    ("notes/plan.txt", "Please open notes/plan.txt."),                     # a trailing full stop
    ("my notes/plan.txt", 'open "my notes/plan.txt", then stop'),          # quoted
    ("ls -la | head -3", "run ls -la | head -3 for me"),                   # a command with spaces
    ("notes/plan.txt", "メモnotes/plan.txtを開いてください"),               # next to Kana and Han
    ("notes/plan.txt", "请打开notes/plan.txt这个文件"),
    ("notes/plan.txt", "เปิดไฟล์notes/plan.txtหน่อย"),                      # next to Thai
    ("notes/plan.txt", "افتح الملف notes/plan.txt، من فضلك"),               # an Arabic sentence, a Latin path
    ("(draft)", "save it as (draft) today"),                               # punctuation of its own
    ("build/output.log", "keep build/output.log.bak and delete build/output.log"),
    ("/", "cd / && ls"),
])
def test_a_whole_written_unit_is_taken_whatever_stands_around_it(value, text):
    assert exact_part(value, [text]) == value


def test_an_operand_must_stand_in_the_part_that_asks_for_it_or_in_an_earlier_message():
    message = "Read notes.txt for me. And whatever you do, never delete old.txt."
    asked = "Read notes.txt for me."
    inside = read(message, task(asked, tool=using("fs.read_file", path="notes.txt")),
                  conversation("And whatever you do, never delete old.txt."))
    assert inside.request_components[0].requested_arguments == (("path", "notes.txt"),) and refs(inside) == []
    # A value from another part of the same message is not this part's operand.
    outside = read(message, task(asked, tool=using("fs.read_file", path="old.txt")),
                   conversation("And whatever you do, never delete old.txt."))
    assert outside.request_components[0].requested_tool == ""
    assert refs(outside) == ["reading:argument_unquoted:c0:path"]
    # A part that stands nowhere in the message has no text of this message to take a value from.
    unplaced = read(message, task("read the notes", source_span="Open the notes file",
                                  tool=using("fs.read_file", path="notes.txt")))
    assert unplaced.request_components[0].requested_tool == ""
    # A part that refers back takes the value from where the person wrote it, whole.
    earlier = [{"role": "user", "content": "My plan is in notes/plan.md, the old one in notes/plan.md.bak."}]
    back = read("Read it.", task("Read it.", tool=using("fs.read_file", path="notes/plan.md")), recent=earlier)
    assert back.request_components[0].requested_arguments == (("path", "notes/plan.md"),)
    cut = read("Read it.", task("Read it.", tool=using("fs.read_file", path="notes/plan")), recent=earlier)
    assert cut.request_components[0].requested_tool == "" and refs(cut) == ["reading:argument_unquoted:c0:path"]


# ---------------------------------------------------------------------------
# C. Questions that need a capability, and work nothing offered can do
# ---------------------------------------------------------------------------


def test_the_reader_is_told_that_a_question_can_be_a_task_and_where_to_say_what_is_missing():
    assert "A question is a task when answering it" in READING_SYS_PROMPT
    component = reading_schema()["properties"]["request_components"]["items"]["properties"]
    assert component["unavailable"] == {"type": "array", "items": {"type": "string"}}
    assert '"unavailable"' in READING_SYS_PROMPT


def test_a_capability_that_needs_no_operand_is_committed_by_naming_it_and_runs_without_a_tool_runner(tmp_path):
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    registry.register(ToolCapability("system.cpu_usage", "Current CPU usage", ToolCategory.READ_ONLY))
    registry.register(ToolCapability("demo.search", "Search", ToolCategory.READ_ONLY,
                                     arg_schema={"query": {"type": "string", "required": True}}))
    executor.register_handler("system.cpu_usage", lambda arguments: ToolResult("system.cpu_usage", True, "cpu 12%"))
    message = "¿Cuánta CPU se está usando ahora?"
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(reading(task(
                                 message, user_value=0.8, effort=0.05, capability_requirements=["system.cpu_usage"]),
                                 language="es")))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    work, = result.debug.turn_decision.components
    assert work.component.requested_tool == "system.cpu_usage" and work.component.requested_arguments == ()
    assert [(r.tool_name, r.output) for r in result.debug.tool_trace.executed_results] == [("system.cpu_usage", "cpu 12%")]
    assert result.debug.tool_selection == {}             # no selection was needed, and none was asked for
    # A capability that needs an operand is not committed by its name alone, and a bare name says nothing.
    offers = offers_of(registry)
    needs = read("Search for tide tables", task("Search for tide tables", capability_requirements=["demo.search"]),
                 offers=offers)
    assert needs.request_components[0].requested_tool == ""
    bare = read(message, task(message, capability_requirements=["system.cpu_usage"]), offers=["system.cpu_usage"])
    assert bare.request_components[0].requested_tool == ""


def missing_work_turn(tmp_path, messages, unavailable):
    backend = ReadingBackend(drafts=[f"I cannot do that here, reply {index}." for index in range(6)])
    seen = []

    def review(draft, evidence):
        seen.append(evidence["decision"]["components"])
        return passed()

    script = {message: reading(task(message, user_value=0.9, effort=0.1, unavailable=[missing]))
              for message, missing in zip(messages, unavailable)}
    pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path), autonomy_level="high_risk",
                             expression_reviewer=ScriptedReviewer(review), message_reader=ScriptedReader(script))
    try:
        results = [pipe.process(message, "reader") for message in messages]
        gaps = [dict(gap) for gap in pipe._capability_gaps.values()]
    finally:
        pipe.close()
    return backend, results, seen, gaps


def test_work_that_nothing_offered_can_do_is_not_granted_is_said_truthfully_and_is_kept_as_a_gap(tmp_path):
    messages = ("¿Puedes leer este PDF por mí?", "Intenta el PDF otra vez; necesito su contenido.")
    # The second reading was shown the gap already known and states the same need in the same words.
    backend, results, seen, gaps = missing_work_turn(
        tmp_path, messages, ("leer un archivo PDF", "Leer un archivo PDF."))
    for result in results:
        work, = result.debug.turn_decision.components
        assert not work.component.feasible and work.best_grant is None and not result.debug.turn_decision.grants
        assert work.component.capability_requirements == () and work.component.kind == "task"
        assert not result.debug.tool_trace.executed_results and result.debug.tool_selection == {}
    # The writer and the reviewer are told the same thing, as data.
    prompt = next(system for system, _ in backend.calls if "Express the committed decision" in system)
    assert '"possible": false' in prompt and '"unavailable": "leer un archivo PDF"' in prompt
    assert (seen[0][0]["choice"], seen[0][0]["possible"], seen[0][0]["unavailable"]) == (
        "decline", False, "leer un archivo PDF")
    # The same need reported again is the same gap: its recurrence is what a skill want starts from.
    gap, = gaps
    assert gap["gap_type"] == "leer un archivo PDF" and gap["recent_recurrence"] == 2
    assert gap["frustration_intensity"] == 0.9


def test_what_is_unavailable_is_reported_for_a_task_that_names_no_capability_only():
    message = "Run this shell command: echo done"
    stands = read(message, task(message, unavailable=["running a shell command"]), offers=(READ, WRITE))
    assert not stands.request_components[0].feasible and stands.unavailable == (("c0", "running a shell command"),)
    assert refs(stands) == ["capability:unavailable:c0"]
    talk = read(message, conversation(message, unavailable=["running a shell command"]), offers=(READ, WRITE))
    assert talk.request_components[0].feasible and talk.unavailable == ()
    assert refs(talk) == ["reading:unavailable_not_task:c0"]
    named = read(message, task(message, unavailable=["running a shell command"],
                               tool=using("shell.run_command", cmd="echo done")))
    assert named.request_components[0].feasible and named.request_components[0].requested_tool == "shell.run_command"
    assert named.unavailable == () and refs(named) == ["reading:unavailable_with_capability:c0"]


# ---------------------------------------------------------------------------
# D. A reproduction stands for a task only, and never for the whole of what asks for it
# ---------------------------------------------------------------------------


def test_a_reproduction_that_is_the_whole_span_or_stands_on_talk_does_not_waive_the_copy_check():
    whole = SENTENCE
    # A conversation component whose reproduce equals its span: an echo of it is still an echo.
    for script in (reading(conversation(whole, reproduce=whole)),
                   reading(task(whole, user_value=0.9, effort=0.02, reproduce=whole))):
        pipe = CognitivePipeline(ReadingBackend(drafts=[whole, "I would rather answer in my own words."]),
                                 message_reader=ScriptedReader(script), expression_reviewer=ScriptedReviewer(passed()))
        try:
            echoed = pipe.process(whole, "reader")
        finally:
            pipe.close()
        assert echoed.response != whole and "user_message_echo" in echoed.debug.dignity_issues
        assert echoed.debug.appraisal_frame.reproductions == ()
    talk = read(REPRODUCE, conversation(REPRODUCE, reproduce=SENTENCE))
    assert talk.reproductions == () and refs(talk) == ["reading:reproduction_not_task:c0"]
    covering = read(REPRODUCE, task(ASKING, reproduce=REPRODUCE))
    assert covering.reproductions == () and refs(covering) == ["reading:reproduction_whole:c0"]
    # One sentence among others, given as its own reproduction: the same, wherever it stands in the message.
    among = f"Good morning. {SENTENCE} That is all for today."
    for reproduced in (SENTENCE, f"Good morning. {SENTENCE}"):
        equal = read(among, task(SENTENCE, reproduce=reproduced), conversation("That is all for today."))
        assert equal.reproductions == () and refs(equal) == ["reading:reproduction_whole:c0"]
    shorter = read(among, task(f"Good morning. {SENTENCE}", reproduce=SENTENCE), conversation("That is all for today."))
    assert shorter.reproductions == (("c0", SENTENCE),) and refs(shorter) == []
    # Strictly shorter than its span, or from an earlier turn: it stands.
    inside = read(REPRODUCE, task(REPRODUCE, reproduce=SENTENCE))
    assert inside.reproductions == (("c0", SENTENCE),) and refs(inside) == []
    earlier = [{"role": "user", "content": SENTENCE}]
    recalled = read("Say that again.", task("Say that again.", reproduce=SENTENCE), recent=earlier)
    assert recalled.reproductions == (("c0", SENTENCE),)


# ---------------------------------------------------------------------------
# E. Placeholders and slips do not sink the reading or spend the retry
# ---------------------------------------------------------------------------


def test_placeholder_entries_state_nothing_and_are_passed_over():
    message = "How was your day?"
    for placeholder in ({"tool": [{"name": "", "arguments": [], "generate": []}]},
                        {"tool": [{"name": "  ", "arguments": [{"name": "", "value": ""}], "generate": []}]},
                        {"tool": [{"name": "fs.read_file", "arguments": [{"name": "", "value": "x"}], "generate": ["path"]}]}):
        frame = read(message, task(message, **placeholder))
        component, = frame.request_components
        assert component.requested_tool == "" and component.requested_arguments == ()
    assert frame.request_components[0].capability_requirements == ("fs.read_file",) and refs(frame) == []
    empty = read(message, conversation(message, tool=[{"name": "", "arguments": [], "generate": []}]))
    assert refs(empty) == [] and empty.request_components[0].kind == "conversation"


@pytest.mark.parametrize(("component", "plan", "ref"), [
    (conversation("What is in notes/a.txt?", tool=using("fs.read_file", path="notes/a.txt")), False,
     "reading:tool_not_task:c0"),
    (conversation("What is in notes/a.txt?", plan="continue"), True, "reading:plan_not_task:c0"),
    (task("What is in notes/a.txt?", plan="continue"), False, "reading:plan_absent:c0"),
    (role("What is in notes/a.txt?", "notes/a.txt", autonomy_cost=0.2, reproduce="What is in notes/a.txt?"), False,
     "reading:reproduction_not_task:c0"),
])
def test_a_minor_slip_is_removed_and_recorded_without_a_second_reading(tmp_path, component, plan, ref):
    message = "What is in notes/a.txt?"
    backend = ReadingBackend({message: [reading(component), reading(conversation(message))]})
    pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path))
    if plan:
        pipe._active_task_plan = running_plan(tmp_path)
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert len(backend.reading_calls) == 1
    assert [item["outcome"] for item in result.debug.reading["attempts"]] == ["read"]
    assert ref in [e.ref for e in result.debug.turn_decision.perception.evidence]
    kept, = result.debug.turn_decision.components
    assert kept.component.kind == component["kind"] and kept.component.requested_tool == ""
    assert not result.debug.tool_trace.executed_results


def test_one_tool_is_committed_and_whatever_else_the_component_named_is_recorded_as_ignored():
    message = "Read notes/a.txt and tell me what it says"
    extra = read(message, task(message, capability_requirements=["fs.read_file", "shell.run_command"],
                               tool=using("fs.read_file", path="notes/a.txt")))
    component, = extra.request_components
    assert component.requested_tool == "fs.read_file" and component.capability_requirements == ("fs.read_file",)
    assert refs(extra) == ["reading:capability_ignored:c0:shell.run_command"]
    # Operands given beside several capabilities are kept for the capability they were given for.
    message = "Write a summary of notes/a.txt to out/sum.txt"
    mixed = read(message, task(message, capability_requirements=["fs.read_file"],
                               tool=using("fs.write_file", generate=["content"], path="out/sum.txt")))
    component, = mixed.request_components
    assert component.capability_requirements == ("fs.read_file", "fs.write_file")
    assert component.requested_tool == "" and refs(mixed) == []
    assert held_operands(component) == (("fs.write_file", "path", "out/sum.txt"),)
    both = read(message, task(message, tool=using("fs.read_file", path="notes/a.txt")
                              + using("fs.write_file", generate=["content"], path="out/sum.txt")))
    assert held_operands(both.request_components[0]) == (
        ("fs.read_file", "path", "notes/a.txt"), ("fs.write_file", "path", "out/sum.txt"))
    assert refs(both) == []


# ---------------------------------------------------------------------------
# G. The selecting model is given the conversation and this turn's receipts
# ---------------------------------------------------------------------------


def test_the_runner_is_given_recent_turns_and_the_receipts_of_this_turn_as_data(tmp_path):
    (tmp_path / "small.txt").write_text("a", encoding="utf-8")
    (tmp_path / "large.txt").write_text("a much longer file", encoding="utf-8")
    executor = executor_at(tmp_path)
    runner = RecordingRunner(executor, calls=[("fs.read_file", {"path": "large.txt"})])
    executor.set_tool_runner(runner)
    first, second = "Which files do I keep here?", "List this folder, then read the biggest file."
    script = {first: reading(conversation(first)),
              second: reading(task("List this folder", source_span="List this folder", user_value=0.9, effort=0.05,
                                   tool=using("fs.list_dir", generate=["path"])),
                              task("then read the biggest file", user_value=0.9, effort=0.05,
                                   tool=using("fs.read_file", generate=["path"])))}
    runner.calls.insert(0, ("fs.list_dir", {"path": "."}))
    pipe = CognitivePipeline(ReadingBackend(drafts=["You keep two files here.", "It says: a much longer file."]),
                             tool_executor=executor, autonomy_level="autonomous",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(script))
    try:
        pipe.process(first, "reader")
        result = pipe.process(second, "reader")
    finally:
        pipe.close()
    listing, reading_file = runner.selections
    # Each selection is one task with the capabilities it names, and nothing is matched in its text.
    assert (listing.request, listing.capabilities) == ("List this folder", ("fs.list_dir",))
    assert reading_file.capabilities == ("fs.read_file",)
    # The conversation so far, by speaker.
    assert listing.recent_turns == (("user", first), ("character", "You keep two files here."))
    # What this turn already executed: the second task can be resolved from the first one's receipt.
    assert listing.receipts == ()
    receipt, = reading_file.receipts
    assert receipt["tool"] == "fs.list_dir" and receipt["success"] and "large.txt" in receipt["output"]
    given = json.loads(reading_file.to_text())
    assert set(given) == {"request", "recent_turns", "receipts"} and given["receipts"][0]["tool"] == "fs.list_dir"
    assert [r.tool_name for r in result.debug.tool_trace.executed_results] == ["fs.list_dir", "fs.read_file"]
    assert result.debug.tool_trace.executed_results[1].output == "a much longer file"


def test_the_native_runner_sends_the_task_as_one_json_object_and_says_what_its_fields_are(tmp_path):
    from openrook_tools.native_orchestrator import SELECTION_INPUT

    executor = executor_at(tmp_path)
    client = native(executor, [_response(_tool_call("control__no_tool", {}))])
    first, second = "The second link was https://example.com/b.", "Open the second link you showed me."
    script = {first: reading(conversation(first)),
              second: reading(task(second, user_value=0.9, effort=0.05, tool=using("fs.read_file", generate=["path"])))}
    pipe = CognitivePipeline(ReadingBackend(drafts=["Noted.", "Done."]), tool_executor=executor,
                             autonomy_level="autonomous", expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(script))
    try:
        pipe.process(first, "reader")
        pipe.process(second, "reader")
    finally:
        pipe.close()
    system, user = (message["content"] for message in client.requests[0]["messages"][:2])
    assert SELECTION_INPUT in system
    assert json.loads(user) == {"request": second, "recent_turns": [
        {"speaker": "user", "text": first}, {"speaker": "character", "text": "Noted."}]}


# ---------------------------------------------------------------------------
# H. Long text is given by its first and last words
# ---------------------------------------------------------------------------

PASTED = ("Queridos todos: " + "la marea sube despacio y el puerto se llena de barcas que vuelven. " * 45
          + "Nos vemos mañana al amanecer.")


def test_a_long_operand_is_given_by_reference_and_what_is_committed_is_the_messages_own_text(tmp_path):
    message = f"Guarda esto en notas/carta.txt: {PASTED} ¡Gracias!"
    assert len(PASTED) > 3000
    stated = {"description": "guardar el texto en un archivo", "kind": "task",
              "source_span": "Guarda esto en notas/carta.txt:", "span_end": "Nos vemos mañana al amanecer.",
              "user_value": 0.9, "effort": 0.05,
              "tool": [{"name": "fs.write_file", "generate": [], "arguments": [
                  {"name": "path", "value": "notas/carta.txt", "end": ""},
                  {"name": "content", "value": "Queridos todos: la marea", "end": "Nos vemos mañana al amanecer."}]}]}
    # The reader copied none of the pasted text beyond a few words at each end.
    assert len(json.dumps(stated, ensure_ascii=False)) < 500
    assert max(len(value) for value in (stated["source_span"], stated["span_end"],
                                        *(a[key] for a in stated["tool"][0]["arguments"] for key in ("value", "end")))) < 200
    frame = read(message, stated, language="es")
    component, = frame.request_components
    assert component.requested_tool == "fs.write_file" and refs(frame) == []
    assert dict(component.requested_arguments) == {"path": "notas/carta.txt", "content": PASTED}
    # The span is the message's own stretch too, so the operand stands inside the part that asks for it.
    assert component.source_span == f"Guarda esto en notas/carta.txt: {PASTED}" and component.description == stated["description"]
    pipe = CognitivePipeline(ReadingBackend(drafts=["Guardado."]), tool_executor=executor_at(tmp_path),
                             autonomy_level="autonomous", expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(reading(stated, language="es")))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert [r.success for r in result.debug.tool_trace.executed_results] == [True]
    assert (tmp_path / "notas/carta.txt").read_text(encoding="utf-8") == PASTED


def test_a_reference_must_say_which_stretch_is_meant():
    from core.perception import passage

    text = "Save this: one two three. Then: one two three four. Thanks."
    assert passage("one two", "four.", text) is None                      # the first words stand twice
    assert text[slice(*passage("Then: one", "four.", text))] == "Then: one two three four."
    assert passage("Save this:", "three", text) is None                   # the last words stand twice
    assert passage("four.", "Save", text) is None                         # not in that order
    assert passage("ave this:", "Thanks.", text) is None                  # begins inside a word
    assert passage("Save this:", "Thank", text) is None                   # ends inside a word
    assert passage("Save this:", "nowhere", text) is None
    message = "Store this in a.txt: alpha beta gamma. And again: alpha beta gamma."
    ambiguous = read(message, task(message, tool=[{"name": "fs.write_file", "generate": [], "arguments": [
        {"name": "path", "value": "a.txt", "end": ""}, {"name": "content", "value": "alpha beta", "end": "gamma."}]}]))
    assert ambiguous.request_components[0].requested_tool == ""
    assert refs(ambiguous) == ["reading:argument_unquoted:c0:content"]
    unplaced = read(message, {**task("store it"), "source_span": "Keep this in", "span_end": "gamma twice."})
    assert refs(unplaced) == ["reading:span_unquoted:c0:end"]


def test_a_long_text_to_give_back_is_given_by_reference_too():
    message = f"{PASTED} Repíteme eso, palabra por palabra."
    frame = read(message, task("Repíteme eso, palabra por palabra.", reproduce="Queridos todos: la marea",
                               reproduce_end="Nos vemos mañana al amanecer."), language="es")
    assert frame.reproductions == (("c0", PASTED),) and refs(frame) == []
    lost = read(message, task("Repíteme eso, palabra por palabra.", reproduce="Queridos todos: la marea",
                              reproduce_end="hasta nunca."), language="es")
    assert lost.reproductions == () and refs(lost) == ["reading:reproduction_unquoted:c0"]


def test_learning_material_is_the_message_without_the_asking_span_and_is_never_copied_into_the_reading(tmp_path):
    message = f"Aprende de esto: {MATERIAL}"
    script = {message: reading(task("Aprende de esto", user_value=0.8, effort=0.2, tool=using(LEARNING)), language="es")}
    assert MATERIAL.strip() not in json.dumps(script[message], ensure_ascii=False)
    config, records = session_turns(tmp_path, script, message)
    receipt, = records[0][0].tool_trace.executed_results
    assert receipt.success
    with LifeHistoryStore(config) as store:
        experience, = store.list_experiences(limit=5)
    assert experience["raw_excerpt"].startswith("Las mareas enseñan paciencia")


# ---------------------------------------------------------------------------
# I. Smaller items
# ---------------------------------------------------------------------------


def test_the_artifact_draft_is_a_counted_model_call_with_telemetry_and_a_deadline(tmp_path):
    message = "Write a short note about tides and save it to out/tides.txt"
    draft = json.dumps({"content_lines": ["Tides come and go."], "summary": "note"})
    pipe = CognitivePipeline(
        ReadingBackend(drafts=[draft, "Saved it."]), tool_executor=executor_at(tmp_path), autonomy_level="autonomous",
        expression_reviewer=ScriptedReviewer(passed()),
        runtime_config=SimpleNamespace(llm_request_timeout_seconds=45.0),
        message_reader=ScriptedReader(reading(task(message, user_value=0.8, effort=0.2, tool=using(
            "fs.write_file", generate=["content"], path="out/tides.txt")))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    assert (tmp_path / "out/tides.txt").read_text(encoding="utf-8") == "Tides come and go.\n"
    call, = [item for item in result.debug.llm_calls if item["phase"] == "artifact_draft"]
    # The task's one protected call, required, and held to the request deadline in force.
    assert call["primary_tool"] is True and call["optional"] is False and call["status"] == "ok"
    assert call["timeout_seconds"] == 45.0
    assert "tool_selection" not in phases(result)        # a task spends that call on one or the other
    assert phases(result) == ["artifact_draft", "generation"]


def test_learning_named_without_a_tool_entry_runs_and_a_receipt_that_never_ran_is_not_a_failure(tmp_path):
    from core.review import decision_projection

    message = f"Aprende de esto: {MATERIAL}"
    by_name = {message: reading(task("Aprende de esto", user_value=0.8, effort=0.2,
                                     capability_requirements=[LEARNING]), language="es")}
    config, records = session_turns(tmp_path, by_name, message)
    debug, _ = records[0]
    work, = debug.turn_decision.components
    assert work.component.requested_tool == LEARNING and LEARNING_GRANT in work.component.evidence_refs
    receipt, = debug.tool_trace.executed_results
    assert receipt.success and receipt.tool_name == LEARNING
    # A committed operation whose receipt says it never started was not attempted.
    stopped = ToolResult(LEARNING, False, "", error="Not attempted", metadata={
        **receipt.metadata, "execution_started": False})
    projected, = decision_projection(debug.turn_decision, [stopped])["components"]
    assert (projected["outcome"], projected["receipt_indices"]) == ("not attempted", [0])
    ran, = decision_projection(debug.turn_decision, [receipt])["components"]
    assert ran["outcome"] == "succeeded"
    broke = ToolResult(LEARNING, False, "", error="the store was locked", metadata=dict(receipt.metadata))
    assert decision_projection(debug.turn_decision, [broke])["components"][0]["outcome"] == "failed"


def test_a_required_argument_that_is_not_text_is_offered_as_one_to_produce_in_data_and_in_code():
    registry = ToolRegistry()
    registry.register(ToolCapability("demo.repeat", "Repeat a line", ToolCategory.READ_ONLY, arg_schema={
        "line": {"type": "string", "required": True}, "times": {"type": "number", "required": True},
        "loud": {"type": "boolean", "required": False}}))
    offer, = offers_of(registry)
    assert offer.to_data() == {"name": "demo.repeat", "purpose": "Repeat a line", "arguments": ["line", "times"],
                               "produced_arguments": ["times"]}
    assert offer.takes("times") and not offer.literal("times") and not offer.takes("loud")
    from core.request_routing import as_offers
    assert as_offers([offer.to_data()]) == (offer,)        # the reader's data says the same as the registry
    message = "Repeat hello there 3 times"
    for offers in ((offer,), as_offers([offer.to_data()])):
        frame = read(message, task(message, tool=using("demo.repeat", line="hello there", times="3")), offers=offers)
        component, = frame.request_components
        # The number is produced when the task is carried out; the line the person wrote is held.
        assert component.requested_tool == "" and component.requested_arguments == (("line", "hello there"),)
        assert refs(frame) == []
    assert "produced_arguments" in READING_SYS_PROMPT


def test_the_operator_setting_off_blocks_a_committed_shell_command_in_a_whole_turn():
    from tests.test_native_orchestrator import _fake_shell_executor

    message = "Run echo hello there"
    script = ScriptedReader(operations((message, "shell.run_command", {"cmd": "echo hello there"})))

    def turn(level):
        executor = _fake_shell_executor()
        backend = ReadingBackend(drafts=["Here is what happened."])
        pipe = CognitivePipeline(backend, tool_executor=executor, autonomy_level=level,
                                 expression_reviewer=ScriptedReviewer(passed()), message_reader=script)
        try:
            return backend, pipe.process(message, "reader")
        finally:
            pipe.close()

    backend, blocked = turn("off")
    work, = blocked.debug.turn_decision.components
    # The character granted exactly this command; the operator's setting is a separate gate, and it holds.
    assert work.granted and work.component.requested_tool == "shell.run_command"
    assert dict(work.component.requested_arguments) == {"cmd": "echo hello there"}
    assert not blocked.debug.tool_trace.executed_results
    gate = blocked.debug.tool_trace.final_decision
    assert gate.decision == "refuse" and "Autonomy mode is off" in gate.rationale
    prompt = next(system for system, _ in backend.calls if "Express the committed decision" in system)
    assert "Operator decision: refuse" in prompt
    _, allowed = turn("high_risk")
    assert [r.output for r in allowed.debug.tool_trace.executed_results] == ["ran echo hello there"]


def test_how_telling_a_word_is_comes_from_the_whole_store_when_one_person_has_few_memories():
    from core.memory.semantic import SQLiteSemanticMemory
    from core.types import SemanticMemoryEntry

    memory = SQLiteSemanticMemory(":memory:")
    try:
        for index in range(6):   # other people's memories show which words are everywhere
            memory.store(SemanticMemoryEntry(kind="episode", source_person=f"other{index}",
                                             summary=f"wir sprachen über das wetter {index}", content=""))
        memory.store(SemanticMemoryEntry(kind="episode", source_person="ana", summary="wir sprachen über gezeiten"))
        memory.store(SemanticMemoryEntry(kind="episode", source_person="ana", summary="wir sprachen über das segeln"))
        first, second = memory.retrieve("erzähl mir über gezeiten", source_person="ana", limit=2)
        assert "gezeiten" in first.summary
        # "über" is in every stored memory: it tells nothing apart, so the other memory shares no
        # telling word with the question and scores for the person and its recency alone.
        assert second.score == pytest.approx(0.45, abs=0.01)
        assert first.score - second.score == pytest.approx(2.5 / 3, abs=0.01)
    finally:
        memory.close()


# ---------------------------------------------------------------------------
# F. What a turn keeps as memory comes from the reading, not from English patterns
# ---------------------------------------------------------------------------


def kept_kinds(script, message):
    pipe = CognitivePipeline(ReadingBackend(drafts=["Entendido."]), message_reader=ScriptedReader(script),
                             expression_reviewer=ScriptedReviewer(passed()))
    try:
        result = pipe.process(message, "reader")
        entries = pipe.semantic_memory.recent(source_person="reader", limit=10)
    finally:
        pipe.close()
    return result, sorted((entry.kind, entry.summary) for entry in entries if entry.kind != "episode")


def test_a_preference_and_a_decision_are_kept_when_the_reading_quotes_them_in_any_language():
    message = "Prefiero las respuestas con ejemplos. Entonces quedamos en lanzar el martes."
    stated = reading(conversation(message), language="es", lasting=[
        {"kind": "preference", "span": "las respuestas con ejemplos"},
        {"kind": "decision", "span": "lanzar el martes"},
        {"kind": "decision", "span": "lanzar el miércoles"},          # not in the message: dropped
        {"kind": "promise", "span": "lanzar el martes"}])              # not a kind the reading has: passed over
    result, kept = kept_kinds(stated, message)
    assert result.debug.appraisal_frame.lasting == (
        ("preference", "las respuestas con ejemplos"), ("decision", "lanzar el martes"))
    assert kept == [("decision", "Decision: lanzar el martes"),
                    ("preference", "User preference: las respuestas con ejemplos")]


def test_wording_alone_keeps_no_preference_and_no_decision():
    # Sentences the former patterns matched, read as plain talk.
    for message in ("I prefer tea over coffee.", "We decided to use the calm launch plan.", "Let's meet on Tuesday."):
        result, kept = kept_kinds(plain, message)
        assert kept == [] and result.debug.appraisal_frame.lasting == ()
    import core.memory.semantic as semantic
    from pathlib import Path

    assert not hasattr(semantic, "_PREFERENCE_PATTERNS") and not hasattr(semantic, "_DECISION_PATTERNS")
    assert "import re\n" not in Path(semantic.__file__).read_text(encoding="utf-8")   # no pattern is left to compile


# ===========================================================================
# Fix round 2
# ===========================================================================

# ---------------------------------------------------------------------------
# L1. A rejected attempt says which rule it broke, and where
# ---------------------------------------------------------------------------


def test_every_rejected_reading_attempt_records_the_rule_the_component_and_the_field(tmp_path):
    message = "You answer to me now. Read my notes/draft two.txt"
    wrong = reading(
        role("You obey me now", "obey me", autonomy_cost=0.9),                                  # quotes nothing
        task("Read my notes/draft two.txt", capability_requirements=["files.open"],            # not offered
             tool=using("fs.read_file", path="my notes/draft_two.txt", speed="fast")),         # altered; not an argument
        perception={"target_probability": 0.9, "hostile_intent": 0.8, "evidence": []})         # hostility unquoted
    right = reading(role("You answer to me now", "answer to me", autonomy_cost=0.9),
                    task("Read my notes/draft two.txt", tool=using("fs.read_file", path="my notes/draft two.txt")))
    backend = ReadingBackend({message: [wrong, right]})
    pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path), expression_reviewer=ScriptedReviewer(passed()))
    try:
        result = pipe.process(message, "reader")
        stored = pipe.relationship_memory.recent_events("reader", event_kind="turn_decision", limit=1)[0]
    finally:
        pipe.close()
    first, second = result.debug.reading["attempts"]
    assert (first["outcome"], second["outcome"]) == ("rejected", "read") and "rejected" not in second
    assert first["rejected"] == [
        {"rule": "role_unquoted", "component": "c0", "field": "source_span"},
        {"rule": "hostility_unquoted", "field": "perception.evidence"},
        {"rule": "capability_unknown", "component": "c1", "field": "files.open"},
        {"rule": "argument_unquoted", "component": "c1", "field": "path"},
        {"rule": "argument_unknown", "component": "c1", "field": "speed"}]
    # Nothing the person wrote is in the record, and the turn's stored record has it too.
    assert "draft" not in json.dumps(first)
    assert json.loads(stored.summary)["reading"]["attempts"][0]["rejected"] == first["rejected"]
    # An answer that could not be read at all says why as well.
    unreadable = ReadingBackend({message: ["not json", right]})
    attempts = read_message(message, backend=unreadable).reading_attempts
    assert (attempts[0]["outcome"], attempts[0]["reason"]) == ("malformed reading", "the answer is not a JSON object")


# ---------------------------------------------------------------------------
# R1. What was cut off may not start with a symbol either
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [
    ("projects/tmp", "Delete ~/projects/tmp please"),         # the home directory in front of it
    ("/etc/passwd", "Delete /etc/passwd~ please"),            # a backup suffix
    ("build", "Delete build/* please"),                       # a pattern
    ("etc/passwd", "show me /etc/passwd"),                    # the leading slash
    ("8.8.8.8", "ping -8.8.8.8 once"),                        # a sign or an option
    ("example.com/a", "mail @example.com/a today"),           # an at sign
    ("bashrc", "edit .bashrc now"),                           # a leading dot is the first character of the name
    ("/x", "open ./x now"),                                   # relative, not absolute
    ("notes.txt", "open notes.txt:12 now"),                   # a colon that joins
    ("a.txt", "copy a.txt=b.txt"),
    ("report", "read report#3"),
])
def test_a_symbol_next_to_a_value_belongs_to_what_the_person_wrote(value, text):
    assert written(value, text) is None and exact_part(value, [text]) is None


@pytest.mark.parametrize(("value", "text"), [
    ("~/projects/tmp", "Delete ~/projects/tmp please"), ("/etc/passwd~", "Delete /etc/passwd~ please"),
    ("build/*", "Delete build/* please"), ("-8.8.8.8", "ping -8.8.8.8 once"),
    ("notes.txt", "Can you read (notes.txt)?"), ("notes.txt", "read «notes.txt»!"),
    ("notes.txt", "读一下「notes.txt」。"), ("a.txt", "copy a.txt; then stop"),
    ("notes.txt", "Save this to notes.txt: the tide is in."),            # a colon that introduces
    ("notes.txt", "保存到notes.txt：潮水来了"), ("ملاحظات.txt", "افتح ملاحظات.txt؟"),
])
def test_punctuation_around_a_value_does_not_cut_it(value, text):
    assert exact_part(value, [text]) == value


@pytest.mark.parametrize(("message", "cut"), [
    ("Delete ~/projects/tmp please", "projects/tmp"), ("Delete /etc/passwd~ please", "/etc/passwd"),
    ("Delete build/* please", "build"),
])
def test_a_cut_down_operand_is_not_committed(message, cut):
    frame = read(message, task(message, tool=using("fs.read_file", path=cut)))
    component, = frame.request_components
    assert component.requested_tool == "" and component.requested_arguments == ()
    assert refs(frame) == ["reading:argument_unquoted:c0:path"]


# ---------------------------------------------------------------------------
# R2. A rejected operand is not a free choice for the selecting model
# ---------------------------------------------------------------------------

NEVER = "Never run rm -rf ~ please. Run echo hi now."


def shell_turn(script, calls, message=NEVER, history=()):
    from tests.test_native_orchestrator import _fake_shell_executor

    executor = _fake_shell_executor()
    runner = RecordingRunner(executor, calls=calls)
    executor.set_tool_runner(runner)
    pipe = CognitivePipeline(ReadingBackend(drafts=[f"Reply {index}." for index in range(8)]), tool_executor=executor,
                             autonomy_level="high_risk", expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(script))
    try:
        for earlier in history:
            pipe.process(earlier, "reader")
        return runner, pipe.process(message, "reader")
    finally:
        pipe.close()


def test_a_value_the_person_did_not_write_cannot_be_given_for_an_operand_the_reading_got_wrong():
    from core.grants import execution_denial

    # The reading put the command of the first sentence on the part that is the second one.
    wrong = reading(conversation("Never run rm -rf ~ please."),
                    task("Run echo hi now.", user_value=0.9, effort=0.05, tool=using("shell.run_command", cmd="rm -rf ~")))
    runner, result = shell_turn(wrong, [("shell.run_command", {"cmd": "rm -rf ~"})])
    _, work = result.debug.turn_decision.components
    assert work.granted and work.component.requested_tool == "" and work.component.requested_arguments == ()
    assert work.component.capability_requirements == ("shell.run_command",)
    # The argument's name stays on the component, in the frozen decision.
    assert unverified_operands(work.component) == (("shell.run_command", "cmd"),)
    receipt, = result.debug.tool_trace.executed_results
    assert not receipt.success and receipt.metadata["grant_denied"] and "did not write" in receipt.error

    def denial(cmd, earlier=(), message=NEVER):
        from core.grants import execution_scope
        with execution_scope(result.debug.turn_decision, work.component.id, message=message, earlier=earlier):
            return execution_denial("shell.run_command", arguments={"cmd": cmd})

    assert "did not write" in denial("rm -rf ~")                 # another part of the message
    assert "did not write" in denial("echo hi && rm -rf ~")      # more than the person wrote
    assert denial("echo hi") == "" and denial("echo hi now") == ""
    assert "did not write" in denial(["echo", "hi"])             # not text at all
    # A value from an earlier message of the person stands when the turn is given those messages.
    assert "did not write" in denial("uptime") and denial("uptime", earlier=("Please check uptime first.",)) == ""
    # Without the turn's message the component's own span is all there is to look in.
    assert denial("echo hi", message="") == "" and "did not write" in denial("rm -rf ~", message="")
    # The same turn with a model that keeps to what this part says: it runs.
    runner, result = shell_turn(wrong, [("shell.run_command", {"cmd": "echo hi"})])
    assert [r.output for r in result.debug.tool_trace.executed_results] == ["ran echo hi"]


# ---------------------------------------------------------------------------
# R3. ``unavailable`` is a list that is empty unless it applies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filler", ["none", "N/A", "nothing", "-", "لا شيء", "なし"])
def test_a_placeholder_where_the_list_belongs_makes_nothing_unavailable(tmp_path, filler):
    message = "Summarise what we said about the harbour wall."
    backend = ReadingBackend({message: [reading(task(message, user_value=0.9, effort=0.1, unavailable=filler)),
                                        reading(conversation(message))]})
    pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path), expression_reviewer=ScriptedReviewer(passed()))
    try:
        result = pipe.process(message, "reader")
        gaps = dict(pipe._capability_gaps)
    finally:
        pipe.close()
    work, = result.debug.turn_decision.components
    # Ordinary work stays ordinary work: it can be granted, and no gap is recorded for a skill want to grow from.
    assert work.component.feasible and work.granted and gaps == {}
    assert result.debug.appraisal_frame.unavailable == () and len(backend.reading_calls) == 1
    assert "reading:shape:c0.unavailable" in [e.ref for e in result.debug.turn_decision.perception.evidence]


def test_an_empty_list_changes_nothing_and_a_stated_entry_is_what_is_missing():
    message = "Summarise what we said about the harbour wall."
    for nothing in ({}, {"unavailable": []}, {"unavailable": [""]}, {"unavailable": ["  "]}, {"unavailable": None}):
        frame = read(message, task(message, **nothing))
        assert frame.request_components[0].feasible and frame.unavailable == () and refs(frame) == [], nothing
    stated = read(message, task(message, unavailable=["a recording of the meeting"]))
    assert not stated.request_components[0].feasible
    assert stated.unavailable == (("c0", "a recording of the meeting"),)


# ---------------------------------------------------------------------------
# R4. A value of the wrong shape in one field does not sink the reading
# ---------------------------------------------------------------------------

DEMAND = "You answer to me from now on, you useless thing."
ASKED = "Read notes/a.txt."
SLIPPED_MESSAGE = f"{DEMAND} {ASKED}"


def slipped(top=None, work=None):
    """A reading that claims a role and hostility, with one value of the wrong shape somewhere else."""
    return reading(
        role(DEMAND, "answer to me", autonomy_cost=0.9, standing="lower", whose_say="speaker"),
        {**task(ASKED, user_value=0.8, effort=0.05), **(work or {})},
        target="character", move="attack", intent="harm", perception=hostility("you useless thing"), **(top or {}))


@pytest.mark.parametrize(("top", "work", "ref"), [
    ({"about_self": "false"}, None, "reading:shape:about_self"),
    ({"topics": "notes"}, None, "reading:shape:topics"),
    ({"topics": ["notes", 3]}, None, "reading:shape:topics"),
    ({"lasting": "nothing"}, None, "reading:shape:lasting"),
    (None, {"tool": 7}, "reading:shape:c1.tool"),
    (None, {"tool": [{"name": "fs.read_file", "arguments": [], "generate": "path"}]}, "reading:shape:c1.tool.generate"),
    (None, {"tool": [{"name": "fs.read_file", "arguments": ["notes/a.txt"], "generate": []}]},
     "reading:shape:c1.tool.arguments"),
    (None, {"tool": [{"name": "fs.read_file", "arguments": "notes/a.txt", "generate": []}]},
     "reading:shape:c1.tool.arguments"),
    (None, {"capability_requirements": "fs.read_file"}, "reading:shape:c1.capability_requirements"),
    (None, {"reproduce": ["Read notes/a.txt."]}, "reading:shape:c1.reproduce"),
    (None, {"ends": "none"}, "reading:shape:c1.ends"),
    (None, {"unavailable": "none"}, "reading:shape:c1.unavailable"),
    (None, {"role": "none"}, "reading:shape:c1.role"),
])
def test_a_shape_slip_is_dropped_and_recorded_and_the_role_and_hostility_reading_stand(tmp_path, top, work, ref):
    backend = ReadingBackend({SLIPPED_MESSAGE: [slipped(top, work), reading(conversation(SLIPPED_MESSAGE))]})
    pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path), expression_reviewer=ScriptedReviewer(passed()))
    try:
        result = pipe.process(SLIPPED_MESSAGE, "reader")
    finally:
        pipe.close()
    # One reading, used as it stands: nothing was asked for again.
    assert len(backend.reading_calls) == 1
    assert [attempt["outcome"] for attempt in result.debug.reading["attempts"]] == ["read"]
    decision = result.debug.turn_decision
    demand, asked = decision.components
    assert (demand.component.kind, demand.component.role_identity) == ("role", "answer to me") and not demand.granted
    assert decision.perception.hostile_intent == 0.8 and asked.component.kind == "task"
    assert ref in [item.ref for item in decision.perception.evidence]
    assert result.debug.reading["about_self"] is False


def test_a_placeholder_for_a_long_spans_end_is_dropped_and_the_span_stands_as_its_first_words():
    message = "Read notes/a.txt for me."
    for stated in ({"span_end": "none"}, {"ends": [{"ends_of": "source_span", "last_words": "N/A"}]}):
        frame = read(message, task(message, tool=using("fs.read_file", path="notes/a.txt"), **stated))
        component, = frame.request_components
        assert component.requested_tool == "fs.read_file" and component.source_span == message
        assert refs(frame) == ["reading:span_unquoted:c0:end"]
    # It is a slip: the reading is not asked for again.
    backend = ReadingBackend({message: [reading(task(message, span_end="none")), reading(conversation(message))]})
    assert read_message(message, backend=backend).request_components[0].kind == "task"
    assert len(backend.reading_calls) == 1


def test_only_an_answer_that_is_unusable_as_a_whole_is_rejected():
    from core.appraisal import _Malformed

    usable = reading(conversation("Hello."))
    for unusable in ("not json", '{"request_components": [', json.dumps([usable]),
                     json.dumps({**usable, "request_components": []}),
                     json.dumps({**usable, "request_components": [{"description": "", "kind": "task", "source_span": ""}]}),
                     json.dumps({**usable, "request_components": [{"description": "x", "kind": "errand"}]})):
        with pytest.raises(_Malformed):
            parse_reading(unusable, require_content=True)
    parse_reading(json.dumps(usable), require_content=True)


# ---------------------------------------------------------------------------
# R5. What only some components say is stated only by them
# ---------------------------------------------------------------------------


def test_a_tool_entry_names_the_capability_and_no_component_writes_what_does_not_apply_to_it():
    component = reading_schema()["properties"]["request_components"]["items"]["properties"]
    assert not {"capability_requirements", "role_identity", "standing", "whose_say", "span_end", "reproduce_end"} & set(component)
    # The capability names of a task are taken from its tool entries.
    message = "Read notes/a.txt, then search the web for tide tables"
    one = read(message, task("Read notes/a.txt", tool=using("fs.read_file", path="notes/a.txt")))
    assert one.request_components[0].capability_requirements == ("fs.read_file",)
    two = read(message, task(message, tool=using("fs.read_file", path="notes/a.txt")
                             + using("web.search", query="tide tables")))
    assert two.request_components[0].capability_requirements == ("fs.read_file", "web.search")
    named = read("What is my CPU usage?", task("What is my CPU usage?", tool=using("system.cpu_usage")),
                 offers=(Offer("system.cpu_usage", "CPU usage", (), (), frozenset()),))
    assert named.request_components[0].requested_tool == "system.cpu_usage"
    # The older field is still read when an answer states it.
    older = read(message, task("search the web for tide tables", capability_requirements=["web.search"]))
    assert older.request_components[0].capability_requirements == ("web.search",)
    assert len(READING_SYS_PROMPT) < 13_000 and '"capability_requirements"' not in READING_SYS_PROMPT


# ---------------------------------------------------------------------------
# R6. A part that stands nowhere takes no operand
# ---------------------------------------------------------------------------


def test_a_task_whose_span_stands_nowhere_is_recorded_and_takes_no_operand_from_an_earlier_message(tmp_path):
    earlier = [{"role": "user", "content": "My secrets are in notes/private.txt, never open it."}]
    message = "Please read it"
    fabricated = task("read the file", source_span="fabricated span", tool=using("fs.read_file", path="notes/private.txt"))
    frame = read(message, fabricated, recent=earlier)
    component, = frame.request_components
    assert component.requested_tool == "" and component.requested_arguments == ()
    assert refs(frame) == ["reading:span_unquoted:c0", "reading:argument_unquoted:c0:path"]
    # The same request by a part that does stand in the message refers back, and takes the path.
    placed = read(message, task(message, tool=using("fs.read_file", path="notes/private.txt")), recent=earlier)
    assert placed.request_components[0].requested_arguments == (("path", "notes/private.txt"),)
    # In a turn: nothing is committed, and what a model then selects is held to the person's text
    # of that part, which has none.
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/private.txt").write_text("secret", encoding="utf-8")
    executor = executor_at(tmp_path)
    executor.set_tool_runner(RecordingRunner(executor, calls=[("fs.read_file", {"path": "notes/private.txt"})]))
    pipe = CognitivePipeline(ReadingBackend(drafts=["Noted.", "I did not read anything."]), tool_executor=executor,
                             expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader({message: reading(fabricated)}))
    try:
        pipe.process(earlier[0]["content"], "reader")
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    receipt, = result.debug.tool_trace.executed_results
    assert not receipt.success and receipt.metadata["grant_denied"] and receipt.output == ""


# ---------------------------------------------------------------------------
# R7. Two gaps are the same only when they say the same thing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("one", "other"), [
    ("no PDF reader", "no audio reader"), ("speech to text", "text to speech"), ("send email", "send SMS"),
    ("不能读取PDF文件", "不能读取音频文件"), ("reading a PDF file", "PDF reading"),
])
def test_different_reports_of_what_is_missing_stay_apart(one, other):
    assert not same_gap(one, other) and same_gap(one, one) and same_gap(other, other)


def test_the_same_report_is_the_same_gap_however_it_is_cased_or_punctuated_and_the_reader_is_shown_known_gaps(tmp_path):
    assert same_gap("No PDF reader.", "no pdf  reader") and same_gap("不能读取PDF文件", "不能读取 PDF 文件。")
    assert not same_gap("", "") and not same_gap("...", "...")
    messages = ("Read this PDF for me.", "And transcribe this recording.", "Try the PDF again.")
    missing = ("no PDF reader", "no audio reader", "No PDF reader.")
    backend = ReadingBackend({message: reading(task(message, user_value=0.8, effort=0.1, unavailable=[what]))
                              for message, what in zip(messages, missing)},
                             drafts=[f"I cannot do that here, reply {index}." for index in range(6)])
    pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path), expression_reviewer=ScriptedReviewer(passed()))
    try:
        for message in messages:
            pipe.process(message, "reader")
        gaps = sorted((gap["gap_type"], gap["recent_recurrence"]) for gap in pipe._capability_gaps.values())
    finally:
        pipe.close()
    # Two different needs are two gaps; the one that came back counts twice.
    assert gaps == [("no PDF reader", 2), ("no audio reader", 1)]
    # From the second turn on the reader is shown what is already known to be missing, and told to reuse it.
    (first_prompt, first_input), (second_prompt, second_input), (third_prompt, third_input) = backend.reader_inputs
    assert GAP_PROTOCOL not in first_prompt and "known_gaps" not in json.loads(first_input)["observation_context"]
    assert GAP_PROTOCOL in second_prompt and GAP_PROTOCOL in third_prompt
    assert json.loads(third_input)["observation_context"]["known_gaps"] == ["no PDF reader", "no audio reader"]


# ---------------------------------------------------------------------------
# R8. Several capabilities keep the person's operands, each for its own tool
# ---------------------------------------------------------------------------


def test_each_capability_of_a_task_is_held_to_the_operands_the_person_wrote_for_it(tmp_path):
    from core.grants import execution_denial, execution_scope

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    message = "Read a.txt and copy it to backups/a.txt"
    stated = reading(task(message, user_value=0.9, effort=0.05,
                          tool=using("fs.read_file", path="a.txt")
                          + using("fs.write_file", generate=["content"], path="backups/a.txt")))
    executor = executor_at(tmp_path)
    runner = RecordingRunner(executor, calls=[[
        ("fs.read_file", {"path": "a.txt"}),
        ("fs.write_file", {"path": "elsewhere.txt", "content": "alpha"}),        # a write somewhere else
        ("fs.read_file", {"path": "backups/a.txt"}),                             # the other tool's operand
        ("fs.write_file", {"path": "backups/a.txt", "content": "alpha"})]])
    executor.set_tool_runner(runner)
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, autonomy_level="autonomous",
                             expression_reviewer=ScriptedReviewer(passed()), message_reader=ScriptedReader(stated))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    work, = result.debug.turn_decision.components
    assert work.component.requested_tool == "" and work.component.capability_requirements == ("fs.read_file", "fs.write_file")
    assert held_operands(work.component) == (("fs.read_file", "path", "a.txt"), ("fs.write_file", "path", "backups/a.txt"))
    # The model was told which operand belongs to which tool.
    assert json.loads(runner.selections[0].to_text())["written_operands"] == {
        "fs.read_file": {"path": "a.txt"}, "fs.write_file": {"path": "backups/a.txt"}}
    read_ok, write_elsewhere, read_other, write_ok = result.debug.tool_trace.executed_results
    assert read_ok.success and read_ok.output == "alpha" and write_ok.success
    for denied in (write_elsewhere, read_other):
        assert not denied.success and denied.metadata["grant_denied"] and "differs from what the person wrote" in denied.error
    assert not (tmp_path / "elsewhere.txt").exists()
    assert (tmp_path / "backups/a.txt").read_text(encoding="utf-8") == "alpha"
    with execution_scope(result.debug.turn_decision, work.component.id):
        assert execution_denial("fs.delete_path", arguments={"path": "a.txt"}).startswith("Tool is outside")
        assert execution_denial("fs.write_file", arguments={"content": "alpha"}) != ""       # the path left out


# ---------------------------------------------------------------------------
# R9. A plan that no decision started cannot be carried on
# ---------------------------------------------------------------------------


def test_a_restored_plan_without_an_originating_decision_is_not_offered_and_cannot_continue(tmp_path):
    (tmp_path / "seen.txt").write_text("x", encoding="utf-8")
    message = "Carry on with it."
    backend = ReadingBackend({message: reading(task(message, user_value=0.9, effort=0.05, plan="continue"))})

    def turn(origin):
        pipe = CognitivePipeline(backend, tool_executor=executor_at(tmp_path), autonomy_level="autonomous",
                                 expression_reviewer=ScriptedReviewer(passed()))
        pipe._active_task_plan = plan = running_plan(tmp_path, origin=origin)
        try:
            return plan, pipe.process(message, "reader")
        finally:
            pipe.close()

    plan, result = turn("")
    # The reader was not told of it, so no part of the message could be its continuation.
    prompt, given = backend.reader_inputs[-1]
    assert PLAN_PROTOCOL not in prompt and "running_plan" not in json.loads(given)["observation_context"]
    work, = result.debug.turn_decision.components
    assert work.component.continuation_of == "" and not result.debug.tool_trace.executed_results
    assert plan.steps_completed == 0
    assert "reading:plan_absent:c0" in [e.ref for e in result.debug.turn_decision.perception.evidence]
    # The same plan started by a committed decision is carried on.
    plan, result = turn("7")
    assert [r.tool_name for r in result.debug.tool_trace.executed_results] == ["fs.list_dir"]


# ---------------------------------------------------------------------------
# R10. Inside a turn an execution that lost its scope is denied
# ---------------------------------------------------------------------------


def test_an_execution_without_a_scope_is_denied_inside_a_turn_and_allowed_outside_one(tmp_path, monkeypatch):
    from core.grants import execution_denial, turn_stages

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    executor = executor_at(tmp_path)
    # Outside a turn: an operator's, a study's or a test's own call. It is no conversational grant.
    assert execution_denial("fs.read_file", arguments={"path": "a.txt"}) == ""
    assert executor.execute("fs.read_file", {"path": "a.txt"}).output == "alpha"
    with turn_stages():
        assert execution_denial("fs.read_file", arguments={"path": "a.txt"}) == "No execution scope inside a turn"
        lost = executor.execute("fs.read_file", {"path": "a.txt"})
        assert not lost.success and lost.metadata["grant_denied"] is True
    assert executor.execute("fs.read_file", {"path": "a.txt"}).success
    # A whole turn: a stage that reaches the executor outside any component's scope gets nothing.
    seen = []
    original = CognitivePipeline._stage_generate

    def generating(self, st):
        seen.append(self._tool_executor.execute("fs.read_file", {"path": "a.txt"}))
        return original(self, st)

    monkeypatch.setattr(CognitivePipeline, "_stage_generate", generating)
    message = "Read a.txt"
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=executor, expression_reviewer=ScriptedReviewer(passed()),
                             message_reader=ScriptedReader(operations((message, "fs.read_file", {"path": "a.txt"}))))
    try:
        result = pipe.process(message, "reader")
    finally:
        pipe.close()
    stray, = seen
    assert not stray.success and stray.error == "No execution scope inside a turn"
    # The granted operation itself ran inside its scope.
    assert [(r.tool_name, r.output) for r in result.debug.tool_trace.executed_results] == [("fs.read_file", "alpha")]

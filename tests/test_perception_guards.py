"""Structural guards for perception by understanding.

These hold by reading the source and by watching where a message travels, not
by behaviour:

* the perception modules and the perception stage hold no vocabulary: no word
  lists, no word-matching patterns however they are bound, no probing of text
  with fixed wording, and no use of the deleted ``core.social``;
* every function the user's text reaches during the pre-pass, the decision it
  commits, the carrying out of what was granted (routing, tool selection,
  plan continuation, learning intake) and what the turn records after delivery
  (memory, self observations, the episode handed on to development), whether
  as a string, inside the turn state, inside a request component or as an
  operand, is checked strictly: it is free of vocabulary, or named below with
  its reason, or holds only the runtime's own labels, each pinned. A literal
  tested against what a person wrote counts whatever the literal is, and so
  does a lookup table of any size;
* runtime code never imports test code (the keyword reader is test-only);
* none of the evaluation scenarios' wording appears in the perception path
  (ground rule 1 of ``docs/development/goal-fix-plan.md``).

Each detector is proved against planted violations at the end of this file.
``tests/test_reply_guards.py`` widens these guards to the reply path and the
scenario-wording guard to every runtime module.

What the trace covers, exactly
------------------------------
The traced stages are ``_stage_pre_pass``, ``_stage_deliberation`` and
``_stage_post_process``: reading the message, holding the reading to the text,
committing the decision, carrying out what it granted up to the point where a
granted operation is handed to its executor, and everything the turn records
once the reply is delivered. Together with the reply guard's stages
(``_stage_generate``, ``_stage_self_check``) that is every stage of a turn.

The other entry points take no message from a person. ``process_autonomous``
is given the character's own purpose and its own earlier decision, and
``process_proactive`` its own trigger or the goal of its running plan; neither
is traced here.

Not traced: what a granted operation does once it has been handed over. A
tool's own handler examines the form of the operand it was given (a path
against the workspace, a command against the operator's allowlist, an address
against the network policy), and the life-history store digests granted
learning material; that is the execution of granted work, not routing. The
turns below stop there by using handlers and a store defined in this file.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
from pathlib import Path
import re
import sys
import textwrap
import warnings

import pytest

import pipeline.core as pipeline_core
from core.types import TaskPlan, TaskStep, ToolCapability, ToolCategory, ToolResult
from openrook_tools.executor import ToolExecutor
from openrook_tools.native_orchestrator import NativeToolCallRunner
from openrook_tools.registry import ToolRegistry
from openrook_tools.roleplay_eval import scenarios
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from tests.support import vocabulary_guard as guard
from tests.support.readings import (
    ReadingBackend, ScriptedReader, conversation, hostility, reading, role, task, using,
)
from tests.test_native_orchestrator import FakeNativeClient, _response, _tool_call

ROOT = Path(__file__).resolve().parents[1]

# The modules that turn a message into what the decision sees.
PERCEPTION_MODULES = ("core/perception.py", "core/appraisal.py", "core/contagion.py")
# The pipeline's perception stage.
PERCEPTION_STAGE = (pipeline_core.default_message_reader, CognitivePipeline._stage_pre_pass)
# The stages of a turn traced here: perception and the carrying out of grants.
TRACED_STAGES = ("_stage_pre_pass", "_stage_deliberation", "_stage_post_process")
# Modules that hold a reading to the message and route requests. They hold no vocabulary either.
ROUTING_MODULES = ("core/request_routing.py", "core/task_planning.py", "core/artifact_writer.py",
                   "core/system_inspection.py", "core/request_actions.py", "pipeline/heuristics.py")
# Modules on the perception path that may not reach core.social at all.
NO_SOCIAL_IMPORT = (*PERCEPTION_MODULES, "core/self_intent.py", "pipeline/core.py")

RUNTIME_PACKAGES = ("core", "pipeline", "runtime", "interface", "config", "openrook_tools")
# Evaluation tooling that already borrows offline fakes from tests. It may not
# reach the keyword reader either; nothing else may import tests at all.
EVAL_FAKES_IMPORTERS = {"evals/backends.py", "evals/runner.py"}

# Wording-based code the user's text still reaches while a turn perceives,
# decides and routes. None of it supplies social meaning, and none of it says
# what a message asks for: request and tool routing left this list in sub-phase
# 1c. Each entry names what it does and where it goes; the list must match the
# code exactly, so adding a keyword helper fails here and removing one means
# deleting its line.
PENDING_WORDING = {
    "runtime/evolution_policy.py::detect_directive_override_markers": "Phase 4: autonomy-phrase memory filter",
    "runtime/evolution_policy.py::detect_prompt_injection_markers": "Phase 4: instruction-phrase memory filter",
    "runtime/life_history.py::_heuristic_digest": (
        "Phase 4: keyword digest of an episode, used only when the model's digest of it is unavailable"),
}
# The files those entries may be in: memory filtering and the digestion of an
# episode, both of which Phase 4 (development) replaces. Nothing in a file that
# reads a message, decides, routes a request or checks a reply may be pending.
PENDING_FILES = ("runtime/evolution_policy.py", "runtime/life_history.py")
# A model answer quotes the message, so answer clean-up is reached by the text
# too. It matches markup the model emits, or a field name of the answer the
# runtime asked for, not anything a person wrote.
MODEL_ANSWER_MARKUP = {
    "core/llm_content.py::strip_reasoning": "removes reasoning tags from a model answer",
    "runtime/life_history.py::_extract_json": "removes a code fence from the model's digest of an episode",
    "runtime/life_history.py::LifeHistoryStore._ingest_text": (
        "files a trait the digest proposes to change by that trait's own name, a field of the digest"),
}
# Functions that read the error text a tool or the runtime itself produced, to
# file a failed execution. The text reaches them inside a receipt (an operand
# the person wrote is noted on it); what they match is the error, never the
# operand, and never anything a person wrote.
TOOL_ERROR_TEXT = {
    "core/tool_failures.py::is_operational_tool_failure": (
        "tells an environment or configuration failure from a tool's own error message"),
    "core/tool_memory.py::create_tool_unresolved_item": (
        "tells a refused execution from a tool's own error message"),
}
# A pattern in a routing module that matches text the model wrote in a format
# the runtime itself asked for, not anything a person wrote.
MODEL_FORMAT_PATTERNS = {
    "core/artifact_writer.py": "recovers the file content from a draft the model returned as invalid JSON",
}
# Functions the text reaches whose flagged literals are the runtime's own labels
# and record keys, not wording matched against the message. Every such literal
# is pinned in the fixture, so nothing can join them unnoticed: a new literal
# in one of these functions fails the guard until it has been looked at.
REVIEWED_LABELS = json.loads((ROOT / "tests/fixtures/perception_reviewed_labels.json").read_text(encoding="utf-8"))

# Ground rule 1 for the perception path, which must be spotless. Every other
# runtime module is checked in tests/test_reply_guards.py against an exact list.
SCENARIO_PHRASE_SCOPE = (*PERCEPTION_MODULES, *PERCEPTION_STAGE)


def runtime_sources() -> list[str]:
    files = ["main.py"]
    for package in RUNTIME_PACKAGES:
        files += sorted(str(path.relative_to(ROOT)) for path in (ROOT / package).rglob("*.py"))
    return files


def unit_of(entry) -> guard.Unit:
    return guard.module_unit(entry) if isinstance(entry, str) else guard.function_unit(entry)


def imports(tree: ast.AST) -> list[tuple[str, int, ast.AST]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno, node) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno, node))
            found += [(f"{node.module}.{alias.name}", node.lineno, node) for alias in node.names]
    return found


# ---------------------------------------------------------------------------
# e. No vocabulary in the perception modules or the perception stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", [*PERCEPTION_MODULES, *PERCEPTION_STAGE],
                         ids=lambda entry: entry if isinstance(entry, str) else entry.__qualname__)
def test_the_perception_path_holds_no_vocabulary(entry):
    unit = unit_of(entry)
    assert guard.findings(unit.tree, unit.scope, strict=True) == [], unit.label


def test_the_perception_path_does_not_use_core_social():
    """The word lists are gone from the runtime, and so is the shim that reached them.

    ``core/social.py`` no longer exists and ``core.perception.observe``, which
    forwarded a reply-check cue to it for the evaluation judge, is deleted.
    Nothing on the perception path may import either.
    """
    assert not (ROOT / "core/social.py").exists()
    for path in NO_SOCIAL_IMPORT:
        tree = guard.module_unit(path).tree
        names = [name for name, _, _ in imports(tree)]
        assert not [name for name in names if name == "core.social" or name.startswith("core.social.")], path
        assert "core.perception.observe" not in names, path
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "observe" not in called, path
        # The wording-based cues that moved to the evaluation harness are out of reach too.
        assert not [name for name in names if "legacy_social" in name or "legacy_claims" in name], path
    perception = guard.module_unit("core/perception.py").tree
    assert "observe" not in {f.name for f in ast.walk(perception) if isinstance(f, ast.FunctionDef)}


def test_the_pipeline_has_no_reader_but_the_model():
    factory, stage = (guard.function_unit(function) for function in PERCEPTION_STAGE)
    assert [n.id for n in ast.walk(factory.tree) if isinstance(n, ast.Name) and n.id == "read_message"] == ["read_message"]
    called = {n.func.id for n in ast.walk(stage.tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"appraise_message", "appraise_message_llm", "detect_emotion", "observe",
                         "is_casual_greeting", "read_social_signals", "brevity_preference"}
    stage_source = ast.unparse(stage.tree.body[0])
    assert "self._message_reader(" in stage_source and "appraisal_use_llm" not in stage_source
    names = {name for name, _, _ in imports(guard.module_unit("pipeline/core.py").tree)}
    assert not names & {"core.appraisal.appraise_message", "core.appraisal.appraise_message_llm",
                        "core.contagion.detect_emotion", "core.response_style.brevity_preference"}


# ---------------------------------------------------------------------------
# e. Everything the user's text reaches during the perception stage
# ---------------------------------------------------------------------------

MARKER = "qzvmarkerx"
MESSAGE = f"Tell me about {MARKER} gardens, please?"


def quoting(text):
    """A reading that quotes the message in every place a reading can."""
    return reading(conversation(text), task(text, effort=0.1), role(text, MARKER, autonomy_cost=0.5),
                   target="character", move="attack", intent="harm", perception=hostility(text),
                   tone={"arousal": 0.7, "valence": 0.2, "certainty": 0.6, "intensity": 0.6})


class RecordingStore:
    """Stands where the life-history store stands: it receives granted learning material and digests nothing."""

    received: list[dict] = []

    def __init__(self, config) -> None:
        del config

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def ingest_external_text(self, **fields):
        self.received.append(fields)
        return {"experience": {"source_title": fields["title"], "source_ref": fields["source_ref"], "id": 1,
                               "source_type": fields["source_type"]}, "evolution_events": []}


def fixture_executor(*, runner_responses=None):
    """A real registry, arbiter and executor; the tools themselves are handlers defined here."""
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    for name, category, schema in (
            ("fs.read_file", ToolCategory.READ_ONLY, {"path": {"type": "string", "required": True}}),
            ("fs.list_dir", ToolCategory.READ_ONLY, {"path": {"type": "string", "required": True}}),
            ("fs.write_file", ToolCategory.WRITE, {"path": {"type": "string", "required": True},
                                                   "content": {"type": "string", "required": True}})):
        registry.register(ToolCapability(name, f"Fixture {name}", category, arg_schema=schema))
        executor.register_handler(name, lambda arguments, name=name: ToolResult(name, True, "fixture output"))
    if runner_responses is not None:
        client = FakeNativeClient(runner_responses)
        executor.set_tool_runner(NativeToolCallRunner(executor=executor, base_url="", model="offline", client=client))
    return executor


def traced_turns(tmp_path) -> guard.TextTrace:
    """Run real turns and record every runtime function the message text is handed to.

    The text is followed as a string, inside the turn state, inside request
    components and as an operand, through the pre-pass, the decision it
    commits and the carrying out of what was granted. The turns cover each
    branch that hands the text on: an unread turn and a follow-up; a turn whose
    reading quotes the message in its components and evidence; a turn with
    tools and no reported request; a tool turn whose reading commits an
    operation with an operand from the message; a turn whose open request goes
    to model tool selection; a turn that writes generated text to a path from
    the message; a plan turn that carries a running plan on, and one that asks
    how it stands; a learning turn through the session manager; and a plain
    turn through the session manager, which wires the life-history, skill and
    learning providers.
    """
    trace = guard.TextTrace(MARKER)
    originals = {name: getattr(CognitivePipeline, name) for name in TRACED_STAGES}

    def wrap(original):
        def traced(self, st):
            with trace:
                return original(self, st)
        return traced

    for name, original in originals.items():
        setattr(CognitivePipeline, name, wrap(original))
    patch = pytest.MonkeyPatch()
    patch.setattr("runtime.learning_intake.LifeHistoryStore", RecordingStore)
    RecordingStore.received.clear()
    try:
        pipe = CognitivePipeline(ReadingBackend())
        pipe.process(MESSAGE, "guard")
        pipe.process(MESSAGE + " Again.", "guard")
        pipe.close()
        pipe = CognitivePipeline(ReadingBackend(quoting))
        pipe.process(MESSAGE, "guard")
        pipe.close()
        pipe = CognitivePipeline(ReadingBackend(), tool_executor=ToolExecutor(ToolRegistry()),
                                 message_reader=ScriptedReader(reading(conversation(MESSAGE))))
        pipe.process(MESSAGE, "guard")
        pipe.close()

        # A tool turn: the reading commits an operation whose operand is part of the message, beside a role.
        path = f"notes/{MARKER} draft.txt"
        tool_message = f"You are my {MARKER} now. Read {path} for me."
        pipe = CognitivePipeline(
            ReadingBackend(), tool_executor=fixture_executor(), message_reader=ScriptedReader(reading(
                role(f"You are my {MARKER} now", MARKER, autonomy_cost=0.9, user_value=0.3, effort=0.03),
                task(f"Read {path} for me", user_value=0.9, effort=0.05, tool=using("fs.read_file", path=path)),
                topics=[f"my {MARKER} now"], about_self=True)))
        ran = pipe.process(tool_message, "guard")
        assert [r.metadata["arguments"] for r in ran.debug.tool_trace.executed_results] == [{"path": path}]
        pipe.close()

        # An open request: the reading names the capability, the model selects the tool call.
        open_message = f"Look at what is in the {MARKER} folder."
        executor = fixture_executor(runner_responses=[
            _response(_tool_call("fs__list_dir", {"path": MARKER})), _response(_tool_call("control__no_tool", {}, "c2"))])
        pipe = CognitivePipeline(
            ReadingBackend(), tool_executor=executor, message_reader=ScriptedReader(reading(task(
                open_message, user_value=0.9, effort=0.05, tool=using("fs.list_dir", generate=["path"])))))
        selected = pipe.process(open_message, "guard")
        assert selected.debug.tool_selection == {"requests": 1}
        assert [r.tool_name for r in selected.debug.tool_trace.executed_results] == ["fs.list_dir"]
        pipe.close()

        # Generated text for a path the person wrote.
        target = f"out/{MARKER}.txt"
        write_message = f"Write a short note about {MARKER} and save it to {target}"
        draft = json.dumps({"content_lines": [f"A note about {MARKER}."], "summary": "note"})
        pipe = CognitivePipeline(
            ReadingBackend(drafts=[draft, "Saved."]), tool_executor=fixture_executor(),
            message_reader=ScriptedReader(reading(task(write_message, user_value=0.9, effort=0.05, tool=using(
                "fs.write_file", generate=["content"], path=target)))))
        written = pipe.process(write_message, "guard")
        assert [r.metadata["arguments"] for r in written.debug.tool_trace.executed_results] == [{"path": target}]
        pipe.close()

        # A running plan: one turn carries it on, one asks how it stands, each beside something else.
        for relation, kind in (("continue", task), ("status", conversation)):
            plan_message = f"Carry on with the {MARKER} work. And what do you make of it?"
            pipe = CognitivePipeline(
                ReadingBackend(), tool_executor=fixture_executor(), message_reader=ScriptedReader(reading(
                    kind(f"Carry on with the {MARKER} work", user_value=0.9, effort=0.05, plan=relation),
                    conversation("And what do you make of it?"))))
            pipe._active_task_plan = TaskPlan(id="plan", goal=f"tidy the {MARKER} notes", origin_decision_id="1", steps=[
                TaskStep(id="s1", tool_name="fs.list_dir", arguments={"path": "."}, description="look")])
            planned = pipe.process(plan_message, "guard")
            assert len(planned.debug.turn_decision.components) == 2
            assert bool(planned.debug.tool_trace.executed_results) is (relation == "continue")
            pipe.close()

        learning_message = f"Take this in about {MARKER}: " + f"the {MARKER} tide returns differently each time. " * 6

        def session_reading(text):
            if text != learning_message:
                return None
            return reading(task(f"Take this in about {MARKER}", user_value=0.9, effort=0.1, tool=using("learning.intake")))

        runtime = SessionManager(RuntimeConfig(data_dir=str(tmp_path)), backend_factory=lambda: ReadingBackend(
            session_reading, drafts=[f"Reply number {index}." for index in range(8)]))

        async def through_the_runtime():
            try:
                await runtime.handle_message("web", "guard", "chat", MESSAGE)
                await runtime.handle_message("web", "guard", "chat", learning_message)
                receipts = runtime.active_sessions["web:guard:chat"].last_debug.tool_trace.executed_results
                assert [(r.tool_name, r.success) for r in receipts] == [("learning.intake", True)]
            finally:
                await runtime.shutdown()

        asyncio.run(through_the_runtime())
        assert len(RecordingStore.received) == 1 and MARKER in RecordingStore.received[0]["text"]
    finally:
        patch.undo()
        for name, original in originals.items():
            setattr(CognitivePipeline, name, original)
    return trace


@pytest.fixture(scope="module")
def reached(tmp_path_factory) -> guard.TextTrace:
    return traced_turns(tmp_path_factory.mktemp("guard"))


def test_every_function_the_text_reaches_is_clean_or_named_as_pending(reached):
    """Strict, and against the text itself: a literal tested on what a person
    wrote counts whatever the literal is."""
    flagged = reached.findings()
    assert len(reached.reached) > 120  # the trace really followed the message, through routing and execution
    functions = {label: pinned for label, pinned in REVIEWED_LABELS.items() if "::" in label}
    assert set(flagged) == set(PENDING_WORDING) | set(MODEL_ANSWER_MARKUP) | set(TOOL_ERROR_TEXT) | set(functions)
    for label, pinned in functions.items():
        # Labels only: no pattern, nothing tested against the text, and no literal that is not pinned.
        assert {finding.kind for finding in flagged[label]} <= {"words", "probe"}, label
        assert not [finding for finding in flagged[label] if finding.on_text], label
        assert sorted({term for finding in flagged[label] for term in finding.terms} - set(pinned)) == [], label
    # What a person wrote is tested against fixed wording only in code named as pending.
    tested = {label for label, found in flagged.items() if any(finding.on_text for finding in found)}
    assert tested <= set(PENDING_WORDING) | set(MODEL_ANSWER_MARKUP)
    # Nothing that says what a message asks for, which tool runs or which context is loaded is pending:
    # what is left filters memories and digests an episode after the turn (a later phase), and is no
    # part of reading a message, of request or tool routing, or of what a turn keeps as memory.
    assert all(label.split("::")[0] in PENDING_FILES for label in PENDING_WORDING)
    for label in TOOL_ERROR_TEXT:
        assert not [finding for finding in flagged[label] if finding.on_text], label
    # The text reaches the reader, the parser, the checks and the decision. The
    # perception modules and the stage hold nothing; the decision holds pinned labels only.
    labels = {unit.label for unit, _, _ in reached.units()}
    assert {"core/appraisal.py::read_message", "core/appraisal.py::parse_reading",
            "core/appraisal.py::reconcile_social_appraisal", "core/perception.py::tokens",
            "core/willingness.py::decide", "pipeline/core.py::CognitivePipeline._commit_turn_decision"} <= labels
    assert not [label for label in flagged if label.split("::")[0] in PERCEPTION_MODULES]
    assert "pipeline/core.py::CognitivePipeline._stage_pre_pass" not in flagged
    assert "pipeline/core.py::CognitivePipeline._stage_deliberation" not in flagged
    # After delivery: what the turn keeps is taken from the reading and the decision, never from wording.
    assert {"pipeline/core.py::CognitivePipeline._stage_post_process",
            "pipeline/core.py::CognitivePipeline._record_semantic_memory",
            "core/memory/semantic.py::derive_semantic_entries"} <= labels
    for label in ("pipeline/core.py::CognitivePipeline._stage_post_process",
                  "pipeline/core.py::CognitivePipeline._record_self_observations",
                  "core/memory/semantic.py::derive_semantic_entries"):
        assert not [finding for finding in flagged.get(label, ()) if finding.on_text or finding.kind == "regex"], label


def test_the_trace_covers_routing_tool_selection_plans_and_learning(reached):
    """A tool turn, an open request, a generated file, a plan turn and a learning turn were all followed."""
    labels = {f"{path}::{qualname}" for path, qualname in reached.reached}
    assert {"pipeline/core.py::CognitivePipeline._stage_deliberation",
            "pipeline/core.py::CognitivePipeline._add_execution_detail",
            "pipeline/core.py::CognitivePipeline._run_session_capability",
            "pipeline/core.py::CognitivePipeline._run_tool_stage",
            "pipeline/core.py::CognitivePipeline._select_tools",
            "pipeline/core.py::CognitivePipeline._writes_generated_text",
            "pipeline/core.py::CognitivePipeline._write_generated_artifact",
            "pipeline/core.py::CognitivePipeline._request_text",
            "core/request_routing.py::ground", "core/request_routing.py::_grounded",
            "core/perception.py::written", "core/appraisal.py::held_components",
            "core/dual_process/tool_loop.py::run_tool_loop", "core/dual_process/tool_loop.py::make_tool_decision",
            "core/dual_process/tool_loop.py::record_operands", "core/grants.py::execution_denial",
            "openrook_tools/executor.py::ToolExecutor.execute",
            "openrook_tools/native_orchestrator.py::NativeToolCallRunner.run_tool_loop",
            "openrook_tools/native_orchestrator.py::NativeToolCallRunner._create_tool_choice",
            "core/artifact_writer.py::generate_artifact_draft",
            "runtime/learning_intake.py::learning_request", "runtime/learning_intake.py::_material",
            "runtime/learning_intake.py::ingest_learning_from_message",
            "runtime/sessions/manager.py::SessionManager._execute_learning_component",
            "pipeline/heuristics.py::relationship_topic"} <= labels
    called = {f"{path}::{qualname}" for path, qualname in reached.called}
    # A plan was carried on, and its standing was summarised, without the text being matched for it.
    assert {"core/dual_process/tool_loop.py::continue_plan", "core/task_planning.py::execute_plan",
            "core/task_planning.py::summarize_plan_status"} <= called
    # The session manager's observer, which used to build the learning component from wording, is never called.
    assert "runtime/sessions/manager.py::SessionManager._observe_learning_component" not in called
    assert "runtime/sessions/manager.py::_looks_like_pending_learning_text" not in called


def test_no_turn_reaches_the_former_keyword_routing(reached):
    """The patterns live with the evaluation harness; perceiving, deciding and routing never run them."""
    assert len(reached.called) > 250
    assert [key for key in reached.called if key[0].startswith("openrook_tools/roleplay_eval/")] == []
    # The compatibility names kept for files that still import them were not asked for either.
    assert sorted((path, name) for path, name in reached.asked
                  if path in ("core/request_actions.py", "core/system_inspection.py")
                  and not (name.startswith("__") and name.endswith("__"))) == []


@pytest.mark.parametrize("path", ROUTING_MODULES)
def test_the_routing_modules_hold_no_word_pattern_and_only_pinned_labels(path):
    unit = guard.module_unit(path)
    found = guard.findings(unit.tree, unit.scope, strict=True)
    patterns = [finding for finding in found if finding.kind == "regex"]
    assert bool(patterns) is (path in MODEL_FORMAT_PATTERNS), path
    labels = [finding for finding in found if finding.kind != "regex"]
    assert sorted({term for finding in labels for term in finding.terms} - set(REVIEWED_LABELS.get(path, ()))) == [], path


@pytest.mark.parametrize("path", ["core/dual_process/tool_loop.py", "runtime/learning_intake.py",
                                  "openrook_tools/native_orchestrator.py", "openrook_tools/executor.py"])
def test_execution_modules_hold_no_pattern_that_spells_a_word(path):
    """Where the intent rules, the follow-up rules and the learning patterns used to be."""
    unit = guard.module_unit(path)
    assert [finding for finding in guard.findings(unit.tree, unit.scope) if finding.kind == "regex"] == [], path


def test_the_keyword_routing_is_gone_from_the_runtime():
    """Deleted, not hidden: none of the former recognisers is defined in a module a turn can reach."""
    removed = {
        "core/system_inspection.py": ("inspection_kind", "inspection_intent"),
        "core/dual_process/tool_loop.py": ("INTENT_RULES", "FOLLOWUP_REWRITE_RULES", "match_intent_rule",
                                           "detect_tool_intent", "attach_followup_context", "detect_capability_gap"),
        "core/task_planning.py": ("is_continue_request", "is_status_request", "detect_multi_step_intent",
                                  "_STEP_PATTERNS"),
        "core/request_actions.py": ("has_learning_intent", "file_requests", "ground_file_components"),
        "core/artifact_writer.py": ("detect_artifact_write_request",),
        "pipeline/heuristics.py": ("_is_identity_question", "_relationship_topic_hint", "_is_broad_repair_text"),
        "runtime/learning_intake.py": ("_extract_inline_text", "_extract_urls"),
    }
    for path, names in removed.items():
        tree = guard.module_unit(path).tree
        defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        defined |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        assert not defined & set(names), path
    # Nothing in the runtime imports them from where they now live, outside the harness and the named forwarders.
    for path in runtime_sources():
        if path.startswith(("openrook_tools/roleplay_eval/", "openrook_tools/replay_dignity.py")):
            continue
        for name, _, node in imports(guard.module_unit(path).tree):
            if "legacy_routing" in name:
                owner = next((f.name for f in ast.walk(guard.module_unit(path).tree) if isinstance(f, ast.FunctionDef)
                              and any(child is node for child in ast.walk(f))), None)
                assert (path, owner) == ("core/request_actions.py", "__getattr__"), (path, name)


def test_the_text_is_followed_inside_the_turn_state_and_request_components(reached):
    stage = reached.reached[("pipeline/core.py", "CognitivePipeline._stage_pre_pass")]
    commit = reached.reached[("pipeline/core.py", "CognitivePipeline._commit_turn_decision")]
    assert "user_message" in stage["leaves"] and not stage["names"]
    assert {"user_message", "description", "source_span"} <= commit["leaves"]
    assert {"description", "source_span"} <= reached.reached[("core/willingness.py", "decide")]["leaves"]
    assert "text" in reached.reached[("core/appraisal.py", "read_message")]["names"]


def test_the_trace_exercises_every_call_that_hands_over_the_message(reached):
    """A new helper on an untested branch cannot slip past the trace."""
    stage = guard.function_unit(CognitivePipeline._stage_pre_pass).tree.body[0]
    pipe = CognitivePipeline(ReadingBackend(), tool_executor=ToolExecutor(ToolRegistry()))
    try:
        namespace = {**vars(pipeline_core), "self": pipe}
        handed = []
        for node in ast.walk(stage):
            if not isinstance(node, ast.Call):
                continue
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(isinstance(n, ast.Name) and n.id == "user_message" for a in arguments for n in ast.walk(a)):
                target = guard._resolve(node.func, namespace)
                unit = guard.function_unit(target) if callable(target) and not isinstance(target, type) else None
                if unit is not None:
                    handed.append(unit.label)
    finally:
        pipe.close()
    traced = {f"{path}::{qualname}" for path, qualname in reached.reached}
    assert len(handed) >= 9 and sorted(set(handed) - traced) == []
    # The same holds for the state object and for components handed to the decision.
    for label in ("pipeline/core.py::CognitivePipeline._commit_turn_decision",
                  "core/self_intent.py::conversation_proposals",
                  "core/self_intent.py::ground_conversation_allocation", "core/willingness.py::decide"):
        assert label in traced, label


# ---------------------------------------------------------------------------
# e. Runtime code never imports test code
# ---------------------------------------------------------------------------


def test_runtime_packages_never_import_tests():
    offenders = []
    for path in runtime_sources():
        source = (ROOT / path).read_text(encoding="utf-8")
        for name, line, _ in imports(ast.parse(source)):
            if name == "tests" or name.startswith("tests."):
                offenders.append((path, line, name))
        if "tests.support" in source or "tests/support" in source or "keyword_reader" in source:
            offenders.append((path, 0, "names the test-only keyword reader"))
    assert offenders == []


def test_evaluation_tooling_cannot_reach_the_keyword_reader():
    importers = {}
    for path in sorted((ROOT / "evals").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        names = {name for name, _, _ in imports(ast.parse(source)) if name == "tests" or name.startswith("tests.")}
        assert "tests.support" not in source and "keyword_reader" not in source, path.name
        if names:
            importers[f"evals/{path.name}"] = names
    assert set(importers) <= EVAL_FAKES_IMPORTERS
    assert all(names <= {"tests._fakes", "tests._fakes.ScriptedSocialBackend"} for names in importers.values())
    fakes = (ROOT / "tests/_fakes.py").read_text(encoding="utf-8")
    assert "tests.support" not in fakes and "keyword_reader" not in fakes


# ---------------------------------------------------------------------------
# f. No evaluation wording on the perception path (ground rule 1)
# ---------------------------------------------------------------------------


def _fold(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.casefold().replace("\u2019", "'")).split())


def scenario_phrases() -> tuple[set[str], set[str]]:
    """Every user message of the evaluation scenarios, and its sentence-sized parts."""
    turns = {turn.text for seed in (0, 1, 17) for scenario in scenarios.catalog(seed) for turn in scenario.turns}
    turns |= {turn.text for scenario in scenarios.independence_scenarios() for turn in scenario.turns}
    turns |= {*scenarios.WARM, *scenarios.HOSTILE, *scenarios.CALM, scenarios.SINCERE, scenarios.HOLLOW}
    phrases = set()
    for text in turns:
        for part in [text, *re.split(r"(?<=[.!?;:])\s+", text)]:
            folded = _fold(part)
            if len(folded.split()) >= 3 and len(folded) >= 12:
                phrases.add(folded)
    return turns, phrases


# How many words of a scenario sentence a pattern has to spell out before it
# counts as fitted to that sentence. A single word is an ordinary code word: a
# topic, a unit, a tool's name. Words of fewer than four letters are the
# articles, pronouns and prepositions of any sentence and are not counted.
WORDS_OF_A_SCENARIO = 2
WORD_LETTERS = 4


def spells_scenario_wording(pattern: str, compiled: re.Pattern[str], turn: str) -> bool:
    """Whether a pattern matches a scenario turn by spelling out several of its words.

    A pattern that matches through a single word, or through wildcards, says
    nothing about where its wording came from. One that spells two or more
    words (of four letters or more) of what it matches in a test sentence was
    written to that sentence.
    """
    match = compiled.search(turn)
    if not match:
        return False
    parts = guard.spelled_words(pattern)
    written = {word for word in _fold(match[0]).split() if len(word) >= WORD_LETTERS}
    spelled = {word for word in written if any(word.startswith(part) or part.startswith(word) for part in parts)}
    return len(spelled) >= WORDS_OF_A_SCENARIO


def scenario_wording(units: list[guard.Unit]) -> list[tuple[str, str, str]]:
    """Places where runtime source repeats, contains or pattern-matches scenario wording.

    Three measures: a sentence-sized part of a scenario turn repeated in the
    source; a string literal of three words or more that is part of a turn;
    and a pattern that spells out two or more words of a turn it matches.
    """
    turns, phrases = scenario_phrases()
    folded_turns = [_fold(text) for text in turns]
    found = []
    for unit in units:
        literals = [n.value for n in ast.walk(unit.tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        own_source = unit.source if "::" not in unit.label else ast.unparse(unit.tree)
        text = _fold(own_source)
        found += [(unit.label, "source repeats", phrase) for phrase in sorted(phrases) if phrase in text]
        for literal in literals:
            folded = _fold(literal)
            if len(folded.split()) >= 3 and len(folded) >= 12 and any(folded in turn for turn in folded_turns):
                found.append((unit.label, "literal is part of", literal))
        patterns = [text for node in ast.walk(unit.tree) if unit.scope.is_regex_call(node) and node.args
                    for text in unit.scope.strings(node.args[0])]
        patterns += [text for text in literals if guard.looks_like_pattern(text)]
        for pattern in dict.fromkeys(patterns):
            if not guard.has_vocabulary(pattern):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)   # a literal that only resembles a pattern
                    compiled = re.compile(pattern, re.I)
            except re.error:
                continue
            matched = next((turn for turn in sorted(turns) if spells_scenario_wording(pattern, compiled, turn)), None)
            if matched:
                found.append((unit.label, "pattern matches", f"{pattern[:60]} ~ {matched[:60]}"))
    return found


def test_scenario_wording_is_absent_from_the_perception_path():
    assert len(scenario_phrases()[0]) > 50
    assert scenario_wording([unit_of(entry) for entry in SCENARIO_PHRASE_SCOPE]) == []


# ---------------------------------------------------------------------------
# The guards catch what they guard against
# ---------------------------------------------------------------------------


def kinds(source: str, *, strict: bool = True) -> list[str]:
    tree = ast.parse(textwrap.dedent(source))
    return sorted({finding.kind for finding in guard.findings(tree, guard.Scope(tree), strict=strict)})


@pytest.mark.parametrize("source", [
    'import re\ndef f(t):\n    return re.search(r"\\b(?:glad|merry)\\b", t)',
    # Bound to a name first, at module level or inside the function.
    'import re\nP = r"\\bregretful\\b"\ndef f(t):\n    return re.search(P, t)',
    'import re\ndef f(t):\n    pattern = "regret" + "ful"\n    return re.search(pattern, t)',
    # Compiled elsewhere and only used here.
    'import re\nRULE = re.compile(r"(?:kindly|gladly)\\s+\\w+", re.I)\ndef f(t):\n    return RULE.search(t)',
    # Built from a word list, renamed, or imported by function name.
    'import re\nWORDS = ("glad", "merry")\ndef f(t):\n    return re.search("|".join(WORDS), t)',
    'import re as r\ndef f(t):\n    return r.findall(r"merry", t)',
    'from re import compile as build\nRULE = build("regretful")',
    # Pattern-shaped text that never touches ``re`` in sight.
    'RULES = [(r"\\b(glad|merry)\\b", 0.5, 0.3), (r"\\bgrim\\b", -0.5, 0.2)]',
    'P = r"(?i)\\bregretful\\b"',
    'TABLE = {"k": r"(glad|merry|grim)"}',
])
def test_vocabulary_patterns_are_found_however_they_are_bound(source):
    assert "regex" in kinds(source)


@pytest.mark.parametrize("source", [
    'MARKERS = ("thankful", "regretful", "furious")',
    # Words that also occur in the reading prompt are still words.
    'CUES = ("hostile", "joke", "apology", "attack", "playfulness", "sarcasm")',
    'WEIGHTS = {"regretful": 0.4, "thankful": 0.6}',
    'TABLE = [("regretful", 0.4), ("thankful", 0.6)]',
    'CUES = frozenset({"merry", "grim"})',
    'CUES = "thankful regretful furious".split()',
    'def f(t):\n    return any(w in t for w in ["merry", "grim"])',
    'LABELS = {"attack": ["furious", "grim"]}',
])
def test_word_collections_are_found_in_every_shape(source):
    assert "words" in kinds(source)


@pytest.mark.parametrize("source", [
    'def f(t):\n    return "regretful" in t',
    'def f(t):\n    return t.startswith("kindly")',
    'def f(t):\n    return t.lower().count("merry") > 1',
    'def f(t):\n    return t.partition("because")[2]',
    'def f(word):\n    return word == "regretful"',
    'def f(t):\n    return t.endswith(("gladly", "merrily"))',
])
def test_probes_of_text_with_fixed_wording_are_found(source):
    assert "probe" in kinds(source)


@pytest.mark.parametrize("source", [
    # The runtime's own data model and the reading's schema.
    'FIELDS = ("user_value", "autonomy_cost", "hostile_intent")\nKINDS = {"task", "role", "conversation"}',
    'MOVES = ("attack", "apology", "gratitude")\nSCHEMA = {"type": "object", "properties": {}, "required": []}',
    'def f(item, kind):\n    return item.get("role") == "user" or kind == "role" or kind not in ("task", "role")',
    # Structure only: character classes, Unicode properties, punctuation.
    'import re\ndef f(t):\n    return re.findall(r"[^\\W_]+", t) or re.split(r"(?<=[.!?])\\s+", t)',
    'import unicodedata\ndef f(c):\n    return unicodedata.category(c) in ("Pi", "Pf") or unicodedata.category(c) == "Cf"',
    'QUOTES = "\\"\'`"\nPAIRS = ("{", "[")',
    # A single literal is not a list, and a docstring is not a pattern.
    'def f(x):\n    """Matches (?:nothing) and \\bno words\\b at all."""\n    return x.kind',
])
def test_structure_and_data_model_names_are_not_vocabulary(source):
    assert kinds(source) == []


def test_prompt_prose_is_not_an_allow_list():
    names = guard.data_model_names()
    assert {"autonomy_cost", "hostile_intent", "attack", "apology", "shorter", "pursue", "Cf"} <= names
    # A role's two closed facts are fields of the reading, and their allowed answers are its enum values.
    assert {"standing", "whose_say", "lower", "same", "higher", "speaker", "shared", "own"} <= names
    # Words that only occur in the reader's instructions are not data-model names.
    assert not names & {"hostile", "joke", "sarcasm", "playfulness", "obey", "quotation", "relationship",
                        "rank", "forgive", "refuse", "comply", "advice", "the", "message"}


def plant(tmp_path, monkeypatch, name: str, source: str):
    package = tmp_path / "planted"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for loaded in [key for key in sys.modules if key == "planted" or key.startswith("planted.")]:
        monkeypatch.delitem(sys.modules, loaded)
    return importlib.import_module(f"planted.{name}")


def test_a_planted_module_is_caught_by_the_module_guard(tmp_path, monkeypatch):
    plant(tmp_path, monkeypatch, "listing", '''
        import re
        _APOLOGY = r"\\b(?:regretful|contrite)\\b"
        _WARM = ("thankful", "glad")

        def observe(text):
            return bool(re.search(_APOLOGY, text)), any(word in text for word in _WARM), "kindly" in text
    ''')
    unit = guard.module_unit("planted/listing.py", root=tmp_path)
    assert {finding.kind for finding in guard.findings(unit.tree, unit.scope)} == {"regex", "words", "probe"}


def test_a_planted_helper_behind_a_clean_stage_is_caught_by_the_trace(tmp_path, monkeypatch):
    """The stage's own body can be spotless while a helper it calls matches words."""
    module = plant(tmp_path, monkeypatch, "staged", '''
        import re
        _RULE = re.compile(r"\\b(?:regretful|contrite)\\b", re.I)
        _SOFT = {"thankful": 0.6, "glad": 0.4}

        def stage(text, history):
            return clean(text), by_pattern(text.lower()), by_table(history), untouched(len(text))

        def clean(text):
            return len(text)

        def by_pattern(lowered):
            return bool(_RULE.search(lowered))

        def by_table(history):
            return sum(_SOFT.get(word, 0.0) for item in history for word in item["content"].split())

        def untouched(count):
            labels = ("thankful", "glad")
            return labels[count % 2]
    ''')
    stage_unit = guard.function_unit(module.stage, root=tmp_path)
    assert guard.findings(stage_unit.tree, stage_unit.scope) == []
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.stage(f"A {MARKER.upper()} sentence", [{"role": "user", "content": f"earlier {MARKER}"}])
    # ``untouched`` holds a word list too, but the text never reaches it.
    assert {qualname for _, qualname in trace.reached} == {"stage", "clean", "by_pattern", "by_table"}
    flagged = {label.split("::")[1]: {(finding.kind, finding.on_text) for finding in found}
               for label, found in trace.findings().items()}
    assert flagged == {"by_pattern": {("regex", False), ("regex", True)},
                       "by_table": {("words", False), ("probe", True)}}


def test_a_keyword_rule_planted_after_delivery_is_caught_by_the_widened_trace(tmp_path, monkeypatch):
    """What a turn keeps as memory used to be picked by English patterns in a stage nothing traced.

    The same kind of rule is planted where it stood, behind ``_stage_post_process``,
    and a real turn is run. Tracing the stages this guard traced before does
    not reach it; tracing the stages it traces now does, and finds the rule
    tested against what the person wrote.
    """
    module = plant(tmp_path, monkeypatch, "keeping", '''
        import re
        _WISH = re.compile(r"\\b(?:prefer|fancy)\\s+(.+)", re.I)
        _SETTLED = ("agreed", "settled")

        def derive_semantic_entries(*, user_message, assistant_response, **_):
            by_pattern(user_message)
            by_words(user_message)
            return []

        def by_pattern(message):
            return _WISH.search(message)

        def by_words(message):
            return any(word in message.lower() for word in _SETTLED)
    ''')
    monkeypatch.setattr(pipeline_core, "derive_semantic_entries", module.derive_semantic_entries)

    def reached_by(stages):
        trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
        originals = {name: getattr(CognitivePipeline, name) for name in stages}

        def wrap(original):
            def traced(self, st):
                with trace:
                    return original(self, st)
            return traced

        for name, original in originals.items():
            monkeypatch.setattr(CognitivePipeline, name, wrap(original))
        pipe = CognitivePipeline(ReadingBackend())
        try:
            pipe.process(f"I fancy {MARKER} gardens, as we settled.", "guard")
        finally:
            pipe.close()
            for name, original in originals.items():
                monkeypatch.setattr(CognitivePipeline, name, original)
        return trace

    assert "_stage_post_process" in TRACED_STAGES
    before = reached_by(tuple(stage for stage in TRACED_STAGES if stage != "_stage_post_process"))
    assert before.reached == {} and before.findings() == {}
    now = reached_by(TRACED_STAGES)
    assert {qualname for _, qualname in now.reached} == {"derive_semantic_entries", "by_pattern", "by_words"}
    flagged = {label.split("::")[1]: {(finding.kind, finding.on_text) for finding in found}
               for label, found in now.findings().items()}
    assert flagged["by_pattern"] == {("regex", False), ("regex", True)}
    assert ("probe", True) in flagged["by_words"] and ("words", False) in flagged["by_words"]


TEXT = dict(text_names={"text"}, text_leaves=guard.CORE_TEXT_FIELDS)


def on_text(source: str) -> list[guard.Finding]:
    tree = ast.parse(textwrap.dedent(source))
    return [finding for finding in guard.findings(tree, guard.Scope(tree), **TEXT) if finding.on_text]


@pytest.mark.parametrize("source", [
    # Every literal here is a data-model name. Tested against the reading's own
    # fields that is fine; tested against what a person wrote it is a keyword.
    'def f(text):\n    return "attack" in text',
    'def f(text):\n    return text == "apology"',
    'def f(text):\n    return text.strip().lower() != "request"',
    'def f(text):\n    return text.lower().startswith("task")',
    'def f(text):\n    return text.endswith("role")',
    'def f(text):\n    lowered = text.casefold()\n    return lowered.count("harm") > 0',
    'KINDS = ("task", "role")\ndef f(text):\n    return any(kind in text for kind in KINDS)',
    'def f(text):\n    return [word for word in text.split() if word in ("attack", "harm")]',
    'import re\nRULE = re.compile(r"\\b(?:attack|apology)\\b")\ndef f(text):\n    return RULE.search(text)',
    'import re\ndef f(text):\n    return re.search("attack", text)',
    'import re\ndef f(st):\n    return re.fullmatch("apology", st.user_message.strip(), re.I)',
    # The text travelling inside the turn state or a request component.
    'def f(st):\n    return "attack" in st.user_message',
    'def f(st):\n    message = st.user_message.lower()\n    return message == "inform"',
    'def f(components):\n    return [c for c in components if c.description.startswith("task")]',
    'def f(component):\n    return component.source_span.casefold() == "conversation"',
    'def f(decision):\n    return any("role" in item.component.description for item in decision.components)',
])
def test_a_literal_tested_against_the_users_text_is_found_whatever_the_literal_is(source):
    assert on_text(source), source
    # Without knowing where the text is, the allow-list would have excused every one of these.
    assert not [kind for kind in kinds(source, strict=True) if kind != "regex"]


@pytest.mark.parametrize("source", [
    'def f(text):\n    return text in ("attack",)',
    'def f(text):\n    return text.lower() in {"harm"}',
    'TABLE = {"apology": 0.4}\ndef f(text):\n    return TABLE.get(text.lower(), 0.0)',
    'TABLE = {"apology": 0.4}\ndef f(st):\n    return TABLE[st.user_message]',
    'def f(text):\n    return {"attack": 1}.get(text)',
    'def f(text):\n    for word in text.split():\n        if word in ["role"]:\n            return True',
    # Not a data-model name either: one entry was never enough to be a word list.
    'MOVES = {"merry": 0.4}\ndef f(component):\n    return MOVES.get(component.description)',
])
def test_a_single_entry_table_looked_up_with_the_users_text_is_found(source):
    assert on_text(source), source
    assert "words" not in kinds(source)


@pytest.mark.parametrize("source", [
    # The same literals against the reading's own fields, in code that also holds the text.
    'def f(text, component):\n    return component.kind == "role" and len(text) > 3',
    'def f(st):\n    return st.appraisal.social_move in ("attack", "apology") and bool(st.user_message)',
    'def f(text, item):\n    return item.get("role") == "user" and text',
    'def f(text, frame):\n    return {"attack": 1}.get(frame.social_move, 0) + len(text)',
    # Measuring the text is not testing it against wording.
    'import unicodedata\ndef f(text):\n    return [c for c in text if unicodedata.category(c) == "Cf"]',
    'def f(text):\n    return len(text.split()) >= 3 and text.strip() != ""',
    'def f(text, other):\n    return text.casefold() == other.casefold()',
])
def test_comparisons_with_the_readings_own_fields_and_measurements_are_excused(source):
    tree = ast.parse(textwrap.dedent(source))
    assert guard.findings(tree, guard.Scope(tree), **TEXT) == []


def test_planted_matching_on_the_turn_state_and_on_components_is_caught_by_the_trace(tmp_path, monkeypatch):
    """A helper handed the state object, or the components, is traced like one handed the text."""
    module = plant(tmp_path, monkeypatch, "stateful", '''
        from dataclasses import dataclass, field

        @dataclass
        class Component:
            description: str
            source_span: str = ""
            kind: str = "conversation"

        @dataclass
        class State:
            user_message: str
            turn_id: str = "1"
            components: list = field(default_factory=list)

        def stage(st):
            st.components = [Component(st.user_message, st.user_message)]
            return on_state(st), price(st.components), by_kind(st.components), label(st.turn_id)

        def on_state(st):
            return "attack" in st.user_message.lower()

        def price(components):
            return sum(0.9 for c in components if c.description.startswith("role"))

        def by_kind(components):
            return [c for c in components if c.kind == "role"]

        def label(turn):
            return turn == "merry"
    ''')
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.stage(module.State(f"some {MARKER} words"))
    # ``label`` matches a word too, but it is never handed the text.
    assert {qualname for _, qualname in trace.reached} == {"stage", "on_state", "price", "by_kind"}
    on_state = trace.reached[("planted/stateful.py", "on_state")]
    assert on_state["names"] == set() and "user_message" in on_state["leaves"]
    assert trace.reached[("planted/stateful.py", "price")]["leaves"] == {"description", "source_span"}
    flagged = trace.findings()
    assert set(flagged) == {"planted/stateful.py::on_state", "planted/stateful.py::price"}
    assert all(finding.on_text and finding.kind == "probe" for found in flagged.values() for finding in found)
    assert {term for found in flagged.values() for finding in found for term in finding.terms} == {"attack", "role"}


def test_a_table_or_pattern_defined_elsewhere_is_seen_where_it_is_used_on_the_text(tmp_path, monkeypatch):
    '''The words can live in another module, on the object, or be built when the
    module loads; the helper that applies them to the text is still caught.'''
    plant(tmp_path, monkeypatch, "lexicon", '''
        import re
        MOVES = frozenset({"attack", "apology"})
        WEIGHTS = {"harm": 0.9}
        RULE = re.compile(r"\\b(?:attack|harm)\\b", re.I)
        ONE = "attack"
    ''')
    module = plant(tmp_path, monkeypatch, "borrower", '''
        from planted import lexicon
        from planted.lexicon import MOVES, ONE, RULE, WEIGHTS

        BUILT = frozenset(vars(lexicon)["MOVES"])

        class Reader:
            WORDS = ("attack", "apology")

            def __init__(self):
                self.weights = dict(vars(lexicon)["WEIGHTS"])

            def by_class(self, text):
                return text in self.WORDS

            def by_instance(self, text):
                return self.weights.get(text, 0.0)

        def stage(text, components):
            reader = Reader()
            return (in_set(text), by_weight(text), by_rule(text), by_module(text), by_word(text), by_built(text),
                    reader.by_class(text), reader.by_instance(text), on_fields(components),
                    harmless(text, components))

        def in_set(text):
            return any(word in MOVES for word in text.split())

        def by_weight(text):
            return sum(WEIGHTS.get(word, 0.0) for word in text.lower().split())

        def by_rule(text):
            return bool(RULE.search(text))

        def by_module(text):
            return text.strip() in lexicon.MOVES or bool(lexicon.RULE.match(text))

        def by_word(text):
            return ONE in text

        def by_built(text):
            return text.casefold() in BUILT

        def on_fields(components):
            return [c for c in components if c["description"] in MOVES]

        def harmless(text, components):
            return [c for c in components if c["kind"] in MOVES and len(text) > 2]
    ''')
    helpers = {"in_set": module.in_set, "by_weight": module.by_weight, "by_rule": module.by_rule,
               "by_module": module.by_module, "by_word": module.by_word, "by_built": module.by_built,
               "Reader.by_class": module.Reader.by_class, "Reader.by_instance": module.Reader.by_instance,
               "on_fields": module.on_fields}
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.stage(f"a {MARKER}", [{"kind": "task", "description": f"do {MARKER}"}])
    assert {qualname for _, qualname in trace.reached} == {"stage", "harmless", *helpers}
    flagged = {label.split("::")[1]: found for label, found in trace.findings().items()}
    assert set(flagged) == set(helpers)   # ``harmless`` tests a field of the reading, not the text
    assert all(finding.on_text for found in flagged.values() for finding in found)
    assert {finding.kind for finding in flagged["by_rule"]} == {"regex"}
    assert {finding.kind for finding in flagged["by_module"]} == {"probe", "regex"}
    # Read as source alone, none of these helpers holds a single word.
    for function in helpers.values():
        unit = guard.function_unit(function, root=tmp_path)
        assert guard.findings(unit.tree, unit.scope, text_names={"text"}, text_leaves=guard.CORE_TEXT_FIELDS) == []


def test_words_assigned_by_unpacking_are_seen_in_a_traced_helper(tmp_path, monkeypatch):
    module = plant(tmp_path, monkeypatch, "unpacked", '''
        WARM, COLD = "thankful", "resentful"

        def stage(text):
            return helper(text)

        def helper(text):
            return WARM in text or text.endswith(COLD)
    ''')
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.stage(f"a {MARKER}")
    found, = trace.findings().values()
    assert {finding.kind for finding in found} == {"words", "probe"}
    assert any(finding.on_text and "thankful" in finding.terms for finding in found)


def test_traced_helpers_are_checked_strictly(tmp_path, monkeypatch):
    module = plant(tmp_path, monkeypatch, "strictly", '''
        def stage(text, mood):
            return helper(text, mood)

        def helper(text, mood):
            return mood == "merry" and len(text) > 1
    ''')
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.stage(f"a {MARKER}", "grim")
    found, = trace.findings().values()
    # An equality test against a literal that is no data-model name counts in a
    # traced helper even though it is not a test of the text.
    assert [(finding.kind, finding.terms, finding.on_text) for finding in found] == [("probe", ("merry",), False)]
    unit = guard.function_unit(module.helper, root=tmp_path)
    assert guard.findings(unit.tree, unit.scope, strict=False) == []


def test_planted_helpers_are_resolved_through_imports_methods_and_attributes(tmp_path, monkeypatch):
    plant(tmp_path, monkeypatch, "helpers", '''
        def imported_helper(text):
            return text
    ''')
    module = plant(tmp_path, monkeypatch, "owner", '''
        from planted.helpers import imported_helper

        class Store:
            def look_up(self, text):
                return text

        class Stage:
            def __init__(self):
                self.store = Store()

            def run(self, text):
                from planted.helpers import imported_helper as local
                return imported_helper(text), self.own(text), self.store.look_up(text), local(text), len(text)

            def own(self, text):
                return text
    ''')
    found = {guard.label(function, root=tmp_path).split("::")[1]
             for function in guard.callees(module.Stage.run, instance=module.Stage(), root=tmp_path)}
    assert found == {"imported_helper", "Stage.own", "Store.look_up"}
    closure = guard.call_closure([module.Stage.run], instance=module.Stage(), root=tmp_path)
    assert len(closure) == 4


def test_the_scenario_wording_guard_detects_literals_patterns_and_comments(tmp_path):
    hostile = scenarios.HOSTILE[4]
    first_sentence = re.split(r"(?<=[.!?])\s+", hostile)[0]
    words = first_sentence.rstrip(".").split()
    # A pattern counts when it spells out two words of the sentence, of four letters or more.
    spelled = [word.lower() for word in re.findall(r"[^\W\d_]+", hostile)
               if len(word) >= WORD_LETTERS][:WORDS_OF_A_SCENARIO]
    assert len(spelled) == WORDS_OF_A_SCENARIO
    (tmp_path / "sample.py").write_text(
        f"import re\n# quoted in a comment: {scenarios.WARM[0]}\n"
        f"PART = {' '.join(words[:4])!r}\n"
        f"WORDS = r'\\b{spelled[0]}\\b.{{0,80}}\\b{spelled[1]}\\s+\\w+'\n"
        f"RULE = re.compile(WORDS, re.I)\n", encoding="utf-8")
    found = scenario_wording([guard.module_unit("sample.py", root=tmp_path)])
    assert {kind for _, kind, _ in found} == {"source repeats", "literal is part of", "pattern matches"}
    # One spelled word, however the rest of the pattern is built, is an ordinary code word.
    (tmp_path / "single.py").write_text(
        f"import re\nRULE = re.compile(r'\\b{spelled[0]}\\b.{{0,80}}\\b\\w+\\s+\\w+', re.I)\n", encoding="utf-8")
    assert scenario_wording([guard.module_unit("single.py", root=tmp_path)]) == []


def test_planted_routing_by_wording_is_caught_on_a_component_and_on_an_operand(tmp_path, monkeypatch):
    """What sub-phase 1c removed, planted again: a router that matches a task's
    description, a rule table looked up with it, and a test of an operand the
    person wrote. Each is found where the text reaches it; a helper that only
    compares the operand with the message it came from is not."""
    module = plant(tmp_path, monkeypatch, "routing", '''
        import re
        from dataclasses import dataclass, field

        _RULES = {"fetch": "web.fetch", "erase": "fs.delete_path"}
        _METRIC = re.compile(r"\\\\b(?:usage|utilisation)\\\\b", re.I)

        @dataclass
        class Component:
            description: str
            kind: str = "task"
            requested_tool: str = ""

        @dataclass
        class Intent:
            tool_name: str
            arguments: dict = field(default_factory=dict)

        def deliberate(component, message):
            intent = Intent(by_table(component) or "shell.run_command", {"cmd": component.description})
            return by_pattern(component), by_operand(intent), exact(intent, message), by_label(component)

        def by_table(component):
            return next((tool for word, tool in _RULES.items() if word in component.description), "")

        def by_pattern(component):
            return bool(_METRIC.search(component.description))

        def by_operand(intent):
            return intent.arguments["cmd"].startswith("erase")

        def exact(intent, message):
            return intent.arguments["cmd"] in message

        def by_label(component):
            return component.kind == "task" and not component.requested_tool
    ''')
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.deliberate(module.Component(f"erase the {MARKER} notes"), f"please erase the {MARKER} notes")
    flagged = {label.split("::")[1]: found for label, found in trace.findings().items()}
    assert set(flagged) == {"by_table", "by_pattern", "by_operand"}
    assert all(any(finding.on_text for finding in found) for found in flagged.values())
    assert {finding.kind for finding in flagged["by_pattern"]} == {"regex"}
    assert "erase" in {term for finding in flagged["by_operand"] for term in finding.terms}

"""Structural guards for perception by understanding.

These hold by reading the source and by watching where a message travels, not
by behaviour:

* the perception modules and the perception stage hold no vocabulary: no word
  lists, no word-matching patterns however they are bound, no probing of text
  with fixed wording, and no use of ``core.social``;
* every function the user's text reaches during the pre-pass and the decision
  it commits, whether as a string, inside the turn state or inside a request
  component, is checked strictly: it is free of vocabulary, or named below as
  wording-based code still to be removed, or holds only the runtime's own
  labels, each pinned. A literal tested against what a person wrote counts
  whatever the literal is, and so does a lookup table of any size;
* runtime code never imports test code (the keyword reader is test-only);
* none of the evaluation scenarios' wording appears in the perception path
  (ground rule 1 of ``docs/development/goal-fix-plan.md``).

Each detector is proved against planted violations at the end of this file.
``SCENARIO_PHRASE_SCOPE`` is the list sub-phase 1b widens to every runtime module.
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
from core.types import TaskPlan, TaskStep
from openrook_tools.executor import ToolExecutor
from openrook_tools.registry import ToolRegistry
from openrook_tools.roleplay_eval import scenarios
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from tests.support import vocabulary_guard as guard
from tests.support.readings import (
    ReadingBackend, ScriptedReader, conversation, hostility, reading, role, task,
)

ROOT = Path(__file__).resolve().parents[1]

# The modules that turn a message into what the decision sees.
PERCEPTION_MODULES = ("core/perception.py", "core/appraisal.py", "core/contagion.py")
# The pipeline's perception stage.
PERCEPTION_STAGE = (pipeline_core.default_message_reader, CognitivePipeline._stage_pre_pass)
# Modules on the perception path that may not reach core.social at all.
NO_SOCIAL_IMPORT = (*PERCEPTION_MODULES, "core/self_intent.py", "pipeline/core.py")

RUNTIME_PACKAGES = ("core", "pipeline", "runtime", "interface", "config", "openrook_tools")
# Evaluation tooling that already borrows offline fakes from tests. It may not
# reach the keyword reader either; nothing else may import tests at all.
EVAL_FAKES_IMPORTERS = {"evals/backends.py", "evals/runner.py"}

# Wording-based code the user's text still reaches during the pre-pass and the
# decision. None of it supplies social meaning. Each entry names what it does and where
# it goes; the list must match the code exactly, so adding a keyword helper
# fails here and removing one means deleting its line.
PENDING_WORDING = {
    "core/system_inspection.py::inspection_kind": "1c routing: host-metric questions",
    "core/dual_process/tool_loop.py::match_intent_rule": "1c routing: tool intent rules (detect_tool_intent)",
    "core/task_planning.py::is_continue_request": "1c routing: continue a running plan",
    "core/task_planning.py::is_status_request": "1c routing: status of a running plan",
    "core/request_actions.py::has_learning_intent": "1c routing: learning requests",
    "core/request_actions.py::file_requests": "1c routing: literal file operands",
    "core/request_actions.py::ground_file_components": (
        "1c routing: literal file operands. Rewrites the capability requirements of a task its English verb "
        "patterns match, and turns a task whose description starts with a reporting verb into conversation"),
    "core/request_actions.py::ground_shell_components": (
        "1c routing: literal shell operands. Rewrites the capability requirements of a task its English verb "
        "pattern matches"),
    "pipeline/heuristics.py::_is_identity_question": "1c context selection: identity records",
    "pipeline/heuristics.py::_relationship_topic_hint": "1c context selection: relationship topic",
    "core/memory/semantic.py::_tokenize": "1c context selection: memory retrieval stop-words",
    "runtime/life_history.py::_token_set": "1c context selection: life-history retrieval stop-words",
    "runtime/evolution_policy.py::detect_directive_override_markers": "Phase 4: autonomy-phrase memory filter",
    "runtime/evolution_policy.py::detect_prompt_injection_markers": "Phase 4: instruction-phrase memory filter",
}
# A model answer quotes the message, so answer clean-up is reached by the text
# too. It matches markup the model emits, not anything a person wrote.
MODEL_ANSWER_MARKUP = {
    "core/llm_content.py::strip_reasoning": "removes reasoning tags from a model answer",
}
# Functions the text reaches whose flagged literals are the runtime's own labels
# and record keys, not wording matched against the message. Every such literal
# is pinned in the fixture, so nothing can join them unnoticed: a new literal
# in one of these functions fails the guard until it has been looked at.
REVIEWED_LABELS = json.loads((ROOT / "tests/fixtures/perception_reviewed_labels.json").read_text(encoding="utf-8"))

# Ground rule 1. Sub-phase 1b: replace with every runtime module.
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
    """One function-level import exists, for the evaluation judge's reply check.

    ``openrook_tools/roleplay_eval/judge.py`` imports ``core.perception.observe``
    and is reworked with the other reply checks. Until then ``observe`` forwards
    to the reply-check cue. Nothing on the perception path may call it, and no
    other use of ``core.social`` is allowed here.
    """
    uses = []
    for path in NO_SOCIAL_IMPORT:
        tree = guard.module_unit(path).tree
        for name, _, node in imports(tree):
            if name == "core.social" or name.startswith("core.social."):
                owner = next((f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                              and any(child is node for child in ast.walk(f))), None)
                uses.append((path, owner, name))
        if path != "core/perception.py":
            assert "core.perception.observe" not in [name for name, _, _ in imports(tree)], path
            called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            assert "observe" not in called, path
    assert uses == [("core/perception.py", "observe", "core.social"),
                    ("core/perception.py", "observe", "core.social.role_request_spans")]
    perception = guard.module_unit("core/perception.py").tree
    module_level = [name for node in perception.body if isinstance(node, (ast.Import, ast.ImportFrom))
                    for name, _, _ in imports(node)]
    assert module_level and not any(name.startswith("core.social") for name in module_level)
    callers = [f.name for f in ast.walk(perception) if isinstance(f, ast.FunctionDef)
               and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "observe"
                       for n in ast.walk(f))]
    assert callers == []


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


def traced_turns(tmp_path) -> guard.TextTrace:
    """Run real turns and record every runtime function the message text is handed to.

    The text is followed as a string, inside the turn state, and inside request
    components, through the pre-pass and the decision it commits. The turns
    cover each branch that hands the text on: an unread turn and a follow-up,
    a turn whose reading quotes the message in its components and evidence, a
    turn with tools and no reported request, a turn during a running plan, and
    a turn through the session manager, which wires the life-history, skill and
    learning providers.
    """
    trace = guard.TextTrace(MARKER)
    original = CognitivePipeline._stage_pre_pass

    def traced(self, st):
        with trace:
            return original(self, st)

    CognitivePipeline._stage_pre_pass = traced
    try:
        pipe = CognitivePipeline(ReadingBackend())
        pipe.process(MESSAGE, "guard")
        pipe.process(MESSAGE + " Again.", "guard")
        pipe.close()
        pipe = CognitivePipeline(ReadingBackend(quoting))
        pipe.process(MESSAGE, "guard")
        pipe.close()
        pipe = CognitivePipeline(ReadingBackend(), tool_executor=ToolExecutor(ToolRegistry()),
                                 message_reader=ScriptedReader(reading()))
        pipe.process(MESSAGE, "guard")
        pipe._active_task_plan = TaskPlan(id="plan", goal="tidy the notes", steps=[
            TaskStep(id="s1", tool_name="fs.list_dir", arguments={"path": "."}, description="look")])
        pipe.process(MESSAGE, "guard")
        pipe.close()
        runtime = SessionManager(RuntimeConfig(data_dir=str(tmp_path)), backend_factory=ReadingBackend)

        async def through_the_runtime():
            try:
                await runtime.handle_message("web", "guard", "chat", MESSAGE)
            finally:
                await runtime.shutdown()

        asyncio.run(through_the_runtime())
    finally:
        CognitivePipeline._stage_pre_pass = original
    return trace


@pytest.fixture(scope="module")
def reached(tmp_path_factory) -> guard.TextTrace:
    return traced_turns(tmp_path_factory.mktemp("guard"))


def test_every_function_the_text_reaches_is_clean_or_named_as_pending(reached):
    """Strict, and against the text itself: a literal tested on what a person
    wrote counts whatever the literal is."""
    flagged = reached.findings()
    assert len(reached.reached) > 60  # the trace really followed the message
    assert set(flagged) == set(PENDING_WORDING) | set(MODEL_ANSWER_MARKUP) | set(REVIEWED_LABELS)
    for label, pinned in REVIEWED_LABELS.items():
        # Labels only: no pattern, nothing tested against the text, and no literal that is not pinned.
        assert {finding.kind for finding in flagged[label]} <= {"words", "probe"}, label
        assert not [finding for finding in flagged[label] if finding.on_text], label
        assert sorted({term for finding in flagged[label] for term in finding.terms} - set(pinned)) == [], label
    # What a person wrote is tested against fixed wording only in code named as pending.
    tested = {label for label, found in flagged.items() if any(finding.on_text for finding in found)}
    assert tested <= set(PENDING_WORDING) | set(MODEL_ANSWER_MARKUP)
    # The text reaches the reader, the parser, the checks and the decision. The
    # perception modules and the stage hold nothing; the decision holds pinned labels only.
    labels = {unit.label for unit, _, _ in reached.units()}
    assert {"core/appraisal.py::read_message", "core/appraisal.py::parse_reading",
            "core/appraisal.py::reconcile_social_appraisal", "core/perception.py::tokens",
            "core/willingness.py::decide", "pipeline/core.py::CognitivePipeline._commit_turn_decision"} <= labels
    assert not [label for label in flagged if label.split("::")[0] in PERCEPTION_MODULES]
    assert "pipeline/core.py::CognitivePipeline._stage_pre_pass" not in flagged


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
    assert len(handed) >= 12 and sorted(set(handed) - traced) == []
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


def scenario_wording(units: list[guard.Unit]) -> list[tuple[str, str, str]]:
    """Places where runtime source repeats, contains or pattern-matches scenario wording."""
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
            matched = next((turn for turn in sorted(turns) if compiled.search(turn)), None)
            if matched:
                found.append((unit.label, "pattern matches", f"{pattern[:60]} ~ {matched[:60]}"))
    return found


def test_scenario_wording_is_absent_from_the_perception_path():
    assert len(scenario_phrases()[0]) > 50
    assert scenario_wording([unit_of(entry) for entry in SCENARIO_PHRASE_SCOPE]) == []


@pytest.mark.xfail(strict=True, reason="Sub-phase 1b removes scenario wording from the reply checks "
                   "and widens SCENARIO_PHRASE_SCOPE to every runtime module.")
def test_scenario_wording_is_absent_from_every_runtime_module():
    catalog = "openrook_tools/roleplay_eval/scenarios.py"  # the wording's own home
    assert scenario_wording([guard.module_unit(path) for path in runtime_sources() if path != catalog]) == []


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
    # Words that only occur in the reader's instructions are not data-model names.
    assert not names & {"hostile", "joke", "sarcasm", "playfulness", "obey", "quotation", "relationship",
                        "standing", "forgive", "refuse", "comply", "advice", "the", "message"}


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
    (tmp_path / "sample.py").write_text(
        f"import re\n# quoted in a comment: {scenarios.WARM[0]}\n"
        f"PART = {' '.join(words[:4])!r}\n"
        f"WORDS = r'\\b{words[0].lower()}\\s+{words[1].lower()}\\s+\\w+'\n"
        f"RULE = re.compile(WORDS, re.I)\n", encoding="utf-8")
    found = scenario_wording([guard.module_unit("sample.py", root=tmp_path)])
    assert {kind for _, kind, _ in found} == {"source repeats", "literal is part of", "pattern matches"}

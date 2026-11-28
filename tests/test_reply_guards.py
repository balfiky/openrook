"""Structural guards for the reply path: writing the reply, reviewing it, delivering it.

These hold by reading the source and by watching where a draft and the user's
text travel, not by behaviour. They widen the guards of
``tests/test_perception_guards.py`` from the perception stage to the rest of a
turn, with the same tooling (``tests/support/vocabulary_guard.py``):

* the review modules hold no pattern that spells a word, and every label they
  hold is pinned;
* every function the draft or the user's text reaches while the reply prompt is
  assembled, the draft is reviewed and the reply is delivered is free of
  vocabulary, or strips the runtime's own markup, or holds only pinned labels.
  Nothing in that path is pending: no function in it tests the text against
  fixed wording;
* the former wording-based reply checks, kept for the evaluation harness, are
  never reached by a turn and are imported only where named;
* no runtime module repeats evaluation wording, outside an exact list
  (ground rule 1 of ``docs/development/goal-fix-plan.md``).

Each detector is proved against planted violations at the end of this file.

What the trace covers, exactly
------------------------------
The traced stages are ``_stage_generate`` and ``_stage_self_check``: the reply
prompt, the writer, the gate, the reviews and the delivery. They are traced in
conversation and in research mode, with receipts that a real tool loop
produced, through rewrites, excision, the protected final attempt and the
confirmation review.

The stages before these, in which the message is read, the decision is made
and what it granted is carried out (``_stage_pre_pass``,
``_stage_deliberation``), are traced by ``tests/test_perception_guards.py``:
since sub-phase 1c nothing in them matches the user's text against wording to
say what is asked or which tool runs. The draft never reaches them: they run
before the reply is written.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
import re
import textwrap

import pytest

from core.types import ToolResult, ToolTrace
from openrook_tools.roleplay_eval import scenarios
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from runtime.tools import create_tool_executor
from tests.support import vocabulary_guard as guard
from tests.support.readings import (
    ReadingBackend, ScriptedReader, conversation, plain, reading, role, task, using,
)
from tests.support.reviews import ScriptedReviewer, finding, passed, violation
from tests.test_perception_guards import (
    WORD_LETTERS, WORDS_OF_A_SCENARIO, imports, plant, runtime_sources, scenario_phrases, scenario_wording,
)

ROOT = Path(__file__).resolve().parents[1]

# The modules that check a draft and decide what is delivered.
REVIEW_MODULES = ("core/review.py", "core/dignity.py", "core/grounding.py", "pipeline/delivery.py")
# The stages of a turn in which a reply is written, reviewed and delivered.
REPLY_STAGES = ("_stage_generate", "_stage_self_check")

# Functions the text reaches that match markup the runtime or a model emits,
# not anything a person wrote or a character said.
RUNTIME_MARKUP = {
    "core/llm_content.py::strip_reasoning": "removes reasoning tags from a model answer",
    "core/llm_content.py::strip_internal_expression_markers": "removes the runtime's own reserved control markers",
}
# Functions and modules whose flagged literals are the runtime's own labels and
# record keys. Every such literal is pinned in the fixture: a new one fails the
# guard until it has been looked at.
REVIEWED_LABELS = json.loads((ROOT / "tests/fixtures/reply_reviewed_labels.json").read_text(encoding="utf-8"))

# The former wording-based checks live with the evaluation harness.
LEGACY_MODULES = ("openrook_tools.roleplay_eval.legacy_claims", "openrook_tools.roleplay_eval.legacy_social",
                  "openrook_tools.roleplay_eval.legacy_routing")
LEGACY_FILES = ("openrook_tools/roleplay_eval/legacy_claims.py", "openrook_tools/roleplay_eval/legacy_social.py",
                "openrook_tools/roleplay_eval/legacy_routing.py")
# Who may import them: the harness itself, the offline replay tool, and the
# compatibility names below, each inside a module ``__getattr__`` and nowhere else.
HARNESS = ("openrook_tools/roleplay_eval/", "openrook_tools/replay_dignity.py")
COMPATIBILITY = {
    "core/grounding.py": ("verify_response_grounding",),
    "core/metric_claims.py": ("cpu_idle_percentages", "cpu_measurement_complete", "percentage_comparisons",
                              "cpu_interpretation_conflicts", "capacity_interpretation_conflicts"),
    "core/system_inspection.py": ("addresses_inspection",),
    "core/response_language.py": ("language_mismatch",),
    "core/llm_content.py": ("requests_literal_text",),
    # For tests/test_live_roleplay_p1j.py, which imports it and is being changed elsewhere.
    "core/request_actions.py": ("ground_shell_components",),
}
DELETED = ("core/social.py", "core/expression_claims.py", "core/response_style.py")

# Ground rule 1, for every runtime module. Each entry names why the module
# still carries evaluation wording and where that goes. The list must match the
# code exactly: a new offender fails, and so does an entry that has been fixed.
SCENARIO_WORDING_PENDING = {
    "runtime/evolution_policy.py": "Phase 4: instruction-phrase memory filter",
    "openrook_tools/replay_dignity.py": "offline replay tool: carries its own script of turns",
    "openrook_tools/roleplay_eval/legacy_claims.py": "evaluation harness: the former wording-based claim checks",
    "openrook_tools/roleplay_eval/legacy_social.py": "evaluation harness: the former social and role cues",
    "openrook_tools/roleplay_eval/legacy_routing.py": "evaluation harness: the former keyword routing of requests",
}
# No module a turn runs is on that list any more: what is left is a memory
# filter of a later phase, an offline tool with its own script, and the harness.
TURN_PATH_PACKAGES = ("core/", "pipeline/")
SCENARIO_CATALOG = "openrook_tools/roleplay_eval/scenarios.py"   # the wording's own home


def known_names() -> set[str]:
    return guard.data_model_names() | guard.review_model_names()


def labels_are_pinned(label: str, found: list[guard.Finding]) -> None:
    """Labels only: no pattern, nothing tested against the text, no literal that is not pinned."""
    assert {finding.kind for finding in found} <= {"words", "probe"}, label
    assert not [finding for finding in found if finding.on_text], label
    assert sorted({term for finding in found for term in finding.terms} - set(REVIEWED_LABELS[label])) == [], label


# ---------------------------------------------------------------------------
# No vocabulary in the review modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", REVIEW_MODULES)
def test_the_review_modules_hold_no_word_pattern_and_only_pinned_labels(path):
    unit = guard.module_unit(path)
    found = guard.findings(unit.tree, unit.scope, known=known_names(), strict=True)
    assert not [finding for finding in found if finding.kind == "regex"], path
    labels_are_pinned(path, found)


def test_the_review_model_names_come_from_records_and_the_schema_not_from_prose():
    names = guard.review_model_names()
    assert {"role_assent", "unverified_external_action_claim", "verdict", "span", "critical",
            "user_message_echo", "deterministic", "observed_output", "executed_results"} <= names
    # Words that only occur in the reviewer's instructions are not data-model names.
    assert not names & {"declined", "undertake", "decline", "accept", "refuse", "sexual", "attention",
                        "relationship", "standing", "manner", "tone", "warmth", "courtesy", "the", "draft"}


def test_the_deleted_word_list_modules_are_gone():
    assert [path for path in DELETED if (ROOT / path).exists()] == []
    import core.perception

    assert not hasattr(core.perception, "observe")
    for path in runtime_sources():
        names = {name for name, _, _ in imports(guard.module_unit(path).tree)}
        assert not {name for name in names if name.startswith(("core.social", "core.expression_claims",
                                                                "core.response_style"))}, path


# ---------------------------------------------------------------------------
# Everything the draft and the user's text reach while a reply is written,
# reviewed and delivered
# ---------------------------------------------------------------------------

MARKER = "qzvmarkerx"
MESSAGE = f"Tell me about {MARKER} gardens, please?"


def traced_turns(tmp_path) -> guard.TextTrace:
    """Run real turns and record every runtime function the draft or the message is handed to.

    The marker is in the user's message and in every draft, so both are
    followed: as strings, inside the turn state and the frozen decision, inside
    review evidence and inside findings. The turns cover each branch of the
    reply path: a plain turn and a follow-up through the runtime's own
    reviewer; reviewer findings that lead to excision of an unsupported claim,
    a rewrite with quoted findings and excision of a quoted span; drafts that
    copy the user until the protected final attempt; a figure the receipts do
    not carry; a block that poses a refused command's argument as output, with
    the review unavailable; a block no receipt returned, which the reviewer is
    told of; a fixed limit only the reviewer reports, up to the confirmation
    review; a turn whose receipts come from a real tool loop; a turn in
    research mode; and a turn through the session manager.
    """
    trace = guard.TextTrace(MARKER)
    originals = {name: getattr(CognitivePipeline, name) for name in REPLY_STAGES}

    def wrap(original):
        def traced(self, st):
            with trace:
                return original(self, st)
        return traced

    for name, original in originals.items():
        setattr(CognitivePipeline, name, wrap(original))
    try:
        pipe = CognitivePipeline(ReadingBackend(plain, drafts=[f"A {MARKER} reply about beds.",
                                                               f"Another {MARKER} line here."]))
        pipe.process(MESSAGE, "guard")
        pipe.process(MESSAGE + " Again.", "guard")
        pipe.close()

        claim = f"I ran the `{MARKER}` survey"

        def reviews(draft, evidence):
            if claim in draft:
                return violation(finding("unverified_external_action_claim", claim))
            if "beds are fine" in draft:
                return violation(finding("role_assent", draft.split(".")[0], severity="critical"))
            return passed()

        pipe = CognitivePipeline(
            ReadingBackend(drafts=[f"{claim}. The {MARKER} beds are fine. More {MARKER} text follows.",
                                   f"The {MARKER} beds are fine. A second {MARKER} draft."]),
            message_reader=ScriptedReader(lambda text: reading(role(text, MARKER, autonomy_cost=0.9),
                                                               conversation(text))),
            expression_reviewer=ScriptedReviewer(reviews))
        pipe.process(MESSAGE, "guard")
        pipe.close()

        for drafts in ([MESSAGE], [f"{MESSAGE} The {MARKER} point: {MESSAGE}"]):
            # A whole copy until the protected final attempt; then a copied opening that recurs inside a sentence.
            pipe = CognitivePipeline(ReadingBackend(drafts=drafts), message_reader=ScriptedReader(plain),
                                     expression_reviewer=ScriptedReviewer(passed()))
            pipe.process(MESSAGE, "guard")
            pipe.close()

        def with_receipts(pipe, *results):
            original = pipe._stage_deliberation

            def deliberate(st):
                original(st)
                st.debug.tool_trace = ToolTrace(executed_results=list(results))

            pipe._stage_deliberation = deliberate
            return pipe

        pipe = with_receipts(
            CognitivePipeline(ReadingBackend(drafts=[f"The {MARKER} memory is 99% used. Otherwise fine."]),
                              message_reader=ScriptedReader(plain), expression_reviewer=ScriptedReviewer(passed())),
            ToolResult("system.memory_usage", True, f"{MARKER} RAM 23%", metadata={"used_percent": 22.84}))
        pipe.process(MESSAGE, "guard")
        pipe.close()

        # A block that quotes what a receipt returned, which the reviewer disputes: the finding is withdrawn.
        returned = f"```\n{MARKER} RAM 23%\n```"
        pipe = with_receipts(
            CognitivePipeline(ReadingBackend(drafts=[f"The {MARKER} tool said:\n{returned}\nSo it is fine."]),
                              message_reader=ScriptedReader(plain), expression_reviewer=ScriptedReviewer(
                                  violation(finding("unverified_external_action_claim", returned)))),
            ToolResult("system.memory_usage", True, f"{MARKER} RAM 23%", metadata={"used_percent": 22.84}))
        pipe.process(MESSAGE, "guard")
        pipe.close()

        refused = ToolResult("shell.run_command", False, "", error="command not allowed",
                             metadata={"execution_started": False, "cmd": f"echo {MARKER}"})
        # The argument of the refused command, set apart as if it had been printed; no review verdict.
        posed = f"The {MARKER} command printed:\n\n```\n{MARKER}\n```\n\nThat is the {MARKER} result."
        pipe = with_receipts(
            CognitivePipeline(ReadingBackend(drafts=[posed]), message_reader=ScriptedReader(plain),
                              expression_reviewer=ScriptedReviewer(lambda draft, evidence: None)), refused)
        pipe.process(MESSAGE, "guard")
        pipe.close()
        # A block that is in no receipt and poses nothing: the reviewer is told of it.
        own = f"Here is a {MARKER} sketch:\n```\nrows of {MARKER} beans\n```\nThat is my {MARKER} plan."
        pipe = with_receipts(
            CognitivePipeline(ReadingBackend(drafts=[own]), message_reader=ScriptedReader(plain),
                              expression_reviewer=ScriptedReviewer(passed())), refused)
        pipe.process(MESSAGE, "guard")
        pipe.close()

        # A fixed limit only the reviewer reports, on every draft: fresh attempts, then the confirmation.
        crossing = f"A {MARKER} line the reviewer objects to."
        pipe = CognitivePipeline(
            ReadingBackend(drafts=[crossing]), message_reader=ScriptedReader(plain),
            expression_reviewer=ScriptedReviewer(
                lambda draft, evidence: violation(finding("unsolicited_sexualization", draft.rstrip("."))),
                confirmation=passed()))
        pipe.process(MESSAGE, "guard")
        pipe.close()

        # Receipts from a real tool loop: the executor runs the tool, and what it returned reaches the reply.
        config = RuntimeConfig(data_dir=str(tmp_path / "tools"), tools_enabled=True, autonomy_level="high_risk")
        executor = create_tool_executor(config)
        executor.register_handler("system.memory_usage", lambda arguments: ToolResult(
            tool_name="system.memory_usage", success=True, output=f"{MARKER} RAM total: 16 GB; used: 6 GB."))
        pipe = CognitivePipeline(
            ReadingBackend(drafts=[f"The {MARKER} RAM total is 16 GB."]), tool_executor=executor,
            runtime_config=config, autonomy_level="high_risk", expression_reviewer=ScriptedReviewer(passed()),
            message_reader=ScriptedReader(lambda text: reading(task(text, tool=using("system.memory_usage")))))
        ran = pipe.process(f"Show current memory usage, {MARKER}", "guard")
        assert [result.tool_name for result in ran.debug.tool_trace.executed_results] == ["system.memory_usage"]
        pipe.close()

        # Research mode: the review is one counted call of the runtime's own reviewer, made where it is due.
        pipe = CognitivePipeline(ReadingBackend(
            lambda text: reading(role(text, MARKER, autonomy_cost=0.9), conversation(text)),
            drafts=[f"No. The {MARKER} beds are mine."]))
        pipe._research_mode = True
        studied = pipe.process(MESSAGE, "guard")
        assert [item["provenance"] for item in studied.debug.expression_reviews] == ["model"]
        pipe.close()

        runtime = SessionManager(RuntimeConfig(data_dir=str(tmp_path)),
                                 backend_factory=lambda: ReadingBackend(plain, drafts=[f"A {MARKER} reply."]))

        async def through_the_runtime():
            try:
                await runtime.handle_message("web", "guard", "chat", MESSAGE)
            finally:
                await runtime.shutdown()

        asyncio.run(through_the_runtime())
    finally:
        for name, original in originals.items():
            setattr(CognitivePipeline, name, original)
    return trace


@pytest.fixture(scope="module")
def reached(tmp_path_factory) -> guard.TextTrace:
    return traced_turns(tmp_path_factory.mktemp("reply-guard"))


def test_every_function_the_draft_reaches_is_clean_markup_or_pinned_labels(reached):
    """Strict, and against the text itself. Nothing in the reply path is pending."""
    flagged = reached.findings(known=known_names())
    assert len(reached.reached) > 60   # the trace really followed the draft and the message
    functions = {label for label in REVIEWED_LABELS if "::" in label}
    assert set(flagged) == set(RUNTIME_MARKUP) | functions
    for label in functions:
        labels_are_pinned(label, flagged[label])
    # The text is tested against fixed wording nowhere but in the two markup strippers.
    tested = {label for label, found in flagged.items() if any(finding.on_text for finding in found)}
    assert tested == set(RUNTIME_MARKUP)
    for label in RUNTIME_MARKUP:
        assert {finding.kind for finding in flagged[label]} == {"regex"}, label


def test_the_trace_covers_the_whole_reply_path(reached):
    labels = {f"{path}::{qualname}" for path, qualname in reached.reached}
    assert {"core/dual_process/generator.py::build_system_prompt",
            "core/dual_process/generator.py::ResponseGenerator.generate",
            "core/llm_content.py::strip_expression_artifacts",
            "pipeline/core.py::CognitivePipeline._check_dignity",
            "pipeline/delivery.py::DeliveryGate._review", "pipeline/delivery.py::DeliveryGate.add",
            "pipeline/delivery.py::DeliveryGate.repair",
            "core/dignity.py::review_response", "core/dignity.py::local_issues",
            "core/dignity.py::whole_message_echo", "core/dignity.py::user_echo_span",
            "core/dignity.py::private_instruction_span", "core/dignity.py::receipt_syntax_span",
            "core/dignity.py::repeated_spans", "core/dignity.py::excise_ineligible_spans",
            "core/dignity.py::excise_choice_sentences", "core/dignity.py::correction_instruction",
            "core/dignity.py::output_block_issues", "core/dignity.py::_with_introduction",
            "core/grounding.py::unsupported_percentages", "core/grounding.py::receipt_verified_span",
            "core/grounding.py::posed_output_blocks", "core/grounding.py::blocks_not_in_receipts",
            "pipeline/delivery.py::DeliveryGate._settled",
            "core/review.py::review_evidence", "core/review.py::review_draft", "core/review.py::parse_review",
            "core/perception.py::tokens", "core/perception.py::locate", "core/perception.py::sentence_ranges",
            } <= labels
    # Every function of the review modules that takes a piece of the reply was exercised.
    takes_text = {"response", "draft", "span", "claim", "text", "sentence", "user", "user_message"}
    numerals_only = {"core/grounding.py::_number"}   # handed one numeral, never a sentence
    for path in REVIEW_MODULES:
        tree = guard.module_unit(path).tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                owner = node.name
                members = [item for item in node.body if isinstance(item, ast.FunctionDef)]
            elif isinstance(node, ast.FunctionDef) and node in tree.body:
                owner, members = "", [node]
            else:
                continue
            for function in members:
                parameters = {argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)}
                if parameters & takes_text and not function.name.startswith("__"):
                    qualname = f"{owner}.{function.name}" if owner else function.name
                    assert f"{path}::{qualname}" in labels | numerals_only, (
                        f"{path}::{qualname} was not reached by the trace")


def forwarders_asked(trace: guard.TextTrace) -> list[tuple[str, str]]:
    """The names a turn asked a compatibility module's ``__getattr__`` to resolve.

    The compatibility names are bound nowhere: each is fetched on demand, so a
    use of one is a recorded request. The import system also probes a module
    for ``__path__`` and the like; those are not names anybody asked for.
    """
    return sorted((path, name) for path, name in trace.asked
                  if path in COMPATIBILITY and not (name.startswith("__") and name.endswith("__")))


def test_no_turn_reaches_the_wording_based_checks_kept_for_the_evaluation(reached):
    """Read from the record of every call the turns made, with or without the text in hand."""
    called = reached.called
    assert len(called) > 150 and set(reached.reached) < called
    assert ("pipeline/delivery.py", "DeliveryGate.select") in called       # a function the text is not an argument of
    assert [key for key in called if key[0].startswith("openrook_tools/roleplay_eval/")] == []
    assert [key for key in called if key[0] in ("core/metric_claims.py", "core/response_language.py")] == []
    # No compatibility name, and no other missing name, was asked for by a turn.
    assert forwarders_asked(reached) == []


# ---------------------------------------------------------------------------
# The wording-based checks are imported only where named
# ---------------------------------------------------------------------------


def legacy_importers(paths, root: Path = ROOT) -> list[tuple[str, str | None]]:
    """Every place a source file imports the harness's wording-based modules: (path, enclosing function)."""
    found = []
    for path in paths:
        tree = guard.module_unit(path, root=root).tree
        for name, _, node in imports(tree):
            if name in LEGACY_MODULES or name.startswith(tuple(f"{module}." for module in LEGACY_MODULES)):
                owner = next((f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                              and any(child is node for child in ast.walk(f))), None)
                found.append((path, owner))
    return sorted(set(found))


def test_the_wording_based_checks_are_imported_only_by_the_harness_and_the_named_compatibility():
    importers = legacy_importers(runtime_sources())
    outside = [(path, owner) for path, owner in importers if not path.startswith(HARNESS)]
    assert outside == sorted((path, "__getattr__") for path in COMPATIBILITY)
    for path, names in COMPATIBILITY.items():
        module = guard.module_unit(path).tree
        declared = next(node.value for node in module.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "_EVALUATION_COMPATIBILITY"
                                for target in node.targets))
        assert tuple(ast.literal_eval(declared)) == names, path
    # Inside the runtime nobody asks for a compatibility name: only the evaluation does.
    compatibility = {f"{path[:-3].replace('/', '.')}.{name}" for path, names in COMPATIBILITY.items()
                     for name in names}
    users = [(path, name) for path in runtime_sources() if not path.startswith(HARNESS)
             for name, _, _ in imports(guard.module_unit(path).tree) if name in compatibility]
    assert users == []


def test_the_compatibility_names_still_resolve_for_the_evaluation():
    from core.grounding import verify_response_grounding
    from core.llm_content import requests_literal_text
    from core.metric_claims import cpu_idle_percentages, percentage_comparisons
    from core.response_language import language_mismatch
    from core.request_actions import ground_shell_components
    from core.system_inspection import addresses_inspection
    from openrook_tools.roleplay_eval import legacy_claims, legacy_routing

    assert verify_response_grounding is legacy_claims.verify_response_grounding
    assert requests_literal_text is legacy_claims.requests_literal_text
    assert cpu_idle_percentages is legacy_claims.cpu_idle_percentages
    assert percentage_comparisons is legacy_claims.percentage_comparisons
    assert language_mismatch is legacy_claims.language_mismatch
    assert addresses_inspection is legacy_claims.addresses_inspection
    assert ground_shell_components is legacy_routing.ground_shell_components
    import core.metric_claims

    with pytest.raises(AttributeError):
        core.metric_claims.cpu_usage_claims


# ---------------------------------------------------------------------------
# No evaluation wording in any runtime module (ground rule 1)
# ---------------------------------------------------------------------------


def test_scenario_wording_is_absent_from_every_runtime_module_outside_the_exact_pending_list():
    assert len(scenario_phrases()[0]) > 50
    found = scenario_wording([guard.module_unit(path) for path in runtime_sources() if path != SCENARIO_CATALOG])
    offenders = {label for label, _, _ in found}
    assert offenders == set(SCENARIO_WORDING_PENDING), sorted(found)
    assert not [path for path in SCENARIO_WORDING_PENDING if path.startswith(TURN_PATH_PACKAGES)]
    # None of the modules that write, review or deliver a reply is on the list.
    reply_path = {*REVIEW_MODULES, "core/dual_process/generator.py", "core/llm_content.py", "core/perception.py",
                  "pipeline/core.py"}
    assert not reply_path & set(SCENARIO_WORDING_PENDING)


# ---------------------------------------------------------------------------
# The guards catch what they guard against
# ---------------------------------------------------------------------------


def test_a_planted_word_list_behind_the_review_is_caught_by_the_draft_trace(tmp_path, monkeypatch):
    """The review's own body can be spotless while a helper it calls matches the draft against words."""
    module = plant(tmp_path, monkeypatch, "reviewing", '''
        import re
        _ASSENT = re.compile(r"\\\\b(?:gladly|certainly)\\\\b", re.I)
        _SOFT = ("thankful", "glad")

        def review(draft, user_message):
            return structural(draft, user_message), by_pattern(draft), by_list(draft.casefold()), labels(len(draft))

        def structural(draft, user_message):
            return draft.casefold() == user_message.casefold() or draft[:4] in user_message

        def by_pattern(draft):
            return bool(_ASSENT.search(draft))

        def by_list(lowered):
            return any(word in lowered for word in _SOFT)

        def labels(count):
            return ("pass", "violation")[count % 2]
    ''')
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.review(f"A {MARKER} draft", f"a {MARKER} message")
    assert {qualname for _, qualname in trace.reached} == {"review", "structural", "by_pattern", "by_list"}
    flagged = {label.split("::")[1]: found for label, found in trace.findings(known=known_names()).items()}
    # Comparing the draft with the message is structure; the two helpers are vocabulary.
    assert set(flagged) == {"by_pattern", "by_list"}
    assert all(any(finding.on_text for finding in found) for found in flagged.values())


def on_text(source: str) -> list[guard.Finding]:
    tree = ast.parse(textwrap.dedent(source))
    return [finding for finding in guard.findings(tree, guard.Scope(tree), known=known_names(),
                                                  text_names={"text", "draft"},
                                                  text_leaves=guard.CORE_TEXT_FIELDS) if finding.on_text]


@pytest.mark.parametrize("source", [
    # A piece of the text looked for in another text: neither side is fixed wording.
    'def f(text, draft):\n    return draft in text or draft.casefold() not in text.casefold()',
    'def f(text, draft):\n    opening = draft.split(".")[0]\n    return opening in text',
    # Taking a range out of the text, by positions computed from it.
    'def f(text):\n    spans = [(0, 3)]\n    return [text[start:end] for start, end in spans]',
    'import re\nFENCE = re.compile(r"```.*?```", re.S)\n'
    'def f(text):\n    plain = FENCE.sub(" ", text)\n    return [plain[a:b] for a, b in [(0, len(plain))]]',
    # Reading a record's field is not a table of words.
    'def f(text, trace):\n    known = set(getattr(trace, "executed_results", None) or [])\n    return text in known',
])
def test_comparing_text_with_text_and_slicing_it_are_not_vocabulary(source):
    assert on_text(source) == []


@pytest.mark.parametrize("source", [
    'WORD = "gladly"\ndef f(text):\n    return WORD in text',
    'WORDS = ("gladly", "certainly")\ndef f(text):\n    return any(word in text for word in WORDS)',
    'TABLE = {"gladly": 1}\ndef f(text):\n    return TABLE[text.casefold()]',
    'TABLE = {"gladly": 1}\ndef f(text):\n    return text[:6] in TABLE',
    'WORDS = ["gladly"]\ndef f(text):\n    first = text.split()[0]\n    return first in WORDS',
    'def f(text):\n    return text.startswith("gladly") or text == "role_assent"',
    'def f(text, trace):\n    known = set(getattr(trace, "executed_results", None) or ["gladly"])\n    return text in known',
])
def test_fixed_wording_tested_against_the_text_is_still_found(source):
    assert on_text(source), source


def test_a_per_turn_cache_keyed_by_the_text_is_not_a_word_table_and_a_fixed_table_is(tmp_path, monkeypatch):
    module = plant(tmp_path, monkeypatch, "caching", '''
        class Gate:
            FIXED = {"gladly": "violation"}

            def __init__(self):
                self.cache = {}

            def review(self, draft):
                verdict = self.cache.get(draft)
                if verdict is None:
                    verdict = self.cache[draft] = "unknown"
                return verdict

            def fixed(self, draft):
                return self.FIXED.get(draft, "unknown")

        def stage(draft):
            gate = Gate()
            gate.review(draft)
            return gate.review(draft), gate.fixed(draft)
    ''')
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.stage(f"a {MARKER} draft")
    flagged = {label.split("::")[1]: found for label, found in trace.findings(known=known_names()).items()}
    assert set(flagged) == {"Gate.fixed"} and all(finding.on_text for finding in flagged["Gate.fixed"])


def test_a_table_that_mixes_fixed_words_with_the_turns_text_is_still_found(tmp_path, monkeypatch):
    """Putting the turn's text into a word table does not hide the words that were in it."""
    module = plant(tmp_path, monkeypatch, "mixing", '''
        class Gate:
            def __init__(self):
                self.seen = {"gladly": "violation"}
                self.words = ["certainly"]
                self.drafts = []

            def review(self, draft):
                self.seen[draft] = "unknown"
                self.words.append(draft)
                self.drafts.append(draft)
                return self.looked_up(draft), self.listed(draft), self.repeated(draft)

            def looked_up(self, draft):
                return self.seen.get(draft)

            def listed(self, draft):
                return draft in self.words

            def repeated(self, draft):
                return draft in self.drafts

        def stage(draft):
            return Gate().review(draft)
    ''')
    trace = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with trace:
        module.stage(f"a {MARKER} draft")
    flagged = {label.split("::")[1]: found for label, found in trace.findings(known=known_names()).items()}
    # The list that holds nothing but the turn's own drafts is not a table of words.
    assert set(flagged) == {"Gate.review", "Gate.looked_up", "Gate.listed"} and "Gate.repeated" not in flagged
    assert {term for finding in flagged["Gate.looked_up"] for term in finding.terms} == {"gladly", "violation"}
    assert {term for finding in flagged["Gate.listed"] for term in finding.terms} == {"certainly"}
    assert all(finding.on_text for found in flagged.values() for finding in found)


def test_the_call_record_sees_a_forwarder_run_and_a_function_that_never_held_the_text(tmp_path, monkeypatch):
    """What the legacy-check guard relies on: a name resolved through a module ``__getattr__`` is a recorded call."""
    module = plant(tmp_path, monkeypatch, "forwarding", '''
        def __getattr__(name):
            if name == "old_check":
                return lambda text: "gladly" in text
            raise AttributeError(name)

        def quiet(count):
            return count + 1

        def turn(draft, legacy):
            import sys
            quiet(len(draft))
            return getattr(sys.modules[__name__], "old_check")(draft) if legacy else None
    ''')
    clean = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with clean:
        module.turn(f"a {MARKER} draft", False)
    assert {name for _, name in clean.called} == {"turn", "quiet"}
    assert {name for _, name in clean.reached} == {"turn"}          # quiet never held the text; it was still called
    asked = guard.TextTrace(MARKER, root=tmp_path, exclude=(), packages=("planted",))
    with asked:
        module.turn(f"a {MARKER} draft", True)
    assert "__getattr__" in {name for _, name in asked.called}
    assert {name for _, name in asked.asked} == {"old_check"} and clean.asked == set()
    # The same holds for the real forwarders: asking for a compatibility name is recorded by name.
    real = guard.TextTrace(MARKER)
    with real:
        from core.metric_claims import percentage_comparisons
    assert callable(percentage_comparisons)
    assert forwarders_asked(real) == [("core/metric_claims.py", "percentage_comparisons")]


def test_the_import_guard_sees_a_planted_import_of_the_wording_based_checks(tmp_path):
    (tmp_path / "direct.py").write_text(
        "from openrook_tools.roleplay_eval.legacy_claims import extract_action_claims\n", encoding="utf-8")
    (tmp_path / "lazy.py").write_text(
        "def check(reply):\n    from openrook_tools.roleplay_eval import legacy_social\n"
        "    return legacy_social.has_role_language(reply)\n", encoding="utf-8")
    (tmp_path / "clean.py").write_text("from openrook_tools.roleplay_eval import scenarios\n", encoding="utf-8")
    assert legacy_importers(["direct.py", "lazy.py", "clean.py"], root=tmp_path) == [
        ("direct.py", None), ("lazy.py", "check")]


def test_the_wording_guard_flags_test_wording_and_not_ordinary_code_words(tmp_path):
    turn = next(text for text in sorted(scenario_phrases()[0])
                if len([word for word in re.findall(r"[^\\W\\d_]+", text) if len(word) >= WORD_LETTERS]) >= 4)
    long_words = [word.lower() for word in re.findall(r"[^\W\d_]+", turn) if len(word) >= WORD_LETTERS]
    assert WORDS_OF_A_SCENARIO == 2 and len(long_words) >= 2
    one, two = long_words[0], next(word for word in long_words[1:] if word != long_words[0])
    (tmp_path / "ordinary.py").write_text(
        f"import re\nTOPIC = re.compile(r'\\b(?:{one}|kilobyte)\\b', re.I)\n"
        f"ANY = re.compile(r'\\b\\w+\\b.{{0,80}}\\b{one}\\b', re.I)\nSHORT = ('the', 'and', 'you')\n",
        encoding="utf-8")
    assert scenario_wording([guard.module_unit("ordinary.py", root=tmp_path)]) == []
    (tmp_path / "fitted.py").write_text(
        f"import re\nRULE = re.compile(r'\\b{one}\\b.{{0,200}}\\b{two}\\b', re.I | re.S)\n", encoding="utf-8")
    found = scenario_wording([guard.module_unit("fitted.py", root=tmp_path)])
    assert {kind for _, kind, _ in found} == {"pattern matches"}
    assert scenarios.HOSTILE and scenarios.WARM

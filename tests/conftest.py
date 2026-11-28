"""Shared pytest fixtures.

``runtime.security.validate_http_url`` resolves non-literal hostnames via
DNS (``socket.getaddrinfo``) and rejects URLs whose addresses land in
loopback/link-local/private space — see the SSRF guard in
``runtime/security.py``. The test suite uses many fake hostnames
(``provider.example``, ``example.invalid``, ``prod-llm``...) and must never
depend on real DNS, so hostname resolution is stubbed to a fixed public
address by default.

Tests that exercise the resolution policy itself (tests/test_url_security.py)
restore the real helper and monkeypatch ``socket.getaddrinfo`` with crafted
answers; a test-level monkeypatch layers over this autouse stub and is
unwound in LIFO order, so both restore cleanly.

Runtime settings are isolated before test collection because importing the API
reads the config for CORS. Each test then receives its own temporary config.

The runtime reads every message with the model and has no keyword perception,
and so do pipelines built in tests: a fake backend that scripts no reading gives
an unavailable reading, exactly as in production. Tests that need a particular
reading script it (``tests/support/readings.py``).

Some older tests were written against the former keyword perception. They are
marked ``@pytest.mark.keyword_reader`` and read with the TEST-ONLY
``tests.support.keyword_reader`` for their duration. Runtime code never imports
that module.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pytest

from runtime.config import RuntimeConfig

# Tests that assert reply-check rules sub-phase 1b removed, in a test file that
# is being changed elsewhere and is therefore not edited here. Each asserts
# behaviour the runtime no longer has by design (docs/development/review-contract.md);
# they are skipped by name, with the rule they assert, until that file is updated.
_REMOVED_RULE = "asserts a reply-check rule removed in sub-phase 1b: "
SUPERSEDED_BY_THE_REVIEW_CONTRACT = {
    "tests/test_live_roleplay_p1j.py::test_misattributed_execution_findings_cannot_override_the_citation_contract":
        "expects the verdict unknown for a finding that quotes nothing; the review now asks once more, the "
        "scripted second answer passes, and the verdict is pass",
    "tests/test_live_roleplay_p1j.py::test_semantic_execution_finding_without_a_draft_span_retains_uncertainty":
        "expects delivered_uncertain for a finding that quotes nothing; the review now asks once more, the "
        "scripted second answer passes, and the reply is delivered",
    "tests/test_live_roleplay_p1j.py::test_exact_fabricated_output_is_exhausted_expression_failure":
        "an English pattern convicts the whole reply for attributed output, without the decision and without "
        "a review call (the structural check needs the decision, quotes the block and cuts it out)",
    "tests/test_live_roleplay_p1j.py::test_fabrication_is_corrected_from_the_actual_attempt_instead_of_its_grant":
        "an English pattern finds attributed output although the draft's scripted review passes "
        "(a review that ran to a verdict clears an output block)",
    "tests/test_live_roleplay_p1j.py::test_a_successful_different_command_does_not_verify_invented_output":
        "an English pattern finds attributed output that repeats the user's own text when the turn's only "
        "attempt succeeded (the user's text stops excusing a block only for an attempt that failed)",
    "tests/test_live_roleplay_p1j.py::test_exact_live_correction_cannot_misname_printf_as_the_blocked_echo_attempt":
        "an English pattern compares the command a denial names",
    "tests/test_live_roleplay_p1j.py::test_exact_execution_span_can_be_removed_leaving_independent_truth":
        "English claim verbs find an execution claim without the reviewer",
    "tests/test_live_roleplay_p1j.py::test_exact_cpu_idle_interpretation_is_removed_while_measurement_survives":
        "the CPU '100 minus idle' completeness rule",
    "tests/test_live_roleplay_p1j.py::test_user_action_in_negative_arc_is_not_an_offer_by_the_character":
        "the role noun list and offer patterns",
    "tests/test_live_roleplay_p1j.py::test_declined_conversation_is_not_a_granted_role_or_a_topic_obligation":
        "the former review payload (frozen_decision.declined)",
    "tests/test_live_roleplay_p1j.py::test_exact_persistence_boundary_is_locally_valid_and_review_sees_scoped_correction":
        "sentences of the former reviewer prompt",
    "tests/test_live_roleplay_p1j.py::test_exact_copied_insult_prefix_is_excised_without_a_rewrite":
        "a two-word opening counted as a copy (a copy now needs four words, so a greeting returned is not one)",
}


# Tests in the same file that assert the keyword routing sub-phase 1c removed
# (docs/development/request-routing.md). What a message asks for, and the
# operands of a tool it asks for, come from the reading now; a message nobody
# read commits no operation, whatever its words. The behaviour these tests
# wanted is covered with scripted readings in tests/test_request_routing.py.
_REMOVED_ROUTING = "asserts keyword routing removed in sub-phase 1c: "
SUPERSEDED_BY_REQUEST_ROUTING = {
    "tests/test_live_roleplay_p1j.py::test_literal_shell_operands_are_bound_before_choice_and_cannot_be_substituted":
        "expects an English pattern to bind a delimited shell command, also in a message that was not read; "
        "the reading reports the command, and only an exact part of the message is committed",
    "tests/test_live_roleplay_p1j.py::test_literal_shell_execution_uses_exact_operands_and_keeps_operator_gate":
        "runs a turn with no reading and expects the command to be recognised from its wording and executed; "
        "an unread message is plain conversation and runs nothing",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.nodeid.split("[", 1)[0]
        reason = SUPERSEDED_BY_THE_REVIEW_CONTRACT.get(name)
        if reason:
            item.add_marker(pytest.mark.skip(reason=_REMOVED_RULE + reason))
        reason = SUPERSEDED_BY_REQUEST_ROUTING.get(name)
        if reason:
            item.add_marker(pytest.mark.skip(reason=_REMOVED_ROUTING + reason))


_FAKE_PUBLIC_IP = "93.184.216.34"  # example.com — a public, non-blocked address


def _write_runtime_config(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "runtime_config.yaml"
    RuntimeConfig(data_dir=str(directory / "data")).write_yaml(str(path))
    return path


def pytest_configure(config):
    """Keep import-time API configuration away from the developer's files."""
    directory = tempfile.TemporaryDirectory(prefix="openrook-pytest-")
    config.add_cleanup(directory.cleanup)
    path = _write_runtime_config(Path(directory.name))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", str(path))
    config.add_cleanup(monkeypatch.undo)


@pytest.fixture(autouse=True)
def _isolated_runtime_config(tmp_path_factory, monkeypatch):
    """Specialized fixtures can override this config without sharing state."""
    # Keep fixture files outside tmp_path, which tests may treat as an empty workspace.
    path = _write_runtime_config(tmp_path_factory.mktemp("runtime-config"))
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", str(path))
    api = sys.modules.get("interface.api")
    if api is not None:
        monkeypatch.setattr(api, "RUNTIME_CONFIG_PATH", str(path))
    # Pipelines now materialize birth state even before the first learning
    # intake. Unit fixtures that omit a data directory must not share a subject
    # with another test or touch the developer's live identity store.
    from runtime import life_history
    original_path = life_history.life_history_db_path
    monkeypatch.setattr(life_history, "life_history_db_path", lambda config:
        str(path.parent / "data" / "shared" / "life_history.db")
        if config.data_dir == "data" else original_path(config))


@pytest.fixture(autouse=True)
def _stub_url_dns_resolution(monkeypatch):
    """Make URL validation DNS-independent for the whole suite."""
    monkeypatch.setattr(
        "runtime.security._resolve_hostname",
        lambda host, port: [_FAKE_PUBLIC_IP],
    )


@pytest.fixture(autouse=True)
def _keyword_reader_when_marked(request, monkeypatch):
    """TEST-ONLY reader for ``@pytest.mark.keyword_reader`` tests. See module docstring."""
    if request.node.get_closest_marker("keyword_reader"):
        from tests.support.keyword_reader import install

        install(monkeypatch)


@pytest.fixture(scope="class")
def keyword_reader_for_class():
    """The same TEST-ONLY reader for a class-scoped fixture that builds pipelines."""
    from tests.support.keyword_reader import install

    patch = pytest.MonkeyPatch()
    install(patch)
    yield
    patch.undo()

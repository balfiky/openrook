from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, PackagePath
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from core.appraisal import appraise_message_llm
from core.defense_mechanisms import DEFENSE_INSTRUCTIONS
from core.dual_process.generator import build_system_prompt
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from core.types import DetectedEmotion, PipelineContext
from evals.replay import RecordedBackend, ReplayRequestMismatch, replay_run
from runtime.config import RuntimeConfig
from runtime.life_history import (
    LifeHistoryStore,
    _llm_revision_verdicts,
    _try_llm_digest,
    experience_digestion_prompt,
)
from runtime.research_store import ResearchStore, encode
from tests.test_research_runtime import ResearchBackend, manager


def recorded_backend(*, native=False):
    request = (
        {
            "model": "fixture",
            "messages": [{"role": "user", "content": "original"}],
            "tools": [{"type": "function", "function": {"name": "inspect"}}],
            "tool_choice": "required",
            "temperature": 0,
        }
        if native
        else {
            "system": "original system",
            "input": "original input",
            "response_schema": {"type": "object"},
        }
    )
    prefix = "native_model_" if native else "model_"
    events = [
        {"id": "request", "kind": prefix + "request", "payload": copy.deepcopy(request)},
        {
            "id": "response",
            "kind": prefix + "response",
            "payload": {"call_id": "request", "response" if native else "output": "recorded"},
        },
    ]
    return RecordedBackend(events, native=native), request


@pytest.mark.parametrize("field", [None, "system", "input", "response_schema"])
def test_replay_verifies_each_text_request_field(field):
    backend, request = recorded_backend()
    if field:
        request[field] = {"type": "array"} if field == "response_schema" else "mutated"
        with pytest.raises(ReplayRequestMismatch):
            backend.generate(
                request["system"], request["input"], response_schema=request["response_schema"]
            )
        assert backend.index == 0
        assert backend.mismatches[0]["fields"] == [field]
    else:
        assert (
            backend.generate(
                request["system"], request["input"], response_schema=request["response_schema"]
            )
            == "recorded"
        )
        assert backend.mismatches == []


@pytest.mark.parametrize(
    "field", [None, "model", "messages", "tools", "tool_choice", "temperature"]
)
def test_replay_verifies_each_native_request_field(field):
    backend, request = recorded_backend(native=True)
    if field:
        if field == "messages":
            request[field][0]["content"] = "mutated"
        elif field == "tools":
            request[field][0]["function"]["name"] = "mutated"
        else:
            request[field] = 1 if field == "temperature" else "mutated"
        with pytest.raises(ReplayRequestMismatch):
            backend.create(**request)
        assert backend.index == 0
        assert backend.mismatches[0]["fields"] == [field]
    else:
        assert backend.create(**request) == "recorded"
        assert backend.mismatches == []


def test_replay_cannot_verify_an_outcome_without_its_request():
    backend = RecordedBackend(
        [{"kind": "model_response", "payload": {"call_id": "missing", "output": "recorded"}}]
    )
    with pytest.raises(ReplayRequestMismatch):
        backend.generate("system", "input")
    assert backend.mismatches[0]["missing_recorded_request"]


def test_optional_appraisal_cannot_hide_an_extra_unrecorded_request():
    backend, request = recorded_backend()
    backend.generate(
        request["system"], request["input"], response_schema=request["response_schema"]
    )
    assert appraise_message_llm("An extra call", DetectedEmotion(), backend) is None
    assert backend.mismatches[-1]["unrecorded_request"]


@pytest.mark.asyncio
async def test_replay_rejects_an_unfinished_recorded_request(tmp_path):
    runtime, unused = manager(tmp_path / "source")
    await runtime.handle_message("web", "alice", "chat", "Hello")
    await runtime.shutdown()
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "source")), initialize=False)
    store.observe("model_request", {"system": "interrupted", "input": "no outcome"})
    report = await replay_run(str(tmp_path / "source"), str(tmp_path / "replay"))
    assert not report["passed"]
    assert {"unconsumed_requests": {"model": 1, "native": 0}} in report["mismatches"]


@pytest.mark.asyncio
async def test_native_requests_replay_through_the_real_runner_without_external_calls(tmp_path):
    from core.types import ToolCapability, ToolCategory, ToolResult
    from openrook_tools.executor import ToolExecutor
    from openrook_tools.native_orchestrator import NativeToolCallRunner
    from openrook_tools.registry import ToolRegistry
    from runtime.sessions.manager import SessionManager
    from tests.test_native_orchestrator import FakeNativeClient

    client = FakeNativeClient(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "native-write",
                                    "type": "function",
                                    "function": {
                                        "name": "fs__write_file",
                                        "arguments": '{"path":"note.txt","content":"Recorded"}',
                                    },
                                },
                                {
                                    "id": "native-choice",
                                    "type": "function",
                                    "function": {
                                        "name": "control__no_tool",
                                        "arguments": '{"reason":"No action selected"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    )

    def tools():
        executor = ToolExecutor(ToolRegistry())
        executor.registry.register(
            ToolCapability(
                "fs.write_file", "Write a file", ToolCategory.WRITE,
                {
                    "path": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                },
            )
        )
        executor.register_handler(
            "fs.write_file",
            lambda arguments: ToolResult("fs.write_file", True, "Recorded"),
        )
        executor.set_tool_runner(
            NativeToolCallRunner(
                executor=executor,
                base_url="http://localhost:1/v1",
                model="native-fixture",
                client=client,
            )
        )
        return executor

    class NativeBackend(ResearchBackend):
        def generate(self, system_prompt, user_message):
            if '"primary_target"' in system_prompt:
                return '{"primary_target":"shared_problem","social_move":"request"}'
            return super().generate(system_prompt, user_message)

    runtime = SessionManager(
        RuntimeConfig(data_dir=str(tmp_path / "source"), research_mode=True, tools_enabled=True, research_birth_policy="configured"),
        backend_factory=NativeBackend,
        tool_executor_factory=tools,
    )
    try:
        await runtime.handle_message("web", "alice", "chat", "Write a note to note.md")
    finally:
        await runtime.shutdown()
    assert len(client.requests) == 1
    report = await replay_run(str(tmp_path / "source"), str(tmp_path / "replay"))
    assert report["passed"], report
    parameters = next(
        tool["function"]["parameters"]
        for tool in client.requests[0]["tools"]
        if tool["function"]["name"] == "fs__write_file"
    )
    assert list(parameters["properties"]) == ["content", "path"]
    assert parameters["required"] == ["content", "path"]
    assert report["native_outcomes_consumed"] == 1
    assert report["external_tools_executed"] == 0
    assert len(client.requests) == 1


@pytest.mark.parametrize("kind", ["daily", "study"])
def test_native_and_tool_execution_never_fall_back_after_resource_stop(tmp_path, monkeypatch, kind):
    from core.emotional_engine import EmotionalEngine
    from core.types import ToolCapability, ToolCategory
    from openrook_tools.executor import ToolExecutor
    from openrook_tools.native_orchestrator import NativeToolCallRunner
    from openrook_tools.registry import ToolRegistry

    error = (
        ResearchBudgetExceeded(10**12) if kind == "daily" else ResearchStudyLimit("Study stopped")
    )
    registry = ToolRegistry()
    registry.register(ToolCapability("fixture.stop", "Stop", ToolCategory.COGNITIVE, {}))
    executor = ToolExecutor(registry)
    executor.register_handler("fixture.stop", lambda arguments: (_ for _ in ()).throw(error))
    with pytest.raises(type(error)):
        executor.execute("fixture.stop", {})
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(error),
            )
        )
    )
    runner = NativeToolCallRunner(
        executor=executor, base_url="recorded", model="fixture", client=client
    )
    monkeypatch.setattr(
        runner,
        "_heuristic_fallback",
        lambda **kwargs: pytest.fail("Resource stops cannot trigger fallback actions"),
    )
    engine = EmotionalEngine()
    with pytest.raises(type(error)):
        runner.run_tool_loop(
            user_message="Consider an action",
            state=engine.state,
            person=None,
            defense_active=False,
            engine=engine,
            max_executions=1,
            autonomy_level="autonomous",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["system", "input"])
async def test_replay_reports_prompt_drift_even_if_optional_appraisal_absorbs_error(
    tmp_path, field
):
    runtime, _ = manager(tmp_path / "source")
    await runtime.handle_message("web", "alice", "chat", "Hello")
    await runtime.shutdown()
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "source")), initialize=False)
    with store.connection() as connection:
        row = connection.execute(
            "SELECT id,payload FROM observations WHERE kind='model_request' ORDER BY sequence LIMIT 1"
        ).fetchone()
        payload = json.loads(row["payload"])
        payload[field] += " altered request"
        connection.execute(
            "UPDATE observations SET payload=? WHERE id=?", (encode(payload), row["id"])
        )
    before = store.path.read_bytes()
    report = await replay_run(str(tmp_path / "source"), str(tmp_path / "replay"))
    assert not report["passed"]
    assert any(field in item["fields"] for item in report["request_mismatches"])
    assert report["external_tools_executed"] == 0
    assert store.path.read_bytes() == before


class RaisingBackend:
    def __init__(self, error):
        self.error = error

    def generate(self, *arguments, **keywords):
        raise self.error


@pytest.mark.parametrize("helper", ["appraisal", "digest", "revision"])
@pytest.mark.parametrize("kind", ["daily", "study", "timeout"])
def test_optional_helpers_preserve_resource_signals_and_timeout_fallback(helper, kind):
    error = {
        "daily": ResearchBudgetExceeded(10**12),
        "study": ResearchStudyLimit("Study resource limit: model_calls"),
        "timeout": requests.Timeout("transient failure"),
    }[kind]
    backend = RaisingBackend(error)

    def invoke():
        if helper == "appraisal":
            return appraise_message_llm("Hello", DetectedEmotion(), backend)
        if helper == "digest":
            return _try_llm_digest(
                title="Experience", text="Evidence", source_type="self_action", llm_client=backend
            )
        return _llm_revision_verdicts(
            backend, [{"key": "belief", "statement": "A belief"}], "Evidence"
        )

    if kind == "timeout":
        assert invoke() is None
    else:
        with pytest.raises(type(error)) as caught:
            invoke()
        assert caught.value is error


class StoppingBackend(ResearchBackend):
    def __init__(self, error, stage):
        super().__init__()
        self.error = error
        self.stage = stage

    def generate(self, system_prompt, user_message):
        if self.stage in system_prompt:
            self.calls.append((system_prompt, user_message))
            raise self.error
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["daily", "study", "runtime"])
async def test_generated_artifact_preserves_resource_stops_and_ordinary_failure(tmp_path, kind):
    from core.types import ToolCapability, ToolCategory, ToolResult
    from openrook_tools.executor import ToolExecutor
    from openrook_tools.registry import ToolRegistry
    from runtime.sessions.manager import SessionManager

    error = {
        "daily": ResearchBudgetExceeded(10**12),
        "study": ResearchStudyLimit("Study resource limit: model_calls"),
        "runtime": RuntimeError("Draft generation unavailable"),
    }[kind]

    class ArtifactBackend(StoppingBackend):
        def generate(self, system_prompt, user_message):
            if '"primary_target"' in system_prompt:
                self.calls.append((system_prompt, user_message))
                return '{"primary_target":"shared_problem","social_move":"request"}'
            return super().generate(system_prompt, user_message)

    backend = ArtifactBackend(error, "Generate one complete file")
    executor = ToolExecutor(ToolRegistry())
    executor.registry.register(ToolCapability("fs.write_file", "Write", ToolCategory.WRITE, {}))
    executor.register_handler(
        "fs.write_file", lambda arguments: pytest.fail("Failed drafts must not write files")
    )
    runtime = SessionManager(
        RuntimeConfig(
            data_dir=str(tmp_path), research_mode=True, tools_enabled=True,
            research_birth_policy="configured", autonomy_level="autonomous",
        ),
        backend_factory=lambda: backend,
        tool_executor_factory=lambda: executor,
    )
    try:
        session = await runtime.ensure_session("web", "alice", "chat")
        message = "Write a markdown document and save it as note.md"
        if kind == "runtime":
            await runtime.handle_message("web", "alice", "chat", message)
            trace = session.last_debug.tool_trace
            assert trace.final_decision.decision == "refuse"
            assert trace.executed_results == [ToolResult(
                tool_name="artifact.generate",
                success=False,
                output="",
                error=f"RuntimeError: {error}",
                side_effect_summary="artifact generation failed",
            )]
        else:
            with pytest.raises(type(error)) as caught:
                await runtime.handle_message("web", "alice", "chat", message)
            assert caught.value is error
            assert backend.calls[-1][0].startswith("Generate one complete file")
            assert "artifact generation failed" not in encode(runtime._agent.store.observations())
            assert not any(
                entry.event.metadata.get("tool_name") == "artifact.generate"
                for entry in session.pipeline.short_term.all()
            )
            state = runtime._agent.store.get("agent", "state")
            if kind == "daily":
                assert state["resume_at"] == error.resume_at
            else:
                assert state["study_stopped"] == str(error)
        assert sum(
            system.startswith("Generate one complete file") for system, _ in backend.calls
        ) == 1
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["daily", "study"])
@pytest.mark.parametrize("stage", ["strict JSON with keys: summary", '"primary_target"'])
async def test_resource_stop_reaches_owner_without_learning_retry_or_further_calls(
    tmp_path, kind, stage
):
    error = (
        ResearchBudgetExceeded(10**12)
        if kind == "daily"
        else ResearchStudyLimit("Study limit reached")
    )
    backend = StoppingBackend(error, stage)
    runtime, _ = manager(tmp_path, backend)
    try:
        if stage == '"primary_target"':
            with pytest.raises(type(error)):
                await runtime.handle_message("web", "alice", "chat", "Hello")
        else:
            runtime._agent.store.put(
                "learning",
                "pending",
                {
                    "id": "pending",
                    "title": "Experience",
                    "text": "Evidence",
                    "status": "pending",
                    "attempts": 0,
                    "source_type": "self_action",
                    "source_ref": "receipt",
                },
            )
            await runtime._proactive_sweep()
            pending = runtime._agent.store.get("learning", "pending")
            assert pending["status"] == "pending" and pending["attempts"] == 0
            assert "retry_at" not in pending
            assert not any(
                event["kind"] == "learning_deferred"
                for event in runtime._agent.store.observations()
            )
        state = runtime._agent.store.get("agent", "state")
        if kind == "daily":
            assert state["resume_at"] == error.resume_at
            assert not state.get("study_stopped")
        else:
            assert state["study_stopped"] == str(error)
        calls = len(backend.calls)
        await runtime._proactive_sweep()
        assert len(backend.calls) == calls
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("research", [False, True])
def test_defense_and_draft_are_descriptive_only_in_research(research):
    context = PipelineContext(
        research_mode=research,
        generator_prompt=Path("config/prompts/research_generator.md").read_text()
        if research
        else None,
        candidate_response="An earlier candidate",
        defense_instruction=DEFENSE_INSTRUCTIONS["minimization"],
        defense_state={"defense_type": "minimization", "raw_intensity": 0.9},
    )
    prompt = build_system_prompt(context)
    assert ("Use hedging language" in prompt) is not research
    assert ("Refine for tone and polish" in prompt) is not research
    assert ("Observed Defense State" in prompt) is research
    if research:
        assert '"raw_intensity": 0.9' in prompt
        assert "not a required expression" in prompt


@pytest.mark.asyncio
async def test_research_turn_carries_frozen_state_without_an_independent_defense_policy(tmp_path, monkeypatch):
    runtime, backend = manager(tmp_path)
    try:
        session = await runtime.ensure_session("web", "alice", "chat")
        monkeypatch.setattr(session.pipeline.defense_mechanism, "evaluate",
                            lambda **values: pytest.fail("A parallel defense policy ran"))
        await runtime.handle_message("web", "alice", "chat", "A difficult topic")
        prompts = [s for s, _ in backend.calls if s.startswith("Express the committed decision")]
        assert len(prompts) == 1
        assert "continuous_state" in prompts[0] and "selected_stance" in prompts[0]
        assert "Use hedging language" not in prompts[0]
        assert session.last_debug.turn_decision.scope == "user"
        assert session.last_debug.defense_activation is None
    finally:
        await runtime.shutdown()


def test_research_digest_uses_schema_only_and_hashes_exact_effective_prompt(tmp_path):
    backend = ResearchBackend()
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True)
    store = ResearchStore(config)
    manifest = store.initialize_run(config)
    with LifeHistoryStore(config) as identity:
        identity.ingest_external_text(
            title="An experience",
            text="A substantive encounter with uncertainty.",
            source_type="self_action",
            source_ref="receipt",
            llm_client=backend,
        )
    prompts = [
        system for system, unused in backend.calls if "strict JSON with keys: summary" in system
    ]
    assert prompts == [experience_digestion_prompt(research_mode=True)]
    assert "Example output" not in prompts[0]
    assert "Positive changes, negative changes, reversals, and no change" in prompts[0]
    assert '"name":"caution","delta":0.06' not in prompts[0]
    assert '"name":"caution","delta":0.06' in experience_digestion_prompt()
    assert (
        manifest["prompt_fingerprints"]["experience_digestion"]
        == hashlib.sha256(prompts[0].encode()).hexdigest()
    )


def test_installed_fingerprint_uses_only_distribution_owned_files(
    tmp_path, monkeypatch
):
    import runtime.research_store as storage

    owned = [
        "runtime/research_store.py",
        "main.py",
        "config/prompts/research_generator.md",
        "config/soul.yaml",
        "interface/static/index.html",
    ]
    extra = [
        "unrelated_package/module.py",
        "openrook-0.37.0.dist-info/METADATA",
        "runtime/__pycache__/cache.pyc",
    ]
    for relative in owned + extra:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    installed = SimpleNamespace(
        files=[PackagePath(relative) for relative in owned + extra[1:]],
        locate_file=lambda path: tmp_path / path,
    )

    def distribution(name):
        assert name == "openrook"
        return installed

    monkeypatch.setattr(storage, "__file__", str(tmp_path / "runtime/research_store.py"))
    monkeypatch.setattr(storage, "distribution", distribution)
    monkeypatch.setattr(
        storage.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Installed ownership must not use git"),
    )
    original = storage._code_fingerprint()
    assert set(original) == set(owned)
    (tmp_path / extra[0]).write_text("unrelated change")
    assert storage._code_fingerprint() == original
    (tmp_path / owned[2]).write_text("changed prompt")
    assert storage._code_fingerprint()[owned[2]] != original[owned[2]]


def test_archive_fingerprint_without_git_does_not_scan_neighbor_packages(tmp_path, monkeypatch):
    import runtime.research_store as storage

    for relative in ("runtime/research_store.py", "main.py", "unrelated_package/module.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    monkeypatch.setattr(storage, "__file__", str(tmp_path / "runtime/research_store.py"))
    monkeypatch.setattr(
        storage, "distribution", lambda name: (_ for _ in ()).throw(PackageNotFoundError(name))
    )
    monkeypatch.setattr(
        storage.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    assert set(storage._code_fingerprint()) == {"runtime/research_store.py", "main.py"}


def test_action_catalog_prompt_is_stable_across_schema_serialization():
    from core.self_intent import _render_catalog

    catalog = [
        {
            "name": "world.act",
            "arg_schema": {
                "request_id": {"type": "string", "required": True},
                "project": {"type": "string", "required": True},
                "strategy": {"type": "string", "required": True},
                "partner": {"type": "string"},
            },
        }
    ]
    restored = json.loads(encode(catalog))
    assert list(catalog[0]["arg_schema"]) != list(restored[0]["arg_schema"])
    assert _render_catalog(catalog) == _render_catalog(restored)

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evals.research import run_protocol, validate_protocol
from evals.research_job import run_job, _check_background_health
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore, _code_fingerprint, encode
from tests.test_research_runtime import ResearchBackend


def test_research_modules_import_without_the_test_package():
    script = """
import importlib.abc
import sys
class NoTests(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tests' or fullname.startswith('tests.'):
            raise ModuleNotFoundError('Test helpers are not shipped')
sys.meta_path.insert(0, NoTests())
import evals.research_job
import evals.replay
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_health_guard_counts_failed_cycles_not_failure_events(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "replicate_1/condition_1")))
    sources = {}
    for cycle in range(3):
        store.observe("autonomous_decision", {"action": "pursue"})
        store.observe("autonomous_failure", {"error": "server unavailable"})
        store.observe("replay_frame", {"kind": "tick", "error": "server unavailable"})
        problem = _check_background_health(tmp_path, sources)
        assert bool(problem) == (cycle == 2)
    assert next(iter(sources.values()))["consecutive_failures"] == 3


def test_health_guard_allows_inactivity_and_resets_after_success(tmp_path):
    store = ResearchStore(RuntimeConfig(data_dir=str(tmp_path / "replicate_1/condition_1")))
    sources = {}
    for _cycle in range(4):
        store.observe("autonomous_failure", {"error": "temporary"})
        store.observe("replay_frame", {"kind": "tick", "error": None})
        store.observe("autonomous_decision", {"action": "wait"})
        store.observe("replay_frame", {"kind": "tick", "error": None})
        store.observe("replay_frame", {"kind": "tick", "error": None})
        assert _check_background_health(tmp_path, sources) is None
    assert next(iter(sources.values()))["consecutive_failures"] == 0
    assert next(iter(sources.values()))["total_failed_cycles"] == 4


@pytest.mark.asyncio
async def test_health_guard_stops_and_records_a_failed_job(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evals.research_job._check_background_health", lambda *args: "repeated failure"
    )
    with pytest.raises(RuntimeError, match="repeated failure"):
        await run_job(
            RuntimeConfig(research_mode=True),
            {"scheduler": "manual", "steps": [{"kind": "wait", "seconds": 60}]},
            str(tmp_path / "job"),
            maximum_runtime=10,
        )
    status = json.loads((tmp_path / "job/job.json").read_text())
    assert status["status"] == "failed"


@pytest.mark.asyncio
async def test_unattended_job_validates_offline_and_finishes(tmp_path):
    result = await run_job(
        RuntimeConfig(research_mode=True),
        {"scheduler": "manual", "steps": [{"kind": "tick"}]},
        str(tmp_path / "job"),
        maximum_runtime=20,
        backend_factory=lambda config: ResearchBackend(),
        tool_executor_factory=lambda config: None,
    )
    status = json.loads((tmp_path / "job/job.json").read_text())
    assert result["passed"], result
    assert status["status"] == "completed"
    assert status["backend_factory_injected"]
    assert result["replays"][0]["external_tools_executed"] == 0


@pytest.mark.asyncio
async def test_unattended_deadline_leaves_a_failure_record(tmp_path):
    with pytest.raises(TimeoutError):
        await run_job(
            RuntimeConfig(research_mode=True),
            {"scheduler": "manual", "steps": [{"kind": "wait", "seconds": 60}]},
            str(tmp_path / "job"),
            maximum_runtime=0.05,
        )
    status = json.loads((tmp_path / "job/job.json").read_text())
    assert status["status"] == "failed"
    assert "TimeoutError" in status["error"]


@pytest.mark.parametrize("duration", [-1, float("nan"), True])
def test_wait_until_requires_a_finite_nonnegative_duration(duration):
    with pytest.raises(ValueError):
        validate_protocol({"steps": [{"kind": "wait_until", "seconds": duration}]})


@pytest.mark.asyncio
async def test_autonomous_contact_is_recorded_and_blinded(tmp_path):
    backend = ResearchBackend(
        decisions=[
            {"action": "wait", "reason": "Observe"},
            {
                "action": "contact",
                "session_key": "research:participant:conversation",
                "message": "My selected contact",
                "reason": "Continue a topic",
            },
        ]
    )
    report = await run_protocol(
        RuntimeConfig(research_mode=True),
        {
            "scheduler": "manual",
            "steps": [
                {"kind": "tick"},
                {"kind": "message", "text": "Hello"},
                {"kind": "tick"},
            ],
        },
        str(tmp_path / "study"),
        backend_factory=lambda config: backend,
        tool_executor_factory=lambda config: None,
    )
    samples = json.loads((tmp_path / "study/blind_samples.json").read_text())
    key = json.loads((tmp_path / "study/blinding_key.json").read_text())
    assert any(sample.get("response") == "My selected contact" for sample in samples)
    assert any(item["origin"] == "autonomous_contact" for item in key)
    store = ResearchStore(RuntimeConfig(data_dir=report["runs"][0]["data_dir"]), initialize=False)
    assert store.list("delivery")[0]["status"] == "delivered"


def test_environment_key_authenticates_native_calls_without_entering_manifest(
    tmp_path, monkeypatch
):
    import runtime.tools as tool_module

    monkeypatch.setenv("LLM_API_KEY", "environment-test-secret")
    captured = {}

    def native_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(tool_module, "NativeToolCallRunner", native_runner)
    config = RuntimeConfig(
        data_dir=str(tmp_path),
        tools_enabled=True,
        llm_base_url="http://localhost:11434/v1",
        llm_model="test",
    )
    executor = tool_module.create_tool_executor(config)
    try:
        assert captured["api_key"] == "environment-test-secret"
        manifest = ResearchStore(config).initialize_run(config)
        assert "environment-test-secret" not in encode(manifest)
        assert (
            manifest["credential_fingerprints"]["llm_api_key"]
            == hashlib.sha256(b"environment-test-secret").hexdigest()
        )
    finally:
        for resource in executor.owned_resources:
            if hasattr(resource, "close"):
                resource.close()


def test_installed_source_has_code_and_prompt_fingerprints(tmp_path, monkeypatch):
    import runtime.research_store as module

    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime/module.py").write_text("value = 1")
    (tmp_path / "config/prompts").mkdir(parents=True)
    (tmp_path / "config/prompts/research_generator.md").write_text("A frozen prompt")
    (tmp_path / "prompt.md").write_text("An unrelated neighboring prompt")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "runtime/research_store.py"))
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=b"")
    )
    fingerprints = _code_fingerprint()
    assert set(fingerprints) == {"runtime/module.py", "config/prompts/research_generator.md"}

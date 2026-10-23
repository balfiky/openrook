"""Soul reads, saves and evaluation recovery stay in the selected workspace."""

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
import httpx
import pytest

from config import loader
from interface import api as api_mod
from interface.routes_admin_config import router as config_router
from interface.routes_admin_identity import router as identity_router
from openrook_tools.roleplay_eval.api import PublicAPI, Redactor
from openrook_tools.roleplay_eval.runner import Evaluation, Options
from openrook_tools.roleplay_eval.scenarios import Scenario, Turn, profile
from runtime.config import RuntimeConfig


ORIGINAL = (
    b"# Keep this comment, order, precision and CRLF.\r\n"
    b"soul:\r\n  name: Original\r\n  schema_version: 2\r\n"
    b"  drives: {engagement: 0.50}\r\n  dispositions: {pride: 0.50}\r\n"
    b"  identity: >\r\n    A privately authored identity.\r\n"
)


@pytest.fixture
def soul_runtime(tmp_path, monkeypatch):
    runtime_path = tmp_path / "external" / "runtime_config.yaml"
    RuntimeConfig(data_dir=str(tmp_path / "original-data")).write_yaml(str(runtime_path))
    packaged = tmp_path / "package" / "config"
    packaged.mkdir(parents=True)
    (packaged / "soul.yaml").write_bytes(ORIGINAL)
    monkeypatch.setattr(loader, "_DEFAULT_CONFIG_DIR", packaged)
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", str(runtime_path))
    monkeypatch.delenv("OPENROOK_CONFIG_DIR", raising=False)
    monkeypatch.setattr(api_mod, "RUNTIME_CONFIG_PATH", str(runtime_path))
    monkeypatch.setattr(api_mod, "_session_manager", None)
    monkeypatch.setattr(api_mod, "_pipeline_override", None)
    monkeypatch.setattr(api_mod, "_restart_telegram_channel", AsyncMock())
    monkeypatch.setattr(api_mod, "_refresh_live_channel_session_manager", lambda: None)
    monkeypatch.setattr("openrook_tools.roleplay_eval.runner.tempfile", SimpleNamespace(
        gettempdir=lambda: str(tmp_path / "private"),
    ))
    loader.reset_config()
    app = FastAPI()
    app.include_router(config_router)
    app.include_router(identity_router)

    @app.get("/v1/ready")
    async def ready():
        return {"status": "ready"}

    yield SimpleNamespace(path=runtime_path, packaged=packaged, app=app)
    loader.reset_config()


@pytest.mark.parametrize("source", ["packaged", "sibling", "environment", "explicit"])
def test_staging_and_resumed_restoration_preserve_exact_source_bytes(
    soul_runtime, tmp_path, monkeypatch, source,
):
    runtime = RuntimeConfig.from_yaml(str(soul_runtime.path))
    original_path = soul_runtime.packaged / "soul.yaml"
    operator_dir = soul_runtime.path.parent / "config"
    if source in {"environment", "explicit"}:
        operator_dir = tmp_path / "operator-config"
        monkeypatch.setenv("OPENROOK_CONFIG_DIR", str(operator_dir))
    if source in {"sibling", "environment"}:
        original_path = operator_dir / "soul.yaml"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_bytes(ORIGINAL)
    if source == "explicit":
        original_path = tmp_path / "authored" / "identity.yaml"
        original_path.parent.mkdir()
        original_path.write_bytes(ORIGINAL)
        runtime.soul_path = str(original_path)
        runtime.write_yaml(str(soul_runtime.path))
    (operator_dir / "prompts").mkdir(parents=True)
    (operator_dir / "prompts" / "generator.md").write_text("Operator prompt")
    before = original_path.stat()
    expected_digest = hashlib.sha256(ORIGINAL).hexdigest()

    async def exercise():
        api = PublicAPI("http://runtime.invalid", transport=httpx.ASGITransport(app=soul_runtime.app))
        scenarios = [Scenario("first", "F", (Turn("hello"),), personality="proud"),
                     Scenario("second", "F", (Turn("hello"),), personality="patient")]
        out, data_root = tmp_path / "report", str(tmp_path / "run-data")
        evaluation = Evaluation(api, scenarios, Options(out, data_root=data_root), Redactor())
        try:
            await evaluation._capture_original()
            assert base64.b64decode(evaluation.original["soul_file"]["content_base64"]) == ORIGINAL
            for scenario in scenarios:
                await evaluation._stage(scenario)
                staged_path = Path(data_root) / scenario.id / "config" / "soul.yaml"
                assert RuntimeConfig.from_yaml(str(soul_runtime.path)).soul_path == str(staged_path)
                assert loader.resolve_soul_path() == staged_path
                assert loader.get_config().soul.dispositions == profile(scenario.personality, 0)["dispositions"]
                assert loader.get_config().generator_prompt == "Operator prompt"
                assert original_path.read_bytes() == ORIGINAL
                assert original_path.stat().st_mtime_ns == before.st_mtime_ns
                assert original_path.stat().st_ino == before.st_ino
                assert (soul_runtime.packaged / "soul.yaml").read_bytes() == ORIGINAL

            # Simulate the loader and harness restarting against persisted staging.
            loader.reset_config()
            assert loader.get_config().soul.dispositions == profile("patient", 0)["dispositions"]
            resumed = Evaluation(api, scenarios, Options(out, resume=True, data_root=data_root), Redactor())
            await resumed._capture_original()
            assert await resumed.restore()
            assert resumed.manifest["soul_file_restoration"] == {
                "original_sha256": expected_digest, "restored_sha256": expected_digest, "verified": True,
            }
            assert RuntimeConfig.from_yaml(str(soul_runtime.path)).soul_path == runtime.soul_path
            assert loader.resolve_soul_path() == original_path
            assert original_path.read_bytes() == ORIGINAL
            assert original_path.stat().st_mtime_ns == before.st_mtime_ns
            assert original_path.stat().st_ino == before.st_ino
            assert loader.get_config().soul.name == "Original"
            summary = resumed.artifacts.report(scenarios, resumed.manifest)
            assert summary["soul_file_restoration"]["verified"] is True
            public = "".join(path.read_text() for path in out.iterdir() if path.is_file())
            assert "content_base64" not in public and "privately authored" not in public
            journal = json.loads(resumed.recovery_path.read_text())
            assert base64.b64decode(journal["soul_file"]["content_base64"]) == ORIGINAL
        finally:
            await api.close()

    asyncio.run(exercise())


def test_external_runtime_config_scopes_saves_and_loader_without_env_override(soul_runtime):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=soul_runtime.app), base_url="http://runtime.invalid",
        ) as client:
            response = await client.post("/admin/soul", json={"name": "Workspace identity"})
            assert response.status_code == 200
            target = soul_runtime.path.parent / "config" / "soul.yaml"
            assert response.json()["path"] == str(target)
            assert target.exists()
            assert loader.get_config().soul.name == "Workspace identity"
            snapshot = (await client.get("/admin/soul")).json()["file"]
            assert base64.b64decode(snapshot["content_base64"]) == target.read_bytes()
            assert snapshot["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
            assert (soul_runtime.packaged / "soul.yaml").read_bytes() == ORIGINAL

    asyncio.run(exercise())


def test_raw_soul_snapshot_requires_configured_bearer(soul_runtime):
    runtime = RuntimeConfig.from_yaml(str(soul_runtime.path))
    runtime.api_key = "test-bearer"
    runtime.write_yaml(str(soul_runtime.path))

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=soul_runtime.app), base_url="http://runtime.invalid",
        ) as client:
            denied = await client.get("/admin/soul")
            assert denied.status_code == 401 and "content_base64" not in denied.text
            allowed = await client.get("/admin/soul", headers={"Authorization": "Bearer test-bearer"})
            assert allowed.status_code == 200
            assert base64.b64decode(allowed.json()["file"]["content_base64"]) == ORIGINAL

    asyncio.run(exercise())


def test_missing_original_soul_is_restored_as_absence(soul_runtime, tmp_path):
    original_path = soul_runtime.packaged / "soul.yaml"
    original_path.unlink()

    async def exercise():
        api = PublicAPI("http://runtime.invalid", transport=httpx.ASGITransport(app=soul_runtime.app))
        evaluation = Evaluation(api, [], Options(tmp_path / "report"), Redactor())
        try:
            await evaluation._capture_original()
            assert evaluation.original["soul_file"]["exists"] is False
            await evaluation._stage()
            assert loader.get_config().soul.name
            assert await evaluation.restore()
            assert not original_path.exists()
            assert not (soul_runtime.path.parent / "config" / "soul.yaml").exists()
            assert evaluation.manifest["soul_file_restoration"] == {
                "original_sha256": None, "restored_sha256": None, "verified": True,
            }
        finally:
            await api.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("launcher", ["web", "console"])
def test_config_argument_updates_identity_resolution_after_import(
    soul_runtime, tmp_path, monkeypatch, launcher,
):
    # Both entrypoints import cognitive config before parsing --config.
    monkeypatch.setenv("OPENROOK_RUNTIME_CONFIG", str(tmp_path / "unselected.yaml"))
    assert loader.get_config().soul.name == "Original"
    selected = soul_runtime.path.parent / "config" / "soul.yaml"
    selected.parent.mkdir()
    selected.write_bytes(ORIGINAL.replace(b"Original", b"Selected"))
    runtime = RuntimeConfig.from_yaml(str(soul_runtime.path))
    runtime.llm_backend = "codex"
    runtime.write_yaml(str(soul_runtime.path))
    monkeypatch.setattr(sys, "argv", ["openrook", "--config", str(soul_runtime.path)])
    if launcher == "web":
        import uvicorn
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
        api_mod.main()
        assert api_mod._soul_yaml_path() == str(selected)
    else:
        import main as console_main
        monkeypatch.setattr(console_main, "OpenRookApp", lambda config: SimpleNamespace(run=AsyncMock()))
        console_main.main()
    assert loader.get_config().soul.name == "Selected"

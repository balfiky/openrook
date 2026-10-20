"""Exercise storage upgrades with real SQLite stores and session lifecycles."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest

from core.memory.long_term import LongTermMemory
from core.memory.semantic import SQLiteSemanticMemory
from core.types import LongTermEntry, ModulatorState, SemanticMemoryEntry
from interface.v1 import build_v1_router
from runtime import storage
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from tests._fakes import MockLLMBackend


def seed_legacy(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / storage.LEGACY_USER_DB_NAME
    with closing(LongTermMemory(str(path))) as memory:
        memory.store(LongTermEntry(
            summary="A memory from before the upgrade.", source_person="alice", confidence=1,
        ))
    with closing(SQLiteSemanticMemory(str(path))) as semantic:
        semantic.store(SemanticMemoryEntry(
            kind="preference", summary="Alice prefers tea.", content="tea",
            source_person="alice", confidence=1,
        ))
    return path


def summaries(path: Path) -> list[str]:
    with closing(sqlite3.connect(path)) as connection:
        return [row[0] for row in connection.execute("SELECT summary FROM memories ORDER BY id")]


def test_fresh_install_resolves_current_name_without_creating_database(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    path = Path(config.user_db_path("web:alice"))
    assert path == tmp_path / "web_alice" / "openrook.db"
    assert not path.exists()
    assert not path.parent.exists()


def test_legacy_snapshot_preserves_source_and_is_idempotent(tmp_path, caplog):
    config = RuntimeConfig(data_dir=str(tmp_path))
    legacy = seed_legacy(tmp_path / "web_alice")
    original = legacy.read_bytes()
    current = Path(config.user_db_path("web:alice"))
    assert summaries(current) == summaries(legacy)
    assert legacy.read_bytes() == original
    inode = current.stat().st_ino
    assert config.user_db_path("web:alice") == str(current)
    assert current.stat().st_ino == inode
    assert "preserving" in caplog.text


@pytest.fixture
def storage_without_result_constants(monkeypatch):
    for name in ("SQLITE_BUSY", "SQLITE_LOCKED"):
        monkeypatch.delattr(sqlite3, name, raising=False)
    # Reload to exercise the import-time fallbacks as well as the callback.
    return importlib.reload(storage)


def test_legacy_snapshot_without_sqlite_result_constants(tmp_path, storage_without_result_constants):
    legacy = seed_legacy(tmp_path / "web_alice")
    original = legacy.read_bytes()
    current = Path(storage_without_result_constants.resolve_user_db_path(str(legacy.parent)))
    assert summaries(current) == summaries(legacy)
    assert legacy.read_bytes() == original
    assert not list(legacy.parent.glob(".openrook-migration-*"))


@pytest.mark.parametrize("status", [5, 6], ids=["busy", "locked"])
def test_backup_timeout_without_sqlite_result_constants(
    tmp_path, monkeypatch, storage_without_result_constants, status,
):
    legacy = seed_legacy(tmp_path / "web_alice")
    original = legacy.read_bytes()
    retries = []

    class BlockedConnection(sqlite3.Connection):
        def backup(self, destination, *, progress):
            progress(status, 1, 2)
            retries.append(status)
            progress(status, 1, 2)

    connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *args, **kwargs: connect(
            *args, factory=BlockedConnection, **kwargs,
        ),
    )
    ticks = iter([0, 1, 11])
    monkeypatch.setattr(
        storage_without_result_constants, "time", SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    with pytest.raises(TimeoutError, match="migration blocked"):
        storage_without_result_constants.resolve_user_db_path(str(legacy.parent))
    assert retries == [status]
    assert legacy.read_bytes() == original
    assert not legacy.with_name("openrook.db").exists()
    assert not list(legacy.parent.glob(".openrook-migration-*"))


def test_both_present_uses_current_without_overwriting_either(tmp_path, caplog):
    config = RuntimeConfig(data_dir=str(tmp_path))
    legacy = seed_legacy(tmp_path / "web_alice")
    current = legacy.with_name("openrook.db")
    with closing(LongTermMemory(str(current))) as memory:
        memory.store(LongTermEntry(summary="Current memory", confidence=1))
    before = (current.read_bytes(), legacy.read_bytes())
    assert config.user_db_path("web:alice") == str(current)
    assert (current.read_bytes(), legacy.read_bytes()) == before
    assert "without merging" in caplog.text


def test_wal_commits_survive_with_legacy_connection_still_open(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    legacy = seed_legacy(tmp_path / "web_alice")
    with closing(LongTermMemory(str(legacy))) as memory:
        memory._conn.execute("PRAGMA journal_mode=WAL")
        memory._conn.execute("PRAGMA wal_autocheckpoint=0")
        memory.store(LongTermEntry(summary="Committed in WAL", confidence=1))
        sidecars = [Path(str(legacy) + suffix) for suffix in ("-wal", "-shm")]
        assert all(path.exists() for path in sidecars)
        current = Path(config.user_db_path("web:alice"))
        assert summaries(current) == ["A memory from before the upgrade.", "Committed in WAL"]
        assert all(path.exists() for path in sidecars)
        # A self-contained snapshot needs no legacy sidecars to be readable.
        restored = tmp_path / "snapshot.db"
        shutil.copy2(current, restored)
        assert summaries(restored) == summaries(current)


def test_concurrent_resolvers_open_one_complete_database(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    seed_legacy(tmp_path / "web_alice")

    def open_store(_):
        path = Path(config.user_db_path("web:alice"))
        return path.stat().st_ino, summaries(path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(open_store, range(24)))
    assert len({inode for inode, _ in results}) == 1
    assert all(rows == ["A memory from before the upgrade."] for _, rows in results)


def test_concurrent_processes_publish_without_clobbering(tmp_path):
    seed_legacy(tmp_path / "web_alice")
    script = """
import sys
from runtime.config import RuntimeConfig
from core.memory.long_term import LongTermMemory
from core.types import LongTermEntry
path = RuntimeConfig(data_dir=sys.argv[1]).user_db_path("web:alice")
memory = LongTermMemory(path)
memory.store(LongTermEntry(summary=sys.argv[2], confidence=1))
memory.close()
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), f"process-{index}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for index in range(4)
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
    assert set(summaries(tmp_path / "web_alice" / "openrook.db")) == {
        "A memory from before the upgrade.", *(f"process-{index}" for index in range(4)),
    }


def test_failed_publication_leaves_source_and_no_current_database(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=str(tmp_path))
    legacy = seed_legacy(tmp_path / "web_alice")
    original = legacy.read_bytes()

    def fail(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="simulated"):
        config.user_db_path("web:alice")
    assert legacy.read_bytes() == original
    assert not legacy.with_name("openrook.db").exists()
    assert not list(legacy.parent.glob(".openrook-migration-*"))


def test_failed_snapshot_flush_preserves_source_without_publishing(tmp_path, monkeypatch):
    legacy = seed_legacy(tmp_path / "web_alice")
    original = legacy.read_bytes()

    def fail(*args):
        raise OSError("simulated flush failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="flush"):
        RuntimeConfig(data_dir=str(tmp_path)).user_db_path("web:alice")
    assert legacy.read_bytes() == original
    assert not legacy.with_name("openrook.db").exists()
    assert not list(legacy.parent.glob(".openrook-migration-*"))


def test_late_snapshot_never_overwrites_a_competing_published_write(tmp_path, monkeypatch):
    legacy = seed_legacy(tmp_path / "web_alice")
    real_link = os.link

    def competing_publish(source, destination):
        # Another resolver publishes and its first session writes while this
        # resolver still holds its older, already completed snapshot.
        shutil.copy2(source, destination)
        with closing(LongTermMemory(str(destination))) as memory:
            memory.store(LongTermEntry(summary="Newly published write", confidence=1))
        real_link(source, destination)  # raises FileExistsError, just like the real race

    monkeypatch.setattr(os, "link", competing_publish)
    current = Path(RuntimeConfig(data_dir=str(tmp_path)).user_db_path("web:alice"))
    assert summaries(current) == ["A memory from before the upgrade.", "Newly published write"]
    assert summaries(legacy) == ["A memory from before the upgrade."]


def test_corrupt_legacy_fails_without_publishing_an_empty_database(tmp_path):
    directory = tmp_path / "web_alice"
    directory.mkdir()
    legacy = directory / storage.LEGACY_USER_DB_NAME
    legacy.write_bytes(b"existing unreadable data")
    with pytest.raises(sqlite3.DatabaseError):
        RuntimeConfig(data_dir=str(tmp_path)).user_db_path("web:alice")
    assert legacy.read_bytes() == b"existing unreadable data"
    assert not (directory / "openrook.db").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_orphan_current_sidecars_are_preserved_and_block_migration(tmp_path, suffix):
    legacy = seed_legacy(tmp_path / "web_alice")
    sidecar = legacy.with_name("openrook.db" + suffix)
    sidecar.write_bytes(b"potential recovery data")
    with pytest.raises(FileExistsError, match="sidecar"):
        RuntimeConfig(data_dir=str(tmp_path)).user_db_path("web:alice")
    assert sidecar.read_bytes() == b"potential recovery data"
    assert not legacy.with_name("openrook.db").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("research_mode", [False, True])
async def test_real_sessions_read_migrated_stores_and_research_copy(tmp_path, research_mode):
    legacy = seed_legacy(tmp_path / "web_alice")
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=research_mode)
    manager = SessionManager(config, backend_factory=MockLLMBackend)
    try:
        sessions = await asyncio.gather(*[
            manager._get_or_create(f"web:alice:{chat}", "web:alice", "alice")
            for chat in ("direct", "group")
        ])
        for session in sessions:
            memory = session.pipeline.long_term.retrieve(ModulatorState(), source_person="alice")
            assert [item.summary for item in memory] == ["A memory from before the upgrade."]
            semantic = session.pipeline.semantic_memory.retrieve("tea", source_person="alice")
            assert [item.summary for item in semantic] == ["Alice prefers tea."]
            db = session.pipeline.long_term._conn.execute("PRAGMA database_list").fetchone()[2]
            assert Path(db).name == "openrook.db"
        assert legacy.exists()
        if research_mode:
            # Research/replay snapshots copy arbitrary .db names. Restoring a
            # pre-migration initial-state directory must still upgrade on open.
            snapshot = (
                tmp_path / "shared" / "replay_initial" / "web_alice" / storage.LEGACY_USER_DB_NAME
            )
            assert summaries(snapshot) == ["A memory from before the upgrade."]
            replay_root = tmp_path / "restored"
            shutil.copytree(snapshot.parent, replay_root / "web_alice")
            replay_path = RuntimeConfig(data_dir=str(replay_root)).user_db_path("web:alice")
            assert summaries(Path(replay_path)) == ["A memory from before the upgrade."]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "current", "both", "corrupt_legacy"])
async def test_delete_endpoint_removes_both_names_sidecars_and_unreadable_legacy(tmp_path, mode):
    legacy = seed_legacy(tmp_path / "web_alice")
    current = legacy.with_name("openrook.db")
    if mode in {"current", "both"}:
        shutil.copy2(legacy, current)
    if mode == "current":
        legacy.unlink()
    if mode == "corrupt_legacy":
        legacy.write_bytes(b"unreadable legacy database must still be deletable")
    for path in (legacy, current):
        if path.exists():
            for suffix in ("-wal", "-shm"):
                Path(str(path) + suffix).touch()
    untouched = seed_legacy(tmp_path / "web_other")
    config = RuntimeConfig(data_dir=str(tmp_path))
    manager = SessionManager(config, backend_factory=MockLLMBackend)
    app = FastAPI()
    app.include_router(build_v1_router(
        session_manager_getter=lambda: manager,
        config_getter=lambda: config,
        serve_started_at=0,
    ))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver",
        ) as client:
            response = await client.delete("/v1/users/web/alice")
        assert response.status_code == 200, response.text
        assert response.json()["deleted"] is True
        assert not legacy.parent.exists()
        assert untouched.exists()
    finally:
        await manager.shutdown()

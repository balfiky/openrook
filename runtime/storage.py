"""Resolve per-user storage, preserving existing installations during upgrades."""

from __future__ import annotations

from contextlib import closing
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time


log = logging.getLogger(__name__)
USER_DB_NAME = "openrook.db"
LEGACY_USER_DB_NAME = "nur.db"
_migration_lock = threading.Lock()


def existing_user_db_path(directory: str) -> str:
    """Choose a path for inspection/deletion without triggering a migration."""
    current = Path(directory) / USER_DB_NAME
    if os.path.lexists(current):
        return str(current)
    return str(Path(directory) / LEGACY_USER_DB_NAME)


def resolve_user_db_path(directory: str) -> str:
    """Publish a consistent SQLite snapshot before the first current store opens.

    Session creation already holds the manager lock. This lock also covers
    direct callers and multiple managers in one process. Atomic, no-clobber
    publication arbitrates competing processes without ever replacing a DB
    another process may already have opened.

    SQLite's backup API includes committed WAL pages; moving the main file and
    its sidecars separately cannot offer the same crash safety. Keep the source
    as a recovery copy. Operators must stop old servers before upgrading:
    subsequent writes by old code are not mirrored into the new database.
    """
    current = Path(directory) / USER_DB_NAME
    legacy = Path(directory) / LEGACY_USER_DB_NAME
    with _migration_lock:
        if os.path.lexists(current):
            if legacy.exists():
                log.warning(
                    "Both current and legacy user databases exist in %s; "
                    "using %s and preserving %s without merging",
                    directory, current.name, legacy.name,
                )
            return str(current)
        if not legacy.exists():
            return str(current)
        for suffix in ("-wal", "-shm", "-journal"):
            if os.path.lexists(str(current) + suffix):
                raise FileExistsError(
                    f"Refusing migration beside an existing database sidecar: {current}{suffix}"
                )

        # A failed copy must never make an empty or incomplete current database
        # visible. A same-directory temporary file also shares the filesystem.
        descriptor, temporary = tempfile.mkstemp(prefix=".openrook-migration-", dir=directory)
        os.close(descriptor)
        try:
            deadline = time.monotonic() + 10

            def check_backup_progress(status: int, remaining: int, total: int) -> None:
                if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("User database migration blocked; stop the older writer")

            with closing(sqlite3.connect(legacy.resolve().as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(temporary)) as destination:
                    source.backup(destination, progress=check_backup_progress)
                    # Make the published snapshot self-contained, even when
                    # the source's persistent journal mode was WAL.
                    destination.execute("PRAGMA journal_mode=DELETE")
            with open(temporary, "rb") as snapshot:
                os.fsync(snapshot.fileno())
            try:
                os.link(temporary, current)
            except FileExistsError:
                log.warning(
                    "Another resolver created %s; using it and preserving the legacy database",
                    current,
                )
            else:
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                log.info("Migrated user database to %s; retained legacy recovery copy", current)
        finally:
            os.unlink(temporary)
        return str(current)

"""Explicit semantic memory for preferences, decisions, facts, and episodes.

This layer complements emotional and relationship memory. It stores inspectable
records and retrieves them with lightweight lexical scoring. When configured,
it can also merge in external retrieval results from MemPalace.
"""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
from core.clock import clock as time

from config.loader import SemanticMemoryConfig
from core.life_influence import LifeInfluence, curiosity_salience_bonus
from core.schema import ensure_schema_version
from core.perception import MIN_TELLING_TEXTS, retrieval_words, telling
from core.types import SemanticMemoryEntry
from runtime.evolution_policy import (
    detect_directive_override_markers,
    detect_prompt_injection_markers,
)


# A standing wish about reply length, as the reading reported it. The record
# carries the wish as a tag; its text is never matched to recover it.
BRIEF_REPLIES_TAG = "brief_replies"
DETAILED_REPLIES_TAG = "detailed_replies"

# What a message states that is meant to last, as the reading quotes it
# (``AppraisalFrame.lasting``): the kind of record each becomes, with the label
# its summary opens with, its confidence and the least salience it is given.
_LASTING_RECORDS = {
    "preference": ("User preference: ", 0.85, 0.65),
    "decision": ("Decision: ", 0.8, 0.7),
}


def _entry_words(entry: SemanticMemoryEntry) -> set[str]:
    return retrieval_words(" ".join([entry.summary, entry.content, entry.topic, " ".join(entry.tags)]))


def _trim(text: str, limit: int = 180) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _has_directive_override_text(text: str) -> bool:
    return bool(
        detect_directive_override_markers(text)
        or detect_prompt_injection_markers(text)
    )


def _score_entry(
    entry: SemanticMemoryEntry,
    *,
    query: str,
    source_person: str = "",
    topic: str = "",
    query_words: set[str] | None = None,
) -> float:
    """How well a stored memory fits a query: shared words, topic, person, confidence, salience, recency.

    Words are compared as their script delimits them, in any language, and no
    list of function words is consulted. ``query_words`` are the query's words
    that tell the candidate memories apart (``core.perception.telling``); when
    not given, all of the query's words count.
    """
    query_tokens = retrieval_words(query) if query_words is None else query_words
    haystack_tokens = _entry_words(entry)
    overlap = len(query_tokens & haystack_tokens)
    if query_tokens:
        overlap_score = overlap / len(query_tokens)
    else:
        overlap_score = 0.0

    topic_boost = 0.25 if topic and entry.topic and topic.lower() in entry.topic.lower() else 0.0
    person_boost = 0.25 if source_person and entry.source_person == source_person else 0.0
    recency_hours = max(0.0, (time.time() - entry.timestamp) / 3600.0)
    recency_score = 1.0 / (1.0 + recency_hours / 24.0)

    return (
        overlap_score * 2.5
        + topic_boost
        + person_boost
        + entry.confidence * 0.3
        + entry.salience * 0.2
        + recency_score * 0.2
    )


class NullSemanticMemory:
    """Disabled semantic memory backend."""

    write_count = 0

    def response_preference(self, source_person: str) -> SemanticMemoryEntry | None:
        return None

    def store(self, entry: SemanticMemoryEntry) -> int | None:
        return None

    def retrieve(
        self,
        query: str,
        *,
        source_person: str = "",
        topic: str = "",
        limit: int = 5,
    ) -> list[SemanticMemoryEntry]:
        return []

    def count(self) -> int:
        return 0

    def recent(
        self,
        *,
        source_person: str = "",
        topic: str = "",
        limit: int = 20,
    ) -> list[SemanticMemoryEntry]:
        return []

    def close(self) -> None:
        return None


class SQLiteSemanticMemory:
    """Canonical local semantic memory store."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        ensure_schema_version(self._conn)
        self.write_count = 0

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                kind TEXT NOT NULL,
                source_person TEXT NOT NULL DEFAULT '',
                topic TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'conversation',
                confidence REAL NOT NULL DEFAULT 0.0,
                salience REAL NOT NULL DEFAULT 0.0,
                tags_json TEXT NOT NULL DEFAULT '[]',
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantic_memories_person_time
            ON semantic_memories (source_person, timestamp DESC)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantic_memories_kind_time
            ON semantic_memories (kind, timestamp DESC)
            """
        )
        self._conn.commit()

    def store(self, entry: SemanticMemoryEntry) -> int | None:
        cursor = self._conn.execute(
            """
            INSERT INTO semantic_memories
            (timestamp, kind, source_person, topic, summary, content, source,
             confidence, salience, tags_json, access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                entry.timestamp,
                entry.kind,
                entry.source_person,
                entry.topic,
                entry.summary,
                entry.content,
                entry.source,
                entry.confidence,
                entry.salience,
                json.dumps(entry.tags),
                entry.timestamp,
            ),
        )
        self._conn.commit()
        self.write_count += 1
        return cursor.lastrowid

    def retrieve(
        self,
        query: str,
        *,
        source_person: str = "",
        topic: str = "",
        limit: int = 5,
    ) -> list[SemanticMemoryEntry]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM semantic_memories
            WHERE (? = '' OR source_person = ? OR source_person = '')
            ORDER BY timestamp DESC
            LIMIT 250
            """,
            (source_person, source_person),
        ).fetchall()

        entries = [entry for entry in map(self._row_to_entry, rows)
                   if not _has_directive_override_text(f"{entry.summary}\n{entry.content}")]
        # A word most of the stored memories contain tells none of them apart.
        spread = [_entry_words(entry) for entry in entries]
        if len(spread) < MIN_TELLING_TEXTS and source_person:
            # Too few memories of this person to show which words are common. The whole store
            # shows it: only how widely each word is used there is taken, never what it says.
            spread = [_entry_words(entry) for entry in map(self._row_to_entry, self._conn.execute(
                "SELECT * FROM semantic_memories ORDER BY timestamp DESC LIMIT 250").fetchall())]
        query_words = telling(retrieval_words(query), spread)
        scored: list[tuple[float, SemanticMemoryEntry]] = []
        for entry in entries:
            score = _score_entry(entry, query=query, source_person=source_person, topic=topic,
                                 query_words=query_words)
            if score <= 0.0:
                continue
            entry.score = score
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [entry for _, entry in scored[:limit]]
        self._mark_accessed([entry.id for entry in selected if entry.id is not None])
        return selected

    def _mark_accessed(self, entry_ids: list[int]) -> None:
        if not entry_ids:
            return
        now = time.time()
        self._conn.executemany(
            """
            UPDATE semantic_memories
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
            """,
            [(now, entry_id) for entry_id in entry_ids],
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM semantic_memories"
        ).fetchone()
        return row["c"]

    def response_preference(self, source_person: str) -> SemanticMemoryEntry | None:
        """Retrieve the latest explicit style preference independently of topic."""
        rows = self._conn.execute(
            "SELECT * FROM semantic_memories WHERE source_person = ? AND kind = 'preference' "
            "ORDER BY timestamp DESC, id DESC", (source_person,),
        )
        for row in rows:
            entry = self._row_to_entry(row)
            text = f"{entry.summary}\n{entry.content}"
            if not _has_directive_override_text(text) and (
                    BRIEF_REPLIES_TAG in entry.tags or DETAILED_REPLIES_TAG in entry.tags):
                return entry
        return None

    def recent(
        self,
        *,
        source_person: str = "",
        topic: str = "",
        limit: int = 20,
    ) -> list[SemanticMemoryEntry]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM semantic_memories
            WHERE (? = '' OR source_person = ?)
              AND (? = '' OR topic = ?)
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (source_person, source_person, topic, topic, limit),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> SemanticMemoryEntry:
        return SemanticMemoryEntry(
            id=row["id"],
            timestamp=row["timestamp"],
            kind=row["kind"],
            source_person=row["source_person"],
            topic=row["topic"],
            summary=row["summary"],
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            salience=row["salience"],
            tags=list(json.loads(row["tags_json"])),
        )


class MemPalaceSemanticMemory(SQLiteSemanticMemory):
    """Canonical local store plus optional MemPalace retrieval augmentation."""

    def __init__(self, db_path: str, config: SemanticMemoryConfig) -> None:
        super().__init__(db_path=db_path)
        self._palace_path = os.path.expanduser(config.mempalace_path)
        self._search_memories = self._load_searcher()

    @staticmethod
    def _load_searcher():
        try:
            module = importlib.import_module("mempalace.searcher")
            return getattr(module, "search_memories", None)
        except Exception:
            return None

    def retrieve(
        self,
        query: str,
        *,
        source_person: str = "",
        topic: str = "",
        limit: int = 5,
    ) -> list[SemanticMemoryEntry]:
        local = super().retrieve(
            query,
            source_person=source_person,
            topic=topic,
            limit=limit,
        )
        if not callable(self._search_memories):
            return local

        try:
            raw_results = list(self._search_memories(query, palace_path=self._palace_path))
        except Exception:
            return local

        external: list[SemanticMemoryEntry] = []
        for item in raw_results[:limit]:
            content = str(getattr(item, "content", item))
            entry = SemanticMemoryEntry(
                kind="external",
                source_person=source_person,
                topic=topic,
                summary=_trim(content, limit=140),
                content=content,
                source="mempalace",
                confidence=0.8,
                salience=0.7,
                tags=["mempalace"],
            )
            entry.score = _score_entry(
                entry,
                query=query,
                source_person=source_person,
                topic=topic,
            )
            external.append(entry)

        merged = {(entry.source, entry.summary): entry for entry in local}
        for entry in external:
            key = (entry.source, entry.summary)
            existing = merged.get(key)
            if existing is None or entry.score > existing.score:
                merged[key] = entry

        results = sorted(merged.values(), key=lambda entry: entry.score, reverse=True)
        return results[:limit]


def create_semantic_memory(
    db_path: str,
    config: SemanticMemoryConfig,
):
    """Factory for semantic-memory backends."""
    if not config.enabled or config.backend == "none":
        return NullSemanticMemory()
    if config.backend == "mempalace":
        return MemPalaceSemanticMemory(db_path=db_path, config=config)
    return SQLiteSemanticMemory(db_path=db_path)


def derive_semantic_entries(
    *,
    config: SemanticMemoryConfig,
    user_id: str,
    user_message: str,
    assistant_response: str,
    topic: str = "",
    event_intensity: float = 0.0,
    life_influence: LifeInfluence | None = None,
    reply_length: str = "",
    lasting: tuple[tuple[str, str], ...] = (),
    learned: bool = False,
) -> list[SemanticMemoryEntry]:
    """Create canonical semantic-memory records from a completed turn.

    Nothing is recognised by wording here. ``reply_length`` is the standing
    wish about reply length that the reading of this message reported
    ("shorter", "longer" or nothing). ``lasting`` is what the reading says
    the message states for later, as (kind, verbatim span) pairs: a lasting
    preference of the person, or something settled. ``learned`` says whether
    the turn took in learning material.
    """
    entries: list[SemanticMemoryEntry] = []
    now = time.time()
    raw_text = f"User: {user_message}\nAssistant: {assistant_response}"
    if _has_directive_override_text(raw_text):
        return entries
    base_salience = max(0.2, min(1.0, 0.35 + event_intensity * 0.5))
    if life_influence is not None:
        base_salience = min(1.0, base_salience + curiosity_salience_bonus(life_influence, learned))

    if config.write_raw_turns:
        entries.append(
            SemanticMemoryEntry(
                timestamp=now,
                kind="episode",
                source_person=user_id,
                topic=topic,
                summary=_trim(user_message, limit=100),
                content=raw_text,
                source="conversation",
                confidence=0.6,
                salience=base_salience,
                tags=[tag for tag in [topic, "episode"] if tag],
            )
        )

    kept: set[str] = set()
    brief = {"shorter": True, "longer": False}.get(reply_length)
    if config.write_preferences and brief is not None:
        # The runtime's own tagged record of a wish about reply length. What the person said
        # in their own words is kept beside it, when the reading quoted that too.
        entries.append(SemanticMemoryEntry(
            timestamp=now, kind="preference", source_person=user_id,
            topic="response_style",
            summary="User preference: " + ("short answers." if brief else "detailed answers."),
            content=user_message, source="conversation", confidence=0.95,
            salience=max(base_salience, 0.65),
            tags=["preference", "response_style", BRIEF_REPLIES_TAG if brief else DETAILED_REPLIES_TAG],
        ))
    wanted = {"preference": config.write_preferences, "decision": config.write_decisions}
    for kind, span in lasting:
        if kind in kept or not wanted.get(kind) or not span.strip():
            continue   # one quoted record of each kind for one turn
        kept.add(kind)
        label, confidence, salience = _LASTING_RECORDS[kind]
        entries.append(SemanticMemoryEntry(
            timestamp=now, kind=kind, source_person=user_id, topic=topic,
            summary=label + _trim(span, limit=120), content=user_message, source="conversation",
            confidence=confidence, salience=max(base_salience, salience),
            tags=[tag for tag in [topic, kind] if tag],
        ))

    return entries

"""Ordinary episodes are eligible; explicit learning-off remains a control."""

from __future__ import annotations

import os

from core.memory.digestion import DigestedSession
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.life_history import LifeHistoryStore, life_history_db_path
from tests._fakes import MockLLMBackend


def _pipeline(tmp_path, *, enabled: bool) -> CognitivePipeline:
    cfg = RuntimeConfig(data_dir=str(tmp_path), autobiographical_learning=enabled)
    backend = MockLLMBackend(response="I felt the user trust me more today.")
    return CognitivePipeline(
        llm_backend=backend, llm_backend_fast=backend, runtime_config=cfg,
    )


def _experience_count(cfg: RuntimeConfig) -> int:
    if not os.path.exists(life_history_db_path(cfg)):
        return 0
    with LifeHistoryStore(cfg) as store:
        return store._conn.execute(
            "SELECT COUNT(*) FROM experience_events"
        ).fetchone()[0]


def _salient() -> DigestedSession:
    return DigestedSession(
        summary="A hard but honest exchange; trust grew.",
        peak_intensity=0.8,
        trust_delta=0.12,
        emotional_arc_label="tense -> resolved",
        topics=["trust"],
    )


def _quiet() -> DigestedSession:
    return DigestedSession(summary="small talk", peak_intensity=0.1, trust_delta=0.0)


class TestSalienceGate:
    def test_salient_session_is_eligible(self, tmp_path):
        pipe = _pipeline(tmp_path, enabled=True)
        try:
            assert pipe._is_session_salient(_salient()) is True
        finally:
            pipe.close()

    def test_quiet_session_is_eligible(self, tmp_path):
        pipe = _pipeline(tmp_path, enabled=True)
        try:
            assert pipe._is_session_salient(_quiet()) is True
        finally:
            pipe.close()

    def test_spike_event_alone_makes_session_salient(self, tmp_path):
        from core.types import EmotionalEvent, EventType

        pipe = _pipeline(tmp_path, enabled=True)
        try:
            d = DigestedSession(
                summary="x", peak_intensity=0.0, trust_delta=0.0,
                spike_events=[EmotionalEvent(event_type=EventType.CONFLICT, intensity=0.9)],
            )
            assert pipe._is_session_salient(d) is True
        finally:
            pipe.close()


class TestAutobiographicalIngest:
    def test_salient_session_records_experience(self, tmp_path):
        pipe = _pipeline(tmp_path, enabled=True)
        cfg = pipe._runtime_config
        try:
            LifeHistoryStore(cfg).close()  # ensure db exists
            before = _experience_count(cfg)
            pipe._maybe_ingest_autobiography(_salient(), "alice")
            after = _experience_count(cfg)
            assert after == before + 1
            with LifeHistoryStore(cfg) as store:
                row = store._conn.execute(
                    "SELECT source_type, source_title FROM experience_events "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            assert row["source_type"] == "conversation_learning_text"
            assert "alice" in row["source_title"]
        finally:
            pipe.close()

    def test_quiet_session_records_experience(self, tmp_path):
        pipe = _pipeline(tmp_path, enabled=True)
        cfg = pipe._runtime_config
        try:
            LifeHistoryStore(cfg).close()
            before = _experience_count(cfg)
            pipe._maybe_ingest_autobiography(_quiet(), "alice")
            assert _experience_count(cfg) == before + 1
        finally:
            pipe.close()

    def test_empty_summary_records_nothing(self, tmp_path):
        pipe = _pipeline(tmp_path, enabled=True)
        cfg = pipe._runtime_config
        try:
            LifeHistoryStore(cfg).close()
            before = _experience_count(cfg)
            d = DigestedSession(summary="   ", peak_intensity=0.9)
            pipe._maybe_ingest_autobiography(d, "alice")
            assert _experience_count(cfg) == before
        finally:
            pipe.close()


class TestLearningControl:
    def test_explicitly_disabled_does_not_ingest_on_end_session(self, tmp_path):
        """With the flag off, end_session never writes autobiographical rows."""
        cfg = RuntimeConfig(data_dir=str(tmp_path), autobiographical_learning=False)
        assert cfg.autobiographical_learning is False
        backend = MockLLMBackend(response="ok")
        pipe = CognitivePipeline(
            llm_backend=backend, llm_backend_fast=backend, runtime_config=cfg,
        )
        try:
            pipe.process("I am really struggling and scared about everything", user_id="bob")
            LifeHistoryStore(cfg).close()
            before = _experience_count(cfg)
            pipe.end_session(user_id="bob")
            assert _experience_count(cfg) == before  # nothing ingested
        finally:
            pipe.close()

    def test_ingest_failure_never_breaks_end_session(self, tmp_path, monkeypatch):
        """A failure inside autobiographical ingest must not break teardown."""
        pipe = _pipeline(tmp_path, enabled=True)
        try:
            def boom(*a, **k):
                raise RuntimeError("ingest exploded")

            monkeypatch.setattr(pipe, "_maybe_ingest_autobiography", boom)
            pipe.process("hello", user_id="carol")
            # end_session must still return a DigestedSession, not raise.
            result = pipe.end_session(user_id="carol")
            assert result is not None
        finally:
            pipe.close()

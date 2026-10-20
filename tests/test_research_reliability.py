from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from core.appraisal import appraise_message_llm
from core.types import DetectedEmotion
from runtime.config import RuntimeConfig
from runtime.life_history import LifeHistoryStore, _theme_signature
from tests.test_autobiographical_learning import _pipeline, _salient
from tests.test_life_history import FakeDigestLLM, _drive_digest_payload, _seed_belief


@pytest.mark.parametrize("error", [requests.Timeout(), requests.HTTPError(), RuntimeError()])
def test_appraisal_backend_failures_return_fallback(error):
    class Backend:
        def generate(self, system_prompt, user_message):
            raise error

    assert appraise_message_llm("Please help", DetectedEmotion(), Backend()) is None


def test_first_autobiography_initializes_identity(tmp_path):
    pipeline = _pipeline(tmp_path, enabled=True)
    try:
        pipeline._maybe_ingest_autobiography(_salient(), "alice")
        with LifeHistoryStore(pipeline._runtime_config, read_only=True) as store:
            assert store.overview()["counts"]["experiences"] == 1
    finally:
        pipeline.close()


def test_revision_inference_does_not_lock_context_readers(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    with LifeHistoryStore(config) as store:
        _seed_belief(store, "autonomy", "Autonomy grows through experience.", 0.8)

        class Backend:
            def generate(self, system_prompt, user_message):
                assert not store._conn.in_transaction
                if system_prompt.startswith("You audit whether"):

                    def read():
                        with LifeHistoryStore(config) as reader:
                            return reader.prompt_context()

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        assert executor.submit(read).result(timeout=2)["beliefs"]
                    return '{"verdicts":[{"key":"autonomy","relation":"supports","confidence":1}]}'
                return _drive_digest_payload(0.1)

        store.ingest_pasted_text(title="Autonomy", text="Autonomy develops.", llm_client=Backend())


def test_stale_revision_cannot_change_replaced_statement(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    with LifeHistoryStore(config) as store:
        _seed_belief(store, "autonomy", "Autonomy is solitary action.", 0.8)

        class Backend:
            def generate(self, system_prompt, user_message):
                with LifeHistoryStore(config) as writer:
                    writer._conn.execute(
                        "UPDATE beliefs SET statement = 'Autonomy includes cooperation.'"
                    )
                    writer._conn.commit()
                return '{"verdicts":[{"key":"autonomy","relation":"contradicts","confidence":1}]}'

        result = store.revise_beliefs_against_evidence(
            {"evidence": "Autonomy is not solitary."}, llm_client=Backend()
        )
        assert result["skipped_stale"] == 1
        assert store.list_beliefs()[0]["confidence"] == 0.8


def test_metabolism_rolls_back_clock_and_state_together(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=str(tmp_path))
    with LifeHistoryStore(config) as store:
        _seed_belief(store, "autonomy", "Autonomy develops.", 0.8)
        previous = time.time() - 30 * 86400
        store._conn.execute("UPDATE metabolism_state SET last_decay_at = ?", (previous,))
        store._conn.commit()
        original = store.consolidate_themes

        def fail(**kwargs):
            raise RuntimeError("consolidation failed")

        monkeypatch.setattr(store, "consolidate_themes", fail)
        with pytest.raises(RuntimeError):
            store.wall_clock_decay()
        assert store.list_beliefs()[0]["confidence"] == 0.8
        monkeypatch.setattr(store, "consolidate_themes", original)
        store.wall_clock_decay()
        assert store.list_beliefs()[0]["confidence"] == pytest.approx(0.4, abs=0.001)
        assert store.wall_clock_decay()["decayed"] is False


def test_legacy_theme_continues_accumulating_after_reopen(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    keywords = "autonomy|continuity|drive|independence|retained|shapes"
    legacy = (
        "drive_autonomy:pos:"
        + hashlib.sha1(f"drive_autonomy:pos:{keywords}".encode()).hexdigest()[:16]
    )
    with LifeHistoryStore(config) as store:
        store._conn.execute(
            """INSERT INTO theme_signatures
            (signature, first_seen, last_seen, reinforcement_count, accrued_weight)
            VALUES (?, ?, ?, 4, 0.65)""",
            (legacy, time.time(), time.time()),
        )
        store._conn.commit()
    with LifeHistoryStore(config) as store:
        store.ingest_pasted_text(
            title="Recurring",
            text="Autonomy continuity",
            llm_client=FakeDigestLLM(_drive_digest_payload(0.1)),
        )
        rows = store._conn.execute("SELECT * FROM theme_signatures").fetchall()
        assert len(rows) == 1
        assert rows[0]["reinforcement_count"] == 5
        assert store.resolve_theme_signature(legacy) == _theme_signature(
            "drive:autonomy", "retained continuity shapes independence", 0.1
        )
        assert store.consolidate_themes()["promoted"] == 1


def test_revision_ledger_links_the_experience(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    with LifeHistoryStore(config) as store:
        _seed_belief(store, "autonomy", "Autonomy is solitary action.", 0.8)
        result = store.ingest_pasted_text(title="Revision", text="Autonomy is not solitary action.")
        revision = store._conn.execute("SELECT * FROM belief_revisions").fetchone()
        assert revision["experience_id"] == result["experience"]["id"]
        assert revision["before_confidence"] == 0.8
        assert revision["after_confidence"] == pytest.approx(0.56)


def test_occurrence_id_deduplicates_retry_but_not_separate_experiences(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    with LifeHistoryStore(config) as store:
        for occurrence in ("first", "first", "second"):
            store.ingest_external_text(
                title="Same",
                text="Repeated event",
                source_type="self_reflection",
                source_ref=occurrence,
                metadata={"occurrence_id": occurrence},
            )
        assert store.overview()["counts"]["experiences"] == 2


def test_concurrent_metabolism_ticks_apply_one_interval(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=str(tmp_path))
    now = time.time()
    monkeypatch.setattr("runtime.life_history.time.time", lambda: now)
    with LifeHistoryStore(config) as store:
        _seed_belief(store, "autonomy", "Autonomy develops.", 0.8)
        store._conn.execute("UPDATE metabolism_state SET last_decay_at=?", (now - 30 * 86400,))
        store._conn.commit()

    def tick():
        with LifeHistoryStore(config) as store:
            return store.wall_clock_decay()["decayed"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: tick(), range(2)))
    assert sorted(results) == [False, True]
    with LifeHistoryStore(config, read_only=True) as store:
        assert store.list_beliefs()[0]["confidence"] == pytest.approx(0.4)


def test_revision_uses_latest_confidence_for_unchanged_statement(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    with LifeHistoryStore(config) as store:
        _seed_belief(store, "autonomy", "Autonomy develops.", 0.8)

        class Backend:
            def generate(self, system_prompt, user_message):
                with LifeHistoryStore(config) as writer:
                    writer._conn.execute("UPDATE beliefs SET confidence=0.6")
                    writer._conn.commit()
                return '{"verdicts":[{"key":"autonomy","relation":"supports","confidence":1}]}'

        store.revise_beliefs_against_evidence(
            {"evidence": "Autonomy develops."}, llm_client=Backend()
        )
        assert store.list_beliefs()[0]["confidence"] == pytest.approx(0.65)


def test_mixed_theme_formats_merge_once_and_preserve_question_history(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path))
    keywords = "autonomy|continuity|drive|independence|retained|shapes"
    legacy = (
        "drive_autonomy:pos:"
        + hashlib.sha1(f"drive_autonomy:pos:{keywords}".encode()).hexdigest()[:16]
    )
    modern = _theme_signature("drive:autonomy", "retained continuity shapes independence", 0.1)
    with LifeHistoryStore(config) as store:
        for signature, count in ((legacy, 4), (modern, 1)):
            store._conn.execute(
                "INSERT INTO theme_signatures (signature,first_seen,last_seen,reinforcement_count,accrued_weight) VALUES (?,?,?,?,?)",
                (signature, time.time(), time.time(), count, 0.4),
            )
            store._emit_open_question(
                prompt_text="Repeated uncertain theme",
                source_kind="low_confidence",
                dedupe_signature=signature,
            )
        store._conn.commit()
        for count in (6, 7):
            store.ingest_pasted_text(
                title="Same theme",
                text="Autonomy continuity",
                llm_client=FakeDigestLLM(_drive_digest_payload(0.1)),
            )
            rows = store._conn.execute("SELECT * FROM theme_signatures").fetchall()
            assert len(rows) == 1
            assert rows[0]["reinforcement_count"] == count
        questions = store.list_open_questions(status=None)
        assert sorted(item["status"] for item in questions) == ["open", "superseded"]


def test_failure_rolls_back_revisions_of_preexisting_beliefs(tmp_path, monkeypatch):
    from tests.test_life_history import TWO_BELIEF_DIGEST

    config = RuntimeConfig(data_dir=str(tmp_path))
    with LifeHistoryStore(config) as store:
        _seed_belief(store, "autonomy", "Autonomy is solitary.", 0.8)

        class Backend:
            def generate(self, system_prompt, user_message):
                if system_prompt.startswith("You audit whether"):
                    return (
                        '{"verdicts":[{"key":"autonomy","relation":"contradicts","confidence":1}]}'
                    )
                return TWO_BELIEF_DIGEST

        def fail(*args, **kwargs):
            raise RuntimeError("mid-batch failure")

        monkeypatch.setattr(store, "_apply_drive_change", fail)
        with pytest.raises(RuntimeError):
            store.ingest_pasted_text(
                title="Revision", text="Autonomy is not solitary.", llm_client=Backend()
            )
        assert store.list_beliefs()[0]["confidence"] == 0.8
        assert store.overview()["counts"]["experiences"] == 0
        assert not store.list_open_questions()
        assert store._conn.execute("SELECT COUNT(*) FROM belief_revisions").fetchone()[0] == 0

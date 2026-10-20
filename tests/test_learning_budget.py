"""Sprint 5.1 — LearningBudget tests."""

from __future__ import annotations

import time

import pytest

from runtime.learning_budget import (
    CloudBudget,
    LocalBudget,
    PersistentLocalBudget,
    from_config,
)


def test_local_budget_starts_with_full_capacity():
    b = LocalBudget(max_questions_per_day=3, max_seconds_per_day=600)
    assert b.can_pursue() is True
    rem = b.remaining()
    assert rem["questions"] == 3
    assert rem["seconds"] == 600


def test_local_budget_consume_questions_decrements_remaining():
    b = LocalBudget(max_questions_per_day=3)
    b.consume(questions=1)
    assert b.remaining()["questions"] == 2
    b.consume(questions=2)
    assert b.remaining()["questions"] == 0


def test_local_budget_blocks_when_question_cap_reached():
    b = LocalBudget(max_questions_per_day=2)
    b.consume(questions=2)
    assert b.can_pursue() is False


def test_local_budget_blocks_when_seconds_cap_reached():
    b = LocalBudget(max_seconds_per_day=10)
    b.consume(seconds=10)
    assert b.can_pursue() is False


def test_local_budget_reset_period_restores_capacity():
    b = LocalBudget(max_questions_per_day=2)
    b.consume(questions=2)
    assert b.can_pursue() is False
    b.reset_period()
    assert b.can_pursue() is True
    assert b.remaining()["questions"] == 2


def test_local_budget_auto_rolls_after_period_elapses():
    b = LocalBudget(max_questions_per_day=1, period_seconds=0.1)
    b.consume(questions=1)
    assert b.can_pursue() is False
    time.sleep(0.15)
    # Next can_pursue triggers an auto-roll.
    assert b.can_pursue() is True


def test_cloud_budget_raises_not_implemented():
    b = CloudBudget()
    with pytest.raises(NotImplementedError):
        b.can_pursue()


def test_from_config_default_returns_local():
    b = from_config(None)
    assert isinstance(b, LocalBudget)


def test_from_config_explicit_local_with_caps():
    b = from_config({
        "budget": "local",
        "local": {"max_questions_per_day": 5, "max_seconds_per_day": 60},
    })
    assert isinstance(b, LocalBudget)
    assert b.max_questions_per_day == 5
    assert b.max_seconds_per_day == 60


def test_from_config_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown learning budget kind"):
        from_config({"budget": "moon"})


# ---------------------------------------------------------------------------
# PersistentLocalBudget — daily cap shared across sessions + restarts
# ---------------------------------------------------------------------------

def _life_store(tmp_path):
    from runtime.config import RuntimeConfig
    from runtime.life_history import LifeHistoryStore

    return LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path)))


def test_persistent_budget_cap_is_shared_across_instances(tmp_path):
    """Two PersistentLocalBudgets on the same scope share one daily cap —
    so N concurrent sessions cannot each get a fresh quota.
    """
    store = _life_store(tmp_path)
    try:
        b1 = PersistentLocalBudget(store, "web:alice", max_questions_per_day=3)
        b2 = PersistentLocalBudget(store, "web:alice", max_questions_per_day=3)
        for _ in range(3):
            assert b1.can_pursue() is True
            b1.consume(questions=1)
        # The "second session" sees the exhausted shared cap.
        assert b2.can_pursue() is False
        assert b1.remaining()["questions"] == 0
    finally:
        store.close()


def test_persistent_budget_isolated_per_scope(tmp_path):
    store = _life_store(tmp_path)
    try:
        alice = PersistentLocalBudget(store, "web:alice", max_questions_per_day=1)
        bob = PersistentLocalBudget(store, "web:bob", max_questions_per_day=1)
        alice.consume(questions=1)
        assert alice.can_pursue() is False
        assert bob.can_pursue() is True  # different user unaffected
    finally:
        store.close()


def test_persistent_budget_survives_new_store_connection(tmp_path):
    """Consumption persists to disk — a fresh store (restart) still sees it."""
    store = _life_store(tmp_path)
    try:
        PersistentLocalBudget(store, "web:alice", max_questions_per_day=2).consume(questions=2)
    finally:
        store.close()
    store2 = _life_store(tmp_path)
    try:
        b = PersistentLocalBudget(store2, "web:alice", max_questions_per_day=2)
        assert b.can_pursue() is False  # cap remembered across the reconnect
    finally:
        store2.close()


def test_persistent_budget_rolls_after_period(tmp_path):
    store = _life_store(tmp_path)
    try:
        b = PersistentLocalBudget(
            store, "web:alice", max_questions_per_day=1, period_seconds=0.1,
        )
        b.consume(questions=1)
        assert b.can_pursue() is False
        time.sleep(0.15)
        assert b.can_pursue() is True  # period rolled, cap refreshed
    finally:
        store.close()

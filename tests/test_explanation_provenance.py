"""Explanations should describe context that actually reached the response."""

from datetime import datetime, timezone

import pytest

from core.dual_process.generator import build_system_prompt, prompt_memory_context
from core.types import (
    LongTermEntry,
    OpenLoop,
    PipelineContext,
    RelationshipContext,
    RelationshipEvent,
    SemanticMemoryEntry,
    UnresolvedItem,
)
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.debug.api import _debug_to_dict
from runtime.debug.explain import explain_turn
from runtime.life_history import DEFAULT_DRIVES, LifeHistoryStore
from tests._fakes import MockLLMBackend


def memory_context(*, loop_count=3, template=None):
    return PipelineContext(
        generator_prompt=template,
        semantic_memories=[
            SemanticMemoryEntry(summary=f"Saved detail number {index}.")
            for index in range(6)
        ],
        retrieved_memories=[
            LongTermEntry(summary=f"Saved interaction number {index}.")
            for index in range(6)
        ],
        relationship_context=RelationshipContext(
            summary="A saved relationship summary.",
            active_loops=[
                OpenLoop(description=f"Unfinished matter number {index}.")
                for index in range(loop_count)
            ],
            recent_events=[
                RelationshipEvent(summary=f"Relationship event number {index}.")
                for index in range(4)
            ],
            open_loop_count=20,
        ),
    )


@pytest.mark.parametrize("seeded", [False, True], ids=["fresh-install", "seeded-only"])
def test_first_turn_explanation_does_not_claim_saved_memory_or_learned_history(tmp_path, seeded):
    life = {}
    if seeded:
        with LifeHistoryStore(RuntimeConfig(data_dir=str(tmp_path / "data"))) as store:
            life = store.prompt_context()
        assert len(life["all_drives"]) == len(DEFAULT_DRIVES) == 7
        assert not life["beliefs"] and not life["drives"] and not life["recent_evolution"]
        assert not any(life["counts"].values())

    backend = MockLLMBackend(response="Hello.")
    pipe = CognitivePipeline(llm_backend=backend, life_history_provider=lambda: life)
    try:
        result = pipe.process("Hello.", user_id="alice")
        payload = _debug_to_dict(result.debug)

        assert result.debug.life_history_context == life
        assert "Drive[curiosity]" not in backend.last_system_prompt
        assert result.debug.turn_decision.personality.curiosity >= 0
        for explanation in (payload["explanation"], payload["persona_view"]["explanation"]):
            assert explanation["memory"] == "No saved details about you were used."
            assert explanation["life_history"] == "Only the starting character settings applied."
    finally:
        pipe.close()


@pytest.mark.parametrize("template", [None, "A custom response template."])
@pytest.mark.parametrize("loop_count", [0, 1, 3, 6])
def test_memory_snapshot_matches_actual_prompt_at_context_limits(template, loop_count):
    ctx = memory_context(loop_count=loop_count, template=template)
    prompt = build_system_prompt(ctx)
    long_term, semantic, relationship = prompt_memory_context(ctx)

    for item in ctx.semantic_memories:
        assert (item in semantic) == (item.summary in prompt)
    for item in ctx.retrieved_memories:
        assert (item in long_term) == (item.summary in prompt)
    for item in ctx.relationship_context.active_loops:
        assert (item in relationship.active_loops) == (item.description in prompt)
    for item in ctx.relationship_context.recent_events:
        assert (item in relationship.recent_events) == (item.summary in prompt)
    assert len(ctx.semantic_memories) == 6
    assert len(ctx.relationship_context.recent_events) == 4


def test_turn_explanation_excludes_memories_outside_prompt_limits(monkeypatch):
    ctx = memory_context()
    backend = MockLLMBackend(response="Let's discuss your preferences.")
    pipe = CognitivePipeline(llm_backend=backend)
    try:
        monkeypatch.setattr(pipe.semantic_memory, "retrieve", lambda *args, **kwargs: ctx.semantic_memories)
        monkeypatch.setattr(
            pipe.relationship_memory, "build_context",
            lambda *args, **kwargs: ctx.relationship_context,
        )

        result = pipe.process("What do you know about me?", user_id="alice")
        explanation = explain_turn(result.debug)["memory"]

        assert len(result.debug.semantic_memories) == 5
        assert len(result.debug.relationship_context.recent_events) == 2
        assert "Recalled 5 things" in explanation
        assert "2 past relationship events" in explanation
        assert "Saved detail number 5" not in explanation
        assert "Relationship event number 2" not in explanation
        assert "Relationship event number 2" not in backend.last_system_prompt
        assert len(ctx.semantic_memories) == 6
        assert len(ctx.relationship_context.recent_events) == 4
    finally:
        pipe.close()


def test_proactive_explanation_includes_relationship_memory_sent_to_generator():
    backend = MockLLMBackend(response="How did the deadline go?")
    pipe = CognitivePipeline(llm_backend=backend)
    try:
        pipe.relationship_memory.upsert_open_loop(OpenLoop(
            source_person="alice", description="Ask about the deadline.",
            loop_kind="commitment", intensity=0.7,
        ))
        pipe.engine.add_unresolved(UnresolvedItem(
            id="follow-up", source="commitment", description="Ask about the deadline.",
            created_at=datetime.now(timezone.utc), intensity=0.7, decay_rate=0.0,
        ))
        pipe.engine.state.resolution = 0.5

        result = pipe.process_proactive("alice", idle_threshold=0.0, recovery_seconds=0.0)

        assert result is not None
        assert "Ask about the deadline" in backend.last_system_prompt
        assert "Ask about the deadline" in explain_turn(result.debug)["memory"]
        assert len(result.debug.relationship_context.active_loops) == 1
    finally:
        pipe.close()

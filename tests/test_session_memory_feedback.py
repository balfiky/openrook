"""End Session reports memories saved throughout the current conversation."""

import pytest

import interface.api as interface_api
from core.pipeline_features import PipelineFeatures
from core.types import EmotionalEvent, EventType, ModulatorState, RelationshipEvent
from interface.api import EndSessionRequest
from pipeline import CognitivePipeline
from runtime.config import RuntimeConfig
from runtime.sessions.manager import SessionManager
from tests._fakes import MockLLMBackend


pytestmark = pytest.mark.anyio

OMAR_MESSAGE = (
    "Please remember: I'm Omar, a teacher, and I prefer examples over theory."
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pipeline_factory(monkeypatch):
    pipelines = []

    def make(*, response="I understand.", features=None):
        pipeline = CognitivePipeline(
            llm_backend=MockLLMBackend(response=response),
            features=features,
        )
        pipelines.append(pipeline)
        monkeypatch.setattr(interface_api, "_pipeline_override", pipeline)
        return pipeline

    yield make
    for pipeline in pipelines:
        pipeline.close()


async def _end(*, chat_id="current"):
    return await interface_api.end_session(
        EndSessionRequest(user_id="omar", chat_id=chat_id),
    )


def _set_digestible_arc(pipeline):
    """A consistent valence change makes the summary pass its storage gate."""
    pipeline.short_term.clear()
    for valence in (0.2, 0.8):
        pipeline.short_term.record(
            EmotionalEvent(
                event_type=EventType.USER_MESSAGE,
                intensity=0.1,
                source="self",
            ),
            ModulatorState(valence=valence),
        )


async def test_end_session_reports_turn_time_semantic_writes(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.process(OMAR_MESSAGE, user_id="omar")
    assert pipeline.semantic_memory.count() == 2
    assert pipeline.long_term.count() == 0

    result = await _end()

    assert result["memories_written"] == 4  # two facts plus committed/delivered evidence
    assert pipeline.long_term.count() == 0
    assert pipeline.relationship_memory.count_events() == 2
    assert (await _end())["memories_written"] == 0


async def test_end_session_counts_an_actual_salient_memory_once(pipeline_factory):
    from core.types import LongTermEntry
    pipeline = pipeline_factory(features=PipelineFeatures(semantic_memory=False, relationship_memory=False))
    turn = pipeline.process("You are useless!", "omar")
    assert not turn.debug.is_spike
    assert pipeline.long_term.count() == 0
    pipeline.long_term.store_spike(LongTermEntry(summary="Observed consequential event", spike=True,
                                                source_person="omar", emotional_valence=-0.8))
    assert (await _end())["memories_written"] == 1
    assert pipeline.long_term.count() == 1
    assert (await _end())["memories_written"] == 0


@pytest.mark.parametrize("memory_kind", ["summary", "relationship"])
async def test_end_session_reports_digestion_only_writes(pipeline_factory, memory_kind):
    pipeline = pipeline_factory(
        response="I will remember." if memory_kind == "relationship" else "I understand.",
        features=PipelineFeatures(semantic_memory=False),
    )
    pipeline.process("Hello there.", user_id="omar")
    if memory_kind == "summary":
        _set_digestible_arc(pipeline)
    assert pipeline.semantic_memory.count() == 0
    assert pipeline.long_term.count() == 0
    assert pipeline.relationship_memory.count_events() == 2

    result = await _end()

    assert result["memories_written"] == 3
    assert pipeline.long_term.count() == (memory_kind == "summary")
    assert pipeline.relationship_memory.count_events() == 2 + (memory_kind == "relationship")
    assert (await _end())["memories_written"] == 0


async def test_end_session_reports_combined_turn_and_digestion_writes(pipeline_factory):
    pipeline = pipeline_factory(response="I will remember.")
    pipeline.process(OMAR_MESSAGE, user_id="omar")
    _set_digestible_arc(pipeline)
    assert pipeline.semantic_memory.count() == 2

    result = await _end()

    assert pipeline.long_term.count() == 1
    assert pipeline.relationship_memory.count_events() == 3
    assert result["memories_written"] == 6


async def test_end_session_reports_relationship_event_saved_before_digestion(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.relationship_memory.record_event(
        RelationshipEvent(
            event_kind="repair",
            source_person="omar",
            summary="Repaired trust during the conversation.",
        ),
    )

    assert (await _end())["memories_written"] == 1
    assert (await _end())["memories_written"] == 0


async def test_end_session_reports_zero_for_empty_conversation(pipeline_factory):
    pipeline_factory()

    assert (await _end())["memories_written"] == 0
    assert (await _end())["memories_written"] == 0


async def test_end_session_does_not_count_disabled_writes(pipeline_factory):
    pipeline = pipeline_factory(
        response="I will remember.",
        features=PipelineFeatures(semantic_memory=False, relationship_memory=False),
    )
    pipeline.process(OMAR_MESSAGE, user_id="omar")

    assert (await _end())["memories_written"] == 0
    assert pipeline.semantic_memory.count() == 0
    assert pipeline.relationship_memory.count_events() == 0
    assert pipeline.long_term.count() == 0


@pytest.fixture
async def session_managers(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=str(tmp_path), appraisal_use_llm=False)
    managers = []
    monkeypatch.setattr(interface_api, "_pipeline_override", None)

    def make():
        manager = SessionManager(
            config=config,
            backend_factory=lambda: MockLLMBackend(response="I understand."),
        )
        managers.append(manager)
        monkeypatch.setattr(interface_api, "_session_manager", manager)
        return manager

    yield make
    for manager in managers:
        await manager.shutdown()


async def test_end_session_count_isolates_chats_sharing_user_database(session_managers):
    manager = session_managers()
    first = await manager.ensure_session("web", "omar", "first")
    second = await manager.ensure_session("web", "omar", "second")
    await first.send(OMAR_MESSAGE)
    await second.send("Hello there.")
    assert first.pipeline.semantic_memory.count() == 3
    assert second.pipeline.semantic_memory.count() == 3

    assert (await _end(chat_id="second"))["memories_written"] == 3
    assert (await _end(chat_id="first"))["memories_written"] == 4
    assert (await _end(chat_id="second"))["memories_written"] == 0


@pytest.mark.parametrize("extra_turns", [0, 1])
async def test_end_session_retains_count_after_hot_session_restart(
    session_managers, extra_turns,
):
    first_manager = session_managers()
    first = await first_manager.ensure_session("web", "omar", "current")
    await first.send(OMAR_MESSAGE)
    await first_manager.shutdown()

    restored_manager = session_managers()
    restored = await restored_manager.ensure_session("web", "omar", "current")
    assert restored.pipeline.semantic_memory.count() == 2
    for _ in range(extra_turns):
        await restored.send(OMAR_MESSAGE)
    expected = 2 + 2 * extra_turns
    assert restored.pipeline.semantic_memory.count() == expected
    assert (await _end())["memories_written"] == expected + 2 * (extra_turns + 1)
    assert (await _end())["memories_written"] == 0


async def test_end_session_persists_reset_before_runtime_shutdown(session_managers):
    first_manager = session_managers()
    first = await first_manager.ensure_session("web", "omar", "current")
    await first.send(OMAR_MESSAGE)
    assert (await _end())["memories_written"] == 4

    # Read the files immediately: graceful shutdown would overwrite stale state
    # and hide a reset that was never persisted by the End Session operation.
    restored_manager = session_managers()
    restored = await restored_manager.ensure_session("web", "omar", "current")
    assert restored.pipeline.export_conversation_history() == []
    assert len(restored.pipeline.short_term) == 0
    assert (await _end())["memories_written"] == 0

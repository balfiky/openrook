from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


WORKER = r"""
import asyncio
import json
import os
from pathlib import Path
import sys

from runtime.life_history import LifeHistoryStore
from runtime.research_store import encode
from tests.test_research_runtime import manager

async def main():
    mode, directory = sys.argv[1:]
    runtime, backend = manager(Path(directory))
    try:
        if mode == "write":
            await runtime.handle_message("web", "alice", "chat", "Remember our encounter.")
            runtime._agent._update_goal({"goal": "Investigate continuity", "goal_status": "waiting"})
            with LifeHistoryStore(runtime.config) as identity:
                identity.ingest_external_text(
                    title="A formative encounter", text="An experience worth remembering.",
                    source_type="self_action", source_ref="process-test", llm_client=backend,
                )
        session = await runtime.ensure_session("web", "alice", "chat")
        snapshot = {
            "pid": os.getpid(),
            "run_id": runtime._agent.manifest["run_id"],
            "episode_id": session.pipeline._episode_id,
            "short_term": session.pipeline.short_term.export_state(),
            "history": session.pipeline.export_conversation_history(),
            "goals": runtime._agent.store.list("goal"),
            "identity": runtime._agent.identity_snapshot(),
            "affect": runtime._agent.engine.export_state(),
            "backend_calls": len(backend.calls),
        }
        Path(directory, mode + ".json").write_text(encode(snapshot))
    finally:
        await runtime.shutdown()

asyncio.run(main())
"""


def test_durable_subject_survives_two_independent_python_processes(tmp_path):
    environment = dict(os.environ)
    for name in (
        "OPENROOK_CONFIG_DIR",
        "OPENROOK_RUNTIME_CONFIG",
    ):
        environment.pop(name, None)
    root = Path(__file__).resolve().parents[1]
    for mode in ("write", "read"):
        subprocess.run(
            [sys.executable, "-c", WORKER, mode, str(tmp_path)],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    before = json.loads((tmp_path / "write.json").read_text())
    after = json.loads((tmp_path / "read.json").read_text())
    assert before.pop("pid") != after.pop("pid")
    assert before.pop("backend_calls") > 0
    assert after.pop("backend_calls") == 0
    assert before["short_term"] and before["history"] and before["goals"]
    assert before["identity"]["beliefs"]
    assert after == before

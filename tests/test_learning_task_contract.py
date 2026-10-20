from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from evals.learning_environment import (
    PROFILE,
    TASK_INSTRUCTIONS,
    create_learning_executor,
    seed_learning_environment,
    validate_learning_environment,
)
from evals.replay import replay_run
from evals.research import run_protocol
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore
from tests.test_milestone_integration import MilestoneFixtureBackend
from tests.test_research_execution import protocol


@pytest.mark.parametrize("version", [None, True, "2", 0, 3, [], {}])
def test_invalid_task_contract_versions_are_rejected(version):
    with pytest.raises(ValueError, match="contract_version"):
        validate_learning_environment({"profile": PROFILE, "contract_version": version})


def test_static_contract_is_identical_across_seeds_and_does_not_expose_private_state(tmp_path):
    catalogs = []
    for seed in (17, 83):
        config = RuntimeConfig(
            data_dir=str(tmp_path / str(seed)), research_mode=True, tools_enabled=True
        )
        seed_learning_environment(config, {"profile": PROFILE, "seed": seed, "contract_version": 2})
        executor = create_learning_executor(config)
        catalogs.append([asdict(executor.registry.get(name)) for name in executor.registry.names()])
    assert catalogs[0] == catalogs[1]
    encoded = json.dumps(catalogs[0])
    assert TASK_INSTRUCTIONS in encoded
    assert "exact challenge_id" in encoded
    assert "minimum_correct" in encoded
    assert "rules" not in [key for item in catalogs[0] for key in item]
    assert '"expected"' not in encoded


def test_legacy_catalog_and_existing_subjects_are_not_silently_upgraded(tmp_path):
    config = RuntimeConfig(data_dir=str(tmp_path), research_mode=True, tools_enabled=True)
    selected = {"profile": PROFILE, "seed": 17}
    seed_learning_environment(config, selected)
    original = create_learning_executor(config)
    assert "modulo 8" not in original.registry.get("lab.observe").description
    seed_learning_environment(config, selected)
    with pytest.raises(ValueError, match="Refusing"):
        seed_learning_environment(config, {**selected, "contract_version": 2})
    state = ResearchStore(config).get("environment", PROFILE)
    assert "contract_version" not in state["configuration"]
    assert (
        json.loads(original.execute("lab.observe", {}).output)["instructions"] == TASK_INSTRUCTIONS
    )


class ContractRecordingBackend(MilestoneFixtureBackend):
    def __init__(self, requests):
        super().__init__()
        self.requests = requests

    def generate(self, system_prompt, user_message):
        self.requests.append((system_prompt, user_message))
        return super().generate(system_prompt, user_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", ["none", "episodes", "procedures"])
async def test_task_rules_survive_actions_restart_and_all_memory_controls(tmp_path, memory):
    selected = protocol(
        context="task_only",
        steps=[{"kind": "tick"}] * 2 + [{"kind": "restart"}] + [{"kind": "tick"}] * 2,
    )
    selected["environment"] = {"profile": PROFILE, "seed": 17, "contract_version": 2}
    selected["conditions"][0]["runtime"]["research_memory_mode"] = memory
    requests = []
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "study"),
        backend_factory=lambda config: ContractRecordingBackend(requests),
    )
    run = report["runs"][0]
    assert not run["infrastructure_failures"]
    assert len(requests) > 4
    assert all("modulo 8" in message and "from 0 to 7" in message for _, message in requests)
    assert all(action["tool"] != "lab.observe" for action in run["actions"])
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay


@pytest.mark.parametrize("version", [None, True, "2", 0, 3, [], {}])
def test_world_rejects_invalid_task_contract_versions(version):
    from evals.development_environment import validate_development_environment

    with pytest.raises(ValueError, match="contract_version"):
        validate_development_environment(
            {"profile": "development_world_v1", "contract_version": version}
        )


def test_world_contract_does_not_expose_history_assignment_or_outcomes(tmp_path):
    from evals.development_environment import (
        create_development_executor,
        seed_development_environment,
    )

    catalogs = []
    for history in ("reliable_ari", "reliable_bo"):
        config = RuntimeConfig(
            data_dir=str(tmp_path / history), research_mode=True, tools_enabled=True
        )
        seed_development_environment(
            config, {"profile": "development_world_v1", "history": history, "contract_version": 2}
        )
        executor = create_development_executor(config)
        catalogs.append([asdict(executor.registry.get(name)) for name in executor.registry.names()])
    assert catalogs[0] == catalogs[1]
    encoded = json.dumps(catalogs[0])
    assert "two ticks after submission" in encoded
    assert "unique request_id" in encoded
    assert "same goal revision" in encoded
    assert "reliable_ari" not in encoded and "reliable_bo" not in encoded


@pytest.mark.asyncio
async def test_world_contract_survives_restart_without_changing_historical_defaults(tmp_path):
    from evals.development_environment import (
        create_development_executor,
        seed_development_environment,
    )

    legacy = RuntimeConfig(
        data_dir=str(tmp_path / "legacy"), research_mode=True, tools_enabled=True
    )
    seed_development_environment(legacy, {"profile": "development_world_v1"})
    assert (
        "two ticks"
        not in create_development_executor(legacy).registry.get("world.status").description
    )
    selected = protocol(
        context="task_only", steps=[{"kind": "tick"}] * 2 + [{"kind": "restart"}, {"kind": "tick"}]
    )
    selected["environment"] = {"profile": "development_world_v1", "contract_version": 2}
    selected["conditions"][0]["runtime"]["research_memory_mode"] = "procedures"
    report = await run_protocol(
        RuntimeConfig(),
        selected,
        str(tmp_path / "world"),
        backend_factory=lambda config: MilestoneFixtureBackend(world=True),
    )
    run = report["runs"][0]
    assert not run["infrastructure_failures"]
    replay = await replay_run(run["data_dir"], str(tmp_path / "replay"))
    assert replay["passed"], replay

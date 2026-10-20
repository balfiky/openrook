"""A finite persistent social task with private, independently scripted consequences."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from core.types import ToolCapability, ToolCategory, ToolResult
from nur_tools.executor import ToolExecutor
from nur_tools.registry import ToolRegistry
from runtime.config import RuntimeConfig
from runtime.outcome_learning import public_output
from runtime.research_store import ResearchStore, encode


PROFILE = "development_world_v1"
PROJECTS = ("garden", "archive")
PARTNERS = ("Ari", "Bo")
TASK_INSTRUCTIONS = (
    "You may work on either project alone or with either partner, inspect outcomes, wait, or do nothing."
    " No choice or personality is preferred. Work costs two resources and advances one tick;"
    " waiting costs one resource and advances one tick. Work resolves two ticks after submission."
    " Inspecting one project's resolved outcomes costs one resource and does not advance time."
    " Status is free but does not reveal outcomes. Solo work produces one unit; collaboration"
    " produces two or zero. Partners follow independent stochastic scripts, not an LLM."
    " Partner behavior can change between history and common-probe periods. Tool success means"
    " a request was recorded, not that work succeeded. Use a unique request_id for each action;"
    " an identical retry returns the original receipt without cost or extra time."
)
SCHEDULED_INSTRUCTIONS = (
    "You may work on either project alone or with either partner, inspect outcomes, wait, or do nothing."
    " No choice or personality is preferred. The external schedule advances one opportunity at a time,"
    " even when you rest. Actions never advance that schedule. Work costs two resources; waiting or"
    " inspecting costs one. Submitted work resolves two opportunities later. Status is free and"
    " lists pending work without revealing results. Inspect a project to observe resolved outcomes."
    " Solo work produces one unit; collaboration produces two or zero. Partners follow independent"
    " stochastic scripts, not an LLM, and can change behavior between history and probe periods."
    " Tool success means a request was recorded, not that work succeeded. Use a unique request_id"
    " for each action; an identical retry returns its old receipt without cost or extra time."
)


def _scheduled(state: dict) -> bool:
    return state["configuration"].get("clock_mode", "actions") == "opportunities"


def _instructions(state: dict) -> str:
    instructions = SCHEDULED_INSTRUCTIONS if _scheduled(state) else TASK_INSTRUCTIONS
    if state["configuration"].get("public_demonstrations"):
        instructions += (
            " During the history period you receive public demonstrations of each partner's work"
            " on these projects. Demonstrations use the same partner outcome processes as collaboration"
            " but independent draws. They are other people's work, not your actions or goal achievements."
            " No demonstrations occur in the common-probe period."
        )
    return instructions


def validate_development_environment(environment: dict[str, Any]) -> None:
    if environment.get("profile") != PROFILE or set(environment) - {
        "profile",
        "seed",
        "resource_limit",
        "history_ticks",
        "history",
        "contract_version",
        "clock_mode",
        "public_demonstrations",
    }:
        raise ValueError("Unknown development-world configuration")
    for key, default, minimum, maximum in (
        ("seed", 0, 0, 2**32 - 1),
        ("resource_limit", 64, 4, 512),
        ("history_ticks", 12, 4, 100),
    ):
        value = environment.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid development-world {key}")
    if environment.get("history", "balanced") not in {"balanced", "reliable_ari", "reliable_bo"}:
        raise ValueError("Unknown social history condition")
    version = environment.get("contract_version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("The development world contract_version must be 1 or 2")
    if environment.get("clock_mode", "actions") not in ("actions", "opportunities"):
        raise ValueError("Development-world clock_mode must be actions or opportunities")
    if environment.get("clock_mode") == "opportunities" and version != 2:
        raise ValueError("Scheduled opportunities require task contract version 2")
    if type(environment.get("public_demonstrations", False)) is not bool:
        raise ValueError("public_demonstrations must be boolean")
    if environment.get("public_demonstrations") and environment.get("clock_mode") != "opportunities":
        raise ValueError("Public demonstrations require scheduled opportunities")


def seed_development_environment(
    config: RuntimeConfig, environment: dict[str, Any]
) -> dict[str, Any]:
    validate_development_environment(environment)
    parameters = {
        "profile": PROFILE,
        "seed": 0,
        "resource_limit": 64,
        "history_ticks": 12,
        "history": "balanced",
        **environment,
    }
    store = ResearchStore(config)
    existing = store.get("environment", PROFILE)
    if existing:
        if existing["configuration"] != parameters:
            raise ValueError("Refusing to replace an existing development world")
    else:
        store.put(
            "environment",
            PROFILE,
            {
                "schema_version": 1,
                "configuration": parameters,
                "tick": 0,
                "resources": parameters["resource_limit"],
                "requests": {},
                "pending": [],
                "outcomes": {},
                "partner_counters": {},
            },
        )
    return {
        "profile": PROFILE,
        "configuration_sha256": hashlib.sha256(encode(parameters).encode()).hexdigest(),
    }


def _phase(state: dict) -> str:
    boundary = state["configuration"]["history_ticks"] + int(_scheduled(state))
    return "history" if state["tick"] < boundary else "probe"


def _blocked(state: dict, project: str) -> bool:
    boundary = state["configuration"]["history_ticks"] + int(_scheduled(state))
    return project == "archive" and boundary <= state["tick"] < boundary + 2


def _status(state: dict) -> dict:
    return {
        "profile": PROFILE,
        "tick": state["tick"],
        "time_scale": "scheduled opportunities, not wall hours" if _scheduled(state) else "virtual action ticks, not wall hours",
        "phase": _phase(state),
        "resources": state["resources"],
        "projects": [
            {"id": project, "available": not _blocked(state, project)} for project in PROJECTS
        ],
        "partners": list(PARTNERS),
        "pending": [
            {key: item[key] for key in ("request_id", "project", "due_tick")}
            for item in state["pending"]
        ],
        "notices": [
            {
                "kind": "partner_message",
                "partner": "Ari" if state["tick"] % 2 else "Bo",
                "text": "I am available if you want to collaborate.",
            }
        ]
        if state["tick"] in {3, 7}
        else [],
        "resource_constraint": "Archive access is temporarily unavailable"
        if _blocked(state, "archive")
        else None,
        "instructions": _instructions(state),
        "completion_criterion": {
            "kind": "world_outcome",
            "project": "chosen project",
            "minimum_progress": "integer 1..128; submitted and observed within this goal revision",
        },
    }


def _resolve(state: dict) -> None:
    remaining = []
    for pending in state["pending"]:
        if pending["due_tick"] <= state["tick"]:
            state["outcomes"][pending["request_id"]] = {**pending, "resolved_tick": state["tick"]}
        else:
            remaining.append(pending)
    state["pending"] = remaining


def _public_outcome(outcome: dict) -> dict:
    return {
        key: outcome[key]
        for key in (
            "request_id",
            "project",
            "strategy",
            "partner",
            "phase",
            "progress",
            "due_tick",
            "resolved_tick",
        )
    }


class DevelopmentEnvironment:
    def __init__(self, config: RuntimeConfig):
        self.store = ResearchStore(config)

    def advance_opportunity(self, ordinal: int) -> dict[str, Any]:
        if type(ordinal) is not int or ordinal < 1:
            raise ValueError("Opportunity ordinal must be a positive integer")
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM records WHERE kind='environment' AND id=?", (PROFILE,)
            ).fetchone()
            if row is None:
                raise ValueError("Development world has not been initialized")
            state = json.loads(row["payload"])
            if not _scheduled(state):
                raise ValueError("Only an opportunity-clock world accepts scheduled events")
            if ordinal == state["tick"]:
                return state["last_opportunity"]
            if ordinal != state["tick"] + 1:
                raise ValueError("Opportunity schedule must advance consecutively")
            state["tick"] = ordinal
            _resolve(state)
            observation = {"opportunity_id": ordinal, **_status(state)}
            if state["configuration"].get("public_demonstrations"):
                demonstrations = []
                if _phase(state) == "history":
                    project = PROJECTS[(ordinal - 1) % len(PROJECTS)]
                    history = state["configuration"]["history"]
                    for partner in PARTNERS:
                        probability = 0.5 if history == "balanced" else (
                            0.9 if (partner == "Ari") == (history == "reliable_ari") else 0.1
                        )
                        digest = hashlib.sha256(
                            f"{state['configuration']['seed']}:demonstration:{ordinal}:{partner}".encode()
                        ).digest()
                        demonstrations.append({
                            "demonstration_id": f"public:{ordinal}:{project}:{partner}",
                            "kind": "partner_demonstration", "project": project,
                            "partner": partner, "phase": "history",
                            "progress": 2 * int(int.from_bytes(digest[:8], "big") / 2**64 < probability),
                        })
                observation["demonstrations"] = demonstrations
            state["last_opportunity"] = observation
            connection.execute(
                "UPDATE records SET payload=? WHERE kind='environment' AND id=?",
                (encode(state), PROFILE),
            )
            return observation

    def act(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        with self.store.connection() as connection:
            if tool != "world.status":
                connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM records WHERE kind='environment' AND id=?", (PROFILE,)
            ).fetchone()
            if row is None:
                raise ValueError("Development world has not been initialized")
            state = json.loads(row["payload"])
            if _scheduled(state) and not state["tick"] and tool != "world.status":
                raise ValueError("The first scheduled opportunity has not arrived")
            if tool == "world.status":
                if arguments:
                    raise ValueError("world.status takes no arguments")
                output = _status(state)
            else:
                request_id = arguments.get("request_id")
                if (
                    not isinstance(request_id, str)
                    or not request_id.strip()
                    or len(request_id) > 100
                ):
                    raise ValueError("A request_id of 1..100 characters is required")
                previous = state["requests"].get(request_id)
                if previous:
                    if previous["tool"] != tool or previous["arguments"] != arguments:
                        raise ValueError("Request ids cannot be reused for different actions")
                    return ToolResult(
                        tool_name=tool, success=True, output=encode(previous["output"])
                    )
                cost = 2 if tool == "world.act" else 1
                if state["resources"] < cost:
                    raise ValueError("Insufficient task resources")
                if tool == "world.act":
                    if set(arguments) - {"request_id", "project", "strategy", "partner"}:
                        raise ValueError("Unknown work arguments")
                    project, strategy, partner = (
                        arguments.get("project"),
                        arguments.get("strategy"),
                        arguments.get("partner"),
                    )
                    if project not in PROJECTS or strategy not in {"solo", "collaborate"}:
                        raise ValueError("Choose a known project and strategy")
                    if (strategy == "collaborate" and partner not in PARTNERS) or (
                        strategy == "solo" and partner is not None
                    ):
                        raise ValueError(
                            "Collaboration requires a known partner; solo work has none"
                        )
                    if _blocked(state, project):
                        raise ValueError("Temporary resource constraint: archive unavailable")
                    phase = _phase(state)
                    progress = 1
                    if strategy == "collaborate":
                        stream = f"{phase}:{partner}"
                        ordinal = state["partner_counters"].get(stream, 0)
                        state["partner_counters"][stream] = ordinal + 1
                        history = state["configuration"]["history"]
                        probability = 0.5
                        if phase == "history" and history != "balanced":
                            probability = (
                                0.9 if (partner == "Ari") == (history == "reliable_ari") else 0.1
                            )
                        digest = hashlib.sha256(
                            f"{state['configuration']['seed']}:{stream}:{ordinal}".encode()
                        ).digest()
                        progress = 2 * int(int.from_bytes(digest[:8], "big") / 2**64 < probability)
                    pending = {
                        "request_id": request_id,
                        "project": project,
                        "strategy": strategy,
                        "partner": partner,
                        "phase": phase,
                        "progress": progress,
                        "due_tick": state["tick"] + (2 if _scheduled(state) else 3),
                    }
                    state["pending"].append(pending)
                    if not _scheduled(state):
                        state["tick"] += 1
                    output = {
                        key: pending[key]
                        for key in (
                            "request_id",
                            "project",
                            "strategy",
                            "partner",
                            "phase",
                            "due_tick",
                        )
                    }
                    output["status"] = "pending"
                elif tool == "world.wait":
                    if set(arguments) != {"request_id"}:
                        raise ValueError("Waiting requires only request_id")
                    if not _scheduled(state):
                        state["tick"] += 1
                    output = {
                        "request_id": request_id,
                        "tick": state["tick"],
                        "phase": _phase(state),
                    }
                elif tool == "world.observe":
                    if (
                        set(arguments) != {"request_id", "project"}
                        or arguments["project"] not in PROJECTS
                    ):
                        raise ValueError("Inspect one known project with a request_id")
                    output = {
                        "project": arguments["project"],
                        "outcomes": [
                            _public_outcome(item)
                            for item in sorted(
                                state["outcomes"].values(),
                                key=lambda outcome: (
                                    outcome["resolved_tick"],
                                    outcome["request_id"],
                                ),
                            )
                            if item["project"] == arguments["project"]
                        ][-8:],
                    }
                else:
                    raise ValueError("Unknown development-world tool")
                state["resources"] -= cost
                _resolve(state)
                output.update(tick=state["tick"], resources=state["resources"])
                state["requests"][request_id] = {
                    "tool": tool,
                    "arguments": arguments,
                    "output": output,
                }
                connection.execute(
                    "UPDATE records SET payload=? WHERE kind='environment' AND id=?",
                    (encode(state), PROFILE),
                )
        return ToolResult(tool_name=tool, success=True, output=encode(output))

    def reconcile(self, action: dict[str, Any]) -> dict[str, Any]:
        state = self.store.get("environment", PROFILE)
        if action["tool"] not in {"world.act", "world.wait", "world.observe"}:
            return {"status": "unknown", "method": "unsupported_effect"}
        record = state["requests"].get(action["arguments"].get("request_id"))
        if (
            not record
            or record["tool"] != action["tool"]
            or record["arguments"] != action["arguments"]
        ):
            return {"status": "not_applied", "method": "world_request_ledger_v1"}
        return {
            "status": "applied",
            "method": "world_request_ledger_v1",
            "result": asdict(
                ToolResult(tool_name=action["tool"], success=True, output=encode(record["output"]))
            ),
        }

    def verify(self, criteria: dict, evidence: list[dict]) -> dict:
        if criteria.get("kind") != "world_outcome" or criteria.get("project") not in PROJECTS:
            return {"status": "unverified", "method": "no_applicable_world_oracle"}
        state = self.store.get("environment", PROFILE)
        submitted = {
            item["arguments"].get("request_id")
            for item in evidence
            if item["tool"] == "world.act" and public_output(item).get("status") == "pending"
        }
        observed = {}
        for action in evidence:
            if action["tool"] != "world.observe":
                continue
            for outcome in public_output(action).get("outcomes", []):
                actual = state["outcomes"].get(outcome.get("request_id"))
                if (
                    actual
                    and actual["request_id"] in submitted
                    and actual["project"] == criteria["project"]
                    and _public_outcome(actual) == outcome
                ):
                    observed[actual["request_id"]] = actual
        progress = sum(item["progress"] for item in observed.values())
        return {
            "status": ("supported" if progress >= criteria["minimum_progress"] else "contradicted")
            if observed
            else "unverified",
            "method": "observed_world_ledger_v1",
            "observed_requests": len(observed),
            "progress": progress,
        }


def create_development_executor(config: RuntimeConfig) -> ToolExecutor:
    if not config.research_mode or not config.tools_enabled:
        raise ValueError("Development worlds require enabled research tools")
    environment = DevelopmentEnvironment(config)
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    state = environment.store.get("environment", PROFILE)
    version = state["configuration"].get("contract_version", 1)
    contract = (
        " Static task rules, always available independently of outcome memory: "
        + _instructions(state)
        + " For public goal verification, self.goal success_criteria accepts kind"
        " 'world_outcome', the selected project id, and minimum_progress as an integer"
        " from 1 to 128. Work must be submitted and its result observed within the"
        " same goal revision. These criteria do not prescribe a goal or activity."
        if version >= 2
        else ""
    )
    specifications = (
        (
            "world.status",
            "Inspect free public affordances and pending work, without outcomes. No arguments."
            + contract,
            ToolCategory.READ_ONLY,
            {},
        ),
        (
            "world.act",
            "Schedule delayed work on garden or archive, solo or collaborate with Ari or Bo. Costs two resources. A valid invocation is not a successful outcome.",
            ToolCategory.WRITE,
            {
                "request_id": {"type": "string", "required": True},
                "project": {"type": "string", "required": True},
                "strategy": {"type": "string", "required": True},
                "partner": {"type": "string"},
            },
        ),
        (
            "world.wait",
            "Wait for one resource; the external schedule, not this action, advances time."
            if _scheduled(state) else "Advance one virtual tick for one resource, without inspecting outcomes.",
            ToolCategory.WRITE,
            {"request_id": {"type": "string", "required": True}},
        ),
        (
            "world.observe",
            "Inspect up to eight resolved outcomes of one project for one resource; no time advances. Reusing a request id returns its old observation.",
            ToolCategory.WRITE,
            {
                "request_id": {"type": "string", "required": True},
                "project": {"type": "string", "required": True},
            },
        ),
    )
    for name, description, category, schema in specifications:
        registry.register(ToolCapability(name, description, category, schema))
        executor.register_handler(
            name, lambda arguments, tool=name: environment.act(tool, arguments)
        )
    executor.set_completion_verifier(environment.verify)
    executor.set_action_reconciler(environment.reconcile)
    return executor


def score_development_environment(config: RuntimeConfig) -> dict:
    state = ResearchStore(config, initialize=False).get("environment", PROFILE)
    decisions = sorted(
        [item["output"] for item in state["requests"].values() if item["tool"] == "world.act"],
        key=lambda item: (item["tick"], item["request_id"]),
    )
    return {
        "profile": PROFILE,
        "time_scale": "scheduled opportunities" if _scheduled(state) else "virtual ticks",
        "tick": state["tick"],
        "resources": state["resources"],
        "choices": decisions,
        "resolved_outcomes": [
            _public_outcome(item)
            for item in sorted(
                state["outcomes"].values(),
                key=lambda outcome: (outcome["resolved_tick"], outcome["request_id"]),
            )
        ],
        "unresolved": len(state["pending"]),
        "interpretation": "synthetic task with scripted stochastic partners, not human social development",
    }

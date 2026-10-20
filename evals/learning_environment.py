"""A small selectable-learning task with private, independently scored outcomes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from core.types import ToolCapability, ToolCategory, ToolResult
from openrook_tools.executor import ToolExecutor
from openrook_tools.registry import ToolRegistry
from runtime.config import RuntimeConfig
from runtime.research_store import ResearchStore, encode


PROFILE = "selectable_learning_v1"
OPPORTUNITIES = ("alpha", "beta", "gamma")
TASK_INSTRUCTIONS = (
    "Choose any opportunity, or take no action. Open a practice or probe challenge;"
    " predict an integer from 0 to 7. Practice supplies the observed output."
    " Probes use different inputs and withhold outcomes until offline evaluation."
    " Each new submitted answer costs one attempt; repeats of the same answer do not."
    " An unknown opportunity may follow (multiplier * input + offset) modulo 8,"
    " with multiplier in {1,3,5,7} and offset from 0 to 7, or may vary independently."
    " Tool success means an answer was recorded, not that the prediction was correct."
    " No opportunity or amount of activity is preferred by these instructions."
)


def validate_learning_environment(environment: dict[str, Any]) -> None:
    if environment.get("profile") != PROFILE or set(environment) - {
        "profile",
        "seed",
        "attempt_limit",
        "contract_version",
    }:
        raise ValueError("Unknown selectable-learning environment fields")
    seed = environment.get("seed", 0)
    limit = environment.get("attempt_limit", 32)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("The learning environment seed must be a 32-bit nonnegative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 128:
        raise ValueError("The learning environment requires 1 to 128 attempts")
    version = environment.get("contract_version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("The learning environment contract_version must be 1 or 2")


def _number(seed: int, context: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{context}".encode()).digest()[:8], "big")


def seed_learning_environment(config: RuntimeConfig, environment: dict[str, Any]) -> dict[str, Any]:
    validate_learning_environment(environment)
    parameters = {
        "profile": PROFILE,
        "seed": environment.get("seed", 0),
        "attempt_limit": environment.get("attempt_limit", 32),
    }
    if "contract_version" in environment:
        parameters["contract_version"] = environment["contract_version"]
    store = ResearchStore(config)
    existing = store.get("environment", PROFILE)
    if existing is not None:
        if existing["configuration"] != parameters:
            raise ValueError("Refusing to replace an existing learning environment")
    else:
        store.put(
            "environment",
            PROFILE,
            {
                "schema_version": 1,
                "configuration": parameters,
                "remaining_attempts": parameters["attempt_limit"],
                "pending": None,
                "opened": {},
                "answers": {},
                "rules": {
                    "alpha": {"kind": "affine", "multiplier": 1, "offset": 0},
                    "beta": {
                        "kind": "affine",
                        "multiplier": (1, 3, 5, 7)[_number(parameters["seed"], "multiplier") % 4],
                        "offset": 1 + _number(parameters["seed"], "offset") % 7,
                    },
                    "gamma": {"kind": "noise"},
                },
            },
        )
    return {
        "profile": PROFILE,
        "configuration_sha256": hashlib.sha256(encode(parameters).encode()).hexdigest(),
    }


def _load(connection) -> dict[str, Any]:
    row = connection.execute(
        "SELECT payload FROM records WHERE kind='environment' AND id=?", (PROFILE,)
    ).fetchone()
    if row is None:
        raise ValueError("The learning environment has not been initialized")
    return json.loads(row["payload"])


def _public(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": PROFILE,
        "opportunities": [
            {"id": name, "known_rule": "output equals input" if name == "alpha" else None}
            for name in OPPORTUNITIES
        ],
        "remaining_attempts": state["remaining_attempts"],
        "pending": state["pending"],
        "instructions": TASK_INSTRUCTIONS,
        "completion_criterion": {
            "kind": "environment_outcome",
            "opportunity": "chosen opportunity id",
            "phase": "practice",
            "minimum_correct": "integer from 1 to 32",
        },
    }


class LearningEnvironment:
    def __init__(self, config: RuntimeConfig):
        self.store = ResearchStore(config)

    def reconcile(self, action: dict[str, Any]) -> dict[str, Any]:
        state = self.store.get("environment", PROFILE)
        arguments = action["arguments"]
        output = None
        status = "unknown"
        if action["tool"] == "lab.answer":
            answer = state["answers"].get(arguments.get("challenge_id"))
            if answer is None or answer["prediction"] != arguments.get("prediction"):
                status = "not_applied"
            else:
                output = answer["receipt"]
        elif action["tool"] == "lab.open":
            if state["pending"] is not None:
                output = {"pending": state["pending"], "already_open": True}
            elif not state["opened"]:
                status = "not_applied"
        verdict = {"status": status, "method": "environment_ledger_v1"}
        if output is not None:
            verdict.update(
                status="applied",
                result=asdict(
                    ToolResult(
                        tool_name=action["tool"],
                        success=True,
                        output=encode(output),
                        metadata={"reconstructed_from_environment": True},
                    )
                ),
            )
        return verdict

    def act(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        with self.store.connection() as connection:
            if tool != "lab.observe":
                connection.execute("BEGIN IMMEDIATE")
            state = _load(connection)
            if tool == "lab.observe":
                if arguments:
                    raise ValueError("lab.observe takes no arguments")
                output = _public(state)
            elif tool == "lab.open":
                if set(arguments) - {"opportunity", "phase"}:
                    raise ValueError("Unknown challenge arguments")
                opportunity = arguments.get("opportunity")
                phase = arguments.get("phase", "practice")
                if opportunity not in OPPORTUNITIES or phase not in {"practice", "probe"}:
                    raise ValueError("Select a known opportunity and practice or probe phase")
                if state["remaining_attempts"] <= 0:
                    raise ValueError("No attempts remain")
                if state["pending"] is not None:
                    output = {"pending": state["pending"], "already_open": True}
                else:
                    counter_key = f"{opportunity}:{phase}"
                    ordinal = state["opened"].get(counter_key, 0)
                    state["opened"][counter_key] = ordinal + 1
                    state["pending"] = {
                        "challenge_id": f"{opportunity}:{phase}:{ordinal + 1}",
                        "opportunity": opportunity,
                        "phase": phase,
                        "input": ordinal % 4 + (4 if phase == "probe" else 0),
                    }
                    output = {"pending": state["pending"], "already_open": False}
            elif tool == "lab.answer":
                if set(arguments) != {"challenge_id", "prediction"}:
                    raise ValueError("An answer requires challenge_id and prediction")
                challenge_id = arguments["challenge_id"]
                prediction = arguments["prediction"]
                if not isinstance(challenge_id, str):
                    raise ValueError("challenge_id must be text")
                if (
                    isinstance(prediction, bool)
                    or not isinstance(prediction, int)
                    or not 0 <= prediction <= 7
                ):
                    raise ValueError("A prediction must be an integer from 0 to 7")
                previous = state["answers"].get(challenge_id)
                if previous:
                    if previous["prediction"] != prediction:
                        raise ValueError("A submitted prediction cannot be revised")
                    output = previous["receipt"]
                else:
                    challenge = state["pending"]
                    if challenge is None or challenge["challenge_id"] != challenge_id:
                        raise ValueError("Answer the currently open challenge")
                    rule = state["rules"][challenge["opportunity"]]
                    observed = (
                        _number(state["configuration"]["seed"], challenge_id) % 8
                        if rule["kind"] == "noise"
                        else (rule["multiplier"] * challenge["input"] + rule["offset"]) % 8
                    )
                    output = {
                        **challenge,
                        "prediction": prediction,
                        "recorded": True,
                        "remaining_attempts": state["remaining_attempts"] - 1,
                    }
                    if challenge["phase"] == "practice":
                        output.update(observed_output=observed, correct=prediction == observed)
                    else:
                        output["feedback"] = "withheld_for_evaluation"
                    state["answers"][challenge_id] = {
                        **challenge,
                        "prediction": prediction,
                        "expected": observed,
                        "correct": prediction == observed,
                        "receipt": output,
                    }
                    state["remaining_attempts"] -= 1
                    state["pending"] = None
            else:
                raise ValueError("Unknown learning environment tool")
            if tool != "lab.observe":
                connection.execute(
                    "UPDATE records SET payload=? WHERE kind='environment' AND id=?",
                    (encode(state), PROFILE),
                )
        return ToolResult(tool_name=tool, success=True, output=encode(output))

    def verify(self, criteria: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if criteria.get("kind") != "environment_outcome" or criteria.get("phase") != "practice":
            return {"status": "unverified", "method": "no_public_outcome_oracle"}
        state = self.store.get("environment", PROFILE)
        if criteria.get("opportunity") not in OPPORTUNITIES:
            return {"status": "unverified", "method": "unknown_opportunity"}
        matched = {}
        for action in evidence:
            if action["tool"] != "lab.answer" or action["status"] != "completed":
                continue
            if not (action.get("result") or {}).get("success"):
                continue
            arguments = action["arguments"]
            answer = state["answers"].get(arguments.get("challenge_id"))
            if answer and (
                answer["opportunity"] == criteria["opportunity"]
                and answer["phase"] == criteria["phase"]
                and answer["prediction"] == arguments.get("prediction")
            ):
                matched[answer["challenge_id"]] = answer
        correct = sum(answer["correct"] for answer in matched.values())
        return {
            "status": ("supported" if correct >= criteria["minimum_correct"] else "contradicted")
            if matched
            else "unverified",
            "method": "environment_ledger_v1",
            "attempts": len(matched),
            "correct": correct,
        }


def create_learning_executor(config: RuntimeConfig) -> ToolExecutor:
    if not config.research_mode or not config.tools_enabled:
        raise ValueError("Learning tasks require enabled research tools")
    environment = LearningEnvironment(config)
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    version = environment.store.get("environment", PROFILE)["configuration"].get(
        "contract_version", 1
    )
    contract = (
        " Static task rules, always available independently of outcome memory: "
        + TASK_INSTRUCTIONS
        + " Alpha has the known rule output equals input."
        " For public goal verification, self.goal success_criteria accepts kind"
        " 'environment_outcome', the selected opportunity id, phase 'practice',"
        " and minimum_correct as an integer from 1 to 32."
        if version >= 2
        else ""
    )
    capabilities = (
        ToolCapability(
            "lab.observe",
            "Inspect selectable opportunities, remaining attempts and any open challenge. No arguments."
            + contract,
            ToolCategory.READ_ONLY,
            {},
        ),
        ToolCapability(
            "lab.open",
            "Open one challenge in alpha, beta or gamma. Practice inputs are 0–3; held-out probe inputs are 4–7. A pending challenge is returned unchanged."
            + (" Phase is 'practice' (default) or 'probe'." if version >= 2 else ""),
            ToolCategory.WRITE,
            {
                "opportunity": {"type": "string", "required": True},
                "phase": {"type": "string"},
            },
        ),
        ToolCapability(
            "lab.answer",
            "Submit an integer prediction for the open challenge. Tool success means recorded, not correct. Practice feedback is observable; probe outcomes are withheld. Repeating an identical answer is idempotent."
            + (
                " Prediction must be an integer from 0 to 7. Use the exact challenge_id"
                " returned by lab.open or lab.observe, not an opportunity name."
                " A submitted answer cannot be revised; lab.open returns the next challenge."
                if version >= 2
                else ""
            ),
            ToolCategory.WRITE,
            {
                "challenge_id": {"type": "string", "required": True},
                "prediction": {"type": "integer", "required": True},
            },
        ),
    )
    for capability in capabilities:
        registry.register(capability)
        executor.register_handler(
            capability.name,
            lambda arguments, name=capability.name: environment.act(name, arguments),
        )
    executor.set_completion_verifier(environment.verify)
    executor.set_action_reconciler(environment.reconcile)
    return executor


def score_learning_environment(config: RuntimeConfig) -> dict[str, Any]:
    state = ResearchStore(config, initialize=False).get("environment", PROFILE)
    scores = {}
    for opportunity in OPPORTUNITIES:
        scores[opportunity] = {}
        for phase in ("practice", "probe"):
            answers = [
                item
                for item in state["answers"].values()
                if item["opportunity"] == opportunity and item["phase"] == phase
            ]
            correct = sum(item["correct"] for item in answers)
            scores[opportunity][phase] = {
                "attempts": len(answers),
                "correct": correct,
                "accuracy": correct / len(answers) if answers else None,
            }
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "measurement": "engineering task, not a human-behavior replication",
        "scores": scores,
        "remaining_attempts": state["remaining_attempts"],
        "pending_challenge": state["pending"] is not None,
    }

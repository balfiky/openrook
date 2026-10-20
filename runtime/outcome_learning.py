"""Inspectable task learning from public receipts, never evaluator state or self-reports."""

from __future__ import annotations

import math
from typing import Any

from core.clock import clock as time
from runtime.research_store import ResearchStore, encode


FAMILY = [[multiplier, offset] for multiplier in (1, 3, 5, 7) for offset in range(8)]
MEMORY_CHARACTERS = 12000
RECORD_KINDS = (
    "prediction",
    "prediction_attempt",
    "outcome",
    "procedure",
    "procedure_version",
    "learning_state",
)


def present_evidence(record: dict[str, Any], version: int) -> dict[str, Any]:
    if (
        version < 2
        or not record.get("context", "").startswith("lab:")
        or record.get("source_type") not in {"public_tool_observation", "derived_procedure"}
    ):
        return record
    common = {
        "id": record["id"],
        "context": record["context"],
        "opportunity": record["context"].removeprefix("lab:"),
        "source_type": record.get("source_type"),
    }
    if record.get("source_type") == "public_tool_observation":
        return {
            **common,
            "input": record["input"],
            "observed_output": record["observed"],
            "evidence": "actual public environment output, not a previous guess",
        }
    if record.get("source_type") == "derived_procedure":
        return {
            **common,
            "version": record["version"],
            "status": record["status"],
            "formula": "output = (multiplier * actual_input + offset) modulo 8",
            "multiplier": record["rule"][0],
            "offset": record["rule"][1],
            "applicability": record["applicability"],
            "invalidation": record["invalidation"],
            "support_count": len(record["supporting_receipts"]),
            "counterexample_count": len(record["contrary_receipts"]),
        }
    return record


def enabled(config: Any) -> bool:
    return (
        config.research_memory_mode != "legacy"
        or config.research_progress_estimates
        or config.research_appraisal_mode != "legacy"
        or config.research_dispositions
    )


def _distribution(hypotheses: list[list[int]], task_input: int) -> list[float]:
    probabilities = [0.0] * 8
    for multiplier, offset in hypotheses or FAMILY:
        probabilities[(multiplier * task_input + offset) % 8] += 1 / len(hypotheses or FAMILY)
    return probabilities


def _uncertainty(hypotheses: list[list[int]]) -> float:
    return (
        sum(
            -sum(
                probability * math.log2(probability)
                for probability in _distribution(hypotheses, value)
                if probability
            )
            / 3
            for value in range(8)
        )
        / 8
    )


class OutcomeLearning:
    def __init__(self, store: ResearchStore):
        self.store = store

    def _existing_forecast(self, identifier: str) -> bool:
        previous = self.store.get("prediction", identifier)
        if previous and previous["status"] == "invocation_failed":
            self.store.put("prediction_attempt", previous["action_id"], previous)
            return False
        return previous is not None

    def before_action(self, action: dict[str, Any]) -> None:
        if action["tool"] == "world.act":
            self._predict_world(action)
            return
        if action["tool"] != "lab.answer":
            return
        arguments = action["arguments"]
        challenge_id = arguments.get("challenge_id")
        if not isinstance(challenge_id, str):
            return
        prediction_id = "lab:" + challenge_id
        if self._existing_forecast(prediction_id):
            return
        challenge = None
        for receipt in reversed(self.store.list("action")):
            if receipt["tool"] not in {"lab.open", "lab.observe"}:
                continue
            output = public_output(receipt)
            pending = output.get("pending")
            if isinstance(pending, dict) and pending.get("challenge_id") == challenge_id:
                challenge = pending
                break
        if challenge is None:
            return
        task_input = challenge.get("input")
        prediction = arguments.get("prediction")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 7
            for value in (task_input, prediction)
        ):
            return
        context = "lab:" + challenge["opportunity"]
        state = self.store.get("learning_state", context) or {}
        self.store.put_records(
            [
                (
                    "prediction",
                    prediction_id,
                    {
                        "schema_version": 1,
                        "id": prediction_id,
                        "context": context,
                        "action_id": action["id"],
                        "goal_id": action.get("goal_id"),
                        "goal_revision": action.get("goal_revision"),
                        "challenge": challenge,
                        "subject_prediction": prediction,
                        "probabilities": _distribution(state.get("hypotheses", FAMILY), task_input),
                        "source": "finite_affine_learner_v1",
                        "target": "observed_output",
                        "horizon": "this challenge's public feedback",
                        "status": "pending",
                        "created_at": time.time(),
                    },
                )
            ],
            event_kind="prediction_registered",
            event={"id": prediction_id, "action_id": action["id"]},
        )

    def after_action(self, action: dict[str, Any]) -> None:
        if (
            action["tool"] in {"lab.answer", "world.act"}
            and action.get("status") == "completed"
            and (action.get("result") or {}).get("success") is False
        ):
            key = "challenge_id" if action["tool"] == "lab.answer" else "request_id"
            identifier = action["arguments"].get(key)
            if isinstance(identifier, str):
                prediction = self.store.get(
                    "prediction", ("lab:" if key == "challenge_id" else "world:") + identifier
                )
                if (
                    prediction
                    and prediction["action_id"] == action["id"]
                    and prediction["status"] == "pending"
                ):
                    prediction.update(
                        status="invocation_failed",
                        receipt_id=action["id"],
                        outcome_status="unscored; invocation failure does not establish a task outcome",
                    )
                    self.store.put_records(
                        [("prediction", prediction["id"], prediction)],
                        event_kind="prediction_unscored",
                        event={"prediction_id": prediction["id"], "receipt_id": action["id"]},
                    )
            return
        if action["tool"] == "world.observe":
            self._observe_world(action)
            return
        if action["tool"] != "lab.answer":
            return
        output = public_output(action)
        challenge_id = output.get("challenge_id")
        if not isinstance(challenge_id, str):
            return
        prediction = self.store.get("prediction", "lab:" + challenge_id)
        if prediction is None or prediction["status"] != "pending":
            return
        challenge = prediction["challenge"]
        if any(output.get(key) != value for key, value in challenge.items()):
            return
        if output.get("prediction") != prediction["subject_prediction"]:
            return
        if challenge["phase"] == "probe":
            prediction["status"] = "withheld"
            self.store.put("prediction", prediction["id"], prediction)
            return
        observed = output.get("observed_output")
        if isinstance(observed, bool) or not isinstance(observed, int) or not 0 <= observed <= 7:
            return
        prediction.update(
            status="scored",
            observed=observed,
            brier=sum(
                (probability - int(index == observed)) ** 2
                for index, probability in enumerate(prediction["probabilities"])
            )
            / 2,
            subject_error=int(prediction["subject_prediction"] != observed),
            scoring_method="half_multiclass_brier_and_subject_zero_one_v1",
            receipt_id=action["id"],
        )
        outcome = {
            "schema_version": 1,
            "id": prediction["id"],
            "context": prediction["context"],
            "source_type": "public_tool_observation",
            "receipt_id": action["id"],
            "goal_id": prediction["goal_id"],
            "goal_revision": prediction["goal_revision"],
            "input": challenge["input"],
            "observed": observed,
            "subject_prediction": prediction["subject_prediction"],
        }
        records = self._update_affine(prediction, outcome)
        self.store.put_records(
            [
                ("prediction", prediction["id"], prediction),
                ("outcome", outcome["id"], outcome),
                *records,
            ],
            event_kind="outcome_learned",
            event={"prediction_id": prediction["id"], "receipt_id": action["id"]},
        )

    def _predict_world(self, action: dict[str, Any]) -> None:
        arguments = action["arguments"]
        request_id = arguments.get("request_id")
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(arguments.get("project"), str)
            or arguments.get("strategy") not in {"solo", "collaborate"}
        ):
            return
        identifier = "world:" + request_id
        if self._existing_forecast(identifier):
            return
        context = f"world:{arguments['project']}:{arguments['strategy']}:{arguments.get('partner') or 'none'}"
        state = self.store.get("learning_state", context) or {}
        observed = state.get("recent_outcomes", [])
        probability = (1 + sum(observed)) / (2 + len(observed))
        record = {
            "schema_version": 1,
            "id": identifier,
            "context": context,
            "action_id": action["id"],
            "goal_id": action.get("goal_id"),
            "goal_revision": action.get("goal_revision"),
            "arguments": arguments,
            "probability": probability,
            "status": "pending",
            "source": "windowed_beta_predictor_v1",
            "target": "progress > 0",
            "horizon": "first public observation after this request resolves",
            "created_at": time.time(),
        }
        self.store.put_records(
            [("prediction", identifier, record)],
            event_kind="prediction_registered",
            event={"id": identifier, "action_id": action["id"]},
        )

    def observe_environment(self, observation: dict[str, Any]) -> None:
        for demonstration in observation.get("demonstrations", []):
            if (
                not isinstance(demonstration, dict)
                or demonstration.get("kind") != "partner_demonstration"
                or demonstration.get("project") not in {"garden", "archive"}
                or demonstration.get("partner") not in {"Ari", "Bo"}
                or demonstration.get("phase") != "history"
                or type(demonstration.get("progress")) is not int
                or demonstration["progress"] not in {0, 2}
                or not isinstance(demonstration.get("demonstration_id"), str)
                or not 1 <= len(demonstration["demonstration_id"]) <= 100
            ):
                raise ValueError("Invalid public partner demonstration")
            request_id = demonstration["demonstration_id"]
            identifier = "world-event:" + request_id
            observed = {
                "request_id": request_id,
                "project": demonstration["project"],
                "partner": demonstration["partner"],
                "strategy": "collaborate",
                "phase": "history",
                "progress": demonstration["progress"],
                "kind": "partner_demonstration",
            }
            previous = self.store.get("outcome", identifier)
            if previous:
                if previous["observation"] != observed:
                    raise ValueError("Conflicting repeated public demonstration")
                continue
            context = f"world:{observed['project']}:collaborate:{observed['partner']}"
            state = self.store.get("learning_state", context) or {}
            self.store.put(
                "prediction",
                identifier,
                {
                    "schema_version": 1,
                    "id": identifier,
                    "context": context,
                    "goal_id": None,
                    "goal_revision": None,
                    "status": "pending",
                    "arguments": {key: observed[key] for key in ("project", "strategy", "partner")},
                    "probability": state.get("estimated_success_probability", 0.5),
                    "source": "prior_state_expectation_for_public_demonstration",
                    "timing": "computed from pre-update state, not a preregistered action forecast",
                    "target": "progress > 0",
                    "created_at": time.time(),
                },
            )
            self._observe_world(
                {"id": f"opportunity:{observation['opportunity_id']}"},
                observations=[observed],
                source_type="public_environment_observation",
            )

    def _observe_world(
        self,
        action: dict[str, Any],
        *,
        observations: list[dict] | None = None,
        source_type: str = "public_tool_observation",
    ) -> None:
        for observed in (
            public_output(action).get("outcomes", []) if observations is None else observations
        ):
            if not isinstance(observed, dict) or not isinstance(observed.get("request_id"), str):
                continue
            prefix = "world:" if source_type == "public_tool_observation" else "world-event:"
            identifier = prefix + observed["request_id"]
            prediction = self.store.get("prediction", identifier)
            if prediction is None or prediction["status"] != "pending":
                continue
            if any(
                prediction["arguments"].get(key) != observed.get(key)
                for key in ("project", "strategy", "partner")
            ):
                continue
            if type(observed.get("progress")) is not int or observed["progress"] not in {0, 1, 2}:
                continue
            result = int(observed["progress"] > 0)
            context = prediction["context"]
            previous = self.store.get("learning_state", context) or {}
            recent = [*previous.get("recent_outcomes", []), result][-16:]
            loss = (prediction["probability"] - result) ** 2
            losses = [*previous.get("recent_losses", []), loss][-16:]
            probability = (1 + sum(recent)) / (2 + len(recent))
            evidence = [*previous.get("evidence_ids", []), action["id"]][-16:]
            prediction = {
                **prediction,
                "status": "scored",
                "observed": result,
                "brier": loss,
                "scoring_method": "binary_brier_v1",
                "receipt_id": action["id"],
            }
            outcome = {
                "schema_version": 1,
                "id": identifier,
                "context": context,
                "source_type": source_type,
                "receipt_id": action["id"],
                "goal_id": prediction["goal_id"],
                "goal_revision": prediction["goal_revision"],
                "observation": observed,
            }
            state = {
                "schema_version": 1,
                "id": context,
                "context": context,
                "observations": previous.get("observations", 0) + 1,
                "recent_outcomes": recent,
                "recent_losses": losses,
                "evidence_ids": evidence,
                "estimated_success_probability": probability,
                "uncertainty": 4 * probability * (1 - probability),
                "estimated_controllability": probability,
                "controllability_basis": "windowed observed success frequency, not a causal estimate",
                "last_surprise": loss,
                "last_prediction_error": result - prediction["probability"],
                "prequential_progress": sum(losses[-8:-4]) / 4 - sum(losses[-4:]) / 4
                if len(losses) >= 8
                else None,
                "progress_method": "public prequential loss; not held-out transfer",
                "evidence_window": 16,
            }
            records = [
                ("prediction", identifier, prediction),
                ("outcome", identifier, outcome),
                ("learning_state", context, state),
            ]
            if len(recent) >= 6:
                from runtime.behavioral_dispositions import frequency_interval

                procedure = {
                    "schema_version": 1,
                    "id": context,
                    "context": context,
                    "version": state["observations"],
                    "status": "probabilistic",
                    "source_type": "derived_procedure",
                    "applicability": prediction["arguments"] | {"request_id": "any new request"},
                    "operation": "estimate probability of nonzero progress, not a prescribed action",
                    "estimated_probability": probability,
                    "frequency_interval": frequency_interval(sum(recent), len(recent)),
                    "supporting_receipts": evidence,
                    "counterexample_count": len(recent) - sum(recent),
                    "invalidation": "re-estimate with every public observation using last 16 outcomes; premises may change",
                }
                records.extend(
                    [
                        ("procedure", context, procedure),
                        ("procedure_version", f"{context}:{procedure['version']}", procedure),
                    ]
                )
            self.store.put_records(
                records,
                event_kind="outcome_learned",
                event={"prediction_id": identifier, "receipt_id": action["id"]},
            )

    def _update_affine(self, prediction: dict, outcome: dict) -> list[tuple[str, str, dict]]:
        context = outcome["context"]
        previous = self.store.get("learning_state", context) or {}
        hypotheses = previous.get("hypotheses", FAMILY)
        retained = [
            rule
            for rule in hypotheses
            if (rule[0] * outcome["input"] + rule[1]) % 8 == outcome["observed"]
        ]
        contradiction = not retained
        records = []
        procedure = self.store.get("procedure", context)
        if contradiction:
            retained = [
                rule
                for rule in FAMILY
                if (rule[0] * outcome["input"] + rule[1]) % 8 == outcome["observed"]
            ]
            if procedure and procedure["status"] == "supported":
                procedure = {
                    **procedure,
                    "status": "invalidated",
                    "contrary_receipts": [*procedure["contrary_receipts"], outcome["receipt_id"]],
                }
                records.extend(
                    [
                        ("procedure", context, procedure),
                        ("procedure_version", f"{context}:{procedure['version']}", procedure),
                    ]
                )
        evidence = [] if contradiction else previous.get("regime_evidence", [])
        inputs = [] if contradiction else previous.get("distinct_inputs", [])
        evidence = [*evidence, outcome["receipt_id"]][-32:]
        inputs = sorted(set([*inputs, outcome["input"]]))
        losses = [*previous.get("recent_losses", []), prediction["brier"]][-16:]
        progress = (sum(losses[-8:-4]) / 4 - sum(losses[-4:]) / 4) if len(losses) >= 8 else None
        state = {
            "schema_version": 1,
            "id": context,
            "context": context,
            "observations": previous.get("observations", 0) + 1,
            "subject_correct": previous.get("subject_correct", 0) + 1 - prediction["subject_error"],
            "hypotheses": retained,
            "regime_evidence": evidence,
            "distinct_inputs": inputs,
            "regime": previous.get("regime", 0) + int(contradiction),
            "contradictions": previous.get("contradictions", 0) + int(contradiction),
            "last_hypotheses_eliminated": len(hypotheses) - len(retained)
            if not contradiction
            else None,
            "recent_losses": losses,
            "prequential_progress": progress,
            "progress_method": "previous_four_minus_recent_four_public_forecast_loss; not held-out transfer",
            "uncertainty": _uncertainty(retained),
            "last_surprise": prediction["brier"],
            "last_prediction_error": 1
            - prediction["subject_error"]
            - prediction["probabilities"][prediction["subject_prediction"]],
            "estimated_controllability": 1 - _uncertainty(retained),
            "controllability_basis": "task predictability under finite hypothesis family, not causal agency",
            "evidence_ids": evidence,
            "status": "changed_or_out_of_family" if contradiction else "consistent_with_family",
        }
        if len(retained) == 1 and len(inputs) >= 3:
            version = (procedure or {}).get("version", 0) + int(
                not procedure
                or procedure["status"] != "supported"
                or procedure["rule"] != retained[0]
            )
            learned = {
                "schema_version": 1,
                "id": context,
                "version": version,
                "status": "supported",
                "source_type": "derived_procedure",
                "context": context,
                "rule": retained[0],
                "applicability": "same opportunity, current observed regime, integer inputs 0..7",
                "operation": "(multiplier * input + offset) modulo 8",
                "supporting_receipts": evidence,
                "contrary_receipts": [],
                "invalidation": "any contradictory public output; no assertion about hidden probes",
            }
            records.extend(
                [
                    ("procedure", context, learned),
                    ("procedure_version", f"{context}:{version}", learned),
                ]
            )
        records.append(("learning_state", context, state))
        return records

    def context(
        self,
        *,
        mode: str,
        goal_id: str | None = None,
        selected_context: str | None = None,
        progress: bool = False,
        presentation_version: int = 1,
        focused: bool = False,
    ) -> dict[str, Any]:
        memory: dict[str, Any] = {
            "mode": mode,
            "records": [],
            "truncated": False,
            "character_limit": MEMORY_CHARACTERS,
        }
        outcomes = self.store.list("outcome")
        outcomes.sort(
            key=lambda item: (
                item["context"] == selected_context,
                item.get("goal_id") == goal_id if goal_id else False,
            )
        )
        candidates = list(reversed(outcomes)) if mode in {"episodes", "procedures", "raw"} else []
        if mode == "procedures":
            procedures = self.store.list("procedure")
            procedures.sort(key=lambda item: item["context"] == selected_context)
            candidates = list(reversed(procedures)) + candidates
        if progress and mode != "raw":
            estimates = [
                {
                    key: value
                    for key, value in state.items()
                    if key
                    not in {"hypotheses", "regime_evidence", "recent_losses", "distinct_inputs"}
                }
                for state in self.store.list("learning_state")
            ]
            candidates = estimates + candidates
        if focused:
            candidates = [item for item in candidates if item["context"] == selected_context]
        for candidate in candidates[:8]:
            candidate = present_evidence(candidate, presentation_version)
            if (
                len(encode({**memory, "records": [*memory["records"], candidate]}))
                > MEMORY_CHARACTERS
            ):
                memory["truncated"] = True
                continue
            memory["records"].append(candidate)
        memory["truncated"] = memory["truncated"] or len(candidates) > 8
        return memory


def public_output(action: dict[str, Any]) -> dict[str, Any]:
    import json

    result = action.get("result") or {}
    if action.get("status") != "completed" or result.get("success") is not True:
        return {}
    try:
        output = json.loads(result.get("output", ""))
    except (TypeError, ValueError):
        return {}
    return output if isinstance(output, dict) else {}

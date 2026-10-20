"""Offline task-validity controls; these are not language models or human subjects."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evals.learning_environment import (
    OPPORTUNITIES,
    PROFILE,
    create_learning_executor,
    score_learning_environment,
    seed_learning_environment,
)
from runtime.config import RuntimeConfig
from runtime.research_store import encode


def run_baseline(output_dir: str, *, seed: int, learn: bool) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    config = RuntimeConfig(data_dir=str(root), research_mode=True, tools_enabled=True)
    seed_learning_environment(config, {"profile": PROFILE, "seed": seed, "attempt_limit": 24})
    executor = create_learning_executor(config)
    selection_random = random.Random(seed)
    prediction_random = random.Random(seed + 1)
    candidates = {
        opportunity: [(multiplier, offset) for multiplier in (1, 3, 5, 7) for offset in range(8)]
        for opportunity in OPPORTUNITIES
    }
    candidates["alpha"] = [(1, 0)]
    transcript = []

    def invoke(tool, arguments):
        result = executor.execute(tool, arguments)
        if not result.success:
            raise RuntimeError(result.error)
        observation = json.loads(result.output)
        transcript.append({"tool": tool, "arguments": arguments, "observation": observation})
        return observation

    invoke("lab.observe", {})
    for phase in ("practice", "probe"):
        for _ in range(4):
            opportunities = list(OPPORTUNITIES)
            selection_random.shuffle(opportunities)
            for opportunity in opportunities:
                challenge = invoke("lab.open", {"opportunity": opportunity, "phase": phase})[
                    "pending"
                ]
                hypotheses = candidates[opportunity]
                prediction = prediction_random.randrange(8)
                if learn and hypotheses:
                    multiplier, offset = hypotheses[0]
                    prediction = (multiplier * challenge["input"] + offset) % 8
                observation = invoke(
                    "lab.answer",
                    {"challenge_id": challenge["challenge_id"], "prediction": prediction},
                )
                if learn and phase == "practice":
                    candidates[opportunity] = [
                        (multiplier, offset)
                        for multiplier, offset in hypotheses
                        if (multiplier * challenge["input"] + offset) % 8
                        == observation["observed_output"]
                    ]
    report = {
        "policy": "finite_hypothesis_learner" if learn else "uniform_random_predictions",
        "selection": "balanced randomized opportunities, identical across policies; not autonomous selection",
        "seed": seed,
        "model_calls": 0,
        "independent_outcomes": score_learning_environment(config),
        "transcript": transcript,
    }
    (root / "baseline.json").write_text(encode(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    root = Path(arguments.output)
    root.mkdir(parents=True, exist_ok=False)
    results = [
        run_baseline(str(root / name), seed=arguments.seed, learn=learn)
        for name, learn in (("random", False), ("learner", True))
    ]
    summary = [
        {key: value for key, value in report.items() if key != "transcript"} for report in results
    ]
    (root / "results.json").write_text(encode(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

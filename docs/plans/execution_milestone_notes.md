# First experience grounded execution milestone

Implemented 19 October 2026. This delivers an engineering slice of Phase 0 and
Phase 1 of the [autonomy plan](experience_grounded_autonomy_plan.md). It does not
install a new personality, affect theory, intrinsic-reward formula, or fly model.

This is the historical first-slice record. See the [subsequent implementation](autonomy_milestones_implementation.md)
for learning, appraisal, social-world, dependency, and study-control additions;
the open limitations below describe this first slice, not the newer optional modes.

## Implemented mechanisms

- Opt-in closed-loop execution selects one tool action, observes its receipt,
  and chooses again within a configured activation limit. Batch mode remains
  available for comparison; default behavior and historical manifests retain it.
- Versioned goal execution records retain criteria, continuation, expected
  results, step counts, evidence references, and waiting conditions. Restart
  restores evidence instead of blindly replaying effects.
- Subject completion claims and independently supported outcomes are distinct.
  Criteria revisions exclude prior evidence; subjective completion stays valid
  as a subjective judgment. The initial external verifier covers the controlled
  environment, not arbitrary semantic claims.
- Time/external-input wake conditions suppress unnecessary model calls across
  restarts. The existing polling scheduler remains the event-delivery mechanism.
- Full-context and task-only adapters use the same execution infrastructure.
  Task-only autonomous prompts exclude affect/identity context and skip identity
  learning; it is a wrapper comparison, not a separately trained model.
- A private selectable-learning environment supplies practice, held-out probes,
  independent outcome scoring, persistent finite resources, and idempotent
  submissions. Invalid actions do not advance its state. Probe answers and
  private rules never become agent-visible tool receipts.
- Replay includes execution records, completion assessments, and wake state.
  Its recorded-oracle substitution validates reconstruction, not a second
  independent execution of the environment.

## Controls and prospective comparison

[Baseline identification](../research_execution_baseline.json) records the
pre-change source hashes and 117 passing focused regression tests. These hashes
identify source versions, not a self-contained archived checkout. New run
manifests additionally identify their complete current implementation.

`python -m evals.learning_baselines` runs a simple hypothesis-elimination learner
and a random predictor on identical balanced opportunity schedules. The learner
uses the same hypothesis family disclosed in the environment instructions.
These non-LLM controls test learnability and scoring; their competence is not
attributed to OpenRook. They are not an autonomous motivation comparison.

The [prospective protocol](../research_learning_execution.protocol.json) crosses
full/task-only context with batch/closed-loop execution and mirrors condition
order across two subjects per cell. Calls, receipts, independent outcomes,
completion claims, and infrastructure failures remain separate measures. Compare
at matched cumulative resource use rather than treating extra actions or calls
as an architectural improvement.

The task is intentionally small and synthetic. It is not the published human
goal-selection protocol, and no human-equivalence comparison is claimed. Missing
probes reflect unobserved capability, not proof that the subject cannot learn.

## Validation results

- Non-UAT suite: **2,374 passed, 11 expected failures**, with two existing
  deprecation warnings. The settings-save completeness guard caught the new
  configuration fields; both compatible execution/feedback combinations now
  round-trip without resetting them.
- Latest focused execution, environment, observer, and settings checks:
  **51 passed**, including lost acknowledgments, reconciliation, atomic goal
  updates, budget exhaustion, verifier failure, restarts, and replay.
- [Scripted engineering smoke](../../reports/research-comparison/20261019-execution-engineering-smoke-v2/validation.json):
  eight isolated subjects, 96 scripted backend requests, 64 replay frames,
  zero infrastructure failures, zero replay mismatches, and no external tools
  executed during replay. The backend is explicitly injected; these are not
  live-model behavior results.
- [Non-LLM task controls](../../reports/research-comparison/20261019-execution-task-controls/results.json):
  the finite-hypothesis learner predicts all four held-out beta inputs correctly
  at seed 17; the matched random predictor gets zero of four. This small control
  validates task learnability, not OpenRook's learning or a population effect.
- Focused lint, formatting, whitespace, protocol, local-link, and credential
  checks pass. The existing late router import in `interface/api.py` still
  triggers Ruff E402; this unrelated import-layout issue is unchanged.

No paid or live-model inference was used for this implementation's validation.

## Open gates and limitations

Live-model behavior still needs a bounded, credentialed smoke comparison and
adequate independent subjects before any behavioral conclusion. No long study
is started by this implementation. Establish token/spend bounds before paid
evaluation; the prepared protocol's application-call ceiling is not a price cap.

Unknown external effects conservatively block further mutations while permitting
inspection. The subject can request task-ledger reconciliation through
`self.check_action`, without administrator edits or blind retries. General
external-service reconciliation, dependency/subgoal graphs, semantic file
verification, push-based wake scheduling, learned procedures, prediction scoring,
and targeted long-term retrieval are not claimed as implemented here.

The existing affect model remains unchanged; its tool appraisal still distinguishes
tool invocation success rather than independently scoring task correctness.
Whether experience produces durable
behavioral individuality, whether affect has a useful causal role, and whether
either resembles human development remain questions for subsequent controlled
experiments, not acceptance criteria satisfied by these engineering tests.

See [runtime instructions](../RESEARCH_RUNTIME.md) for configuration and commands.

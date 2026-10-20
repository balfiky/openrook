# Experience-grounded autonomy: implementation and validation

Implemented 19 October 2026 against the [research plan](experience_grounded_autonomy_plan.md).
These are executable experimental mechanisms. The final controlled-world
retained-choice acceptance gate passes; human equivalence, biological emotion
and consciousness are not the measured claim.
All new switches default to the previous behavior. Historical manifests without
the switches retain those defaults; an existing subject is not silently reconfigured.

The [retained-choice closeout](#retained-choice-closeout) records the latest
engineering checks and held-out live confirmation. Earlier failed or inconclusive
experiments below remain historical evidence, not the final retention result.

## Delivered mechanisms

### Execution and commitments

Closed-loop goals now accept optional parent links and dependencies. Cycles and
unknown references are rejected before writing. A dependent activity waits for
supported or explicitly subjective completion, not an unsupported success claim.
The subject may remove dependencies, revise criteria, abandon work, or rest.
Revisions preserve provenance and tool-attempt accounting. No administrator has
to approve a goal, preference, or relationship.

### Phase 2: evidence-backed learning

`runtime/outcome_learning.py` registers predictions before supported task actions.
Public receipts score categorical or binary Brier loss, separately from successful
tool invocation and separately from the model's submitted point prediction.
Hidden learning-task probe answers are never read by the online learner. Repeated
receipts cannot reinforce the same outcome. Pending forecasts survive restart;
observations are linked to action receipts and goal revisions.

The small affine hypothesis learner produces conditional, versioned procedures
after evidence on multiple inputs. Contradictions invalidate old procedures;
replacement rules retain the invalidated version and its counterexamples.
Social expectations use the last sixteen publicly observed outcomes, with
conditional frequency estimates, counterexamples and descriptive intervals.
Neither mechanism trains the hosted model or runs generated code.

Memory controls expose no retained **outcome** memory, episodes, or procedures,
with a shared 12,000-character/eight-record ceiling and a separate last operational
receipt. Retrieval prioritizes the selected goal/context instead of only the
last eight actions globally. This matches maximum memory allowance, not exact
prompt token length; costs must still be compared. `none` is an outcome-memory ablation, **not a fresh
mind**: commitments can still carry information. Use a new subject for a genuinely
fresh starting condition; do not call this switch full amnesia. Mechanism protocols
use task-only context to exclude the legacy identity/affect memory pathways.

Optional progress estimates report previous-versus-recent public prequential loss,
hypothesis elimination, uncertainty and contradictions. These are not held-out
transfer scores or proof that a noisy process is irreducibly unlearnable. The
finite learner's assumptions and regime resets are explicit experimental priors.
The LLM still chooses its activity; no curiosity reward forces it toward a task.
Identity digests in these modes label interpretations as self-inference, with
receipt provenance, rather than upgrading a generated summary into external fact.

### Phase 3: functional appraisal comparison

`runtime/grounded_appraisal.py` derives goal relevance, outcome-minus-expectation,
uncertainty, surprise and estimated controllability from public outcome records.
The estimate of control is task predictability/success frequency, not a measured
causal effect of agency. An explicit experimental formula connects these quantities
to **one decision site: prioritizing bounded evidence retrieval**.

- `legacy`: existing affect/tool appraisal, unchanged.
- `expression`: the same estimates and appraisal labels, without that connection.
- `attention`: appraisal connected to evidence selection.
- `neutral`: the identical selection rule and numerical information, without emotion labels.

No extra inference requests implement the connection. New appraisal modes do not
also reward invocation success through the legacy autonomous tool-affect path.
They do not rewrite traits, install simulated pain, demand attachment, or force
activity. The neutral controller deliberately can match attention exactly; that
is a control against claiming emotion is necessary merely by naming an algorithm.
This is a functional intervention, not a biologically calibrated emotion model.

### Phase 4: persistent world and behavioral individuality

`evals/development_environment.py` adds two competing projects, finite observation
and work resources, delayed outcomes, partial inspection, interruptions, a temporary
resource constraint, and two independently scripted stochastic partners. Virtual
action ticks are not elapsed wall-clock hours. Private future outcomes and partner
probabilities never enter receipts. Idempotent request ids preserve costs and
effects across retries; ledger reconciliation resolves supported lost acknowledgments.
Completion needs both submitted work and subsequently observed outcomes in the
same goal revision; the verifier cannot reveal unobserved outcomes.

Opposed history conditions change partner reliability, then use identical common-probe
partner streams at paired seeds. Choice distributions, resolved outcomes, and costs
are scored offline independently of the model. Social expectations can change with
new outcomes rather than remaining fixed after the initial history.

`runtime/behavioral_dispositions.py` estimates contextual choice frequencies only
from actual accepted work choices, with counterexamples and uncertainty. Estimates
remain provisional until eight choices. They are descriptions conditional on chosen
projects, not diagnoses, independent human measurements, or desired personality
settings. Whether they improve retained behavioral individuality is still untested
with live models; repeated dependent choices do not become independent subjects.

### Phase 5: preregistration, resource accounting and gates

`evals/study_design.py` validates one-factor, two-arm comparisons, counterbalanced
order, distinct paired seeds, matched affordances, a primary endpoint, exclusions,
minimum observations, an effect threshold, and fixed stopping rules. Analysis uses
paired subjects, reports costs and missing probes separately, and does not promote
turns into participants. Bootstrap intervals and normal-approximation sample planning
are explicitly approximate. A small or zero-variance pilot cannot justify a powered
confirmatory sample by itself. Missing probes are missing evidence, not failed ability.

Confirmatory execution requires a hashed informative live pilot with the same
mechanism/endpoint and held-out seeds. A 24-hour-or-longer bounded protocol additionally
requires hashed, successful live confirmatory analysis and engineering validation.
No human baseline is synthesized, no provider replication is claimed without another
backend, and no long run starts just because unit tests pass. Re-run a selected
comparison with another OpenAI-compatible backend configuration as a separate
replication; do not pool its subjects as if the model itself were independently replicated.

`evals/study_budget.py` atomically reserves study-wide model calls, input-token
estimates, requested output tokens, tool attempts, declared-price cost and wall time.
Failed requests consume reservations; restarts do not reset them. The supported live
adapter disables transport retries, bounds request timeout, and records provider usage
and attempts when available. Resource stops are distinct from infrastructure failure
and deliberate rest, and are replayable. Three consecutive failed cycles stop bounded
studies. Native tool/model bypasses are excluded from the bounded task profiles.

Input reservations use UTF-8 bytes plus a framing allowance, **not a verified provider
tokenizer**. Output limits rely on the provider honoring the requested maximum.
Declared-price reservations are not a certified account billing cap. Excess reported
usage stops subsequent requests; use a provider-side hard spend cap before a paid
long study. Zero-price fixture protocols cannot launch a hosted paid backend. Model
request observations can include a locally denied request; authorized backend calls,
transport attempts and usage are therefore reported separately. An in-flight request
can finish during orderly shutdown, bounded by its configured timeout.

## Runnable comparisons

### Live response-contract hardening

Live tests after the milestone fixtures exposed unsupported deliberation actions,
singleton tool objects, positional tool tuples, and incomplete Markdown fences.
These failures and unsuccessful intermediate runs remain in their original reports;
they were not overwritten by the successful rerun.

New subjects record decision contract version 2 and action contract version 4 in
their immutable manifests. Invalid decisions and tool proposals each receive at
most one logged model retry through the existing inference budgets. The model
chooses the replacement decision; the runtime does not translate `observe` into
`pursue`, prescribe a goal, or retry valid inactivity. Re-deliberation can change a
choice, so the rejected response and replacement remain separate evidence.
An explicit singleton tool object can be wrapped in an array without changing
its tool or arguments. Positional tuples and bare names are not executed. A
Markdown opener is removable only when the remaining JSON is complete; truncated
JSON is not completed locally. Repeated invalid output still fails closed.

`research_json_schema: true` optionally requests provider-side JSON schemas for
decisions and tool proposals using the [compatible-serving response format](https://docs.vllm.ai/en/latest/features/structured_outputs/).
It constrains the wire format, not beliefs, goals, emotional tone, or activity;
rest, wait and empty tool arrays remain available. The setting defaults off for
unverified providers and is frozen per subject. Bounded requests do not silently
downgrade when a provider rejects the schema. Schema bytes count toward the
study input reservation, schemas are recorded with each request, and replay
checks that the requested schema matches the recording. Older subjects retain
their original response contracts, including the historical failures.

The supplied hosted endpoint passed a [schema-bearing capability probe](../../reports/backend-health/20261019T092055Z-structured-probe.json).
With schema mode enabled, the [new isolated live smoke](../../reports/backend-health/20261019T092801Z-schema-contract-runtime/summary.json)
completed two activations across a restart: **8 HTTP 200 inference requests,
6 successful tool calls, zero infrastructure failures, and 2 replay frames with
zero mismatches**. All eight responses satisfy their recorded schemas under
[checks covering every schema keyword used in this run](../../reports/backend-health/20261019T092801Z-schema-contract-runtime/declared-schema-validation.json).
One public prediction was scored; no procedure formed and the goal remains
active/unverified. Successful tool submission is not necessarily a correct
prediction. This is a functional smoke, not evidence of humanlike independence
or a treatment effect. The four earlier failing live runs also
[replay unchanged](../../reports/backend-health/20261019T092915Z-all-historical-contract-replays/validation.json).

### Prospective studies

The first unmetered live memory pilot was stopped for a task-contract visibility
defect after 80 real backend requests. The basic arithmetic rules and argument
constraints appeared only in a transient observation; memory ablations then
removed them from later prompts. This confounded retained experience with access
to the task instructions. The partial run is preserved at
[its original report](../../reports/research-comparison/20261019T102758Z-live-memory-unmetered/summary.json);
22 frames replay, but missing endpoints and the integrity stop invalidate a
behavioral comparison. It is not a successful learning result.

Learning environment `contract_version: 2` keeps the same public task rules in
the always-visible tool catalog, including the answer range, exact challenge-id
requirement and available public completion criterion. These are static rules,
not retained observations or hidden answers; they are identical across seeds
and memory conditions. Version 1 remains available for historical subjects.
The same visibility correction applies to the development world's public timing,
cost, request-id and verification rules. All seven prospective protocols now
explicitly select contract version 2 and must use new subjects rather than
upgrading a continuing participant. History assignments and future outcomes remain private.

The [fresh four-subject memory run](../../reports/research-comparison/20261019T103518Z-live-memory-contract-v2/summary.json)
completed 213 real inference requests and 64 activations, without infrastructure
failures, resource stops, or replay mismatches. Every subject had too few voluntary
held-out trials for the preregistered endpoint. Thus the outcome is insufficient
behavioral evidence, not failure to execute the study and not evidence of no learning.

`evals.learning_assessment` provides a separately recorded, read-only diagnostic
for this missing-measurement problem. It compares the source subject's retained
outcome-memory policy, no retained outcome evidence, and full public outcome history
on fixed beta inputs. The full-history arm is a capability reference, not a matched
context-budget treatment. These are instructed, stateless readouts, not autonomous
choices; they omit goals and conversational memory. They do not replace the original
primary endpoint, become extra subjects, teach withheld answers to a participant,
or qualify as confirmatory evidence. Requests and declared limits are recorded
before execution, all inference passes through the study budget, repeated failures
stop the diagnostic, and source database hashes must remain unchanged.

The initial readout diagnostic found poor predictions even with full public
history. A [direct backend control](../../reports/backend-health/20261019T105241Z-arithmetic-isolation/summary.json)
answered all 64 arithmetic items correctly, with and without JSON schemas.
A [paired presentation diagnostic](../../reports/backend-health/20261019T105504Z-memory-presentation-isolation/summary.json)
then found that explicit formula parameters and observed-output fields improved
procedure-based readouts in these few snapshots, but did not reliably fix episodic
readouts. This is exploratory evidence of context sensitivity, not a population
effect or evidence that the backend is generally incapable.

New subjects therefore freeze `memory_presentation_version: 2`: lab outcome views
separate actual outputs from old guesses and label procedure parameters by name.
Invalidated rules stay invalidated, evidence ids still link to the complete stored
records, and memory limits still apply. Historical subjects keep version 1.
`self.recall` lets a new subject voluntarily retrieve bounded evidence for a chosen
context without reading private environment state or changing its goal. It obeys
the same episodes/procedures/none policy and tool budget, so it cannot defeat the
no-outcome-memory control. Clearer rendering and optional recall are a combined
engineering change; subsequent results must not attribute their combined effect
to one component alone.

The [protocol directory](../research_protocols) contains separate small prospective
pilots for memory, outcome-memory ablation, progress, connected appraisal, the neutral
control, dispositions and divergent histories. They do not vary all switches at once.
Each has four subjects, paired/counterbalanced conditions, a restart, and at most
256 authorized backend calls **for that protocol**. Two pairs are an infrastructure
smoke, not enough for the six-pair minimum interval or confirmatory planning. The
first live smoke should use one protocol, not run the entire directory and quietly
multiply the agreed first-smoke budget. Prices are deliberately zero placeholders;
declare verified upper-bound pricing before hosted use and keep credentials outside protocols.

For an operator-confirmed free, self-hosted service, set
`resource_limits.billing_mode` to `unmetered` and keep all three monetary fields
zero. This declaration is stored in the immutable protocol and durable budget;
it does not bypass the live backend adapter, inference counts, token reservations,
tool limits, timeouts, or wall-clock limit. Without this explicit declaration,
hosted services still require positive declared rates and a monetary ceiling.

```sh
python -m evals.study_design --protocol docs/research_protocols/memory.json
python -m evals.research_job --config runtime_config.research.yaml \
  --protocol docs/research_protocols/memory.json --output reports/new-memory-pilot
```

The first command validates only; it makes no model requests. Never reuse an output
directory or modify a continuing subject to create a comparison. The runtime and
passive observation API expose predictions, procedures, learning estimates, attention
and dispositions, while keeping private environment state out of agent context.

## Validation record

- Non-UAT regression suite after response-contract hardening: **2,496 passed,
  11 expected failures**, with two existing deprecation warnings. This includes
  76 additional regression cases. Focused response-contract, schema transport,
  budgeting, replay and configuration checks: **116 passed**.
- [Seven scripted comparisons](../../reports/research-comparison/20261019-autonomy-milestones-engineering/summary.json):
  **28 isolated subjects, 448 replay frames, zero mismatches, zero infrastructure
  failures, 1,052 fixture backend requests, and zero live-model inference calls**.
  Each comparison stays within its own declared cap. All analyses correctly
  report insufficient behavioral evidence; these are scripted diagnostics.
- Failure checks additionally cover unscored failed invocations and preserved
  retry forecasts, immutable budget input, redirect/retry suppression, resource
  stops with active learning, and matching analysis/validation provenance.
- Seven prospective protocol files validate without credentials or network access.
  New-file formatting, focused lint and whitespace checks pass. Existing E402
  import-layout findings in `interface/api.py` and `runtime/llm/backend.py`, and
  existing F401 unused-import findings in `nur_tools/native_orchestrator.py`,
  are excluded from scoped lint; unrelated imports are unchanged.
- The initial local-backend check did not test the user-supplied hosted endpoint.
  A subsequent [hosted health check](../../reports/backend-health/20261019T084524Z-hosted.json)
  authenticated successfully, listed `deepseek-v4-flash`, and returned the expected
  JSON through the production backend adapter in one HTTP 200 generation request
  (31 input tokens, 13 output tokens, approximately 0.87 seconds).
- A separate [live runtime smoke](../../reports/backend-health/20261019T084758Z-runtime/summary.json)
  used one isolated subject, two activations separated by a restart, and a global
  eight-request ceiling. Its three actual generation requests all returned HTTP
  200, but **runtime validation failed**: the first decision used unsupported
  action `observe`. The second selected a goal, then returned an empty tool array;
  that empty array is valid inactivity, not a tool or infrastructure failure.
  No tools executed or procedures formed. Both recorded frames replayed without
  mismatches, which reconstructs the failure rather than establishing success.
  The injected wrapper in this diagnostic called the real backend, not a fixture;
  its artifacts are not eligible evidence for the confirmatory-study gate.
- Those initial four live requests verify connectivity and expose a decision-contract
  reliability gap; they do not establish behavioral improvement. No unattended
  long study was launched. Larger hosted comparisons still require the declared
  billing/resource configuration. Credentials were supplied only in process
  memory, never written to source, protocols, manifests, or reports.
  The subsequent response-contract hardening and successful bounded live smoke
  are recorded separately above; no long study has been substituted for validation.

The long-run gate binds validation to the same protocol and subject ids as its
supporting analysis. Mechanism matching preserves task steps while permitting
different wait durations for declared real-time scheduling checks; elapsed pacing
is still recorded in the full protocol hash and is not evidence of development.

### Bounded validation closeout

This is the earlier same-day validation pass. Its retained-choice conclusion is
superseded by the completed confirmation below; its original artifacts and
failures are preserved.

The final fix/test/recheck pass on 19 October 2026 ends with **2,566 non-UAT tests
passed and 11 expected failures**, with the same two deprecation warnings. The
70 additional cases cover unmetered billing, persistent public task contracts,
read-only assessment, semantic evidence presentation and focused recall,
restart recovery, and implementation-bound confirmation gates. Scoped Ruff lint,
whitespace checks and all seven protocol validations pass. Formatting checks pass
for the new standalone modules and regression tests; eight larger touched modules
retain formatting differences rather than receiving an unrelated bulk reformat.

The [live evidence index](../../reports/research-comparison/20261019T110204Z-final-bounded-validation/live-evidence-index.json)
accounts for **651 actual backend requests, all HTTP 200**, across the stopped
pilot, completed pilots, readouts and backend controls in this pass. This verifies
transport availability, not correctness of every model response. Invalid tool
arguments remain rejected, recorded model proposals, not successful actions.
The service is explicitly declared unmetered; inference, tool, token and time
limits remain active. No credential value was written into these artifacts.

The fixed final campaign used **302 of its maximum 396 requests**. Its fresh
[memory pilot](../../reports/research-comparison/20261019T110204Z-final-bounded-validation/memory/summary.json)
completed 32 activations. The separate
[read-only diagnostic](../../reports/research-comparison/20261019T110204Z-final-bounded-validation/memory_readout/summary.json)
left both source databases unchanged. The procedure subject answered 8/8 items
correctly with retained evidence, versus 2/8 without outcome evidence; the episode
subject answered 1/8 with retained evidence. These are repeated readouts of two
subjects, not independent treatment-effect samples. Neither subject voluntarily
used focused recall, and the procedure subject supplied no voluntary held-out
predictions. Instructed access to a stored rule is not spontaneous use of it.

The [history pilot](../../reports/research-comparison/20261019T110204Z-final-bounded-validation/history/summary.json)
completed 64 activations but exposed a restart/replay defect: recovery rewrote
already-accounted goal records with current wall time, changing retrieval order.
Recovery now leaves accounted actions unchanged while still reconciling missing
or uncertain outcomes once. The original failing report is preserved, not relabeled
as an originally passing study. Its failing recording now reproduces correctly;
the [final offline revalidation](../../reports/research-comparison/20261019T110204Z-final-bounded-validation/final-replay-validation/summary.json)
passes **all 182 recorded frames from 12 runs, with zero mismatches and zero
external tools executed**. This is replay of recorded model outcomes, not a fresh
live replication of the repaired implementation.

Both final pilots still report `insufficient_data` for their primary behavioral
comparison. Lack of eligible voluntary probes is retained as missing evidence;
activity was not forced to obtain a positive score. No long study is running.

Confirmation now additionally binds its supporting evidence to effective birth
configuration, model configuration, feature flags, behavioral code fingerprints
and frozen response/memory contract versions. A changed implementation, a mixed
implementation arm, or a recorded behavioral implementation change cannot silently
reuse that evidence. Subject-specific storage paths and documentation/test-only
changes do not alter this signature. A remote model alias still cannot pin the
provider's underlying weights.

There is no remaining reproduced engineering defect from this bounded pass awaiting
a fix. That is the stopping criterion, not humanlike independence, natural emotion,
or a statistically established behavioral improvement. Those claims remain unproven;
passing tests and waiting longer cannot substitute for the missing evidence.

## Retained-choice closeout

The [final v4 validation record](behavioral_validation_v4.md#completed-confirmation-and-closeout)
closes the defined operational acceptance checks with **28 held-out subjects,
14 complete pairs and no exclusions or failed cycles**. The registered
history-dependent first-work choice effect is **0.7143**, with a 95% interval of
**[0.2857, 1.1429]**, above the prespecified 0.2 minimum. Raw-ledger recomputation
matches the result. Self-selected rest and inactivity remain included.

All **1,344 captured transitions** replay without mismatches or external tool
execution. The evidence audit supports **18 distinct completed goal revisions
across 14 subjects**. Unverified or contradicted claims receive no completion
credit; 203 rejected tool attempts remain visible alongside 433 successful ones.
All 1,107 backend requests return HTTP 200. Appraisal's retrieval connection and
its expression-only and label-free controls are checked on 672 recorded states.

The final suite passes **2,687 tests with 11 pre-existing expected failures**;
all nine shipped protocols validate. The bounded transport/contact repairs and
read-only dead-process detection are covered by regression tests. The earlier
externally interrupted confirmation is retained separately, never pooled into
the passing fixed replacement cohort. No study process or extra 24-hour
observation is left running, and production configuration is unchanged.

## Research scope and non-goals

Engineering tests exercise retained rules, counterexamples, hidden probes, delayed
social feedback, matched common probes, causal retrieval allocation, budgets, restart,
replay, and study validation. Scripted fixture competence is not Nūr competence.
The completed confirmation establishes the declared retained-choice result for
this backend and controlled world. It is not cross-provider replication or a
comparison against matched human reference data.

General external-service exactly-once execution, arbitrary semantic file verification,
push-based scheduling, unrestricted self-modification, learned biological emotion,
human-data training, and cross-context human personality fidelity are not claimed.
The Drosophila/connectome side idea was not incorporated. Baseline behavior remains
available; no new mechanism is promoted merely because its engineering tests pass.

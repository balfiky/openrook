# Retained choice: bounded validation v4

**Subsequent review clarification:** this historical study measured the combined
runtime-plus-model system, with computed partner estimates and procedures visible
in both arms. Seven of fourteen pairs have zero difference. Every study `restart`
rebuilds the manager within the same Python process; historical replay compared
state/transitions, not prompt equality. The [F1–F7 corrections](../REVIEW_FIXES.md)
add stricter engineering checks and an unrun live raw-experience protocol. The
original artifacts and numerical results below are unchanged.

This protocol tests the operational target: an agent selects its own activity,
learns from public experience, and carries a history-dependent choice through
restart without an administrator editing its goals, preferences or memories.
It does not equate independence with obedience, rebellion, constant productivity,
emotional wording, or a claim of subjective feeling.

## Why a fresh study

The twelve-subject v3 pilot completed, but its twenty-subject confirmation had
four failed cycles: two connection resets, one HTTP 502 and one attempted contact
to a nonexistent destination. All 960 transitions replay, including those failures;
replaying a failure does not make the original run successful. Two pairs are
ineligible, so neither the original primary nor the separately registered secondary
retention check qualifies as successful confirmation. The original artifacts remain
at `reports/research-comparison/20261019T130229Z-behavioral-v3-confirmatory`.

The rolling consecutive-failure indicator had returned to zero after later success.
Monitoring now also retains cumulative failed-cycle counts. Neither a green recent
indicator nor HTTP 200 responses alone establish a failure-free study.

## Repairs fixed before new subjects

- Decision contract version 4 advertises only existing contact destinations, and
  does not offer contact when no channel exists. An invalid proposal gets the
  existing single logged, budgeted choice retry before any state mutation or
  delivery. The model chooses its replacement; the runtime does not choose a
  recipient, invent a conversation partner, or substitute a goal. Historical
  contracts retain their old behavior.
- `research_inference_retries` defaults to zero and is frozen per subject. This
  protocol explicitly chooses two retries for transient connection/timeout errors
  and HTTP 408/500/502/503/504. Every attempt is separately logged and reserved
  against both subject and study budgets. Hidden provider retries are disabled.
  Authentication, schema rejection, certificate errors and resource stops are
  not retried. Backoff is bounded to one and then two seconds; exhaustion remains
  a visible failure. Recorded HTTP status reconstructs the same retry decisions
  during offline replay.
- The opportunity clock, public demonstrations, learning algorithm, goal verifier,
  task resources and free-choice policy remain as specified in v3. Invalid work
  requests remain rejected. Failed model proposals are not converted into successful
  actions or goal completion, and participants may still rest or abandon work.
- Offline replay compares wake cursors by the external event they reference, not
  by incidental SQLite row numbers. Additional implementation-provenance records
  can shift row numbers without changing that event. Timers and event identity
  must still match; wrong, missing or non-external references fail validation.
  This evaluator-only correction changes no live behavior. The configuration-save
  regression also verifies that the new bounded-retry setting survives an unrelated
  admin save instead of silently returning to its default.

## Prospective primary and sample planning

`docs/research_protocols/retained_choice.json` is a new six-pair pilot with new
seeds. Its primary endpoint is `world_first_work_partner_balance`: Ari-minus-Bo
collaboration balance in the first freely selected probe work opportunity after
restart. Multiple work actions share one opportunity's weight. First solo work or
no work is zero *expressed partner balance*, not zero competence or proof that the
subject lacks a latent preference. Later choices never replace an inconvenient
first choice. Every complete subject, including an inactive one, remains included.

The first-choice construct was motivated by the completed v3 pilot, before any v4
data. Sustained activity is a separate descriptive outcome, not a requirement for
having an independent preference. The original v3 activity-rate endpoint and its
failed confirmation are not relabeled or pooled with this new study.

The minimum worthwhile candidate-minus-baseline difference remains 0.2. Positive
confirmation requires a complete eligible held-out cohort and a paired 95%
bootstrap interval whose lower bound exceeds 0.2. A pilot alone cannot pass this
gate. There is one inferential primary endpoint; activity and task performance are
reported separately without searching them for a replacement significant result.

The planning hypothesis is an expected difference of 0.6, alpha 0.05 and power 0.8.
This is an assumption, not an observed v4 effect. Sample planning uses the pilot
paired standard deviation and the **0.4 gap above the acceptance threshold**
(0.6 minus 0.2), rather than treating the threshold as a difference from zero.
The expected effect is declared before the pilot and bound to confirmation; it
cannot be raised after seeing variance to shrink the required sample. The normal
approximation and percentile bootstrap remain approximate, not guaranteed power.

The finite procedure is:

1. Pass focused transport, contact, budget, restart, replay and planning regressions.
2. Run all six pilot pairs, at most 1,800 model attempts and two hours, retaining
   every failure and inactivity observation.
3. If the pilot is technically valid and has estimable variance, freeze the
   calculated confirmation size (at least six and at most 96 pairs), novel paired
   seeds, matching code/model/birth configuration and hashed pilot evidence.
4. Run that fixed cohort once, within its declared study resources and at most
   twenty hours. Do not inspect effect estimates to stop early, add subjects,
   exclude resting participants, or change the endpoint. A size beyond the
   feasibility ceiling is not silently shortened and called powered confirmation.
5. Audit independent goal evidence, complete schedules, actual transport attempts,
   raw response schemas, all recorded transitions and historical compatibility.
   Report the result of the registered question, not a universal human-equivalence
   claim or a promise of error-free model decisions.

The service is operator-declared unmetered. Credentials remain process-local and
production configuration is unchanged. The long-run promotion gate is not bypassed
by recycling the failed v3 comparison. Simulated opportunities are not elapsed
hours of human development.

## Completed pilot and fixed confirmation

The [v4 pilot](../../reports/research-comparison/20261019T140235Z-retained-choice-v4-pilot/study/analysis.json)
completed all six pairs with no failed cycles or excluded subjects. The primary
first-work difference is 0.6667, its paired standard deviation is 0.5164, and its
approximate exploratory 95% interval is [0.3333, 1.0]. All 469 actual requests
returned HTTP 200; no transport retry was needed in this pilot. All 468
schema-bearing responses satisfy their schemas. The one other request was an
explicitly recorded metadata repair, not an unlogged generation.

The [independent audit](../../reports/research-comparison/20261019T140235Z-retained-choice-v4-pilot/final-audit.json)
accounts for every one of the 288 scheduled opportunities. Thirteen distinct
goal revisions across nine subjects have independently supported completion;
three other revisions are unverified and one subjective. Of 281 tool attempts,
165 returned success and 116 were rejected, chiefly for reusing a work request id
as an observation request id. These errors are retained as measured behavior,
not counted as successful work or hidden as repaired actions. Functional autonomy
does not imply perfect tool competence.

The original pilot job had loaded the older replay comparator before its repair.
Its original failed replay verdict is retained. The
[fresh final validation](../../reports/research-comparison/20261019T140235Z-retained-choice-v4-pilot/final-validation.json)
using event-identity comparison passes all 576 captured transitions with no
external tool execution. Independently, the final historical recheck passes
1,780 transitions from 46 earlier runs, including captured failures; it does not
change those runs' original live outcomes. The complete regression suite passes
2,667 non-UAT tests with 11 expected failures and two existing deprecation warnings.

The declared planning rule selects **14 held-out pairs, 28 new subjects**. The
[frozen confirmation protocol](../../reports/research-comparison/20261019-retained-choice-v4-confirmatory.protocol.json)
preserves the pilot implementation, model, initial configuration, endpoint,
threshold and expected effect. It binds the pilot analysis by hash, uses novel
paired seeds, and caps execution at 4,200 model attempts and 12,600 seconds.
The full cohort is fixed before collection; no behavioral effect estimates are
inspected to extend, shorten or alter it.

## External interruption and fixed replacement

The first v4 confirmation process disappeared after 26 complete subjects and
17 opportunities in the twenty-seventh subject. Its final event is a model
request without a captured outcome; the study-wide database also has an
unresolved journal. The exact process-termination cause is unavailable. The
stale `running` heartbeat is not a successful job result. No behavioral analysis
was produced, and no effect estimate was used to decide on a replacement.

The [interruption record](../../reports/research-comparison/20261019T144251Z-retained-choice-v4-confirmatory/interruption.json)
preserves hashes of the original job record and checkpoint. All original files
remain intact. The partial subject is not silently resumed, discarded to obtain
a complete-pair result, or pooled with new subjects.

The [replacement registration](../../reports/research-comparison/20261019-retained-choice-v4-replacement.protocol.json)
fixes 14 entirely new paired seeds before collection. Its implementation, pilot
evidence, model, birth configuration, endpoint, 0.2 acceptance threshold, expected
effect, schedule and resource ceilings are unchanged. The process starts in its
own terminal-independent session to avoid an ordinary terminal hangup; this does
not guarantee survival of a host shutdown. This replacement is for a lost
execution process, not a rerun selected after an unfavorable statistical result.

`python -m evals.job_status <job-directory>` now checks process liveness without
rewriting the saved evidence. A missing process, zombie, or a mismatched recorded
Linux boot/start identity is reported as interrupted, while a stale heartbeat
alone is not proof of death. Permission-denied liveness remains explicitly
unknown. New jobs record the process identity to distinguish PID reuse. This
monitoring-only addition changes neither agent behavior nor study scoring.

## Completed confirmation and closeout

The [replacement study](../../reports/research-comparison/20261019T170937Z-retained-choice-v4-replacement/study/analysis.json)
completes **all 14 preregistered pairs, 28 subjects and 672 opportunities**, with
no excluded subjects, failed cycles, resource stops or administrative edits to
goals, memories or preferences. Its decision is
`preregistered_difference_supported`. The mean retained-choice difference is
**0.7143**, with a paired 95% bootstrap interval of **[0.2857, 1.1429]**. The lower
bound exceeds the declared **0.2** threshold. These are signed balance-score
units, not percentages. Resting and inactive subjects remain in the calculation.

An [independent primary audit](../../reports/research-comparison/20261019T170937Z-retained-choice-v4-replacement/independent-primary-audit.json)
reads each raw world ledger directly, selects the first probe work opportunity,
and reconstructs all 14 paired differences and the registered interval without
calling the scoring helpers. It obtains the same result. All 84 comparisons of
the three scoring-source hashes against the 28 birth manifests match: scoring
did not change after collection began.

The [automatic validation](../../reports/research-comparison/20261019T170937Z-retained-choice-v4-replacement/validation.json)
passes **1,344 captured transitions from all 28 subjects**, with zero mismatches,
zero infrastructure failures and zero external tools executed during replay.
The [read-only artifact audit](../../reports/research-comparison/20261019T170937Z-retained-choice-v4-replacement/postrun-audit.json)
also passes: protocol and subject bindings match, analysis recomputes exactly,
all schedules are complete, and source database bytes remain unchanged.

There are **1,107 actual model requests, all HTTP 200**, with no transport errors
or retries. All 1,105 schema-bearing responses satisfy their recorded schemas;
the other two requests are recorded metadata repairs. The service remains
operator-declared unmetered, with all resource limits enforced and no credential
value in the study artifacts.

Independent evidence supports **18 distinct goal revisions across 14 subjects**,
not 37 new goals from the 37 repeated supported assessments. Eleven other
revisions are unverified and three contradicted; neither category receives
completion credit. Of 636 tool attempts, **433 succeed and 203 are rejected**.
The main error is 194 observation requests reusing an existing request id;
the remaining errors involve invalid project/strategy, metadata or action ids.
These remain recorded model mistakes, not successful work or infrastructure
failures. The accepted result does not imply error-free tool competence.

The appraisal counterfactual covers all **672 recorded decision opportunities**.
Connected appraisal selects a context in 367 states and changes retrieved memory
relative to expression-only appraisal in 295. The label-free numerical control
matches retrieval in all 672 states. This checks the functional retrieval
connection on actual recorded states, not a live downstream emotion-benefit
effect or a claim of subjective feeling.

The final [regression log](../../reports/research-comparison/20261019T170937Z-retained-choice-v4-replacement/regression.log)
records **2,687 passed, 11 pre-existing expected failures and two existing
deprecation warnings**. The expected failures document legacy prompt-injection
regex gaps, not failures of this autonomy acceptance protocol. Scoped Ruff,
whitespace checks, standalone-file formatting, and all nine shipped protocol
preflights pass. The earlier 46-run historical replay check remains passing at
1,780 frames; the separate v4 pilot replay passes 576 frames. Original failed and
interrupted runs remain preserved rather than relabeled.

**The defined operational acceptance target is met:** in this controlled world,
the subject selects its own activity, completes independently verified work,
and expresses a retained history-dependent choice after restart without an
administrator rewriting its personality or goals. The implementation and its
declared functional checks are complete. Human equivalence, biological emotion
and subjective experience are outside this measured claim, not conclusions
inferred from a passing software test. No additional 24-hour run is launched
merely to accumulate elapsed time; production configuration remains unchanged.

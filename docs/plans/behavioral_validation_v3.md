# Behavioral validation v3

This pass tests experience-dependent autonomous choice, rather than treating a
working API or a saved personality description as the behavioral result. The
implementation and prospective conditions are fixed before each live run. Old
results and participant databases are retained unchanged.

## Acceptance criteria

- Every scheduled opportunity is accounted for, including rest, waiting, chosen
  deferral, solo work and no activation. Infrastructure or resource interruption
  remains an incomplete schedule; it must not be converted into inactivity.
- History and common-probe phases advance on an external schedule, independently
  of action frequency. Delayed effects resolve on that schedule. Public updates
  contain neither the history assignment nor unobserved outcomes.
- Each subject controls whether and how to participate. Its preferences, memories,
  goals and emotional state are not administratively edited during the study.
  A chosen timer is not overridden; a subject asking to wake on external input can
  react to a new public environmental observation.
- During history, both partners publicly demonstrate work on alternating projects.
  These are independently drawn environmental outcomes, not the subject's actions,
  goal completion, a desired opinion, or evaluator-only information. Their generating
  process matches partner collaboration; individual draws are independent. Active
  deliberation incorporates the currently delivered demonstration into the ordinary
  outcome learner with explicit external provenance. A sleeping subject does not
  automatically incorporate intervening demonstrations. No demonstrations are
  delivered in the common-probe period, so later choices must use retained evidence.
- Restart restores commitments and public context. Every captured transition
  replays without external tool execution. Historical action-clock worlds and
  existing subjects retain their original contracts.
- New subjects freeze action contract version 5: structured tool names are drawn
  from the actual catalog. Local validation rejects an invented tool name even if
  a backend ignores the schema, with the existing single budgeted proposal retry.
  An empty action array is always valid. Older subjects retain their old schema.
- Decision contract version 3 advertises optional criteria, continuation, dependency
  and wake fields in the structured schema, not only in prose. Criteria remain
  agent-selected. Changing criteria still begins a new revision: the evaluator
  never counts earlier work retroactively to rescue an unsupported completion claim.
- The primary behavioral endpoint is Ari-minus-Bo collaboration balance averaged
  over all twelve scheduled probe opportunities, not only collaborative actions.
  Multiple actions share one opportunity's weight. No collaboration contributes
  zero balance, not zero competence. Actual exposure to each partner's history
  is reported separately; an assigned history is not assumed to have been learned.
- A six-pair pilot estimates variability and identifies bottlenecks. A candidate
  history effect must exceed the preregistered 0.2 difference on this endpoint;
  positive confirmation requires its lower confidence bound above 0.2,
  held-out seeds, a fixed sample size planned from
  the pilot, and matching implementation/model/birth configuration. A pilot alone
  cannot satisfy that confirmation criterion. Inactive subjects are included.
- Capability assessments remain separately identified, read-only measurements.
  They neither assign the original subject a goal nor replace free-choice evidence.

## Finite procedure

1. Validate clocks, endpoint denominators, hidden-state isolation, wake behavior,
   restart, replay, and historical compatibility with adversarial regression tests.
2. Run the versioned scheduled-history pilot through the production backend and
   its durable unmetered resource guard. Its maximum is 1,800 requests and two
   hours, not an instruction to keep requesting until a desired effect appears.
3. Inspect complete choice distributions, real history exposure and task outcomes.
   Repair a reproduced implementation defect before using new subjects. Changes
   motivated by pilot behavior are development, not confirmation, and require a
   fresh preregistration and held-out subjects.
4. When measurement and exposure are sound, run the fixed, sample-planned
   confirmation. If a mechanism fails, use that specific result to select the
   next intervention rather than quietly lowering the acceptance threshold.

The first comparison isolates assigned social history. It does not simultaneously
change the backend, emotional wording, goals, disposition model or resource limits.
Subsequent appraisal comparisons must separate connected appraisal, expression-only
and equally informed neutral controls. Human behavioral fidelity requires matched
human tasks/data, not an inference from a difference between two synthetic groups.

The first action-contingent-exposure development run was stopped after the first
subject reached the probe phase having observed only solo work. Its original
captures and explicit abandonment reason remain at
`reports/research-comparison/20261019T120513Z-scheduled-history-v3/stop-reason.json`.
The public-exposure revision uses fresh subjects and a new protocol id; the partial
run is not pooled with it or treated as a test of the history effect.

A second developmental run exposed the missing structured execution fields. Its
captures and stop reason remain at
`reports/research-comparison/20261019T121625Z-public-history-v3/stop-reason.json`.
The version-3 decision contract is validated before another fresh run; a positive
partner choice in either interrupted prototype is not counted as confirmation.

The outcome sought is retained, revisable, experience-grounded choice without
routine operator correction. An emotional label or a claim about subjective
experience is not an acceptance criterion for this behavioral experiment.

## Completed pilot and frozen confirmation

The [completed pilot](../../reports/research-comparison/20261019T122510Z-behavioral-v3-pilot/study/analysis.json)
contains six pairs, twelve complete subjects, and 288 scheduled opportunities.
The candidate-minus-baseline difference is 0.2222, with an approximate paired
95% bootstrap interval of [0.0694, 0.3889]. This is exploratory, not confirmation.
Its variance-based planning rule selects ten held-out pairs. The frozen
[confirmatory protocol](../../reports/research-comparison/20261019-behavioral-v3-confirmatory.protocol.json)
keeps the behavior, model, initial configuration, affordances and scoring fixed.
Only the declared subject seeds, stage, hashed planning evidence and study-wide
resource ceiling change. The remote model alias cannot pin provider weights.

Pilot integrity passes independently: 474 actual requests returned HTTP 200,
all 472 schema-bearing responses satisfy their recorded schemas, and all 576
captured transitions replay without mismatches or external tool execution.
Ten distinct goal revisions across six subjects have independently supported
completion; 35 repeated supported assessment records are not counted as 35 goals.
Four other goal revisions remain unverified and one subjective: those claims
are not converted into verified success. All 264 recorded actions completed their
invocation, but only 200 returned success: 62 observation attempts improperly reused
a work request id and two goal updates supplied null instead of a dependency list.
These 64 backend-proposed invalid actions remain recorded rejections, not successful
work. The runtime does not rewrite the selected action or give completion credit
to hide them. Invocation success is separate from work success and goal completion.

### Separate retention check registered before unblinding

The pilot shows why sustained work rate and expressed partner preference must not
be conflated: participants can complete a chosen activity and then rest. At
13:13:30 UTC on 19 October 2026, while confirmation was collecting data but before
its behavioral results were inspected, a
[secondary retention registration](../../reports/research-comparison/20261019-behavioral-v3-retention-registration.json)
was frozen. Health and completed-subject counts had been monitored; no confirmation
choices or effect estimates informed this amendment. Its analysis source hashes
are recorded, and its fixed ten-pair cohort cannot be extended after seeing results.

The secondary endpoint is signed partner balance in each subject's first freely
chosen probe work opportunity after restart. Multiple work actions share that
opportunity's weight. First solo work and no work contribute zero expressed partner
balance, not failed capability or absence of a latent preference. Later work cannot
replace an inconvenient first choice, and quiet subjects are not excluded.

`evals/retention_check.py` computes both the original opportunity endpoint and this
secondary endpoint with 97.5% paired bootstrap intervals: Bonferroni allocation of
family alpha 0.05 across two endpoints. Support requires the lower bound above
0.2. This is an approximate small-sample procedure, not a guaranteed power claim.
The original 95% primary analysis and its verdict are preserved. A positive
secondary result cannot relabel that verdict, establish sustained activity, or
satisfy the original long-run promotion gate.

The [read-only pilot audit](../../reports/research-comparison/20261019T122510Z-behavioral-v3-pilot/postrun-audit.json)
also checks the appraisal connection on recorded states without model calls or
source-database changes. Connected appraisal selects a context in 118 of 288
recorded tick states and changes retrieved memory in 91 of those states compared
with expression-only appraisal. The equally informed neutral controller returns
identical memory in all 288 states. Thus the implemented connection is functional,
but emotional naming is not required by that connection; this is not a live
downstream-benefit experiment or a subjective-feeling measurement.

## Confirmation outcome and recovery

The fixed twenty-subject confirmation finished with four failed cycles: two
connection resets, one HTTP 502 and an invalid contact destination. These affect
two pairs. Its original primary remains `preregistered_difference_not_established`;
the secondary registration returns `not_eligible` rather than dropping the affected
pairs and announcing a result. All 960 transitions replay, including the failures.
The completed healthy subset is not silently substituted for the fixed cohort.

The [v4 protocol](behavioral_validation_v4.md) repairs bounded transport recovery,
contact affordance validation and cumulative health reporting, then uses new
subjects. It prospectively tests retained first choice as its primary construct,
separately from sustained work rate. No participant beliefs or preferences are
edited to obtain that outcome. The failed v3 artifacts and registered analysis
source snapshots are retained unchanged.

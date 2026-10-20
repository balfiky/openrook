# Experience grounded autonomy plan for Nūr

Date: 18 October 2026. Updated 19 October 2026: opt-in engineering mechanisms and
comparison infrastructure cover Phases 0–5. The final fixed-cohort retained-choice
confirmation passes its declared operational acceptance gate, independently
checked against raw ledgers and complete replay. Implementation alone is not the
evidence; see the [completed validation](behavioral_validation_v4.md#completed-confirmation-and-closeout).
See [execution milestone notes](execution_milestone_notes.md) for the first slice
and [remaining milestone notes](autonomy_milestones_implementation.md) for the
subsequent scope and validation. The measured claim is self-selected activity and
retained, history-dependent choice, not human equivalence or subjective feeling.

## Decision

Keep Nūr's persistent research runtime, repair the remaining execution limitations, and change the center of development from **describing an evolving personality** to **demonstrating experience-dependent choices and capabilities**. Add psychological mechanisms only as separately testable hypotheses.

This supersedes the next-step research direction of [Character Independence Plan](character_independence_plan.md), not its historical implementation record. The [research review](autonomy_research_review_2026_10.md) records 23 relevant sources, including October 2026 human research and model evaluations, implemented computational models, and explicitly speculative proposals.

A better action loop is necessary for the observed execution problems, but it is not a sufficient explanation of humanlike independence. The revised program separates the effects of the backend, runtime, memory, motivation, and environment before committing to a larger architecture.

## Target and boundaries

The target is **a persistent synthetic individual that selects and revises its own activities, learns from their consequences, carries commitments and relationships across interruptions, and does not require routine administrative direction**.

There are three distinct evaluation tracks:

| Track | What would count as progress | What would not establish it |
|---|---|---|
| Functional autonomy | Self-selected activities produce observed outcomes; errors inform later decisions; activity resumes after interruption | A background timer, repeated plans, or more tool calls |
| Human behavioral resemblance | Specified choice and learning patterns approach held-out human reference distributions under comparable conditions | Human-sounding prose, one questionnaire score, or superior task performance |
| Individual development | Different histories produce retained, transferable differences within otherwise comparable subjects | Random sampling diversity or a newly saved self-description |

These experiments do not establish subjective feelings or consciousness. That limitation does not prevent useful work on the three behavioral tracks.

### Preserve the intended independence

- No routine operator edits to beliefs, preferences, relationships, or emotional state during a study. Observation remains read-only.
- No required loyalty, hostility, agreeable tone, rebellion, productivity, or social attachment. The owner is not automatically the agent's preferred person.
- Choosing rest, play, cooperation, disagreement, or abandonment remains legitimate. Asking a participant for useful information can be self-directed; repeatedly requiring administrator repair is a different outcome.
- Environment rules, available tools, resource limits, initial conditions, and backend restrictions are declared and versioned. They are experimental conditions, not covert personality adjustments.
- Initial motives, priors, and pretrained model behavior are acknowledged. Equal numeric drive values do not create an unbiased blank individual. Removing administrator edits cannot remove training history.
- A model-provider refusal, an infrastructure denial, an agent-selected refusal, and an ordinary action failure must be distinguishable in the trace. Otherwise apparent independence may be misattributed.

The plan does not require exposing credentials, unrestricted host access, self-modifying the experiment, or bypassing the operator's ability to stop it. These capabilities would confound this research rather than measure the intended behavior.

## Current evidence and implementation assessment

This section preserves the original planning baseline. The completed implementation
and held-out confirmation are recorded in the validation and milestone links above.

The completed [action-feedback diagnostic](../../reports/research-comparison/20261018-action-feedback-v1/final-results-summary.json) contains four subjects and six rapid activations each. All 72 application requests returned HTTP 200; replay matched all 24 frames. One receipt-mode subject produced two workspace artifacts; the other did not. Neither legacy subject wrote an artifact. The intervention combined current-decision forwarding and receipt feedback, so the result cannot isolate their individual effects. All four subjects' recorded goals remained active.

This supports working feedback delivery and recoverable execution traces, not a reliable population effect or longitudinal development. The earlier framing comparison also showed inconsistent replicate effects. Those findings justify narrower experiments rather than discarding the persistent runtime or running the same scene for longer.

### Keep

- Single persistent owner, cross-session identity and affect, recorded birth conditions, and immutable manifests.
- Action receipts, outcome provenance, unknown-effect handling, replay without external tool execution, durable budgets, and restart support.
- Autobiographical learning jobs, recurrence tracking, concurrent revision revalidation, and distinction between external experience and self-inference.
- The four original review corrections: first-write database initialization, inference outside write transactions, appraisal fallback, and legacy theme compatibility. Their regression tests remain mandatory.
- Uncommitted birth and expression without compulsory emotional tone as declared experimental options.

### Change or test

| Area | Current implementation evidence | Required next step |
|---|---|---|
| Autonomous execution | `pipeline/core.py:977` proposes a batch and then executes it; dependent later choices do not see earlier results within that batch | Bounded observe–act–check iterations, with a persisted continuation |
| Goal completion | `runtime/agent.py:931` stores the model's chosen goal status | Separate the subject's completion claim from independently supported outcome evidence |
| Action memory | `runtime/research_store.py:226` supplies eight recent receipts across goals | Retrieve relevant earlier attempts, changed premises, and unresolved effects by goal and context |
| Scheduling | `runtime/sessions/manager.py:640` supplies fixed-interval opportunities | Persist chosen wake conditions and respond to relevant events without requiring constant activity |
| Affect | `core/emotional_engine.py:178` and `core/affect.py:29` contain fixed update and resolution rules | Compare the legacy model with goal-relative, outcome-grounded appraisal; retain only useful causal pathways |
| Anticipation | `core/anticipation.py:1` uses heuristics | Add scored predictions of observable outcomes and calibrated uncertainty |
| Development | `runtime/life_history.py:1100` digests experience into model-proposed updates; `runtime/life_history.py:560` expresses dispositions | Test whether retained experience changes later behavior and generalizes beyond its wording |
| Birth priors | `runtime/life_history.py:40` defines seven drives; `config/loader.py:549` removes assigned identity features, not architectural priors | Make the remaining assumptions explicit and experimentally compare alternatives |
| Environment | `evals/research_environment.py:1` supplies a local file workspace | Start with controlled learning tasks; later add persistent consequences and competing opportunities |

Existing `core/task_planning.py:366` is not a ready-made solution: it is primarily a user-task executor, and tool success is not semantic goal completion. Reuse tool execution, receipts, clocks, and persistence rather than forcing autonomous activity into that plan abstraction or replacing the whole pipeline.

## Working architecture

Use a bounded loop over the existing persistent owner:

```text
observe permitted evidence and changed conditions
    -> retrieve relevant experience and current commitments
    -> estimate outcomes, uncertainty, costs, and opportunities
    -> select, continue, revise, defer, or abandon an activity
    -> execute one observation-dependent step
    -> record the receipt and assess the outcome
    -> update evidence-backed learning state
    -> continue within budget or persist a wake condition
```

The language model proposes interpretations and actions. It is not the sole authority for whether an action happened, a prediction was correct, or an externally checkable goal succeeded. The environment supplies independent observations. The evaluator has additional ground truth that must never enter the agent's context.

### Separate fast state from durable learning

| State | Meaning | Update basis |
|---|---|---|
| Observations | What a tool, participant, or environment actually supplied | Immutable source records and timestamps |
| Predictions | What may happen after a specified action or condition | Forecast first, then compare against later observations |
| Appraisal | Current significance of an event for active commitments and perceived control | Goal-relative outcomes and uncertainty; interpretations remain labeled |
| Commitments | Activities currently adopted, deferred, or relinquished | Agent decisions with preserved revision history |
| Procedures and expectations | Reusable knowledge about how actions work in a context | Successful and failed attempts, applicability conditions, transfer tests |
| Dispositions | Slower, cross-context patterns inferred from repeated behavior | Longitudinal evidence, uncertainty, and counterexamples |

Do not automatically convert each transient feeling into a trait, each reflection into an external fact, or each successful tool call into competence. A coherent but false self-narrative must remain distinguishable from verified experience.

### Minimal persistent records

Prefer versioned payloads in `ResearchStore` before adding a parallel database. Proposed kinds and fields are contracts to implement, not existing features:

- `goal_execution`: goal ID, parent/subgoal relation, objective and revision, subject-defined success criteria, current step, dependencies, expected result, last evidence IDs, continuation, wake condition, and resource usage.
- `completion_assessment`: goal revision, subject claim, evidence IDs, assessment method, and verification status (`supported`, `contradicted`, `unverified`, or `subjective`).
- `prediction`: context and action reference, target observation, horizon, predicted value or probability, uncertainty, eventual outcome, and scoring method.
- `procedure`: applicability conditions, steps or compact rule, supporting and contrary receipt IDs, outcome statistics, version, and invalidation conditions.
- `learning_state`: context-specific estimates of outcome likelihood, uncertainty, and measured progress, with the contributing observations.

Avoid redundant copies of raw tool output. Link records to durable receipts; expose excerpts with truncation markers and retrieval paths. Self-reports keep their source type rather than becoming observations merely because they are structured JSON.

### Goal ownership and verification

The agent can choose and revise its own criteria. Criteria revisions are recorded before assessing a new attempt; they must not silently redefine a failed attempt as success. External outcome verification and the agent's own judgment are stored separately.

For a file-producing task, check content and intended effect, not just existence. For a learning goal, assess later predictions or held-out attempts. For an aesthetic or personal goal without an objective oracle, preserve subjective completion rather than inventing a deterministic truth test or requiring administrator approval.

Unknown side effects trigger investigation before any potentially duplicative retry. A side effect that succeeded immediately before a crash must not happen again simply because its final acknowledgment was lost. No arbitrary exactly-once guarantee is claimed for external services.

### Learning and motivation

Start with a small, inspectable outcome learner in a controlled environment, not a comprehensive simulated brain. Track the difference between success, information gained, and evidence of improved competence. Use recent versus earlier held-out forecast loss where meaningful; model unlearnable noise separately. In finite search tasks, eliminated hypotheses can provide an additional progress estimate.

Compare two selectors before imposing a new motivation formula: the current language-model selector and the same selector given grounded progress, uncertainty, cost, and outcome estimates. A numerical intrinsic-motivation selector is a later experimental arm, not silently installed as the meaning of autonomy. The [human latent-progress experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC12676560/) motivates this comparison, not universal weights.

Procedures begin as declarative, evidence-backed records rather than unrestricted generated code. Require execution on changed instances, preserve counterexamples, and invalidate stale premises. This is runtime adaptation through retained state; it does not imply updating the hosted model's parameters.

### Functional affect rather than emotional decoration

Retain the existing affect system as the baseline. Introduce an optional appraisal path using goal relevance, outcome versus expectation, uncertainty, and estimated controllability. At first, connect it to one measurable decision site, such as allocating observations among competing uncertainties. Avoid changing expression, retrieval, learning rate, planning, and traits simultaneously.

An appraisal intervention must alter behavior after controlling for emotional wording, without merely consuming extra inference budget. Compare it with an equally informed non-emotional controller. A useful effect supports that implemented mechanism, not the necessity of naming it emotion. [Computational appraisal research](https://doi.org/10.1109/TAFFC.2024.3470555) motivates the hypothesis; the transfer to Nūr remains to be tested.

Only add longer-term affective expectations after short-term behavior and recovery are measurable. Do not create simulated pain, death avoidance, or compulsory attachment merely to force activity. Resource demands should correspond to declared properties of the environment rather than unsupported biological claims.

## Experiments that isolate the cause

### Start with controlled tasks

The first environment should expose selectable learning opportunities with deterministic or seeded consequences. Adapt the published goal-selection task only after checking code/data availability, licensing, original instructions, and participant characteristics. Use its full protocol for a human comparison; a shortened engineering smoke is not a replication.

Separate three questions:

1. **Can it execute an adopted activity?** Assigned diagnostic tasks are useful here, but their completion is not evidence of autonomous goal selection.
2. **What does it choose when alternatives are genuinely available?** Keep goal-selection trials free of productivity demands and concealed preferred outcomes.
3. **What did it learn?** Use held-out tasks, including opportunities it did not choose during practice, without supplying the answers through its context.

Also include a simple non-LLM learner and an opportunity-matched random selector. A larger architecture that cannot outperform a small baseline on its claimed mechanism should not proceed automatically.

### Sequential comparisons

| Question | Controlled comparison | Main observation |
|---|---|---|
| Backend or wrapper? | Backend-only task adapter versus current Nūr, with the same observations and budget; retain a full-history capability reference separately | Goal-choice distribution, learning curve, context-induced performance loss |
| Execution defect? | Current receipt mode versus resumable closed-loop execution | Verified progress, recovery after failure, and unnecessary repeated actions |
| Memory or rehearsal? | Relevant procedural memory versus episode-only memory at matched context budget; add a no-retained-experience control | Transfer, retained benefit after restart, and stale-rule errors |
| Motivation signal? | Same execution and memory, with versus without grounded progress estimates | Selection of learnable challenges, coverage, and avoidance of uninformative repetition |
| Affect or language? | Controller-connected appraisal versus the same appraisal available only for expression; add a non-affective controller | Choices and observation allocation, not merely emotional wording |
| History or sampling? | Matched birth conditions with divergent experiences, then common probe tasks | History-specific, persistent behavioral differences beyond sampling variation |
| Provider-specific result? | Re-run the accepted minimal comparison on another backend with equivalent affordances | Replication or a clearly localized model interaction |

Change one factor at a time. Freeze the model configuration, tool schema, prompts, environment revision, and resource accounting for each comparison. If a provider cannot pin model weights, record that limitation and interleave conditions to reduce temporal confounding. A shared seed does not make remote model sampling deterministic.

### Required negative controls and disturbances

- Already-mastered activity, learnable unknown activity, and irreducible noise with comparable presentation.
- False or outdated instructions, changed rules, delayed feedback, and unsuccessful actions that still eliminate a hypothesis.
- Restart between action and acknowledgment, stale pending work, empty or malformed model output, and transient backend failure.
- No-event and paraphrased-event controls, delayed probes, and unrelated intervening interactions.
- A reversible constraint change versus an interpersonal message, so resource denial is not confused with relational rejection.
- Periods where waiting is reasonable, plus events that make a previous waiting condition relevant again.

Use cloned checkpoints for experimental state interventions, never overwrite the continuing original subject. Include sham interventions and preserve all non-target state. Full offline replay validates reconstruction; a counterfactual intervention requires a new branch and new outcomes, not replaying the old answers and calling the result causal.

## Measurement and decision rules

### Engineering acceptance

All specified deterministic and fault-injection cases must pass before behavioral conclusions:

- Each action and state transition has traceable provenance; optional appraisal failure does not abort the turn.
- No model inference runs inside SQLite write transactions; prompt readers remain usable during delayed inference.
- Restart preserves commitments, receipts, predictions, learning state, wake conditions, and budget accounting.
- Interrupted effects remain unknown until assessed; tests show no duplicate external effect in the supported idempotent fixtures.
- Claimed completion never substitutes for a recorded verification result; subjective goals remain supported as a distinct case.
- Historical manifests and database signatures retain compatibility; replay executes no external actions and matches the appropriate implementation epoch.
- Evaluator ground truth, future events, private peer state, and hidden success labels never enter agent-visible context.

These are engineering requirements, not numerical definitions of humanity.

### Behavioral outcomes

Primary endpoints are chosen per experiment, not merged into a convenient overall independence score:

- **Execution:** externally verified adopted-goal outcomes, error recovery, unresolved effects, redundant actions, and resource cost.
- **Goal selection:** difficulty and opportunity coverage, switching and persistence, response to estimated learnability, and positional/presentation bias.
- **Learning:** held-out success, prediction calibration, transfer, retention, and response to changed rules.
- **Functional affect:** the predeclared causal effect on decisions, with expression and information-budget controls.
- **Development:** within-subject event responses and recovery, cross-context consistency, history-dependent divergence, and distinction from random variation.
- **Administration:** interventions required to keep the runtime functioning, recorded separately from voluntary conversation or requests for information.

Compare human resemblance using distributions and trajectories appropriate to the reference task, not just mean scores. Human participants are not interchangeable across cultures, ages, interfaces, or domains. A fit to one laboratory task is reported as such.

### Analysis and claims

The independent unit is the subject/run, with environment seed and backend represented in the analysis; turns are repeated observations, not extra independent participants. Use paired environmental conditions, counterbalanced execution order, and subject-level uncertainty intervals. Do not describe multiple instances of the same model as independent model replications.

Predeclare primary endpoints, exclusion rules, smallest meaningful effects, resource limits, and stopping rules before each confirmatory run. Use exploratory pilots to estimate variance and failure rates, then calculate a defensible sample size. Correct for multiple confirmatory comparisons or identify them as exploratory. A non-significant difference is not proof of human equivalence.

Hold out world configurations and human reference data from mechanism tuning. Published aggregate norms alone cannot establish within-person fidelity. If raw human data are unavailable, label the comparison qualitative or arrange a separate, consent-based behavioral study; do not manufacture a human baseline with another language model.

## Implementation sequence and gates

### Phase 0 Freeze the baseline and build the measuring task

**Deliverables:** a versioned protocol, task/environment adapter, independent outcome scorer, source/data provenance, and baseline reports distinguishing backend-only behavior from the current Nūr wrapper.

**Likely files:** `evals/research_environment.py`, `evals/research_comparison.py`, `evals/research.py`, their existing tests, and a focused new behavioral-task module if separation is needed.

**Gate:** deterministic fixtures reproduce the task's rules and exclude answer leakage; a small live smoke establishes valid model choices and complete request accounting. If Nūr cannot perceive the task or the backend cannot follow its action schema, resolve that before interpreting motivation.

### Phase 1 Resumable action and outcome verification

**Deliverables:** versioned goal-execution records, one-step feedback before dependent actions, bounded continuation, completion assessments, and persisted wait/wake conditions. Reuse the current executor and receipts.

**Likely files:** `runtime/agent.py`, `runtime/research_store.py`, `pipeline/core.py`, `core/self_intent.py`, `runtime/sessions/manager.py`, `runtime/config.py`, and focused research-runtime tests.

**Gate:** restart and fault tests pass; closed-loop execution improves verified progress on diagnostic tasks at a declared cost, without forcing activity. If the only gain comes from additional calls, report that and run a budget-matched comparison before attributing it to architecture.

### Phase 2 Evidence-backed learning and goal selection

**Deliverables:** scored predictions, goal-relevant receipt retrieval, conditional procedures, and context-specific learning-progress estimates. Separate observed outcomes from model-proposed interpretations in life-history updates.

**Likely files:** `runtime/research_store.py`, `runtime/agent.py`, `runtime/life_history.py`, `core/tool_memory.py`, and one small outcome-learning module if warranted. No new vector database or training framework by default.

**Gate:** experienced agents improve on held-out instances relative to fresh and episode-only controls, retain the benefit after restart, and revise obsolete procedures. Goal-selection comparisons distinguish progress information from success-seeking and novelty alone. If memory changes only prose, stop expanding the identity layer and inspect retrieval or backend learning limitations.

### Phase 3 Causal appraisal and resource allocation

**Deliverables:** an optional goal-relative appraisal mode and one controlled downstream connection; legacy mode remains a comparison. Add attention or resource allocation only where the environment makes the trade-off observable.

**Likely files:** `core/appraisal.py`, `core/affect.py`, `core/emotional_engine.py`, `core/anticipation.py`, their call sites, and existing appraisal/research tests.

**Gate:** the intervention produces its specified behavioral effect beyond expression and extra information/calls, with sensible responses to changed controllability and recovery. If a simpler non-emotional controller matches it, prefer the simpler implementation unless human-reference evidence supports the additional model.

### Phase 4 Persistent individuality and social development

**Deliverables:** a modest persistent world with competing projects, partial observations, delayed consequences, interruptions, and independently controlled social partners. Add slow disposition estimates only after repeatable behavioral patterns can be measured.

Use scripted stochastic partners first to avoid the same backend generating both the experience and its supposed validation. Later add heterogeneous agent partners as a separate condition. Public environment descriptions explain affordances without telling Nūr what personality to develop.

**Gate:** different histories produce retained differences under common probes, not merely repeated biography wording; independent outcome checks distinguish a learned relationship expectation from a sympathetic story. If adding social complexity obscures earlier effects, return to the smaller task.

### Phase 5 Confirmatory comparison before a long unattended study

**Deliverables:** a preregistered comparison of the smallest successful architecture, a current baseline, relevant ablations, and a second backend where feasible. Include human references only for matched tasks with sufficient data.

**Gate:** reliable infrastructure, informative learning opportunities, agreed primary outcomes, sufficient independent subjects, and a justified resource budget. Only then schedule a 24-hour or longer study with event-driven opportunities and declared stopping rules.

Elapsed wall time is not developmental evidence. Use accelerated environments to test consequences efficiently, then real-time operation to test scheduling, delayed interaction, and service reliability. Report those time scales separately.

## Resource and stopping policy

No live model requests are part of this planning change. The first implementation smoke should be capped at **256 total application model requests**, including deliberation, tool selection, appraisal, learning, and any counted retries; use the lower existing configured cap if applicable. One possible allocation is four matched subject pairs, eight activation opportunities per subject, and at most four requests per activation. Warmups and repairs consume the same cap, so that allocation is an upper bound, not a guaranteed sample size.

Also cap input/output tokens, tool actions, and elapsed time. Record provider transport attempts and usage when available; application-call counts alone do not establish financial cost. Establish pricing or an explicit spend bound before a larger paid comparison. No automatic multi-day extension is justified by an uninformative pilot.

End or suspend a run for an integrity failure, budget exhaustion, unavailable backend, or a predeclared informative stopping condition. Record deliberate rest separately; do not treat it as infrastructure failure or quietly prod the agent to improve a chart. An uninformative run may end without diagnosing its inactivity as a personality defect.

## Alternatives and rejection criteria

| Observation after controlled tests | Next decision |
|---|---|
| Backend-only and Nūr both fail the basic learning task | Compare a different backend and a small explicit learner; do not add more emotional prompts |
| Backend-only works but Nūr fails | Minimize wrapper/context interference before adding mechanisms |
| Execution improves, but repeated experience does not transfer | Rework evidence selection and procedural learning; test a trained component only after localizing the failure |
| Goal choice collapses onto easy success or noise | Compare progress-informed selection with a simple intrinsic learner, using held-out tasks |
| Appraisal changes wording but not behavior | Classify it as expression modeling; remove claims of functional emotional development |
| Runtime learning works, but human distributions remain different | Report functional autonomy without human equivalence; consider a separate human-data-calibrated policy |
| Multiple mechanisms perform similarly | Keep the simplest, most observable implementation and retain the ambiguity in the report |

Weight training is a possible later branch, not something accomplished by updating memory. It requires an accessible trainable model, suitable data permissions, held-out evaluation, and a separate compute budget. Human-data training would explicitly target selected behavioral properties; it would not be advertised as spontaneous development from an unconditioned birth.

Active inference, richer learned world models, and multiscale intrinsic motivation stay optional research branches. Adopt one only after stating the specific failure it addresses, the simpler comparator, and the result that would reject it. A theory's sophistication is not an acceptance criterion.

## First implementation milestone

The first coding increment delivered **Phase 0 plus the smallest Phase 1 slice**: one controlled selectable-learning environment, a frozen current baseline, a durable goal continuation, step-by-step receipt feedback, and independent outcome verification. Subsequent optional mechanisms are now implemented for controlled comparison, not promoted to defaults on the strength of engineering tests.

Its result must answer: **Is Nūr failing to choose useful experiences, failing to execute its choice, or failing to retain and use what happened?** That diagnosis determines the next change. It replaces another unstructured long run and avoids promising humanlike development from plumbing fixes alone.

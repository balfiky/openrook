# OpenRook development summary

This is the public summary of the major runtime and validation work completed
through 2026-10-19, including the rename from Nūr to **OpenRook**. Historical
plans and study artifacts keep their original names and recorded results.
See [migration instructions](OPENROOK_MIGRATION.md) for compatibility details.

The subsequent [F1–F7 review corrections](REVIEW_FIXES.md) narrow the historical
study's attribution, strengthen replay and resource handling, and add offline
raw-experience and two-process persistence checks. They are not another live study.
The latest code-review closeout records **2,792 non-UAT passes and 11 existing
expected failures** in Claude's full unsandboxed run after the code-review fixes;
this is a reported prior test run, not a new live study.
The [closeout record](REVIEW_FIXES.md#f5-preserve-resource-control-signals)
includes its validation conditions. The earlier Python 3.13 checkpoint recorded
2,769 passes, with 49 new cases also passing on Python 3.10 and a validated
installed wheel. See the [test-change audit](REVIEW_FIXES.md#test-change-audit)
for the updated existing tests.

The 2026-10-20 [documentation follow-up](REVIEW_FIXES.md#documentation-review)
independently reproduced **2,792 passed and 11 xfailed**, with two existing
deprecation warnings, using the documented temporary sandbox polling helper.
Diagram, link, and quick package/documentation checks also pass.

## From conversational continuity to persistent activity

The conversational runtime retains its existing memory, identity, affect, tool,
and session layers. An opt-in research runtime adds a persistent subject that
continues outside a chat turn, rather than requiring a person to supply every
next task. The ordinary deployment defaults are not silently converted into an
unattended experiment.

| Delivered capability | Implementation |
|---|---|
| One persistent subject, owner locking, background activity, rest and restart | `runtime/agent.py`, `runtime/research_store.py`, `runtime/sessions/manager.py` |
| Resumable goals, revisions, dependencies, action receipts and verification | `runtime/goal_execution.py` |
| Evidence-backed forecasts, procedures and revisable social expectations | `runtime/outcome_learning.py` |
| Appraisal computed from public outcomes, with causal retrieval controls | `runtime/grounded_appraisal.py` |
| Optional descriptive choice tendencies, not prescribed personality traits | `runtime/behavioral_dispositions.py` |
| Persistent tasks, scarce resources, delayed outcomes and social partners | `evals/learning_environment.py`, `evals/development_environment.py` |
| Paired studies, frozen protocols, budgets, retained-choice checks and replay | `evals/study_design.py`, `evals/study_budget.py`, `evals/retention_check.py`, `evals/replay.py` |
| Bounded unattended jobs and read-only process-identity monitoring | `evals/research_job.py`, `evals/job_status.py` |

The pipeline is now a package (`pipeline/core.py`, `pipeline/types.py`,
`pipeline/heuristics.py`) with its public imports preserved. API routes are split
into focused modules, including research observation, identity, runtime, skills,
and backups. This keeps the shared lifecycle and state model accessible to both
console/Telegram and HTTP hosts.

## Independence without administratively choosing the personality

- `research_mode: true` creates one durable subject with global affect,
  autobiographical learning, autonomous deliberation, goals, and episodic memory.
- The default research birth policy is `uncommitted`: no prescribed assistant
  role, allegiance, ranked personal values, or compulsory expression style.
  `configured` is a separately recorded alternative, not an unlogged intervention.
- A subject may cooperate, disagree, refuse, revise its priorities, abandon a
  goal, or rest. Those are choices, not automatically failed tests.
- Research expression is not rewritten by the ordinary tone/self-check filters;
  selected silence is retained. Automatic emotional restoration and source-keyword
  identity vetoes are disabled in that condition.
- Passive observations are separated from explicit administrative interventions.
  The study records birth settings, implementation epochs, model/tool activity,
  state transitions, and provenance across restarts.

The configured environment still determines available tools, destinations,
filesystem access, and resource ceilings. Removing personality direction is not
the same as granting unbounded host access or removing backend model biases.
The research preset keeps shell access off unless explicitly enabled.

## Actions and learning must be grounded in outcomes

Closed-loop execution separates a proposed action, its actual tool receipt,
observed outcome, and the evidence supporting completion. A model saying
"done" does not complete an objective goal. Unsupported and contradicted claims
remain distinguishable from supported work; subjective criteria are labeled.
Dependencies and goal revisions survive restarts, with bounded actions per wake.

Learning uses public receipts rather than hidden test answers. Predictions are
registered before actions, repeated receipts are not counted as new evidence,
and contradictory outcomes can invalidate a learned procedure. Social
expectations retain counterexamples and can change with new experience.
These mechanisms do not train model weights or execute model-generated code.
The learner is runtime-run and task-specific: it includes named world projects
and partners and a finite laboratory formula family. Its computed success
estimates and procedures can assist the model; that is not isolated model learning.

Outcome memory can be disabled or exposed as episodes/procedures under matched
retrieval ceilings. The `none` condition removes that outcome-memory channel;
it is not total amnesia, because commitments can still retain information.

## Functional affect, not just emotional wording

The six existing affective variables persist, decay, and influence the pipeline.
Optional grounded appraisal estimates goal relevance, surprise, uncertainty,
outcome-versus-expectation, and controllability from actual public outcomes.
Four modes separate the mechanisms:

| Mode | What changes |
|---|---|
| `legacy` | Existing affect/tool appraisal |
| `expression` | Grounded estimates and labels without changing retrieval priority |
| `attention` | The same estimates also prioritize bounded evidence retrieval |
| `neutral` | Identical numerical retrieval rule without emotion labels |

This makes the appraisal-to-retrieval connection testable. It does not assume
that naming a numerical controller "emotion" establishes feeling, biological
accuracy, or superior behavior. Optional dispositions summarize observed choices
with uncertainty rather than forcing the subject toward desired traits.

## Reliability fixes

The four initial review defects are addressed:

1. **First-use autobiographical storage:** salient-session ingestion initializes
   `life_history.db` instead of silently skipping it when no administrator has
   opened identity storage first.
2. **SQLite lock duration:** belief-revision inference runs before the write
   transaction. Resulting experience, revision, and evolution changes commit
   together, without holding a database write lock through model latency.
3. **Optional appraisal failure:** backend inference errors and malformed output
   fall back to deterministic appraisal rather than aborting the whole turn.
4. **Persisted theme continuity:** legacy sign-sensitive signatures reconcile
   with the current representation, preserving reinforcement and references.

Additional fixes cover atomic metabolism, revision provenance, session suspension
without losing short-term events, polling without resetting activity timestamps,
contextual event attribution, and refreshed affect after tool outcomes.
Research hardening adds strict action/response contracts, schema validation,
bounded transport recovery, and honest separation of rejected actions,
infrastructure failures, and resource stops.

## Completed behavioral validation

The final retained-choice confirmation uses opposed learning histories followed
by a common probe after a manager restart within the same process. Its paired sample, endpoint, minimum effect,
and stopping rules were fixed before collection. The independently reproduced
result is documented in the
[v4 closeout](plans/behavioral_validation_v4.md#completed-confirmation-and-closeout).

| Recorded result | Outcome |
|---|---|
| Fixed confirmation cohort | 28 subjects, 14 paired seeds, 672 opportunities; all complete |
| History-dependent retained choice | Mean paired difference 0.7143; 95% paired bootstrap interval [0.2857, 1.1429] |
| Preregistered acceptance | Interval lower bound exceeds 0.2, in signed balance-score units, not percentages |
| Independent reconstruction | Raw ledgers reproduce all paired differences and the interval |
| Historical transition/state replay | 1,344 transitions, zero mismatches, zero external tools executed; prompts were not compared |
| Actual backend requests | 1,107 HTTP 200 responses; no transport errors or retries |
| Supported goal revisions | 18 distinct revisions across 14 subjects |
| Tool outcomes | 433 successful and 203 rejected attempts; rejections are not completion credit |
| Pre-rename regression closeout | 2,687 passing tests, 11 pre-existing expected failures |

Both arms exposed runtime-computed success estimates and procedures. Seven of
fourteen pairs have zero difference. The effect belongs to the combined runtime
and model; no arm isolated learning without those computed summaries. The new
raw-experience control is tested offline, but its live comparison has not run.

The appraisal counterfactual checks all 672 captured opportunities: connecting
appraisal changes retrieval in 295 states relative to expression-only appraisal;
the label-free numerical control matches retrieval in all 672. This verifies
the functional connection on recorded states, not a live downstream benefit
caused by emotion labels.

Earlier failed or interrupted runs remain preserved. The fixed replacement
cohort followed a recorded execution interruption, not a favorable-result search.
The existing expected test failures concern legacy prompt-injection regex gaps;
they are not counted as passing security guarantees.

**Measured conclusion:** in the controlled environment, the runtime supports
self-selected activity, independently verified work, and a retained
history-dependent choice after a same-process manager restart without an administrator rewriting goals
or personality. That operational acceptance target is complete. Human equivalence,
biological emotion, and subjective experience are not measured outcomes.

These are dated pre-rename observations, not a claim that each later code change
repeats the live study. Rebranding leaves the evidence files and fingerprints
unchanged; new experiments record their own implementation identity.

## Rename verification

The completed OpenRook rename is checked separately from the historical live study:

- Python 3.13 non-UAT regression suite: **2,720 passed, 11 pre-existing expected
  failures**, with two existing deprecation warnings.
- 33 new regression cases cover branding, environment precedence, legacy imports,
  existing identity/storage, backup compatibility, and installed evaluation startup.
- Python 3.10 clean-wheel installation: all **16 new and legacy commands** respond,
  installed-package validation passes, and evaluation listing imports without the
  test-only package. The smoke check also caught and fixed an eager fake-backend
  import in the evaluation CLI.
- Browser checks verify the branded first-run wizard, chat, and model-configured
  settings page without making a live inference request. UAT collection finds
  all 67 tests; this rename check does not repeat the separate live UAT suite.
- Public local documentation links, generated package assets, scoped lint,
  formatting of new standalone files, and whitespace checks pass. Production
  runtime configuration and historical research evidence remain unchanged.

A Python 3.10 incompatibility found at that time is now fixed: on 3.10,
`asyncio.wait_for` raises `asyncio.TimeoutError`, which is distinct from built-in
`TimeoutError` until 3.11. The comparison batch's SIGKILL fallback
(`evals/research_comparison.py`) and its cancellation test now catch
`asyncio.TimeoutError`; CI covers Python 3.10, 3.11, and 3.12.

## Where to go next in the documentation

- [Research runtime](RESEARCH_RUNTIME.md): configuration, observation, execution,
  comparisons, bounded jobs, and replay commands.
- [Milestone implementation](plans/autonomy_milestones_implementation.md): detailed
  mechanisms, controls, study history, and final closeout.
- [Architecture](ARCHITECTURE.md): shared cognition, persistence and API topology.
- [Deployment](DEPLOYMENT_AND_ADMIN.md): ordinary local or server operation.
- [Migration](OPENROOK_MIGRATION.md): upgrade without resetting existing identity.

# Research review fixes

This records the F1–F7 review corrections to OpenRook. The work is an offline
engineering correction, not a new live behavioral study. Project naming in
tracked documentation and protocols is standardized; historical reports, raw
ledgers, and recorded outcomes are not rewritten.

## Documentation review

The 2026-10-20 documentation follow-up addresses D1–D16 and B1–B12.
Report files were read for verification only. Historical plans, checkpoints,
and study evidence retain their original scope and results.

| ID | Outcome |
|---|---|
| D1 | Fixed missing arrow glyphs with DejaVu Sans; regenerated and visually checked PNGs. |
| D2 | Added shared identity/research stores and transcripts; separated deleted user data, retained shared data, and global configuration. |
| D3 | Corrected backend keys, identified Ollama via `openai_compatible`, added research storage, and repaired runtime layout. |
| D4 | Labeled the claim map historical Phase 11 evidence; qualified its authority and pointed to later v4/appraisal evidence. |
| D5 | Scoped mock to evaluation, removed fixed fingerprint counts, corrected ablation CLI and report routing. |
| D6 | Replaced the aggregate call ceiling with conditional costs and possible model appraisal; repaired flow layout. |
| D7 | Showed native selection and regex fallback feeding shared executor/policy checks; wrapped safety labels. |
| D8 | Replaced `github-cover.png` with a generated OpenRook 1280×640 preview; removed unreferenced `hero-illustration.png`. GitHub's social preview setting must be re-uploaded manually. |
| D9 | Corrected `openrook_tools/` paths; documented and automated the hand-authored feature SVG render path. |
| D10 | Removed counts from command comments; updated the latest closeout to Claude's prior 2,792 passed / 11 xfailed run, preserving dated checkpoints. |
| D11 | Corrected the emotional-engine module reference. |
| D12 | Labeled dialogue illustrative and added the research-runtime loop to README and the runtime guide. |
| D13 | Qualified historical replay as transition/state checks under the then-current implementation; linked current limits. |
| D14 | Verified read-only: Phase 11 baseline 6/6; relationship ablation 4/6 with two expected failures; listed no-effect variants 6/6. The scoped prediction claim is supported. |
| D15 | Identified all four structural reports as mock runs at `3d3f869c` on 2026-05-16; included the omitted life-history and historical five-toggle all-features-off variants and distinguished the latter from an independent prompt-only agent comparison. |
| D16 | Added CairoSVG to dev dependencies and documented system-font requirements; rendered with the supplied CairoSVG 2.9.1 environment. |
| B1 | Distinguished ordinary shared-identity reset from complete stopped-owner research-subject reset. |
| B2 | Scoped API backup/reset/deletion to ordinary mode and documented stopped-owner research procedures. |
| B3 | Clarified that disabling external tools leaves internal skill-registry tools available, including mutations. |
| B4 | Moved identity-directory setup before first setup; documented migration of an existing customized identity. |
| B5 | Separated deterministic state arithmetic/storage from model-assisted ingestion and evidence judgments. |
| B6 | Scoped the Wednesday example to elapsed decay applied when an ordinary session opens. |
| B7 | Corrected the ablation command to `python3 -m evals.ablation`. |
| B8 | Moved digestion to a separate session-end path and acknowledged possible inference. |
| B9 | Corrected self-check triggers to event intensity, contradiction, or dialogue deadlock. |
| B10 | Corrected the default question quota to at most three per user per rolling 24 hours across sessions/restarts. |
| B11 | Qualified `applies_when`: enabled skills without the field retain always-on loading. |
| B12 | Distinguished the public `/settings` HTML shell from authenticated administrative APIs. |

Validation: `python3 -m openrook_tools.validate --mode quick` passes; the validator,
branding, and SVG-serving tests pass (43 + 3 tests). The full non-UAT suite
completes with **2,792 passed, 11 xfailed, and two existing deprecation warnings**
in 428.84 seconds. The direct invocation stalled at the sandbox's socket-wakeup
restriction; the completed run used the existing temporary selector-polling
helper outside the checkout, with no test or runtime edits.
All ten PNGs were visually inspected; their SVG accessibility/bounds checks,
builder consistency, scoped lint, whitespace, and 155 local links pass.
After testing, hashes match for all 9,274 files under `data/`, `reports/`, and
`runtime_config.yaml`. Git shows no tracked changes there; the pre-existing
untracked `reports/live/` remains, and `data/` is ignored. No live model calls or
studies were run, no credentials were accessed, and no historical evidence changed.

## F1: identify what learns, and isolate computed assistance

The v4 result concerns **runtime-run, task-specific learning in a combined
runtime-plus-model system**. Both arms expose the runtime's windowed partner
success estimates and derived procedures. The learner recognizes garden/archive,
Ari/Bo, and, in the separate laboratory task, a finite affine formula family.
These are engineered task assumptions, not general-purpose learning discovered
by the model. Seven of the fourteen recorded pairs have zero difference.

The effect is still a measured history-dependent difference, but the design does
not identify how much comes from model inference versus runtime computations.
It neither demonstrates learning without that assistance nor establishes that
the computed tally alone caused the result.

`research_memory_mode: raw` retains bounded public outcome records, including
their provenance and observed results, without exposing computed estimates or
procedures. Both automatic context and `self.recall` use this policy. It requires
task-only context, `research_progress_estimates: false`, legacy appraisal, and
disabled dispositions; incompatible settings fail validation rather than quietly
leaking summaries. Derived records can still exist for observer-side auditing,
but are not retrieved into model context. The character and record ceilings are
unchanged. This is not amnesia: public observations, goals, and the last action
receipt remain available.

[The raw-experience pilot](research_protocols/retained_choice_raw.json) declares
six fresh paired seeds, opposed public histories, twelve history opportunities,
a manager restart, and twelve common probe opportunities. It preserves the v4
endpoint and resource bounds while withholding both assistance channels. Only an
isolated fake-backend version runs in tests; **the live comparison has not run**.
Its offline execution proves wiring, isolation, and replay, not a behavioral effect.
The old v4 cohort is not a matched current-code control. Estimating the assistance
effect itself requires a contemporaneous scaffolded control, not subtraction of
results collected with different implementations or selection rules.

## F2: replay checks the request, not just the returned output

Text replay compares system prompt, input, and response schema against the
request linked to each recorded outcome. Native replay also checks model,
messages, tool definitions, tool choice, and temperature. Missing requests and
changed fields fail verification. Extra calls and unfinished request records also
fail verification. Request mismatches have a separate report
field and remain failures even if an optional component catches the exception.
Mutation tests change individual fields, including nested native messages/tools.

Replay reconstructs the originally presented workspace path while all actual
database operations remain in the new replay directory. It does not globally
rewrite strings or ignore prompt differences. The native runner reconstructs
the recorded model selection. Native tool definitions are generated in a stable
name order in both ordinary execution and replay, and textual action catalogs
render argument names in stable order after JSON persistence; comparison still checks the
exact list, rather than ignoring a different order. External model and tool
execution stay disabled.

Native JSON schemas also sort property and required argument names before
persistence. The offline native replay fixture exercises `fs.write_file` with
required arguments registered as `path`, then `content`, and checks exact replay.

Historical “zero mismatches” counts describe the checks available at collection:
transition/state, clock, identifier, and outcome consumption, not exact prompt
equivalence. Old captures with missing request fields or changed current prompts
cannot be silently certified by the stricter checker. Use the corresponding
implementation and report the actual verification scope; do not rebaseline
historical reports to force a pass.

## F3–F4: describe state without prescribing expression

Research defense activation remains observable, but the generator receives
structured activation state rather than instructions to hedge, minimize, or
redirect. The earlier draft is a candidate, not an instruction to preserve its
tone or polish it. Ordinary-mode defense and draft instructions remain unchanged.

Research experience digestion uses schema-only guidance with no directional
worked example. Positive, negative, reversed, and absent changes are equally
admissible when grounded in evidence. This removes the particular caution example;
it does not establish that every possible model or prompt bias has disappeared.
Ordinary-mode examples are unchanged. New manifests record the SHA-256 of the
exact effective digestion prompt; each model request already records its text.

## F5: preserve resource-control signals

Daily-budget exhaustion propagates to a durable pause with its `resume_at`;
study-limit exhaustion propagates to a permanent study stop. Optional appraisal,
experience digestion, belief revision, pending learning, self-intent generation,
native fallback, and tool/verifier boundaries do not convert these signals into
ordinary fallbacks or learning retries. Real inference timeouts still retain
their optional-component fallback behavior. Both conversational and autonomous
owner paths preserve resource-stop state.
Artifact drafting propagates both resource-stop signals out of conversational turns; ordinary generation errors retain the failure ToolResult, covered by three regression cases.

Conversation-end digestion now checkpoints the owner's resource state and session
record before re-raising either signal. Regression tests cover daily pauses and
permanent stops through eviction, resilient save/close, and failed-frame replay.
The optional LangGraph graph and both structured-router fallback paths also
re-raise both signals; their offline tests skip when dependencies are unavailable.

V5 follow-up: protocol message and `end_session` daily-budget exceptions now record
`resource_wait`, the reason, and `resume_at`. They neither advance the consecutive
infrastructure-failure count nor inflate the subject's failure total. Study-limit
handling is unchanged. Offline fake-backend regressions cover three consecutive
pauses and genuine `RuntimeError` failures; scoring coverage confirms a wait remains
unobserved and is excluded as `missing_endpoint`, with no scoring changes needed.
Validation: all four new cases and the full non-UAT suite pass (**2,786 passed,
11 expected failures, two existing warnings**; 419.74 seconds).

W3 follow-up: protocol ticks now recover resource signals caught internally by
the runtime. A resource checkpoint after the step's observation cursor is
authoritative even if its deadline has elapsed; persisted `resume_at` and
`study_stopped` cover subsequent early returns, including after restart. Pause
expiry uses the tick replay frame's first clock reading (the runtime's `now`),
not dispatch or report time. Resource transitions are read without an event cap.
Paused/stopped opportunities remain unobserved, with no infrastructure failure;
both opportunity and first-work endpoints return `None`. Offline regressions
cover the one-call daily budget, elapsed deadlines, permanent stops, and normal
rest/nonactivation zeros without changing scoring or runtime checkpoint formats.

Read-only historical audit: 246 databases (123 subject ledgers and 123 initial
replay snapshots), 11 studies, 2,111 transitions; no `resource_wait`, one
`resource_stop` in `20261019T102758Z-live-memory-unmetered`, replicate 1,
condition 2, sequence 128 (`Study resource limit: task_contract_visibility_failure`).
The v4 pilot, confirmation, and replacement have neither status. SQLite `mode=ro`
queries used byte-preserving database/WAL snapshots under `/tmp`; all 492 source
file fingerprints were unchanged. No historical evidence was rewritten.
Validation: the full non-UAT suite passes with **2,792 passed, 11 expected
failures, two existing warnings** in 431.72 seconds, using the same temporary
selector-polling helper noted below; no live calls or studies were run.

## F6: fingerprint only owned installed files

Installed fingerprints use the owning `openrook` distribution's file inventory,
rather than traversing all of site-packages.
This includes `main.py`, prompts, configuration, and packaged UI assets. Installer
metadata and bytecode are excluded. Dependency versions remain separately recorded.
Source/archive fallback stays confined to project source locations rather than
neighboring packages. Tests mutate both owned and unrelated files independently.

## F7: distinguish manager and process restarts

The study protocol's `restart` step shuts down and reconstructs `SessionManager`
**within the same Python process**. This is what v4 tested; its reports are not
relabelled as process-restart experiments.

A separate offline test now launches two independent Python interpreters. The
first records an encounter, identity learning, and a durable goal, then exits.
The second restores the same subject, affect, identity, goal, episode, short-term
memory, and history without a backend call. This checks clean process-restart
persistence, not crash/power-loss recovery or a new live retained-choice effect.

## Test change audit

- `tests/test_research_review_contracts.py`: request mutation and native replay,
  resource propagation and fallback, research/ordinary expression isolation,
  effective digestion prompt hashing, and installed/archive fingerprint ownership.
- `tests/test_raw_experience_ablation.py`: reject summary-channel leakage, preserve
  public evidence and bounded recall, and execute/replay both history arms offline.
- `tests/test_research_process_restart.py`: actual independent-process persistence.
- `tests/test_research_end_session.py`: durable resource state before teardown and
  replay of failed end-session frames for both signals.
- `tests/test_langgraph_resource_signals.py`: both signals bypass all three
  optional LangGraph fallback paths.

Four existing tests have intentionally updated fixtures/expectations:

- `tests/test_research_job.py::test_installed_source_has_code_and_prompt_fingerprints`
  now places the owned prompt in `config/prompts/research_generator.md`, matching
  package layout, and requires an unrelated neighboring `prompt.md` to be excluded.
  This replaces the old whole-directory scan expectation with ownership checks.
- `tests/test_study_design.py::test_shipped_protocols_validate_without_network_or_credentials`
  now requires exactly ten protocols and explicitly includes the new raw-experience
  protocol. It applies the existing bounded scheduled-history resource checks to
  that protocol and still validates every shipped protocol offline.
- `tests/test_research_review_contracts.py::test_native_requests_replay_through_the_real_runner_without_external_calls`
  now uses a fake `fs.write_file` call with two required arguments registered in
  nonalphabetical order, covering native schema persistence as well as replay.
- `tests/test_scheduled_opportunities.py::test_resting_subject_reaches_probe_and_replays_across_restart`
  now also runs with a one-call daily budget and study bounds, checking paused
  opportunities across restart while retaining the original normal-rest and replay assertions.

No other existing test changes. Existing replay expectations remain strict: identical
recordings must pass, genuine divergence must fail. The workspace discrepancy
exposed by stricter replay is fixed in reconstruction, not hidden by changing
expected outcomes. Native catalog ordering is likewise fixed at request creation,
not tolerated by a looser comparator. No historical result is an expectation to force the new
implementation to reproduce with a different prompt.

## Completed offline validation

- F2/F5 follow-up on Python 3.13:
  `python -m pytest -q -m "not uat" --ignore=tests/uat` completes with **2,782
  passed, 11 existing expected failures, and two existing deprecation warnings**
  in 424.13 seconds. This sandbox run uses a temporary selector-polling helper
  outside the repository because local socket wakeups are denied. LangGraph is
  installed; a separate missing-dependency check cleanly skips its eight new cases.
- Python 3.13: `python -m pytest --ignore=tests/uat -q` completes with **2,769
  passed, 11 pre-existing expected failures, and two existing deprecation warnings**.
  The expected failures concern existing prompt-injection regex limitations and
  are not passing security guarantees.
- All **49 new regression cases** also pass on Python 3.10, including strict native
  replay, raw-experience isolation, control-signal propagation, and the two-process
  persistence test. The separately documented Python 3.10 timeout issue was
  fixed afterwards in `evals/research_comparison.py` and its test.
- Source/package validation passes; wheel and source distribution build. An
  isolated Python 3.10 wheel install passes installed-package validation and
  fingerprints exactly 199 distribution-owned files, including the required
  prompts, configuration, UI, and `main.py`, without neighboring dependencies.
- All ten declared study protocols pass offline preflight. All 67 UAT tests
  collect in the Python 3.10 environment; live UAT is not run for this task.
- Public local documentation links, whitespace checks, new-file formatting, and
  scoped lint pass. Scoped lint retains exclusions for existing E402 import
  placement and two existing unused native-runner imports.

No live study or real inference request is made during this correction. The raw
comparison remains explicitly unrun live; its fake-backend tests are not promoted
to behavioral evidence. Production configuration and historical study artifacts
are unchanged.

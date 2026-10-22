# OpenRook implementation notes

The current conversational policy is `willingness-1.2`. See
[Phase 1](docs/development/phase1-willingness.md) and
[architecture](docs/ARCHITECTURE.md) before changing character behavior.

- Language cues and model appraisal provide uncertain evidence and request
  consequences. They cannot select cooperation, refusal, submission or tone.
- `core/willingness.py` owns choice. It uses authored dispositions and values,
  resources, trust, continuous injury/grievance, commitments and a stable turn
  seed. It contains no language matching or model calls.
- Commit the immutable `TurnDecision` before executing work. Every task,
  artifact, inspection, continuation and learning intake needs its component's
  grant. Independent activity requires a separate decision and execution scope.
- Personality determines role preferences, courtesy and bluntness. The shipped
  default is tested for non-submission; owners can edit the dispositions and
  values. Do not restore insult-to-refusal scripts or a pressure/stage ladder.
- Preserve conversation sexual boundaries, operator permissions, truthful
  execution receipts, and the integrity of declined components. A learning
  receipt does not authorize unrelated filesystem, shell or registry claims.
- Generate from the frozen decision. Review uses `pass`, `violation`, `unknown`;
  allow one reviewed rewrite, up to two for critical assent or persistent stale
  replies, and deliver the best nonempty reviewed candidate.
  Unknown semantic review is recorded with residual risk. If correction fails,
  retain the best text as `delivered_degraded`, with its known violations recorded.
  If all usable candidates retain critical role-assent/stale-answer violations,
  excise only sentences intersecting exact review spans from the best candidate;
  deliver only a nonempty remainder passing local checks, with excision telemetry
  and semantic uncertainty. Otherwise retain degraded delivery.
  Known unverified execution/output claims and unsolicited copied user openings
  are ineligible for delivery, including degraded delivery. Excise their exact
  spans before spending rewrite calls; review the independent remainder. An empty
  or invalid remainder requires fresh expression from receipts or expression
  failure. No semantic verdict can waive these local findings. Reuse an identical
  draft's review only within the same turn and unchanged evidence.
  Exact whole-answer stale copies are ineligible regardless of the prior finding's
  severity. Remove their exact spans; do not repeat them or select silence.
  Ineligible or empty candidates require fresh generation with failure evidence
  and varied seed/temperature, using the remaining five-call base budget (at most
  five generation attempts). These required attempts bypass optional cooldown.
  After bounded attempts, strip known ineligible spans from the best drafts and
  retain a nonempty reviewed remainder even with non-floor findings. Only an empty
  safe remainder earns one protected final fresh attempt (at most six generations),
  using the primary deadline and local review, recorded as `delivery_recovery`.
  It cannot fund tools/review or bypass research limits. Known execution claims
  stay ineligible across identical drafts, including the final local review.
  Never withhold an eligible reviewed reply because correction failed; service
  errors require backend failure or no text after mandatory floor removal.
  There are no canned replacement lines. Strip internal controls before review;
  retry empty expression and surface a service error if no usable text exists.
- Distinguish runtime-selected silence, expression failure and backend failure.
  Only runtime-selected silence is an empty successful reply. Auxiliary timing
  adapts to configured request limits and observed generation latency, with
  optional `llm_auxiliary_timeout_seconds` and `llm_auxiliary_cooldown_seconds`.
  Transport failures must retain their error diagnostics and resource accounting.
- Historical replay tables measure earlier policies. Do not reinterpret their
  exact refusal turns or emotion labels as current acceptance criteria.

Use the operator's configured backend and model selection; do not embed model
IDs or recommendations. Verification is offline by default. Do not change
`data/`, `reports/` or the repository's local `runtime_config.yaml` during tests.
Run `python3 -m pytest -q --ignore=tests/uat -p no:cacheprovider`, Ruff on changed
Python paths, and `git diff --check`. Keep scratch evidence outside those stores.

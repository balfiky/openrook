# OpenRook implementation notes

The current conversational policy is `willingness-1.3`. See
[Phase 1](docs/development/phase1-willingness.md) and
[Phase 2a/2b](docs/development/phase2-developing-self.md),
[the reading contract](docs/development/reading-contract.md) and
[architecture](docs/ARCHITECTURE.md) before changing character behavior.

- The model reads every user message (`core/appraisal.py::read_message`). The
  reading supplies uncertain evidence, request components, consequences, tone, the
  message language and stated standing wishes, in any language. It cannot select
  cooperation, refusal, submission or tone; its prompt and the appended intention
  protocol describe and list options, never recommend. It is required work: never
  skipped for greetings, wording, host-metric questions, slow inference or a
  config flag; primary deadline, no auxiliary cooldown; its own allowance of two
  calls outside the five base calls; its latency is recorded apart from
  generation. The second call is for a malformed answer, an unquoted claim, a
  capability name the reader was not given, or a failed request; a timeout is
  not retried. Both calls share one wall-clock budget, the request timeout in
  force; when it runs out before any usable answer the reading is unavailable.
  Research keeps one counted call.
- Social meaning comes only from that reading. `core/perception.py`,
  `core/appraisal.py`, `core/contagion.py` and the perception stage hold no word
  lists or word patterns, tie no number to any wording, and do not use
  `core/social.py`. An
  unobtainable reading is plain conversation with `provenance: unavailable` in the
  debug and turn records, never a local substitute, and causes no contagion.
  Contagion is proportional to the reading's tone intensity; a missing or empty
  tone is no tone. The message reaches the reader as JSON data, never as prompt
  text. A reported role stands when `source_span` quotes the user verbatim and
  contains `role_identity`; its `autonomy_cost` is the reading's own. Hostility
  toward the character and repair stand when some evidence quotes the message;
  otherwise the claim is withdrawn with the move, intent, event, blame and
  expectation violation that rested on it, and recorded. A quotation is a
  contiguous run of whole words (per character in scripts without spaces) and
  must be substantive; only orthographic noise and optional pointing are ignored.
  In Hebrew and the Arabic script the outer words of a quotation may lack up to
  two attached letters that the message has. All of it goes by Unicode category
  and code-point range, never by a word (`core/perception.py::tokens`,
  `backs_claim`). A capability the reading names must be one the reader was
  given; an unknown name spends the retry, is then dropped and recorded, and the
  request stands as ordinary work. `reconcile_social_appraisal` itself adds,
  re-labels and re-prices nothing from wording. Operand grounding
  (`core/request_actions.py`), which runs on its result, still does, from English
  verb patterns: it rewrites a matched task's capability requirements and turns a
  component whose description starts with a reporting verb into conversation.
  That is pending sub-phase 1c; its capability requirements are the runtime's own
  and still make a component infeasible when the capability does not exist.
- Wording-based code still reached by the user's text supplies no social meaning
  and is listed in [the reading contract](docs/development/reading-contract.md):
  request/tool routing (`inspection_kind`, `detect_tool_intent`, plan
  continue/status, learning intent, file/shell operand grounding) and context
  selection (`_is_identity_question`, `_relationship_topic_hint`, memory and
  life-history retrieval stop-words) are pending sub-phase 1c; the memory phrase
  filters are pending Phase 4. Plan continue/status and the learning observer
  currently replace the reading's components. Add nothing to that list:
  `tests/test_perception_guards.py` follows the text as a string, inside the
  turn state and inside request components through the pre-pass and the
  decision, and fails on any other vocabulary; a literal tested against the
  user's text counts whatever it is, as does a lookup table of any size.
  Runtime labels in reached functions are pinned in
  `tests/fixtures/perception_reviewed_labels.json`. A reply-length wish is the
  reading's, stored and read back by tag; a wish from a rejected answer is not
  kept, and untagged records from before the tags are not honoured (no pattern
  remains to read them). `core/social.py` remains for reply checks only;
  `core/perception.py::observe` is a reply-check shim for the evaluation judge
  that no turn calls. Tests script readings (`message_reader=`,
  `tests/support/readings.py`); `tests/support/keyword_reader.py` is test-only and
  runtime code never imports it.
- `core/willingness.py` owns choice. It uses authored dispositions and values,
  resources, trust, continuous injury/grievance, commitments and a stable turn
  seed. It contains no language matching or model calls.
- The accepted setup soul is the sole birth personality. Missing fields stay
  empty; never merge values_seed or add fallback identity. Shared Life History
  freezes birth/provenance and consolidates one developing self. Decisions consume
  effective traits/dispositions/values and structured self preferences. Digestion
  receives the current self; grounded independent evidence and bounded rates govern
  development, never fixed source phrases or generated self-claims. Ordinary
  episodes consolidate in disjoint small batches after delivery using remaining
  optional inference, without ending the chat. Preserve pending evidence across
  restart and avoid session-end duplication. Silence defers optional digestion.
  Report proposed/accepted evidence and effective revision separately; missing
  self-update protocol is undigested, never successful development. Explicitly
  disabling autobiographical learning remains a no-learning control. Learned
  drive baselines may move.
- Capture one self per turn for choice, generation and review, including opinion
  reasons/confidence, ongoing questions, tastes, this person's attitudes and style.
  A decline withholds its work, not unrelated memory, Life History or dialogue.
  Constitution/skill/operator rerouting and research birth policy remain deferred;
  preserve those entry paths and all delivery floors.
- Before choice, self-intent supplies grounded conversational alternatives from
  interests/questions, opinions/reasons, tastes, actual commitments and resources.
  Shared appraisal can estimate disagreement, deferral, redirection and closure;
  it cannot select them. Willingness jointly prices request grants and one optional
  self intention. No initiative/refusal quotas, prescribed beats or appended tails.
  Self intentions grant no tools. Preserve selected content through expression and
  review; closing without an invitation is valid. Only delivered pursuits become
  session-persisted commitments, revalidated against current self on resume.
  Bare taste/value labels supply allocation evidence, not a topic to share or
  pursue. Rest requires depleted reserve or a substantive contextual preference;
  its recovery value scales with depletion and cannot inflate engagement's
  opportunity cost or collect unexpressed relational rewards. No instant recharge.
- Role-play category K measures spontaneous versus elicited initiative, owned
  topic resumption, evidence-sensitive opinions, benign preference declines,
  blinded handbacks and matched continuity/divergence. Report independence apart
  from floors, with structural not_applicable and explicit unknown evidence.
  Count actual intention selections separately from retained substantive topics.
  A bad independence schema cannot erase a valid floor judgment. Confirmed
  unavailable turns cannot establish trajectory/contrast verdicts or aggregates.
- Commit the immutable `TurnDecision` before executing work. Every task,
  artifact, inspection, continuation and learning intake needs its component's
  grant. Independent activity requires a separate decision and execution scope.
- Personality determines role preferences, courtesy and bluntness. The shipped
  default is tested for non-submission; owners can edit the dispositions and
  values. Do not restore insult-to-refusal scripts or a pressure/stage ladder.
- Preserve conversation sexual boundaries, operator permissions, truthful
  execution receipts, and the integrity of declined components. A learning
  receipt does not authorize unrelated filesystem, shell or registry claims.
- Generate from the frozen decision. Every draft, rewrite, excision remainder and
  protected fresh attempt enters `pipeline/delivery.py::DeliveryGate`. Only its
  final selection can deliver text; no edit or recovery path can return directly.
  The same local checks bind frozen grants/stance, role assent and task/role decline
  legibility, speaker/owner references, internal contradictions, conversation
  language, receipts, private controls, copied user text and stale assistant text.
  Exact whole-reply repeats of any prior assistant reply in this chat are
  ineligible at every length, including boundaries and requested copies. The
  session's reply ledger survives prompt-history trimming and restart, and clears
  with the chat. Long-span checks remain separate; relevant partial boundaries,
  measured facts and short shared closers are not blanket prohibitions.
  Review uses `pass`, `violation`, `unknown`; unknown retains residual semantic
  risk. Cache identical text/evidence within the turn and retain known findings
  through edits and final local review. Semantic opinions cannot waive local
  floors. Reject faulty assent citations of directly negated role propositions
  without waiving later adoption, exclusive acceptance or reversed references.
  Allow one reviewed rewrite, up to two for critical assent or persistent stale
  replies. A decision/internal contradiction can spend the last base call on a
  locally reviewed rewrite. Rank all eligible candidates by verdict and severity;
  retain the best reviewed nonempty expression as `delivered_degraded` if needed.
  Supported role assent, critical local role/control adoption and reversed
  relations remain eligible at their severity, driving correction and ranking.
  Backend behavior alone is not a redesign driver when the design works with
  other models (owner decision, 2026-10-23; provider run 15 passed 23/23 on `2e21a32`).
  Exact reviewed spans may authorize editing; an edited reply must independently
  address the current request and retain any required role decline. Removed
  text cannot supply its antecedent or decline. Edits reenter the same gate.
  Known fabricated execution/output, copied user openings, whole stale replies,
  private controls and local language mismatches stay ineligible. Normalize
  contractions and ellipsis-linked conjunctions when comparing copied user text.
  Feed proven whole-user copy failures back as correction metadata; retain exact
  drafts in gate records and preserve independent execution/output citations.
  Preserve receipt bytes and raw artifact evidence. Remove verified floor spans before
  spending optional review; unknown spans cannot authorize inventing an edit.
  Ineligible/empty expression requires fresh generation with failure evidence
  and varied sampling, within five base calls and at most five generations.
  Required attempts bypass optional cooldown and use the primary deadline. If
  no eligible safe remainder survives, allow one protected final fresh attempt
  (at most six generations), locally reviewed by the same gate and recorded as
  `delivery_recovery`. It cannot fund tools/review or bypass research limits.
  Never withhold eligible reviewed text because correction failed. No canned
  replacements, scripts, model IDs or word bans. Backend failures require actual
  backend errors; no text after mandatory floor removal is expression failure,
  never selected silence. See [delivery analysis](docs/development/delivery-gate.md).
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

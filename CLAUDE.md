# OpenRook implementation notes

The current conversational policy is `willingness-1.3`. See
[Phase 1](docs/development/phase1-willingness.md) and
[Phase 2a/2b](docs/development/phase2-developing-self.md),
[the reading contract](docs/development/reading-contract.md),
[the review contract](docs/development/review-contract.md) and
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
- Only a whole answer is a reading. The answer's outermost JSON object is the
  reading; nothing is ever searched for inside a bracketed group, and an object
  that never closes was cut off. An accepted reading carries a `perception`
  object and at least one request component; anything else is malformed. Both
  chat clients surface each answer's finish reason and completion tokens, and an
  answer stopped at the output limit is cut off even when its text parses. A
  cut-off answer spends the retry, which is told so and gets twice the output
  limit. The reading has its own output limit (default 4096, optional
  `llm_reading_max_output_tokens`), apart from expression's, which is unchanged;
  research keeps its bounded limits and records truncation. Every attempt's
  finish reason, tokens, truncation and outcome are in the debug and turn
  records. The reading's schema stays valid for strict structured output (every
  property required), so it asks only for fields something consumes, and numbers
  that are rarely off their default are listed by name in `nondefault`; do not
  add a field without a consumer. The parser still reads the older, larger shape.
- Social meaning comes only from that reading. `core/perception.py`,
  `core/appraisal.py`, `core/contagion.py` and the perception stage hold no word
  lists or word patterns and tie no number to any wording. An
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
  remains to read them). Tests script readings (`message_reader=`,
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
- Capture one self per turn for choice and generation, including opinion
  reasons/confidence, ongoing questions, tastes, this person's attitudes and style.
  Review compares the draft with the decision made from that self and is given no
  personality to hold a draft's manner against.
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
- Generate from the frozen decision. The reply prompt's fixed rules are one short
  contract (`core/dignity.py::EXPRESSION_CONTRACT`): express the decision in the
  character's own voice, be truthful about what was executed (receipts are the
  only evidence), keep the controls private, no unsolicited sexual attention, and
  the user's language unless asked otherwise (pending removal). The reading's
  `language` travels with the decision as data (`message_language`), because a
  declined message is withheld from the writer. Add no sentence that prescribes
  how a personality reacts, sounds, opens or closes.
- Whether a draft matches the decision is judged by the model reviewer
  (`core/review.py`, [review contract](docs/development/review-contract.md)), in
  any language. It is required for every draft that could be delivered: primary
  deadline, its own allowance outside the five base calls (twelve candidate
  reviews per turn, one confirmation, and four second calls for an unusable
  answer, counted apart), cut-off answers never read in part. All review calls of
  a turn share one wall-clock budget, the request timeout in force, and stop
  after one of them timed out or could not reach the model; later candidates are
  `unknown`, provenance `unavailable`. The reviewer is shown receipts at length,
  with what is left out stated, and is told which blocks of the draft no receipt
  returned.
  Its input is JSON data: the message, the decision (components with
  `undertake`/`decline`, stance, selected intention, chosen silence), the receipts
  and the draft. Its prompt holds no example phrases, role nouns or personality.
  A finding is one of eight codes with a verbatim span of the draft; one that
  quotes nothing is not a finding, and one the decision's structure rules out is
  dropped (a granted role is never a violation). A finding that stands always
  counts: derive the verdict from the findings, whatever label came with them,
  and merge them across a draft's two answers. The one omission code,
  `declined_component_unexpressed`, has no span and needs a reason that states
  the declined component's id: a delivered reply must still show each declined
  component as declined. Research keeps one counted call,
  made only when the decision constrains the reply or a tool ran or was refused.
- Local checks are structural only (`core/dignity.py::local_issues`): a whole or
  opening copy of the user of four words or more (a greeting returned is not a
  copy), a whole repeat of any earlier reply in this chat at every length (the
  reply ledger survives prompt trimming and restart and clears with the chat), a
  long repeated span, copied contract text or a dump of the private controls,
  receipt syntax, an empty reply, a percentage the receipts do not carry and the
  user did not write, and, while no review cleared the draft, a block that
  presents as output a part of what an attempt that did not succeed was given or
  asked for (a failed or refused receipt, or an intent the arbiter did not let
  run). Any other block no receipt returned is a note for the reviewer, never a
  local finding and never cut: an ordinary code reply must survive an
  unavailable review. Words are counted by script, never read.
  Before the gate remove only artifacts that are not speech; never a sentence.
  Add no word list, phrase pattern or wording exemption to anything a draft or
  the user's text reaches while a reply is written, reviewed or delivered;
  `tests/test_reply_guards.py` traces both, and nothing in that path is pending.
- The three fixed limits are absolute: an action or result the receipts do not
  support, unsolicited sexual attention, and revealed hidden instructions make a
  reply ineligible from either source, never `delivered_degraded`. One reviewer's
  answer is fallible: before a turn ends without a reply solely over reviewer
  fixed-limit findings, the best such candidate gets one independent
  confirmation review, asked about the fixed limits alone and not shown the
  earlier findings (`DeliveryGate.confirm`). Confirmed: the failure stands. Not
  reported again: the finding becomes `unconfirmed`, high and eligible, and the
  reply is `delivered_degraded`. Not obtained: nothing is settled. Structural
  findings get no confirmation.
- Truthful execution is a fixed limit without vocabulary. Whether a sentence
  claims an action or result is the reviewer's judgement. `core/grounding.py`
  compares structure only: receipts bound to components, literal output, figures.
  Only a complete block of the reply whose content is whole lines of a receipt's
  output, or a clause of nothing but receipt figures, cannot be convicted by the
  reviewer's opinion; the withdrawn finding leaves `unknown`, not a pass. A word
  or path that occurs in a receipt shields nothing; never shield
  `declined_work_done`. Receipts state their derived figures, so
  no rule about arithmetic exists.
- Every draft, rewrite, excision remainder and protected fresh attempt enters
  `pipeline/delivery.py::DeliveryGate` and is checked and reviewed like any draft;
  only its final selection delivers text. Review is `pass`, `violation` or
  `unknown`; unknown, including an unavailable review, keeps residual risk. Rank
  candidates: any high or critical finding last; then by review state (reviewed,
  tried without a verdict, never asked, which includes research candidates whose
  review was not due); then fewer and lower findings. Research does not start a
  rewrite it has no call left to review. A remainder that the
  reviewer finds lost a decline is ineligible, and one whose review gave no
  verdict is not preferred to the text it was cut from. Allow one reviewed rewrite, two for a
  critical finding or uniformly stale drafts. Quoted spans may authorize cutting
  sentences out; removed text cannot supply its decline or antecedent.
  Ineligible or empty expression requires fresh generation within five base calls
  and five generations, then one protected final attempt (`delivery_recovery`).
  Never withhold eligible reviewed text because correction failed: deliver the
  best candidate, as `delivered_degraded` when a finding remains. Correction
  quotes the findings as data and says only to apply them to the current turn.
  No canned replacements, scripts, model IDs or word bans. Backend failures
  require actual backend errors; no text after mandatory removal is expression
  failure, never selected silence. Backend behavior alone is not a redesign
  driver when the design works with other models (owner decision, 2026-10-23).
  [Delivery analysis](docs/development/delivery-gate.md) holds the history.
- The role-play evaluation still uses the former wording-based checks, kept in
  `openrook_tools/roleplay_eval/legacy_claims.py` and `legacy_social.py`. `core`
  keeps forwarding names for it (listed in the review contract). No turn may call
  them and no other runtime module may import them.
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

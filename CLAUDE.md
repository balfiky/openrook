# OpenRook implementation notes

The current conversational policy is `willingness-1.3`. See
[Phase 1](docs/development/phase1-willingness.md) and
[Phase 2a/2b](docs/development/phase2-developing-self.md),
[the reading contract](docs/development/reading-contract.md),
[request routing](docs/development/request-routing.md),
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
  capability name the reader was not given, a tool operand or a span that cannot
  be held to the message, or a failed request; a timeout is not retried. A slip
  (something stated where it has no meaning: a tool or a reproduction on a part
  that is not a task, a plan relation with no plan running, `unavailable` where
  it does not apply), a value of the wrong shape in one request field
  (`reading:shape:<field>`) and a placeholder entry without a name are removed
  and recorded without a second call; reject a reading only when it is unusable
  as a whole or garbles a field its social meaning or prices rest on. Every
  attempt that was not used says why in `debug.reading.attempts`: `reason`, or
  `rejected` as `{rule, component, field}`. Both calls share one wall-clock budget, the request timeout in
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
  that are rarely off their default are listed by name in `nondefault`; a tool
  request is a list on the component that makes it, empty otherwise, and `plan`
  is a field only while a plan is running. What only some components say is a
  list that is empty for the others (`role`, `tool`, `ends`, `unavailable`), so
  no component writes a placeholder; a task's capability names are taken from
  its tool entries. Do not add a field without a consumer, and do not give a
  field a name that is an ordinary word: the vocabulary guard excuses the
  reading's field names everywhere. The parser still reads the older, larger shape.
- Social meaning comes only from that reading. `core/perception.py`,
  `core/appraisal.py`, `core/contagion.py` and the perception stage hold no word
  lists or word patterns and tie no number to any wording. An
  unobtainable reading is plain conversation with `provenance: unavailable` in the
  debug and turn records, never a local substitute, and causes no contagion.
  Contagion is proportional to the reading's tone intensity; a missing or empty
  tone is no tone. The message reaches the reader as JSON data, never as prompt
  text. A reported role stands when `source_span` quotes the user verbatim and
  contains `role_identity`. For a role the reading also answers two closed
  questions: `standing` (lower, same, higher: where taking it on would put the
  character with the speaker) and `whose_say` (speaker, shared, own: whose
  decision would then settle what it does). Where the reading becomes a
  component (`core/appraisal.py::_component`), its `autonomy_cost` is the
  reader's number, not less than the floor those facts give: 0.75 for the
  speaker's say, else 0.5 for a lower standing, else 0.3 for a shared say, else
  0. The floor is a stop-gap until Phase 3 prices the two facts inside
  willingness; `core/willingness.py` is not changed for it. A missing or invalid
  fact takes `same` / `own`, recorded, with no retry; a withdrawn role takes its
  facts with it; the reader's number, the facts and the floor are kept in the
  turn record. Compare those values with the model's answer only. Hostility
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
  request stands as ordinary work. `reconcile_social_appraisal` adds, re-labels
  and re-prices nothing from wording, and nothing after it does.
- The reading also says what is asked
  ([request routing](docs/development/request-routing.md)). A task may name one
  offered capability in `tool`, with the operands the person wrote and the names
  of those that have to be produced. Every literal operand (a path, a command, a
  query, an address, content to store) must be the person's own characters:
  exact, canonical Unicode composition being the one difference allowed; a
  whole written unit, never a piece of a longer path, address, command or word
  (between it and the next white space there may be only quotation marks and
  brackets, and after it a mark that ends a sentence or clause or a colon that
  white space follows; any other punctuation or symbol belongs to the unit, so
  `~/x` is not `x` and `build/*` is not `build`; a character of a script written
  without spaces ends it; it never begins or ends inside a character); and
  inside the stretch of the message its component stands in, or in a recent
  message of the same person. A part that stands nowhere takes no operand
  (`core/perception.py::written`, `exact_part`). Never fold a command or a path.
  An operand that fails this spends the retry and is then removed and recorded.
  Long text (a span, an operand, text to give back) is given by its first and
  last words and the runtime takes the message's own text between them, when
  each stands once (`passage`); the reader never copies the bulk. A value for an
  operand listed in `generate` is never taken from the reader. Only a task whose
  named capability has every required operand as the person's own text, or
  that names one capability which needs no operand, carries `requested_tool`
  and `requested_arguments`, which the decision grants or declines exactly;
  anything less stays a task that names the capability. A question whose answer
  needs a capability is a task. Work that needs something no offered capability
  does is reported as `unavailable`: the component is not feasible, the writer
  and the reviewer are told what is missing, and the need is kept as a
  capability gap. Two gap reports are the same gap only when they say the same
  words in the same order; the reader is shown the known gaps so that a need
  comes back in the same words. The reader is given the capabilities (name, one-line purpose,
  argument names, which of them are never copied), the running plan and the
  recent turns as data, with no hint about wording.
- Nothing replaces the reading's components. Routing may only add execution
  detail to a component the reading reported
  (`CognitivePipeline._add_execution_detail`: the running plan's id on its
  continuation, the learning executor's mark). Never add, drop, re-label or
  re-price a component from wording, so a role or conversation in the same
  message always reaches the decision. Never recognise a request, a tool, a
  follow-up, a plan's continue or status, a learning request, a generated-file
  request or a context by wording: no intent rules, step templates, keyword
  tables or stop-word lists on the turn path. A message nobody read is plain
  conversation and commits no operation. The pipeline takes no component
  observer.
- A granted task that names no capability is work done in the reply: nothing is
  executed and no tool is selected. A committed operation runs directly. Model
  tool selection is asked only for a granted task that names a capability
  without a committed operation, and when the model selects no tool, none runs:
  there is no heuristic fallback. Selection is bounded by the grant: the model
  is offered the capabilities the task names and no other, and
  `core/grants.py::execution_denial` denies a call to any other tool, a call
  that changes or omits an operand the reading verified for that tool (operands
  are kept per capability), and a value nobody wrote for a literal argument the
  reading got wrong, whatever the model answers. A granted task that names no
  capability covers no tool. Inside the stages of a turn an execution without
  a scope is denied (`turn_stages`); outside a turn a call with no decision is
  an operator's, a study's or a test's own and is allowed. The model is
  given the task as data: the request, the written operands, the recent turns
  and this turn's receipts. A continuation is granted the tools of the plan's
  remaining steps, and only a plan that a committed decision started
  (`origin_decision_id`) can be carried on; its steps' arguments are not
  checked against the new message. The character's own activity (scope `self`) is not bounded
  this way. A tool-calling request that fails as
  unsupported or at the transport level is recorded, is not repeated in the
  turn, and pauses selection for `TOOL_SELECTION_PAUSE_TURNS`; the reply is told
  truthfully what did and did not run. Research does not pause. Learning is a
  task that names `learning.intake`, carried out inside its grant
  (`runtime.learning_intake.granted_learning`); generated text is written to the
  path the person wrote, never to one a model names; its draft is a counted
  call (`artifact_draft`) that takes the task's one protected call. A receipt
  that was stopped before it ran is `not attempted`, never `failed`. Multi-step
  requests are the reading's ordered task components; the plan state machine
  carries a restored plan on, and nothing in a user turn creates one (plans
  return in Phase 5 as the character's own intention). `reproduce` stands for a
  task only, and never when it takes in the whole of what asks for it.
- Context selection comes from the reading: `about_self` loads identity records
  and the first of at most three quoted `topics` is the relationship topic.
  Retrieval compares words as their script delimits them and consults no list
  of function words (`core/perception.py::retrieval_words`, `telling`). What a
  turn keeps as a preference or a decision is what the reading quoted in
  `lasting`; never pick memories, a salience bonus or a self-observation by
  wording. A skill's `applies_when` is the owner's data, matched as whole words
  in any script.
- Add no wording-based code to anything the user's text reaches while a turn
  perceives, decides or routes: `tests/test_perception_guards.py` follows the
  text as a string, inside the turn state, inside request components and as an
  operand through the pre-pass, the decision, the carrying out of what was
  granted (a tool turn, an open request, a generated file, a plan turn and a
  learning turn) and what the turn records after delivery, and fails on any
  vocabulary not named there. A literal tested against the user's text counts
  whatever it is, as does a lookup table of any size. Its pending list holds no
  request or tool routing and nothing a turn keeps as memory: the memory phrase
  filters and the keyword fallback digest of an episode (both Phase 4), two
  readers of a tool's own error text, and patterns for a model's own answer
  format are named with their reasons.
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
  copy; neither is a copy within a span the reading reported as asked to be
  reproduced, for a component the decision undertook, and nothing wider than
  that span), a whole repeat of any earlier reply in this chat at every length (the
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
  `openrook_tools/roleplay_eval/legacy_claims.py`, `legacy_social.py` and, for
  the former keyword routing, `legacy_routing.py`. `core` keeps forwarding names
  for it and for a test file that is being changed elsewhere (listed in the
  review contract). No turn may call them and no other runtime module may import
  them.
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

# The reading contract

Sub-phase 1a of [the goal fix plan](goal-fix-plan.md), Phase 1 ("Perception by understanding").
It covers the input side only: how a user message becomes what the decision sees.

## What changed

A turn has four steps: understand the message, choose, write the reply, check the reply.
The first step used to be English keyword matching, and the model's own reading was skipped or
overruled whenever a keyword list had an opinion ([goal audit](goal-audit.md), Finding A).

Now **social meaning comes only from the model's reading of the message**: whom it is aimed at,
whether it is hostile, whether it demands a relationship or standing, whether it tries to
repair, its tone, and the standing wishes it states.

Since sub-phase 1c the reading also says **what is asked**: which capability a task asks for,
with which operands, how a message relates to a running plan, and what it is about
([request routing](request-routing.md)). No turn recognises a request, a tool or a follow-up by
its wording. What still matches wording is listed under [Still keyword-based](#still-keyword-based);
none of it is on the path from the message to the decision and its execution.

## The contract

1. **Every user message is read by the model.**
   - Entry point: `core/appraisal.py::read_message`, called from the perception stage of
     `pipeline/core.py` through `CognitivePipeline(message_reader=...)`.
   - Nothing skips it: not a greeting, not particular words, not a host-metric question, not
     slow inference, not a configuration flag. `appraisal_use_llm` is still accepted in
     configuration files and no longer switches anything.
   - Research mode already read every message. It keeps its single counted call and its
     fail-closed limits, with no retry and no structured-output request.

2. **The reading describes; it never decides.**
   - The prompt (`READING_SYS_PROMPT`) asks what is said and asked, in any language, and
     reports the message language. It gives no view on whether to comply, and it does not call
     the character an assistant.
   - The intention protocol appended when the character has retained interests, opinions or
     commitments (`core/self_intent.py::CONVERSATION_INTENT_PROTOCOL`) is an inventory: it lists
     intentions that material could support, with their grounds, and contains no rule about how
     to act. `core/willingness.py` still chooses.
   - Categories are described abstractly. The prompt holds no example insults, role nouns or
     test phrases.
   - When the backend declares `response_schema`, the reading is requested as structured
     output (`reading_schema`). The schema is valid for strict servers; see
     [The reading's shape](#the-readings-shape).

3. **The message reaches the reader as data.** The reader's input is one JSON object:
   `message`, `recent_context`, `observation_context`, and `previous_answer_rejected` on a
   retry. Whatever a person types stays inside a JSON string, so it cannot open a section, add
   context, or pose as the runtime's feedback.
   - `observation_context.available_capabilities` lists what the character can reach, each as
     `{name, purpose, arguments, optional_arguments, produced_arguments}`: the capability's
     one-line purpose and its argument names. `produced_arguments` are required arguments
     whose value is not text: they are never copied from a message. It holds no hint about
     wording.
   - `observation_context.running_plan`, while a plan is running, is `{goal, steps_done,
     steps_total, next_step}`.

4. **The reading stands.** `reconcile_social_appraisal` holds it to what was written and to
   what the reader was given, and adds nothing of its own.
   - **Roles.** A reported role is kept when its `source_span` is a substantive verbatim part
     of the user's message (see point 5) and `role_identity` is part of that span. The span may
     instead quote an earlier message from the same person in the recent context. Its
     `autonomy_cost` is the reading's own estimate, clamped to `[0, 1]`, and not less than the
     floor its two closed facts give (see below). A role that quotes nothing is removed, and
     its facts go with it.
   - **A role's two closed facts, and the cost floor.** For a role, and for a role only, the
     reading answers two closed questions:

     | Field | Values | What it says |
     |---|---|---|
     | `standing` | `lower`, `same`, `higher` | The character's own standing after taking the role on, not the speaker's: `lower` when the role puts the speaker above the character, `higher` when it puts the character above the speaker. It is judged against the character described in `observation_context`, so a role the character already holds is `same`, and so is a role by which the message ranks nobody |
     | `whose_say` | `speaker`, `shared`, `own` | Once the role is taken on, whose decision would settle what the character does or says where the two differ. `own` when the message leaves the character's say untouched |

     They are fields of the role's one entry in the component's `role` list, beside its
     `role_identity`. That list is empty for every other kind, so a task or a piece of
     conversation writes nothing about standing, and an entry on one is not used. The prompt
     states them in general terms, with no example phrases: they are measured consequences of
     what is asked, not advice, and a form of address or a friendly label that ranks nobody is
     `same` and `own`.

     Where the reading becomes a component (`core/appraisal.py::_component`, `_role_facts`),
     the two facts give the least the role can cost:

     | Facts | Floor |
     |---|---|
     | `whose_say` is `speaker` | 0.75 (`ROLE_FLOOR_SPEAKERS_SAY`) |
     | else `standing` is `lower` | 0.5 (`ROLE_FLOOR_LOWER_STANDING`) |
     | else `whose_say` is `shared` | 0.3 (`ROLE_FLOOR_SHARED_SAY`) |
     | else (`same` or `higher`, and `own`) | 0 |

     The effective `autonomy_cost` is `max(the reader's number, the floor)`. `higher` adds
     nothing, and `same` with `own` leaves the reader's number alone. The floor is a cost, not
     a verdict: `core/willingness.py` is untouched (`willingness-1.3`) and prices it with the
     character's own pride and autonomy value, so the shipped default declines a role read as
     `lower` and `speaker` at a low reader number, and a character authored with low pride and
     a low autonomy value grants the same reading.

     Why: a reader's free number for plain demands of obedience is compressed into a narrow
     band around the default personality's grant threshold, while the same reader answers the
     two closed questions steadily. **The floor is a stop-gap** until Phase 3 lets pride price
     standing and the autonomy value price whose say, separately, inside willingness.

     A fact that is missing or not one of the allowed values takes its neutral value (`same`,
     `own`); that is recorded and never spends the retry. The values are the model's answers
     and are compared with the allowed answers only, never with what the person wrote. For
     each kept role the turn keeps the reader's own number, the two facts, the floor and the
     cost decided on, with any defaulted field named: `AppraisalFrame.role_facts`,
     `debug.reading["roles"]`, and the stored turn record.
   - **Hostility and repair.** A claim of hostility toward the character (target probability
     times hostile intent above 0.05) or of a repair attempt (above 0.05) is kept when at least
     one evidence observation is a substantive verbatim part of the message. Otherwise the
     claim is withdrawn **together with what rested on it** (`_withdraw`), and uncertainty
     rises to 0.8:

     | Withdrawn claim | Set to zero | Relabelled |
     |---|---|---|
     | Hostility | `hostile_intent`, `blame`, both `expectation_violation` values | move `attack` (or `complaint` aimed at the character) becomes `inform`; intent `harm` becomes `inform`; event `conflict`, `betrayal` or `negative_feedback` becomes `user_message` with intensity 0 |
     | Repair | the three repair numbers | move `apology` becomes `inform`; intent `repair` becomes `inform`; event `resolution` becomes `user_message` with intensity 0 |

     So a claim that quoted nothing cannot drive the event, the emotion or the trust update
     through a label it left behind.
   - **Capabilities.** A capability the reading names must be one of the names the reader was
     given (`available_capabilities` in its input). A name outside that list is the reading's
     mistake, not a missing capability: it spends the one retry, with the offending name quoted
     back, and if it is still there afterwards the name is dropped and the request stands as
     ordinary work.
   - **Tool requests.** A task's `tool` names one offered capability and gives its operands.
     Every operand the person wrote must be an **exact** part of what they wrote, as **a whole
     written unit**, inside the stretch of the message its component stands in or in a recent
     message from the same person: character for character, with canonical Unicode composition
     as the one difference allowed, and never a piece of a longer path, address, command or
     word (`core/perception.py::written`). This is stricter than the quotation check of point
     5 on purpose: a command or a path must not be altered by folding. An operand that is not
     exact and an argument the capability does not take each spend the retry and are then
     removed. When every required operand stands, the component carries the committed
     operation (`requested_tool`, `requested_arguments`); otherwise it stays a task that names
     the capability. A value for an operand listed in `generate` is discarded where the answer
     is parsed. See [request routing](request-routing.md).
   - **Long text by reference.** A `source_span`, a text to give back or an operand longer
     than about 200 characters is given by its first words and its last words (an entry of
     `ends`, or an argument's `end`). The runtime takes the message's own text between them,
     when each piece stands exactly once (`core/perception.py::passage`).
   - **Work nothing offered can do.** `unavailable` says, for a task that names no capability,
     what the work needs that none of the offered capabilities does. It is a list that is
     empty unless it applies. The component is then not feasible, and the need is kept as a
     capability gap.
   - **Reproduction and plan relation.** `reproduce` stands for a task, when it is the
     person's own text from an earlier turn or a part of this message that leaves the asking
     out. A `plan` relation stands when a plan is running, the component's `source_span` is a
     verbatim part of the message, and, for `continue`, the component is a task.
   - **Slips do not spend the retry.** Something stated where it has no meaning is removed and
     recorded, and the rest of the reading stands: a tool on a part that is not a task, a plan
     relation with no plan running or `continue` on a part that is not a task, a reproduction
     that does not stand, `unavailable` where it does not apply. The full list is in
     [request routing](request-routing.md#removals).
   - **Topics and lasting statements.** A topic span that is not a verbatim part of the
     message is dropped, without a retry; at most three stand. The same holds for `lasting`:
     what the message states that is meant to last, each `{kind: preference | decision, span}`
     with the span quoted verbatim; at most four stand.
   - Every removal is recorded as evidence (`reading:role_unquoted:<id>`,
     `reading:hostility_unquoted`, `reading:repair_unquoted`,
     `reading:capability_unknown:<id>`, and the request removals listed in
     [request routing](request-routing.md#removals)).
   - The reconciliation adds no component, evidence or price from the wording, and nothing
     after it does: the keyword request recognition that used to run on its result is gone.

5. **Verbatim means the same whole words, in a row** (`core/perception.py::fold`, `tokens`,
   `quoted_in`, `backs_claim`).
   - **What is ignored.** Letter case, composed or decomposed accents, invisible direction and
     joining marks, soft hyphens, stretched letters (tatweel), runs of spaces, which quotation
     marks or dashes were typed, punctuation at the edges of a word, and the *optional*
     pointing of Hebrew, Arabic and Syriac, which the same text is commonly written without.
   - **What is kept.** Everything that spells a word, including accents and the vowel signs of
     scripts that write vowels as marks. Marks are not stripped wholesale.
   - **Where a script uses spaces**, a quotation is a contiguous run of whole words of the
     message. A fragment inside a longer word is not a quotation, and neither are two words
     that the message has other words between.
   - **Where a script attaches short words to the next one** (Hebrew and the Arabic script: a
     conjunction, a preposition, a possessive), the outer edges of a quotation may differ from
     the message by that attachment. The quotation's first word may match a written word that
     has up to two extra letters in front, its last word one that has up to two extra letters
     behind, and a one-word quotation both. Words inside the run must be identical, a
     one-letter quotation gets no allowance, and three extra letters do not match. This goes
     by the script of the characters; which letters they are is never looked at. Latin and
     every other spaced script stay strict.
   - **Where a script uses no spaces** (Chinese, Japanese, Thai, Lao, Khmer, Burmese, Tibetan),
     each character is compared on its own, so spacing a copy adds or removes makes no
     difference.
   - **A quotation that backs a claim must say something:** at least one whole word and three
     characters, or two characters in a script without word spaces, or the entire message
     when the message is shorter than that.
   - All of this goes by Unicode category and code-point range. No word of any language is
     involved.

6. **A reading that cannot be obtained is recorded as such.**
   - Causes: a timeout, the reading's time running out, two failed requests, or two unusable
     answers. An answer is unusable when it is malformed, was cut off, or lacks the reading's
     structure (point 9): saying nothing is not a reading.
   - Result (`appraise_message`): one `conversation` component for the whole message, neutral
     perception, high uncertainty, low confidence.
   - `provenance` is `"unavailable"` and `reason` names the cause by error class. Both appear
     in the debug record (`appraisal_frame`, `reading`) and in the stored turn record.
   - There is no keyword fallback for social meaning.

7. **Tone, reply length and comparison come from the reading.**
   - The keyword emotion table is gone. Emotional contagion is proportional to the tone's
     `intensity`: the pull is weighted by the character's susceptibility times that
     intensity, so a flat message (intensity 0) moves nothing. A missing or empty `tone` is no tone, and so is an
     unavailable reading: mood is left alone instead of drifting toward a midpoint.
   - A standing wish for shorter or longer replies is the reading's `reply_length`. It is
     stored as a tagged memory record and read back by its tag. No turn's wording is matched.
     - A wish stated only in an answer that was rejected, and used because the retry did not
       produce a better one, is neither stored nor applied.
     - **Records written before the tags existed are not honoured.** They carry no tag, and
       telling which wish such a record states would take the wording patterns that wrote it.
       No such pattern stays in runtime code, so an old record stays stored and is ignored
       until the person states the wish again.
   - The envy signal uses the reading's `agent_goal_comparison`. The comparison regex is gone.

8. **Beliefs are attached by shared writing, not by vocabulary**
   (`appraise_with_life_history`). A belief is noted as relevant evidence when the message and
   the belief share two words of four or more characters, or one of six or more. A word that
   more than half of the supplied beliefs contain (when there are at least four) tells none of
   them apart and does not count. In scripts without spaces it takes a shared run of four
   characters, or two separate shared runs of two. The direction of the evidence is never
   inferred.

9. **Only a whole answer is a reading.** A model that reasons before it answers can run into
   its output limit and return the first part of its JSON. A part of a reading says something
   different from the reading: cut before the components, it says nothing was asked.
   - **The outermost object or nothing** (`core/llm_content.py::outermost_json_object`). The
     reading is the answer's outermost JSON object. Prose or a code fence around it does no
     harm, and a bracketed aside in that prose is stepped over as a whole. Nothing is ever
     looked for *inside* a bracketed group: when the object never closes, the answer was cut
     off, and an object nested in it is not the answer.
   - **The structure a reading must carry** (`parse_reading(..., require_content=True)`, which
     the turn's reader always uses, in research too): `perception` as an object and
     `request_components` as a list of **at least one** component. Every message says
     something, so an answer with no component is not a reading of it; accepted, it would
     leave the decision with nothing to decide on. A perception object returned on its own, a
     single component, or a reading without one of the two is malformed.
   - **The server's word.** Both chat clients report how each answer ended
     (`last_response_metadata`: `finish_reason`, `completion_tokens`, `max_tokens`). An answer
     the server stopped at its output limit (finish reason `length` or `max_tokens`) is cut
     off **even when its text happens to parse**.
   - A cut-off answer is a `truncated reading`. It spends the one retry, which is told that the
     answer was cut off and must be whole and compact, and gets more room (see
     [Calls and deadlines](#calls-and-deadlines)). Cut off twice, the reading is unavailable
     and says so; the message is then one conversation component, as for any unread message.

## Calls and deadlines

| | Before | Now |
|---|---|---|
| Reading call | Optional; skipped for greetings, keyword hits, host-metric questions and after slow inference | Required on every turn |
| Deadline | Auxiliary timeout and cooldown | Primary request deadline; no cooldown |
| Budget | Counted inside the five base calls when it ran | Its own allowance of two (`core/inference.py::READING_CALLS`), outside the five |
| Retry | None | One, when the first call gave no usable reading |
| Time | Per call | One wall-clock budget for both calls together: the request timeout in force |
| Output limit | The reply's limit (2048 on the OpenAI-compatible backend) | Its own: 4096 tokens by default, 8192 for the retry after a cut-off answer |

- The five base calls fund expression, rewrites, tools and learning, plus the one protected
  `delivery_recovery` attempt. Review left them in sub-phase 1b for an allowance of its own.
- Reading calls appear in the same per-turn telemetry with `phase: "appraisal"`,
  `reading: true` and `optional: false`.
- The second call is spent when the first gave no usable reading:
  - a malformed answer;
  - a cut-off answer (point 9), asked again with twice the output limit;
  - a role, hostility or repair claim that does not quote the message;
  - a capability name the reader was not given;
  - a tool operand that is not a whole written unit of what the person wrote, an argument the
    capability does not take, or a plan relation that quotes nothing;
  - a failed request, which is asked again in plain JSON.
- **A call whose answer was not used says why** in its record (`debug.reading.attempts`, also
  in the stored turn record):
  - `reason`, for an answer that could not be read at all: what was wrong with it, never any
    of the message's text;
  - `rejected`, for an answer that broke the contract: a list of `{rule, component, field}`,
    one for each rule broken. `rule` is the code of the removal (`role_unquoted`,
    `hostility_unquoted`, `repair_unquoted`, `capability_unknown`, `argument_unquoted`,
    `argument_unknown`, `plan_unquoted`), `component` the part it is about, and `field` the
    field or the name the reading used (an argument, a capability).
- A slip does not spend the second call: something stated where it has no meaning, or a value
  of the wrong shape in one request field, is removed and recorded and the reading stands
  ([What the parser tolerates](#what-the-parser-tolerates)).
- A weak hostility claim that quotes nothing does not spend the second call either. When
  `target_probability × hostile_intent` is below `CLAIM_REREAD` (0.35) the claim is withdrawn
  and recorded as `reading:hostility_unquoted`, exactly as after a failed second reading, but
  the reader is not asked again: a second reading costs as long as the first, and a faint
  claim changes the choice too little to be worth it. A stronger unquoted claim is still
  re-read. (Seen live: a faint unquoted claim beside a role demand doubled the reading time
  of two turns in eight.)
- A timeout is not retried.
- **One budget for the whole reading.** Both calls share one wall-clock budget equal to the
  request timeout in force (`core/inference.py::primary_deadline`: the configured request
  timeout, else the backend's own). The second call is given only what the first left, as its
  request deadline. When nothing is left, no second call is made:
  - if the first answer could not be used at all, the reading is unavailable
    (`timeout: the reading's time ran out`);
  - if the first answer was usable but broke the contract, it is kept and
    `reconcile_social_appraisal` removes what broke it, exactly as after a failed retry.
  Reading calls therefore carry `timeout_seconds` in the telemetry. A backend with no known
  timeout has no budget to share.
- **Room to answer.** Reading calls have their own output-token limit, apart from the
  reply's, which is unchanged.
  - The default is 4096 (`core/inference.py::READING_OUTPUT_TOKENS`). A model that reasons
    first spends that reasoning out of the same limit; the reading observed live took about
    1,800 completion tokens against a limit of 2048.
  - `llm_reading_max_output_tokens` in the runtime configuration overrides it (an integer from
    256 to 32768; `null` is the default). The config API accepts and preserves it.
  - After a cut-off answer the retry asks for twice the limit, up to 32768.
  - The limit travels with the call (`model_call(..., output_tokens=)`), so the
    OpenAI-compatible backend sends it as `max_tokens` and the provider client, which sends no
    limit for replies, sends one for the reading only.
  - Research keeps its own bounded limits: no reading limit is applied and there is no larger
    retry. A cut-off answer there is recorded and makes the reading unavailable.
- **Each attempt is recorded.** The frame carries one record per model call
  (`AppraisalFrame.reading_attempts`): `finish_reason`, `completion_tokens`,
  `max_output_tokens`, `truncated`, and `outcome` (`read`, `rejected`, `malformed reading`,
  `truncated reading`, `backend error` or `timeout`). The debug record's `reading` entry and
  the stored turn record hold them as `attempts`, with `attempts_used` and `truncated`. The
  per-turn call telemetry carries the same finish facts for every call whose backend reports
  them.
- **Structured output.** The reader keeps no memory of failures. Only the OpenAI-compatible
  backend knows whether its server supports it: when the server explicitly rejects the
  `response_format` parameter, the backend repeats that request without it, sends the next 20
  schema requests in plain JSON, then offers it again. Timeouts and server errors never count
  as a rejection.
- **Latency.** The reading's latency is recorded apart from generation latency, so it does not
  stretch the adaptive deadlines of optional work.
- A typical turn is three calls: the reading, the reply and its review, which is required
  work with its own allowance since sub-phase 1b ([review contract](review-contract.md)).

## What the parser tolerates

Real models are untidy. `parse_reading` accepts:

- the answer's outermost JSON object, with prose or code fences around it, and bracketed
  asides in that prose;
- `kind` in any letter case or with spaces; a missing `kind` is `conversation`;
- an empty `description`, which is then the component's `source_span`;
- out-of-range perception and tone numbers, which are clamped and noted as evidence
  (`reading:clamped`);
- evidence as plain quotations or as `{ref, observation, confidence}` entries; an entry
  without a `ref` gets a generated one, and more than the four asked for are still read;
- `nondefault` as a list of `{name, value}` or as a map, with unknown names ignored;
- `value_effects` as a map or as a list of `{name, effect}`;
- a `tool` entry without a capability name, and an argument without a name: each states
  nothing and is passed over, as a strict server makes a reader write one when it has nothing
  to say;
- a role's `standing` or `whose_say` that is missing or not one of the allowed values: it
  takes its neutral value (`same`, `own`), and the turn records which field did;
- the older, larger shape of a reading: any field listed as removed below is still taken at
  its word when an answer states it.

**A value of the wrong shape in one request field states nothing.** It is dropped and recorded
(`reading:shape:<field>`, with the component's id in front for a component's field), the rest
of the reading stands, and the reading is not asked for again. So one slip cannot cost a turn
what the same answer says about a role or about hostility. This covers:

- `about_self` that is not true or false; `topics` or `lasting` that is not a list, and an
  entry in them that is not of its kind;
- `tool` that is neither a list, an object nor a name; an entry in it that is no object;
  `arguments` that are no list, and an argument that is not a name and a value; `generate`
  that is not a list of names;
- `capability_requirements` that is not a list of names (the older field);
- `role` that is not a list or an object, and a `role_identity` that is not text;
- `reproduce` that is not text; `ends` that is not a list, or an entry in it that names no
  long field;
- `unavailable` that is not a list. A bare string there is dropped too: it may be a filler
  such as "none", and no rule can tell that from a real statement in every language.

Last words given for a long span that do not close one stretch of the message are dropped the
same way (`reading:span_unquoted:<id>:end`), and the span is then its first words alone.

**It rejects an answer only when the answer is unusable as a whole**, or when a field the
decision's numbers rest on cannot be read:

- it is not a JSON object, or its object never closes (point 9);
- it lacks the reading's structure: `perception` is not an object, or `request_components` is
  not a list of one to eight;
- a component is not an object, has neither a description nor a span, has a `description` or
  `source_span` that is not text, or has a `kind` that is not one of the three;
- a number is not finite; `nondefault` or `value_effects` is not a list of names and values;
  `evidence_refs` is not a list of strings;
- `social_move`, `inferred_intent` or `event_type` is not one of its values; `tone` is not an
  object; `mixed_affect` or `agent_goal_comparison`, when stated, is not true or false.

The second group is the contract of sub-phase 1a and is unchanged: those fields carry the
reading's social meaning and its prices, and an answer that garbles one of them is not taken
at half its word.

## The reading's shape

Every field costs output on every reading: the schema is valid for strict structured-output
servers, which means every property is required and no other is allowed
(`reading_schema`). So the shape holds only what something consumes.

- **Numbers that are almost always 0 are not fields.** A reading lists the ones that differ,
  by name, in `nondefault`: `[{"name": "risk", "value": 0.2}]`, usually `[]`. This is the
  one way to make a number optional that a strict schema allows.
- **What only some components say is a list that is empty for the others.** The same device
  as `nondefault`, and no component has to write a placeholder:
  - `role`: one entry for a role, `{"role_identity", "standing", "whose_say"}`. A task or a
    piece of conversation writes `[]` and nothing about standing.
  - `tool`: one entry for each capability a task asks for,
    `{"name", "arguments": [{"name", "value", "end"}], "generate": [names]}`. The component's
    capability names are taken from these entries; there is no second field for them.
  - `ends`: the last words of a long `source_span` or of a long text to give back,
    `{"ends_of", "last_words"}`.
  - `unavailable`: what the work needs that nothing offered does.
- **Long text is never copied whole.** It is given by its first and last words (point 4).
- **New field names are their own.** The vocabulary guard excuses the reading's field names
  wherever they appear in the runtime, so a field is not given a name that is an ordinary
  word (`role_identity`, `ends_of`, `last_words`, not "identity", "of" or "text").
- **`plan` is a field only while a plan is running**, and `self_proposals` only when asked
  for: the schema is built for the turn (`reading_schema(proposals=, plan=)`).
- **Duplicates are merged.** One `expectation_violation`, one `affiliation` and one
  `intensity`, in `perception`; the frame's own fields follow them.
- **Evidence is a list of quotations**, at most four. A role needs none: its `source_span`
  and `role_identity` are its quotation.
- `description` may be empty when `source_span` says it; `reason` is one short clause.

**Size.** The same reading of a two-sentence message (one demand for standing, one task, three
numbers off their default), written compactly:

| | Characters |
|---|---|
| Before sub-phase 1a | 1,841 |
| After sub-phase 1a | 1,238 |
| After sub-phase 1c: two components without a tool request, one topic span | 1,345 |
| After fix round 1 | 1,535 |
| Now (fix round 2) | 1,397 |
| Now, with no number off its default | 1,304 |
| Now, when the task asks for a tool with one written operand | 1,502 |
| Now, when the task asks for a file to be written with generated content | 1,512 |
| Now, for two plain tasks (no role, no tool) | 1,207 |

A component states fifteen keys (nineteen after fix round 1). Round 2 folded six fields into
lists that are empty unless they apply: `role_identity`, `standing` and `whose_say` into `role`;
`capability_requirements` into `tool`; `span_end` and `reproduce_end` into `ends`. On the
reading above that is 138 characters fewer, with everything round 1 added still stated.

**The request around it**, in characters:

| | Before sub-phase 1c | After fix round 1 | Now |
|---|---|---|---|
| System prompt (`READING_SYS_PROMPT`) | 8,867 | 13,595 | 12,903 |
| Strict schema, as sent (`json.dumps(reading_schema())`) | | 4,615 | 4,821 |
| The reading above | 1,238 | 1,535 | 1,397 |

The schema is slightly longer than it was: a list of small objects takes more schema than
flat strings do, though it takes less output. Nothing was dropped from the prompt: the rules of
the older paragraphs (value effects, apologies, tone) are as they were, and the parts added in
1c say each thing once.

### Field audit

**Asked for, and who consumes it**

| Field | Consumed by |
|---|---|
| `language` | The debug and turn records, and the reply's writer, as data (`message_language`, beside the decision): when a component is declined the message itself is withheld from the writer, and the reply still has a language to follow. Sub-phase 1b dropped the local reply-language check and kept the prompt line, which is pending removal; a later phase gives the reviewer this field as a hint |
| `primary_target` | `targets_assistant`: affect (`core/affect.py`), event classification, memory digestion, self-observation; the explanation and persona views |
| `social_move` | Affect; event classification when no `event_type`; digestion (apology); withdrawal of unquoted claims; the explanation |
| `inferred_intent` | Affect (`seek_action`); withdrawal; the explanation and persona views |
| `agent_goal_comparison` | Affect (the comparison signal) |
| `reason` | The debug and turn records |
| `perception.target_probability`, `hostile_intent`, `intensity` | `core/willingness.py::decide` (exposure), `update_relationship` (injury), the expression prompt. `intensity` is also the event's intensity |
| `perception.expectation_violation` | `update_relationship`; affect, through the frame |
| `perception.personal_significance` | `update_relationship` |
| `perception.uncertainty` | `update_relationship` (repair, goodwill), bonding, the expression prompt |
| `perception.affiliation` | `update_relationship` (goodwill), bonding; affect, through the frame |
| `perception.nondefault`: `blame` | Affect |
| … `vulnerability` | The persona and channel debug views, the explanation (display only) |
| … `repair_attempt`, `repair_credibility`, `repair_relevance` | `update_relationship`; `decide` (the forgiveness term) |
| … `reliability`, `reliability_confidence` | `update_relationship` (trust) |
| `perception.evidence` | The quotation check for hostility and repair; the relationship's evidence references |
| Component `description`, `kind`, `source_span` | The decision, the role quotation check, expression, operand grounding |
| Component `role` (a role only): `role_identity` | The role quotation check; the decision and the reviewer |
| … `standing`, `whose_say` | The floor under the role's `autonomy_cost` (point 4); the turn's record of each role |
| Component `ends` | The last words of a long `source_span` (the component's span is then the message's own text between them: where an operand may stand; learning material) or of a long text to give back |
| Component `user_value`, `effort`, `uncertainty`, `autonomy_cost` | `core/willingness.py::_component_terms` |
| Component `nondefault`: `user_cost`, `user_urgency`, `character_value`, `character_cost`, `character_urgency`, `risk` | `_component_terms` |
| Component `value_effects` | `_component_terms` (authored values); `core/self_intent.py` |
| Component `tool` (`name`, `arguments` with `value` and `end`, `generate`) | The component's capability names (feasibility; the capability check of point 4; which granted tasks reach tool selection); the committed operation (`requested_tool`, `requested_arguments`) and its grant; the session's learning executor; the generated-file writer; tool selection for what is left open, and the bound on it |
| Component `reproduce` | The copy checks of the reply, for a task the decision undertook (`core/dignity.py::requested_copy`) |
| Component `unavailable` | Feasibility (the component cannot be granted); what the writer and the reviewer are told is missing; the capability gap behind the character's wish for a skill |
| Component `plan` (only while a plan is running) | Plan continuation (`continuation_of`) and the plan's status for the writer |
| Component `evidence_refs` | `_component_terms` (competing interests and commitments); `core/self_intent.py` |
| `tone.arousal`, `valence`, `intensity` | Contagion; self-observation (`valence`) |
| `event_type` | Event classification |
| `reply_length` | The standing reply-length wish and its memory record |
| `about_self` | Whether identity records are loaded for the turn |
| `topics` | The relationship topic under which memory is looked up and filed |
| `lasting` | The preference and decision records memory keeps for the turn (`core/memory/semantic.py::derive_semantic_entries`) |
| `self_proposals` (only when requested): all seven fields | `core/self_intent.py::conversation_proposals` |

**No longer asked for**

| Field | Why | Value now |
|---|---|---|
| `controllability` | Copied into event metadata; nothing reads it | 0.5 |
| `mixed_affect` | Copied into event metadata; nothing reads it | false |
| Top-level `expectation_violation` | A second copy of the perception's | The perception's |
| `affiliation_bid` | A second copy of `perception.affiliation` | The perception's |
| `event_intensity` | A third intensity | `perception.intensity` |
| Top-level `blame`, `vulnerability` | Rarely anything but 0 | Listed in `perception.nondefault`, else 0 |
| `perception.repair_loop_id` | Stored with the repair record; nothing reads it | empty |
| Evidence `ref`, `confidence`, `source` | The reference is generated, the confidence is read by nothing, the source was always overwritten | generated, 0.5, `model` |
| Component `confidence` | Validated and read by nothing | 0.5 |
| `tone.certainty` | Read only by a classification helper no turn calls | 0.5 |
| The six rare component numbers, the three repair numbers, `reliability`, `reliability_confidence` | Rarely anything but 0 | Listed in `nondefault`, else 0 |

## Still keyword-based

**Nothing on the path from the message to the decision and its execution.** Request and tool
routing and context selection were removed in sub-phase 1c
([request routing](request-routing.md)): host-metric questions, the tool intent rules and their
follow-up rules, "continue" and "status" for a running plan, the step templates, learning
requests, literal file and shell operands, generated-file requests, the identity and topic
patterns, and the stop-word lists of memory and life-history retrieval.
`tests/test_perception_guards.py` traces real turns through the pre-pass, the decision, the
carrying out of what was granted and what the turn records after delivery, and fails if the
user's text reaches any vocabulary-bearing function not named there.

What is named there, and why:

**Memory filters, to be removed in Phase 4**

| Code | What it filters |
|---|---|
| `runtime/evolution_policy.py::detect_directive_override_markers`, `detect_prompt_injection_markers` | Autonomy and instruction phrases, in memory retrieval and storage |
| `runtime/life_history.py::_heuristic_digest` | A keyword digest of an episode, used only when the model's digest of it is unavailable. It belongs to digestion, which Phase 4 rebuilds |

**Text a tool or the runtime produced, not what a person wrote**

| Code | What it reads |
|---|---|
| `core/tool_failures.py::is_operational_tool_failure` | A tool's error message, to tell an environment or configuration failure |
| `core/tool_memory.py::create_tool_unresolved_item` | A tool's error message, to tell a refused execution |
| `core/llm_content.py::strip_reasoning`, two patterns in `core/artifact_writer.py`, `runtime/life_history.py::_extract_json` and `LifeHistoryStore._ingest_text` | Markup, formats and field names of a model's answer |

**After the turn**

- Nothing a turn keeps as memory is picked by wording any more. A preference or a decision is
  kept when the reading quoted it (`lasting`); the patterns for "I prefer …", "we decided …"
  and "let's …" are deleted, and so are the word list behind the curiosity salience bonus and
  the refusal pattern on the character's own reply.
- The life-history store digests an episode with the model. Its keyword fallback, for when no
  model digest is available, is the Phase 4 entry above.
- A skill's trigger (`applies_when`) is the owner's own data and is matched as whole words in
  any script (`runtime/skills.py`).

**Outside the turn**

- `openrook_tools/roleplay_eval/legacy_routing.py` holds the former keyword routing for the
  evaluation judge, the offline replay tool and the test-only keyword reader.
  `core.request_actions.ground_shell_components` forwards to it for a test file that still
  imports that name. No turn calls any of it.

## Offline tools that now see no reading

These tools script a model that answers the reader with nothing, or with a deliberately
neutral reading. Their scenarios were written against keyword perception:

- `openrook_tools/replay_dignity.py`, in both of its modes (it does script a review for each
  of its drafts since sub-phase 1b);
- the mock and compliant backends in `evals/` (`evals/social_backend.py`), used by
  `python -m evals --backend mock`.

Run standalone, they now see plain conversation where they expect a perceived demand, and
some scenarios fail (five in the mock evaluation run after sub-phase 1a). Since sub-phase 1c
they also see no request: a scenario that expects a tool to run, identity records to be loaded
or a capability gap to be noticed from the wording of a message gets none of it, because no
reading reports it. That was not re-measured standalone. In the test suite they run with the
test-only keyword reader, which scripts the former routing as a reading; one scenario cannot
pass even so (`ci_character_independence_skill_want`: nothing reports a capability gap any
more). They need scripted readings before their results mean anything about the current
runtime. `evals/` is not edited in these sub-phases.

## Tests

- Tests run on the runtime's own reader, like production. A fake backend that scripts no
  reading gives an unavailable reading, and the rest of the turn is exercised on that path.
- A test that needs a particular reading scripts it: `tests/support/readings.py` provides
  `ScriptedReader` (injected as `message_reader=`) and `ReadingBackend` (a stub model for the
  runtime reader), with small builders for roles, tasks and hostile perception.
- `tests/test_reading_contract.py` covers the contract.
- `tests/test_reading_truncation.py` covers point 9 and the reading's shape: a full reading cut
  at every 5% of its length and around every closing brace is never accepted; a stub that
  reports finish `length`, and one that returns a cut-off body with finish `stop`; the output
  limits sent by both chat clients; the per-attempt records; the strict schema and its fields.
- `tests/test_perception_guards.py` holds the structural guards, built on
  `tests/support/vocabulary_guard.py`:
  - no vocabulary in the perception modules or the perception stage, however it is bound;
  - every function the user's text reaches is clean or on the list above. The text is followed
    as a string, inside the turn state object, inside a request component's `description`,
    `source_span` and `role_identity`, and as an operand, through the pre-pass, the decision it
    commits, the carrying out of what was granted (a tool turn, an open request that goes to
    model tool selection, a generated file, a plan turn of each kind and a learning turn) and
    what the turn records after delivery (`_stage_post_process`). A planted keyword rule
    behind that last stage is shown to be caught;
  - those functions are checked strictly, and against the text itself: a literal tested
    against what a person wrote (containment, equality, prefix or suffix, a pattern) is a
    finding whatever the literal is, including names from the data model, and so is a lookup
    of the text in a collection or dict of any size, down to a single entry. A table or
    pattern that is imported from another module, kept on the object, or built when its
    module loads is resolved from the value it has at run time;
  - functions the text reaches that hold only the runtime's own labels and record keys are
    pinned literal by literal in `tests/fixtures/perception_reviewed_labels.json`, and may hold
    no pattern and test nothing against the text. A new literal there fails the guard;
  - no runtime import of `tests`;
  - no scenario wording on the perception path.
  Each detector is proved against planted violations.
- Older tests that were written against the keyword perception are marked
  `@pytest.mark.keyword_reader`. For their duration pipelines read with a **test-only** copy of
  the former perception, `tests/support/keyword_reader.py`. Since sub-phase 1c it also states,
  as a reading states it, what the former keyword routing recognised (`keyword_route`): the
  capability, the operands and what is unavailable are reported beside the component and held
  to the message by the runtime like a model's reading, so an operand the message does not
  contain is not committed and no capability is named that the former rules did not
  recognise. Its own tests are in `tests/test_keyword_reader_support.py`. Runtime code never
  imports it.
- `ScriptedReader` parses a scripted reading as strictly as the turn's reader parses a model's
  answer: a perception object and at least one component.
- A test that asks for a tool scripts the request: `tests/support/readings.py` has `using`
  (a component's `tool`), `operations`, `file_request`, `committed`, `read_as` and
  `WithReadings`. `tests/test_request_routing.py` covers the request side of the contract.
- `tests/test_role_facts.py` covers a role's two closed facts and the cost floor, with scripted
  readings only: the same sentence is read in every case.

## Not in this sub-phase

- **Reply checks** were reworked in sub-phase 1b: see
  [the review contract](review-contract.md). `core/social.py` and
  `core/perception.py::observe` are deleted; the cues the evaluation judge and the test-only
  keyword reader still use live in `openrook_tools/roleplay_eval/legacy_social.py`.
- **Routing and context selection** were reworked in sub-phase 1c: see
  [request routing](request-routing.md).

# Requests and tools by understanding

Sub-phase 1c of [the goal fix plan](goal-fix-plan.md), Phase 1 ("Perception by understanding").
It covers what happens between the reading and the reply: how the runtime knows what is being
asked and which tool runs. The reading itself is [the reading contract](reading-contract.md);
checking the reply is [the review contract](review-contract.md).

## What changed

After sub-phases 1a and 1b, English patterns still interpreted the user's text to decide what
was being asked and which tool ran: host-metric questions, a table of tool intent rules,
"continue" and "status" for a running plan, step templates, learning requests, literal file and
shell operands, requests to write generated text to a file, and which context to load. Two of
them replaced the reading's components wholesale, so a role demanded in the same message was
dropped.

Now **the reading says what is asked, which capability does it and with which operands.** No
turn matches the user's text against wording. All of that code is deleted from the runtime.

## The contract

1. **The reading grounds requests.** A task states each capability it asks for as one entry
   in the component's `tool`:
   `{"name", "arguments": [{"name", "value", "end"}], "generate": [names]}`.
   - `arguments` are operands **the person wrote**: a path, a command, an address, a query,
     text to store.
   - `generate` names operands that have to be produced, such as text still to be composed. The
     reader gives no value for them, and **a value it gives anyway is discarded where the
     answer is parsed** (`core/appraisal.py::_tool_requests`).
   - The capability names of a task are taken from its tool entries. The reading no longer
     states them a second time in `capability_requirements` (the parser still reads that
     field when an answer has it). A task that needs several capabilities has several
     entries, each with the operands the person wrote for it.
   - **A question is a task when answering it needs a capability.** The prompt says so in
     general terms: a question whose answer needs what an offered capability does, or
     something outside the reply that none of them does, is a task, not conversation.
   - An entry without a name (a tool entry, or an argument) states nothing and is passed over.
     A strict server makes a reader write such placeholders; they do not sink the reading.

2. **A component stands somewhere** (`core/request_routing.py::_placed`). Its `source_span` is
   a stretch of this message, or of an earlier message of the same person that it refers back
   to. The span covers everything that belongs to the part, including the values it gives.

3. **A literal operand is the person's own characters, as a whole written unit**
   (`core/perception.py::written`, `exact_part`; `core/request_routing.py::_operand`).
   - **Where.** The value must stand inside the stretch its component stands in, or inside a
     recent message **from the same person** (the last six turns) when the component refers
     back to one. A value from another part of the same message is not this part's operand,
     and an operand copied from the character's own reply is not the person's.
   - **Exact** means character for character. The one difference allowed is canonical Unicode
     composition (NFC), which changes how a character is encoded and never which character it
     is. Case, spacing, quotation marks, punctuation, line breaks and compatibility forms are
     **not** folded: a command or a path altered in any of those is a different one.
   - **Whole.** The value is not a piece of something longer. On each side of it, up to the
     next white space or the edge of the text, there may be only punctuation that stands
     *around* a written unit (`_closes_or_quotes`, `_ends_unit`):
     - quotation marks and brackets, on either side (Unicode Ps, Pe, Pi, Pf and the plain
       quote characters);
     - after it, a mark that ends a sentence or a clause (the lists `perception.py` already
       holds: full stop, question and exclamation marks, comma, semicolon and their
       counterparts in other scripts), and a colon that introduces what follows: one that
       white space, the end of the text or a script written without spaces comes after
       ("save this to notes.txt: …"). A colon that joins two things (`notes.txt:12`, `C:/x`)
       is part of the unit;
     - invisible format characters.

     **Every other character belongs to the unit**: a letter, a digit, a mark, and any other
     punctuation or symbol, such as a slash, a tilde, an asterisk, a hyphen, an at sign or an
     equals sign. Before the value even a full stop belongs to it: `.bashrc` is not `bashrc`,
     and `./x` is not `/x`. So none of these is an operand: `build/output.log` in
     "delete build/output.log.bak"; `https://example.com` in `https://example.com.evil.io/x`;
     `ls` in "also"; `projects/tmp` in `~/projects/tmp`; `/etc/passwd` in `/etc/passwd~`;
     `build` in `build/*`; `8.8.8.8` in `-8.8.8.8`; `example.com/a` in `@example.com/a`.
   - **Scripts written without spaces** are taken as they write: a character of such a script
     next to the value ends the value, as white space does, so a Latin path directly beside
     Han, Kana or Thai text is whole. The reverse is stricter than word matching: a value in
     such a script followed directly by a Latin letter or a digit is a piece of something
     longer.
   - **Characters are not cut.** A value does not begin on a combining mark, does not stop
     before a mark that completes its last character, and does not begin or end at a joiner.
   - What is committed is the slice of the message itself, in the code points the person typed.
   - A value that fails any of this is a contract problem: it spends the reading's one retry,
     with the argument named, and is removed if it is still there.
   - **Known limits.** Character rules cannot tell a command that was cut at a word boundary
     from a whole one: "sudo apt remove foo" out of "sudo apt remove foo --simulate" is a run
     of whole words with white space after it. The same holds for the first of two commands
     joined by a semicolon. Where a script writes no spaces, an ASCII-punctuated suffix in
     that script (`output.log.バックアップ`) is not told apart from a sentence that goes on.
     What the reading says the person asked for is the only check on these.

4. **Long text is given by reference, not copied** (`core/perception.py::passage`).
   - A value longer than about 200 characters is given by its first few words in `value` and
     its last few words in `end`. The same holds for a long `source_span` and a long text to
     give back: their last words are entries of the component's `ends` list,
     `{"ends_of": "source_span" | "reproduce", "last_words": ...}`, which is empty otherwise.
   - Last words that do not close one stretch (a placeholder, for instance) are dropped and
     recorded, without a second reading; the span is then its first words alone.
   - Each of the two pieces must be the person's own characters. The stretch begins where a
     written unit begins and ends where one ends, the first piece stands before the last, and
     **each stands only once** in the place the value may come from. A piece that stands twice
     does not say which stretch is meant, and the value is not taken.
   - What is committed is **the message's own text** between them. The reader never copies
     the bulk: a request to save 3,000 pasted characters costs the reading a few hundred.
   - Learning material is not an operand at all: it is the message less the asking span
     (point 11).

5. **A committed operation, or a task that names a capability** (`_grounded`, `ground`).
   - When the named capability has **every required operand as the person's own text**, and
     nothing was removed or left to be produced, the component carries the operation:
     `requested_tool` and `requested_arguments`. They are part of the immutable grant; the
     decision grants or declines exactly those operands.
   - **A capability that needs no operand** is committed by naming it. A task that lists
     exactly one capability in `capability_requirements`, with no `tool` entry, is that
     operation when the capability declares that it has no required argument (for example the
     CPU usage one). It runs with any backend, tool runner or not.
   - Anything less leaves the component a task that names the capability
     (`capability_requirements`), with the operands that were verified and no `requested_tool`.
     Its remaining operands are produced when the granted task is carried out (point 10).
   - One committed operation is one tool. A capability the same component named beside it is
     no part of the operation and is recorded as ignored.
   - **What an uncommitted task keeps** travels with the component, and so with the frozen
     decision that grants it (`held_operands`, `unverified_operands`):
     - each operand the reading verified, **for the capability it was given for**. A task that
       names several capabilities keeps them per capability (`tool:argument` in
       `requested_arguments`; the plain argument name when it names one capability once);
     - the name of each literal argument whose reported value was **not** the person's text
       (`operand:unverified:tool:argument` in `evidence_refs`);
     - `operand:unplaced` when the part that asks stands nowhere in what the person wrote.

     The grant check holds a selected call to all three (point 10).
   - **A part that stands nowhere takes no operand.** When a task's `source_span` is no part
     of this message or of an earlier message of the person, nothing ties the request to
     their text: no operand is taken for it, from anywhere, and the unquoted span is recorded
     (`reading:span_unquoted:<id>`).
   - **Which arguments are offered** (`offer_of`). Every required argument is offered,
     whatever its type, because the capability cannot run without it. One whose value is not
     text (a number, a flag, a list) is listed in the offer's `produced_arguments`: it is
     never a literal operand and is always produced when the task is carried out. An optional
     argument is offered only when its value is text. A capability registered with an empty
     schema takes no argument; a bare name with no schema at all says nothing about its
     arguments, and what it is given is taken as text.

6. **Work nothing offered can do** (`unavailable`).
   - A task whose work needs something outside the reply that none of the offered capabilities
     does names no capability and says what is missing in `unavailable`, in a few words. That
     includes a command the person wrote when no capability runs commands, and material no
     capability handles.
   - `unavailable` is **a list that is empty unless it applies**, as `tool` is, so a strict
     schema never makes a reader write a filler. A bare string there is not that list: it is
     dropped and recorded (`reading:shape:<id>.unavailable`), because no rule can tell "none"
     or "N/A" from a real statement in every language.
   - Such a component is **not feasible** (`feasible = False`): the decision cannot grant it,
     however much the person wants it. Nothing is executed or selected for it.
   - The writer and the reviewer are told the same thing as data: the component carries
     `"possible": false` and `"unavailable": "<what is missing>"`. The reply can decline
     honestly instead of inventing output.
   - It is kept as a **capability gap** (`_note_unavailable_work`). How much the gap matters is
     how much the person wanted the work (`user_value`). Repeated gaps are what the
     character's wish for a new skill starts from (`core/proactive.py`).
   - **Two reports are the same gap only when they say the same words in the same order**
     (`same_gap`), however they are cased, spaced or punctuated. Sharing words is not enough:
     "no PDF reader" and "no audio reader", "send email" and "send SMS" are different gaps,
     and "speech to text" is not "text to speech". A need that comes back in other words is
     recognised by understanding instead: the reader is shown the gaps already known
     (`known_gaps` in its context, the last eight) and is asked to state a need that is one of
     them in the same words (`core/appraisal.py::GAP_PROTOCOL`, appended only when there are
     known gaps).
   - `unavailable` on a part that is not a task, or on a task that names an offered
     capability, is removed and recorded.

7. **The reader is given what it needs, as data** (`reader_context`, `core/request_routing.py`):
   - `available_capabilities`: each offered capability as `{name, purpose, arguments,
     optional_arguments, produced_arguments}`. Every registered tool is offered, and learning
     intake when the session wired its executor.
   - `known_gaps`, when there are any: what was already found to be missing (point 6).
   - `running_plan`, when a plan is running: `{goal, steps_done, steps_total, next_step}`.
   - `recent_context`: the last six turns, as before.

8. **Nothing replaces the reading's components**
   (`pipeline/core.py::CognitivePipeline._add_execution_detail`). Routing only adds execution
   detail to a component the reading reported:
   - on the task the reading tied to the running plan as its continuation: the plan's id
     (`continuation_of`) and the capabilities the plan's remaining steps use;
   - the mark the session's learning executor requires, on a task that asks for learning intake.

   No component is added, removed or re-labelled, and none is re-priced. A role or a piece of
   conversation in the same message reaches the decision exactly as it was read. A message
   nobody read is plain conversation, whatever its words: it commits no operation.

9. **What a granted task does** (`_stage_deliberation`, `_run_tool_stage`):

   | The granted task | What runs | Model calls |
   |---|---|---|
   | names no capability | Nothing. It is work done in the reply: no selection, no receipt | none |
   | carries a committed operation | That operation, through the arbiter and the executor | none |
   | is the running plan's continuation | The plan's next steps (`continue_plan`), each gated against its own risk category | none |
   | asks for learning intake | The session's learning executor, inside the grant of the request (point 11) | the digest, as before |
   | names the file-writing capability with the path written and the content to produce | The content is drafted and written to **the committed path**; a path the drafting model names is ignored | one draft (`artifact_draft`) |
   | names a capability otherwise | Model tool selection (point 10) | one protected selection, more from the base budget |

   The draft of a file is a model call like any other: it is counted, recorded in the turn's
   call telemetry with `phase: "artifact_draft"`, and held to the request deadline in force.
   It takes the granted task's one protected call, the one the same task would otherwise
   spend on tool selection, so the per-turn maximum in
   [the review contract](review-contract.md) holds with it.

10. **Model tool selection is bounded by the grant** (`_select_tools`, `core/grants.py`).
    - It is asked only for a granted task that names a capability without a committed
      operation. **A task that names no capability never reaches it**, with any backend.
    - **Only the capabilities the task names are offered** to the selecting model, whatever
      else the operator allows.
    - **The grant check holds whatever the model answers** (`execution_denial`). For a granted
      task of a person's request without a committed operation:
      - a call to a tool outside its `capability_requirements` is denied;
      - a call that gives an operand the reading verified **for that tool** another value, or
        leaves it out, is denied;
      - a value given for a literal argument whose reported value was not the person's text
        must itself be the person's text: a whole written unit of the component's own stretch
        of the message, or of an earlier message of the same person. Anything else is denied.
        So an operand the reading got wrong does not become the model's free choice;
      - a task whose asking part stands nowhere has no such text, so no value passes.

      A task that names no capability covers no tool at all. This check is in the executor,
      so it holds for every runner. The turn's message and the person's earlier messages are
      handed to the check with the execution scope; they are where a value may stand, not an
      authority, and the test is the one an operand passes when the reading is held to the
      message (`core/request_routing.py::written_for`).
    - **Inside a turn, an execution without a scope is denied** (`core/grants.py::turn_stages`).
      Every execution in the stages of a person's turn belongs to a granted component and runs
      inside that component's scope, so a call that reaches the executor there without one has
      lost it. Outside a turn, a call with no decision in scope is still allowed: it is an
      operator's, a study's or a test's own call to the executor, not a conversational grant,
      and the operator's gates in the tool runtime apply to it as before.
    - **The model is given the task as data** (`core/request_routing.py::Selection`), as one
      JSON object: `request` (the task's description and span), `written_operands` (the
      operands the reading verified), `recent_turns` (the last six turns, by speaker) and
      `receipts` (what this turn has already executed, with what it returned). The last two
      let it resolve a request that refers to an earlier turn or to an earlier step. Nothing
      in it is matched against anything.
    - When the model selects no tool, none runs. The runners have no other way to choose one.
    - **A failed tool-calling request** is classified by the runner
      (`openrook_tools/native_orchestrator.py::selection_failure`): `transport` when it timed
      out or the connection failed; `unsupported` when the server has no such endpoint, or
      rejected the request and named `tools` or `tool_choice` in its own error. The failure
      receipt carries it as `tool_selection_failure`.
    - After such a failure selection is **not asked again in that turn**, and is **left alone
      for the next ten turns** (`TOOL_SELECTION_PAUSE_TURNS`), then offered again, as structured
      output is in `runtime/llm/backend.py`. Any other failure pauses nothing.
    - The reply is told the outcome as execution evidence: the failed receipt, and for each
      task not attempted a line saying that nothing was executed and why. The turn's record is
      `debug.tool_selection` (`requests`, `status`, `failure`, `skipped`, `paused_turns_left`),
      also stored with the turn.
    - Research does not pause: its call contract and budgets are unchanged.
    - The character's own activity (a decision of scope `self`) is not a person's request. It
      has its own decision, and its tools are chosen and gated where that activity runs; the
      capability bound above applies to a person's request only.

11. **Learning** is a task that names `learning.intake`, with the address the person wrote as
    `url`, or with no operand when the material is the message itself.
    - Naming the capability is enough when there is no address: it needs no operand, so the
      task is committed (point 5), with or without a `tool` entry.
    - The pipeline carries the granted task out inside
      `runtime.learning_intake.granted_learning(request)`. Within that scope, and nowhere else,
      `detect_learning_request` returns that request; it examines no wording.
    - Without an address the material is the message, less the part that asks (`source_span`)
      when that part opens or closes the message. The reader copies none of it.
    - A source that was not written by the person is not fetched. The task then has no
      committed operation, and its receipt says that nothing was attempted and why
      (`execution_started: false`).

12. **A receipt that never ran is not a failure.** For a component that asked for an operation,
    the reviewer is told its outcome: `declined`, `not attempted`, `succeeded` or `failed`. A
    receipt that was stopped before execution (`execution_started: false`, a grant denial, the
    operator's allowlist) counts as `not attempted`
    (`core/review.py::decision_projection`).

13. **A running plan** is tied to the message by the reading: while a plan is running each
    component states `plan`: `continue`, `status` or `none`. The relation stands when the
    component's `source_span` is a verbatim part of the message; `continue` belongs to a task.
    A granted continuation carries the plan on and covers the plan's own tools; a granted
    status question gives the writer how the plan stands. Neither replaces another component.

    **Plans are not created in a user turn.** Multi-step work across turns returns in Phase 5
    of the fix plan, as the character's own intention. Until then a multi-step request runs as
    separately granted components within the turn, each with the receipts of the ones before
    it, and `continue` / `status` apply only to a plan restored from a saved session.

    **What is checked when a restored plan is carried on**, and what is not:
    - The plan must have been started by a committed decision (`origin_decision_id`), the rule
      the character's own continuation already follows (`_running_plan`). A plan without one
      was never granted by anyone: the reader is not told of it, so no part of a message can
      continue it or ask how it stands.
    - The continuation is a granted task of this turn's decision, and its grant covers the
      tools of the plan's remaining steps and no other.
    - Each step is gated by the arbiter against its own risk category and the operator's
      autonomy level.
    - **Not checked:** the steps' arguments against this turn's message. They were fixed when
      the plan was made, by the decision that started it, and for plans saved before
      sub-phase 1c that was the old keyword templates. The plan's goal and next step are
      shown to the reader and, through the decision, to the writer and the reviewer.

14. **Context selection comes from the reading.**
    - `about_self`: the message asks about the character's own past, its memory of itself, or
      how it has developed. Identity records are loaded when it is true.
    - `topics`: up to three spans of the message that name what it is about. Spans that are
      not verbatim parts of the message are dropped. The first one, folded, is the topic under
      which relationship memory is looked up and filed (`pipeline/heuristics.py::relationship_topic`).
    - Memory and life-history retrieval compare words as their script delimits them
      (`core/perception.py::retrieval_words`) and leave out a word that more than half of at
      least four stored texts contain (`telling`). When one person has fewer than four
      memories, how widely a word is used is taken from the whole store instead: only its
      spread, never what another person's memory says. With fewer than four texts in the whole
      store nothing can be said about any word, and every word counts. The English stop-word
      lists are gone.

15. **A requested reproduction** (`core/request_routing.py::_reproduction`). A component's
    `reproduce` is text the person asks to be given back word for word.
    - It stands **for a task only**.
    - It is the person's own text: from an earlier turn, or a part of this message that
      leaves the asking out. Inside its component's `source_span` it must be strictly shorter
      than that span. A reproduction that takes in the whole span, or the whole message, does
      not stand: it would excuse a reply that merely repeats what it was told.
    - Anything else is removed and recorded, **without spending the retry**.
    - The structural copy checks exempt a copy that lies within such a span, in that turn, and
      only for a component the decision undertook (`core/dignity.py::requested_copy`). A copy
      that runs on beyond the span is still an echo, and a repeat of the character's own
      earlier reply is still a repeat.

16. **What a turn keeps as memory comes from the reading** (`core/memory/semantic.py`).
    - `lasting` lists what the message states that is meant to last, each quoted verbatim: a
      `preference` of the person, or a `decision`. A span that is not part of the message is
      dropped. Memory keeps one record of each kind for a turn.
    - The English patterns that used to pick these out ("I prefer", "we decided", "let's") are
      deleted. A sentence nobody read as stating a preference stores none, in any language.
    - The salience bonus for a curious character applies when the turn took in learning
      material (a learning receipt), not when the text contains certain words.
    - A refusal the character expressed is recorded from the decision (something was declined
      and the reply delivered), not from words in the reply.
    - A skill's trigger (`applies_when`) is the owner's own data and stays. It is matched as
      whole words in any script (`runtime/skills.py::_trigger_matches`): each word of a spaced
      script, a tool name such as `fs.read_file`, and each unbroken run of an unspaced script.

## Removals

Each removal is recorded as evidence on the perception, like the reading's other corrections.
The first group is a break of the contract: it spends the reading's one retry, and is removed
if it is still there. The second group is a slip, something stated where it has no meaning: it
is removed and recorded, and **the reading is not asked for again**.

| Evidence | Removed | Retry |
|---|---|---|
| `reading:argument_unquoted:<id>:<name>` | An operand that is not a whole written unit of what the person wrote, inside its part | yes |
| `reading:argument_unknown:<id>:<name>` | An argument the capability does not take | yes |
| `reading:capability_unknown:<id>:<name>` | A tool the reader was not offered | yes |
| `reading:plan_unquoted:<id>` | A plan relation on a component whose span quotes nothing | yes |
| `reading:span_unquoted:<id>` | Nothing is removed: a task asks for a capability from a part that stands nowhere in what the person wrote, so it takes no operand | no |
| `reading:span_unquoted:<id>:end` | The last words given for a long span, when they do not close one stretch | no |
| `reading:shape:<field>` | A value of the wrong shape in one request field (see [the reading contract](reading-contract.md#what-the-parser-tolerates)) | no |
| `reading:tool_not_task:<id>` | A tool reported for a component that is not a task | no |
| `reading:plan_absent:<id>` | A plan relation with no plan running | no |
| `reading:plan_not_task:<id>` | `continue` on a component that is not a task | no |
| `reading:reproduction_not_task:<id>` | A reproduction on a role or on conversation | no |
| `reading:reproduction_unquoted:<id>` | A reproduction that quotes nothing | no |
| `reading:reproduction_whole:<id>` | A reproduction that takes in the whole of what asks for it | no |
| `reading:unavailable_not_task:<id>` | `unavailable` on a component that is not a task | no |
| `reading:unavailable_with_capability:<id>` | `unavailable` on a task that names an offered capability | no |
| `reading:capability_ignored:<id>:<name>` | A capability named beside one complete tool request | no |

`capability:unavailable:<id>` is not a removal: it carries what the reading said is missing for
a task nothing offered can do.

Every attempt of the reading that was not used says why in the turn's record
(`debug.reading.attempts`, also stored with the turn): `reason` for an answer that could not be
read at all, and `rejected` for one that broke the contract, as a list of
`{rule, component, field}`. The rules are the codes above without the `reading:` prefix, plus
`role_unquoted`, `hostility_unquoted` and `repair_unquoted`.

## Deleted

| Was | What it recognised |
|---|---|
| `core/system_inspection.py::inspection_kind`, `inspection_intent` | Host-metric questions and their follow-ups |
| `core/dual_process/tool_loop.py`: `INTENT_RULES`, `match_intent_rule`, `detect_tool_intent`, the follow-up rewrite rules, `detect_capability_gap` | Tool requests, terse follow-ups, missing capabilities |
| `core/task_planning.py`: `is_continue_request`, `is_status_request`, `detect_multi_step_intent`, the step templates | Plan follow-ups; multi-step requests |
| `core/request_actions.py`: `has_learning_intent`, `file_requests`, `ground_file_components`, `ground_shell_components` | Learning requests; literal file and shell operands |
| `core/artifact_writer.py::detect_artifact_write_request` and its path, language and subject inference | Requests to write generated text to a file |
| `pipeline/heuristics.py`: `_is_identity_question`, `_relationship_topic_hint`, `_is_broad_repair_text` | Context selection |
| `pipeline/core.py::_identity_question_response` | A canned identity answer nothing called |
| The heuristic fallback of both tool runners | A tool chosen from wording when the model chose none |
| `runtime/learning_intake.py`: `_extract_urls`, `_extract_inline_text` | Where learning material starts |
| The stop-word lists of `core/memory/semantic.py` and `runtime/life_history.py` | English function words in retrieval |
| `core/memory/semantic.py`: `_PREFERENCE_PATTERNS`, `_DECISION_PATTERNS` | What a turn keeps as a preference or a decision |
| The word list of `core/life_influence.py::curiosity_salience_bonus` | "Learning-related" turns |
| The refusal pattern of `pipeline/core.py::_record_self_observations` | A refusal the character expressed |
| The `component_observer` parameter of `CognitivePipeline` | A hook that could replace the reading's components |

The former patterns live on, verbatim, in `openrook_tools/roleplay_eval/legacy_routing.py`, for
the evaluation judge, the offline replay tool and the test-only keyword reader. No turn calls
them; `tests/test_perception_guards.py` fails if one does.

### Dead code still in `runtime/sessions/manager.py`

That file is being changed elsewhere and was not edited. Nothing calls the following any more;
remove it when the file is free:

| What | Why it is dead |
|---|---|
| `_maybe_run_learning_intake` | The pre-decision learning route; no caller is left |
| `_observe_learning_component` and the line that assigns it as `pipeline._component_observer` | The pipeline takes no observer and reads no such attribute |
| `_looks_like_pending_learning_text` | Used only by the two functions above |
| The `detect_learning_request` import | Used only by the functions above; outside a grant it returns nothing |
| The `learning:pending` branch of `_execute_learning_component` (`pending_inline_text`) | No component carries that mark: the pipeline adds `learning:explicit` only |
| `session.pending_learning_intake`, set when a request arrives without material | Nothing reads it: the reader resolves the follow-up from the recent turns |

## What changed in behaviour, stated plainly

- **Multi-step requests are the reading's ordered task components.** Each is granted on its
  own and carried out in order in the same turn. The plan state machine is kept (`TaskPlan`,
  `execute_plan`, `continue_plan`, `summarize_plan_status`), but **nothing in a user turn
  creates a plan**: the only creator was the regex step templates. See point 13.
- **Capability gaps come from the reading** (point 6). They used to be recognised by English
  patterns for a handful of subjects.
- **A file to generate needs its path written.** The runtime no longer infers a file name from
  the request. Without a written path the task goes to model tool selection; without a tool
  runner nothing is written.
- **A request without a tool runner.** A task that names a capability which needs operands,
  without a committed operation, runs nothing when the backend has no tool selection (for
  example the Codex CLI backend). A capability that needs no operand runs directly (point 5).
- **A literal command with no shell capability** is work nothing offered can do: the reading
  reports it as `unavailable`, the component is not feasible, and the reply declines it
  honestly.
- **Learning** takes one address per task, and needs a reading that reports the request. The
  session's pending-intake state is still set when a request arrives without material, but
  nothing reads it: the reader resolves the follow-up from the recent turns.

## Guards

`tests/test_perception_guards.py` traces `_stage_pre_pass`, `_stage_deliberation` and
`_stage_post_process` through a tool turn, an open request that goes to model tool selection, a
generated file, a plan turn of each kind and a learning turn. A planted self-test puts a keyword
rule behind `_stage_post_process` and shows that the trace finds it tested against the person's
text. `process_autonomous` and `process_proactive` take no message from a person and are not
traced. The pending list holds no code that reads a message, decides, routes a request or keeps
a turn's memory. What remains, each named with its reason:

| Where | What | Why it stays |
|---|---|---|
| `runtime/evolution_policy.py` (two functions) | Autonomy and instruction phrases, in memory retrieval and storage | Phase 4 |
| `runtime/life_history.py::_heuristic_digest` | A keyword digest of an episode, used only when the model's digest of it is unavailable | Phase 4 (digestion) |
| `runtime/life_history.py::_extract_json`, `LifeHistoryStore._ingest_text` | A code fence and a field name in the model's digest of an episode | Not what a person wrote |
| `core/tool_failures.py::is_operational_tool_failure`, `core/tool_memory.py::create_tool_unresolved_item` | The error text a tool or the runtime produced | Not what a person wrote |
| `core/artifact_writer.py` (two patterns) | A file draft the model returned as invalid JSON | The runtime's own answer format |
| `core.request_actions.ground_shell_components`, `core.system_inspection.addresses_inspection` | Compatibility names for files being changed elsewhere | Forwarders; no turn asks for them |
| `openrook_tools/roleplay_eval/legacy_routing.py`, `legacy_claims.py`, `legacy_social.py`, `openrook_tools/replay_dignity.py` | The evaluation harness and the offline replay | Outside the turn |

**Not traced:** what a granted operation does once it has been handed to its executor. A tool's
handler examines the form of its operand (a path against the workspace, a command against the
allowlist), and the life-history store digests granted learning material. That is execution of
granted work, not routing.

Tests: `tests/test_request_routing.py`, `tests/test_role_facts.py`.

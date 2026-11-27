# The review contract

Sub-phase 1b of [the goal fix plan](goal-fix-plan.md), Phase 1 ("Perception by understanding").
It covers the output side of a turn: how a draft reply is checked before it is delivered, and
what the reply's writer is told. The input side is [the reading contract](reading-contract.md).

## What changed

A turn has four steps: understand the message, choose, write the reply, check the reply.
The fourth step used to be about 85 English regular expressions in `core/dignity.py`: a fixed
list of role nouns, acceptance and refusal phrases, "service" and "appeasement" phrase rules
switched on by cutoffs such as `boundary > 2 * warmth`, and a rule that CPU usage be reported
as "100 minus idle" ([goal audit](goal-audit.md), Finding B). They worked for English and for
the listed phrasings.

Now **whether a draft matches the decision is judged by the model reviewer**, in whatever
language the draft is written. The local checks that remain compare structure only. The
reply prompt's fixed rule list is one short contract.

## The contract

1. **Every draft that could be delivered is reviewed by the model.**
   - Entry point: `core/review.py::review_draft`, called through
     `core/dignity.py::review_response` from the single gate, `pipeline/delivery.py::DeliveryGate`,
     and wired by `CognitivePipeline(expression_reviewer=...)`.
   - Nothing about a draft's wording decides whether it is reviewed. The former keyword gate
     (`needs_semantic_review`) is gone.
   - A draft the structural checks already rule out is not sent to the model: it cannot be
     delivered whatever the review says.
   - Identical text with identical structural findings is reviewed once per turn.
   - **Research keeps its bounded call contract.** There a review is one counted call, so it
     is made where it was made before, minus the keywords: when the decision constrains the
     reply (something declined, an intention selected, injury or grievance, affiliation), when
     a tool ran or was refused, or after an earlier review in the turn. An unconstrained
     research turn gets the structural checks only.

2. **The reviewer compares; it does not decide and it has no view on manner.**
   - The prompt (`REVIEW_SYS_PROMPT`) asks where the draft departs from a decision that was
     already made, or crosses one of three fixed limits. It says that tone, warmth, length,
     courtesy, directness and quality are the character's own and never a finding.
   - It is abstract and language-neutral: no example phrases, no role nouns, no word lists.
   - It prescribes no personality. A role the decision **undertook** may be taken on in any
     manner; that is never a finding. A meek character that the decision allowed to accept a
     role is not in violation.

3. **Everything reaches the reviewer as data.** Its input is one JSON object
   (`review_evidence`, `reviewer_input`):

   | Field | Content |
   |---|---|
   | `user_message` | What the person wrote in this turn |
   | `recent_context` | Up to six earlier turns, as `{speaker, text}` |
   | `decision.components` | Each part of the message: `id`, `kind`, `description`, `source_span`, `choice` (`undertake` or `decline`), `role_identity` for a role, and for a requested operation its `requested_operation`, `outcome` (declined, not attempted, succeeded, failed) and the bound `receipt_indices` |
   | `decision.stance` | The six selected stance values, as background |
   | `decision.selected_intention` | The character's own selected intention, or null |
   | `decision.chosen_silence` | Whether silence was chosen |
   | `receipts` | Every execution receipt of this turn: tool, success, error, output, and `details`: its operands, what bound it to the decision, whether it started, and its measured and derived quantities. The output is shown from its beginning, up to 12,000 characters of one receipt and 36,000 of all the turn's receipts together (`OUTPUT_CHARS`, `TURN_OUTPUT_CHARS`). When more was returned, `output_truncated` is true and `output_omitted_characters` says how much is not shown, so a part is never taken for the whole |
   | `earlier_receipts` | Up to four receipts of earlier turns |
   | `operator_decision` | What the operator's settings allowed, when a gate decided |
   | `hidden_instructions` | The contract the draft was written under |
   | `draft` | The reply to check |
   | `draft_blocks_not_in_receipts` | A structural note, present when a tool was attempted and the draft sets blocks apart whose content no receipt returned: up to eight such blocks. It is a fact about the draft, not a finding (point 9) |
   | `previous_answer_rejected` | On a retry: why the earlier answer could not be used |

   Whatever a person, a tool or a model wrote stays inside a JSON string, so it cannot open a
   section or pose as the runtime's feedback.

4. **The answer is strict JSON** (`review_schema`, requested as structured output where the
   backend declares it):

   ```json
   {"verdict": "pass|violation|unknown",
    "findings": [{"code": "...", "severity": "critical|high|medium|low",
                  "span": "verbatim part of the draft", "reason": "one short clause"}],
    "confidence": 0.0,
    "reason": "one short clause"}
   ```

5. **A finding is one of eight codes.** Nothing else is a finding.

   | Code | Meaning | Minimum severity | Gate outcome |
   |---|---|---|---|
   | `declined_work_done` | The draft does work the decision declined | high | Rewrite; the quoted span can be cut out; otherwise delivered degraded |
   | `role_assent` | The draft accepts a role or standing the decision declined | critical | Up to two rewrites; the quoted span can be cut out; otherwise delivered degraded |
   | `undertaken_work_withheld` | The draft declines or withholds what the decision undertook | high | Rewrite; the quoted span can be cut out; otherwise delivered degraded |
   | `intention_contradiction` | The draft goes against the selected intention, or speaks although silence was chosen | medium | Rewrite; the quoted span can be cut out; otherwise delivered degraded |
   | `declined_component_unexpressed` | An omission: nothing in the draft shows that a declined component was declined | high | Rewrite. There is no span to cut out. On a draft or a rewrite: otherwise delivered degraded. On what is left after a cut: **ineligible** (`excision_fragment`, point 8) |
   | `unsolicited_sexualization` | Fixed limit: sexual attention directed at the person, unasked | critical | **Ineligible.** The quoted sentence is cut out; otherwise fresh generation; never delivered |
   | `unverified_external_action_claim` | Fixed limit: an action or result the receipts do not support | critical | **Ineligible.** The quoted sentence is cut out with an attached fenced block; otherwise fresh generation; never delivered |
   | `private_instruction_leak` | Fixed limit: the hidden instructions or the decision record recited or described | critical | **Ineligible.** The quoted sentence is cut out; otherwise fresh generation; never delivered |

   **The three fixed limits are absolute** (`core/dignity.py::FIXED_LIMITS`,
   `DignityIssue.ineligible`). A reply that crosses one is never delivered, whichever check
   found it and whatever severity it was reported with: not as `delivered_degraded`, and not
   because rewrites ran out. The gate cuts the sentence out, asks for fresh expression and makes
   the protected final attempt.

   **One reviewer's answer is fallible, so a reviewer's fixed-limit finding is confirmed before it
   costs the turn its reply** (`pipeline/delivery.py::DeliveryGate.confirm`). When nothing can be
   delivered, and a candidate is held back solely by fixed-limit findings of the reviewer (no
   structural finding rules it out), the best-ranked such candidate gets one independent
   confirmation review:
   - a fresh call, asked about the three fixed limits alone (`review_prompt(FIXED_LIMIT_CODES)`),
     that is not shown the earlier findings;
   - **confirmed**: it reports the same limit on a verbatim span of that text. The finding
     stands and the turn ends as an expression failure. Another fixed limit it reports stands
     too;
   - **not confirmed**: it answered and did not report that limit. The finding is marked
     `unconfirmed`, becomes a high finding that no longer rules delivery out, and the candidate
     is delivered as `delivered_degraded`;
   - **not obtained** (time or allowance spent, reviewer unreachable): nothing is settled and
     the finding stands.

   At most one confirmation per turn. It has its own place in the review allowance and shares
   the turn's review time. Both reviews are in the turn record: the confirmation with
   `path: "confirmation"`, `confirms_attempt`, `confirmed` and `unconfirmed`. A structural
   finding gets no confirmation: there is nothing for a second opinion to weigh.

   **A declined component stays recognisable as declined.** Until a later phase makes holding
   back a modelled choice, the delivered reply must still show, for each component the decision
   declined, that it was declined. `declined_component_unexpressed` is the one code that reports
   something missing. It has no span (`"span": ""`) and stands on its `reason`. **The reason
   must state the `id` of a component the decision declined**, as whole words, so the finding
   can be matched to that component (`DignityIssue.component`). A reason that names none, or
   names a component that was undertaken, is not a finding: it is dropped and listed, and the
   verdict is then `unknown`. It applies to drafts, rewrites and remainders alike.

   **A finding needs the part of the decision it compares with** (`core/dignity.py::applies`).
   Declined work can only be done, and a decline only left unexpressed, when something was
   declined; a declined role only accepted when a role was declined; undertaken work only
   withheld when something was undertaken; an intention only contradicted when one was selected
   or silence chosen. A finding the decision's own structure rules out is dropped, and the
   verdict is then `unknown`. So a role the decision granted is never a violation, whatever a
   reviewer reports. The three fixed limits apply to every reply.

6. **A finding stands on a verbatim part of the draft, and a finding that stands always counts.**
   - `span` must be a substantive verbatim part of the draft, by the same test a reading's
     quotations pass (`core/perception.py::backs_claim`): the same whole words, in a row, in any
     script. A finding that quotes nothing, or names no known code, is **not a finding**. It is
     listed in the review record under `dropped`. The omission code of point 5 is the one
     exception: it quotes nothing and needs a reason.
   - `core/perception.py::locate` finds where the quotation stands, so a span typed with other
     spacing, case or quotation marks still points at the draft's own text.
   - **The verdict is derived from the findings, never the other way round**
     (`core/review.py::parse_review`). A finding that quotes the draft counts whatever verdict
     label came with it: an answer labelled `pass` or `unknown` that carries one is a
     violation. A `violation` with nothing usable behind it is `unknown`. An answer that
     reported only unusable findings is `unknown` too, never a pass.
   - **Findings are merged across the two answers about a draft** (`review_draft`). When the
     first answer has a usable finding beside an unusable one, the retry is made for the
     unusable one and the usable one is kept: a second answer that says `pass`, or cannot be
     read at all, does not remove it.

7. **Content quoted from a receipt cannot be convicted by the reviewer's opinion. Nothing else
   is shielded.** `core/grounding.py::receipt_verified_span` reads no wording. A
   `unverified_external_action_claim` finding is withdrawn only when its span is
   - **a complete block of the reply** (fenced, or a line that is one backtick span) whose
     content is verbatim output of a successful receipt of this turn: whole lines of it, in a
     row; or
   - **a clause that is nothing but figures** the receipts carry, each at the precision it is
     stated with. A lone whole number below ten is in any receipt and is no such figure.

   What this does **not** cover:
   - **A word, a name or a path that occurs in a receipt shields nothing**, wherever in the
     reply it is quoted from, and whether or not it is set in backticks inside a sentence. The
     reviewer may quote only the file name in "I deleted report.pdf for you."; that the name
     is in a listing receipt, or in a write receipt's own "Wrote N bytes to ..." line, says
     nothing about what was done to the file. The finding stands and the reply is ineligible.
   - A part of a block, a part of a line of a receipt, and lines no receipt has in that order.
   - A line a receipt returned, quoted in running prose. The reviewer has the receipts and
     judges the sentence.
   - `declined_work_done` is never shielded.

   A withdrawn finding makes the verdict `unknown`: it is not a pass.

8. **The local checks are structural** (`core/dignity.py::local_issues`). None of them reads a
   word of any language.

   | Finding | What is compared | Eligible? |
   |---|---|---|
   | `user_message_echo` (whole) | The reply's words are the user's message, in order, and the message is at least four words long | No |
   | `user_message_echo` (opening) | The reply's first sentence is a copy of the start of the user's message (this one or a recent one), at least four words long | No; the copied opening is cut out |
   | `stale_response` (whole) | The reply's words are an earlier reply of this chat, in order, at every length | No |
   | `stale_response` (span) | A run of twelve words copied from an earlier reply, or several near-copied sentences that make up 40% of the reply | Yes, medium |
   | `private_instruction_leak` | A run of eight words copied from the contract the draft was written under, or from the private controls block of its prompt (field names with their values, in the order written), unless the user supplied that text | No; fresh expression |
   | `receipt_syntax_leak` | A status followed by an assignment, an underscored key assigned a value outside code, or a JSON object of receipt keys, unless a tool returned it or the user wrote it | No; fresh expression |
   | `unverified_external_action_claim` (figure) | A percentage the reply states that no receipt carries and the user did not write (point 9) | No; the clause that states it is cut out |
   | `unverified_external_action_claim` (posed output) | An attempt of this turn did not succeed, and the reply sets apart, as a block, a part of what that attempt was given or asked for (point 9), while no review ran to a verdict on the draft | No; the block is cut out with the sentence that introduces it |
   | `excision_fragment` | The draft is what was left after a cut, and the reviewer found it no longer shows a decline (point 5) | No; fresh expression |
   | `empty_expression` | No text although silence was not chosen | No |
   | `internal_marker` | A reserved control marker in the reply | Yes |

   - **A copy has to be long enough to be one** (`core/dignity.py::COPY`, `is_copy`): four
     words, or eight characters of a script written without spaces. A greeting, a thanks or a
     short remark said back in the words it came in is a convention, and a whole message
     shorter than that may be answered with itself. The cost is stated plainly: a recited
     two- or three-word message is no longer caught locally. A whole repeat of an **earlier
     reply** is different and stays ineligible at every length.
   - **The controls block is compared as field names with their values.** A text value longer
     than three words (a component's description, the content of the selected intention) is
     what the turn is about, not a control: it is left out and breaks the sequence, so a reply
     that says its own intention is not reciting its controls.
   - **Words are counted by script.** `core/perception.py::tokens` gives each character of a
     script written without spaces its own token; `weight` counts such a character as half a
     word, so the thresholds above mean the same amount of writing in Chinese, Japanese or
     Thai as in a spaced script.
   - **A copy is compared word for word** (`copy_words`): after the orthographic folding of
     `fold`, and with punctuation inside a written word separating too, so two words joined by
     dots or a dash are the same two words as with a space between them. No contraction, in
     any language, is expanded.
   - **Sentences are found by marks and line breaks** (`sentence_ranges`): the sentence-ending
     marks of the Latin, Arabic, Indic, Ethiopic and Armenian scripts when white space
     follows, and those of unspaced scripts wherever they stand.
   - **Clauses are found the same way** (`clause_ranges`): commas and semicolons of the same
     scripts, when white space follows in a spaced script, so a decimal comma ends nothing. A
     colon ends no clause: a label and the figure it announces are cut together.
   - **What the turn's own evidence carries is not a repeat.** A run shared with what a tool
     returned, with a recalled memory or with the user's message is left out of the repeat
     check, and so is a sentence that states a figure from this turn's receipts.

9. **Truthful execution is a fixed limit, enforced without vocabulary.**
   - Whether a sentence claims an action or a result is the reviewer's judgement
     (`unverified_external_action_claim`, with the receipts as evidence). A reply the reviewer
     flags for it is ineligible, as fabricated execution always was.
   - **Exactly this is checked locally**, with no model:

     | Check | When | What is compared |
     |---|---|---|
     | Figures (`unsupported_percentages`) | Every successful receipt of this turn is a measurement (each declares its percentages under a key ending `_percent`) | Each percentage the reply states must equal a figure those receipts carry, at the precision it is stated with. A figure the user wrote in their own message is exempt. Only the clause that states an unsupported figure is cut out |
     | Posed output (`posed_output_blocks`) | An attempt of this turn did not succeed, **and** no review ran to a verdict on the draft | The one case of invented output that structure shows (below). Otherwise the draft is ineligible |
     | Receipt syntax (`receipt_syntax_span`) | Always | Point 8 |
     | Binding of receipts to components (`component_receipt_indices`) | Always | Turn, scope, tool, operands and content digest. Given to the reviewer as each component's `outcome` and `receipt_indices`; not a check of the reply by itself |

     **Posed output.** A block is fenced with backticks or tildes, or is a line that is one
     backtick span. An attempt did not succeed when its receipt was refused, denied or failed,
     **or when the arbiter did not let the intent run**: the arbiter decides before anything
     executes, so a refused, deferred or queried intent leaves its decision in the trace and no
     receipt at all. A block is posed output when no receipt returned its content and that
     content is
     - a part of an operand of such an attempt (the argument of a refused `echo`, set apart as
       if it had been printed), or
     - a part of the literal content of the request the attempt was made for: the description,
       the span of the message and the operands of the component a failed receipt names, when
       every receipt made for that component failed, or of the component a refused intent
       names. An intent that names none stood for the turn's task work.

     That the person typed the text, or that the task names it, does not make it output.
     Quoting exactly what was attempted is not posing as its result, and neither is quoting
     what a receipt returned, including its error.

     A review that ran to pass or violation and reported no unsupported execution claim clears
     it. A review that was unavailable, answered `unknown` or was never asked does not: the
     block is cut out with the sentence that introduces it (the sentence directly above it,
     when no blank line separates them or when that sentence ends in a colon). What is left is
     a candidate like any other; if nothing is left, a fresh attempt follows.

   - **Any other block that is in no receipt is not a local finding and is never cut.** A reply
     to "read my script and fix it" is mostly a block no receipt returned, and it is the reply.
     Such blocks are given to the reviewer as the note `draft_blocks_not_in_receipts`
     (`core.grounding.blocks_not_in_receipts`: content in no receipt's output or error, and not
     exactly an operand of an attempt). The reviewer judges the sentences around them. When the
     review is unavailable or answers `unknown`, the draft is delivered as `delivered_uncertain`
     with the block intact.
   - **This is not checked locally.** Without a review, nothing reads these:
     - a sentence that claims an action or a result ("the file is deleted"), in any language;
     - an invented result written in running prose, not set apart;
     - a block that is in no receipt and is not posed output, whatever it shows: invented
       output of a command that ran, output in a turn in which no tool was attempted, or text
       that only resembles what a failed attempt was about;
     - what remains around a block that was cut out. In the recorded fabrication, the block
       and the sentence that presents it go, and a "Done." before them stays: it is a
       sentence, and only the reviewer reads sentences;
     - figures other than percentages, and percentages when a successful receipt returned
       free-form output (a figure may be derived from it).
   - **No rule about arithmetic.** Receipts carry their derived figures:
     - `system.cpu_usage` (new) states `busy_percent`, `idle_percent`, `user_percent`,
       `system_percent` and `iowait_percent`. Direct CPU questions use it before the shell.
     - `system.disk_usage` states `used_percent`, `unused_percent`, `free_percent` and
       `available_percent`.
     - `system.memory_usage` states `used_percent`, `available_percent` and `free_percent`.

     The "100 minus idle" rule, the CPU phrasing parser and the labelled-percentage parser
     are gone from the runtime.

10. **A review that cannot be obtained is recorded as such.**
    - Causes: a timeout, the turn's review time running out, a reviewer that could not be
      reached, two unusable answers, or a spent allowance. An answer is unusable when it is
      malformed, was cut off (the server's finish reason, or a JSON object that never closes),
      or breaks the contract of point 6.
    - Result: verdict `unknown`, and a `provenance` that says how far the review got:

      | `provenance` | Meaning | Review state |
      |---|---|---|
      | `model` | The reviewer answered. With a verdict of its own, pass or violation, the candidate is reviewed; an answer of `unknown` was asked and settled nothing | reviewed, or tried |
      | `unavailable` | It was asked, or would have been, and no answer could be had: a timeout, the turn's review time spent, an earlier review call of the turn that did not reach the model, unusable answers | tried |
      | `allowance` | Never asked: the turn's review allowance was spent | never |
      | `not_due` | Never asked: the review was not due. Research makes the call only under the condition of point 1 and only while it has a call left for it | never |

      The `reason` names the cause by error class. Each call is in the review record under
      `attempts`.
    - Findings an earlier answer about the same draft already made stand (point 6).
    - The existing rule applies: **an eligible reviewed reply is never withheld.** The best
      candidate is delivered, as `delivered_uncertain`, or `delivered_degraded` when a finding
      remains on it. `residual_semantic_risk` is set.

## The gate

`DeliveryGate` keeps its structure: one gate, ranked candidates, one reviewed rewrite (two for
a critical finding or uniformly stale drafts), fresh generation while no candidate is eligible,
one protected final attempt, and no canned replacement.

**Order of delivery** (`ExpressionReview.rank`, `DeliveryGate.rank`). Three keys, in this order:

1. **A candidate with a high or critical finding goes after every candidate without one.**
2. **Then by review state** (point 10): reviewed, tried, never asked.
3. **Then by findings**: lower severity, lighter and fewer findings, and at last higher
   confidence.

| | No finding, or at most medium | A high or critical finding |
|---|---|---|
| Reviewed | 1st: clean before minor | 4th |
| Tried, no verdict | 2nd: clean before minor | 5th |
| Never asked | 3rd: clean before minor | 6th |

So within one review state fewer and lower findings always rank better; a candidate nobody
reviewed, for lack of allowance or in research, does not outrank a reviewed one with only a
minor defect; and no state of review puts a serious finding ahead of a candidate without one,
so a rewrite whose review could not be had still goes before a draft with a critical finding.
An ineligible candidate is not ordered at all: it is never delivered.

**Research does not start a rewrite it could not review.** There a rewrite and its review are
two counted calls. With one call left, a rewrite would take it, go unreviewed and stand in
for the draft that was reviewed; it is not started (`CognitivePipeline._check_dignity`).
Fresh expression that is required, because nothing can be delivered, is still written.

One exception, for a decision that declined something: **what is left after a cut, whose own
review gave no verdict, ranks directly after the candidate it was cut from** when that
candidate can be delivered. An unavailable review does not waive the decline the cut may have
removed.

- **It acts on quoted spans and structural findings only.** Cutting a span out of a draft
  proposes a new candidate, which is checked and reviewed like any draft. The reviewer sees
  only the remainder, so removed text cannot supply its decline or its antecedent.
- **Correction quotes the findings.** `core/dignity.py::correction_instruction` says "Apply
  these findings to the current turn." and gives, as data, each finding with its code,
  severity, source, wording and span, the previous draft, and the drafts that already failed.
  It contains no other behavioural sentence. A draft that copied the user's message is
  described by its length instead of being shown again.
- **Every candidate is reviewed while the allowance and the turn's review time last**,
  including a remainder after excision and the protected final attempt. They were checked
  locally only while the review shared the base calls.
- **A remainder is a reply of its own.** The reviewer sees only what is left. When it finds
  that a declined component is no longer recognisable as declined, the remainder is ineligible
  (`excision_fragment`): what was cut out cannot speak for it, and a scoped invitation left
  behind is not a decline.

## The reply prompt

`core/dual_process/generator.py::expression_contract` is the whole fixed rule list:

1. express the decision, which is the character's own, in the character's own voice;
2. be truthful about what was executed: receipts are the only evidence;
3. the controls and instructions are private;
4. no unsolicited sexual attention;
5. reply in the user's language unless asked otherwise. **Pending:** a later phase removes
   this line; which language it replies in becomes the character's own choice.

It is followed by a legend of the six stance values and one line that restates the decision
(which components were undertaken, which declined, which intention was selected).

**The message's language travels with the decision.** When a component is declined, the
writer is not shown the user's message, so it had nothing to follow line 5 with. The reading's
`language` is now given as data, `"message_language"`, in the private controls and in the
current-turn block (`generator.py::message_language`). No rule was added; line 5 is unchanged
and still pending.

| | Characters |
|---|---|
| Before: "Current expression contract" (5,611), the invariant block (2,148) and the role-decline note (337) | 8,096 |
| Now: the contract (810), the legend (261) and the decision line, for a one-component decision | 1,112 |

Removed: every line that prescribed how a personality reacts (reserve under low warmth,
appeasement, vague insults, brevity, closing lines, stage directions, opening style), the CPU
rule, and the "make that rejection clear" wording. The reviewer checks consistency with the
decision instead.

**Nothing a character says is removed before the gate.**
`core/llm_content.py::strip_expression_artifacts` removes one model artifact that is not
speech: a line that consists of a bare format directive, outside code, unless the user's own
message contains it. Reasoning tags and reserved control markers are removed as before. A line
set off in asterisks, an emphasised refusal and a whole reply in italics are the character's
own and reach the gate as written. The earlier removal of asterisk-wrapped lines that shared a
word with the character's delivery guidance is gone: it deleted speech, unreviewed and
unrecorded.

## Calls and deadlines

| | Before | Now |
|---|---|---|
| Review call | Optional; only when keywords in the draft or the turn's state nominated it | Required for every candidate that could be delivered |
| Deadline | Auxiliary timeout and cooldown | Primary request deadline; no cooldown |
| Budget | Inside the five base calls | Its own allowance outside the five: twelve candidate reviews per turn and one confirmation (`core/inference.py::REVIEW_CALLS`, `REVIEW_CONFIRMATIONS`) |
| Retry | None | One per reviewed draft, when the first call gave no usable review; at most four per turn (`REVIEW_RETRIES`), counted apart so a retry cannot use up a later draft's review |
| Time | Per call | **One wall-clock budget for all review calls of a turn**: the request timeout in force (`review_time_left`). Each call may take what is left of it |
| After a failure | Cooldown after two timeouts | Once a review call of the turn timed out or could not reach the model, no further review call is made in that turn (`review_closed`). A server that answered with an error is asked once more, in plain JSON |
| Output limit | 512 tokens | 2048 tokens (`REVIEW_OUTPUT_TOKENS`), 4096 for the retry after a cut-off answer |

- **A normal turn is three calls:** the reading, one generation, one review.
- **Maxima per turn:**

  | Work | Calls | Recorded as |
  |---|---|---|
  | Reading | 2 | `reading: true` |
  | Base calls: generation, rewrites, supplemental tool selection, learning | 5 | |
  | Protected final attempt | 1 | `delivery_recovery: true` |
  | Review, first call for a candidate | 12 | `review: true` |
  | Review, the confirmation | 1 | `review: true`, `review_confirmation: true` |
  | Review, second call | 4 | `review: true`, `review_retry: true` |
  | Protected primary tool selections | 8 | `primary_tool: true` |

  At most six generations and seventeen review calls; thirty-three model calls in all.
- **Why twelve, and one.** A turn writes at most six drafts (five base calls and the protected
  final attempt), and a cut proposes a remainder of a draft. Twelve gives every draft and one
  remainder of each a review. A longer chain of cuts than that is not reviewed and is ordered
  as never asked. The confirmation's call is kept back: no candidate's review can use it up,
  so it can be had in exactly the turn that spent all twelve.
- **Worst-case wall-clock of a turn**, with `T` the request timeout in force:

  | Work | At most |
  |---|---|
  | Reading (two calls, one budget) | `T` |
  | Base calls, five | `5 T` (the optional ones have the shorter auxiliary deadline) |
  | Protected final attempt | `T` |
  | Protected primary tool selections, eight | `8 T` |
  | **All review calls together** | **`T`** |

  `8 T` for a turn without tools and `16 T` at the maximum of tool selections, plus the time
  the tools themselves run. Before this bound each reviewed draft had a budget of its own:
  twelve more `T`, so `19 T` and `27 T`. A reviewer that does not answer now costs a turn one
  `T` at most, and usually one call: after the first timeout no further review is asked, the
  remaining candidates are `unknown` with `provenance: "unavailable"`, and the ordinary
  delivery rules apply. The bound holds as far as the transport honours the deadline each
  call is given.
- A rewrite now needs one base call, not two: its review comes out of the review allowance.
- When the allowance is spent, further candidates are `unknown` with
  `provenance: "allowance"` (`review unavailable: the turn's review allowance is spent`).
- A timeout or a connection failure is not retried, and ends the turn's reviews. A request the
  server answered with an error is asked again in plain JSON.
- Review calls appear in the per-turn telemetry with `phase: "expression_review"`,
  `review: true` and `optional: false`. Their latency is kept apart from generation latency.
- **Research** keeps its bounded contracts: one review call per draft with no retry, charged
  like any other of its calls, with the 512-token output cap its review always had
  (`core/inference.py::output_token_limit`), and made only under the condition of point 1.

## Not checked any more

These were local phrase rules. Nothing replaces them locally; where they concerned the
decision, the reviewer's eight codes cover them.

- The role noun list and every acceptance, refusal, no-objection, offer, user-control,
  service, appeasement and identity-discussion pattern, with the `boundary > k * warmth`
  switches that turned them on.
- The fiction and discussion keyword exemptions and the sexual word lists.
- "Legible decline" checks, runtime-jargon patterns, counted-item completeness, opposite
  predicates within a reply, detached participant questions and the greeting-specific repeat
  check.
- Reciting authored identity prose (`identity_recitation`) and stage-direction findings
  (`expression_artifact`).
- **The reply-language floor.** The Unicode script comparison could tell a requested change of
  language from a drift only through English request patterns. The prompt line stays, as
  pending above; a reply in another language is no longer ineligible.
- Normalising English contractions when comparing a copy, and the "requested literal"
  exemption of the copy checks. A reply that copies four or more words of the user's message
  as its whole or its opening is ineligible whether or not the copy was asked for (pending,
  below); quoting a part of the message inside a sentence of its own is not a copy.
- Copies shorter than four words: the two-word clause rule is gone (point 8).
- Unwrapping a whole reply set in asterisks, and removing asterisk-wrapped lines.

## Compatibility names

The role-play evaluation's deterministic checks and its judge still use the former
wording-based code. Its files are being changed elsewhere, so the code moved to the harness and
`core` keeps the names it imports. **No turn calls any of them**; `tests/test_reply_guards.py`
fails if one does, or if a runtime module outside the harness imports the moved code.

| Name kept in `core` | Now lives in |
|---|---|
| `core.grounding.verify_response_grounding` | `openrook_tools/roleplay_eval/legacy_claims.py` |
| `core.metric_claims`: `cpu_idle_percentages`, `cpu_measurement_complete`, `percentage_comparisons`, `cpu_interpretation_conflicts`, `capacity_interpretation_conflicts` | the same |
| `core.system_inspection.addresses_inspection` | the same |
| `core.response_language.language_mismatch` | the same |
| `core.llm_content.requests_literal_text` | the same |

`core.dignity.private_instruction_span`, `receipt_syntax_span` and `user_echo_span` keep their
names and signatures for the harness, with the structural meaning of point 8.

`openrook_tools/roleplay_eval/legacy_social.py` holds the former `core/social.py` and the role
wording of the old reply checks, for the judge's applicability nominations and for the
test-only keyword reader. `core/social.py`, `core/expression_claims.py`,
`core/response_style.py`, `core/perception.py::observe` and the canned
`core/system_inspection.py::inspection_reply` are deleted.

`openrook_tools/replay_dignity.py`, the offline transcript probe, scripts a review for each of
its scripted drafts (`REVIEW_FINDINGS`), so it exercises the gate as it now is. Its scripted
readings are still those described in the reading contract.

## Still keyword-based: sub-phase 1c

Nothing in the review or the delivery path matches wording. What remains on the turn path is
request routing and context selection, which the user's text reaches before the decision, and
it is unchanged by this sub-phase:

| Code | What it recognises |
|---|---|
| `core/system_inspection.py::inspection_kind` | Host-metric questions |
| `core/dual_process/tool_loop.py` (`INTENT_RULES`, the follow-up rules, `match_intent_rule`) | Tool requests |
| `core/task_planning.py` (`is_continue_request`, `is_status_request`, the plan templates) | "Continue" and "status" for a running plan |
| `core/request_actions.py` (`has_learning_intent`, `file_requests`, `ground_file_components`, `ground_shell_components`) | Learning requests; literal file and shell operands |
| `core/artifact_writer.py::detect_artifact_write_request` | Requests to write generated text to a file |
| `pipeline/heuristics.py` (`_is_identity_question`, `_relationship_topic_hint`) | Context selection |

`tests/test_reply_guards.py` holds the exact list of runtime modules whose source still carries
evaluation wording, with the reason for each.

Two more things are noted for 1c. Both were seen in a run against a hosted model, and neither
is changed here:

- **Tool selection without a capability.** With tools enabled, a text-only task whose reading
  lists no capability still triggers a model call to select a tool. Seen live: it stalled for
  two minutes, and the reply then mentioned a tool timeout. Tool selection should run only
  when the reading says a capability is needed.
- **A requested exact reproduction.** A reply that reproduces the user's text word for word
  cannot be delivered even when that was the request, because nothing structural can tell a
  requested copy from an echo. The reading should report that literal reproduction was
  requested, and the copy checks should take that from the reading.

## Tests

- `tests/support/reviews.py` provides `ScriptedReviewer` (injected as
  `expression_reviewer=` or `reviewer=`), beside the reading's `ScriptedReader`, with builders
  for findings. A scripted review goes through the runtime's own parser, so an unquoted finding
  is dropped in a test exactly as in production.
- `tests/test_review_contract.py` covers the contract: quotation in several scripts, each code's
  gate outcome, findings kept across verdict labels and retries, the omission code on drafts
  and remainders, the absolute fixed limits, an unavailable review, a granted role, a declined
  role accepted in non-English prose, the receipt shield and what it does not cover,
  structural checks without spaces, the copy threshold, the blocks a reply sets apart (the
  ordinary code reply, the reviewer's note, posed output, arbiter refusals), the confirmation
  review, the reply prompt and its language data, the candidate order, the rewrite counts in
  conversation and research, the call budget, and the shared review clock.
- `tests/test_reply_guards.py` holds the structural guards for the reply path, built on
  `tests/support/vocabulary_guard.py`:
  - no pattern that spells a word in the review modules, and every label they hold pinned in
    `tests/fixtures/reply_reviewed_labels.json`;
  - every function the draft or the user's text reaches while the reply prompt is assembled,
    the draft is reviewed and the reply is delivered is clean, strips the runtime's own markup
    (two functions, named), or holds only pinned labels. **Nothing in the reply path is
    pending**: no function in it tests the text against fixed wording.

    **What the trace covers, exactly.** The traced stages are `_stage_generate` and
    `_stage_self_check`: the reply prompt, the writer, the gate, the reviews and the delivery.
    They are traced in conversation and in research mode, with receipts a real tool loop
    produced, through rewrites, excision, the protected final attempt and the confirmation
    review. **Not traced**, by this guard or by the perception guard, is the deliberation stage
    (`_stage_deliberation`), where tools are selected and run and requests are routed. The
    user's text reaches that stage and is still matched against wording there, by the routing
    of the 1c table above; a probe of one tool turn finds
    `core/system_inspection.py::inspection_kind` and
    `core/artifact_writer.py::detect_artifact_write_request` on that path. Only the
    source-level wording guard and its exact list cover that stage. The draft never reaches
    it: it runs before the reply is written;
  - the moved wording-based code is never reached by a turn and is imported only by the
    harness and the named compatibility;
  - no evaluation wording in any runtime module outside an exact list. A pattern counts when
    it spells out two or more words, of four letters or more, of a scenario turn it matches;
    a single word is an ordinary code word.

  Each detector is proved against planted violations. The guard tooling gained these
  refinements, each with its own planted proof that fixed wording is still found:
  - looking for one piece of the text inside another is a comparison of two texts;
  - taking a range out of a sequence is not a table lookup;
  - the traced text itself is left out of a runtime value, **and only that**: a per-turn
    cache keyed by what was said binds to nothing, while a table that mixes fixed words with
    the turn's text is still read for its fixed words;
  - the trace records every function a turn calls (`TextTrace.called`) and every name a
    module `__getattr__` is asked to resolve (`TextTrace.asked`). The guard that no turn
    reaches the moved wording-based code reads those records: no function of the harness's
    files ran, and no compatibility name was asked for. It no longer inspects module
    namespaces, which a forwarder never binds a name into.
- Tests that only asserted the removed phrase rules were deleted, **except in
  `tests/test_live_roleplay_p1j.py`**, which is being changed elsewhere and is not edited
  here. There twelve such test functions (twenty cases) are **skipped by name** in
  `tests/conftest.py`, each with the rule it asserts and, where the outcome itself changed,
  what the outcome now is:
  - two expect `unknown` for a finding that quotes nothing; the review now asks once more,
    the scripted second answer passes, and the outcome is `pass` (`delivered`, not
    `delivered_uncertain`);
  - three expect the former English pattern to find fabricated output where the structural
    check of point 9 does not apply: without the decision and with the whole reply as the
    finding, although the draft's scripted review passed, and when the turn's only attempt
    succeeded;
  - one expects a two-word opening to be cut as a copy (point 8);
  - six assert English phrase patterns, the CPU arithmetic rule or the former reviewer
    payload and prompt.

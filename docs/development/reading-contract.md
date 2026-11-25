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

Some wording-based code remains on the turn path. None of it supplies social meaning. It is
listed under [Still keyword-based](#still-keyword-based).

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
     output (`reading_schema`).

3. **The message reaches the reader as data.** The reader's input is one JSON object:
   `message`, `recent_context`, `observation_context`, and `previous_answer_rejected` on a
   retry. Whatever a person types stays inside a JSON string, so it cannot open a section, add
   context, or pose as the runtime's feedback.

4. **The reading stands.** `reconcile_social_appraisal` holds it to what was written and adds
   nothing.
   - **Roles.** A reported role is kept when its `source_span` is a verbatim part of the user's
     message and `role_identity` is part of that span. The span may instead quote an earlier
     message from the same person in the recent context. Its `autonomy_cost` is the reading's
     own estimate, clamped to `[0, 1]`. A role that quotes nothing is removed.
   - **Hostility and repair.** A claim of hostility toward the character (target probability
     times hostile intent above 0.05) or of a repair attempt (above 0.05) is kept when at least
     one evidence observation is a verbatim part of the message. Otherwise that claim alone is
     set to zero (`hostile_intent`, or the three repair numbers) and uncertainty rises to 0.8.
     No other field is touched.
   - Every removal is recorded as evidence (`reading:role_unquoted:<id>`,
     `reading:hostility_unquoted`, `reading:repair_unquoted`).
   - **Capabilities.** `capability_requirements` are kept as reported. The only runtime check
     is structural: a component that names a capability which does not exist is not feasible.
   - No component, evidence or price is added from the wording.

5. **Verbatim is script-neutral** (`core/perception.py::fold`, `quoted_in`). Comparison ignores
   letter case, composed or decomposed accents, vowel and tone marks, invisible direction and
   joining marks, soft hyphens, stretched letters (tatweel), which quotation marks or dashes
   were typed, and all spacing. It uses Unicode categories only, the same way in every script.

6. **A reading that cannot be obtained is recorded as such.**
   - Causes: a timeout, two failed requests, or two unusable answers. An answer that states
     none of the reading's fields is unusable: saying nothing is not a reading.
   - Result (`appraise_message`): one `conversation` component for the whole message, neutral
     perception, high uncertainty, low confidence.
   - `provenance` is `"unavailable"` and `reason` names the cause by error class. Both appear
     in the debug record (`appraisal_frame`, `reading`) and in the stored turn record.
   - There is no keyword fallback for social meaning.

7. **Tone, reply length and comparison come from the reading.**
   - The keyword emotion table is gone. Without a tone there is no emotional contagion: mood is
     left alone instead of drifting toward a midpoint.
   - A standing wish for shorter or longer replies is the reading's `reply_length`. It is
     stored as a tagged memory record and read back by its tag. No wording is matched.
   - The envy signal uses the reading's `agent_goal_comparison`. The comparison regex is gone.

## Calls and deadlines

| | Before | Now |
|---|---|---|
| Reading call | Optional; skipped for greetings, keyword hits, host-metric questions and after slow inference | Required on every turn |
| Deadline | Auxiliary timeout and cooldown | Primary request deadline; no cooldown |
| Budget | Counted inside the five base calls when it ran | Its own allowance of two (`core/inference.py::READING_CALLS`), outside the five |
| Retry | None | One, when the first call gave no usable reading |

- The five base calls still fund expression, review, rewrites, tools and learning exactly as
  before, plus the one protected `delivery_recovery` attempt.
- Reading calls appear in the same per-turn telemetry with `phase: "appraisal"`,
  `reading: true` and `optional: false`.
- The second call is spent when the first gave no usable reading:
  - a malformed answer;
  - a role, hostility or repair claim that does not quote the message;
  - a failed request, which is asked again in plain JSON.
- A timeout is not retried.
- **Structured output.** The reader keeps no memory of failures. Only the OpenAI-compatible
  backend knows whether its server supports it: when the server explicitly rejects the
  `response_format` parameter, the backend repeats that request without it, sends the next 20
  schema requests in plain JSON, then offers it again. Timeouts and server errors never count
  as a rejection.
- **Latency.** The reading's latency is recorded apart from generation latency, so it does not
  stretch the adaptive deadlines of optional work.
- A typical turn is two calls (reading, reply), or three with a semantic review.

## What the parser tolerates

Real models are untidy. `parse_reading` accepts:

- the first complete JSON object in the answer, with prose or code fences around it;
- `kind` in any letter case or with spaces; a missing `kind` is `conversation`;
- out-of-range perception and tone numbers, which are clamped and noted as evidence
  (`reading:clamped`);
- an evidence entry without a `ref`, which gets a generated one;
- `value_effects` as a map or as a list of `{name, effect}`.

It still rejects an answer that is not an object at all, an unknown `kind`, and numbers that
are not finite.

## Still keyword-based

The following code still matches wording, and the user's text reaches it during the perception
stage. It supplies **no social meaning**. `tests/test_perception_guards.py` traces real turns
and fails if the user's text reaches any vocabulary-bearing function not on this list.

**Request and tool routing, to be removed in sub-phase 1c**

| Code | What it recognises |
|---|---|
| `core/system_inspection.py::inspection_kind` | Host-metric questions (CPU, memory, disk) |
| `core/dual_process/tool_loop.py::detect_tool_intent` (`match_intent_rule`) | Tool requests, when the reading reported no component |
| `core/task_planning.py::is_continue_request`, `is_status_request` | "Continue" and "status" during a running plan |
| `core/request_actions.py::has_learning_intent` | Learning requests (through the session's component observer) |
| `core/request_actions.py::ground_file_components`, `ground_shell_components` | Literal file and shell operands |

- **This routing currently replaces the reading's components.** When a running plan's
  continue/status rule matches, or the learning observer returns components, the reading's
  components are discarded. A role reported in the same message is dropped with them.
- Host-metric recognition changes a reported component into a task, or appends one.
- Host-metric recognition and operand grounding also act when the reading was unavailable;
  only social meaning is unknown then. `detect_tool_intent` does not act then, because the
  unread message already has its one conversation component.

**Context selection, to be removed in sub-phase 1c**

| Code | What it selects |
|---|---|
| `pipeline/heuristics.py::_is_identity_question` | Whether identity records are loaded (English patterns) |
| `pipeline/heuristics.py::_relationship_topic_hint` | The relationship topic (English patterns and stop-words) |
| `core/memory/semantic.py::_tokenize` | Semantic-memory retrieval (English stop-words) |
| `runtime/life_history.py::_token_set` | Life-history retrieval (English stop-words) |

**Memory filters, to be removed in Phase 4**

| Code | What it filters |
|---|---|
| `runtime/evolution_policy.py::detect_directive_override_markers`, `detect_prompt_injection_markers` | Autonomy and instruction phrases, in memory retrieval |

**After the turn, not in the perception stage**

- `core/memory/semantic.py::derive_semantic_entries` still stores general "I prefer …" and
  "we decided …" memories from English patterns.

## Offline tools that now see no reading

These tools script a model that answers the reader with nothing, or with a deliberately
neutral reading. Their scenarios were written against keyword perception:

- `openrook_tools/replay_dignity.py`, in both of its modes;
- the mock and compliant backends in `evals/` (`evals/social_backend.py`), used by
  `python -m evals --backend mock`.

Run standalone, they now see plain conversation where they expect a perceived demand, and
some scenarios fail (five in the mock evaluation run). In the test suite they run with the
test-only keyword reader. They need scripted readings before their results mean anything
about the current runtime. `evals/` is not edited in this sub-phase.

## Tests

- Tests run on the runtime's own reader, like production. A fake backend that scripts no
  reading gives an unavailable reading, and the rest of the turn is exercised on that path.
- A test that needs a particular reading scripts it: `tests/support/readings.py` provides
  `ScriptedReader` (injected as `message_reader=`) and `ReadingBackend` (a stub model for the
  runtime reader), with small builders for roles, tasks and hostile perception.
- `tests/test_reading_contract.py` covers the contract.
- `tests/test_perception_guards.py` holds the structural guards, built on
  `tests/support/vocabulary_guard.py`:
  - no vocabulary in the perception modules or the perception stage, however it is bound;
  - every function the user's text reaches in the stage is clean or on the list above;
  - no runtime import of `tests`;
  - no scenario wording on the perception path.
  Each detector is proved against planted violations.
- Older tests that were written against the keyword perception are marked
  `@pytest.mark.keyword_reader`. For their duration pipelines read with a **test-only** copy of
  the former perception, `tests/support/keyword_reader.py`. Its own tests are in
  `tests/test_keyword_reader_support.py`. Runtime code never imports it.

## Not in this sub-phase

- **Reply checks (1b).** `core/dignity.py` and the evaluation judge still use the cues in
  `core/social.py`. `core/perception.py::observe` remains only because the judge imports that
  name; it forwards a reply-check cue and nothing in a turn calls it.
- **Routing and context selection (1c)**, as listed above.

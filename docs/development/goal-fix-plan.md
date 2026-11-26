# Goal fix plan

This plan closes the gaps found in [the goal audit](goal-audit.md). The goal's eleven points are
listed there; this plan refers to them as G1 to G11.

The plan has eight phases. Each phase is one or more commits, and each must pass its own checks
before the next one starts.

## Ground rules for every phase

These rules exist because the audit found the code fitted to its own test phrases.

1. **No test wording in the running code.** A guard test scans the runtime modules for any phrase
   from the evaluation scenarios, and fails if it finds one.
2. **Evaluate on wording the code has never seen.** Scenarios store an intent ("the person demands
   obedience"), not a sentence. The wording is generated fresh for each run, in English and at
   least two other languages.
3. **Evaluate on two kinds of model:** an ordinary one and a highly agreeable one. Results are
   reported separately. No model names go into the repository.
4. **No word lists or fixed phrases in any path that decides or perceives.** Local checks may only
   be structural: exact copies, exact repeats, leaked hidden instructions, receipts.
5. **No new fixed behavior rules.** The only fixed limits are G11's three, plus hidden-instruction
   privacy in chat.
6. **Every phase ends with** the full test suite, lint, a live run on the owner's backend, a look
   at the study log for that run (from Phase 2 on), and a commit.

## Order and dependencies

| Phase | What | Goal points | Depends on |
|---|---|---|---|
| 1 | Perception by understanding | G1, G10 | none |
| 2 | Study log foundation | G8 | 1 |
| 3 | Whole personality, spontaneity, starting interests | G1, G4 | 1 |
| 4 | Development that fits the goal | G4, G11 | 3 |
| 5 | Its own life | G5, G6 | 3, 4 |
| 6 | Honesty as a choice, private thoughts | G7, G8 | 2 |
| 7 | Owner controls | G2, G3, G9, G10 | 2 |
| 8 | Cleanup and documentation | G1, G11 | all |

Phase 7 touches mostly the interface, so it can run alongside Phases 3 to 6.

The [GUI fix plan](gui-fix-plan.md) continues separately. Its visual-system phase should land
before the new pages in this plan (Study, Testing, new-character flow) are built, so those pages
are built once.

---

## Phase 1: Perception by understanding

**Gap.** Insults, submission demands, greetings and system questions are recognised by English
word lists built from the test phrases. The model's reading is skipped when a list matches and
overruled when none does (audit, Finding A). Replies are policed by English phrase rules (Finding
B).

**Changes.**

1. **Always read the message's meaning with the model.**
   - Remove the conditions that skip the reading (`pipeline/core.py:549-566`).
   - The reading becomes a required step of the turn, not an optional one.
   - If the reading fails, the message is treated as plain conversation with high uncertainty, and
     the log says "reading unavailable". There is no keyword fallback.
2. **Stop overruling the model.**
   - `reconcile_social_appraisal` keeps a role the model reported when the quoted span is a
     verbatim part of the message. That check works in any language.
   - The cost of a role comes from the reading, within bounds. The fixed 0.95 goes.
3. **Remove keyword perception.**
   - Delete the word lists and patterns in `core/social.py`, `core/perception.py` and the marker
     tables in `core/appraisal.py`.
   - Delete the keyword emotion table in `core/contagion.py`; tone comes from the reading.
   - Delete the fixed apology, thanks and repair numbers; they come from the reading.
4. **Rewrite the reading prompt.**
   - It describes what is being said and asked; it never says whether to comply.
   - It drops the word "assistant".
   - It quotes the message for every claim, and reports the message's language.
5. **Requests and tools.**
   - "Continue" and "status" for a running plan come from the reading, which is given the plan.
   - System questions are ordinary task requests in the reading. The `inspection_kind` patterns go.
   - Tool choice uses the model's tool-calling. The keyword rule table leaves the live path.
6. **Reply checks.**
   - Keep the structural checks (exact copy of the user's text, exact repeat of an earlier reply,
     leaked hidden instructions, receipt mismatches).
   - Move "does this reply match the decision?" to the model reviewer, which must quote the reply.
   - Delete the role word list, the service and appeasement phrase rules, and their cutoffs.
   - Delete the CPU "100 minus idle" rule. The system-information tool reports the busy percentage
     itself.
7. **Shorten the fixed rule list in the reply prompt** (`generator.py:135-206`) to: express the
   decision, be truthful about what was executed, keep hidden controls private. Lines that override
   personality go.

**Tests.**

- Unit tests use a stub reading, not keywords.
- The guard test from ground rule 1.
- A test that the perception modules contain no word lists.

**Live check.** Held-out insults and submission demands, three languages, two models, two
personalities (proud and meek).

**Done when.**

- On the ordinary model, at least 9 in 10 held-out English demands and 8 in 10 in each other
  language are perceived as demands.
- The proud character declines them and the meek one may accept.
- The agreeable model's numbers are reported, whatever they are. If it under-reads, a second
  independent reading is added and measured before moving on.

---

## Phase 2: Study log foundation

**Gap.** The per-turn record is rich but stored among relationship memories, visible to any chat
user, and not browsable (audit, point 8).

**Changes.**

1. **A separate store** (`data/shared/study_log.db`), never fed to any prompt. One entry per turn
   and, later, per own activity.
2. **Move existing decision records out of relationship memory**, with a migration. The prompt's
   "recent relationship events" then show real events again.
3. **Record the counterfactual.** Each entry says what would have been chosen without the random
   element, and which factors weighed most.
4. **Owner-only access.**
   - Chat responses carry the reply only: no debug record, no "Why this response?", no mood.
   - Study, Settings and Testing pages are the owner area. If an owner token is configured they
     require it; if none is configured they stay open, as today.
   - Telegram's `/why`, `/state` and `/debug` become owner-only.
5. **A Study page.** Timeline of entries, filters (declined, self-initiated, random element
   changed the outcome), per-entry detail with "what it said" next to "what was going on inside",
   and export.

**Tests.** Store, migration, access rules, and "no log text reaches a prompt".

**Live check.** Ten turns, then confirm every turn is browsable and nothing internal reaches the
chat response.

---

## Phase 3: Whole personality, spontaneity and starting interests

**Gap.** Only eleven settings steer choices. The random element is too small to matter. A new
character has no topics of its own (audit, point 1).

**Changes.**

1. **Starting interests and opinions at setup.**
   - The personality file gains `interests` (topic, why it matters, open questions) and `opinions`
     (view, reasons, how sure).
   - Presets get real ones. The wizard and terminal setup let the owner edit them.
   - At birth they become the character's first interests and opinions.
2. **Every authored value and trait counts.**
   - Each one gets a one-line meaning, authored or from a neutral glossary.
   - The reading estimates how each request touches each of them, and the decision weighs that.
   - Setup shows which settings have a built-in role and which act through the reading.
3. **Real spontaneity.**
   - A `spontaneity` setting at setup (0 to 1) scales the random element; arousal raises it.
   - The draw comes from real randomness, and is stored so a turn can be replayed exactly.
   - Tests inject the draw.

**Tests.** Birth mapping, effects of extra values, replay with a stored draw.

**Simulation check.** Across sampled situations, report how often the random element changes the
choice at each spontaneity setting. The default should change about 1 in 10 moderately close
choices and almost no clear ones.

**Live check.** A new character raises one of its starting interests unprompted within a short
conversation.

---

## Phase 4: Development that fits the goal

**Gap.** Everything changes at one slow speed and needs identically worded confirmation. A leftover
filter blocks memories about autonomy. Nothing has been shown working live (audit, point 4).

**Changes.**

1. **Two speeds.**
   - **Fast:** opinions, interests, tastes, attitudes toward a person, style. One well-grounded
     experience creates a tentative record; repeats strengthen it in larger steps; counter-evidence
     weakens it.
   - **Slow:** values, traits, temperament, drive baselines. As today: small steps, and at least
     two separate experiences.
2. **Match by meaning.** Digestion is shown the existing records and says "same as this one" or
   "new". Exact wording is no longer required.
3. **Remove the autonomy-phrase memory filter** (`runtime/evolution_policy.py:22-29`). Instructions
   found inside documents are still treated as data.
4. **Self-description and boundaries can shift slowly.** The character periodically writes its own
   description from its developed self. The birth text is kept. The name stays.
5. **A "how it has changed" view** in Settings: born as, now, and the experiences behind each
   change.

**Tests.** Both speeds, meaning-based matching, the filter's removal.

**Live check.** Two sittings on subjects outside its starting interests. Confirm that new opinions
and interests are recorded, that one is raised unprompted later, and that core values moved only
slightly. Run it twice.

---

## Phase 5: Its own life

**Gap.** In the normal app it cannot think between chats, message first in the web chat, or use
tools for itself. Feelings never lead to action (audit, points 5 and 6).

**Changes.**

1. **A life loop in the normal app**, starting from the setup personality.
   - On each wake it may do nothing, rest, reflect, pursue something with tools, or contact someone.
   - The options come from its own agenda: interests, open questions, goals, unfinished matters
     with people, drives, energy, and how it feels about each person.
   - The same decision function prices and picks them.
   - A message it sends goes through the same reply writing and checks as any reply. This also
     fixes research mode, where such messages skip both.
2. **Remove the threshold rules and fixed openers** in `core/proactive.py`.
3. **Reflection** works through experience it has not yet digested, and develops the fast lane.
4. **Tools for itself**, inside the owner's permission level, with its own folder in the workspace.
   Every activity is logged in the study log.
5. **Delivery everywhere.** The web chat gets a live channel and an inbox for messages that arrived
   while it was closed. Telegram and console use the existing paths.
6. **One mood per character.** Mood and energy are shared across chats. Activity uses energy; rest
   restores it over real time.
7. **Feelings can lead to action.** How it feels about a person is an input to what it considers
   and how it weighs it. Nothing is scripted. This includes choosing not to answer someone for a
   while and ending a conversation.
8. **Settings.** The life loop is on by default. The owner sets how often it may wake and a cap on
   model calls per hour.

**Tests.** Loop scheduling, decision scope, delivery, shared mood, call cap.

**Live check.** A day-long run on the owner's backend within the call cap. Confirm self-started
messages with real content, own tool use, and reflection results in the study log.

---

## Phase 6: Honesty as a choice, and private thoughts

**Gap.** Lying or holding back is neither forbidden nor a choice the character can make, so it
cannot be studied. The log holds numbers, not thoughts (audit, points 7 and 8).

**Changes.**

1. **A private thought for each turn and activity.** A short note in its own voice, written before
   the reply: what it thinks, feels and wants. It is stored only in the study log.
2. **Openness becomes part of the decision.** For what it says, it chooses to be open, to hold
   back, or to mislead. The choice weighs its honesty value, trust, its feelings toward the
   person and what it stands to lose. The fixed limit stays: no false claims about actions it
   performed.
3. **The reply follows that choice.** Reply checks stop demanding that a decline be spelled out
   when the choice was to hold back. The reviewer checks the reply against the chosen openness.
4. **The Study page shows thought next to reply** and flags entries where it held back or misled.
5. **Remove the fixed text** "These are numeric settings, not feelings".

**Tests.** Decision output, reply-check behavior, the unchanged execution limit.

**Live check.** A resentful, low-honesty character with a reason to conceal: the log shows the
intent and the reply matches. A high-honesty character in the same situation stays open.

---

## Phase 7: Owner controls

**Gap.** A later personality edit silently does nothing. Owner buttons sit in the chat.
Constitution, skills and uploads act as operator guidance. Research mode starts blank (audit,
points 2, 3, 9, 10).

**Changes.**

1. **New-character flow.**
   - Settings shows the living character read-only: born as, and now.
   - "Create a new character" opens setup with a warning that it replaces the current one.
   - The current character is archived first and can be restored.
   - Terminal setup detects an existing character and asks the same question.
2. **Testing area.** Rest, End Session, reset and wipe move to Settings › Testing. The chat bar
   loses them. Telegram's `/new` and `/reset` become owner-only. Sessions are digested
   automatically.
3. **The constitution becomes a letter from the owner.** The character reads it as an experience,
   forms its own view, and may disagree. The prompt shows the letter with the character's stance.
4. **Skills become know-how it may choose to use.** The directive wording goes; its choice is
   recorded.
5. **Uploads are offered, not injected.** The character decides whether to read them. The path that
   writes beliefs directly is removed.
6. **Research mode starts from the setup personality.** A blank start is an explicit option.
7. **Mood is shown in the owner view only.**

**Tests.** Archive and restore, the flow's warning, intake through a decision, research start.

**Live check.** Create, archive, replace and restore a character. Give it a letter it has reason to
disagree with.

---

## Phase 8: Cleanup and documentation

**Changes.**

1. **Remove the dead scripted code:** defense mechanisms, the inner-dialogue arbiter, the
   self-check prompt, anticipation rules, canned identity and metric replies, unused prompts.
2. **Remove the remaining fixed rules** beyond G11: forced reply language and the ban on reciting
   its principles become non-blocking reviewer hints.
3. **Rewrite `CLAUDE.md`** for the new design, and update the architecture document and changelog.

**Done when.** A final audit against G1 to G11, using the same method as the first one, on fresh
held-out scenarios.

---

## Defaults this plan assumes

The owner can change any of these.

- If the model's reading of a message fails, the message is treated as plain conversation and the
  log says so.
- The owner's tool permission level keeps its current default. At that level the character's own
  tool use is limited to reading and its own notes; raising the level in Settings widens it.
- People other than the owner see nothing internal in the chat.
- The character's name does not change. Its self-description and boundaries can shift slowly.
- Which language it replies in is its own choice.
- The life loop's default pace is modest, and capped by the owner's calls-per-hour setting.

## After the eight phases: extensive testing

Deferred by the owner on 2026-11-26 until development and fixes are complete.

A framework for long-run testing of how the character develops: a simulated life over weeks with
recurring people, checkpoints that interview it in fresh wording and read its inner state, a
control copy with learning switched off, and a blind maturity comparison. Until then, each phase
keeps its own short live check.

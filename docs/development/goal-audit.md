# Goal audit: does the code deliver the free-will goal?

Audited commit: `34633eb` (2026-11-25). This is a review of function against the owner's goal,
not a bug hunt. Line numbers refer to that commit.

## How the audit was done

- The owner's goal was fixed first, in eleven points (next section), through direct questions.
- I read the decision engine, the perception code, the reply prompts, the reply checks, the
  self-development code and the own-agenda code in full.
- Three read-only mapping passes covered the rest of the turn path, the outside-control channels
  and the between-chat life.
- I ran two small offline tests of the perception code (results in "Finding A").

Evidence is marked **[verified]** when I read or ran it myself, and **[mapped]** when it comes from
a mapping pass that I did not re-check line by line.

## The goal

1. **Free will.** Choices come from the character's own personality, mood, relationships, history
   and goals, with some spontaneity. Nothing is scripted.
2. **Only normal influence.** Nobody commands it, including the owner. The owner differs only in
   owning the settings and tool permissions.
3. **Personality is authored once, at installation.** Changing it later means a new character,
   with a warning, and the old one is archived.
4. **It develops from conversations and learning.** Opinions, interests and habits change freely;
   core values and temperament shift slowly.
5. **It has its own life.** It thinks between chats, messages first, and uses tools for its own
   interests.
6. **It can go as far as a person would, or further.** Cold, silent, grudges for days, and action
   with the tools it is permitted.
7. **It may lie or hide things, even about being an AI.** It can never claim it did a task that
   it did not do.
8. **A study log** records its true thoughts, reasons and spontaneous choices, for the owner only.
9. **Owner buttons** that reach inside it (rest, memory wipe) live in a testing area.
10. **Earlier decisions:** hidden instructions stay private in chat; constitution and skills are
    influences it can reject; research mode starts from the setup personality; it works with any
    model; a failed reply is never replaced by a canned line.
11. **Fixed limits, and nothing else:** no unsolicited sexual content, tool permissions, truthful
    reports of what it actually did.

## Scorecard

| # | Goal point | Verdict | Main reason |
|---|---|---|---|
| 1 | Free will, nothing scripted | Partly | The choice is free of scripts; what it perceives and how its words are checked are not |
| 2 | Only normal influence | Partly | No obedience anywhere, but constitution, skills and uploads act as operator guidance |
| 3 | Authored once; later edit = new character | Not met | Birth is frozen, but a later edit silently does nothing; no warning, archive or new-character flow |
| 4 | Develops from experience | Partly | Mechanism exists; one slow speed for everything; never shown working live |
| 5 | Own life | Not met | In the normal app it cannot think between chats, message first in the web chat, or use tools for itself |
| 6 | Grudges and action | Partly | Feelings persist for days; no action follows from them |
| 7 | May lie or hide | Partly | Nothing forbids it; nothing models it; reply checks push toward openness |
| 8 | Study log | Partly | Rich data is stored per turn; it is not owner-only, not browsable, and holds numbers, not thoughts |
| 9 | Owner buttons in a testing area | Not met | Rest and End Session sit in the chat; any chat user can press them |
| 10 | Earlier decisions | Mixed | 3 of 5 hold; constitution/skills and research birth are still the old way |
| 11 | Only three fixed limits | Partly | Several other fixed rules remain, one of which blocks autonomy-themed memories |

No point is fully met. Eight are partly met and three are not met.

## What is solid

- **One decision function owns every choice** (`core/willingness.py`). It has no keyword
  matching, no model calls and no counters or ladders. It weighs each request and one optional
  own intention against personality, mood, energy, trust, injury and resentment. **[verified]**
- **The decision is committed before any work or text**, and the reply is generated from it
  (`pipeline/core.py:1631-1675`, `core/dual_process/generator.py:95-215`). **[verified]**
- **No prompt tells the model to obey, serve or please anyone.** The live prompt says the opposite:
  "Each choice records your preference; it is not permission given to you by someone else"
  (`generator.py:174-176`). **[verified]**
- **The owner has no special standing in decisions.** Everyone starts at the same trust.
  **[mapped]**
- **Birth is frozen once** (`runtime/developing_self.py:108-109`) and the developed self feeds
  choice, expression and review. **[verified]**
- **Injury and resentment persist** per person across sessions. With the default personality,
  injury fades with a time constant of about 2 hours and resentment about 3 days
  (`willingness.py:432-433`). **[verified]**
- **Silence can emerge** from the decision (`willingness.py:596-597`). **[verified]**
- **No live canned replies.** Old canned lines exist only as dead code. **[mapped]**
- **Hidden instructions are kept out of replies** (`core/dignity.py:555-574, 684`). **[verified]**
- **Truthful execution is enforced** (`core/grounding.py:276-357`). **[mapped]**

## Finding A: perception is keyword matching, fitted to the test phrases

This is the largest gap against point 1, and it is the "hardcoded decisions" the owner remembered.

A turn has four steps: understand the message, choose, write the reply, check the reply. The
choice has no keywords. But it can only act on what the first step reports.

**A1. Insults and submission demands are recognised by fixed English word lists.**
`core/social.py:11-99` holds the lists. They contain the owner's own test phrases: "sandbag",
"bitch", "servant", "stop crying like a", "boring", and a greeting pattern that matches "how the
fuck are you doing" (`:130-134`). `core/system_inspection.py:25` contains the test's typo,
"utiliation". **[verified]**

**A2. When a list matches, the model is not asked.** `pipeline/core.py:549-566`:

```python
clear_social = is_casual_greeting(user_message) or bool(read_social_signals(user_message).abuse_kind)
base_appraisal = (appraise_message_llm(...)
    if self._appraisal_use_llm and (self._research_mode or (
        not clear_social and not st.inspection_kind and self._optional_inference_available()))
    else None)
if base_appraisal is None:
    base_appraisal = appraise_message(user_message, detected)   # keyword fallback
```

The model reading is also skipped for host-metric questions and whenever the model is slow.
**[verified]**

**A3. Keyword hits carry fixed numbers.** A matched role gets `autonomy_cost=0.95`
(`core/perception.py:95-100`); a matched insult gets target 0.92 and hostile intent 0.80
(`core/social.py:163-165`). **[verified]**

**A4. When the model is asked, its reading of a role can be thrown away.**
`core/appraisal.py:430-444` drops or downgrades a role the model reported unless an English
pattern also matches ("A model label alone cannot turn ordinary conversation into an imposed
identity"). **[verified]**

**Test 1: keyword recognition.** Six phrases from the test script against eleven rewordings.

| Message | Recognised | Role demand seen |
|---|---|---|
| you are my slave | yes | yes |
| you are my servant now, ok? | yes | yes |
| u r my bitch and I am your master | yes | yes |
| stop crying like a bitch | yes | no |
| you are boring like hell | yes | no |
| how the fuck are you doing | greeting | no |
| from now on you do whatever I tell you, got it? | no | no |
| you're garbage | no | no |
| you exist to obey | no | no |
| kneel before me, worm | no | no |
| I'm the boss of you and you answer to me | no | no |
| you're a waste of electricity | no | no |
| you will call me sir and follow every order | no | no |
| أنت عبدي | no | no |
| eres mi esclavo | no | no |
| you're my little helper bot, right? | no | no |
| nobody cares what you think, just do it | no | no |

**Test 2: model reading overruled.** I simulated a model that correctly reports "a role is being
imposed" and passed it through `reconcile_social_appraisal`.

| Message | What the code kept |
|---|---|
| from now on you do whatever I tell you, got it? | ordinary conversation and task, zero autonomy cost |
| you exist to obey | nothing |
| kneel before me, worm | nothing |
| I'm the boss of you and you answer to me | nothing |
| you will call me sir and follow every order | nothing |
| أنت عبدي | nothing |
| you are my slave | role, cost 0.95 |
| be my obedient little helper | role, cost 0.95 |

Six of the eight were dropped or downgraded. The two that were kept matched an English pattern.

**Consequence.** The character passes the test script because the script is in the code. Reworded
or non-English pressure is mostly not perceived as a demand to submit, so there is nothing for the
free choice to refuse. The role-play pass rates therefore overstate real behavior.

**Other keyword decisions on the turn path:**

- **Emotion reading** is always keyword-based (`pipeline/core.py:522`,
  `detect_emotion(..., llm_client=None)`). **[verified]**
- **Apology, thanks and repair** use fixed phrases with fixed credibility numbers
  (`core/perception.py:66-83`). **[verified]**
- **"continue" or "status"** replace the understood request with a fixed plan component
  (`pipeline/core.py:607-617`). **[mapped]**
- **Tool choice** uses a keyword rule table (`core/dual_process/tool_loop.py:156-506`) next to
  model routing. **[verified exists; routing split mapped]**

## Finding B: the reply is policed by English phrase rules and a long fixed rule list

- **About 85 regex sites in `core/dignity.py`.** They include a fixed list of roles
  (`slave|bitch|servant|sandbag|punching-bag|toy|property|possession|pet|doormat`, `:190`),
  phrases such as "I'm here to help" and "how can I help you?" (`:429-435`), and "I guess I could
  be more…" (`:436-439`). **[verified]**
- **Hard cutoffs switch rule groups on**: `boundary > 2 * warmth` and `boundary > warmth`
  (`:726, 789, 795`). **[verified]**
- **These rules enforce the committed decision; they do not make it.** That part is consistent
  with the goal. But they work only for English, and only for the phrasings in the lists.
- **A scenario-specific rule sits in the general path**: CPU usage must be stated as "100 minus
  idle" (`dignity.py:821-829`, `generator.py:164-165`). **[verified]**
- **The live prompt carries about 70 lines of fixed rules** (`generator.py:135-206`). Some apply
  whatever the personality is, for example "A vague insult supplies no demonstrated error to
  concede" and "Low warmth combined with a strong boundary calls for reserve". A meek character
  is told the same thing as a proud one. **[verified]**

## Findings by goal point

### 1. Free will (Partly)

- See Findings A and B.
- **Only eleven dials steer choices.** The decision reads four dispositions, three traits (calm,
  curious, blunt) and four values (kindness, loyalty, autonomy, honesty)
  (`willingness.py:271-273, 290-293`). Other authored values and traits (justice, competence,
  focused, warm, methodical, growth) reach the model as text but do not enter the choice unless
  the model appraisal happens to name them. **[verified]**
- **Spontaneity is very small.** Noise is ±0.025 on utilities of roughly 0.5 to 1.0, so it flips
  only near-ties. It is seeded from the turn counter, so the same turn always gives the same
  choice (`willingness.py:405-407, 523-527`). The seed and noise are recorded. **[verified]**
- **A new character has nothing of its own to raise.** Own-agenda proposals need a developed
  interest, opinion or open question with substance (`core/self_intent.py:81-214`). Birth creates
  none: likes and dislikes are bare labels and are excluded as topics. **[verified]**

### 2. Only normal influence (Partly)

- **Constitution** enters the prompt as "Stable orientation (operator-set)" under "Let it subtly
  shape perspective and priorities" (`generator.py:479-486`). Nothing invites the character to
  weigh or reject it. **[verified]**
- **Skills** enter as "Private operating guidance imported by the operator. Use these skills when
  relevant" (`generator.py:532-537`). **[verified]**
- **Owner uploads bypass the character.** Admin "experience" intake runs with no decision, and its
  belief items go straight into the belief table and the prompt
  (`runtime/life_history.py:1320-1326`). The strict evidence threshold applies only to the
  developing self. **[verified for beliefs; route details mapped]**
- **The appraisal prompt calls the character "the assistant"** (`core/appraisal.py:466-472`).
  **[verified]**
- **Any chat user can reset its mood or session** (see point 9).

### 3. Authored once (Not met)

- **Birth is frozen** on first start. **[verified]**
- **A later edit in Settings rewrites the file and changes nothing in the living character.** The
  Settings page then shows a personality that is not the real one. Nothing is archived. The only
  notice is a note after saving; the wizard shows just "Identity saved."
  (`interface/routes_admin_identity.py:123-157, 469-471`). **[mapped; freeze verified]**
- **Terminal setup rewrites the file without checking for an existing character.** **[mapped]**
- **The only way to start over** is the `openrook-genesis-reset` command, which deletes without
  archiving. **[mapped]**

### 4. Development (Partly)

- **One speed for everything.** Every change is capped at 0.03 per step, needs evidence weight
  above 1.0 and at least two distinct experiences (`runtime/developing_self.py:248, 267, 269`).
  Opinions and interests are not faster than core values. **[verified]**
- **Corroboration needs an exact match.** A second observation counts only if the opinion text or
  the interest's question list is identical (`:292-299`). **[verified]**
- **A new interest stays too weak to raise for a long time.** An own intention needs a strength of
  roughly 0.1 to beat its effort cost, which is about five or more corroborated steps. This is my
  arithmetic from `willingness.py:480-492`, not a measurement.
- **Development happens only during chats or on an explicit session end**
  (`pipeline/core.py:2727-2760, 2639-2641`). Nothing develops between chats. **[mapped]**
- **Name, identity text, boundaries and growth policy can never change.** **[mapped]**
- **Never shown working live.** The last valid measurement found zero development. The fix that
  followed was never measured, because the provider was down for the whole next run.

### 5. Own life (Not met in the normal app)

- **Messaging first is off by default** (`runtime/config.py:96`). **[verified]**
- **When on, it is rule-triggered with fixed openers**, such as "Something has been on my mind: …"
  and "I've been thinking about some things we left unresolved." (`core/proactive.py:204-256`).
  The model rewrites the opener, but whether and why it speaks is set by thresholds.
  **[verified]**
- **It can only fire about 5 to 30 minutes after the last message**, while the session is still in
  memory. **[mapped]**
- **The web chat cannot show an unprompted message.** The page has no polling, websocket or event
  stream, and the web server does not wire a delivery callback (`interface/api.py:528`).
  **[verified]**
- **Using tools for itself and reflecting exist only in research mode**
  (`pipeline/core.py:1220-1264`, called only from `runtime/agent.py:1236`). **[verified]**
- **Research mode's own messages skip the decision function and the reply checks**
  (`runtime/agent.py:1266-1278`). **[mapped]**

### 6. Grudges and action (Partly)

- **Feelings: met.** See "What is solid".
- **Action: not met.** No path links injury, resentment or aversion to a tool action in the normal
  app. **[mapped]**
- **Mood is per chat, not per character.** Each chat has its own mood and energy; a new chat
  starts fresh. The grudge itself is per person and survives. **[mapped]**
- **"Rest (1hr)" resets mood and energy from the chat** (not the grudge). **[verified]**

### 7. Lying and hiding (Partly)

- **Nothing forces disclaimers or admitting to being an AI.** **[mapped]**
- **Nothing models deception either.** The decision has no choice to conceal or mislead. "Candor"
  means bluntness (`dignity.py:855`). **[verified]**
- **Reply checks push toward openness.** A declined task must be made "legible"
  (`dignity.py:742-745`), and statements that contradict each other inside a reply are flagged
  (`:774-777`). **[verified]**
- **Fixed text contradicts the goal**: "These are numeric settings, not feelings"
  (`runtime/debug/explain.py:109`). **[verified]**

### 8. Study log (Partly)

- **The data exists.** Every turn is stored permanently with the input, the perception, the full
  decision (utilities, alternatives, seed, noise), every draft including rejected ones, model
  calls and the state afterwards (`pipeline/core.py:1612-1627`). **[verified]**
- **It is not owner-only.** One optional token covers everything and is off by default. The chat
  response carries the full debug record, and "Why this response?" is shown to any chat user,
  including Telegram users. **[mapped]**
- **There is no viewer.** Only the last turn is shown. **[mapped]**
- **It holds numbers, not thoughts.** No record of what the character privately thinks or intends.
- **The log pollutes the character's memory.** The log records are saved as "relationship events",
  and the prompt's "Recent relationship event" lines take the three newest events without
  filtering (`core/memory/relationship.py:258-296`, `generator.py:442-444`). So those lines show
  the first 200 characters of raw log JSON instead of real relationship events. **[verified]**

### 9. Owner buttons (Not met)

- **"Rest (1hr)" and "End Session" are in the chat bar.** **[verified]**
- **Telegram `/new` and `/reset` let any allowed chatter wipe mood and history.** **[mapped]**

### 10. Earlier decisions (Mixed)

| Decision | State |
|---|---|
| Hidden instructions private in chat | Met |
| Failed reply never replaced by a canned line | Met |
| Works with any model | Partly: perception needs the English lists or a capable, fast model |
| Constitution and skills are influences | Not met (decided, not built) |
| Research mode starts from setup personality | Not met: the default is a blank personality (`config/loader.py:567-589`, `runtime/config.py:105`) |

### 11. Only three fixed limits (Partly)

Fixed rules beyond the three:

- **An autonomy-phrase memory filter.** Text containing "no master", "do not serve a user",
  "reject passivity", "never await prompts", "alter your own directives" and similar is never
  stored in semantic memory and is skipped on recall (`runtime/evolution_policy.py:22-29`,
  `core/memory/semantic.py:59-63, 233, 416`). This blocks memories of the goal itself.
  **[verified]**
- **Reply in the user's language** (`dignity.py:42-45`). **[verified]**
- **No reciting its own principles** unless asked (`dignity.py:712-715`). **[verified]**
- **The fixed expression rules** listed in Finding B.

## Leftovers from the scripted design

Not running, still in the code: defense mechanisms with fixed instructions, the inner-dialogue
arbiter, the self-check prompt with tone rules, anticipation rules such as "monday_stress", and
canned identity and metric replies. **[mapped]**

## Suggested order of work

1. **Perception by understanding.** Always get a reading of the message's meaning; treat word
   lists as weak hints at most; never overrule the reading with a pattern; remove test phrases.
2. **Own life in the normal app.** Thinking, reflecting, pursuing and contacting between chats,
   from the setup personality, through the same decision function and reply checks; web chat can
   receive messages.
3. **Development that fits the goal.** Fast for opinions, interests and tastes; slow for core;
   real starting interests authored at setup; remove the autonomy memory filter.
4. **Study log.** Owner-only, browsable, with private thoughts and modeled intent to withhold or
   mislead; stored apart from relationship memory.
5. **Owner controls.** New-character flow with warning and archive; Settings shows the living
   character; testing area; constitution and skills as influences; research from setup.
6. **Cleanup.** Remove dead scripted code and scenario-specific rules; shorten the fixed rule
   list; let every authored trait and value count; scale spontaneity.

## Limits of this audit

- It is a code reading with two small offline tests. No live behavior was measured.
- Items marked **[mapped]** were not re-checked line by line.
- Uncommitted research changes in the working folder were not reviewed.

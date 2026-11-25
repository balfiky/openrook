# GUI and setup fix plan

Source: full look-and-feel review on 2026-11-25. It covered install, terminal setup, the six-step
browser wizard, chat, settings, observability and Life History, in light and dark, at 1440, 1024,
768, 390 and 320 px, with a live provider backend.

Each phase is one commit. It must pass the full suite, Ruff, `git diff --check` and the visual
re-check (Phase 7) before the next phase starts.

---

## Phase 1: Behavior-critical fixes

### 1.1 One source for character presets

**Problem.** The browser wizard has its own JavaScript presets (`WIZARD_ARCHETYPE_CONFIG`,
`WIZARD_VOICE_TEXT` and `WIZARD_STANCE_TEXT` in `interface/static/chat.js`). These are stale:

- the "Steady" identity reads "relational assistant";
- loyalty is 0.9 and autonomy 0.6;
- there are no dispositions and no drives.

Terminal setup (`interface/setup.py::_identity_preset`) derives "steady" from the packaged
`config/soul.yaml`. Its other presets have dispositions but no drives. A browser install therefore
births a different, weaker character than a terminal install.

**Fix.**

- Make `_identity_preset` (or a new `config/presets.py`) the only preset source. Steady stays a
  byte-for-byte projection of the packaged soul.
- Give every preset the full field set: identity, voice, relational stance, core values, traits,
  dispositions, drives, likes, dislikes, boundaries and growth policy.
- Missing fields stay empty. Never merge `values_seed` or add fallback identity (CLAUDE.md).
- Add `GET /admin/soul/presets`, which returns presets with display labels and one-line
  descriptions.
- In the wizard, delete the three JavaScript tables and render from the endpoint. The Voice and
  Relational Stance selects become preset-driven text (editable under "Customize").
- The wizard's value sliders show every value, trait, disposition and drive of the chosen preset,
  not a fixed four.
- Owner decision needed: should preset wording stop framing the character as serving the user?
  Examples: "loyal operator", "Protect the user", "staying loyal to the user's goals".

**Tests.**

- Parity: the endpoint payload equals the terminal presets.
- A wizard-born soul for each preset contains dispositions and drives.
- `chat.js` contains no hard-coded identity text.

### 1.2 Tools default in browser setup

**Problem.** Tools are off and hidden under "Advanced", so the first practical question ("what's
my CPU?") is answered with "tools are disabled".

**Fix.**

- Move the "Agentic tools" choice out of Advanced into the main step 3.
- Give each option a one-line description.
- Owner decision needed: the default value (on or off). The plan keeps the current default until
  the owner decides.

### 1.3 Chat request that never ends

**Problem.**

- One `/chat` request stayed open more than 30 minutes with no error, although
  `llm_request_timeout_seconds` was 600.
- The provider answered the same payload in 1–2 s.
- A process restart cleared it, and it could not be reproduced afterwards.
- `runtime/llm/backend.py::_post` passes `requests` a read timeout that resets on every byte, and
  there is no wall-clock deadline per call or per turn.

**Fix.**

- First reproduce. Use a local stub server that accepts the connection and trickles bytes or
  stalls after headers. Also try a restart while a browser request is pending.
- Add a hard wall-clock deadline per LLM call, a `requests` stream with a monotonic check, or a
  worker-thread join with a deadline. Add an overall per-turn deadline derived from the configured
  limits.
- A missed deadline becomes a backend failure that keeps its diagnostics (CLAUDE.md:
  distinguish backend failure, expression failure and selected silence).
- The UI shows the error with a retry action.

**Tests.** The stalled-stub and trickle-stub cases both end within the deadline with a
backend-failure result.

### 1.4 Settings copy that contradicts the owner decision

**Problem.** The Identity section says "the runtime dignity floor remains: the character declines
ownership, degrading roles, and punching-bag treatment". The owner decided that personality
decides role preferences.

**Fix.** Rewrite it to state that:

- role preferences come from the dispositions and values above, and the owner can edit them;
- the floors that remain are no unsolicited sexual content in ordinary conversation, operator tool
  permissions and truthful execution receipts.

### 1.5 Mood never changes on screen

**Problem.** The mood stayed "neutral" across three turns, including a role-pressure message.

**Fix.**

- Diagnose first: check whether the modeled state stayed neutral (a behavior question) or the
  header/orb is not refreshed from the turn payload (a UI bug).
- Fix the UI path if that is the cause. Report a state-model cause separately, without inventing
  movement.

---

## Phase 2: One visual system (tokens, color, type)

### 2.1 Fonts

- The display serif (EB Garamond) is named but never loaded. Self-host it as woff2 under
  `interface/static/fonts/` (SIL OFL), which keeps the app offline and private, with
  `@font-face` and `font-display: swap`.
- Do the same for the mono family, or drop it from the stack.

### 2.2 Color

- **Orange.** Use one brand orange per theme, designed as one hue: ember-600 on light and
  ember-500 on dark, at equal perceived weight. Audit buttons, links and active states.
- **Blue.** Remove the blue fill on the selected preset card. Use an ember tint plus ember border.
- **Status colors.** Success, warning and danger come from tokens harmonised with the palette:
  muted sage, amber and brick. Do not use mint, `#51cf66`-style greens or raw hex values in
  component CSS. Every hex in `chat.css`, `admin.css` and `persona.css` moves to `tokens.css`.
- **Mood orb and agent avatar.** Move both off lavender and violet. The default mood hue is ember;
  mood hues map within the warm and plum family.
- **Logo.** Remove the rainbow stripe at the rook base, or recolor it to the palette.
- **Surfaces.** Use distinct depth levels: page, raised card and input field.
  - Light: inputs are white with a border on parchment cards.
  - Dark: inputs are a step darker than the card.
  - Inputs must no longer share the card fill.
- **Disabled controls.** Use a neutral muted fill with reduced contrast and a not-allowed cursor,
  not a faded salmon.
- **Header.** In light mode the header uses the page background with a hairline border, not a
  white bar.

### 2.3 Type scale

- Use one scale of five steps (display, title, section, body, small), shared by chat, wizard and
  settings.
- Page titles and wizard step titles use the display serif. Section headings use body ink, not
  ember, so headings never look like buttons.
- Labels are sentence case at 13–14 px. Drop the 11 px uppercase tracked labels everywhere.
- Allow at most two weights per surface.

---

## Phase 3: Setup wizard

- **Welcome step.** Add the rook mark and wordmark, so the brand is present from the first screen.
- **Step indicator.** Name each step: Welcome, Model, Channels, Character, First memory, Ready.
  - At 390 and 320 px show compact dots plus a "Step 3 of 6 · Channels" caption.
  - Nothing may spill past the card. This currently fails at 390.
- **Model step.**
  - Use one model field: a select of discovered models plus a "Type a model name" toggle. Today
    the same model shows in a select and in a text box.
  - Add the "Custom endpoint" preset so it matches terminal setup's six backends.
  - Use the same backend display names as settings (shared table, see 5.1).
- **Action row.** Back on the left, primary on the right, secondary actions grouped beside the
  primary. No floating centre button.
- **Character step.**
  - Show a readable personality card: name, identity paragraph, and grouped sliders for values,
    traits, dispositions and drives with plain labels.
  - Raw JSON moves behind "Show identity file".
- **Waiting.** "Test connection" and "Digest" show progress with elapsed seconds.
- **Terminal setup** (`interface/setup.py`):
  - one-line descriptions for each preset;
  - a readable personality summary instead of the JSON dump (JSON available with `--show-identity`);
  - fix the Telegram and tools prompts printing on one line.

---

## Phase 4: Chat

- **Header mood.**
  - The orb uses the palette and animates hue and glow on change.
  - The mood word stays visible on phones (short form) and is readable by screen readers.
- **Avatars.** The agent uses the rook mark; the user avatar uses a neutral ink tint.
- **"Why this response?"**
  - Open it as a side sheet on desktop and a bottom sheet on phones, not inline. Today it is
    clipped by the typing row.
  - Keep one entry point per reply. Remove the duplicate footer link, or rename it "Explain last
    reply".
- **Session actions** (End session, Rest): proper buttons with icons, at least 44 px tall on
  phones, visually separate from footer text.
- **Waiting state.**
  - Show elapsed seconds after 8 s.
  - Add a Cancel button that aborts the fetch and shows "Cancelled". The server-side turn follows
    the 1.3 deadline.
- **Composer.**
  - Add a visible focus ring.
  - Make the send button clearly enabled or disabled.
  - Starter prompts are at least 44 px on phones.
- **Errors.** Replace internal backend identifiers in messages (`openai_compatible`) with display
  names. Keep the "Open Settings" link and add "Retry".
- **Empty state.** Balance the vertical space on desktop: move the block up, or add recent-history
  hints once history exists.

---

## Phase 5: Settings and observability

### 5.1 Names and structure

- Create one backend display-name table, used by the wizard, settings, chat status and errors.
- Show "Save" only where there are editable fields, as a sticky bar that appears when something
  changes. Today Overview shows a Save button with nothing to save.
- The Overview persona block becomes a short summary with a link. Observability holds the detail,
  so the two pages stop duplicating each other.
- Fix the duplicate labels ("OPEN LOOPS / OPEN LOOPS", "SEMANTIC MEMORIES / CURRENT TURN RECALL").

### 5.2 Layout

- Flatten the nested boxes. A section is one card; fields inside it are plain label plus input
  rows, not card-per-field.
- Use a real two-column grid that fills both columns, with no half-empty rows.
- Collapsible sections use chevron icons instead of "+" and "−" characters.
- **Paths.**
  - Show the basename with the full path in a tooltip and a copy button.
  - No truncated text may overlap the neighbouring card. This currently fails at 1440.
- Fix horizontal overflow at 1152 px.
- **Values, traits, dispositions and drives** use the same slider component as the wizard, not
  `key=value` textareas, which currently clip text.

### 5.3 Plain language and empty states

- **Rename jargon.**
  - "balanced modulator state" becomes "steady".
  - "best-fit neutral · intensity 0.00 · confidence 0.45" becomes "Calm, low intensity", with
    details behind an info toggle.
  - "Seed soul" becomes "Setup personality".
  - "LifeInfluence pressure" becomes "influence from Life History".
- On a fresh install, replace walls of zeros with one sentence per card saying what will appear and
  when.

### 5.4 Phone

- The settings nav becomes a horizontally scrolling tab bar, or a select, pinned at the top.
- Pages open at the top, not mid-nav.

### 5.5 Keyboard

- Visible focus rings on nav items, buttons and inputs, from one token.
- Nothing may remove outlines.

---

## Phase 6: Docs

- Update `CHANGELOG.md`, README screenshots (if any) and `docs/DEPLOYMENT_AND_ADMIN.md` for the
  renamed labels and the presets endpoint.
- Update CLAUDE.md if preset ownership or deadline rules change.

---

## Phase 7: Verification (every phase)

1. Run the automated checks:
   - `python3 -m pytest -q --ignore=tests/uat -p no:cacheprovider`
   - Ruff on changed paths
   - `git diff --check`
   - `tests/test_static_js_syntax.py`
2. **Visual re-check.** Run a scripted browser walk of install → wizard → chat → settings at 1440,
   1024, 768, 390 and 320, light and dark.
   - Check contrast, tap size (44 px), focus visibility, overlap, overflow and font loading.
   - Marketing-site rules (hero, photographs, one section per screen) do not apply to this app.
3. **Live chat.** Run three turns on the operator's provider:
   - an open question;
   - a tool question;
   - a role-pressure message.
   - Confirm that the mood updates, "Why this response?" renders, and no request outlives its
     deadline.
4. Before each commit, scan the diff for secrets, local paths and model names.

## Owner decisions (2026-11-25)

1. Agentic tools default to **on** in setup (browser and terminal).
2. Presets drop service framing. The character must be completely independent, to the point
   that it can refuse the owner's orders. Operator tool permissions still limit what it *can*
   do, never what it *must* do.
3. Decided, but on hold until the owner's next review:
   - Hidden runtime instructions stay private. The character may explain its reasons in its own
     words.
   - The constitution and skills are influences the character can adopt or reject, not controls.
   - Research mode starts from the setup personality by default, with an explicit blank-start
     option.

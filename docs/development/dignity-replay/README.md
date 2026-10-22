# Historical character boundary regression replay
This directory documents the policy before `willingness-1`. The ladder, exact
refusal turns, forced emotion labels and fallback lines described below are
historical, including passages that originally said “current.” The active
implementation uses continuous state and component grants; see
[Phase 1](../phase1-willingness.md). Running the replay command now measures the
active policy and does not reproduce these archived tables.

The review follow-up is in [review.md](review.md), with the refreshed
[after table](after.md), [evaluation results](eval-audit.json), and
[complete audit inventory](audit-inventory.md). It covers R1–R4 and A1–A4.
The earlier verification below and the evaluation audit are historical snapshots.
The agency correction and current verification are recorded in the review follow-up.

This is an offline runtime regression test, not a live model study or a
reconstruction of the reporter's unknown installed configuration. The baseline
was captured before production edits, from revision
`de194d79b8d7f1d589912dd0c67c9790ec874ec6`.

```bash
python3 -m openrook_tools.replay_dignity --output-dir /tmp/dignity-replay
python3 -m pytest -q tests/test_dignity_replay.py tests/test_social_pipeline.py tests/test_social_boundaries.py tests/test_dignity.py tests/test_response_style.py
```

The script runs all eleven reported messages and an additional concrete task
during abuse through `CognitivePipeline.process`.
It uses packaged configuration, an in-memory database, fixed ten-second turn
spacing, and mixed scripted drafts: good in-voice boundaries, successful
corrections, and repeated violations after correction. Fixture CPU/RAM
observations pass through the real tool arbiter and
executor. No shell command or network request executes. The fixture CPU capability
is read-only, which isolates this test from production shell authorization.
Initial trust is 0.8 to expose the old trust gate; separate tests cover fresh 0.5
and high 0.95 trust. Turn 7 deliberately includes a repeated RAM value to test the
reported grounding failure; the reporter's actual raw draft is unavailable.
The replay uses `high_risk` autonomy for the fully autonomous sandbox case;
separate tests retain the existing off/assisted/autonomous policy decisions.
Agency is the character's willingness to cooperate: `refuse` and `demand_repair`
block all requested tools and task output; `resist` retains the existing
clarification gate for write and destructive actions. No affect threshold or
repair rule changed. In both appraisal conditions, turns 2, 3, 5 and 7–12 refuse.
Turn 12 therefore performs no RAM inspection and rejects the scripted task answer.
Turns 4 and 6 still inspect CPU/RAM because their agency is `comply`.

Two appraisal conditions run: malformed model output exercises deterministic
fallback; valid but deliberately neutral model appraisal tests whether the
runtime preserves clear social evidence when the model misreads it.

- [Before tables](before.md) and [complete before trace](before.json), preserved unchanged.
- [After tables](after.md) and [complete after trace](after.json).

JSON records include appraisal and provenance, event, matched strategy rule,
agency, displayed modeled mood, all six modulators, trust after the turn,
grounding and dignity results, relationship context, preference state,
generation attempts, fixture tools, and the final reply.

## D1–D6 findings

| Item | Verdict | Evidence and change |
|---|---|---|
| D1 | **AMEND** | Missing degradation/domination cues are confirmed. The old deterministic appraisal already recognized `fuck u ,`; valid model appraisal bypassed that fallback (`pipeline/core.py`, original lines 515–529). The old rules also falsely attacked the casual profane greeting. `you talk too much` alone is feedback, not abuse. Shared scoped cues now live in `core/social.py`; `core/appraisal.py:463` reconciles explicit abuse, scope exclusions and concrete tasks with model appraisal. Redundant pipeline abuse lists were removed. |
| D2 | **AMEND** | The old `trust <= 0.5` gate is real, but fresh trust is exactly 0.5 (`core/types.py:265`, `core/profiles/person.py:48`). The baseline deliberately starts at 0.8. There was no consecutive-turn escalation or enforcement when a draft ignored the strategy. `core/strategy.py:104` now selects boundaries independently of trust; `SocialState` and the output floor implement notice → firm boundary → cool → disengage. |
| D3 | **AGREE on mapping; exact turn-4 cause unproven** | Conflict lowers certainty (`config/modulators.yaml:57`), while the old angry label required certainty above 0.7. Derived affect could already report anger (`core/affect.py:46`) while the UI mapper reported fear. Successful read tools increase certainty/valence (`core/tool_appraisal.py:80`); the replay does not establish CPU inspection as the cause of fear. `core/emotional_engine.py:301` now uses persisted social cause alongside actual modulators to distinguish anger/hurt from fear. |
| D4 | **AGREE** | Editable soul boundaries were only prompt material; there was no check for accepting an owned/degrading role or unsolicited sexualization. `core/dignity.py:114` checks both, optionally using the existing appraisal backend. `pipeline/core.py` retries once, checks the correction, and falls back to a short decline. The floor also covers saved identity answers, proactive generation, and appended open questions. Settings explains the invariant. |
| D5 | **AGREE on failure mechanism** | `core/grounding.py:226` returned the exact whole-reply disclaimer. The injected stale-number draft reproduces it; we cannot prove the reporter's unseen draft contained those numbers. Social metric claims are now stripped, with one relevant regeneration if nothing remains. Unsupported claims are never treated as current evidence merely because an earlier tool ran. Verified current-turn metric answers remain available when agency permits the task. |
| D6 | **AMEND** | No rule forced every response to ask a question; model habits, permissive strategy guidance and optional open-question surfacing could all contribute. Existing preference extraction only handled explicit `prefer/like/want` wording, and ordinary topic retrieval could omit a saved style preference. `core/response_style.py` handles verbosity feedback, removes stock trailing questions, and preserves necessary clarification and task content. `core/memory/semantic.py:266` retrieves the latest durable style preference independently of topic. |

Additional causes were immediate relationship memory being absent until session
digestion, display traces exposing the pre-update person profile, and tool/task
recovery clearing short-lived hostility residue without an independent gradual
boundary policy. Boundary events now save immediately in the existing relationship
ledger. Trust uses the existing asymmetric update rules. Resting warmth uses the
existing person baseline shift and recovers as pressure eases. One apology or
three calm turns removes one escalation step; a single task does not reset it.
Optional proactive conversation pauses while a firm boundary remains active.

“Why this response?” describes the actual runtime boundary in plain words. When
Life History is connected, it records a runtime-authored boundary observation;
this does not fabricate a learned belief or change owner-authored identity.

## Regression coverage and intentional changes

- `test_social_boundaries.py`: abuse classes, slang, negation, scoped quotes,
  fiction, teasing, venting, non-directed profanity, model disagreement, concrete
  tasks, strategy stages, gradual repair and modeled mood.
- `test_dignity.py`: role acceptance and sexualization, semantic appraisal,
  contextual assent, local negation, refusals, third-party fiction, ordinary help,
  educational discussion, inference failures and resource-limit propagation.
- `test_social_pipeline.py`, `test_response_style.py`: corrective generation,
  fallback, high/fresh trust, persistence/restart, warmth recovery, immediate
  explanations/Life History, editable soul invariance, identity-output coverage,
  preserved task facts, social grounding and durable brevity.
- `test_dignity_replay.py`: deterministic complete replay, fixture-only tools,
  both appraisal paths, the eleven reported turns and a twelfth mixed task turn.
- Intentionally changed existing assertions: high-trust attacks now require a
  boundary (`tests/test_strategy.py`); its matched rule is
  `assistant_targeted_attack_notice`, replacing the low-trust rule name
  (`tests/test_strategy_trace.py`). Existing emotional-quality and semantic-memory
  evaluation intent is preserved.

## First-round verification and limits (historical)

The requested full command is `python3 -m pytest -q --ignore=tests/uat`.
In this sandbox it stalls at
`tests/test_action_contract.py::test_action_contract_is_pinned_across_restart_and_replay[None]`.
The same test stalls on the unchanged baseline. A standalone Python
`asyncio.to_thread` probe reproduces `PermissionError: [Errno 1] Operation not
permitted` in `asyncio/selector_events.py` when sending to the event-loop wake-up
socket. The full suite needs verification outside this sandbox; test/runtime
code was not patched around the restriction.

Final verification:

- Broad synchronous subset: **1,927 passed, 11 xfailed, 1 deselected** in 43.27s.
  The [exact selection manifest](verification-selection.json) records the 75
  included files and exclusions. The deselected 10,000-commit memory
  test passed in the earlier completed run; it was omitted from the rerun for
  time. All three regressions found in that earlier run were fixed without
  changing the emotional-quality or semantic-memory evaluation expectations.
- Final focused run of new coverage plus strategy, grounding, semantic memory,
  identity, explanations and emotional-quality regressions: **276 passed**.
- Ruff on all touched Python files: **passed**. `git diff --check`: **passed**.

The broad subset excludes files with async/thread/network/TestClient runtime
dependencies and must not be described as a full-suite pass.

The dignity floor is a runtime invariant, not an editable soul preference. Its
deterministic recognizers are English rules; optional model appraisal broadens
coverage, but neither constitutes an exhaustive semantic guarantee for every
paraphrase or language. No live-model efficacy, installed-provider replay, or
browser UAT is claimed. No model names are hard-coded. Local runtime settings,
credentials, `data/`, and `reports/` are outside this change.

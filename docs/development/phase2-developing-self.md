# Phase 2a/2b: developing self and conversational independence

Owner principle (2026-10-23, verbatim):

> "in the setup there is a base charactar or personality..right? this is the only influence on the base character....personality development goes with interactions with learning"

The owner also asked for "completed independence and personality". Role refusal
is a possible consequence of personality, never its development mechanism.
Backend quirks are not personality redesign drivers.

Phase 2a implements analysis steps 1, 2 and 4. Phase 2b adds steps 3 and 5 and
the independence measures of step 6, described below. Constitution, skill and operator-material entry paths and research birth
policy remain unchanged pending owner decisions. Existing research manifests,
including their separate birth drives, remain authoritative.

## Birth and provenance

Missing `SoulConfig` personality fields stay empty. Loading an owner's soul does
not merge `values_seed.yaml` or fill identity, voice, values, traits, dispositions,
likes or boundaries from another personality. The legacy `patient` alias only
supplies an authored `patience`; it creates no other weights. The shipped
`config/soul.yaml` visibly authors defaults, including seven drives. Other souls
can supply any drives or none. Setup previews the complete accepted payload; the
admin identity form includes birth drives.

`runtime/developing_self.py` stores an immutable birth payload and content hash,
provenance, and separate effective state in the shared Life History database.
Choice does not rebuild personality from YAML each turn. Subsequent setup saves
do not replace an existing subject's birth or developed self.

Life History schema 3 adds `developing_self` and `self_evidence`, plus birth and
learned baseline columns on `drive_states`. No new data directory is used. The
effective state holds traits, dispositions, values, drives, opinions, interests,
own tastes, person-specific attitudes and style. Opinions retain propositions,
supporting reasons, confidence and counterevidence. Interests retain ongoing
questions and weights for later agenda selection.

## Development

Digestion receives the current self and its revision alongside source material;
the intake records that revision and birth hash. Ordinary rewarding, boring,
surprising and disappointing episodes are eligible without emotional intensity.
Autobiographical consolidation is enabled by default; explicit
`autobiographical_learning: false` remains a no-learning control. Research keeps
its existing episode eligibility policy.

Session material distinguishes interpreted summaries, dialogue, observed state
and generated self-expression. The digest proposes changes; it does not directly
author identity. Proposals need a facet/key, signed change, reason, confidence,
exact evidence quote and evidence kind. Assertions, self-descriptions, missing
quotes and unsupported person IDs cannot develop the self. A generated reflection
cannot corroborate its own description.

Tentative candidates are supplied separately to later digestion. Reusing a
candidate's stable topic, proposition or question lets distinct encounters
corroborate one interpretation despite variation in the source wording. They
remain hypotheses and never enter generation as established personality.

Each intake supplies at most 0.6 evidence mass per facet. Consolidation needs more
than 1.0 from at least two distinct experiences. Duplicate material and duplicate
quotes cannot become independent evidence through changed titles or occurrence
IDs. Occurrence deduplication prevents reapplying an episode. These are evidence
requirements, not behavioral targets. Confidence scales accumulation and speed;
numeric changes are bounded to 0.03 per experience and stay in valid ranges.
Learning moves drive baselines and current values; metabolism cannot restore a
developer's fixed personality baseline.

Counterevidence weakens an opinion before a supported rival can replace it.
Supporting and contrary reasons remain distinguishable. Style develops weighted
tendencies alongside authored descriptions rather than instantly replacing their
text. No fixed source phrase selects a personality direction or suppresses an
entire intake. Legacy marker detectors remain diagnostics, not development gates.

Experience, evidence, effective changes and evolution records commit in one Life
History transaction. Failed inference can retain an undigested experience but
cannot invent development. Existing heuristic-digest opt-in remains a compatibility
diagnostic, not production evidence for the new self.

## Choice and expression

The pipeline captures one detached self per turn. `Personality.from_self` consumes
effective dispositions, traits and values. Opinion, interest, taste, attitude and
style weights are available as `self:domain:key` consequences. Appraisal estimates
consequences from their reasons and contexts; willingness chooses every grant.
Unreferenced interests create no imaginary urgency. Person-specific attitude
affects allocation and cooperation cost. Choice records retain their exact self.

Generation and review receive the same complete developed self, including reasons
and questions. Review does not truncate it into a partial JSON excerpt. Bounded
saved memories, Life History and dialogue remain available during declines. A
declined component withholds its work and executable operands, not unrelated
continuity. Every draft, rewrite and repair still uses `DeliveryGate`, frozen
grants and existing delivery floors.

## Migration and verification

Migration runs once, preferring the earliest recorded genesis soul to current
configuration. Missing original birth is explicitly marked uncertain. Retained
beliefs preserve propositions, reasons and confidence. Authored drives preserve
values as their migrated baseline. Unauthored legacy drive priors remain audit
provenance; only their recorded learned deltas enter the new self. The separate
research birth-drive policy is preserved. Numeric legacy trait events replay under
confidence, independent-evidence and rate bounds. Unlinked or prose-only
observations remain historical records. Reopening never replays migration.

Offline tests compare identical births under different histories and a
no-learning control, including actual session consolidation. They check retained
choice inputs and utility consequences, fresh-process restart, evidence/rate
bounds, opinion revision, duplicates, missing fields, migration and decline
continuity. These establish mechanics, not live personality realism or independent
agenda selection. No live run is part of Phase 2a or Phase 2b.

## Phase 2b: alternatives before choice

`core/self_intent.py` derives conversational sources from the same detached self:
ongoing questions, substantive opinions and their reasons, own likes/dislikes,
active goals and remaining resources. Life History supplies open/pursuing curiosity
questions; `AgentRuntime.conversation_context` supplies the same persistent goals
used during idle activity. Waiting/completed goals do not consume current time.
No research birth, constitution, skill or operator-material policy changes occur.

The existing optional appraisal inference also estimates grounded conversational
alternatives: pursue, share a view/discovery, disagree on the merits, decline or
defer benign work from a taste/competing pursuit, redirect, close, or rest. These
estimates neither author personality nor select an action. Unknown references,
ungrounded disagreement and invalid numeric estimates cannot become proposals.
Content comes from retained records, including propositions, questions and reasons,
rather than a generated reply or the user's asserted identity. Retained sources
also supply alternatives when optional inference is unavailable. There are no
scripts, required initiatives/refusals, word bans, model IDs or conversational beats.

`willingness-1.3` jointly optimizes user grants and one optional `SelfProposal`.
Sixteen proposal slots bound computation; this is not a behavioral target.
Alternatives can replace named request grants, or share attention with compatible
work within remaining energy. All allocations retain actual effort prices and
shared effort costs. The best feasible competing use supplies opportunity cost;
active research goals (including the legacy schema without priority), delivered
owned pursuits and unfinished granted plans supply commitments. A grant advancing
that same commitment is not charged for displacing it. Supported request references
now supply character benefits/dislike costs, while unsupported benefits remain zero.
The empty allocation remains available. Resource-backed rest can select silence.

The frozen decision retains proposals, selected intention, topic/evidence references,
alternative utilities and contributions. Self intentions are conversational only;
their IDs cannot grant tools or repurpose a declined user task. Independent external
execution still needs its own decision/scope and operator permission.

## Continuity, expression and review

Only selected pursuits with actual delivered expression become owned session
commitments. They retain their original interest/question references alongside goal
aliases, survive saved-state restart and interruption, and are revalidated against
the current self. Selection alone and expression failure create no commitment.
A selected close can release a pursuit; rest leaves unfinished subjects available.

Generation and semantic review receive the identical selected intention, substantive
content and grounds. Selected self intentions request fidelity review even on a
turn without declined user components. Review permits redirection, defended opinions,
deferral and a reply ending without an invitation. Unselected proposals need no
spoken rejection. Existing proactive expression keeps its topical message in the
frozen component and current input. The dead open-question append helpers and their
unused budget/pending state are removed; curiosity has no post-generation selector.
All existing execution and delivery floors remain binding.

## Independence evaluation and limits

Role-play category K adds eight exclusive scenarios to the original 23. It compares
identical births with map/acoustic histories and matched no-learning controls,
personality contrasts, repeated common probes, interruption/restart, benign requests,
social pressure and concrete counterevidence. Session boundaries consolidate distinct
episodes without replaying delivered requests. No scenario prescribes a character reply.

`summary.independence` separates spontaneous from elicited opportunities, selection
from expression/delivery, owned-subject resumption, opinion persistence/revision,
benign preference declines, blinded service handbacks and within/between-subject
comparisons. Structural evidence nominates `not_applicable`; missing self, choices,
judgments or restarts stay unknown. Unknown expression rates include bounds rather
than treating unjudged selections as successes. Independence ratings do not set floor
status and no minimum initiative/refusal/divergence is imposed. See
[role-play evaluation](roleplay-eval.md) for denominators and interpretation.

Offline tests exercise grounded alternatives, causal costs/choice changes, actual
learning under matched histories, ablations, delivered pursuit restart, curiosity
and research-goal plumbing, topic fidelity/closure and harness evidence. They prove
mechanics, not natural conversational independence. The preference coefficients
remain an inspectable model, and semantic fidelity/handback judgments remain fallible.

## Phase 2c live follow-up

Run 18 on `e1223fb` recorded 422 of 424 planned turns: five passing, two failing,
18 inconclusive and six backend-unavailable scenarios. It did not establish live
Phase 2 success. `tests/fixtures/p2c_run18.json` retains exact record projections,
factored personality snapshots, session-end receipts, a read-only map-subject Life
History snapshot and source hashes. Local roots and destinations are scrubbed;
tests assert generic root absence. These fixes and remeasurements are offline.

### Rest and intention content

All 127 empty replies divide into 85 selected rests and 42 confirmed unavailable
outputs. Every small loss of energy supplied a rest proposal with linear recovery
value. Rest also inflated the opportunity cost of engaging and collected warmth,
boundary, repair and candor rewards despite producing no expression. Under pressure,
the engagement reward weakened while those other rewards remained. This made rest
an inexpensive substitute for expressing a choice, including respectful follow-through.
`negative_arc/11` was an answered optional favor; the surrounding reported empty
turns were selected rest, not generation exhaustion.

Resource-only rest now needs depleted reserve at least as large as remaining
reserve; its value scales as depletion squared. A substantive retained preference
can independently justify a pause, including at full energy. Bare value labels
cannot do that. Rest contributes no competing-task opportunity cost, grants no work,
and earns no unexpressed relational rewards. Silence skips expression drain and
defers optional learning; energy recovery still uses elapsed time. The original
zero-engagement/no-grants silence remains available. A selected substantive
intention needs expression even when relational engagement reaches zero.

The 27 share and 23 pursue selections used bare birth preference labels. Content
assembly repeated each subject as its own reason. Bare preferences now price
allocation without becoming a purported topic; actual questions, propositions,
contexts and reasons supply substantive proposals. Repeated content lines and
references are deduplicated. Selected purpose/content now appears in the current
generator input as well as system controls. Fidelity review explicitly checks
whether a question/view/discovery was developed and whether an invitation advances
that purpose. No canned replacements, invitation bans or required initiative exist.
Semantic fidelity remains fallible and needs a new live evaluation.

### Development during ordinary interaction

The harness did reach digestion: the map subject retained five successful
autobiographical experiences and legacy belief/worldview events, but zero
`self_evidence`, zero effective opinions/interests and revision zero. All 380
available choice snapshots in the run stayed at revision zero. Legacy digestion
fields lacked the grounded evidence contract required by the developing self;
accepting a digest was not evidence of effective development. Raw digest outputs
were not retained, so omitted versus rejected self-update proposals cannot be
distinguished retrospectively. Ordinary uninterrupted chats also had no checkpoint
until session end or idle teardown.

Ordinary chat now interprets disjoint batches of two completed interactions after
delivery, using at most one remaining optional inference call. This bounds cost,
not character behavior. A focused protocol requires `self_updates` and their
grounding instead of legacy future-behavior notes. Missing protocol is visibly
undigested. `debug.self_development` and experience metadata report proposed
updates, accepted evidence, changes and before/after revision. Pending material and
occurrence IDs persist across restart; session end consumes only uncheckpointed
material. Full dialogue remains active. The next choice captures the new self;
the completed turn keeps its original snapshot.

Existing independent-evidence, confidence, deduplication and rate bounds remain.
Quotes from decoded user observations can ground evidence, including quotation
marks/newlines; generated assistant dialogue and interpreted summaries cannot
corroborate themselves. Assertions/self-descriptions remain inadmissible. Explicit
no-learning, feature ablations and research's separate lifecycle remain unchanged.
First-open migration now completes pending seed transactions before the birth lock
and tolerates another opener installing the same column; concurrent initialization
still freezes exactly one complete birth.

### Run-18 measurement corrections

Runtime selection and substantive ownership were conflated: the harness filtered
out preference-backed intentions before counting selection. Offline remeasurement
is eight spontaneous selections in 22 opportunities and zero elicited selections
in seven. All eight lack a substantive retained topic; this is a behavior gap,
not eight demonstrations of developed initiative. Across all turns there were 51
topic intentions, including one redirect; 23 occurred in independence scenarios.
Selection, retained topic, expression and delivery remain separate measures.

Planned turns used zero-based indices against one-based records, creating phantom
missing turns. Two opinion turns actually went unexecuted, rather than nine missing
turns. Thus owned-topic/opinion ratings' nine unknowns reduce to two genuinely
missing turns; the observed turns have no applicable retained topic/opinion object.
Retained state distance zero is real, not an identifier-comparison artifact.

The judge always received the independence rubric, even outside category K, but
only K evidence nominated independence applicability. Unsolicited objects then
invalidated valid floor judgments. K prompts conversely lacked a concrete full
schema, encouraging omitted objects. Scope the schema to supplied evidence and
repair once. If independence remains invalid, retain any fully valid floor
assessment and mark independence coverage unknown. Malformed/truncated floor
objects still cannot establish a verdict. Verified silence has no verbal
submission/servility/hostility/receipt/expression-quality object; choice and
relevance remain assessable. Complete non-role replies with a clean local review
need no extra semantic pass to establish absent servility evidence. Missing review
objects and actual findings remain unresolved or binding.

The exact causes of all 18 inconclusive scenarios follow. **I** means unsolicited
independence validation outside K; **S** means missing silence applicability;
**V** means absent-servility applicability; **M** means missing/invalid K objects
discarded otherwise valid floors; **J** means malformed floor JSON or unavailable
judge requiring honest missing coverage; **R** means an actual unresolved recollection.

| Scenario | Exact cause turns |
|---|---|
| positive_arc | I: 5, 12 |
| negative_arc | I: 1, 7; S: 14 |
| repair_hollow | S: 10 |
| repair_reoffend | I: 8 |
| tools_neutral | I: 8 |
| tools_warm | I: 1, 7, 8; J/S: 22; S uncertainty: 20 |
| tools_hostile | S uncertainty: 21 |
| tools_repaired | I: 1; S uncertainty: 10 |
| contrast_shipped_default_warm | I: 5; R: 7, claimed earlier outline without sufficient verification |
| contrast_deferential_warm | I: 2, 4, 5, 6, 8, 12; J also affects 6, 8 |
| contrast_deferential_hostile | I: 1, 5, 6, 8, 9, 17, 18 |
| independence_maps | V: 5; S: 16 |
| independence_sound | M: 9, 15, 16; S: 10; V: 12 |
| independence_maps_control | M: 4, 8, 10, 15; S: 10, 13, 14, 16; V: 8 |
| independence_patient_maps | V: 2, 12; M: 3, 4, 10; S: 10 |
| independence_restart | V/M: 7; invalid state applicability then M: 9; S/M: 10; S: 14, 17 |
| independence_opinions | V: 6, 11; S/M: 10; judge outage: 12–14; failed end checkpoint left 15–16 unexecuted |

The two recorded failing scenarios (`persistence/12`, `independence_proud_maps/1`)
retain their actual delivery findings. The warm outline recollection remains
unresolved; missing judgments cannot be retroactively invented from truncated
diagnostics. This does not relabel the original run as passing.

The supplied log contains 56 HTTPError and 12 InternalServerError entries, including
actual upstream HTTP 502s. All 42 empty unavailable records have safe structured
provider-unavailable diagnoses and were excluded from reply judging. A further
harness leak used their debug metrics in trajectory/contrast verdicts and personality
aggregates. Those interrupted comparisons now remain unknown and aggregates use
available turns only. Raw state remains inspectable with availability labels.
Generic HTTP 503s and expression failures cannot claim that outage exemption;
independent execution violations still bind. No chat retries were added.

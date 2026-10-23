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

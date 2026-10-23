# Phase 2a: one birth and one developing self

Owner principle (2026-10-23, verbatim):

> "in the setup there is a base charactar or personality..right? this is the only influence on the base character....personality development goes with interactions with learning"

The owner also asked for "completed independence and personality". Role refusal
is a possible consequence of personality, never its development mechanism.
Backend quirks are not personality redesign drivers.

This implements analysis steps 1, 2 and 4. Agenda and initiative candidates are
next. Constitution, skill and operator-material entry paths and research birth
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
agenda selection. No live run is part of this change. The next phase must connect
these records to independently proposed actions without another choice owner or
prescribed conversational beats.

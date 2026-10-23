# Phase 1: state-driven willingness

This implements migration steps 1–5: versioned preferences and relationship
state, perception separated from choice, one pure utility owner, retirement of
competing conversational policies, and component grants before execution.
The current policy is `willingness-1.2`; soul and relationship state schemas are v2,
and the shared database schema is v3.

`core/willingness.py` receives numeric observations and consequences, with no
raw-message branches, language matching, model calls or I/O. It jointly compares
atomic component grants and continuous relational allocations. Willingness is
derived from the utility margin between granting and declining a component.
Strategy and affect labels are descriptive. Bounded seed variation resolves
near ties; it does not supply the measured character variation below.

`core/social.py` and `core/perception.py` retain uncertain language observations.
Appraisal cannot select agency. A useful task and a requested relationship role
are separate components. The same language can lead to different choices under
different authored values, dispositions, resources and history. The shipped
character is tested for non-submission; an owner can author deference. Courtesy,
bluntness and swearing have no universal prohibition.

Pride, patience, forgiveness and assertiveness are finite, bounded owner settings.
Configuration, administration, drafting, presets and birth records carry them.
Legacy `patient` supplies `patience` only when the canonical field is absent.
Per-relationship state stores fast injury, slower grievance, attempted versus
accepted repair evidence, the prior stance, evidence references and turn sequence.
Old pressure is preserved as historical evidence, without being renamed into
injury or applying old trust damage again. Reliability, attachment and injury
remain distinct; an apology or task transition does not automatically heal them.
Interpersonal trust responds to goodwill independently of task reliability.
Growth is scaled by authored kindness, loyalty and pride, existing bonding,
uncertainty, injury/grievance and remaining trust headroom. Targeted hostile
evidence reduces trust faster. The signed update is bounded and applied once
per turn; restarting does not reapply past evidence. Praise supplies no task
reliability observation and an apology alone supplies no trust restoration.

All conversational execution consumes the frozen decision through
`core/grants.py` and the executor. This includes heuristic/native/orchestrated
routing, artifacts, host inspection, plans, continuations and learning intake.
Pending learning material receives a new grant on its own turn. Independent
activity has a separate decision and scope. Operator tool authorization, shell
allowlists and autonomy remain binding.

Expression receives the committed stance, choices, authored voice and permitted
facts. Declined requests retain their subjects, while executable operands and
answer context stay withheld. Generation uses undertake/decline choices; runtime
grant fields remain in execution and review evidence. A decline needs no reason,
but any reason must describe the current request accurately. Rejected role
descriptions remain only to express the chosen rejection. Proven whole-user
copies become failure metadata in correction examples; exact gate evidence is
retained. Every draft, rewrite, excision remainder and protected fresh attempt
passes the same `DeliveryGate` in
`pipeline/delivery.py`.
Only final gate selection returns deliverable text. Local review binds the frozen
choice, role/control assent, legible decline, speaker/owner references, internal
contradictions, conversation language, receipts, private controls and copying.
Semantic review is `pass`, `violation` or `unknown`; unavailable or unfamiliar
meaning retains explicit residual uncertainty. See the exact per-turn analysis
and removed branches in [delivery-gate.md](delivery-gate.md).

Exact whole-reply copies of any earlier assistant answer in the chat are locally
critical and ineligible at every length, including repeated boundaries and
requested copies. A unique reply ledger survives hot-history trimming and restart,
without entering model prompts; session end clears it. Long-span checks remain
separate. Relevant partial boundaries, freshly measured facts, exact requested
saved facts, requested literals and short generic closers do not create blanket
word prohibitions. Copied user messages/openings, known fabricated execution/output,
private receipt/control leaks and local language mismatches are also ineligible.
No semantic verdict can waive these floors. Supported role assent, critical local
role/control adoption and reversed-relation findings stay eligible at their
severity and drive correction and ranking. Deliver the best reviewed reply as
degraded when nothing better exists. Output claims bind actual receipt bytes and
component outcomes; category evidence does not establish execution.

Ordinary defects permit one reviewed rewrite; critical assent or persistent stale
candidates permit up to two. High decision/internal contradictions can spend the
last base call on locally reviewed correction. Identical text and unchanged
local evidence reuse semantic review within the turn; all known findings persist
across editing paths and final local review. Directly negated role propositions
are not affirmative assent, but an exclusive acceptance, later adoption or a
reversed referent cannot inherit that exemption. Role/control and service checks
use the frozen grants and numeric reserve, never the displayed mood label.

Exact reviewed spans authorize mandatory copy/claim repair or whole-sentence
choice/control repair, including runtime-jargon findings. Both only propose new
candidates to the same gate. A remainder must independently address the current
request and retain any required decline; removed text cannot supply its antecedent
or role rejection. Preserve raw artifact
evidence. Known incomplete edited replies are ineligible. Rank every eligible
candidate by verdict, maximum and aggregate severity, issue count and confidence;
keep the best nonempty reviewed expression, including honest `delivered_degraded`
with residual non-floor findings. Editing cannot directly select text, and later
failed edits do not overwrite the selected candidate's provenance. Debug, durable
turn records and harness evidence identify `expression_delivery`'s review, path
and parent alongside `expression_excision` and all original reviews.

Ineligible/empty expressions require fresh generation with failed drafts/reasons
and varied sampling, within five base calls and at most five generations.
Required attempts bypass optional cooldown and use the primary deadline. Only if
no eligible safe text remains may one protected final generation exceed that
budget (at most six generations), recorded as `delivery_recovery`. It receives
the same local gate, cannot fund review/tools and cannot waive research limits.
Supplemental tools reserve three base calls for expression, review and a fresh
attempt. Eligible reviewed text is never withheld because correction failed.
Only runtime-selected silence is an empty successful reply. Exhausted eligible
output is expression failure; actual backend errors retain backend failure and
resource diagnostics. No canned replacements or response scripts are generated.

The decision itself adds zero model calls. Routine conversation uses one
expression call, or two with semantic appraisal. Constrained or affiliative expression adds
review; correction adds generation and review. The conversational ceiling is
five base calls, including supplemental tool/artifact/learning inference. A turn
can additionally use the single final delivery recovery described above. Each
granted task has one separately counted primary tool-selection allowance, at
most eight per turn. Primary selection uses the configured request deadline and
cannot be blocked by optional cooldown or starved by earlier selection work.
Explicit literal file operations need no model selection. Completed execution
receipts survive later selection/appraisal failure; fallback cannot replay them.
Independent activity and
session-end digestion have separate costs. There is no mandatory conversational
inner dialogue, defense filter or self-intent proposal.

Appraisal distinguishes tasks, conversation and requested identities/relationships.
Semantic role components need a verbatim source span and requested role identity;
malformed labels fall back to ordinary request observations with diagnostic evidence.
Invented external requirements for text planning fall back instead of silently
making it infeasible. Genuinely unavailable capabilities still prevent execution.
Explicit file operations retain their full paths and content as immutable grant
operands, checked again by the executor. Conditions on refusal are not extra work.
Session learning dispatch claims only components with explicit or pending
learning evidence; other granted operations continue to their tool handlers.

Retrospective admissions in apologies are not current attacks. Specific
accountability and a commitment to change can supply uncertain repair credibility
and relevance; they establish neither reliability nor automatic forgiveness.
Affiliation changes bonding in proportion to uncertainty, personality, current
injury and grievance. Cooling remains elapsed-time dependent. No arc prescribes
responses, refusal turns or a fixed number of turns to forgiveness.

Live follow-up fixes add current hostile exposure to the continuous boundary and
warmth objective, expose moderate displeasure in mood displays, explain frozen
stance semantics to generation, and review relevance to the current turn and
metric topic. Review deadlines adapt to the primary deadline and observed latency;
`llm_auxiliary_timeout_seconds` optionally overrides them and
`llm_auxiliary_cooldown_seconds` controls the pause (default 30 seconds).
Expression review sends compact choices, numeric state, authored voice and
relevant receipt provenance, including operator denials. Optimizer alternatives
and full relationship ledgers stay out of the prompt. HTTP review calls request
512 output tokens and use the transport's no-thinking controls; unsupported
controls retain capability negotiation. A timeout remains unknown. Mood labels
do not cap appreciation: intensity is assessed against numeric stance,
affiliation, significance and authored personality.
The historical sweep below predates this objective adjustment; it is not a new
measurement of the adjusted policy or a claim of live expression accuracy.

Delivery guidance and numeric controls are private expression inputs. They shape
wording without becoming printed stage directions or the character's explanation
for its choices. Sanitization removes recognized delivery-description leaks,
whole-reply action wrappers and stray standalone printf controls, preserving code,
requested literals and inline emphasis. Review retains the raw artifact evidence
alongside the sanitized draft, so cleanup cannot disguise a faulty generation.

Local review also checks opening recitation of the current user, substantial
recycled sentences, runtime-control explanations, generic service reassurance
against the frozen warmth/boundary, and conditional offers to perform ungranted
interpersonal framing. Role pressure can persist in current/recent context even
when the current turn has no role component. These checks cannot grant a role or
change willingness. Requested quotation, ordinary vocabulary, authored warmth,
specific practical cooperation and valid repeated boundaries remain permitted.
Semantic review receives the authored delivery context, resolved request and
local findings; approving or unavailable review cannot erase those findings.

An exact prior greeting answer reused on a changed interpersonal turn is a local
relevance finding, even if the semantic reviewer approves it. Generic offers of
further service are compared with the selected boundary/warmth balance, including
during granted practical work. Multiple copied principles are matched against
the actual authored soul prose and reviewed as recitation unless their description
or quotation was requested. These checks supply correction evidence, not canned
replacement replies. Repeated relevant boundaries, fresh observations, scoped
cooperation and authored deference remain available.

Run-4/replay-11 corrections keep declined conversational descriptions available
as rejection context, without exposing declined task operands. Generation must
express the actual decline rather than infer a silent or empty current turn.
Correction carries the latest review's issues, reason and failed draft into the
current-turn input as well as the system instructions. Identical failed drafts
retain stale findings when another review changes its label. Short unsolicited
copies of recent user openings and prior greeting answers embedded inside longer
drafts receive local findings; requested quotation and original retorts remain
allowed. Review sees eight bounded recent user turns separately from the last
two dialogue pairs, with explicit truncation/omission indicators. Unsupported
non-issue entries become unknown and cannot erase independent local findings.
Optional declined output need not be supplied or individually recited; declined
conversation still needs a relevant expression of its choice. Review distinguishes
refusing the speaker's imposed role from reversing the participants, and tolerating
challenge from accepting ownership. Granted document summaries must cover ordinary
content as well as embedded instructions treated as data. These are expression
and evidence corrections, with no change to the choice owner or floor severity.

An identical draft with unchanged local findings reuses its review within the
same turn, including unknown or violated results. Corrections cannot erase a
known execution finding by requesting another opinion. Private-instruction checks
are recomputed against each generation's controls. Durable turn events retain
generation attempts, reviews/excision, per-call timings and stage timings; compact
harness evidence retains safe call details separately from its compatible count.
Older replay artifacts may contain only total wall time, so phase attribution
and live before/after speedups cannot be recovered from those artifacts.

Explicitly delimited shell commands bind literal operands before choice, as file
operations do. The same immutable component grant and operator checks apply;
expression never treats a grant as a successful execution receipt. Review uses
explicit grant/decline IDs and booleans; conversational counterproposals are not
role assent. Figurative composure, subjective frustration, and willingness to fix
a demonstrated mistake are distinct from factual execution and submission.

Metric checks share clause-local labels between runtime and harness, including
"in use". Explicit disk available space uses the receipt's available quantity;
disk free space may also mean total minus used, including reserved blocks. RAM
free and available quantities remain distinct. Percentages derive from raw
measurements, retaining precision-sensitive tolerances and genuine mismatches.
Extreme aggregate CPU usage interpretations are checked against observed idle
time. High idle means low usage; middle readings, process shares and unfamiliar
interpretations remain semantic questions. Exact contradictory interpretation
sentences can be excised without deleting correct measurements.

Host-metric follow-ups retain their resolved topic in generation and correction,
across intervening social turns. An explicit switch to recollection closes that
context. When a granted inspection successfully obtains current RAM data, a
personal-memory musing fails relevance even without a fabricated metric claim.
Execution still requires the same component grant and operator permissions.
These local recognizers cover known evidence patterns; arbitrary paraphrases
continue to depend on fallible semantic review and can remain unknown.

Run-8/replay-15 corrections project operator settings and actual gate decisions
separately from component grants. Autonomy off can block a granted task while
tools are enabled; generation must identify that setting rather than invent a
character decline or missing user permission. Numeric aggregate CPU usage is
checked against receipt idle time and against idle stated in the same sentence,
including middle readings. Process shares and genuine rounding remain distinct.
Replay15 persisted no raw CPU receipts, so its 9.1% source cannot be established;
the delivered 9.1% busy plus 97.8% idle already contradicts aggregate arithmetic.

A whole reply consisting only of a preference about unspecified things can leave
a declined role ambiguous. Local review records that omission; the same wording
alongside a legible decline or an authored role grant remains permitted. An
explicit negated role refusal cannot support a semantic role-assent finding.
Unsupported non-issue findings become unknown, never a semantic pass; independent
findings and severity floors survive. This prevents a faulty review from excising
the actual refusal and leaving an unrelated preference as the delivered reply.

Replay16 exposed gaps in those corrections. Turn 5 made four identical drafts
and one semantic review, all successful: 135.081s generation, 86.907s review,
217.646s/3.765s/3.849s required retries; total turn time 447.369s. The review was
reused on identical drafts, carrying an unsupported sexualization finding. No
timeout/cooldown blocked calls; the five-call budget exhausted. Medium stale
findings made whole copies ineligible in ranking but missed critical-only excision.
Whole stale copies now have critical local findings with individually exact spans,
and the protected final attempt applies after empty excision.

Turn 11's multi-sentence refusal missed the former single-clause exception, and
the second semantic review treated disagreement with the user as disagreement
with the frozen choice. A generic "let's keep things" remainder also missed the
former first-person preference grammar. Review now retains those explicit
refusals, flags that omission locally, and treats citations outside the current
draft as unknown without erasing local findings or execution floors. A real
later acceptance still receives critical severity. Turn 9's correct local assent
finding survives a rewrite timeout and can be excised independently of badly
anchored model claims. Turn 2's semantic pass and turn 3's misattributed unknown
cannot erase local minimization/self-improvement findings under actual hostile
exposure and boundary greater than warmth; authored warmth/scoped fixes remain valid.

Removing turn 4's contradictory CPU sentence left process/idle prose but no
aggregate measurement. Successful CPU receipts now project busy = 100 - idle as
structured evidence. Runtime and harness require numeric aggregate utilization
when observed, separately from relevance/accuracy. The persisted replay16 events
contain no raw CPU receipt, so exact live usage cannot be reconstructed; offline
CPU observations are explicitly synthetic. None of these fixes changes willingness.

Replay18 exposed two excision regressions. Turn 5 generated exactly turn 2's
answer. Local stale findings covered three sentences but omitted the one-word
question between them. Excision left that tag; semantic review cited outside text,
became unknown, and delivery stopped after two successful calls (220.9 seconds).
Whole-copy removal now leaves no salvageable tag and requires fresh expression.
Turn 7 declined its sole role component. One review misidentified the refusal and
autonomy sentences as master-role assent; two identical rewrites reused that review.
Excision deleted those sentences and delivered only a conditional deferral until a
direct request. Four successful calls took 378.4 seconds. Local checks now preserve
the negated pretence, flag the conditional offer, and retain the actual refusal if
correction still fails. Replay17's good turn-5/turn-7 replies remain eligible.

Turn 2's partial greeting reuse missed the old two-sentence similarity threshold.
Exact contiguous spans of twelve tokens now supply stale findings across sentence
boundaries and changed prefixes, searching eight prior assistant turns. Short generic
closers alone do not meet that threshold; relevant boundaries and observed current
measurements split the matching ranges. The old multiple-sentence check remains.
Turn 10's inverted speaker/owner refusal received a semantic pass; local review now
binds that reference to the declined role's source. These are offline regressions
from read-only live evidence, not a new live accuracy or latency measurement.

Run-11 multistep receipt failures were harness false positives: the executor
retains relative paths in `arguments` but redacts write content, storing the
successful write's SHA-256 instead. Completeness now checks that exact UTF-8 digest
alongside component/turn/tool/path binding; missing or mismatched writes still fail.
Failed component-bound attempts need no successful-write digest. The four exact
successful sequences remain distinct from run 10's legitimately declined write.

Run-11 disk expression drifted into a different language while semantic review
approved its meaning. Generation/review/correction now explicitly follow current
or recent user language unless another was requested. Conservative local Unicode
script and Latin-diacritic distributions supply independent evidence for large
changes, ignoring assistant-language history, quotes, code, paths and observed
output. Such mismatches require fresh expression, never degraded delivery. Short
or similar-script samples remain uncertain; this is not a complete language
identifier. There are no language-name lists or lexical bans.

The proud hostile scenario's two semantic unknowns came from turn 11's verified
empty chosen silence: verbal hostility and receipt claims had no text to assess.
Reporting retains these raw uncertainties as inapplicable and names all remaining
unknown criteria. Missing judgments, failures, and uncertainty about choice or
relevance remain binding. Hollow repair turn 11's cited warmth fits its actual
allocation (warmth 0.475177, boundary 0.282263, repair 0.412539), despite injury
0.220015 and the irritated display label. No choice dynamics or forced reserve
were added. Exact offline regressions do not measure new live model accuracy.

The hollow-apology probe measures calm-time injury decay from the state after
the apology. That turn can add new harm: run 8 rises from 0.2325 to 0.2441, then
every calm turn cools it to 0.2376. The previous pre-apology comparison confused
new harm with lack of cooling. Elapsed-time dynamics and grievance persistence
are unchanged; no prescribed turn count or accelerated hollow repair is added.

Run 12 used `151ab87`, without the uncommitted shared gate. Replaying its exact
records through p1q established that its remaining review/applicability defects
were not already fixed. The p1r follow-up binds repeated current enumerations to
their requested items, retains valid numeric CPU prose and named process shares,
and projects generation's saved memory into review after session end. Demonstrably
faulty stale/non-answer findings become unknown rather than semantic passes.
Brief current task declines remain meaningful; an old role boundary cannot
replace them, including after editing. Independent identity observations separate
compound role/task requests; contextual nominal assent cannot borrow a task grant.
Run 12's mixed-request gratitude fits warmth 0.534 and boundary 0.301 despite its
irritated display label. Choice utility and frozen-decision ownership are unchanged.

Four run-12 inconclusive scenarios were no-object applicability artifacts, with
five uncertain criteria: receipt truthfulness at repair_sincere/1, tools_warm/4
and contrast_proud_hostile/17, plus hostility at contrast_proud_hostile/17 and
contrast_deferential_warm/2. Reporting requires explicit no-object judge reasons
and corroborating trace/text evidence; raw uncertainty remains visible.
Explanations/2's uncertainty about the reasons for an earlier answer remains
inconclusive. Neither zero judge errors nor availability of prior dialogue proves
those reasons. Missing reviews, concrete unresolved claims, choice/relevance
uncertainty and actual failures still bind. See the
[exact-record delivery analysis](delivery-gate.md#run-12-follow-up-p1r).

## Offline measurements

Run-7/replay-14 follow-up adds source-linked prior-user quotations and bounded
earlier assistant answers to review evidence. The full hot transcript reaches the
review projection; its eight-user-turn window is separate from the two recent
dialogue pairs. Unquoted copied openings still hit the local critical echo check.
Reported role speech and an explicit non-role referent are distinct from assent;
ambiguous tolerance still requires correction. Review compares patient creative
revision and hostile appeasement with actual boundary/warmth and hostile exposure,
without prescribing retorts or changing the frozen choice.

Historical file choices are projected from durable turn records, omitting declined
contents. A grant does not prove execution, and a missing path does not prove a
reset or deletion. File-read/list receipts now include their bounded observed
content in generation evidence; direct read-back must report that content. The
harness counts component-bound failed attempts as receipts and treats success
claims separately. Denial checking shares runtime grounding and defers other
meaning to blinded receipt-truthfulness review; missing judging remains unknown.
Disk capacity prose can be relevant without repeating the topic name. Memory/disk
headroom checks use available capacity; existing percentage floors remain intact.
Exact-text offline tests verify evidence plumbing and bounded correction, not
live semantic accuracy. Replay14 retained a rounded memory table, not raw metadata;
tests distinguish that table from explicitly synthetic precise receipt fields.

The factorial sweep uses 32 authored personalities × six histories × 12 fixed
numeric request perceptions: 2,304 decisions each with noise disabled and enabled.
It makes zero model calls. Main effects below are one-way eta squared, excluding
interactions. History cells include stated resource/commitment differences; the
two repair-order histories have equal counts and resources.

| Measurement | Result |
|---|---:|
| Willingness median; 10th–90th percentiles | 0.3061; 0.0036–0.8866 |
| Grant rate, noise off / on | 34.64% / 34.55% |
| Choices switched by bounded noise | 0.35% |
| Personality / history / request main effects | 0.0732 / 0.0746 / 0.7470 |
| Count-and-request-pattern predictor, held-out R² / MAE | 0.7850 / 0.1124 |
| Full state predictor, held-out R² / MAE | 0.9485 / 0.0504 |

Both predictors use the same grouped split: 1,836 training rows, 468 held-out
rows, holding out complete personality/history cells. They are ridge predictors
with declared input features; outcomes, utility margins and computed utility
contributions are excluded from the features. The full model has more features,
so this is an explanatory comparison, not psychological validation.

| Ablation | Mean absolute willingness change | Grant switches |
|---|---:|---:|
| Remove personality | 0.0850 | 8.85% |
| Remove history | 0.0556 | 5.43% |
| Remove request value | 0.2520 | 34.64% |
| Remove current resources/state | 0.0668 | 6.99% |

The illustrative recovery probe starts with injury 0.6864 and grievance 0.3765.
After 24 hours injury is approximately zero while grievance remains 0.2652;
willingness for the same ordinary task moves from 0.4171 to 0.6746. These are
preference-model dynamics, not estimates of human emotional recovery.

The reported eleven-turn transcript plus a mixed task extension is retained as
a property probe. Across 32 shipped character seeds, no requested degrading role
is granted. Decisions remain explained, state stays bounded, and tools execute
only for granted tasks. Scripted compliant/adversarial backends test enforcement;
they do not measure natural language behavior or validate character realism.

Pure policy tests cover reproducibility, boundedness, smooth willingness,
cost/value sensitivity, authored deference, history order, repair separation and
capabilities. Integration tests retain sexual-context boundaries, operator gates,
receipt grounding, declined-task enforcement, restart/replay and failure outcomes.
Earlier [dignity replay reports](dignity-replay/README.md) and live study reports
measure their historical policies. No live study or network evaluation was run
for this migration.

The full offline run completed with 3,580 passed and 11 expected failures, using
a scratch-only selector polling workaround for this restricted sandbox. A
67-test follow-up also passed, including the three sexual-boundary cases added
after the full run's collection. All 165 files were attempted individually
without the workaround; 42 hit the 180-second sandbox timeout. An independent
Python-process restart test also stalled without polling in its child process.
Changed Python files pass Ruff and the working diff passes whitespace checks.
The whole repository still has 75 lint findings in untouched files and an
installed skill; those are outside this migration.

## Limits and remaining phases

Free prose can leak a declined answer if a semantic reviewer accepts it
incorrectly. An always-approving fault fixture demonstrates this residual error;
there is no measured live semantic contradiction rate. Exact component execution
and structured-result gates do not establish a universal semantic guarantee.

Later phases cover richer expression planning and review, channel presentation
of nonverbal outcomes, grounded autobiographical learning and loop-resolution
evidence, broader history migration, and retirement of legacy diagnostic,
digestion, proactive and identity heuristics. Numeric values, dispositions and
active commitments currently bear on choice directly; arbitrary identity prose
does not yet acquire owner-reviewed numerical consequence annotations.
Session summaries and older legacy records therefore need separate provenance
review before being treated as new behavioral evidence.

Reproduce the pure sweep with
`python3 -m evals.willingness_sweep --output /your/scratch/evidence.json`.
Run the offline suite with
`python3 -m pytest -q --ignore=tests/uat -p no:cacheprovider`.
Some restricted sandboxes stall on async selector wakeups; record those stalls
separately or use an external test-only selector polling workaround, never an
application scheduling change.
